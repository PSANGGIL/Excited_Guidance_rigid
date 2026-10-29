"""Optional FINAL joint MAP decoder. It never changes the forward/reverse kernel.

MILP variables select atom/H/charge/allowed-valence and integer Kekule bonds
jointly. A super-root single-commodity flow connects every selected real atom.
Shared ring atoms/edges have ONE variable, including fused/hetero/charged rings.
Required aromatic edges (when explicitly supplied) are checked globally by
RDKit; rejected integer assignments receive no-good cuts, under a total timeout.

Raw samples and coordinates are never overwritten. This is constrained MAP,
not an unbiased molecular sample or a certificate of geometry/EST/synthesis.
"""
from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from rdkit import Chem


@dataclass
class JointDecodeResult:
    status: str
    molecule: object = None
    states: object = None
    bonds: object = None
    retained_indices: object = None
    optimal: bool = False
    iterations: int = 0
    reason: str = ""


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=float, copy=True)


def _probabilities(value, shape):
    value = _array(value)
    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"expected finite nonnegative probabilities of shape {shape}")
    if np.any(value.sum(-1) <= 0):
        raise ValueError("empty categorical support")
    return value / value.sum(-1, keepdims=True)


def _build_molecule(states, bonds, pairs, metadata, positions, aromatic_edges):
    keep = [i for i, state in enumerate(states) if not metadata[state].get("is_fake", False)]
    mapping = {old: new for new, old in enumerate(keep)}
    rw = Chem.RWMol()
    for i in keep:
        state = metadata[states[i]]
        atom = Chem.Atom(state["symbol"])
        atom.SetFormalCharge(int(state["formal_charge"]))
        atom.SetNumExplicitHs(int(state["hydrogens"]))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    types = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    for (i, j), order in zip(pairs, bonds):
        if order:
            if i not in mapping or j not in mapping:
                raise ValueError("bond to fake atom")
            rw.AddBond(mapping[i], mapping[j], types[order])
    mol = rw.GetMol()
    conf = Chem.Conformer(len(keep))
    for new, old in enumerate(keep):
        conf.SetAtomPosition(new, positions[old].tolist())
    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    for atom, old in zip(mol.GetAtoms(), keep):
        state = metadata[states[old]]
        if atom.GetFormalCharge() != state["formal_charge"] or atom.GetTotalNumHs() != state["hydrogens"]:
            raise ValueError("sanitization altered the selected atom state")
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("disconnected or open-shell decoder solution")
    for edge in aromatic_edges:
        i, j = pairs[edge]
        bond = mol.GetBondBetweenAtoms(mapping.get(i, -1), mapping.get(j, -1))
        if bond is None or not bond.GetIsAromatic():
            raise ValueError("required aromatic edge not realized by a global Kekule assignment")
    # Round-trip Kekulization must work globally, not ring by ring.
    Chem.Kekulize(Chem.Mol(mol), clearAromaticFlags=True)
    return mol, keep


def joint_decode(node_probabilities, edge_probabilities, positions, metadata, *,
                 required_atoms=None, net_charge=0, aromatic_edges=(),
                 time_limit=5., max_rounds=16, max_slots=96):
    positions = _array(positions)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions must be finite [N,3]")
    n, k = len(positions), len(metadata)
    if not 2 <= n <= max_slots:
        return JointDecodeResult("rejected", reason="slot count outside solver bounds")
    if not math.isfinite(time_limit) or time_limit <= 0 or max_rounds < 1:
        raise ValueError("positive bounded solver time/rounds required")
    if required_atoms is not None and (int(required_atoms) != required_atoms or not 2 <= required_atoms <= n):
        raise ValueError("required_atoms must be an integer between 2 and N")
    if net_charge is not None and (not math.isfinite(net_charge) or int(net_charge) != net_charge):
        raise ValueError("net_charge must be an integer or None")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    nodes = _probabilities(node_probabilities, (n, k))
    edges = _probabilities(edge_probabilities, (len(pairs), 4))
    aromatic_edges = set(aromatic_edges)
    if any(not isinstance(e, (int, np.integer)) or not 0 <= e < len(pairs) for e in aromatic_edges):
        raise ValueError("aromatic_edges must index the canonical upper triangle")
    variants = []
    for s, state in enumerate(metadata):
        values = [0] if state.get("is_fake", False) else [
            int(v) for v, count in state.get("train_heavy_valence_counts", {}).items() if count > 0
        ]
        variants.extend((s, v) for v in values if v >= 0)
    if not variants:
        raise ValueError("no train-supported atom/valence states")

    def valid_discrete(states, bonds):
        vals = np.zeros(n, dtype=int)
        for (i, j), order in zip(pairs, bonds):
            vals[i] += order
            vals[j] += order
        if any((s, int(v)) not in variants for s, v in zip(states, vals)):
            return None
        count = sum(not metadata[s].get("is_fake", False) for s in states)
        if count < 2 or (required_atoms is not None and count != required_atoms):
            return None
        if net_charge is not None and sum(int(metadata[s].get("formal_charge", 0)) for s in states) != net_charge:
            return None
        try:
            return _build_molecule(states, bonds, pairs, metadata, positions, aromatic_edges)
        except (ValueError, RuntimeError, OverflowError):
            return None

    # A feasible independent argmax attains the unconstrained score upper bound.
    # This fast path is therefore a certified MAP optimum, not a repair shortcut.
    raw_states, raw_bonds = nodes.argmax(-1), edges.argmax(-1)
    decoded = valid_discrete(raw_states, raw_bonds)
    if decoded is not None:
        return JointDecodeResult("optimal", decoded[0], raw_states, raw_bonds, decoded[1], True)

    costs, integrality, upper_bounds = [], [], []
    def variable(cost=0., binary=True, ub=1.):
        index = len(costs)
        costs.append(cost)
        integrality.append(int(binary))
        upper_bounds.append(ub)
        return index
    z, b = [], []
    for i in range(n):
        z.append({(s, v): variable(-math.log(nodes[i, s]))
                  for s, v in variants if nodes[i, s] > 0})
        if not z[-1]:
            return JointDecodeResult("infeasible", reason="no supported atom state")
    for e in range(len(pairs)):
        b.append({order: variable(-math.log(edges[e, order]))
                  for order in range(4) if edges[e, order] > 0
                  and (e not in aromatic_edges or order in (1, 2))})
        if not b[-1]:
            return JointDecodeResult("infeasible", reason="empty bond support")
    roots = [variable() for _ in range(n)]
    sources = [variable(binary=False, ub=n) for _ in range(n)]
    flows = [(variable(binary=False, ub=n-1), variable(binary=False, ub=n-1)) for _ in pairs]
    rows, lower, upper = [], [], []
    def constraint(coefficients, lo, hi):
        rows.append(coefficients)
        lower.append(lo)
        upper.append(hi)
    for selections in z + b:
        constraint({idx: 1. for idx in selections.values()}, 1., 1.)
    real = [{idx: 1. for (s, v), idx in entry.items() if not metadata[s].get("is_fake", False)} for entry in z]
    constraint({idx: 1. for row in real for idx in row},
               required_atoms if required_atoms is not None else 2,
               required_atoms if required_atoms is not None else n)
    if net_charge is not None:
        constraint({idx: int(metadata[s].get("formal_charge", 0))
                    for entry in z for (s, v), idx in entry.items()}, net_charge, net_charge)
    constraint({idx: 1. for idx in roots}, 1., 1.)
    for i in range(n):
        valence = {idx: -v for (s, v), idx in z[i].items()}
        balance = {sources[i]: 1., **{idx: -1. for idx in real[i]}}
        for e, (a, c) in enumerate(pairs):
            if i not in (a, c):
                continue
            valence.update({idx: order for order, idx in b[e].items()})
            outward, inward = flows[e] if i == a else flows[e][::-1]
            balance[outward], balance[inward] = -1., 1.
        constraint(valence, 0., 0.)
        constraint(balance, 0., 0.)
        constraint({roots[i]: 1., **{idx: -1. for idx in real[i]}}, -np.inf, 0.)
        constraint({sources[i]: 1., roots[i]: -n}, -np.inf, 0.)
    for e, (forward, backward) in enumerate(flows):
        for flow in (forward, backward):
            constraint({flow: 1., **{idx: -(n-1) for order, idx in b[e].items() if order}}, -np.inf, 0.)

    start = time.monotonic()
    reason = ""
    for attempt in range(max_rounds):
        remaining = time_limit - (time.monotonic() - start)
        if remaining <= 0:
            break
        rr, cc, vv = [], [], []
        for r, row in enumerate(rows):
            for c, value in row.items():
                rr.append(r); cc.append(c); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), len(costs))).tocsc()
        result = milp(np.asarray(costs), integrality=np.asarray(integrality),
                      bounds=Bounds(np.zeros(len(costs)), np.asarray(upper_bounds)),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options=dict(time_limit=remaining, mip_rel_gap=0.))
        if result.x is None:
            return JointDecodeResult("infeasible" if result.status == 2 else "unresolved",
                                     iterations=attempt + 1, reason=str(result.message))
        x = result.x
        # Independently check a time-limited incumbent; never round fractional
        # values and silently declare a feasible molecule.
        binary = np.asarray(integrality, dtype=bool)
        if (np.max(np.abs(x[binary] - np.round(x[binary]))) > 1e-5
            or np.any(matrix @ x < np.asarray(lower) - 1e-5)
            or np.any(matrix @ x > np.asarray(upper) + 1e-5)):
            return JointDecodeResult("unresolved", reason="incumbent violates constraints")
        states = np.array([max(entry, key=lambda key: x[entry[key]])[0] for entry in z])
        bonds = np.array([max(entry, key=lambda key: x[entry[key]]) for entry in b])
        decoded = valid_discrete(states, bonds)
        if decoded is not None:
            return JointDecodeResult("optimal" if result.status == 0 else "feasible_time_limit",
                                     decoded[0], states, bonds, decoded[1], result.status == 0, attempt + 1)
        chosen = [idx for entry in z + b for idx in entry.values() if x[idx] > .5]
        constraint({idx: 1. for idx in chosen}, -np.inf, len(chosen) - 1.)
        reason = "integer assignment failed global RDKit/closed-shell/aromatic validation"
    return JointDecodeResult("unresolved", iterations=max_rounds, reason=reason or "total timeout")


def attach_joint_decoding(sample, metadata, options=None, required_atoms=None):
    """Attach a SEPARATE molecule/report; raw rdkit_mol/slot_graph stay untouched."""
    import torch
    graph = sample.slot_graph
    if "s_endpoint_probs" not in graph.ndata or "e_endpoint_probs" not in graph.edata:
        raise ValueError("joint decoding requires saved endpoint probabilities, not sampled one-hot surrogates")
    src, dst = graph.edges()
    selected = torch.where(src < dst)[0]
    selected = selected[torch.argsort(src[selected] * graph.num_nodes() + dst[selected])]
    result = joint_decode(
        graph.ndata["s_endpoint_probs"], graph.edata["e_endpoint_probs"][selected],
        graph.ndata["x_1"], metadata, required_atoms=required_atoms, **(options or {}),
    )
    sample.joint_rdkit_mol = result.molecule
    sample.joint_decode_report = dict(status=result.status, optimal=result.optimal,
                                    iterations=result.iterations, reason=result.reason,
                                    method="final constrained MAP; raw sample unchanged")
    if result.molecule is not None:
        raw_states = graph.ndata["s_1"].argmax(-1).cpu().numpy()
        raw_bonds = graph.edata["e_1"][selected].argmax(-1).cpu().numpy()
        sample.joint_decode_report.update(
            atom_state_edits=int(np.sum(result.states != raw_states)),
            bond_edits=int(np.sum(result.bonds != raw_bonds)),
            retained_indices=list(map(int, result.retained_indices)),
            coordinates_reembedded=False,
        )
    return sample
