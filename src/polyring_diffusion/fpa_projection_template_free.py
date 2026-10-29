"""Constraint projection for template-free ring-cell samples."""
from __future__ import annotations

import math
import torch

from .chem import element_record


def _components(n, edges):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for a, b in edges: union(a, b)
    return find, union


def _target_overlap(relation: int) -> int:
    return relation-1 if relation >= 3 else 1 if relation == 2 else 0


def _best_cycle(atoms, edge_score):
    """Maximum-score Hamiltonian cycle on a small cell (3-10 atoms)."""
    atoms = tuple(sorted(map(int, atoms)))
    if len(atoms) < 3:
        return []
    start, n = 0, len(atoms)
    dp = {(1 << start, start): (0., (start,))}
    for mask_size in range(1, n):
        updated = dict(dp)
        for (mask, last), (value, path) in list(dp.items()):
            if mask.bit_count() != mask_size:
                continue
            for nxt in range(1, n):
                if mask & (1 << nxt):
                    continue
                key = (mask | (1 << nxt), nxt)
                candidate = value + float(edge_score[atoms[last], atoms[nxt]])
                if key not in updated or candidate > updated[key][0]:
                    updated[key] = (candidate, path+(nxt,))
        dp = updated
    full = (1 << n)-1
    choices = [(value+float(edge_score[atoms[last], atoms[start]]), path)
        for (mask, last), (value, path) in dp.items() if mask == full and last != start]
    if not choices:
        return []
    path = max(choices, key=lambda item: item[0])[1]
    ordered = [atoms[i] for i in path]
    return [tuple(sorted((ordered[i], ordered[(i+1) % n]))) for i in range(n)]


def _rdkit_graph_valid(sample, row, na, atom_map, charge_min):
    """Chemical identity check used to make projection idempotent on valid graphs."""
    try:
        from rdkit import Chem
        rw = Chem.RWMol()
        for atom_type, charge, hydrogen, radical in zip(
                sample["atom_types"][row, :na].tolist(), sample["charges"][row, :na].tolist(),
                sample["hydrogens"][row, :na].tolist(), sample["radicals"][row, :na].tolist()):
            atom = Chem.Atom(atom_map[int(atom_type)])
            atom.SetFormalCharge(int(charge)+charge_min)
            atom.SetNumExplicitHs(int(hydrogen)); atom.SetNumRadicalElectrons(int(radical)); atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
        bonds = sample["bonds"][row, :na, :na]
        aromatic_flags = sample.get("bond_aromatic")
        aromatic_flags = bonds.eq(4) if aromatic_flags is None else aromatic_flags[row, :na, :na].bool()
        for i in range(na):
            for j in range(i+1, na):
                kind = int(bonds[i, j])
                if kind not in kinds:
                    continue
                rw.AddBond(i, j, kinds[kind])
                if bool(aromatic_flags[i, j]):
                    rw.GetAtomWithIdx(i).SetIsAromatic(True); rw.GetAtomWithIdx(j).SetIsAromatic(True)
                    rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
        mol = rw.GetMol(); Chem.SanitizeMol(mol)
        return len(Chem.GetMolFrags(mol)) == 1
    except Exception:
        return False


def _feasible_bond_kind(prob, left_capacity, right_capacity, preferred=None,
                        allow_aromatic=True, left_aromatic=False, right_aromatic=False):
    order_value = (0., 1., 2., 3., 1.)
    ranked = sorted(range(1, 5 if allow_aromatic else 4), key=lambda kind: float(prob[kind]), reverse=True)
    if preferred == 4 and allow_aromatic:
        # A mandatory aromatic ring edge must remain aromatic; silently
        # downgrading it would make cell aromaticity disagree with the graph.
        ranked = [4]
    elif preferred in ranked:
        ranked.remove(preferred)
        ranked.insert(0, preferred)
    for kind in ranked:
        left_order = order_value[kind]+float(kind == 4 and not left_aromatic)
        right_order = order_value[kind]+float(kind == 4 and not right_aromatic)
        if left_order <= float(left_capacity)+1e-6 and right_order <= float(right_capacity)+1e-6:
            return kind
    return 0


def _valence_limit(symbol, formal_charge):
    base = float(element_record(symbol).default_valence)
    if symbol == "B" and formal_charge < 0:
        return 4.
    if symbol == "C" and formal_charge != 0:
        return 3.
    if symbol == "N":
        return 4. if formal_charge > 0 else 2. if formal_charge < 0 else base
    if symbol == "O":
        return 3. if formal_charge > 0 else 1. if formal_charge < 0 else base
    return base


def _complete_valence(remaining, hydrogen_probability, radical_probability, closed_shell,
                      radical_threshold):
    radical_confidence = float(radical_probability[1:].sum())
    max_radical = 0 if closed_shell or radical_confidence < radical_threshold else min(2, remaining)
    choices = []
    for radical in range(max_radical+1):
        hydrogen = remaining-radical
        if 0 <= hydrogen <= 4:
            cost = -math.log(max(float(hydrogen_probability[hydrogen]), 1e-12))
            cost -= math.log(max(float(radical_probability[radical]), 1e-12))
            # Deterministic ties prefer hydrogen completion, not a radical.
            choices.append((cost, radical, hydrogen))
    if not choices:
        return min(4, remaining), 0, radical_confidence
    _, radical, hydrogen = min(choices)
    return hydrogen, radical, radical_confidence


def _memberships(score, sizes, relations):
    nc, na = score.shape
    chosen = [set(torch.topk(score[c], int(sizes[c])).indices.tolist()) for c in range(nc)]
    for _ in range(max(2, 2*nc)):
        changed = False
        for left in range(nc):
            for right in range(left+1, nc):
                target = _target_overlap(int(relations[left, right]))
                shared = chosen[left] & chosen[right]
                while len(shared) > target:
                    atom = min(shared, key=lambda a: float(score[left, a]+score[right, a]))
                    loser = left if float(score[left, atom]) < float(score[right, atom]) else right
                    chosen[loser].remove(atom); shared.remove(atom); changed = True
                while len(shared) < target:
                    candidates = []
                    for atom in range(na):
                        if atom in shared: continue
                        feasible = True
                        for cell in (left, right):
                            if atom not in chosen[cell] and len(chosen[cell]) >= int(sizes[cell]) and not (chosen[cell]-shared):
                                feasible = False; break
                        if feasible: candidates.append(atom)
                    if not candidates: break
                    atom = max(candidates, key=lambda a: float(score[left, a]+score[right, a]))
                    before = len(shared)
                    for cell in (left, right):
                        if atom not in chosen[cell] and len(chosen[cell]) >= int(sizes[cell]):
                            removable = chosen[cell]-shared
                            chosen[cell].remove(min(removable, key=lambda a: float(score[cell, a])))
                        chosen[cell].add(atom)
                    shared = chosen[left] & chosen[right]
                    if len(shared) <= before: break
                    changed = True
        for cell in range(nc):
            while len(chosen[cell]) < int(sizes[cell]):
                allowed = []
                for atom in range(na):
                    if atom in chosen[cell]: continue
                    if all(atom not in chosen[other] or len(chosen[cell]&chosen[other]) < _target_overlap(int(relations[cell, other])) for other in range(nc) if other != cell):
                        allowed.append(atom)
                if not allowed: break
                chosen[cell].add(max(allowed, key=lambda a: float(score[cell, a]))); changed = True
        if not changed: break
    return chosen


@torch.no_grad()
def project_template_free_fpa(sample: dict, prediction: dict, atom_map, closed_shell=False,
                              radical_threshold=.8, charge_min=-2):
    mscore = prediction["membership_logits"].softmax(-1)[..., 1]
    bprob = prediction["bond_logits"].softmax(-1)
    aprob = prediction["bond_aromatic_logits"].softmax(-1)[..., 1] if "bond_aromatic_logits" in prediction else bprob[..., 4]
    if "hydrogen_logits" in prediction:
        hprob = prediction["hydrogen_logits"].softmax(-1)
    else:
        hprob = torch.nn.functional.one_hot(sample["hydrogens"].clamp(0, 4), 5).float()
    if "radical_logits" in prediction:
        rprob = prediction["radical_logits"].softmax(-1)
    else:
        rprob = torch.nn.functional.one_hot(sample["radicals"].clamp(0, 2), 3).float()
    valid, repairs, reasons = [], [], []
    for row in range(len(sample["atom_types"])):
        na = int(sample["atom_mask"][row].sum()); nc = int(sample["cell_mask"][row].sum())
        reason = 0; changed = 0
        pos = sample["positions"][row, :na]
        # Projection repairs generated invalid graphs only. A chemically valid
        # graph is already a fixed point and its molecular identity is preserved.
        if _rdkit_graph_valid(sample, row, na, atom_map, charge_min):
            aromatic_cells = int(sample["cell_aromatic"][row, :nc].sum())
            sample.setdefault("projection_aromatic_cells", torch.zeros(len(sample["atom_types"]), dtype=torch.long, device=pos.device))[row] = aromatic_cells
            sample.setdefault("projection_complete_aromatic_cells", torch.zeros(len(sample["atom_types"]), dtype=torch.long, device=pos.device))[row] = aromatic_cells
            valid.append(True); repairs.append(0); reasons.append(0)
            continue
        sizes = sample["cell_sizes"][row, :nc].clamp(3, 10)
        if bool((sizes > na).any()): reason |= 32
        safe_sizes = sizes.clamp_max(na)
        relations = sample["cell_relations"][row, :nc, :nc]
        chosen = _memberships(mscore[row, :nc, :na], safe_sizes, relations)
        for cell in range(nc):
            if len(chosen[cell]) != int(safe_sizes[cell]): reason |= 1
        for left in range(nc):
            for right in range(left+1, nc):
                if len(chosen[left]&chosen[right]) != _target_overlap(int(relations[left, right])): reason |= 2
        member = torch.zeros((nc, na), dtype=torch.long, device=pos.device)
        for cell, atoms in enumerate(chosen):
            if atoms: member[cell, torch.tensor(sorted(atoms), device=pos.device)] = 1
        sample["membership"][row, :nc, :na] = member

        centers = sample["cell_centers"][row, :nc]
        frames = sample["cell_frames"][row, :nc]
        radii = sample["cell_radii"][row, :nc]
        for _ in range(3):
            targets = [[] for _ in range(na)]
            for cell, atoms in enumerate(chosen):
                tangent, bitangent = frames[cell, 0], frames[cell, 1]
                for slot, atom in enumerate(sorted(atoms)):
                    rel = pos[atom]-centers[cell]
                    u, v = (rel*tangent).sum(), (rel*bitangent).sum()
                    norm = torch.sqrt(u*u+v*v)
                    if float(norm) < 1e-5:
                        angle = 2*math.pi*slot/max(1, len(atoms))
                        direction = math.cos(angle)*tangent+math.sin(angle)*bitangent
                    else:
                        direction = (u*tangent+v*bitangent)/norm
                    targets[atom].append(centers[cell]+radii[cell]*direction)
            for atom, values in enumerate(targets):
                if values: pos[atom] = torch.stack(values).mean(0); changed += 1
        sample["positions"][row, :na] = pos

        mandatory = {}
        aromatic_cycles = []
        def require_edge(edge, kind):
            # A shared edge belonging to any aromatic cell must remain aromatic.
            old = mandatory.get(edge)
            mandatory[edge] = 4 if old == 4 or kind == 4 else kind
        for cell, atoms in enumerate(chosen):
            aromatic = bool(sample["cell_aromatic"][row, cell])
            if len(atoms) < 3: reason |= 1; continue
            edge_score = aprob[row] if aromatic else bprob[row, :, :, 1:4].amax(-1)
            cycle = _best_cycle(atoms, edge_score)
            if len(cycle) != len(atoms):
                reason |= 1
                continue
            for a, b in cycle:
                kind = 4 if aromatic else int(torch.argmax(bprob[row, a, b, 1:4]))+1
                require_edge((a, b), kind)
            if aromatic:
                aromatic_cycles.append(cycle)
        # Relation-specific topology: fused cells share an edge; linked cells have one direct inter-ring bond.
        priority_edges = set()
        for left in range(nc):
            for right in range(left+1, nc):
                relation = int(relations[left, right])
                shared = sorted(chosen[left] & chosen[right])
                if relation == 3 and len(shared) == 2:
                    i, j = shared
                    both_aromatic = bool(sample["cell_aromatic"][row, left] and sample["cell_aromatic"][row, right])
                    kind = 4 if both_aromatic else int(torch.argmax(bprob[row, i, j, 1:4]))+1
                    require_edge((i, j), kind)
                    priority_edges.add(tuple(sorted((i, j))))
                elif relation == 1:
                    pairs = [(float(1-bprob[row, i, j, 0]), i, j) for i in chosen[left] for j in chosen[right] if i != j]
                    if pairs:
                        _, i, j = max(pairs)
                        kind = int(torch.argmax(bprob[row, i, j, 1:4]))+1
                        require_edge(tuple(sorted((i, j))), kind)
                    else:
                        reason |= 16
        projected = torch.zeros((na, na), dtype=torch.long, device=pos.device)
        projected_aromatic = torch.zeros((na, na), dtype=torch.bool, device=pos.device)
        used = torch.zeros(na, device=pos.device)
        aromatic_used = torch.zeros(na, dtype=torch.bool, device=pos.device)
        limits = torch.tensor([_valence_limit(atom_map[int(t)], int(q)+charge_min)
            for t, q in zip(sample["atom_types"][row, :na], sample["charges"][row, :na])], device=pos.device)
        order_value = (0., 1., 2., 3., 1.)
        edges = []
        ordered_mandatory = sorted(mandatory.items(), key=lambda item: 0 if item[0] in priority_edges else 1)
        for (i, j), kind in ordered_mandatory:
            kind = _feasible_bond_kind(bprob[row, i, j], limits[i]-used[i],
                limits[j]-used[j], preferred=kind,
                left_aromatic=bool(aromatic_used[i]), right_aromatic=bool(aromatic_used[j]))
            if kind == 0:
                reason |= 8
                continue
            left_order = order_value[kind]+float(kind == 4 and not aromatic_used[i])
            right_order = order_value[kind]+float(kind == 4 and not aromatic_used[j])
            projected[i, j] = projected[j, i] = kind
            used[i] += left_order; used[j] += right_order
            if kind == 4:
                aromatic_used[i] = aromatic_used[j] = True
                projected_aromatic[i, j] = projected_aromatic[j, i] = True
            edges.append((i, j))
        find, union = _components(na, edges)
        candidates = []
        for i in range(na):
            for j in range(i+1, na):
                if (i, j) in mandatory: continue
                candidates.append((float(1-bprob[row, i, j, 0]), i, j))
        candidates.sort(reverse=True)
        # Constrained maximum-spanning-tree pass first: Kruskal ordering by
        # predicted edge confidence connects components before adding cycles.
        for connect_only in (True, False):
            for confidence, i, j in candidates:
                if projected[i, j] != 0: continue
                if connect_only and find(i) == find(j): continue
                if not connect_only and confidence < .5: continue
                distance = float((pos[i]-pos[j]).norm())
                left_radius = element_record(atom_map[int(sample["atom_types"][row, i])]).covalent_radius
                right_radius = element_record(atom_map[int(sample["atom_types"][row, j])]).covalent_radius
                if distance > 1.45*(left_radius+right_radius): continue
                kind = _feasible_bond_kind(bprob[row, i, j], limits[i]-used[i], limits[j]-used[j],
                    allow_aromatic=False, left_aromatic=bool(aromatic_used[i]),
                    right_aromatic=bool(aromatic_used[j]))
                if kind == 0: continue
                left_order = order_value[kind]+float(kind == 4 and not aromatic_used[i])
                right_order = order_value[kind]+float(kind == 4 and not aromatic_used[j])
                projected[i, j] = projected[j, i] = kind
                used[i] += left_order; used[j] += right_order
                if kind == 4: aromatic_used[i] = aromatic_used[j] = True
                union(i, j); changed += 1
        remaining = (limits-used).floor().clamp(0, 4).long()
        # Complete the graph while retaining explicitly predicted open-shell
        # states. Closed-shell generation is an opt-in constraint.
        completion = [_complete_valence(int(remaining[i]), hprob[row, i], rprob[row, i],
            closed_shell, radical_threshold) for i in range(na)]
        assigned_h = torch.tensor([item[0] for item in completion], device=pos.device)
        assigned_radical = torch.tensor([item[1] for item in completion], device=pos.device)
        radical_confidence = torch.tensor([item[2] for item in completion], device=pos.device)
        # QCDGE contains even-electron open-shell records (2, 4, ...), but no
        # monoradicals. Remove the least-confident electron if sampling produces
        # an unsupported odd molecular radical count.
        if int(assigned_radical.sum()) % 2:
            candidates = torch.where(assigned_radical.gt(0))[0]
            atom = int(candidates[radical_confidence[candidates].argmin()])
            assigned_radical[atom] -= 1
            assigned_h[atom] += 1
            changed += 1
        sample["hydrogens"][row, :na] = assigned_h
        sample["radicals"][row, :na] = assigned_radical
        complete_aromatic = sum(all(bool(projected_aromatic[i, j]) for i, j in cycle)
            for cycle in aromatic_cycles)
        if complete_aromatic != len(aromatic_cycles): reason |= 8
        if na and len({find(i) for i in range(na)}) > 1: reason |= 4
        if bool((used > limits+1e-5).any()): reason |= 8
        sample.setdefault("projection_aromatic_cells", torch.zeros(len(sample["atom_types"]), dtype=torch.long, device=pos.device))[row] = len(aromatic_cycles)
        sample.setdefault("projection_complete_aromatic_cells", torch.zeros(len(sample["atom_types"]), dtype=torch.long, device=pos.device))[row] = complete_aromatic
        sample["bonds"][row, :na, :na] = projected
        if "bond_aromatic" not in sample:
            sample["bond_aromatic"] = torch.zeros_like(sample["bonds"], dtype=torch.bool)
        sample["bond_aromatic"][row, :na, :na] = projected_aromatic
        valid.append(reason == 0); repairs.append(changed); reasons.append(reason)
    device = sample["atom_types"].device
    sample["projection_valid"] = torch.tensor(valid, device=device)
    sample["projection_repairs"] = torch.tensor(repairs, device=device)
    sample["projection_reason"] = torch.tensor(reasons, device=device)
    return sample
