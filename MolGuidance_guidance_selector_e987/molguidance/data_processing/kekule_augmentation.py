"""Equivalent Kekule representations, chosen BEFORE forward corruption.

Random maximum-weight perfect matchings are not uniform over every Kekule
assignment. We sample a finite pool of verified, identity-preserving variants.
This is representation augmentation, not new molecules or new EST labels.
"""
import networkx as nx
import torch
from rdkit import Chem


def _identity(atom_states, bonds, orders, metadata):
    mol = Chem.RWMol()
    for state in atom_states.tolist():
        info = metadata[state]
        atom = Chem.Atom(info["symbol"])
        atom.SetFormalCharge(int(info["formal_charge"]))
        atom.SetNumExplicitHs(int(info["hydrogens"]))
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for (i, j), order in zip(bonds.tolist(), orders.tolist()):
        mol.AddBond(i, j, types[order])
    result = mol.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result, isomericSmiles=False)


def kekule_variants(atom_states, bonds, orders, aromatic, metadata, *,
                    attempts=8, seed=17041):
    """Keep per-atom aromatic double demand, including fused shared atoms.

    Non-S/D perimeters or demand > 1 are kept intact, not coerced into a
    benzene-like rule. A final whole-molecule RDKit identity check guards rare
    hetero/charge cases that local demand alone would not characterize.
    """
    if attempts < 0 or int(attempts) != attempts:
        raise ValueError("attempts must be a nonnegative integer")
    if bonds.shape != (len(orders), 2) or aromatic.shape != orders.shape:
        raise ValueError("Kekule bonds, orders and aromatic labels must align")
    canonical = orders.detach().cpu().long().clone()
    bonds = bonds.detach().cpu().long()
    aromatic = aromatic.detach().cpu().bool()
    original = _identity(atom_states.cpu(), bonds, canonical, metadata)
    graph = nx.Graph()
    for edge in torch.where(aromatic)[0].tolist():
        i, j = bonds[edge].tolist()
        graph.add_edge(i, j, edge=edge)
    eligible = []
    skipped = 0
    for nodes in nx.connected_components(graph):
        edges = [d["edge"] for _, _, d in graph.subgraph(nodes).edges(data=True)]
        demand = {i: 0 for i in nodes}
        for edge in edges:
            if canonical[edge] == 2:
                i, j = bonds[edge].tolist()
                demand[i] += 1
                demand[j] += 1
        if any(int(canonical[e]) not in (1, 2) for e in edges) or max(demand.values()) > 1:
            skipped += 1
            continue
        eligible.append((edges, {i for i, value in demand.items() if value == 1}))
    pool = {tuple(canonical.tolist()): canonical}
    generator = torch.Generator().manual_seed(seed)
    rejected = 0
    for _ in range(attempts):
        candidate = canonical.clone()
        for edges, demand_nodes in eligible:
            matching_graph = nx.Graph()
            matching_graph.add_nodes_from(sorted(demand_nodes))
            weights = torch.randn(len(edges), generator=generator).tolist()
            for edge, weight in zip(edges, weights):
                i, j = bonds[edge].tolist()
                if i in demand_nodes and j in demand_nodes:
                    matching_graph.add_edge(i, j, weight=weight, edge=edge)
            matching = nx.max_weight_matching(matching_graph, maxcardinality=True)
            if 2 * len(matching) != len(demand_nodes):
                raise ValueError("Stored Kekule demand has no perfect matching")
            candidate[edges] = 1
            for i, j in matching:
                candidate[matching_graph.edges[i, j]["edge"]] = 2
        try:
            identity = _identity(atom_states.cpu(), bonds, candidate, metadata)
        except (ValueError, RuntimeError):
            rejected += 1
            continue
        if identity != original:
            rejected += 1
            continue
        pool.setdefault(tuple(candidate.tolist()), candidate)
    return torch.stack(list(pool.values())), {
        "variants": len(pool), "unsupported_components_preserved": skipped,
        "identity_rejections": rejected,
    }
