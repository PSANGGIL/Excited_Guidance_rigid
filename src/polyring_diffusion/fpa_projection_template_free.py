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
                    candidates = [a for a in range(na) if a not in shared]
                    if not candidates: break
                    atom = max(candidates, key=lambda a: float(score[left, a]+score[right, a]))
                    for cell in (left, right):
                        if atom not in chosen[cell] and len(chosen[cell]) >= int(sizes[cell]):
                            removable = chosen[cell]-shared
                            if not removable: break
                            chosen[cell].remove(min(removable, key=lambda a: float(score[cell, a])))
                        chosen[cell].add(atom)
                    shared = chosen[left] & chosen[right]; changed = True
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
def project_template_free_fpa(sample: dict, prediction: dict, atom_map):
    mscore = prediction["membership_logits"].softmax(-1)[..., 1]
    bprob = prediction["bond_logits"].softmax(-1)
    valid, repairs, reasons = [], [], []
    for row in range(len(sample["atom_types"])):
        na = int(sample["atom_mask"][row].sum()); nc = int(sample["cell_mask"][row].sum())
        reason = 0; changed = 0
        pos = sample["positions"][row, :na]
        sizes = sample["cell_sizes"][row, :nc].clamp(3, 10)
        relations = sample["cell_relations"][row, :nc, :nc]
        chosen = _memberships(mscore[row, :nc, :na], sizes, relations)
        for cell in range(nc):
            if len(chosen[cell]) != int(sizes[cell]): reason |= 1
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
        for cell, atoms in enumerate(chosen):
            if len(atoms) < 3: reason |= 1; continue
            atoms = sorted(atoms); predicted_edges = []
            for ai, a in enumerate(atoms):
                for b in atoms[ai+1:]:
                    kind = int(torch.argmax(bprob[row, a, b]))
                    if kind: predicted_edges.append((a, b, kind))
            find_ring, union_ring = _components(na, [])
            for a, b, _ in predicted_edges: union_ring(a, b)
            has_cycle = len(predicted_edges) >= len(atoms) and len({find_ring(a) for a in atoms}) == 1
            if has_cycle:
                for a, b, kind in predicted_edges: mandatory[(a, b)] = kind
                continue
            tangent, bitangent = frames[cell, 0], frames[cell, 1]
            ordered = sorted(atoms, key=lambda a: math.atan2(float(((pos[a]-centers[cell])*bitangent).sum()), float(((pos[a]-centers[cell])*tangent).sum())))
            aromatic = bool(sample["cell_aromatic"][row, cell])
            for i in range(len(ordered)):
                a, b = sorted((ordered[i], ordered[(i+1)%len(ordered)]))
                kind = 4 if aromatic else int(torch.argmax(bprob[row, a, b, 1:]))+1
                mandatory[(a, b)] = kind
        # Relation-specific topology: fused cells share an edge; linked cells have one direct inter-ring bond.
        for left in range(nc):
            for right in range(left+1, nc):
                relation = int(relations[left, right])
                shared = sorted(chosen[left] & chosen[right])
                if relation == 3 and len(shared) == 2:
                    i, j = shared
                    kind = int(torch.argmax(bprob[row, i, j]))
                    if kind == 0: kind = 4 if bool(sample["cell_aromatic"][row, left] and sample["cell_aromatic"][row, right]) else 1
                    mandatory[(i, j)] = kind
                elif relation == 1:
                    pairs = [(float(1-bprob[row, i, j, 0]), i, j) for i in chosen[left] for j in chosen[right] if i != j]
                    if pairs:
                        _, i, j = max(pairs)
                        kind = int(torch.argmax(bprob[row, i, j, 1:]))+1
                        mandatory[tuple(sorted((i, j)))] = kind
                    else:
                        reason |= 16
        projected = torch.zeros((na, na), dtype=torch.long, device=pos.device)
        used = torch.zeros(na, device=pos.device)
        limits = torch.tensor([float(element_record(atom_map[int(t)]).default_valence)+(1. if atom_map[int(t)] in ("N", "O") and int(q)-2 > 0 else 0.) for t, q in zip(sample["atom_types"][row, :na], sample["charges"][row, :na])], device=pos.device)
        limits = (limits-sample["hydrogens"][row, :na].float()-sample["radicals"][row, :na].float()).clamp_min(0)
        order_value = (0., 1., 2., 3., 1.)
        edges = []
        for (i, j), kind in mandatory.items():
            order = order_value[kind]; projected[i, j] = projected[j, i] = kind
            used[i] += order; used[j] += order; edges.append((i, j))
        find, union = _components(na, edges)
        candidates = []
        for i in range(na):
            for j in range(i+1, na):
                if (i, j) in mandatory: continue
                kind = int(torch.argmax(bprob[row, i, j])); order = order_value[kind]
                candidates.append((float(1-bprob[row, i, j, 0]), i, j, kind, order))
        candidates.sort(reverse=True)
        for connect_only in (True, False):
            for confidence, i, j, kind, order in candidates:
                if projected[i, j] != 0 or kind == 0: continue
                if connect_only and find(i) == find(j): continue
                if not connect_only and confidence < .5: continue
                if used[i]+order > limits[i] or used[j]+order > limits[j]: continue
                projected[i, j] = projected[j, i] = kind
                used[i] += order; used[j] += order; union(i, j); changed += 1
        if na and len({find(i) for i in range(na)}) > 1: reason |= 4
        if bool((used > limits+1e-5).any()): reason |= 8
        sample["bonds"][row, :na, :na] = projected
        valid.append(reason == 0); repairs.append(changed); reasons.append(reason)
    device = sample["atom_types"].device
    sample["projection_valid"] = torch.tensor(valid, device=device)
    sample["projection_repairs"] = torch.tensor(repairs, device=device)
    sample["projection_reason"] = torch.tensor(reasons, device=device)
    return sample
