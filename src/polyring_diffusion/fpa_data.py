"""Heavy-atom graphs augmented with cells only for fused aromatic systems.

An FPA cell is an auxiliary geometric object, not an output atom.  Every heavy
atom remains unique; cells carry a pose, size and fusion ports and are removed
before chemistry/RDKit post-processing.
"""
from __future__ import annotations

import bisect
from pathlib import Path
import torch
from torch import Tensor

from .hybrid_data import HybridRingAtomDataset


def _components(edges: list[tuple[int, int]], nodes: set[int]) -> list[list[int]]:
    parent = {i: i for i in nodes}
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for a, b in edges:
        union(a, b)
    groups: dict[int, list[int]] = {}
    for i in nodes:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class FPAEntityDataset(HybridRingAtomDataset):
    """Return all heavy atoms plus cells for fused aromatic rings only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        by_id = sorted(self.manifest['templates'], key=lambda x: x['id'])
        self.template_signatures = [x['signature'] for x in by_id]
        self.max_cell_ports = max(len(x) for x in self.template_signatures)
        self.max_heavy_atoms = max(self._heavy_count(s, i) for s in self.shards for i in range(s['length']))
        self.max_fpa_cells = max(1, max(self._fpa_count(s, i) for s in self.shards for i in range(s['length'])))

    def _heavy_count(self, shard, i: int) -> int:
        a, b = self._bounds(shard['atom_offsets'], i)
        types = shard['atom_types'][a:b].to(torch.uint8).argmax(-1)
        return int((types != self.h_index).sum()) if self.h_index is not None else len(types)

    def _ring_lists(self, shard, i: int) -> list[list[int]]:
        ra, rb = self._bounds(shard['ring_atom_offsets'], i)
        flat = shard['ring_atom_indices'][ra:rb].long()
        offsets = shard['per_ring_atom_offsets'][i].long()
        return [flat[int(offsets[j]):int(offsets[j + 1])].tolist() for j in range(len(offsets) - 1)]

    def _fpa_rings(self, shard, i: int) -> tuple[list[int], list[tuple[int, int]]]:
        rs, re = self._bounds(shard['ring_offsets'], i)
        rings = self._ring_lists(shard, i)
        aromatic = {
            j for j, tid in enumerate(shard['ring_types'][rs:re].tolist())
            if self.template_signatures[int(tid)] and all(bool(row[2]) for row in self.template_signatures[int(tid)])
        }
        fused = []
        for left in sorted(aromatic):
            for right in sorted(aromatic):
                if left < right and len(set(rings[left]) & set(rings[right])) >= 2:
                    fused.append((left, right))
        keep = {r for component in _components(fused, aromatic) if len(component) > 1 for r in component}
        return sorted(keep), [(a, b) for a, b in fused if a in keep and b in keep]

    def _fpa_count(self, shard, i: int) -> int:
        return len(self._fpa_rings(shard, i)[0])

    @staticmethod
    def _edge_port(ring: list[int], shared: set[int]) -> int:
        n = len(ring)
        for p in range(n):
            if ring[p] in shared and ring[(p + 1) % n] in shared:
                return p
        return next((p for p, atom in enumerate(ring) if atom in shared), 0)

    def __getitem__(self, index):
        si = bisect.bisect_right(self.ends, index)
        base = 0 if si == 0 else self.ends[si - 1]
        i, shard = index - base, self.shards[si]
        a, b = self._bounds(shard['atom_offsets'], i)
        ba, bb = self._bounds(shard['bond_offsets'], i)
        rs, re = self._bounds(shard['ring_offsets'], i)
        all_types = shard['atom_types'][a:b].to(torch.uint8).argmax(-1).long()
        heavy_ids = [j for j, t in enumerate(all_types.tolist()) if self.h_index is None or t != self.h_index]
        heavy_lookup = {old: new for new, old in enumerate(heavy_ids)}
        h = len(heavy_ids)
        bonds = torch.zeros((h, h), dtype=torch.long)
        for edge, kind in zip(shard['bond_indices'][ba:bb].long().tolist(), shard['bond_types'][ba:bb].long().tolist()):
            u, v = map(int, edge)
            if u in heavy_lookup and v in heavy_lookup:
                u, v = heavy_lookup[u], heavy_lookup[v]
                bonds[u, v] = bonds[v, u] = int(kind)

        ring_lists = self._ring_lists(shard, i)
        keep, fused = self._fpa_rings(shard, i)
        cell_lookup = {old: new for new, old in enumerate(keep)}
        c = len(keep)
        membership = torch.zeros((c, h), dtype=torch.bool)
        fusion = torch.zeros((c, c), dtype=torch.long)
        fusion_ports = torch.zeros((c, c, 2), dtype=torch.long)
        for old, new in cell_lookup.items():
            for atom in ring_lists[old]:
                if atom in heavy_lookup:
                    membership[new, heavy_lookup[atom]] = True
        for left, right in fused:
            li, ri = cell_lookup[left], cell_lookup[right]
            shared = set(ring_lists[left]) & set(ring_lists[right])
            fusion[li, ri] = fusion[ri, li] = 1
            lp = self._edge_port(ring_lists[left], shared)
            rp = self._edge_port(ring_lists[right], shared)
            fusion_ports[li, ri] = torch.tensor([lp, rp])
            fusion_ports[ri, li] = torch.tensor([rp, lp])

        positions = shard['atom_positions'][a:b].float()
        centers, normals, tangents, radii, sizes, tids = [], [], [], [], [], []
        for old in keep:
            atom_ids = torch.tensor(ring_lists[old], dtype=torch.long)
            xyz = positions[atom_ids]
            center = xyz.mean(0)
            centered = xyz - center
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            normal = torch.nn.functional.normalize(vh[-1], dim=0)
            tangent = torch.nn.functional.normalize(centered[0] - (centered[0] * normal).sum() * normal, dim=0)
            centers.append(center); normals.append(normal); tangents.append(tangent)
            radii.append(centered.norm(dim=-1).mean()); sizes.append(len(atom_ids)); tids.append(int(shard['ring_types'][rs + old]))

        ids = torch.tensor(heavy_ids, dtype=torch.long)
        zero3 = positions.new_zeros((0, 3))
        return {
            'atom_types': all_types[ids], 'atom_charges': shard['atom_charges'][a:b][ids].long(),
            'atom_positions': positions[ids], 'atom_bond_types': bonds,
            'cell_types': torch.tensor(tids, dtype=torch.long), 'cell_sizes': torch.tensor(sizes, dtype=torch.long),
            'cell_centers': torch.stack(centers) if centers else zero3,
            'cell_normals': torch.stack(normals) if normals else zero3,
            'cell_tangents': torch.stack(tangents) if tangents else zero3,
            'cell_radii': torch.stack(radii) if radii else positions.new_zeros(0),
            'cell_membership': membership, 'fusion_types': fusion, 'fusion_ports': fusion_ports,
            'property': super().__getitem__(index)['property'], 'source_row': torch.tensor(int(shard['source_rows'][i])),
        }


def collate_fpa(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    b = len(batch); na = max(1, max(len(x['atom_types']) for x in batch)); nc = max(1, max(len(x['cell_types']) for x in batch))
    out = {
        'atom_types': torch.zeros((b, na), dtype=torch.long), 'atom_charges': torch.zeros((b, na), dtype=torch.long),
        'atom_positions': torch.zeros((b, na, 3)), 'atom_mask': torch.zeros((b, na), dtype=torch.bool),
        'atom_bond_types': torch.zeros((b, na, na), dtype=torch.long),
        'cell_types': torch.zeros((b, nc), dtype=torch.long), 'cell_sizes': torch.zeros((b, nc), dtype=torch.long),
        'cell_centers': torch.zeros((b, nc, 3)), 'cell_normals': torch.zeros((b, nc, 3)),
        'cell_tangents': torch.zeros((b, nc, 3)), 'cell_radii': torch.ones((b, nc)),
        'cell_mask': torch.zeros((b, nc), dtype=torch.bool), 'cell_membership': torch.zeros((b, nc, na), dtype=torch.bool),
        'fusion_types': torch.zeros((b, nc, nc), dtype=torch.long), 'fusion_ports': torch.zeros((b, nc, nc, 2), dtype=torch.long),
        'property': torch.stack([x['property'] for x in batch]), 'source_row': torch.stack([x['source_row'] for x in batch]),
    }
    for row, x in enumerate(batch):
        a, c = len(x['atom_types']), len(x['cell_types'])
        center = x['atom_positions'].mean(0, keepdim=True)
        out['atom_types'][row, :a] = x['atom_types']; out['atom_charges'][row, :a] = x['atom_charges']
        out['atom_positions'][row, :a] = x['atom_positions'] - center; out['atom_mask'][row, :a] = True
        out['atom_bond_types'][row, :a, :a] = x['atom_bond_types']
        if c:
            for key in ('cell_types', 'cell_sizes', 'cell_normals', 'cell_tangents', 'cell_radii'):
                out[key][row, :c] = x[key]
            out['cell_centers'][row, :c] = x['cell_centers'] - center; out['cell_mask'][row, :c] = True
            out['cell_membership'][row, :c, :a] = x['cell_membership']
            out['fusion_types'][row, :c, :c] = x['fusion_types']; out['fusion_ports'][row, :c, :c] = x['fusion_ports']
    return out
