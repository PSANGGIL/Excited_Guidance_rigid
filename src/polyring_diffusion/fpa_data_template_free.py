"""All-heavy-atom graphs augmented by template-free 3-10 member ring cells.

Cells carry geometry and cardinality only.  Chemical identity remains on atoms;
cell relations are 0=disjoint, 1=linked, 2=spiro, and shared_count+1 for 2-10 shared atoms.
"""
from __future__ import annotations

import bisect
import torch
from torch import Tensor

from .hybrid_data import HybridRingAtomDataset


class TemplateFreeFPAEntityDataset(HybridRingAtomDataset):
    def __init__(self, *args, **kwargs):
        split = kwargs.get("split", args[1] if len(args) > 1 else "train")
        super().__init__(*args, **kwargs)
        info = next(x for x in self.manifest["splits"] if x["name"] == split)
        self.has_hydrogen_counts = True
        for shard, rel in zip(self.shards, info["shards"]):
            sidecar = self.root/"hydrogen_counts"/rel
            if not sidecar.exists():
                self.has_hydrogen_counts = False
                break
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            if not torch.equal(payload["atom_offsets"].long(), shard["atom_offsets"].long()):
                raise ValueError(f"hydrogen sidecar offsets mismatch: {sidecar}")
            shard["atom_hydrogens"] = payload["atom_hydrogens"].long()
            shard["atom_hydrogen_mask"] = payload.get("atom_hydrogen_mask", torch.ones_like(payload["atom_hydrogens"], dtype=torch.bool))
            shard["atom_radicals"] = payload.get("atom_radicals", torch.zeros_like(payload["atom_hydrogens"])).long()
            if "atom_formal_charges" in payload: shard["atom_charges"] = payload["atom_formal_charges"].long()
            if "bond_types" in payload:
                if "bond_offsets" in payload and not torch.equal(payload["bond_offsets"].long(), shard["bond_offsets"].long()): raise ValueError(f"bond sidecar offsets mismatch: {sidecar}")
                shard["bond_types"] = payload["bond_types"].long()
        by_id = sorted(self.manifest["templates"], key=lambda x: x["id"])
        self.template_signatures = [x["signature"] for x in by_id]
        self.selected_template_ids = {
            i for i, sig in enumerate(self.template_signatures) if 3 <= len(sig) <= 10
        }
        self.max_cell_ports = max(
            len(self.template_signatures[i]) for i in self.selected_template_ids
        )
        self.max_heavy_atoms = max(
            self._heavy_count(s, i) for s in self.shards for i in range(s["length"])
        )
        self.max_fpa_cells = max(
            1, max(self._cell_count(s, i) for s in self.shards for i in range(s["length"]))
        )

    def _heavy_count(self, shard, i: int) -> int:
        a, b = self._bounds(shard["atom_offsets"], i)
        types = shard["atom_types"][a:b].to(torch.uint8).argmax(-1)
        return int((types != self.h_index).sum()) if self.h_index is not None else len(types)

    def _ring_lists(self, shard, i: int) -> list[list[int]]:
        ra, rb = self._bounds(shard["ring_atom_offsets"], i)
        flat = shard["ring_atom_indices"][ra:rb].long()
        offsets = shard["per_ring_atom_offsets"][i].long()
        return [flat[int(offsets[j]):int(offsets[j + 1])].tolist() for j in range(len(offsets)-1)]

    def _selected_rings(self, shard, i: int) -> list[int]:
        rs, re = self._bounds(shard["ring_offsets"], i)
        return [
            j for j, tid in enumerate(shard["ring_types"][rs:re].tolist())
            if int(tid) in self.selected_template_ids
        ]

    def _cell_count(self, shard, i: int) -> int:
        return len(self._selected_rings(shard, i))

    def __getitem__(self, index):
        si = bisect.bisect_right(self.ends, index)
        base = 0 if si == 0 else self.ends[si-1]
        i, shard = index-base, self.shards[si]
        a, b = self._bounds(shard["atom_offsets"], i)
        ba, bb = self._bounds(shard["bond_offsets"], i)
        all_types = shard["atom_types"][a:b].to(torch.uint8).argmax(-1).long()
        heavy_ids = [j for j, t in enumerate(all_types.tolist()) if self.h_index is None or t != self.h_index]
        heavy_lookup = {old: new for new, old in enumerate(heavy_ids)}
        h = len(heavy_ids)
        bonds = torch.zeros((h, h), dtype=torch.long)
        for edge, kind in zip(shard["bond_indices"][ba:bb].long().tolist(), shard["bond_types"][ba:bb].long().tolist()):
            u, v = map(int, edge)
            if u in heavy_lookup and v in heavy_lookup:
                u, v = heavy_lookup[u], heavy_lookup[v]
                bonds[u, v] = bonds[v, u] = int(kind)

        ring_lists = self._ring_lists(shard, i)
        keep = self._selected_rings(shard, i)
        c = len(keep)
        membership = torch.zeros((c, h), dtype=torch.bool)
        for new, old in enumerate(keep):
            for atom in ring_lists[old]:
                if atom in heavy_lookup:
                    membership[new, heavy_lookup[atom]] = True

        relations = torch.zeros((c, c), dtype=torch.long)
        for left in range(c):
            for right in range(left+1, c):
                shared = int((membership[left] & membership[right]).sum())
                if shared >= 2:
                    relation = min(shared, 10)+1
                elif shared == 1:
                    relation = 2
                else:
                    linked = bool((bonds[membership[left]][:, membership[right]] > 0).any())
                    relation = 1 if linked else 0
                relations[left, right] = relations[right, left] = relation

        positions = shard["atom_positions"][a:b].float()
        ring_start, _ = self._bounds(shard["ring_offsets"], i)
        centers, normals, tangents, radii, sizes, aromatic = [], [], [], [], [], []
        for old in keep:
            atom_ids = torch.tensor(ring_lists[old], dtype=torch.long)
            xyz = positions[atom_ids]
            center = xyz.mean(0)
            centered = xyz-center
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            normal = torch.nn.functional.normalize(vh[-1], dim=0)
            tangent = centered[0]-(centered[0]*normal).sum()*normal
            tangent = torch.nn.functional.normalize(tangent, dim=0)
            centers.append(center); normals.append(normal); tangents.append(tangent)
            radii.append(centered.norm(dim=-1).mean()); sizes.append(len(atom_ids))
            mapped_ring = [heavy_lookup[a] for a in ring_lists[old] if a in heavy_lookup]
            aromatic.append(bool(mapped_ring) and all(int(bonds[mapped_ring[j], mapped_ring[(j+1) % len(mapped_ring)]]) == 4 for j in range(len(mapped_ring))))

        ids = torch.tensor(heavy_ids, dtype=torch.long)
        zero3 = positions.new_zeros((0, 3))
        return {
            "atom_types": all_types[ids],
            "atom_charges": shard["atom_charges"][a:b][ids].long(),
            "atom_hydrogens": shard["atom_hydrogens"][a:b][ids].long() if "atom_hydrogens" in shard else torch.zeros(len(ids), dtype=torch.long),
            "atom_hydrogen_mask": shard["atom_hydrogen_mask"][a:b][ids].bool() if "atom_hydrogen_mask" in shard else torch.zeros(len(ids), dtype=torch.bool),
            "atom_radicals": shard["atom_radicals"][a:b][ids].long() if "atom_radicals" in shard else torch.zeros(len(ids), dtype=torch.long),
            "atom_positions": positions[ids],
            "atom_bond_types": bonds,
            "cell_sizes": torch.tensor(sizes, dtype=torch.long),
            "cell_aromatic": torch.tensor(aromatic, dtype=torch.long),
            "cell_centers": torch.stack(centers) if centers else zero3,
            "cell_normals": torch.stack(normals) if normals else zero3,
            "cell_tangents": torch.stack(tangents) if tangents else zero3,
            "cell_radii": torch.stack(radii) if radii else positions.new_zeros(0),
            "cell_membership": membership,
            "cell_relations": relations,
            "property": super().__getitem__(index)["property"],
            "source_row": torch.tensor(int(shard["source_rows"][i])),
        }


def collate_template_free_fpa(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    b = len(batch)
    na = max(1, max(len(x["atom_types"]) for x in batch))
    nc = max(1, max(len(x["cell_sizes"]) for x in batch))
    out = {
        "atom_types": torch.zeros((b, na), dtype=torch.long),
        "atom_charges": torch.zeros((b, na), dtype=torch.long),
        "atom_hydrogens": torch.zeros((b, na), dtype=torch.long),
        "atom_hydrogen_mask": torch.zeros((b, na), dtype=torch.bool),
        "atom_radicals": torch.zeros((b, na), dtype=torch.long),
        "atom_positions": torch.zeros((b, na, 3)),
        "atom_mask": torch.zeros((b, na), dtype=torch.bool),
        "atom_bond_types": torch.zeros((b, na, na), dtype=torch.long),
        "cell_sizes": torch.zeros((b, nc), dtype=torch.long),
        "cell_aromatic": torch.zeros((b, nc), dtype=torch.long),
        "cell_centers": torch.zeros((b, nc, 3)),
        "cell_frames": torch.eye(3).reshape(1, 1, 3, 3).repeat(b, nc, 1, 1),
        "cell_radii": torch.ones((b, nc)),
        "cell_mask": torch.zeros((b, nc), dtype=torch.bool),
        "cell_membership": torch.zeros((b, nc, na), dtype=torch.bool),
        "cell_relations": torch.zeros((b, nc, nc), dtype=torch.long),
        "property": torch.stack([x["property"] for x in batch]),
        "source_row": torch.stack([x["source_row"] for x in batch]),
    }
    for row, x in enumerate(batch):
        a, c = len(x["atom_types"]), len(x["cell_sizes"])
        center = x["atom_positions"].mean(0, keepdim=True)
        out["atom_types"][row, :a] = x["atom_types"]
        out["atom_charges"][row, :a] = x["atom_charges"]
        out["atom_hydrogens"][row, :a] = x["atom_hydrogens"]
        out["atom_hydrogen_mask"][row, :a] = x["atom_hydrogen_mask"]
        out["atom_radicals"][row, :a] = x["atom_radicals"]
        out["atom_positions"][row, :a] = x["atom_positions"]-center
        out["atom_mask"][row, :a] = True
        out["atom_bond_types"][row, :a, :a] = x["atom_bond_types"]
        if c:
            out["cell_sizes"][row, :c] = x["cell_sizes"]
            out["cell_aromatic"][row, :c] = x["cell_aromatic"]
            out["cell_centers"][row, :c] = x["cell_centers"]-center
            out["cell_radii"][row, :c] = x["cell_radii"]
            out["cell_mask"][row, :c] = True
            out["cell_membership"][row, :c, :a] = x["cell_membership"]
            out["cell_relations"][row, :c, :c] = x["cell_relations"]
            normal, tangent = x["cell_normals"], x["cell_tangents"]
            bitangent = torch.linalg.cross(normal, tangent, dim=-1)
            out["cell_frames"][row, :c] = torch.stack((tangent, bitangent, normal), -2)
    return out
