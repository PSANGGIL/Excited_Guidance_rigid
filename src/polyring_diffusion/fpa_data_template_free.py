"""All-heavy-atom graphs augmented by template-free 3-10 member ring cells.

Cells carry geometry and cardinality only.  Chemical identity remains on atoms;
cell relations are 0=disjoint, 1=linked, 2=spiro, and shared_count+1 for 2-10 shared atoms.
"""
from __future__ import annotations

import bisect
import math
import torch
from torch import Tensor

from .hybrid_data import HybridRingAtomDataset


class TemplateFreeFPAEntityDataset(HybridRingAtomDataset):
    def __init__(self, *args, **kwargs):
        self.condition_mode = kwargs.pop("condition_mode", "full_spectrum")
        self.frame_augmentation = bool(kwargs.pop("frame_augmentation", False))
        self.sidecar_dir = kwargs.pop("sidecar_dir", "hydrogen_counts")
        self.exclude_mapping_failures = bool(kwargs.pop("exclude_mapping_failures", True))
        self.exclude_disconnected = bool(kwargs.pop("exclude_disconnected", True))
        self.aromatic_only = bool(kwargs.pop("aromatic_only", False))
        supplied_atom_map = kwargs.pop("atom_map", None)
        supplied_condition_stats = kwargs.pop("condition_stats", None)
        split = kwargs.get("split", args[1] if len(args) > 1 else "train")
        if self.condition_mode == "full_spectrum" and supplied_condition_stats is not None:
            kwargs["condition_stats"] = supplied_condition_stats
        super().__init__(*args, **kwargs)
        # Raw shards use a database-wide one-hot schema which may contain
        # elements that never occur in training. Empty output classes have no
        # learned positive examples, so their logits can win during sampling.
        source_atom_map = tuple(self.atom_map)
        observed = torch.zeros(len(source_atom_map), dtype=torch.bool)
        for shard in self.shards:
            observed |= shard["atom_types"].bool().any(0)
        if supplied_atom_map is None:
            active_atom_map = tuple(symbol for symbol, keep in
                zip(source_atom_map, observed.tolist()) if keep)
        else:
            active_atom_map = tuple(supplied_atom_map)
            unknown = set(active_atom_map)-set(source_atom_map)
            if unknown:
                raise ValueError(f"atom vocabulary is absent from shard schema: {sorted(unknown)}")
            split_only = [source_atom_map[i] for i in observed.nonzero().flatten().tolist()
                if source_atom_map[i] not in active_atom_map]
            if split_only:
                raise ValueError(f"split contains elements absent from training vocabulary: {split_only}")
        if not active_atom_map:
            raise ValueError("no observed atom types")
        source_to_active = torch.full((len(source_atom_map),), -1, dtype=torch.long)
        for active, symbol in enumerate(active_atom_map):
            source_to_active[source_atom_map.index(symbol)] = active
        self.source_atom_map = source_atom_map
        self.source_h_index = source_atom_map.index("H") if "H" in source_atom_map else None
        self._source_to_active = source_to_active
        self.atom_map = active_atom_map
        self.num_atom_types = len(active_atom_map)
        self.h_index = active_atom_map.index("H") if "H" in active_atom_map else None
        if self.condition_mode == "s1_t1":
            if supplied_condition_stats is not None:
                if supplied_condition_stats.get("mode") != "s1_t1":
                    raise ValueError("incompatible scalar condition statistics")
                self.condition_stats = supplied_condition_stats
            else:
                s1_energy = torch.cat([x["singlet_energies"][:, 0] for x in self.shards]).float()
                t1_energy = torch.cat([x["triplet_energies"][:, 0] for x in self.shards]).float()
                self.condition_stats = {"mode": "s1_t1",
                    "s1_energy_mean": float(s1_energy.mean()), "s1_energy_std": float(s1_energy.std().clamp_min(1e-6)),
                    "t1_energy_mean": float(t1_energy.mean()), "t1_energy_std": float(t1_energy.std().clamp_min(1e-6))}
            self.condition_dim = 2
        elif self.condition_mode != "full_spectrum":
            raise ValueError(f"unknown condition mode: {self.condition_mode}")
        info = next(x for x in self.manifest["splits"] if x["name"] == split)
        self.has_hydrogen_counts = True
        self.has_chirality_labels = True
        for shard, rel in zip(self.shards, info["shards"]):
            sidecar = self.root/self.sidecar_dir/rel
            if not sidecar.exists():
                self.has_hydrogen_counts = False
                break
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            self.has_chirality_labels &= "atom_chirality" in payload
            if not torch.equal(payload["atom_offsets"].long(), shard["atom_offsets"].long()):
                raise ValueError(f"hydrogen sidecar offsets mismatch: {sidecar}")
            shard["atom_hydrogens"] = payload["atom_hydrogens"].long()
            shard["atom_hydrogen_mask"] = payload.get("atom_hydrogen_mask", torch.ones_like(payload["atom_hydrogens"], dtype=torch.bool))
            shard["atom_radicals"] = payload.get("atom_radicals", torch.zeros_like(payload["atom_hydrogens"])).long()
            shard["atom_chirality"] = payload.get("atom_chirality", torch.zeros_like(payload["atom_hydrogens"])).long()
            shard["atom_aromatic"] = payload.get("atom_aromatic")
            if "atom_formal_charges" in payload: shard["atom_charges"] = payload["atom_formal_charges"].long()
            shard["bond_aromatic"] = payload.get("bond_aromatic")
            if "bond_types" in payload:
                if "bond_offsets" in payload and not torch.equal(payload["bond_offsets"].long(), shard["bond_offsets"].long()): raise ValueError(f"bond sidecar offsets mismatch: {sidecar}")
                shard["bond_types"] = payload["bond_types"].long()
        by_id = sorted(self.manifest["templates"], key=lambda x: x["id"])
        self.template_signatures = [x["signature"] for x in by_id]
        self.selected_template_ids = {
            i for i, sig in enumerate(self.template_signatures) if 3 <= len(sig) <= 10
        }
        self._physical_indices = []
        selected_s1, selected_t1 = [], []
        physical = 0
        for shard in self.shards:
            for row in range(shard["length"]):
                a, b = self._bounds(shard["atom_offsets"], row)
                exact = bool(shard["atom_hydrogen_mask"][a:b].all()) if self.has_hydrogen_counts else False
                smiles = shard["smiles"][row].decode() if isinstance(shard["smiles"][row], bytes) else str(shard["smiles"][row])
                connected = "." not in smiles
                aromatic = bool(shard["atom_aromatic"][a:b].any()) if shard.get("atom_aromatic") is not None else False
                if ((exact or not self.exclude_mapping_failures)
                        and (connected or not self.exclude_disconnected)
                        and (aromatic or not self.aromatic_only)):
                    self._physical_indices.append(physical+row)
                    if self.condition_mode == "s1_t1":
                        selected_s1.append(shard["singlet_energies"][row, 0].float())
                        selected_t1.append(shard["triplet_energies"][row, 0].float())
            physical += shard["length"]
        if not self._physical_indices:
            raise ValueError("dataset filter selected no molecules")
        # Train aromatic-only conditioning in its own distribution. Validation
        # receives these training statistics through supplied_condition_stats.
        if self.aromatic_only and self.condition_mode == "s1_t1" and supplied_condition_stats is None:
            s1_energy, t1_energy = torch.stack(selected_s1), torch.stack(selected_t1)
            self.condition_stats = {"mode": "s1_t1",
                "s1_energy_mean": float(s1_energy.mean()), "s1_energy_std": float(s1_energy.std().clamp_min(1e-6)),
                "t1_energy_mean": float(t1_energy.mean()), "t1_energy_std": float(t1_energy.std().clamp_min(1e-6))}
        self.max_cell_ports = max(
            len(self.template_signatures[i]) for i in self.selected_template_ids
        )
        self.max_heavy_atoms = max(
            self._heavy_count(s, i) for s in self.shards for i in range(s["length"])
        )
        self.max_fpa_cells = max(
            1, max(self._cell_count(s, i) for s in self.shards for i in range(s["length"]))
        )

    def __len__(self):
        return len(self._physical_indices) if hasattr(self, "_physical_indices") else super().__len__()

    def physical_index(self, index: int) -> int:
        return self._physical_indices[index] if hasattr(self, "_physical_indices") else index

    def heavy_count_at(self, index: int) -> int:
        index = self.physical_index(index)
        si = bisect.bisect_right(self.ends, index)
        base = 0 if si == 0 else self.ends[si-1]
        return self._heavy_count(self.shards[si], index-base)

    def _heavy_count(self, shard, i: int) -> int:
        a, b = self._bounds(shard["atom_offsets"], i)
        types = shard["atom_types"][a:b].to(torch.uint8).argmax(-1)
        return int((types != self.source_h_index).sum()) if self.source_h_index is not None else len(types)

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
        index = self.physical_index(index)
        si = bisect.bisect_right(self.ends, index)
        base = 0 if si == 0 else self.ends[si-1]
        i, shard = index-base, self.shards[si]
        a, b = self._bounds(shard["atom_offsets"], i)
        ba, bb = self._bounds(shard["bond_offsets"], i)
        source_types = shard["atom_types"][a:b].to(torch.uint8).argmax(-1).long()
        heavy_ids = [j for j, t in enumerate(source_types.tolist())
            if self.source_h_index is None or t != self.source_h_index]
        all_types = self._source_to_active[source_types]
        if bool((all_types < 0).any()):
            missing = sorted({self.source_atom_map[int(source_types[i])]
                for i in (all_types < 0).nonzero().flatten().tolist()})
            raise ValueError(f"sample contains elements absent from training vocabulary: {missing}")
        heavy_lookup = {old: new for new, old in enumerate(heavy_ids)}
        h = len(heavy_ids)
        bonds = torch.zeros((h, h), dtype=torch.long)
        for edge_index, (edge, kind) in enumerate(zip(shard["bond_indices"][ba:bb].long().tolist(), shard["bond_types"][ba:bb].long().tolist())):
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
            # Fix the arbitrary SVD normal sign using the ordered ring, then
            # expose the physically equivalent D_n gauges during training.
            if len(atom_ids) >= 3:
                winding = torch.cross(xyz[1]-xyz[0], xyz[2]-xyz[1], dim=0)
                if float((normal*winding).sum()) < 0: normal = -normal
            bitangent = torch.cross(normal, tangent, dim=0)
            if self.frame_augmentation:
                angle = 2*math.pi*int(torch.randint(max(1, len(atom_ids)), (1,)))/max(1, len(atom_ids))
                c, s = math.cos(angle), math.sin(angle)
                tangent, bitangent = c*tangent+s*bitangent, -s*tangent+c*bitangent
                if bool(torch.randint(2, (1,))):
                    bitangent, normal = -bitangent, -normal
            centers.append(center); normals.append(normal); tangents.append(tangent)
            radii.append(centered.norm(dim=-1).mean()); sizes.append(len(atom_ids))
            mapped_ring = [heavy_lookup[a] for a in ring_lists[old] if a in heavy_lookup]
            aromatic.append(bool(mapped_ring) and all(int(bonds[mapped_ring[j], mapped_ring[(j+1) % len(mapped_ring)]]) == 4 for j in range(len(mapped_ring))))

        ids = torch.tensor(heavy_ids, dtype=torch.long)
        zero3 = positions.new_zeros((0, 3))
        out = {
            "atom_types": all_types[ids],
            "atom_charges": shard["atom_charges"][a:b][ids].long(),
            "atom_hydrogens": shard["atom_hydrogens"][a:b][ids].long() if "atom_hydrogens" in shard else torch.zeros(len(ids), dtype=torch.long),
            "atom_hydrogen_mask": shard["atom_hydrogen_mask"][a:b][ids].bool() if "atom_hydrogen_mask" in shard else torch.zeros(len(ids), dtype=torch.bool),
            "atom_radicals": shard["atom_radicals"][a:b][ids].long() if "atom_radicals" in shard else torch.zeros(len(ids), dtype=torch.long),
            "atom_chirality": shard["atom_chirality"][a:b][ids].long() if "atom_chirality" in shard else torch.zeros(len(ids), dtype=torch.long),
            "atom_aromatic": shard["atom_aromatic"][a:b][ids].bool() if shard.get("atom_aromatic") is not None else bonds.eq(4).any(-1),
            "atom_positions": positions[ids],
            "atom_bond_types": bonds,
            "atom_bond_aromatic": torch.zeros((h, h), dtype=torch.bool),
            "cell_sizes": torch.tensor(sizes, dtype=torch.long),
            "cell_aromatic": torch.tensor(aromatic, dtype=torch.long),
            "cell_centers": torch.stack(centers) if centers else zero3,
            "cell_normals": torch.stack(normals) if normals else zero3,
            "cell_tangents": torch.stack(tangents) if tangents else zero3,
            "cell_radii": torch.stack(radii) if radii else positions.new_zeros(0),
            "cell_membership": membership,
            "cell_relations": relations,
            "property": super().__getitem__(index)["property"] if self.condition_mode == "full_spectrum" else torch.empty(2),
            "source_row": torch.tensor(int(shard["source_rows"][i])),
            "source_smiles": shard["smiles"][i],
        }
        if shard.get("bond_aromatic") is not None:
            for edge, flag in zip(shard["bond_indices"][ba:bb].long().tolist(), shard["bond_aromatic"][ba:bb].bool().tolist()):
                u, v = map(int, edge)
                if flag and u in heavy_lookup and v in heavy_lookup:
                    u, v = heavy_lookup[u], heavy_lookup[v]; out["atom_bond_aromatic"][u, v] = out["atom_bond_aromatic"][v, u] = True
        else:
            out["atom_bond_aromatic"] = bonds.eq(4)
        if self.condition_mode == "s1_t1":
            stats = self.condition_stats
            s1_energy = shard["singlet_energies"][i, 0].float()
            t1_energy = shard["triplet_energies"][i, 0].float()
            out["property"] = torch.stack(((s1_energy-stats["s1_energy_mean"])/stats["s1_energy_std"],
                (t1_energy-stats["t1_energy_mean"])/stats["t1_energy_std"]))
        return out


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
        "atom_chirality": torch.zeros((b, na), dtype=torch.long),
        "atom_aromatic": torch.zeros((b, na), dtype=torch.bool),
        "atom_positions": torch.zeros((b, na, 3)),
        "atom_mask": torch.zeros((b, na), dtype=torch.bool),
        "atom_bond_types": torch.zeros((b, na, na), dtype=torch.long),
        "atom_bond_aromatic": torch.zeros((b, na, na), dtype=torch.bool),
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
        out["atom_chirality"][row, :a] = x["atom_chirality"]
        out["atom_aromatic"][row, :a] = x["atom_aromatic"]
        out["atom_positions"][row, :a] = x["atom_positions"]-center
        out["atom_mask"][row, :a] = True
        out["atom_bond_types"][row, :a, :a] = x["atom_bond_types"]
        out["atom_bond_aromatic"][row, :a, :a] = x["atom_bond_aromatic"]
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
