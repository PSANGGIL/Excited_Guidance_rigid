"""Small reproducible validation-generation panel, separate from denoising loss.

No XTB jobs are launched during optimization. SDF/targets are exported for the
existing external XTB/sTDA workflow; fidelity is explicitly UNMEASURED here.
"""
import json
import random
import warnings
import uuid
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from molguidance.analysis.strict_generation import strict_molecule_record, summarize_records
from molguidance.model_utils.provenance import sha256_file, state_dict_sha256


class GenerationValidation(Callback):
    def __init__(self, output_dir, data_dir, every_n_epochs=50, n_samples=64,
                 batch_size=8, n_timesteps=100, seed=17041):
        super().__init__()
        if min(every_n_epochs, n_samples, batch_size) < 1 or n_timesteps < 2:
            raise ValueError("invalid validation generation dimensions")
        self.output_dir, self.data_dir = Path(output_dir), Path(data_dir).resolve()
        self.every_n_epochs, self.n_samples = int(every_n_epochs), int(n_samples)
        self.batch_size, self.n_timesteps, self.seed = int(batch_size), int(n_timesteps), int(seed)

    def on_fit_start(self, trainer, pl_module):
        from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
        if trainer.world_size != 1 or not isinstance(pl_module, ClassifierFreeGuidance):
            raise ValueError("generation validation currently supports single-device CFG only")

    @torch.no_grad()
    def evaluate(self, model, epoch, step):
        base = self.output_dir / f"epoch_{epoch:05d}_step_{step:07d}"
        directory = base
        attempt = 0
        while True:
            try:
                directory.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                attempt += 1
                directory = base.with_name(f"{base.name}_attempt_{attempt:03d}")
        val = torch.load(self.data_dir / "val_data_processed.pt", map_location="cpu", weights_only=False)
        order = torch.randperm(len(val["node_idx_array"]), generator=torch.Generator().manual_seed(self.seed))
        indices = order[:min(self.n_samples, len(order))]
        counts = (val["node_idx_array"][indices, 1] - val["node_idx_array"][indices, 0]).tolist()
        targets = val["properties"][indices].reshape(-1).tolist()  # raw eV, not normalized
        reference = set()
        for split in ("train", "val", "test"):
            data = torch.load(self.data_dir / f"{split}_data_processed.pt", map_location="cpu", weights_only=False)
            for smiles in data["smiles"]:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    reference.add(Chem.MolToSmiles(mol, isomericSmiles=False))
        was_training = model.training
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        field = model.vector_field
        attributes = ("properties_for_sampling", "property_name", "conditional_generation",
                      "normalization_file_path", "training_mode", "properties_handle_method",
                      "multilple_values_to_one_property", "dataset_name")
        before = {key: getattr(field, key) for key in attributes if hasattr(field, key)}
        rows, png_molecules, seen, raw_states = [], [], set(), []
        devices = [model.device.index] if model.device.type == "cuda" else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)
                with Chem.SDWriter(str(directory / "raw_generated.sdf")) as writer:
                    for start in range(0, len(indices), self.batch_size):
                        sampled = model.sample_random_sizes(
                            min(self.batch_size, len(indices) - start), device=model.device,
                            n_timesteps=self.n_timesteps, number_of_atoms=counts[start:start + self.batch_size],
                            multilple_values_to_one_property=targets[start:start + self.batch_size],
                            property_name="est", dataset_name=model.dataset_name, training_mode=False,
                            normalization_file_path=str(self.data_dir / "train_data_property_normalization.pt"),
                            guide_w={key: 1. for key in model.canonical_feat_order},
                            dfm_type="campbell", guidance_format="linear",
                        )
                        if len(sampled) != min(self.batch_size, len(indices) - start):
                            raise RuntimeError("sampler changed the requested sample count")
                        for offset, sample in enumerate(sampled):
                            index = start + offset
                            mol = sample.rdkit_mol
                            if hasattr(sample, "slot_graph"):
                                graph = sample.slot_graph
                                src, dst = graph.edges()
                                raw_states.append({
                                    "generation_index": index,
                                    "edge_index": torch.stack((src, dst)).cpu(),
                                    "nodes": {key: value.detach().cpu() for key, value in graph.ndata.items() if key.endswith("_1")},
                                    "edges": {key: value.detach().cpu() for key, value in graph.edata.items() if key.endswith("_1")},
                                })
                            record = strict_molecule_record(mol)
                            record.update(generation_index=index, target_est_eV=targets[index],
                                          requested_heavy_atoms=counts[index], validation_index=int(indices[index]))
                            record["heavy_atom_count_error"] = record["heavy_atoms"] - counts[index]
                            record["heavy_atom_retention"] = record["heavy_atoms"] / max(counts[index], 1)
                            smiles = record["smiles"]
                            record["novel_unique"] = bool(smiles and smiles not in reference and smiles not in seen)
                            if smiles:
                                seen.add(smiles)
                            if mol is not None and record["raw_valid"]:
                                mol.SetIntProp("generation_index", index)
                                mol.SetDoubleProp("target_est_eV", targets[index])
                                # Do not ask the writer to repair/Kekulize the raw graph.
                                writer.SetKekulize(False)
                                try:
                                    writer.write(mol)
                                except (ValueError, RuntimeError) as exc:
                                    record["serialization_error"] = str(exc)
                            if record["novel_unique"] and record["connected"]:
                                drawing = Chem.Mol(mol)
                                Chem.SanitizeMol(drawing)
                                rdDepictor.Compute2DCoords(drawing)
                                png_molecules.append((index, drawing))
                            rows.append(record)
            for start in range(0, len(png_molecules), 24):
                page = png_molecules[start:start + 24]
                Draw.MolsToGridImage([item[1] for item in page], molsPerRow=4,
                                    subImgSize=(280, 240), legends=[str(item[0]) for item in page]).save(
                                        directory / f"novel_rdkit2d_{start // 24:03d}.png")
            torch.save(raw_states, directory / "raw_states.pt")
        finally:
            model.train(was_training)
            for key in attributes:
                if key in before:
                    setattr(field, key, before[key])
                elif hasattr(field, key):
                    delattr(field, key)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        report = summarize_records(rows)
        report.update(epoch=epoch, optimizer_step=step, seed=self.seed,
                      time_grid_points=self.n_timesteps, guidance_scale=1.,
                      condition_fidelity_status="not_measured_requires_external_calculation",
                      geometry_definition="unoptimized covalent-radius/planarity screen; not PoseBusters or physical stability",
                      sampling_path="diagonal", data_manifest=model.data_manifest,
                      model_state_sha256=state_dict_sha256(model.state_dict()),
                      sdf_definition="unrepaired RDKit-valid outputs only; invalid endpoint tensors preserved in raw_states.pt",
                      raw_sdf_sha256=sha256_file(directory / "raw_generated.sdf"), records=rows)
        with (directory / "report.json").open("w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        return report

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        pl_module._stage7_generation_epoch = None
        try:
            report = self.evaluate(pl_module, trainer.current_epoch, trainer.global_step)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            # Do not destroy a training run or select stale quality metrics when
            # a rollout fails. Preserve the diagnostic and skip quality saving.
            self.output_dir.mkdir(parents=True, exist_ok=True)
            error_path = self.output_dir / f"evaluation_error_{trainer.current_epoch}_{uuid.uuid4().hex}.json"
            with error_path.open("x") as stream:
                json.dump(dict(epoch=trainer.current_epoch, step=trainer.global_step,
                               error=repr(exc)), stream, indent=2)
            warnings.warn(f"Generation validation failed; no quality checkpoint selected: {exc}")
            pl_module.log("gen_evaluation_failed", 1., on_step=False, on_epoch=True)
            return
        pl_module._stage7_generation_epoch = trainer.current_epoch
        pl_module.log("gen_evaluation_failed", 0., on_step=False, on_epoch=True)
        for key, value in report.items():
            if key.endswith("_rate"):
                pl_module.log(f"gen_{key}", value, on_step=False, on_epoch=True,
                              batch_size=report["requested"], sync_dist=False)


class GeneratedQualityCheckpoint(ModelCheckpoint):
    """Never save on a stale/missing/failed generation panel."""
    def on_validation_end(self, trainer, pl_module):
        if getattr(pl_module, "_stage7_generation_epoch", None) == trainer.current_epoch:
            super().on_validation_end(trainer, pl_module)
