# PolyRing Diffusion

Hierarchical diffusion prototype for generating **new polycyclic aromatic ring combinations** while preserving each ring as a chemically meaningful unit.

The model does not replace every ring by one featureless point. It keeps two coupled resolutions:

1. a ring graph, where nodes carry a template, center, normal, and physical size;
2. an atom graph, deterministically recovered from template atoms and indexed fusion ports.

This is the implementation of the design discussed for avoiding the failure mode where atom-wise noise destroys aromatic cycles before the model can learn poly-ring topology.

```text
ring templates + ring centers/normals
                 │
                 ▼
       mixed continuous/categorical diffusion
                 │
                 ▼
    connection type + fusion-port prediction
                 │
                 ▼
      hard chemical/topological projection
                 │
                 ▼
       deterministic atom-graph decoding
                 │
                 ▼
       optional radius-aware atom refinement
                 │
                 ▼
        optional RDKit sanitization/export
```

## What is implemented

- Variable-size dense ring graphs with padding masks.
- Ring templates for benzene, pyridine, pyrazine, pyrrole, furan, thiophene, and borinine.
- Continuous DDPM corruption of ring centers and normals.
- Absorbing-mask categorical diffusion for ring types and ring-connection types.
- An equivariant message-passing denoiser. Pair distances are normalized by template physical radii, so differently sized sources are distinguishable without treating radius as their identity.
- `SINGLE`, `FUSED`, and `SPIRO` ring connections with explicit vertex/edge ports.
- A constraint projector that first builds a valid spanning structure and can then add bounded closures. The second phase is what permits peri-condensed ring-adjacency cycles rather than only linear/cata assemblies.
- Deterministic ring-to-atom decoding with atom sharing for fused and spiro connections.
- A separate radius-aware atom refiner using atom-ring membership plus van der Waals and covalent radii.
- Optional RDKit conversion only after generation; RDKit is not inside every training step.
- A deterministic synthetic dataset and a small smoke configuration for testing the complete learning path.

## Important scope boundary

The included synthetic dataset is an **engineering smoke dataset**, not scientific training data. It verifies tensor flow, variable ring counts, fusion ports, decoding, and sampling. A meaningful molecular model still requires preprocessing real molecules into the same ring-template/port representation, train/validation splits, and chemistry/property evaluation.

This prototype preserves rings in the generative state but does not guarantee all of the following:

- valid global aromatic electron assignment for every decoded assembly;
- correct 3-D strain or steric feasibility;
- support for bridged systems sharing more than one edge between a pair of ring nodes;
- arbitrary macrocycles or rings absent from the template library;
- target-property conditioning such as `f_osc` (that should be added as a separate conditional embedding or reward stage).

## Installation

Python 3.10+ and PyTorch 2.2+ are required.

```bash
git clone <this-repository>
cd polyring-diffusion
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For RDKit SMILES export and developer tools:

```bash
python -m pip install -e '.[chem,dev]'
```

## Quick smoke run

```bash
python -m polyring_diffusion.train --config scripts/smoke.yaml
python -m polyring_diffusion.sample --config scripts/smoke.yaml
python -m unittest discover -s tests -v
```

The first command writes `runs/smoke.pt`. The second writes decoded ring and atom graphs to `samples/smoke.jsonl`.

For the larger example:

```bash
python -m polyring_diffusion.train --config config.example.yaml
python -m polyring_diffusion.sample --config config.example.yaml
```

If `device: cuda` is configured but CUDA is unavailable, training falls back to CPU.

## Representation

For each ring node `i`, the denoiser receives:

- noisy template token `r_i(t)`;
- noisy center `c_i(t)` and unit normal `n_i(t)`;
- template size, mean van der Waals radius, maximum van der Waals radius, and mean covalent radius;
- noisy pairwise connection token `e_ij(t)`.

It predicts center/normal noise, the clean ring token distribution, connection classes, and the port on each side of a connection. A fused edge consumes one edge port on each ring; a spiro or single link consumes one vertex port. Reusing the same port is rejected.

The key distinction is:

```text
physical radius = geometric feature used in distance normalization
ring template   = chemical identity carrying ordered atom types and ports
```

Radius alone cannot distinguish two chemically different rings of similar size. Conversely, a template ID alone does not tell the network the physical distance scale. The model therefore uses both.

## Source layout

```text
src/polyring_diffusion/
├── templates.py    ring atoms, charges, geometry, and ports
├── graph.py        hierarchical graph records and invariants
├── decoder.py      ring graph -> atom graph
├── projector.py    discrete validity/topology projection
├── diffusion.py    continuous and absorbing-mask noise processes
├── model.py        ring-level equivariant denoiser
├── atom_model.py   optional radius-aware atom refinement
├── data.py         deterministic synthetic training data
├── train.py        training CLI
├── sample.py       reverse process, projection, and JSONL output
└── rdkit_io.py     optional post-generation RDKit bridge
```

## Recommended next scientific steps

1. Build a real-data preprocessor that rejects or explicitly represents bridged and ambiguous SSSR decompositions.
2. Measure round-trip fidelity: molecule -> ring graph -> atom graph -> sanitized molecule.
3. Train the unconditional topology model and establish validity/novelty before adding property bias.
4. Add the desired property as a conditional embedding with classifier-free dropout, but keep guidance scale at 1 during the base-validity audit.
5. Only after base validity is established, compare conditional training, best-of-N, late-stage surrogate reward, and SMC/RL fine-tuning.

Keeping RDKit/xTB out of the differentiable inner loop is intentional. Tensor-level port, topology, radius-overlap, and valence-like penalties can train on GPU; exact RDKit and xTB checks belong in preprocessing, asynchronous evaluation, or a sparse late-stage reward pipeline.

## Real 100k rigid-ring database and conditional training

The repository also contains a reproducible path for the first 100,000 molecules
from `dataset_train.csv`. The existing atom-level Kekule splits are converted
without recomputing their ETKDGv3 coordinates:

```bash
PYTHONPATH=.deps python prepare_rigid_dataset.py \
  --source-dir /home/jovyan/cpu-only-1-datavol-1/psg/generation/STGG-AL/data/data_kekule \
  --output-dir data_ring_rigid --shard-size 2000
PYTHONPATH=.deps:src python verify_rigid_dataset.py \
  --root data_ring_rigid \
  --csv /home/jovyan/cpu-only-1-datavol-1/psg/generation/STGG-AL/data/dataset_train.csv
```

The database keeps the complete atom graph and adds canonical data-derived ring
templates, ring centers/normals/radii, atom-ring membership, connection classes,
and indexed ports. It is therefore lossless with respect to non-ring substituents.
The current dataset discovers 48 ring templates rather than using the seven smoke
templates.

Train the `f_osc`-conditioned ring model and sample it with:

```bash
PYTHONPATH=.deps:src python train_rigid.py --config scripts/real.yaml
PYTHONPATH=.deps:src python sample_rigid.py \
  --checkpoint runs/rigid_100k/best.pt --targets 0.1 0.8 3.0
```

`properties` uses the source preprocessing contract `log10(f_osc + 1e-6)` and is
standardized using train-split statistics. Generated JSONL records are projected
connected ring graphs. They are not an independent quantum-chemical validation
that the requested oscillator strength was achieved.
