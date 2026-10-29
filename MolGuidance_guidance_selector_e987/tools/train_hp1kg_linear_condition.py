#!/usr/bin/env python3
"""Freeze, verify, and freshly train HP1KG on S1 - 2*T1, in a detached session.

All molecules retain their own scalar label. Zero is a generation-evaluation
target, never a replacement label or a filter on the training corpus.
"""
import argparse
import copy
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("CONDITION_WORKSPACE_ROOT", PROJECT.parent)).resolve()
PROPERTY = "s1_minus_2t1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def audit_data(args, data):
    import torch
    source_hashes = {str(args.source/f"{s}.pt"): digest(args.source/f"{s}.pt")
                     for s in ("train", "val", "test")}
    rows, splits = [], {}
    for split in ("train", "val", "test"):
        original = torch.load(args.source/f"{split}.pt", map_location="cpu", weights_only=False)
        previous = torch.load(args.reference_data/f"{split}_data_processed.pt", map_location="cpu", weights_only=False)
        converted = torch.load(data/f"{split}_data_processed.pt", map_location="cpu", weights_only=False)
        assert converted["property_names"] == PROPERTY
        assert len(original) == len(converted["properties"]) == len(previous["properties"])
        assert converted.keys() == previous.keys()
        for key in previous:
            if key in {"properties", "property_names"}:
                continue
            if torch.is_tensor(previous[key]):
                assert torch.equal(previous[key], converted[key]), (split, key)
            else:
                assert previous[key] == converted[key], (split, key)
        exact = []
        for i, row in enumerate(original):
            assert row["smiles"] == converted["smiles"][i]
            value = Decimal(str(row["s1"])) - 2*Decimal(str(row["t1"]))
            exact.append(value)
            assert abs(float(value)-float(converted["properties"][i])) < 2e-6
            rows.append(dict(split=split, index=i, molecule_id=str(row["id"]), s1_eV=str(row["s1"]),
                             t1_eV=str(row["t1"]), s1_minus_2t1_eV=str(value)))
        splits[split] = dict(n=len(exact), near_zero_le_0p1=sum(abs(x)<=Decimal(".1") for x in exact),
                            exact_zero=sum(x==0 for x in exact), min=float(min(exact)), max=float(max(exact)))
    for name in ("joint_atom_state_metadata.json",):
        assert digest(data/name) == digest(args.reference_data/name), name
    norm = torch.load(data/"train_data_property_normalization.pt", weights_only=False)
    train = torch.load(data/"train_data_processed.pt", weights_only=False)["properties"]
    assert norm["property_names"] == PROPERTY and norm["std"] > 0
    torch.testing.assert_close(norm["mean"], train.mean(), rtol=0, atol=0)
    torch.testing.assert_close(norm["std"], train.std(), rtol=0, atol=0)
    for name in ("train_a", "train_b"):
        alias = torch.load(data/f"{name}_data_processed.pt", weights_only=False)
        torch.testing.assert_close(alias["properties"], train, rtol=0, atol=0)
    for path, expected in source_hashes.items():
        assert digest(path) == expected
    with (args.output/"condition_labels.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(property=PROPERTY, formula="S1 - 2*T1", units="eV", training_labels="per molecule; no zero relabeling or target-only filtering",
                  structural_tensors_and_split_order_unchanged=True, original_source_sha256=source_hashes,
                  splits=splits, normalization_train_only=True, mean=float(norm["mean"]), std=float(norm["std"]),
                  evaluation_target_eV=args.target, standardized_evaluation_target=float((args.target-norm["mean"])/norm["std"]))
    write(args.output/"condition_audit.json", result)
    return result


def verify(args, data, config, config_path):
    import torch
    import pytorch_lightning as pl
    from rdkit import Chem
    from molguidance.model_utils.load import data_module_from_config
    from molguidance.model_utils.load_with_guidance import model_from_config_with_guidance
    from molguidance.model_utils.provenance import data_manifest, verify_manifest, state_dict_sha256
    from molguidance.data_processing.utils import get_upper_edge_mask
    from molguidance.analysis.molecule_builder import SampledMolecule
    from molguidance.analysis.generation_callback import GenerationValidation
    from tools.train_stage8 import TrainingStatus

    output = args.output/"verification"
    output.mkdir()
    source_hashes = {str(p.relative_to(PROJECT)): digest(p) for p in (PROJECT/"molguidance").rglob("*.py")}
    data_before = data_manifest(data)
    pl.seed_everything(args.seed, workers=True)
    dm = data_module_from_config(config)
    dm.setup("fit")
    model = model_from_config_with_guidance(config)
    norm = torch.load(data/"train_data_property_normalization.pt", weights_only=False)
    codec = {}
    for split, dataset in (("train",dm.train_dataset),("val",dm.val_dataset),("test",dm.test_dataset)):
        stored = torch.load(data/f"{split}_data_processed.pt", weights_only=False)
        for i, expected in enumerate(stored["smiles"]):
            graph = dataset[i]
            normalized = (stored["properties"][i]-norm["mean"])/norm["std"]
            torch.testing.assert_close(graph.prop.reshape(-1), normalized.reshape(-1), rtol=1e-6, atol=1e-6)
            for key in ("x", "s"):
                graph.ndata[f"{key}_1"] = graph.ndata[f"{key}_1_true"]
            graph.edata["e_1"] = graph.edata["e_1_true"]
            graph.edata["ue_mask"] = get_upper_edge_mask(graph)
            molecule = SampledMolecule(graph, model.atom_type_map, ctmc_mol=False, atom_state_vocab=model.atom_state_vocab)
            Chem.SanitizeMol(molecule.rdkit_mol)
            assert Chem.MolToSmiles(molecule.rdkit_mol,isomericSmiles=False) == Chem.MolToSmiles(Chem.MolFromSmiles(expected),isomericSmiles=False)
        codec[split] = dict(total=len(dataset), identity=len(dataset), normalized_labels_checked=len(dataset))
        print("CODEC AND CONDITION PASS", split, len(dataset), flush=True)
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        sampler = loader.batch_sampler
        batches = list(sampler)
        assert sorted(i for batch in batches for i in batch) == list(range(len(loader.dataset)))
        assert all(int(sampler.costs[b].sum()) <= sampler.max_edges for b in batches)
    before = state_dict_sha256(model.state_dict())
    # Force correction on in the discarded smoke model to exercise that branch.
    smoke_config = copy.deepcopy(config)
    smoke_config["mol_fm"]["correction"].update(start_step=0, warmup_steps=0, probability=1.)
    model = model_from_config_with_guidance(smoke_config)
    before = state_dict_sha256(model.state_dict())
    trainer = pl.Trainer(accelerator="gpu",devices=1,precision="32-true",max_steps=3,max_epochs=1,
        limit_train_batches=3,limit_val_batches=1,num_sanity_val_steps=0,logger=False,
        enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
        gradient_clip_val=1.,gradient_clip_algorithm="value",default_root_dir=output,
        use_distributed_sampler=False,callbacks=[TrainingStatus(output)])
    trainer.fit(model,train_dataloaders=dm.train_dataloader(),val_dataloaders=dm.val_dataloader())
    assert trainer.global_step == 3 and state_dict_sha256(model.state_dict()) != before
    assert all(torch.isfinite(p).all() for p in model.parameters())
    checkpoint = output/"smoke_only.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = type(model).load_from_checkpoint(checkpoint,map_location="cpu",weights_only=False).cuda().eval()
    assert state_dict_sha256(restored.state_dict()) == state_dict_sha256(model.state_dict())
    panel = GenerationValidation(output/"target0_smoke",data,n_samples=2,batch_size=1,n_timesteps=12,
        property_name=PROPERTY,target_value_eV=args.target,joint_decoding=dict(time_limit=2.,max_rounds=4))
    report = panel.evaluate(restored,0,3)
    assert report["condition_name"] == PROPERTY and report["fixed_target_value_eV"] == args.target
    assert "fixed_target_est_eV" not in report
    assert all(r["target_s1_minus_2t1_eV"] == args.target and "target_est_eV" not in r for r in report["records"])
    verify_manifest(data_before,data_manifest(data))
    for path, expected in source_hashes.items():
        assert digest(PROJECT/path) == expected
    verification = dict(complete=True,data_unchanged=True,data_before=data_before,source_sha256=source_hashes,
        expanded_configs={config_path.name:config},codec=codec,
        configs=[dict(config=config_path.name,optimizer_steps=3,checkpoint_roundtrip=True,
            scalar_target_normalization=True,generation_target_and_metadata=True,
            sampling_finite=True,smoke_weights_discarded=True,
            smoke_overrides=smoke_config["mol_fm"]["correction"])])
    write(output/"verification.json",verification)
    del model,restored,trainer,dm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return output


def worker(args):
    import torch
    import yaml
    from molguidance.model_utils.load import read_config_file
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    data = args.output/"data_s1_minus_2t1"
    write(args.output/"status.json",dict(phase="converting",pid=os.getpid(),sid=os.getsid(0)))
    with (args.output/"conversion.log").open("x") as log:
        subprocess.run([sys.executable,str(PROJECT/"tools/convert_hp1kg_joint_kekule.py"),
            "--source",str(args.source),"--output",str(data),"--property",PROPERTY],check=True,stdout=log,stderr=subprocess.STDOUT)
    conversion=json.loads((data/"conversion_report.json").read_text())
    assert conversion["oracle_identity_total"] == conversion["oracle_identity_expected"] == 1581
    audit_data(args,data)
    config=read_config_file(args.output/"reference_est_config.yaml")
    config["dataset"]["processed_data_dir"]=str(data)
    config["dataset"]["raw_data_dir"]=str(args.source)
    config["dataset"]["conditioning"]["property"]=PROPERTY
    config["training"]["output_dir"]=str(args.output/"train")
    config["training"]["trainer_args"]["max_epochs"]=args.max_epochs
    generation=config["training"]["evaluation"]["generation"]
    generation.pop("target_est_eV",None)
    generation.update(property_name=PROPERTY,target_value_eV=args.target)
    config["wandb"].update(mode="disabled",name="hp1kg-s1-minus-2t1-stage8-cfg",save_dir=str(args.output/"train"))
    config_path=args.output/"s1_minus_2t1_cfg.yaml"
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    write(args.output/"status.json",dict(phase="verifying",pid=os.getpid(),condition=PROPERTY,target_eV=args.target))
    verification=verify(args,data,config,config_path)
    from tools.train_stage8 import train
    training=args.output/"train"
    training.mkdir()
    write(args.output/"status.json",dict(phase="training",pid=os.getpid(),sid=os.getsid(0),
        condition=PROPERTY,formula="S1 - 2*T1",target_eV=args.target,fresh_training=True,resume_checkpoint=None,
        live_status=str(training/"status.json"),max_epochs=args.max_epochs))
    train(argparse.Namespace(config=config_path,verification=verification,data=data,
        seed=args.seed,memory_fraction=args.memory_fraction,resume=None),training)
    write(args.output/"status.json",dict(phase="complete",pid=os.getpid(),condition=PROPERTY))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=ROOT/"joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz")
    parser.add_argument("--reference-data",type=Path,default=ROOT/"molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz")
    parser.add_argument("--reference-config",type=Path,default=ROOT/"runs/molguidance_stage8_cfg_resume_e476_20261021_v3/config.yaml")
    parser.add_argument("--target",type=float,default=0.)
    parser.add_argument("--max-epochs",type=int,default=10000)
    parser.add_argument("--memory-fraction",type=float,default=.7)
    parser.add_argument("--seed",type=int,default=17041)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    for name in ("output","source","reference_data","reference_config"):
        setattr(args,name,getattr(args,name).resolve())
    if not 0<args.memory_fraction<=.9 or args.max_epochs<1 or not __import__("math").isfinite(args.target):
        parser.error("invalid training bounds")
    if not args.worker:
        args.output.mkdir(parents=True,exist_ok=False)
        snapshot=args.output/"model_source_snapshot"
        snapshot.mkdir()
        for name in ("molguidance","configs","tests","tools","docs"):
            shutil.copytree(PROJECT/name,snapshot/name,ignore=shutil.ignore_patterns("__pycache__","*.orig"))
        shutil.copy2(PROJECT/"README.md",snapshot/"README.md")
        shutil.copy2(ROOT/"molguidance_hp1kg_baseline/convert_hp1kg_joint_kekule.py",snapshot/"tools/convert_hp1kg_joint_kekule.py")
        shutil.copy2(args.reference_config,args.output/"reference_est_config.yaml")
        env=dict(os.environ,CONDITION_WORKSPACE_ROOT=str(ROOT),PYTHONPATH=f"{snapshot}:{ROOT}",
            CUDA_VISIBLE_DEVICES="0",WANDB_MODE="disabled",OMP_NUM_THREADS="2",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",PYTHONUNBUFFERED="1")
        command=[sys.executable,"-u",str(snapshot/"tools"/Path(__file__).name),*sys.argv[1:],"--worker"]
        with (args.output/"worker.log").open("xb") as log:
            process=subprocess.Popen(command,env=env,cwd=ROOT,start_new_session=True,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,close_fds=True)
        write(args.output/"launch.json",dict(pid=process.pid,detached=True,command=command,condition=PROPERTY,
            formula="S1 - 2*T1",evaluation_target_eV=args.target,max_epochs=args.max_epochs,fresh_training=True,
            source_sha256={str(p.relative_to(snapshot)):digest(p) for p in snapshot.rglob("*.py")}))
        print(json.dumps(dict(pid=process.pid,output=str(args.output),detached=True,condition=PROPERTY)),flush=True)
    else:
        try:
            worker(args)
        except BaseException:
            write(args.output/"status.json",dict(phase="failed",pid=os.getpid(),traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
