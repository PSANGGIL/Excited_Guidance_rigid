"""Read-only source audit; write a new, non-overwriting Stage 6 history package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "MolGuidance_guidance_selector_e987"
TRAIN = ROOT / "runs/molguidance_hp1kg_joint_kekule_stage6_independent_time_bond_first"
RUN = TRAIN / "hp1kg-est-cfg-stage6-b640-expandable_a73c71od"
WANDB = BUNDLE / "wandb/offline-run-20261019_020337-a73c71od/run-a73c71od.wandb"
DB = ROOT / "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz"
PAIRED = ROOT / "runs/cfg_paired_e1074_est0_n1000_seed20261019"
COMPARE = ROOT / "runs/checkpoint_comparison_e1074_e1713_e1944_est0_cfg1_seed20261019"
LATEST = ROOT / "runs/generation_stage6_best_e1944_est0_n1000_cfg1_seed20261019"
XYZ = LATEST / "aromatic_negative_est_xyz"
DIAG = ROOT / "runs/aromatic_bias_diagnostics_20261019"
SOURCES: set[Path] = set()


def text(path):
    SOURCES.add(path)
    return path.read_text()


def data(path):
    return json.loads(text(path))


def table(path):
    SOURCES.add(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid reference SMILES: {smiles}")
    mol = Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def training(out):
    SOURCES.add(WANDB)
    store = DataStore()
    store.open_for_scan(str(WANDB))
    history = []
    while True:
        block = store.scan_data()
        if block is None:
            break
        record = pb.Record()
        record.ParseFromString(block)
        if record.WhichOneof("record_type") == "history":
            history.append({item.key or "/".join(item.nested_key): json.loads(item.value_json)
                            for item in record.history.item})
    steps = [row for row in history if "train_total_loss" in row]
    validation = [row for row in history if "val_total_loss_epoch" in row and "epoch" in row]
    assert len(history) == 8000 and len(steps) == 4000 and len(validation) == 2000
    by_epoch = defaultdict(list)
    for row in steps:
        by_epoch[int(row["epoch"])].append(row)
    val = {int(row["epoch"]): row for row in validation}
    assert sorted(val) == list(range(2000)) and sorted(by_epoch) == list(range(2000))
    train_keys = sorted(key for key in steps[0] if key.endswith("_train_loss") or key == "train_total_loss")
    epochs = []
    for epoch in range(2000):
        batch = by_epoch[epoch]
        assert len(batch) == 2
        record = {"epoch_zero_based": epoch, "train_logged_steps": len(batch),
                  "optimizer_steps_completed": max(row["trainer/global_step"] for row in batch) + 1}
        for key in train_keys:
            record[key + "_step_mean"] = float(np.mean([row[key] for row in batch]))
            record[key + "_step_last"] = batch[-1][key]
        record.update({key: value for key, value in val[epoch].items()
                       if key.startswith("val_") or key.endswith("_val_loss")})
        record["validation_logged_utc"] = datetime.fromtimestamp(val[epoch]["_timestamp"], timezone.utc).isoformat()
        epochs.append(record)
    save_csv(out / "training_steps.csv", steps)
    save_csv(out / "training_epochs.csv", epochs)
    save_csv(out / "validation_epochs.csv", validation)
    best = min(epochs, key=lambda row: row["val_total_loss_epoch"])
    last_path = RUN / "checkpoints/last.ckpt"
    SOURCES.add(last_path)
    ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
    callback = next(value for name, value in ckpt["callbacks"].items() if name.startswith("ModelCheckpoint"))
    best_scores = {path: float(score) for path, score in callback["best_k_models"].items()}
    assert abs(float(callback["best_model_score"]) - best["val_total_loss_epoch"]) < 1e-12
    config = yaml.safe_load(text(RUN / "config.yaml"))
    log = text(TRAIN / "launcher_b640_expandable_20261019.log")
    assert "max_epochs=2000" in log and "Epoch 1999:" in log
    groups = [{key: value for key, value in group.items() if key != "params"}
              for group in ckpt["optimizer_states"][0]["param_groups"]]
    summary = {
        "run": str(RUN), "config": config, "wandb_source": str(WANDB),
        "epochs_completed": 2000, "epoch_label_range": [0, 1999], "optimizer_steps_completed": 4000,
        "stopped_reason": "Trainer max_epochs=2000 reached (not early stopping)",
        "best_epoch": best["epoch_zero_based"], "best_validation_loss": best["val_total_loss_epoch"],
        "best_checkpoint": callback["best_model_path"], "retained_top3_scores": best_scores,
        "first_epoch": epochs[0], "last_epoch": epochs[-1],
        "first_training_metric_utc": datetime.fromtimestamp(steps[0]["_timestamp"], timezone.utc).isoformat(),
        "last_validation_metric_utc": epochs[-1]["validation_logged_utc"],
        "metric_span_hours": (validation[-1]["_timestamp"] - steps[0]["_timestamp"]) / 3600,
        "optimizer": "torch.optim.Adam", "serialized_optimizer_groups": groups,
        "last_ckpt_serialized_epoch": ckpt["epoch"], "last_ckpt_serialized_step": ckpt["global_step"],
        "last_ckpt_is_last_logged_epoch": int(ckpt["epoch"]) == 1999,
        "train_aggregation": "Unweighted arithmetic mean of the two logged step losses; not a sample-weighted Lightning epoch metric.",
        "validation_contract": "Fixed independent coordinate/graph time grid; training uses stochastic times/dropout, so absolute train and val losses are not identically distributed.",
        "selected_epochs": [epochs[i] for i in [0, 9, 49, 99, 499, 709, 999, 1074, 1499, 1713, 1866, 1944, 1983, 1999]],
        "windows": [{"epoch_first": lo, "epoch_last": hi - 1,
                     "mean_train_step_mean": float(np.mean([r["train_total_loss_step_mean"] for r in epochs[lo:hi]])),
                     "mean_val": float(np.mean([r["val_total_loss_epoch"] for r in epochs[lo:hi]])),
                     "min_val": min(r["val_total_loss_epoch"] for r in epochs[lo:hi])}
                    for lo, hi in [(0, 100), (1000, 1100), (1700, 1800), (1900, 2000), (1940, 1970), (1970, 2000)]],
    }
    save_json(out / "training_summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(range(2000), [r["train_total_loss_step_mean"] for r in epochs], lw=.7, label="Train: mean of logged steps")
    ax.semilogy(range(2000), [r["val_total_loss_epoch"] for r in epochs], lw=.8, label="Validation: fixed time grid")
    ax.scatter([best["epoch_zero_based"]], [best["val_total_loss_epoch"]], c="red", s=25, label="Best validation e1944")
    ax.set(xlabel="Epoch (zero based)", ylabel="Loss (log scale)", title="Stage 6: 2000 completed epochs; different train/validation time distributions")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out / "training_loss_history.png", dpi=170); plt.close(fig)
    return summary, epochs


def generations(out):
    specs = [("e624_prepared", ROOT / "runs/generation_stage6_e624_est0_n1000_cfg2_seed20261019", 624, 2),
             ("e709_cfg2", ROOT / "runs/generation_stage6_best_est0_n1000_cfg2_seed20261019", 709, 2),
             ("e1074_cfg2_initial", ROOT / "runs/generation_stage6_best_e1074_est0_n1000_cfg2_seed20261019", 1074, 2),
             *[(f"e1074_cfg{i}_paired", PAIRED / f"cfg{i}", 1074, i) for i in range(3)],
             ("e1713_cfg1", ROOT / "runs/generation_stage6_best_e1713_est0_n1000_cfg1_seed20261019", 1713, 1),
             ("e1944_cfg1", LATEST, 1944, 1)]
    inventory, details = [], []
    for label, path, epoch, cfg in specs:
        analysis_path = path / "analysis/summary.json"
        analysis = data(analysis_path) if analysis_path.is_file() else {}
        status_path = path / "status/supervisor.status"
        status = dict(line.split("=", 1) for line in text(status_path).splitlines() if "=" in line) if status_path.is_file() else {}
        result_path = path / "xtb_stda_diffusion_coords_24x1/work/results.csv"
        rows = table(result_path) if result_path.is_file() else []
        success = [r for r in rows if r["status"] == "ok"]
        gaps = np.array([float(r["s1_energy_eV"]) - float(r["t1_energy_eV"]) for r in success])
        arom = [any(atom.GetIsAromatic() for atom in Chem.MolFromSmiles(r["smiles"]).GetAtoms()) for r in success]
        xtb_summary_path = path / "xtb_stda_diffusion_coords_24x1/work/summary.json"
        xtb_summary = data(xtb_summary_path) if xtb_summary_path.is_file() else {}
        record = {
            "label": label, "run": str(path), "checkpoint_epoch": epoch, "cfg": cfg, "target_est_eV": 0,
            "generation_record_state": "analysis_available" if analysis else "no_completed_generation_evidence",
            "requested": analysis.get("requested"), "sdf_records": analysis.get("records"),
            "rdkit_valid": analysis.get("valid"), "connected": analysis.get("connected"),
            "novel_unique_connected": analysis.get("novel_unique_connected"),
            "aromatic_ring_valid": analysis.get("molecules_with_aromatic_ring"),
            "png_pages": len(analysis.get("pngs", [])), "generation_finished_utc": status.get("generation_finished_utc", status.get("finished_utc")),
            "xtb_attempted": xtb_summary.get("completed"), "xtb_stda_success": len(success),
            "est_mae_eV": float(np.abs(gaps).mean()) if len(gaps) else None,
            "abs_est_le_0p1_inclusive": int((np.abs(gaps) <= .100001).sum()),
            "aromatic_negative_est": int(sum(bool(a) and gap < 0 for a, gap in zip(arom, gaps))),
            "negative_est_all": int((gaps < 0).sum()),
        }
        if (path / "fixed_atom_counts.npy").is_file():
            p = path / "fixed_atom_counts.npy"; SOURCES.add(p); counts = np.load(p)
            record.update(fixed_count_mean=float(counts.mean()), fixed_count_min=int(counts.min()), fixed_count_max=int(counts.max()))
        inventory.append(record)
        details.append({**record, "status": status, "analysis": analysis, "xtb_summary": xtb_summary,
                        "generation_scripts": {str(p): text(p) for p in path.glob("*.sh")}})
    save_csv(out / "generation_inventory.csv", inventory)
    save_json(out / "generation_inventory.json", details)
    return inventory


def novelty(out):
    manifest = data(XYZ / "manifest.json")
    index, split_counts = defaultdict(list), {}
    for split in ("train", "val", "test"):
        path = DB / f"{split}_data_smiles.pkl"; SOURCES.add(path)
        with path.open("rb") as handle:
            smiles = pickle.load(handle)
        split_counts[split] = len(smiles)
        for i, smi in enumerate(smiles):
            index[identity(smi)].append({"split": split, "smiles_file_index": i})
    assert split_counts == {"train": 1264, "val": 158, "test": 159}
    results = []
    for r in manifest["molecules"]:
        key = identity(r["smiles"])
        optimized = Chem.MolFromMolFile(r["source_sdf"], sanitize=True, removeHs=True)
        assert optimized is not None and identity(Chem.MolToSmiles(optimized)) == key
        xyz = XYZ / r["xyz_file"]; SOURCES.add(xyz)
        assert digest(xyz) == r["xyz_sha256"]
        results.append({**r, "canonical_nonstereo_smiles": key, "db_matches": index.get(key, []),
                        "optimized_sdf_stored_graph_matches": True})
    assert len(results) == 5 and all(not r["db_matches"] for r in results)
    result = {"database": str(DB), "split_counts": split_counts, "total": 1581,
              "identity_method": "Sanitized RDKit canonical SMILES; ordinary explicit H normalized; stereochemistry ignored; no tautomer/protonation normalization",
              "novel_count_against_all_splits": 5, "external_databases_searched": False,
              "xyz_folder": str(XYZ), "molecules": results}
    save_json(out / "negative_est_novelty_check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/stage6_training_generation_record_20261020")
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    training_summary, epochs = training(out)
    inventory = generations(out)
    compare = table(COMPARE / "checkpoint_metrics.csv")
    for row in compare:
        row["val_total_loss_exact_from_training_history"] = epochs[int(row["epoch"])]["val_total_loss_epoch"]
    save_csv(out / "checkpoint_comparison.csv", compare)
    save_json(out / "checkpoint_comparison.json", data(COMPARE / "comparison.json"))
    guidance = data(PAIRED / "comparison.json")
    save_csv(out / "cfg_comparison.csv", [{"arm": name, **values} for name, values in guidance["metrics"].items()])
    save_json(out / "cfg_comparison.json", guidance)
    novel = novelty(out)
    diagnostics = data(DIAG / "consolidated_20261020/conclusions.json")
    save_json(out / "diagnostic_results.json", diagnostics)
    save_csv(out / "diagnostic_replications.csv", table(DIAG / "consolidated_20261020/replication_summary.csv"))
    save_json(out / "atom_count_provenance.json", data(DIAG / "consolidated_20261020/atom_count_provenance.json"))
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lines = ["Stage 6 현재 모델 학습 성능 및 생성 이력 저장", "저장일: 2026-10-20 UTC", f"브랜치: {branch}", f"기존 코드 커밋: {commit}",
             "", "1. 모델과 DB", "연속 heavy-atom Cartesian flow + masked categorical CTMC. 단일 EST 조건.",
             "원소/형식전하/붙은 H 수 joint state 37개 + fake state 1개. H는 별도 좌표 노드가 아님.",
             "Kekule NONE/SINGLE/DOUBLE/TRIPLE, self-conditioning, 독립 graph/coordinate time, bond-first geometry.",
             "HP1KG 공식 최적화 XYZ 기반 1581개: train1264, val158, test159. EST 평균0.5285704, std0.5375195.",
             f"DB: {DB}", "batch640, FP32, Adam lr0.0014, warmup50 epoch, weight_decay1e-12, grad value clip1.0.",
             "Condition dropout0.1, self-conditioning rate0.5, graph_clean_probability0.5, fake_atom_fraction0.3.",
             "좌표/노드/결합 loss weight3/1/2; auxiliary edge/valence/charge/connectivity/aromatic=0.5/0.05/0.05/0.25/0.2.",
             "", "2. 학습 완료와 저장된 체크포인트", f"학습 경로: {RUN}",
             "epoch0~1999, 총2000 epoch / optimizer4000 step. max_epochs 도달로 정상 종료; early stop 아님.",
             f"로그 구간: {training_summary['first_training_metric_utc']} ~ {training_summary['last_validation_metric_utc']}",
             f"최저 validation: e{training_summary['best_epoch']}, {training_summary['best_validation_loss']:.12f}",
             f"주의: last.ckpt 실제 내용은 e{training_summary['last_ckpt_serialized_epoch']}, step{training_summary['last_ckpt_serialized_step']}. 마지막 로그 epoch와 다름.",
             "원본 정확도 W&B 8000 history record에서 train4000 step, val2000 epoch를 추출함. 진행바 반올림 값이 아님.",
             "training_epochs.csv의 train은 두 step loss 산술평균. val은 고정 time grid라 절대값을 train과 직접 일반화 격차로 해석하지 않음.",
             "epoch\ttrain_step_mean\tval_total\tcondition_shuffle_gain"]
    for row in training_summary["selected_epochs"]:
        lines.append(f"{row['epoch_zero_based']}\t{row['train_total_loss_step_mean']:.9f}\t{row['val_total_loss_epoch']:.9f}\t{row['val_condition_shuffle_gain']:.9f}")
    lines += ["", "3. 생성 실행 이력", "각 완료 실행1000개. e624는 완료 결과가 없는 준비 폴더이며 성공 생성으로 세지 않음.",
              "초기 e1074 CFG2와 paired e1074 CFG2는 다른 실행/크기 벡터이므로 혼동하지 않음.",
              "label\tRDKit_valid\tnovel_unique_connected\tXTB_success\tabsEST<=0.1\taromatic_EST<0"]
    for row in inventory:
        lines.append("\t".join(str(row[k]) for k in ["label", "rdkit_valid", "novel_unique_connected", "xtb_stda_success", "abs_est_le_0p1_inclusive", "aromatic_negative_est"]))
    lines += ["", "4. 동일 입력 체크포인트 비교", "EST=0, CFG1, seed20261019, 100 time grid points(99 update), 각1000개.",
              "원자 수 벡터5~44/평균24.485는 앞선 생성 결과에서 유래했음. 현재 train DB p(n|EST=0)의 fresh sample이 아님.",
              "checkpoint\tval_exact\tvalid\tnovel_connected\tXTB_ok\tEST_MAE\tabsEST<=0.1\taromatic_hits\tclean_supported_aromatic_hits\tringless_hits"]
    for row in compare:
        lines.append("\t".join(str(row[k]) for k in ["checkpoint", "val_total_loss_exact_from_training_history", "rdkit_valid", "novel_unique_connected", "xtb_ok", "est_mae_eV", "est_abs_le_0p10", "hit_aromatic", "clean_supported_hit_aromatic", "hit_ringless"]))
    lines += ["e1713는 유효율/novel/clean-supported aromatic 후보 수가 우세. e1944는 val loss/전체 EST hit/고리분포 proxy가 우세.",
              "e1713->e1944 전체 nearzero hit91->132지만 aromatic57->56, ringless16->51. 방향족 조건 제어 향상으로 해석하지 않음.",
              "계산 성공률100%는 준비 필터 통과 후 계산한 분모 기준. 전체1000개 성공률이 아님.",
              "Topology-clean proxy: net charge0, radical0, 3/4 ring없음, >=9 ring없음. Ringless와7/8 ring은 허용하므로 물리적/합성 가능성 보장은 아님.",
              "XTB는 diffusion heavy coordinates+RDKit H 좌표에서 시작. ETKDG 재생성 없음. e1944 최적화 heavy RMSD 중앙0.78391A, >1A 144/509.",
              "", "5. e1074 CFG 비교", "CFG0=무조건, CFG1=기본조건부, CFG2=강화. 동일 count/seed, 각1000개.",
              "cfg\tvalid\tnovel_connected\tXTB_ok\tEST_MAE\tabsEST<=0.1\tclean_supported_hits"]
    for name, row in guidance["metrics"].items():
        lines.append("\t".join([name] + [str(row[k]) for k in ["rdkit_valid", "novel_unique_connected", "xtb_ok", "est_mae_eV", "est_abs_le_0p10", "clean_supported_abs_le_0p10"]]))
    lines += ["가이던스에 따른 EST 이동은 관찰되지만, 강한 CFG의 구조 악화 및 계산 생존 선택편향을 함께 고려해야 함.",
              "", "6. 방향족 포함 음수 EST 후보와 XYZ", "e1944 음수 EST5개는 모두 방향족 포함. 아래5개는 train/val/test 전체1581개에서 동일 구조0건.",
              "generation_index\txtb_job\tS1_eV\tT1_eV\tEST_eV\txyz_file"]
    for row in novel["molecules"]:
        lines.append("\t".join(str(row[k]) for k in ["generation_index", "xtb_job", "s1_eV", "t1_eV", "est_eV", "xyz_file"]))
    lines += [f"XYZ 폴더: {XYZ}", "수소 포함 XTB 최적화 좌표를 원자 순서 그대로 변환; 원본 SDF 대비 좌표차0. XYZ에는 결합 정보가 없음.",
              "DB 비교는 비입체 canonical SMILES, H/전하 유지, tautomer/protonation 별도통합 없음. 외부 DB 신규성은 검사하지 않음.",
              "음수는 저장된 sTDA 계산값이며 실제 inverted excited state 인증은 아님. -0.001/-0.002eV처럼 경계에 가까운 사례 포함.",
              "", "7. 완료된 추가 진단 및 핵심 제한", "전체 DB codec identity1581/1581. Denoiser15525조건/93150판정, partial reverse648, 생성 screening416+replication768.",
              "e1944 invalid224는 모두 AtomValenceException. H 제약만 푸는 진단으로53개(방향족48개) sanitize되나 생산 성공 수에 합산하지 않음.",
              "큰 고리 분자에서 오류 누적과 결합/원자/H 불일치가 보임. 방향족 자체 때문이라고 크기 효과와 분리해 확정하지는 못함.",
              "t=.5 test 방향족141: true조건 valid101/141, identity38/141; null조건105/141,43/141. 단일 실험 차이로 조건이 해롭다고 확정하지 않음.",
              "100->250 point 재검증: valid205->208/256, aromatic164->168. 중성 방향족112->136 신호는 있으나 다중검정 후 확정적이지 않음.",
              "DB |EST|<=0.1: 231개 중 방향족195(84.4%); 과거227은 경계 부동소수점 처리 차이. DB XTB 저gap 통과115 중86 방향족.",
              "생성 XTB 저gap132 중56 방향족(42.4%). 계산/선별만으로 편향 전체가 설명되지 않음.",
              "과거 DB XTB는 ETKDG/MMFF 출발, 현재 생성 XTB는 diffusion 출발. 계산 수준 보정값으로 혼동하지 않음.",
              f"상세 진단: {DIAG / 'consolidated_20261020/REPORT_KO.txt'}",
              "", "8. 다음 수정안은 아직 미적용", "EST 조건 유지. 평균 원자가 loss의 허점을 분포 기반 호환성 loss로 시험하고 기존 self-conditioning에 원자/H<->결합 교차 피드백을 검토.",
              "호환성 DP는 국소/factorized 근사이며 전역 Kekule cycle 보장이 아님. 노이즈 낮은 구간의 약한 항부터 별도 ablation 필요.",
              "조건 경로 변경은 동일 noise/heldout 반복에서 구조 악화가 재현될 때만 검토. EST를 무시해서 validity가 오르는 것은 성공이 아님.",
              "추가 학습만으로 aromatic lowgap 성능이 개선된다는 증거 없음. e1713/e1944를 목적별 기준으로 함께 보존.",
              "이 저장 작업은 모델/DB/ckpt를 수정하거나 학습/생성/새XTB를 시작하지 않음. 추가 git commit도 하지 않음.",
              f"정합성 미해결 항목: {BUNDLE / 'docs/STAGE6_LOGIC_MATH_AUDIT_20261019.md'}",
              "", "9. 파일", "training_steps.csv: 4000 optimizer-step loss", "training_epochs.csv / validation_epochs.csv: 2000 epoch와 구성항",
              "training_summary.json: 실제 config, best 및 last 체크포인트 구분", "training_loss_history.png: 전체 loss 곡선",
              "generation_inventory.csv/json: 8개 실행 폴더 및 실제 산출물", "checkpoint_comparison.csv/json, cfg_comparison.csv/json: 통제 비교",
              "negative_est_novelty_check.json:5개 DB 비교와 XYZ provenance", "diagnostic_results.json / diagnostic_replications.csv / atom_count_provenance.json: 추가 검증",
              "source_manifest.json: 원본 파일 경로/해시. 기존 결과는 보존함."]
    with (out / "REPORT_KO.txt").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    SOURCES.add(Path(__file__).resolve())
    save_json(out / "source_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "branch": branch, "model_code_commit": commit,
              "sources": [{"path": str(path), "size": path.stat().st_size, "sha256": digest(path)} for path in sorted(SOURCES)]})
    print(json.dumps({"out": str(out), "epochs": len(epochs), "history_steps": 4000,
                      "generation_runs": len(inventory), "best_epoch": training_summary["best_epoch"],
                      "last_ckpt_epoch": training_summary["last_ckpt_serialized_epoch"], "novel_candidates": 5}, indent=2))


if __name__ == "__main__":
    main()
