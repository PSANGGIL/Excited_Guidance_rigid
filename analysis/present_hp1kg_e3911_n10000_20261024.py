#!/usr/bin/env python3
"""Present completed, read-only structure/energy audit; no quantum calculation."""
import csv
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "analysis/hp1kg_e3911_n10000_structure_energy_20261023_v3"
OUT = AUDIT / "presentation"


def flag(frame, name):
    return frame[name].astype(str).str.lower().eq("true")


def main():
    RDLogger.DisableLog("rdApp.*")
    summary_path = AUDIT / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert json.loads((AUDIT / "status.json").read_text())["phase"] == "complete"
    g = summary["groups"]
    all_rows = pd.read_csv(AUDIT / "all_10000_structure_energy.csv")
    assert len(all_rows) == all_rows.generation_index.nunique() == 10000
    calc_dir = Path(summary["calculation"])
    ids = set(pd.read_csv(calc_dir / "rdkit_2d_novel_unique_spaced/molecules.csv").generation_index)
    unique = all_rows[all_rows.generation_index.isin(ids)].copy()
    success = all_rows[all_rows.calculation_status.eq("ok")].copy()
    near = unique[unique.est_exact.map(lambda x: abs(Decimal(str(x))) <= Decimal(".1"))].copy()
    assert len(success) == 5395 and len(unique) == 4991 and len(near) == 803
    OUT.mkdir(exist_ok=False)

    # All subsets retain the published first-success representative per graph.
    cross = []
    for label, subset in (("all_novel", unique), ("aromatic", unique[flag(unique, "aromatic")]),
                          ("nonaromatic", unique[~flag(unique, "aromatic")]),
                          ("ringless", unique[flag(unique, "ringless")]),
                          ("atomic_formal_charge", unique[flag(unique, "has_atomic_charge")])):
        hits = subset[subset.generation_index.isin(near.generation_index)]
        warnings = flag(hits, "wbo_change_warning")
        cross.append(dict(group=label, n=len(subset), hits=len(hits), hit_rate=len(hits)/len(subset),
                          hit_wbo_warning=int(warnings.sum()), hit_no_wbo_warning=int((~warnings).sum()),
                          warning_fraction_of_hits=float(warnings.mean())))
    pd.DataFrame(cross).to_csv(OUT / "target_hits_by_structure.csv", index=False)
    screened = pd.read_csv(AUDIT / "screened_aromatic_near_zero.csv")
    priority = screened[screened.s1_f >= .01].copy()
    priority["abs_gap_decimal"] = priority.est_exact.map(lambda x: abs(Decimal(str(x))))
    priority = priority.sort_values(["abs_gap_decimal", "generation_index"])
    assert len(screened) == 340 and len(priority) == 71
    priority.to_csv(OUT / "aromatic_5_6_near_zero_f_ge_0p01.csv", index=False)

    prior_gen = ROOT / "runs/generation_stage8_hp1kg_best_e3911_est0_n1000_cfg1_20261022"
    prior_path = prior_gen / "xtb_stda_diffusion_coords_12x1_20261022/summary.json"
    prior = json.loads(prior_path.read_text())
    previous_raw = json.loads((prior_gen / "raw_analysis/records.json").read_text())
    prior_mean_count = float(np.mean([r["requested_heavy_atoms"] for r in previous_raw]))
    comparison = dict(previous_n=1000, current_n=10000,
        previous_checkpoint_sha256=json.loads((prior_gen / "manifest.json").read_text())["checkpoint_sha256"],
        current_checkpoint_sha256=json.loads((Path(summary["generation"])/"manifest.json").read_text())["checkpoint_sha256"],
        previous_est_mae=prior["est_mae_eV"], current_est_mae=g["calculation_success"]["abs_est"]["mean"],
        previous_hit_rate_calculated=prior["hits"]["abs_error_le_0p1"]/prior["xtb_stda_successes"],
        current_hit_rate_calculated=g["calculation_success"]["inclusive_hits"]["0.1"]/len(success),
        previous_requested_atoms_mean=prior_mean_count,
        current_requested_atoms_mean=summary["count_control"]["requested"]["mean"],
        caveat="Same checkpoint, different seed, batch size and atom-count distribution; not a controlled model comparison")
    assert comparison["previous_checkpoint_sha256"] == comparison["current_checkpoint_sha256"]
    (OUT/"previous_1000_comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    ax = axes[0, 0]
    labels = ["Attempts", "RDKit valid", "Connected, neutral, closed shell", "XTB/sTDA completed", "Novel unique", "Novel unique |EST| <= 0.1"]
    counts = [10000, 8207, 5400, 5395, 4991, 803]
    ax.barh(labels[::-1], np.array(counts[::-1])/100, color="#4778a8")
    for j, n in enumerate(counts[::-1]):
        ax.text(n/100 + 1, j, f"{n:,}", va="center", fontsize=10)
    ax.set(xlim=(0, 115), xlabel="% of all 10,000 attempts", title="A. Success funnel")
    ax = axes[0, 1]
    ax.hist(unique.est, bins=np.arange(-2.3, 2.21, .1), weights=np.ones(len(unique))*100/len(unique),
            color="#4778a8", edgecolor="white")
    ax.axvspan(-.1, .1, color="#dc8150", alpha=.25, label="Target window +/- 0.1 eV")
    ax.axvline(0, color="#af4a2e", linewidth=1)
    ax.set(xlabel="EST = S1 - T1 (eV)", ylabel="% of novel unique successes", title="B. XTB/sTDA target distribution (0.1 eV bins)")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    labels = ["3/4 ring", "7+ ring", "9+ ring", "Ringless"]
    keys = ["small", "seven_plus", "nine_plus", "ringless"]
    pos = np.arange(len(keys))
    for delta, name, title, color in ((-.18,"train_db","Train DB","#638c65"),(.18,"novel_unique_success","Novel unique generated","#4778a8")):
        vals = [100*g[name]["flags"][k]/g[name]["n"] for k in keys]
        ax.bar(pos+delta, vals, .36, label=title, color=color)
    ax.set(xticks=pos, xticklabels=labels, ylabel="% of molecules (categories overlap)", title="C. Ring-size composition")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    subgroup = cross[1:4]
    x = np.arange(3)
    for delta, key, title, color in ((-.18,"hits","All target hits","#4778a8"),(.18,"hit_no_wbo_warning","Hits without WBO-change warning","#638c65")):
        vals = [100*c[key]/c["n"] for c in subgroup]
        ax.bar(x+delta, vals, .36, label=title, color=color)
    ax.set(xticks=x, xticklabels=["Aromatic", "Nonaromatic", "Ringless"], ylim=(0,100),
           ylabel="% of each novel unique subgroup", title="D. Target hits and optimization warnings")
    ax.legend(fontsize=9)
    fig.suptitle("HP1KG e3911 | EST target 0 eV | CFG 1 | 10,000 attempts", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT/f"structure_energy_overview.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, key, width, label in zip(axes, ("s1", "t1", "est"), (.5,.5,.1), ("S1", "T1", "EST = S1 - T1")):
        values = unique[key]
        bins = np.arange(np.floor(values.min()/width)*width, np.ceil(values.max()/width)*width+width*1.01, width)
        ax.hist(values, bins=bins, color="#4778a8", edgecolor="white")
        ax.set(xlabel=f"{label} (eV)", ylabel="Novel unique molecules", title=f"{label}: {values.mean():.3f} +/- {values.std(ddof=0):.3f} eV")
        if key == "est":
            ax.axvspan(-.1,.1,color="#dc8150",alpha=.3)
    fig.suptitle("XTB/sTDA energies | n = 4,991 novel unique successful graphs | std: ddof=0", fontsize=14)
    fig.savefig(OUT/"s1_t1_est_histograms.png",dpi=180)
    plt.close(fig)

    renderer_path = ROOT/"analysis/render_stage8_top10_conditions_20261022.py"
    spec = importlib.util.spec_from_file_location("spaced_render", renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    selected = []
    for row in priority.head(20).to_dict("records"):
        mol = Chem.MolFromSmiles(row["smiles"])
        selected.append(dict(index=int(row["generation_index"]), mol=mol,
            identity=Chem.MolToSmiles(mol,isomericSmiles=False), s1=Decimal(str(row["s1"])),
            t1=Decimal(str(row["t1"])), est=Decimal(str(row["est_exact"])), error=row["abs_gap_decimal"]))
    renderer.render(selected, OUT/"screened_aromatic_f_ge_0p01_top20.png", "e3911 diagnostic subset (f >= 0.01)", "est", Decimal("0"))
    priority.head(20).to_csv(OUT/"screened_aromatic_f_ge_0p01_top20.csv",index=False)

    report = [
        "HP1KG e3911 / EST=S1-T1=0 eV / CFG=1 / 10,000개 상세 분석",
        "분석일: 2026-10-24 UTC. 생성·XTB 재실행 및 모델/학습 코드 수정 없음.",
        "",
        "1. 전체 성공률",
        "전체 10,000 -> RDKit valid 8,207 -> connected 7,761 -> connected/net-neutral/closed-shell 5,400 -> XTB/sTDA 성공 5,395.",
        "계산 성공분에서 DB 일치 92건 및 반복 생성 312건을 제외하면 신규 고유 그래프 4,991개.",
        "신규의 범위는 사용한 HP1KG train/val/test 1,581개이며 세계 전체 화학 DB의 신규성을 뜻하지 않음.",
        "계산 대상 이전 제외: invalid 1,793 / disconnected 446 / open-shell 1,227 / nonzero net charge 1,134 (상호 배타적 순차 분류).",
        "invalid 1,793건 모두 RDKit explicit-valence 예외. C valence=5 1,290건, C=6 97건으로 77.36%가 C 과원자가 관련.",
        "계산 실패 5건: #4657/#5011/#5862는 10 eV 창에서 CSF 없음. #6146/#7447는 수소 추가 후 SDF identity guard 실패로 계산 전 제외.",
        "",
        "2. 에너지 및 목표 적중 (eV; 표준편차 ddof=0)",
    ]
    for key, label in (("calculation_success","전체 계산 성공"),("novel_unique_success","신규 고유"),("near_zero_novel_unique","신규 고유 |EST|<=0.1")):
        group=g[key]
        report.append(f"{label} n={group['n']}: "+" / ".join(f"{k.upper()}={group[k]['mean']:.6f} +/- {group[k]['std_ddof0']:.6f}" for k in ("s1","t1","est")))
        report.append(f"  MAE(EST target 0)={group['abs_est']['mean']:.6f}, median EST={group['est']['median']:.6f}, hits={group['inclusive_hits']}")
    report += [
        "전체 계산 5,395개 중 |EST|<=0.1는 1,046개(19.39%). 중복/DB 제외 후 803개(4,991개 중 16.09%, 전체 시도 중 8.03%).",
        "신규 고유 음수 EST는 154개(방향족 151개). 음수 전체와 |EST|<=0.1은 서로 다른 집합.",
        "DB 에너지는 NEVPT2, 생성 에너지는 보정하지 않은 XTB/sTDA. 두 에너지의 절대값/적중률을 직접 동일 척도로 해석하지 않음.",
        "이번 표본만으로 CFG의 인과적 효과를 측정할 수는 없음. 동일 noise/count의 무조건부 대조군이 필요함.",
        "",
        "3. 분자 내 고리 구성 (분자 비율; cycle-basis 정의를 동일하게 사용)",
        "항목 | Train DB 1,264 | 신규 고유 4,991 | 목표 근접 803",
    ]
    for key in ("aromatic","ringless","small","seven_plus","nine_plus","has_atomic_charge"):
        report.append(key+" | "+" | ".join(f"{g[n]['flags'][key]}/{g[n]['n']} ({100*g[n]['flags'][key]/g[n]['n']:.2f}%)" for n in ("train_db","novel_unique_success","near_zero_novel_unique")))
    report.append("전체 인식된 고리 중 5/6원 비율: "+" / ".join(f"{n}={100*g[n]['fraction_5_6_rings']:.2f}%" for n in ("train_db","novel_unique_success","near_zero_novel_unique")))
    report += [
        "평균 heavy atom 수는 train 26.38, 신규 고유 26.52인데 평균 ring 수는 3.42 -> 4.39. 평균 크기는 비슷해도 추가 closure와 비정상 크기 고리가 많음.",
        "3/4원, 7원 이상이 있는 것만으로 불가능한 분자라고 판정하지 않음. DB 빈도 대비 과생성이라는 결론.",
        "",
        "4. XTB 최적화 후 좌표·결합 점검",
        "전체 5,395개의 optimization convergence, 별도 singlet/triplet flags, CSV 에너지와 원본 로그 일치를 확인. 감사 예외 0건.",
        "수소 추가 전후 diffusion heavy XYZ 변화는 0 A. ETKDG 또는 MMFF/UFF를 사용하지 않았으며, 이후 XTB 자체가 좌표를 최적화함.",
        "회전/평행이동 정렬 후 heavy RMSD 평균 0.868 A, 중앙값 0.808 A, 1 A 초과 1,837/5,395. 유연한 비틀림도 RMSD에 기여하므로 RMSD만으로 구조 붕괴 판정 불가.",
        "기존 heavy bond WBO<0.3: 288개 / 신규 heavy contact WBO>0.5: 124개 / 수소 이동 경고: 387개. 합집합 735/5,395(13.62%).",
        "신규 고유 4,991개 중 경고 584개. 목표 근접 803개 중 경고 245개(30.51%), 그중 수소 이동 경고 202개.",
        "목표 근접 방향족 525개 중 53개 경고, 비방향족 278개 중 192개 경고. 무고리 근접 139개 중 109개 경고.",
        "경고 제외 시 목표 근접 558개가 남고 방향족 472개(84.59%), 무고리 30개(5.38%).",
        "WBO 임계값은 구조 변화 가능성을 찾는 경험적 진단이며 정체성/안정성의 확정 판정이 아님. 최적화 graph의 최종 결합차수 재판정과 Hessian 검증은 수행하지 않음.",
        "",
        "5. 목표 근접 집합의 특징과 진단적 후보",
        "학습 DB의 |EST|<=0.1 집합은 183개, 방향족 153개(83.61%), 무고리 8개(4.37%). 생성 hit 집합의 방향족 65.38%, 무고리 17.31%와 차이. 서로 다른 에너지 방법에 유의.",
        "Se 포함은 신규 고유 전체 171/4,991 -> 목표 근접 126/803. P 포함은 1,310/4,991 -> 303/803.",
        "형식전하 +/- 포함은 공명식도 포함하므로 모든 해당 분자를 zwitterion으로 부르지 않음.",
        "표시 정밀도상 EST=0인 20개 중 17개가 f(S1)<0.001, 11개가 Se 포함, 3개가 WBO 변화 경고. 0.000은 수치 반올림 범위의 일치.",
        "803개 -> WBO 경고 없는 558개 -> 방향족 포함/인식된 모든 고리 5·6원/원자 형식전하 없음 조건의 340개.",
        "그 340개 중 f(S1)>=0.01은 71개. 이 분할은 선택적 진단 기준이며 f가 작거나 이온성인 분자를 실패로 정의하지 않음.",
        "71개 중 P 포함 53개, Se 포함 5개, S 포함 7개. 상위 20개 중 P 포함 15개로 조성 편향이 남음.",
        "71개 중 train에서 관측한 local atom/bond signature만으로 구성된 것은 31개. 나머지가 화학적으로 불가능하다는 뜻은 아니며 추가 검토 대상임.",
        "71개 중 목표 오차가 작은 상위 20개를 추가 PNG로 저장. 원래 생성 그래프를 그렸고 최적화 후 에너지를 표시함.",
        "",
        "6. 크기·이전 표본과의 비교",
    ]
    for row in summary["requested_size_strata"]:
        report.append(f"requested atoms {row['requested_heavy_atom_range']}: n={row['n']}, valid={row['raw_valid']/row['n']:.2%}, calculation={row['calculated']/row['n']:.2%}, near-zero={row['target_hits']}/{row['calculated']} 계산성공; 중복 포함")
    report += [
        "요청 원자 수와 실제 real-node 수가 정확히 같은 경우는 1,257/10,000. 요청 mean=28.437/max=69, 실제 mean=28.855/max=77. hard exact-count 생성으로 해석하면 안 됨.",
        f"동일 e3911 이전 1,000개: MAE={comparison['previous_est_mae']:.6f}, hit rate={comparison['previous_hit_rate_calculated']:.2%}; 이번 10,000개: MAE={comparison['current_est_mae']:.6f}, hit rate={comparison['current_hit_rate_calculated']:.2%}.",
        f"요청 크기 평균 {prior_mean_count:.3f} -> {comparison['current_requested_atoms_mean']:.3f}, seed와 batch 차이. 장기학습의 개선 증거가 아니라 같은 checkpoint의 다른 표본 비교임.",
        "",
        "7. 해석과 우선 검토",
        "이 checkpoint는 방향족 신규 분자를 생성한다. 주된 병목은 생성 그래프의 valence/charge/radical 결합 일관성과 목표 EST 집중도, 큰 분자에서의 성공률 저하임.",
        "비방향족의 높은 근접률은 수소 이동 등 최적화 변화 경고와 크게 겹침. 원래 2D 구조에 최적화 뒤 에너지를 무비판적으로 연결하면 성공률을 과대평가할 수 있음.",
        "다음 비교에서는 동일 noise/count, DB 중복 제거, WBO/geometry 경고 분리 후 conditional 대조 평가가 필요. 추후 모델 수정은 이번 분석에서 시행하지 않음.",
        "",
        f"전체 분석: {AUDIT}", f"원본 생성: {summary['generation']}", f"계산: {summary['calculation']}",
    ]
    (OUT/"analysis_report_ko.txt").write_text("\n".join(report)+"\n")
    manifest = dict(source_summary=str(summary_path), source_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    representation="original generated nonstereo graph", energy="uncalibrated optimized XTB/sTDA",
                    warning_definition="old bond WBO<0.3, new contact WBO>0.5, or hydrogen transfer",
                    diagnostic_filter="aromatic; all rings 5/6; no atom formal charge; no WBO warning; abs(EST)<=0.1; S1 f>=0.01",
                    selected_top20_indices=[r["index"] for r in selected], files=[p.name for p in sorted(OUT.iterdir())])
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(output=str(OUT), top20_indices=manifest["selected_top20_indices"], completed=True)))


if __name__ == "__main__":
    main()
