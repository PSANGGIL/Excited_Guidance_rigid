#!/usr/bin/env python3
"""Write a Korean report from the completed read-only e4690 audit."""
import base64
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"analysis/hp1kg_e4690_s1_minus_2t1_n1000_report_20261027"


def pct(k,n):
    return f"{k}/{n} ({100*k/n:.2f}%)" if n else "0/0"


def mean_std(s):
    return f"{s['mean']:.3f} ± {s['std_ddof0']:.3f}"


def main():
    s=json.loads((OUT/"summary.json").read_text()); g=s["groups"]
    success=g["novel_unique_success"]; hit=g["target_hits"]; db=g["train_db"]
    geo=success["optimized_geometry"]; match=s["count_matched_train"]; baseline=s["db_recalculation_comparison"]
    gen=Path(s["generation"]); xtb=Path(s["calculation"])
    manifest=json.loads((gen/"manifest.json").read_text())
    allrows=list(csv.DictReader((OUT/"all_1000_structure_energy.csv").open()))
    near=list(csv.DictReader((OUT/"target_hits.csv").open()))
    candidates=list(csv.DictReader((OUT/"target_hits_aromatic_only56_uncharged_wbo_clear.csv").open()))
    source_paths=[OUT/"summary.json",OUT/"all_1000_structure_energy.csv",OUT/"target_hits.csv",
                  OUT/"target_hits_aromatic_only56_uncharged_wbo_clear.csv",Path(__file__).resolve()]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={str(p):digest(p) for p in source_paths}
    lines=[
        "HP1KG S1−2T1 조건부 분자 생성·XTB/sTDA 분석 리포트",
        "분석일: "+datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "대상: best validation checkpoint e4690 / 목표 S1−2T1=0 eV / CFG=1 / 생성 1,000회",
        "",
        "1. 핵심 결과",
        "분자 후보는 확보되었지만, 생성 성공률과 목표 조건 집중도는 여전히 제한적이다.",
        "1,000회 중 RDKit 유효 574개, 연결된 중성·폐각 구조 452개이며 이 452개는 모두 XTB/sTDA 계산에 성공했다.",
        "기존 HP1KG train/val/test와 일치하거나 생성 결과끼리 중복된 계산 성공 구조는 없었다.",
        f"|S1−2T1|≤0.1 eV 후보는 {pct(hit['n'],1000)}이며, 계산 성공분 기준으로는 {pct(hit['n'],success['n'])}다.",
        "이 34개 모두에서 WBO 기준 원래 결합 약화·새 결합·수소 이동 경고는 없었다.",
        f"기존 topology·입력 geometry·최적화 수렴·WBO 검사를 함께 통과한 목표 근접 후보는 {g['target_hits_conservative_screen']['n']}개다.",
        "방향족을 포함하고 모든 고리가 5·6원이며 원자별 형식전하가 없고 WBO 경고가 없는 근접 후보는 10개다.",
        "후자의 10개는 이번 표본에서는 입력 geometry 및 최적화 후 방향족 평면성 검사도 통과했다.",
        "이는 후속 계산 후보의 선별 결과이며 합성 가능성이나 물리적 안정성을 확정한 수치는 아니다.",
        "",
        "2. 실험 조건과 데이터 범위",
        "조건 Q = S1−2×T1. 통상적인 EST = S1−T1과 구분한다. 이번 모델은 Q 한 개를 조건으로 학습했다.",
        "학습 라벨은 분자마다 다른 실제 Q이며 모두 0으로 바꾼 데이터가 아니다. 생성 시 목표값 0을 입력했다.",
        "HP1KG 1,581개: train 1,264 / validation 158 / test 159. 구조 중복 판정은 이 전체 DB를 대상으로 했다.",
        "체크포인트 e4690은 epoch 인덱스 0 기준이며 validation loss 최저 체크포인트다. 생성 성능으로 고른 best라는 뜻은 아니다.",
        "생성: CFG 1, timepoints 100, batch 16, seed 20261027. 원자 수는 train 조건부 kernel 분포에서 추출했다.",
        f"원자 수 sampler ESS={s['condition_atom_count_sampler']['diagnostics']['effective_sample_size_mean']:.2f}, 요청 평균 {s['count_control']['requested']['mean']:.3f}개.",
        "XTB: 24개 병렬, 각 1코어. 원본 diffusion heavy-atom XYZ를 유지하고 RDKit AddHs(addCoords=True)로 수소만 추가했다.",
        "XTB 입력 전 heavy-atom 좌표 최대 차이는 0.0 Å. ETKDG/MMFF/UFF 재임베딩은 사용하지 않았다.",
        "이후 xtb --opt loose --cycles 500 --chrg 0 --uhf 0으로 최적화했다. 최적화 단계에서는 좌표가 변할 수 있다.",
        "xtb4stda 이후 stda -xtb -e 10.0을 singlet/triplet 별도로 실행했다. 에너지 보정은 적용하지 않았다.",
        "통계는 raw 생성 그래프 기준이다. 병렬로 만든 joint/MAP projection 결과는 이번 XTB·성공률에 포함하지 않았다.",
        "",
        "3. 성공·실패 분석",
        "단계 / 개수 / 전체 1,000회 대비",
        "RDKit 유효 / 574 / 57.4%",
        "연결된 구조 / 565 / 56.5%",
        "연결된 중성·폐각 구조 및 XTB/sTDA 성공 / 452 / 45.2%",
        f"기존 topology+입력 geometry+수렴+WBO 검사 통과 / {g['conservative_screen']['n']} / {g['conservative_screen']['n']/10:.1f}%",
        "|Q|≤0.1 / 34 / 3.4%",
        "|Q|≤0.1 및 위 검사 통과 / 17 / 1.7%",
        "|Q|≤0.1, 방향족 포함, 5·6원 고리만, 원자별 전하 없음, WBO 경고 없음 / 10 / 1.0%",
        "",
        "계산 전 제외 사유는 상호배타적인 파이프라인 판정 순서 기준이다:",
        "invalid graph 426 / open-shell 98 / disconnected 9 / nonzero net charge 15 = 548개.",
        "유효 구조 전체의 radical·전하 집계는 중복 가능한 특성 통계이므로 이 제외 사유별 숫자와 다를 수 있다.",
        "426개 invalid graph의 첫 RDKit 예외:",
    ]
    lines += [f"  {key}: {value}" for key,value in s["invalid_error_groups"].items()]
    lines += [
        "탄소 valence 5/6 예외가 405/426개(95.07%)다. 큰 손실은 XTB 실패 이전의 생성 atom/H/bond 일관성에서 발생했다.",
        "예외에는 원소·valence가 기록되어 있으나 이것만으로 결합 수 과다와 H 상태 과다의 기여도를 분리할 수는 없다.",
        "",
        "4. 에너지와 목표 반영",
        "계산 성공 452개, 모든 ±는 표준편차(ddof=0), 단위 eV:",
    ]
    for key,label in (("s1","S1"),("t1","T1"),("q","Q=S1−2T1"),("est","EST=S1−T1")):
        e=success["energy"][key]
        lines.append(f"  {label}: {mean_std(e)}; 중앙값 {e['median']:.3f}; 범위 [{e['min']:.3f}, {e['max']:.3f}]")
    lines += [f"목표 Q=0에 대한 MAE={success['energy']['abs_q']['mean']:.3f} eV.",
              f"Q<0: {pct(success['energy']['negative'],452)}. Q=0.000인 표본: {success['energy']['exact_zero']}개.",
              "허용 오차 / 근접 개수 / 계산 성공분 대비 / 전체 생성 대비"]
    for threshold,n in success["energy"]["hits"].items():
        lines.append(f"  |Q|≤{threshold} eV / {n} / {100*n/452:.2f}% / {n/10:.2f}%")
    u=s["uncertainty"]
    lines += [
        f"|Q|≤0.1 성공분 비율의 Wilson 95% 구간: {100*u['wilson95_success'][0]:.2f}–{100*u['wilson95_success'][1]:.2f}%.",
        f"전체 생성 대비 구간: {100*u['wilson95_attempts'][0]:.2f}–{100*u['wilson95_attempts'][1]:.2f}%.",
        "위 구간은 한 seed 표본의 이항 비율 구간이다. seed·checkpoint를 바꿀 때의 변동성은 측정하지 않았다.",
        f"목표 근접 34개의 S1={mean_std(hit['energy']['s1'])}, T1={mean_std(hit['energy']['t1'])} eV.",
        f"근접 후보 중 S1<1 eV가 {hit['s1_below_1']}/34개이며 f(S1)≥0.01은 {hit['f_ge_0p01']}/34개다.",
        "Q 하나를 지정하면 S1과 T1 각각의 절대값은 정해지지 않는다. 낮은 S1·T1로 Q≈0을 만족하는 해도 허용된다.",
        "이번 생성만으로 guidance 자체의 기여도를 정량 확정할 수 없다. CFG 및 조건 대조군과 같은 원자 수·noise가 필요하다.",
        "",
        "5. 고리와 원소 조성",
        "항목 / train DB 1,264개 / 생성 계산 성공 452개 / 목표 근접 34개",
    ]
    for key,label in (("aromatic","방향족 포함"),("ringless","무고리"),("only56","고리가 모두 5·6원"),("small","3·4원 고리 포함"),
                      ("seven_plus","7원 이상 포함"),("nine_plus","9원 이상 포함"),("has_atomic_charge","형식전하 원자 포함")):
        lines.append(label+" / "+" / ".join(pct(x["flags"][key],x["n"]) for x in (db,success,hit)))
    lines += [
        "고리 종류별 분자 집계는 중복 가능하다. '고리가 모두 5·6원'은 고리 크기 조건이며 모든 고리가 방향족이라는 뜻이 아니다.",
        f"분자당 평균 고리 수: train {db['ring_count']['mean']:.3f}, 계산 성공 {success['ring_count']['mean']:.3f}, 목표 근접 {hit['ring_count']['mean']:.3f}.",
        f"인식된 개별 고리 중 5·6원 비율: train {100*db['fraction_5_6_rings']:.2f}%, 계산 성공 {100*success['fraction_5_6_rings']:.2f}%.",
        f"heavy atom 평균: train {db['heavy_atoms']['mean']:.3f}, 계산 성공 {success['heavy_atoms']['mean']:.3f}.",
        f"생성 원자 수로 train을 재가중해도 7원 이상 포함률은 {100*match['flags']['seven_plus']:.2f}%로, 생성의 {100*success['flags']['seven_plus']/452:.2f}%보다 작다.",
        f"재가중 train 평균 고리 수 {match['mean_rings']:.3f}, 생성 {success['ring_count']['mean']:.3f}. 크기 차이만으로 설명되지 않는 추가 고리 문제가 남아 있다.",
        f"재가중은 생성 {match['matched_generated']}/452개 원자 수를 커버하며 {match['unmatched_generated']}개는 해당 원자 수 train 표본이 없다.",
        f"고리 크기 JSD={match['ring_size_jsd_bits']:.4f} bit; 크기 재가중 비교={match['count_matched_ring_size_jsd_bits']:.4f} bit.",
        "JSD 범주는 3/4/5/6/7/8/9+이며 개별 고리 빈도로 계산했다. RDKit cycle basis에 따라 고리 집계는 달라질 수 있다.",
        "이 표본에서는 방향족 포함 구조 자체는 충분히 생성된다. 주요 구조 이슈는 고리 수 증가와 3·4원·7원 이상 고리의 과생성이다.",
        "3·4원 또는 큰 고리라는 이유만으로 화학적으로 불가능하다고 판정하지 않는다. 여기서는 DB 분포와의 차이를 지적한다.",
        "원소 포함 분자 수(train / 성공 / 근접):",
    ]
    for element in sorted(db["element_molecule_counts"]):
        lines.append(f"  {element}: {db['element_molecule_counts'].get(element,0)} / {success['element_molecule_counts'].get(element,0)} / {hit['element_molecule_counts'].get(element,0)}")
    lines += [
        f"성공분의 원자별 형식전하 포함은 {success['flags']['has_atomic_charge']}개이며 모두 총전하 0이다. nitro 등의 공명 표기를 포함하므로 모두 실제 zwitterion으로 분류하지 않는다.",
        f"train에서 관찰한 local atom/bond signature만으로 구성된 성공 구조: {success['local_support']['all_atoms_and_bonds_supported']}/452개.",
        f"train 최근접 Morgan radius2/2048 Tanimoto: 평균 {success['local_support']['nearest_train_tanimoto']['mean']:.3f}, 중앙값 {success['local_support']['nearest_train_tanimoto']['median']:.3f}.",
        "local signature가 train에 없는 것은 학습 지원이 약함을 의미하며 화학적 불가능성을 뜻하지 않는다.",
        "",
        "6. 분자 크기에 따른 성공률",
        "요청 heavy atom 범위 / 생성 / RDKit 유효 / XTB 성공 / 목표 근접 / 성공분 Q 평균 / WBO 경고",
    ]
    for r in s["requested_size_strata"]:
        if r["n"]:
            lo,hi=r["requested_atom_range"]
            lines.append(f"  {lo}–{hi}: {r['n']} / {pct(r['valid'],r['n'])} / {pct(r['calculated'],r['n'])} / {r['target_hits']} / {r['mean_q']:.3f} eV / {r['wbo_warnings']}")
    lines += [
        "大분자 구간의 XTB 성공분은 해당 크기에서 살아남은 일부 구조다. 이들의 에너지 오차가 작아도 전체 생성 성공률이 좋아진 것은 아니다.",
        f"요청 원자 수 평균 {s['count_control']['requested']['mean']:.3f}, 실제 endpoint 원자 수 평균 {s['count_control']['actual']['mean']:.3f}.",
        f"요청과 정확히 같은 원자 수는 {s['count_control']['exact']}/1000개. 평균 차이 {s['count_control']['actual_minus_requested']['mean']:+.3f}개.",
        "현재 요청 count는 실제 endpoint 원자 수의 강제 보장으로 해석하면 안 된다. padded slot의 실제 atom 상태도 함께 생성된다.",
        "",
        "7. Diffusion 좌표 및 XTB 최적화 감사",
        "452개 모두에서 optimization convergence 문구, singlet/triplet 설정, CSV 에너지와 원본 sTDA 로그 일치를 확인했다.",
        f"감사 예외 {s['verification']['audit_errors']}건. singlet/triplet 첫 10개 에너지 전체가 같은 로그 {s['verification']['identical_first10_singlet_triplet']}건.",
        f"정렬 후 heavy-atom RMSD: {mean_std(geo['heavy_rmsd_A'])} Å; 중앙값 {geo['heavy_rmsd_A']['median']:.3f} Å; >1 Å {geo['heavy_rmsd_above_1A']}/452개.",
        f"원래 heavy bond WBO<0.3: {geo['weak_old_heavy_bonds']}개 분자. 새 heavy contact WBO>0.5: {geo['new_heavy_contacts']}개 분자.",
        f"H 이동 경고: {geo['hydrogen_transfer']}개. 합쳐서 WBO 변화 경고 {geo['any_wbo_change_warning']}/452개(중복 제거).",
        f"최적화 후 원래 방향족 고리의 평면 RMS>0.15 Å: {geo['postopt_aromatic_plane_warning']}/452개; 이 역시 휴리스틱 경고다.",
        "목표 근접 34개에서는 WBO 변화 경고가 0건이다. 이번 near-zero 결과의 직접 원인을 최적화 중 결합 재편성으로 설명할 근거는 없다.",
        "WBO 경고가 없다는 것이 전자상태·결합차수·분자 identity의 엄밀한 보존 증명은 아니다. 진동수/Hessian 검사도 수행하지 않았다.",
        "SMILES 재파싱 표기 차이 1건(#551)은 저장된 원본 endpoint tensor로 재구성해 판정을 유지했다. 분석에서 누락하지 않았다.",
        "",
        "8. DB와 생성의 에너지 기준 차이",
        f"train에서 NEVPT2 |Q|≤0.1은 {g['train_db_q_abs_le_0p1']['n']}/1264개, |Q|≤0.5는 {g['train_db_q_abs_le_0p5']['n']}/1264개다.",
        "목표 0은 train 최소·최대 범위 안이지만 평균보다 약 2.14σ 높은 희소 영역이다. 단순히 데이터 범위 밖이라고 분류할 수는 없다.",
        f"train |Q|≤0.1 8개 중 방향족은 {g['train_db_q_abs_le_0p1']['flags']['aromatic']}개; |Q|≤0.5 34개 중 방향족은 {g['train_db_q_abs_le_0p5']['flags']['aromatic']}개다.",
        "과거 전체 DB XTB/sTDA 재계산 1,581개와 NEVPT2 Q를 다시 비교한 결과:",
        f"  전체 Q bias(XTB−DB)={baseline['full_db']['bias']:+.3f} eV, MAE={baseline['full_db']['mae']:.3f}, RMSE={baseline['full_db']['rmse']:.3f}, Pearson r={baseline['full_db']['pearson']:.3f}.",
        f"  DB 기준 |Q|≤0.1인 11개: DB 평균 {baseline['reference_near_zero']['reference']['mean']:+.3f} → XTB 평균 {baseline['reference_near_zero']['calculated']['mean']:+.3f} eV.",
        f"  이 11개 중 XTB에서도 |Q|≤0.1인 것은 {baseline['reference_near_zero']['xtb_hits_0p1']}개다. train 8개만 보면 XTB 평균 {baseline['reference_near_zero_train']['calculated']['mean']:+.3f} eV다.",
        "따라서 DB의 Q≈0을 학습한 모델을 XTB Q≈0 기준으로 평가할 때 계산 방법·구조 차이를 함께 고려해야 한다.",
        "전체 DB에서는 양의 bias, 목표 근처 subset에서는 음의 bias가 나타나므로 전체 평균을 빼는 단순 보정은 적절한 검증이 아니다.",
        f"생성 성공분의 Q 평균 {success['energy']['q']['mean']:+.3f} eV는 과거 DB-XTB 전체 평균 {baseline['full_db']['calculated']['mean']:+.3f} eV보다 목표 쪽으로 이동했다.",
        "그러나 conditional count sampling, 구조 선택 편향, geometry protocol 차이가 섞여 있다. 이 이동량 전체를 guidance의 순수 효과로 간주할 수 없다.",
        "과거 DB 재계산은 ETKDG/MMFF-or-UFF로 시작했고 이번 계산은 diffusion 좌표로 시작했다. 동일한 초기 좌표의 대조군은 아니다.",
        "DB 재계산 결과에 어떤 보정식을 적용하거나 생성 에너지를 재라벨링하지 않았다.",
        "",
        "9. 후속 검토 후보 10개",
        "선별 기준: |Q|≤0.1, 방향족 포함, 모든 인식 고리 5/6원, 원자별 형식전하 없음, WBO 경고 없음.",
        "ID / S1 / T1 / Q / f(S1) / 고리 크기 / train 미관찰 atom·bond signature 개수",
    ]
    for r in candidates:
        lines.append(f"  #{int(r['generation_index']):06d} / {float(r['s1']):.3f} / {float(r['t1']):.3f} / {float(r['q']):+.3f} / {float(r['s1_f']):.4f} / {r['ring_sizes']} / {r['unsupported_atoms']},{r['unsupported_bonds']}")
    lines += [
        "#000901, #000881, #000948은 위 기준을 만족하면서 f(S1)≥0.01 및 모든 local atom/bond signature train 관찰 조건도 만족한다.",
        "이는 이번 표본의 검토 우선순위이며 실제 광물성·안정성·합성 가능성이 검증된 순위는 아니다.",
        "#000830에는 O−O 결합이 있다. 5·6원 고리 필터만으로 모든 반응성 motif가 제거되는 것은 아니다.",
        "",
        "10. 해석과 다음 실험",
        "(a) 가장 큰 손실은 생성 그래프의 원자가 불일치다. 특히 큰 분자 구간에서 성공률이 빠르게 떨어진다.",
        "(b) 방향족 포함률은 높으나 고리 크기·고리 수의 DB 분포는 잘 재현되지 않았다. 원자 수 재가중 후에도 차이가 남는다.",
        "(c) Q=0 근처 후보는 얻었지만 전체 분포 집중도가 낮다. Q 하나의 조건은 S1·T1 절대값을 제약하지 않아 저에너지 후보가 다수 포함된다.",
        "(d) 이번 근접 후보는 WBO 변화 경고 없이 계산되었다. 다만 topology·geometry 검사와 후속 높은 수준 계산을 병행해야 한다.",
        "(e) 목표 부근 DB 자체도 XTB로 재계산하면 Q가 달라진다. 조건 성능과 계산 방법 차이를 분리한 평가가 필요하다.",
        "우선 비교 실험: 동일 checkpoint·seed·count 목록으로 CFG/조건 대조를 시행하고 크기 구간별 raw validity와 Q 적중률을 함께 비교한다.",
        "후속 계산: 선별 후보와 DB 목표 근접 8개(train)의 계산 조건을 맞춰 Q를 교차 확인한다. 평균 offset만으로 보정하지 않는다.",
        "모델 수정 전 진단: 426개 valence 예외에서 H 상태와 heavy-bond order의 과잉 기여를 직접 분해하고 요청 count 대비 실제 count 변화를 분석한다.",
        "이번 리포트 작업에서는 모델 수정, 추가 학습, 재생성, XTB 재실행을 하지 않았다.",
        "",
        "11. 산출물 및 재현 정보",
        "분석 결과는 원본 생성·계산 폴더와 별도로 저장했다. 입력 파일 hash는 summary.json의 source_sha256에 기록했다.",
        "분석 스크립트: "+str(ROOT/"analysis/report_hp1kg_e4690_s1_minus_2t1_20261027.py"),
        "리포트 생성 스크립트: "+str(Path(__file__).resolve()),
        "원본 생성: "+str(gen),
        "원본 XTB/sTDA: "+str(xtb),
        "리포트: "+str(OUT/"analysis_report_ko.txt"),
        "HTML 리포트(그림 포함): "+str(OUT/"analysis_report_ko.html"),
        "분석 그림 PDF: "+str(OUT/"analysis_figures.pdf"),
        "에너지 히스토그램(bin=0.25 eV): "+str(OUT/"energy_histograms.png"),
        "전체 1,000개 분석 CSV: "+str(OUT/"all_1000_structure_energy.csv"),
        "452개 최적화 감사 CSV: "+str(OUT/"geometry_audit_452.csv"),
        "목표 근접 34개: "+str(OUT/"target_hits.csv"),
        "선별 10개 PNG: "+str(OUT/"target_hits_aromatic_only56_uncharged_wbo_clear_00.png"),
        "",
        "12. 지표 해석 범위",
        "RDKit validity / XTB 수렴 / WBO warning-free / 목표 조건 근접은 각각 다른 지표이며 모두 물리적 안정성의 충분조건은 아니다.",
        "2D 그림은 원래 생성 그래프를 나타내고 에너지는 XTB 최적화 후 계산값이다. 모든 energy hit은 원본 로그의 출력 정밀도로 판정했다.",
        "새로운 고리계·형식전하·원소 환경이 train에 드문 것은 주의할 근거이지 자동적인 화학적 불가능 판정이 아니다.",
        "신규성은 이 HP1KG DB 내 canonical non-stereo graph 중복 여부이다. 전 세계 DB 신규성이나 stereoisomer 신규성은 검사하지 않았다.",
    ]
    text="\n".join(lines).replace("大분자","큰 분자")+"\n"
    (OUT/"analysis_report_ko.txt").write_text(text,encoding="utf-8")
    picture_names=[("analysis_overview.png","성공률·에너지·고리 구성·최적화 경고"),
                   ("energy_histograms.png","S1, T1, S1−2T1 개별 히스토그램"),
                   ("database_comparison.png","DB 및 분자 크기 재가중 비교"),
                   ("target_hits_aromatic_only56_uncharged_wbo_clear_00.png","선별 후보 10개 — 원본 생성 그래프")]
    pictures=[]
    for name,title in picture_names:
        encoded=base64.b64encode((OUT/name).read_bytes()).decode("ascii")
        pictures.append(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"></figure>')
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HP1KG e4690 생성·XTB 분석 리포트</title><style>
body{font-family:system-ui,sans-serif;max-width:1240px;margin:32px auto;padding:0 28px;color:#182532;background:#fff;line-height:1.8}
h1{line-height:1.4;font-size:28px}pre{font-family:inherit;white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;line-height:1.9}
figure{margin:32px 0;padding:20px;border:1px solid #dbe1e8;border-radius:10px}figcaption{font-size:19px;font-weight:650;margin-bottom:16px}img{max-width:100%;height:auto}
@media print{body{max-width:none;margin:0;padding:0}figure{break-inside:avoid}pre{font-size:10pt}h1{font-size:18pt}}
</style><h1>HP1KG e4690: S1−2T1=0 생성·XTB/sTDA 분석</h1>'''
    document+="".join(pictures)+"<pre>"+html.escape(text)+"</pre></html>"
    (OUT/"analysis_report_ko.html").write_text(document,encoding="utf-8")
    shutil.copyfile(__file__,OUT/"presentation_script.py")
    for path,old in hashes.items():
        assert digest(Path(path))==old
    (OUT/"report_manifest.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes,report_txt=str(OUT/"analysis_report_ko.txt"),report_html=str(OUT/"analysis_report_ko.html"),
        figures=picture_names,verified_candidates=len(candidates)),indent=2)+"\n")
    print(json.dumps(dict(report=str(OUT/"analysis_report_ko.txt"),html=str(OUT/"analysis_report_ko.html"),lines=len(lines)),ensure_ascii=False))


if __name__=="__main__":
    main()
