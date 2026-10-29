#!/usr/bin/env python3
"""Present endpoint, graph, count-reference and existing-projection diagnostics."""
import base64
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"analysis/hp1kg_e4690_deep_diagnostics_20261028_verified"
PREV=ROOT/"analysis/hp1kg_e4690_s1_minus_2t1_n1000_report_20261027"
GEN=ROOT/"runs/generation_stage8_hp1kg_best_e4690_s1_minus_2t1_0_n1000_cfg1_20261027"


def probability_groups():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    stats={key:Counter() for key in ("real_atoms","fake_atoms","real_real_bonds","real_real_no_bond","any_fake_pair")}
    for file in sorted((GEN/"raw_endpoints").glob("*.pt")):
        for item in torch.load(file,map_location="cpu",weights_only=False):
            p=item["nodes"]["s_endpoint_probs"].numpy(); state=item["nodes"]["s_1"].argmax(-1).numpy()
            real=state<p.shape[-1]-1
            src,dst=item["edge_index"].numpy(); upper=src<dst; src,dst=src[upper],dst[upper]
            ep=item["edges"]["e_endpoint_probs"].numpy()[upper]; bond=item["edges"]["e_1"].argmax(-1).numpy()[upper]
            masks={"real_atoms":p[real],"fake_atoms":p[~real],"real_real_bonds":ep[real[src]&real[dst]&(bond>0)],
                   "real_real_no_bond":ep[real[src]&real[dst]&(bond==0)],"any_fake_pair":ep[~(real[src]&real[dst])]}
            for key,prob in masks.items():
                stats[key].update(n=len(prob),singleton=int(np.sum((prob>0).sum(-1)==1)),
                                  near_one=int(np.sum(prob.max(-1)>=.999999)) if len(prob) else 0,
                                  multi_options_above_1e_6=int(np.sum((prob>1e-6).sum(-1)>1)))
    return {key:{**dict(value),"singleton_fraction":value["singleton"]/max(1,value["n"]),
                     "near_one_fraction":value["near_one"]/max(1,value["n"])} for key,value in stats.items()}


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    s=json.loads((OUT/"summary.json").read_text()); old=json.loads((PREV/"summary.json").read_text())
    assert json.loads((OUT/"status.json").read_text())["phase"]=="complete"
    v,p,g,c=s["valence"],s["projection"],s["graph_groups"],s["conditional_reference"]
    w=json.loads((OUT/"weighted_composition.json").read_text())
    temp=json.loads((OUT/"runtime_temperature_check.json").read_text())
    atoms=list(csv.DictReader((OUT/"endpoint_atom_valence_diagnostics.csv").open()))
    rows=list(csv.DictReader((OUT/"endpoint_molecule_diagnostics_1000.csv").open()))
    near=list(csv.DictReader((PREV/"target_hits.csv").open()))
    probabilities=probability_groups()
    (OUT/"probability_support_strata.json").write_text(json.dumps(probabilities,indent=2)+"\n")
    carbon=[a for a in atoms if a["element"]=="C" and a["formal_charge"]=="0" and a["error_with_zero_H"]=="True"]
    degree_le4=sum(int(a["heavy_degree"])<=4 for a in carbon)
    source=GEN/"source_snapshot/molguidance/models"
    def source_line(path,needle):
        return next(i for i,line in enumerate(path.read_text().splitlines(),1) if needle in line)
    decoder=GEN/"source_snapshot/molguidance/analysis/joint_decode.py"
    (OUT/"code_trace.json").write_text(json.dumps(dict(temperature_default=dict(file=str(source/"ctmc_vector_field.py"),line=source_line(source/"ctmc_vector_field.py","cat_temperature_schedule: Union"),value=.05),
        probability_formula=dict(file=str(source/"classifier_free_guidance.py"),line=source_line(source/"classifier_free_guidance.py","def combine_guided_endpoint_probabilities("),formula="q_k proportional to class_mask_k * max(p_k,1e-9)^(1/T); CFG=1, T=.05"),
        endpoint_storage=dict(file=str(source/"classifier_free_guidance.py"),line=source_line(source/"classifier_free_guidance.py",'data_src[f"{feat}_endpoint_probs"] = saved')),
        decoder_support=dict(file=str(decoder),line=source_line(decoder,"for s, v in variants if nodes[i, s] > 0")),
        runtime_verification=str(OUT/"runtime_temperature_check.json")),indent=2)+"\n")
    lines=["HP1KG e4690 세부 진단 — 원자·수소·결합·고리·projection·조건 분포",
        datetime.now(timezone.utc).strftime("작성: %Y-%m-%d %H:%M UTC"),
        "원본: 1,000개, target Q=S1−2T1=0 eV, CFG=1, 100 timepoints. 기존 raw endpoint와 XTB 결과만 재분석했다.",
        "",
        "1. 세부 분석에서 새로 확인한 핵심",
        "(1) 수소 상태만으로 전체 실패를 설명할 수 없다. 무효 426개 중 312개(73.24%)는 수소가 없어도 heavy bond-order 합이 원자가 한도를 넘는다.",
        "(2) 연결 이웃 수 자체보다 bond order 합의 불일치가 두드러진다. 원자·H 상태와 Kekulé 결합차수의 공동 일관성이 핵심이다.",
        "(3) 5·6원 고리의 헤테로 조성이 감소하고, 고리 밖 원자가 부족하며, 탄소 중심 큰 고리계가 늘어난다.",
        "(4) 기존 joint projection은 극도로 좁아진 endpoint 확률과 강제 요청 count의 충돌 때문에 많은 경우 시작부터 가능한 해가 없다.",
        "(5) 계산법 차이를 고려해도 생성 Q 분포의 목표 집중은 약하다. 다만 Q 하나의 조건은 S1·T1 절대 에너지 범위를 지정하지 않는다.",
        "",
        "2. 원자가 실패 426개를 수소와 heavy-bond 문제로 분해",
        "각 endpoint의 원소·형식전하·H 수·Kekulé single/double/triple을 그대로 재구성했다. 1,000개 raw 유효/무효 판정이 기존 결과와 일치했다.",
        "원자별 B_i = sum_j bond_order(i,j), H_i = 예측 H 수로 놓고 원래 상태 및 H_i=0 상태를 RDKit valence 검사에 각각 넣었다.",
        "분자 분류는 원자별 첫 에러만이 아니라 모든 원자를 검사한 결과다.",
        f"수소를 전부 제거해도 overflow가 남는 분자: {v['invalid_with_any_heavy_only_overflow']}/426.",
        f"수소 감소로 모든 국소 overflow를 없앨 수 있는 분자: {v['invalid_all_overflows_H_removable']}/426.",
        f"전체 원자가 초과 원자 {v['valence_error_atoms']}개 중 heavy-bond만으로 초과하는 원자 {v['heavy_only_error_atoms']}개.",
        "주요 원자 패턴(분자 수가 아닌 원자 수):",
    ]
    lines += [f"  {key}: {n}개" for key,n in v["overvalence_atom_patterns"].items()]
    lines += [
        "C(H=0), heavy BO=5는 명백히 수소 추가 이전부터 결합차수 합이 초과한 경우다.",
        "C(H=1), heavy BO=4는 해당 H 상태와 결합 상태가 충돌한다. H가 하나 많을 수도, 결합차수가 하나 높을 수도 있어 원인 attribution은 단독으로 확정할 수 없다.",
        f"heavy-only overflow인 중성 C {len(carbon)}개 중 {degree_le4}개는 heavy 이웃이 4개 이하다.",
        "즉 '탄소 이웃을 4개 이하로 제한'하는 degree 규칙만으로 대부분의 과원자가를 막을 수 없다. 각 결합차수의 합을 함께 제약해야 한다.",
        "",
        "3. H만 재할당하는 진단용 반사실 실험",
        "원소·형식전하·모든 heavy edge와 bond order는 고정하고, train에서 관찰된 같은 원소/전하의 H 상태 중 해당 heavy valence를 만족하는 값을 대입했다.",
        f"무효 426개 중 완전한 H-only assignment가 존재한 것은 {v['invalid_H_only_train_assignment_exists']}개, RDKit 유효 {v['invalid_H_only_counterfactual_valid']}개, 연결된 중성·폐각 조건까지 만족한 것은 {v['invalid_H_only_counterfactual_eligible']}개다.",
        f"원래 RDKit 유효지만 radical이 있던 {v['valid_radical_molecules']}개 중 {v['valid_radicals_H_only_counterfactual_eligible']}개는 같은 진단에서 연결된 중성·폐각 상태가 되었다.",
        "이는 H/bond 일관성을 개선할 여지가 있음을 보여주는 상한성 진단이다. 실제 생성 성공 수나 XTB 성공 수에 더하지 않는다.",
        "이 H 재할당 결과로 새 XTB를 돌리지 않았고 원본 분자도 바꾸지 않았다. 올바른 화학 구조가 어느 쪽인지를 증명하는 실험은 아니다.",
        "312개 heavy-overflow 분자에는 H-only 조정이 충분하지 않다. 모든 H를 지우고 RDKit에 맡기는 방식만으로 근본 문제가 해결된다고 볼 수 없다.",
        "",
        "4. 고리 수뿐 아니라 고리 점유율과 조성이 달라진다",
        "항목 / 전체 train DB / 목표 주변 DB(kernel 재가중) / XTB 성공 생성",
    ]
    for key,label,genvalue in (("heavy_atoms","평균 heavy atom 수",old["groups"]["novel_unique_success"]["heavy_atoms"]["mean"]),
                              ("ring_count","평균 인식 고리 수",old["groups"]["novel_unique_success"]["ring_count"]["mean"]),
                              ("ring_atom_fraction","분자별 고리 소속 원자 비율 평균",g["xtb_success"]["ring_atom_fraction"]["mean"]),
                              ("nonring_atoms","평균 고리 밖 원자 수",g["xtb_success"]["nonring_atoms"]["mean"])):
        lines.append(f"  {label}: {c['metrics'][key]['unweighted']:.4f} / {c['metrics'][key]['kernel_weighted']:.4f} / {genvalue:.4f}")
    lines += [
        f"5·6원 고리 중 헤테로 원자를 포함한 고리 비율: train {100*g['train']['ring_5_6_hetero_fraction']:.2f}%, 조건부 train {100*w['kernel_ring_5_6_hetero_fraction']:.2f}%, 생성 {100*g['xtb_success']['ring_5_6_hetero_fraction']:.2f}%.",
        f"생성 7원 이상 고리 {g['xtb_success']['seven_plus_total']}개 중 탄소만 있는 고리가 {g['xtb_success']['seven_plus_allcarbon']}개다.",
        f"모든 heavy atom이 고리에 속하는 분자: train {g['train']['all_atoms_in_rings']}/1264, 생성 {g['xtb_success']['all_atoms_in_rings']}/452.",
        f"최대 연결 ring system 평균 atom 수: train {g['train']['largest_ring_system']['mean']:.2f}, 생성 {g['xtb_success']['largest_ring_system']['mean']:.2f}.",
        "원자 수뿐 아니라 원자들이 고리계 안에 배치되는 비율이 높아지고, 치환기·연결기 등 고리 밖 원자와 헤테로 고리 조성이 줄어든다.",
        "단, kernel 재가중 후 DB 평균 ring count는 4.81로 높아진다. 전체 DB 평균 3.42와의 차이 전부를 조건을 무시한 오류로 해석하면 안 된다.",
        "앞선 원자 수 재가중 분석과 이번 조건부 비교를 함께 보면 고리 점유율·고리 크기·조성의 차이가 더 일관된 문제다.",
        "",
        "5. 큰 고리가 RDKit cycle-basis 선택의 착시인지 확인",
        "각 edge (i,j)를 제거하고 i→j 최단 경로 길이를 구해 +1했다. 이는 해당 edge를 포함하는 가장 짧은 단순 cycle 길이다.",
        f"7원 이상이 필요한 edge를 하나 이상 가진 분자: train {g['train']['unavoidable_7plus_edge_molecules']}/1264, 생성 {g['xtb_success']['unavoidable_7plus_edge_molecules']}/452.",
        f"9원 이상: train {g['train']['unavoidable_9plus_edge_molecules']}/1264, 생성 {g['xtb_success']['unavoidable_9plus_edge_molecules']}/452.",
        f"실제 adjacency에서 triangle이 있는 분자 {g['xtb_success']['exact_triangle_molecules']}/452, 4-cycle이 있는 분자 {g['xtb_success']['exact_4cycle_molecules']}/452.",
        "이번 7+ 및 9+ 분자 수는 RDKit ring-basis 집계와 일치했다. 큰 고리 문제는 graph 자체에 남아 있다.",
        "이 최단 cycle 진단은 그래프 불변량을 추가한 것이며 고리 크기만으로 합성 가능성이나 안정성을 판정하지는 않는다.",
        "",
        "6. 기존 joint projection 93/1000의 실패 원인",
        f"저장된 결과: optimal {p['status_counts']['optimal']}, infeasible {p['status_counts']['infeasible']}. timeout이나 계산 예외로 집계된 것은 없다.",
        f"raw 무효→projection 유효 {p['raw_invalid_rescued']}개; raw 유효→projection 실패 {p['raw_valid_to_joint_invalid']}개.",
        f"원본과 동일한 그래프 {p['identical_raw_graph']}개. 성공분 중 atom state 편집 {p['successes_with_atom_edits']}개, bond 편집 {p['successes_with_bond_edits']}개(서로 중복 가능).",
        f"양의 확률을 갖는 선택지로 가능한 atom count 범위에 요청 count가 없는 경우 {p['count_bound_impossible']}개.",
        f"국소 valence의 최소/최대 가능 범위가 충돌하는 경우 {p['local_valence_bound_impossible']}개.",
        f"둘 중 하나로 infeasible을 증명할 수 있는 경우 {p['either_bound_impossible']}개. 이는 907개 infeasible의 {100*p['either_bound_impossible']/907:.2f}%다.",
        "예: #000003은 요청 32개인데 허용 support상 real atom 수가 29개로 고정되어 있다. #000005는 요청 32개, 가능한 수가 40개다.",
        "이 경우 최적화 시간을 늘리는 것으로 feasible 해를 만들 수 없다. support 또는 count 제약 정의가 달라져야 한다.",
        "",
        "7. 확률 선택지가 사라지는 수치 경로",
        "원본 체크포인트와 당시 source_snapshot을 CPU에서 로드해 categorical temperature T=0.05를 확인했다.",
        "CFG=1, corrected probability mode에서 q_k ∝ p_k^(1/T) = p_k^20이다.",
        "정확히는 log 직전 p를 1e−9로 clamp하고 class support mask를 적용한다. 허용 class의 q는 max(p_k,1e−9)^20에 비례한다. 이 floor가 있어도 float32 exp 결과의 0을 막지는 못한다.",
        "예를 들어 p=[0.999,0.001]이면 작은 상태 확률은 float64에서 1.02e−60, float32에서는 정확히 0이 된다.",
        "현재 solver는 probability>0인 class만 변수로 만든다. 따라서 수치적으로 0이 된 후보는 단지 불리한 후보가 아니라 금지된 후보가 된다.",
        "모든 pair가 NONE 위주라서 통계가 부풀려지는지 확인하기 위해 실제 결합과 비결합도 나눴다:",
    ]
    for key,val in probabilities.items():
        lines.append(f"  {key}: {val['n']}개; 양의 확률 class가 하나뿐인 비율 {100*val['singleton_fraction']:.3f}%; max p≥0.999999 {100*val['near_one_fraction']:.3f}%.")
    lines += [
        "의도적인 train-unsupported class mask와 float32 underflow로 생긴 0은 구분해야 한다.",
        "raw 생성이 실패한 426개 전부를 temperature 탓으로 확정할 수는 없다. 그러나 현재 최종 projection의 선택지를 제한하는 경로는 직접 확인됐다.",
        "이는 최종 확률을 다시 키우는 식으로 원래 정보를 복원할 수 있는 문제가 아니다. 이미 저장 값이 0이면 원래 작은 확률 크기를 알 수 없다.",
        "",
        "8. 조건부 DB 기준으로 다시 본 Q와 에너지",
        f"기존 count sampler의 동일 가중치를 재현했다: bandwidth=0.15 eV, nearest_k=128, ESS={c['ess']:.2f}, 상위 10개 reference weight 합 {100*c['top10_weight_mass']:.2f}%.",
        "이 가중치는 조건에 따른 원자 수를 정할 때의 reference이며 모델의 복사 확률이나 CFG=0 결과는 아니다.",
        "조건부 DB reference / 생성 성공 전체 / 생성 목표 근접 34개:",
        f"  S1 (XTB): {c['metrics']['s1_xtb']['kernel_weighted']:.3f} / {old['groups']['novel_unique_success']['energy']['s1']['mean']:.3f} / {old['groups']['target_hits']['energy']['s1']['mean']:.3f} eV",
        f"  T1 (XTB): {c['metrics']['t1_xtb']['kernel_weighted']:.3f} / {old['groups']['novel_unique_success']['energy']['t1']['mean']:.3f} / {old['groups']['target_hits']['energy']['t1']['mean']:.3f} eV",
        f"  Q (XTB): {c['metrics']['q_xtb']['kernel_weighted']:.3f} / {old['groups']['novel_unique_success']['energy']['q']['mean']:.3f} / {old['groups']['target_hits']['energy']['q']['mean']:.3f} eV",
        f"조건부 DB의 원래 NEVPT2 Q 평균은 {c['metrics']['q_reference']['kernel_weighted']:.3f} eV다.",
        f"같은 reference 가중치의 과거 DB-XTB |Q|≤0.1 비율은 {100*c['kernel_weighted_XTB_hit_rate']:.2f}%다. 생성 성공분은 7.52%다.",
        "생성 전체 Q 평균은 해당 reference보다 약 0.695 eV 더 음수 쪽이다. 단, 신규 graph·크기 생존 편향·초기 geometry 차이가 있어 모델 오차의 분리 추정치는 아니다.",
        "근접 후보 34개 중 S1<1 eV가 17개다. Q 하나만 조건으로 주었으므로 낮은 S1/T1를 허용하는 것이 현재 수학적 문제 정의다.",
        "S1가 특정 에너지 범위에 있어야 한다면 그 범위를 별도 평가·조건으로 명시해야 한다. Q만으로 개별 S1/T1 목표가 결정되지는 않는다.",
        "",
        "9. 목표 근접 34개를 구조까지 구분",
        "모두 기존 HP1KG DB와 중복 없는 구조이며 WBO 변화 경고는 0건이다. 아래 2D는 원래 graph, 에너지는 최적화 후 값이다.",
        "ID / S1 / T1 / Q / f(S1) / rings / 입력 topology·geometry / 최적화 후 aromatic-plane RMS(Å)",
    ]
    for r in near:
        lines.append(f"  #{int(r['generation_index']):06d} / {float(r['s1']):.3f} / {float(r['t1']):.3f} / {float(r['q']):+.3f} / {float(r['s1_f']):.4f} / {r['ring_sizes']} / {r['topology_screen']},{r['geometry_screen']} / {float(r['postopt_aromatic_plane_rms_A']):.3f}")
    lines += [
        "#000901: S1=2.278, T1=1.115, Q=+0.048 eV, f=0.4361. 5/6원 고리만 있으며 local atom/bond support와 geometry 검사를 통과했다.",
        "#000881: 1.453 / 0.715 / +0.023 eV, f=0.1669. #000948: 1.115 / 0.561 / −0.007 eV, f=0.0537. 같은 필터를 통과했다.",
        "#000052는 오차 0.006 eV로 작지만 f=0.0010이고 일부 local environment가 train 미관찰이다.",
        "#000643은 오차가 가장 작지만 4/7/9원 고리와 입력 geometry 경고가 있다. 오차 순위가 구조 품질 순위는 아니다.",
        "#000699는 기존 topology screen을 통과하지만 7원 고리가 여러 개다. 기존 screen은 3/4원과 9원 이상만 경고하고 7/8원은 허용한다.",
        "#000830에는 O−O 결합이 있다. 5/6원 고리·중성·closed-shell 조건만으로 모든 반응성 motif를 걸러내지는 않는다.",
        "후속 우선 후보는 #901/#881/#948이며 이는 현재 진단 기준의 순위다. 높은 계산 수준의 확인·안정성·합성 가능성을 확정한 것은 아니다.",
        "",
        "10. 수정 전 우선 확인할 사항",
        "P0. 최종 MAP decoder에는 temperature 적용 전의 log-probability 또는 logit과 의도적 class mask를 따로 전달·저장하는 방식을 검토한다.",
        "    가능 상태를 exp 후 0인지로 결정하지 않도록 하고 score 계산은 log-domain으로 유지한다. 원래 학습을 float64로 바꿔야 한다는 의미는 아니다.",
        "    같은 양의 temperature가 모든 항에 적용되면 정확한 산술에서 비용 전체의 공통 배율이므로 MAP 순위는 같다. underflow로 feasible set이 바뀌는 부분을 분리해야 한다.",
        "P0. stochastic PAD count 생성과 strict requested count projection의 목적을 맞춘다. 실제 count 유지와 요청 count 강제를 별도 옵션·평가로 구분한다.",
        "P1. 기존 checkpoint로 temperature와 count 처리만 대조하고, raw와 projection 지표를 분리해 재평가한다. 동일 seed/count 및 XTB 프로토콜을 유지한다.",
        "P1. 원자·H 상태와 bond order 합의 공동 일관성을 직접 평가한다. degree cap이나 H-only 변경만으로 해결될 문제는 아니다.",
        "P1. ring count 외에 ring-atom fraction, non-ring atom 수, 5/6원 hetero 조성, 7+ shortest-cycle edge를 checkpoint 평가에 추가하는 방식을 검토한다.",
        "위 항목은 개선 제안이다. 이번 요청에서는 학습 코드·sampler·checkpoint를 수정하지 않았고 재학습·재생성·XTB 재실행도 하지 않았다.",
        "",
        "11. 산출물과 검증 범위",
        "기존 2026-10-27 리포트는 보존했다. 이 리포트는 원자별 반사실 진단과 조건부 기준, 수치 support 진단을 추가한 것이다.",
        "첫 분석 실행은 마지막 로그 출력 시 NumPy int64 직렬화 예외가 발생했다. 숫자 형식을 수정하고 별도 verified 폴더에서 재실행하여 종료 코드 0을 확인했다.",
        "원본 파일과 실패 실행 폴더는 모두 보존했다. source hash와 1,000개 판정 일치 검증은 summary.json에 기록했다.",
        "상세 분자 원장: "+str(OUT/"endpoint_molecule_diagnostics_1000.csv"),
        "원자별 원자가 검사: "+str(OUT/"endpoint_atom_valence_diagnostics.csv"),
        "조건부 DB reference: "+str(OUT/"train_conditional_reference_1264.csv"),
        "고리별 조성: "+str(OUT/"ring_composition.csv"),
        "확률 support 상세: "+str(OUT/"probability_support_strata.json"),
        "실제 temperature 확인: "+str(OUT/"runtime_temperature_check.json"),
        "원본 코드 추적: "+str(OUT/"code_trace.json"),
        "수소 반사실·WBO·RDKit·graph cycle 지표는 진단이다. 물리적 안정성, 원래 분자의 정답 구조, 모델 개선의 인과효과를 확정하지 않는다.",
    ]
    report="\n".join(lines)+"\n"; (OUT/"detailed_report_ko.txt").write_text(report)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    with PdfPages(OUT/"deep_diagnostics_figures.pdf") as pdf:
        fig,axs=plt.subplots(2,2,figsize=(14,10),layout="constrained")
        ax=axs[0,0]; names=["Heavy-bond overflow\nremains at H=0","All local overflows\nremovable by reducing H"]; counts=[312,114]
        ax.bar(names,counts,color=["#b86552","#4b849e"])
        for i,n in enumerate(counts):ax.text(i,n+5,f"{n} ({100*n/426:.1f}%)",ha="center")
        ax.set_ylim(0,360); ax.set_ylabel("Invalid molecules"); ax.set_title("426 invalid graphs: local H vs heavy-bond diagnosis")
        ax=axs[0,1]; names=["Count bound","Valence bound","Either bound","All infeasible"]; counts=[p["count_bound_impossible"],p["local_valence_bound_impossible"],p["either_bound_impossible"],907]
        ax.bar(names,counts,color=["#4b849e","#4b849e","#d0a047","#9ba4b0"])
        for i,n in enumerate(counts):ax.text(i,n+10,str(n),ha="center")
        ax.set_ylim(0,1030); ax.set_title("Saved projection: infeasibility witnesses (overlap)"); ax.set_ylabel("Molecules / 1,000")
        ax=axs[1,0]; x=np.arange(3); width=.25
        columns=[("Train",[100*g['train']['ring_atom_fraction']['mean'],100*g['train']['ring_5_6_hetero_fraction'],100*old['groups']['train_db']['flags']['seven_plus']/1264]),
                 ("Target-weighted DB",[100*c['metrics']['ring_atom_fraction']['kernel_weighted'],100*w['kernel_ring_5_6_hetero_fraction'],100*w['kernel_seven_plus_ring_molecule_rate']]),
                 ("Generated success",[100*g['xtb_success']['ring_atom_fraction']['mean'],100*g['xtb_success']['ring_5_6_hetero_fraction'],100*232/452])]
        for offset,(label,values),color in zip([-.25,0,.25],columns,["#9ba4b0","#65967a","#4b849e"]): ax.bar(x+offset,values,width,label=label,color=color)
        ax.set_xticks(x,["Atoms in rings\n(mean per molecule)","5/6 rings\nwith heteroatom","Molecules with\n7+ rings"]); ax.set_ylabel("Percent"); ax.legend(fontsize=9); ax.set_title("Topology and ring composition")
        ax=axs[1,1]
        for valid,label,color in [(False,"Invalid","#b86552"),(True,"RDKit valid","#4b849e")]:
            part=[r for r in rows if (r['raw_valid']=="True")==valid]
            ax.scatter([int(r['requested']) for r in part],[int(r['actual']) for r in part],s=13,alpha=.35,label=label,color=color)
        ax.plot([14,68],[14,68],"k--",lw=1); ax.set_xlabel("Requested heavy atom count"); ax.set_ylabel("Actual endpoint count"); ax.legend(); ax.set_title("Count is stochastic; equality is not enforced")
        fig.suptitle("HP1KG e4690 | Endpoint diagnostics | S1 - 2*T1 = 0 eV",fontsize=16)
        fig.savefig(OUT/"deep_diagnostics_overview.png",dpi=180); pdf.savefig(fig); plt.close(fig)
    sys.path.insert(0,str(ROOT/"MolGuidance_guidance_selector_e987/tools"))
    from run_stage8_generation_xtb import render_spaced
    byid={int(r["generation_index"]):r for r in near}
    selected=[901,881,948,52,830,643,671,120,699,956]
    if not (OUT/"candidate_structure_review_10.png").exists():
        render_spaced([dict(generation_index=i,canonical_smiles=byid[i]["smiles"],s1_energy_eV=byid[i]["s1"],t1_energy_eV=byid[i]["t1"]) for i in selected],
                      OUT/"candidate_structure_review_10.png","HP1KG e4690 | Detailed structure review (not an energy ranking)","s1_minus_2t1",0.)
    figures=[]
    for name,title in [("deep_diagnostics_overview.png","수소·결합·고리·projection 세부 진단"),("candidate_structure_review_10.png","후보 구조 비교: 세부 해석은 본문 참고")]:
        encoded=base64.b64encode((OUT/name).read_bytes()).decode()
        figures.append(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="data:image/png;base64,{encoded}"></figure>')
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HP1KG e4690 세부 진단</title>
<style>body{font:16px/1.8 system-ui,sans-serif;max-width:1240px;margin:32px auto;padding:0 26px;color:#192b3b}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}figure{margin:30px 0;border:1px solid #ddd;padding:16px}img{max-width:100%}figcaption{font-size:21px;font-weight:bold} @media print{figure{break-inside:avoid}pre{font-size:10pt}}</style>
<h1>HP1KG e4690 — 분자 생성 세부 진단</h1>'''+"".join(figures)+"<pre>"+html.escape(report)+"</pre></html>"
    (OUT/"detailed_report_ko.html").write_text(document)
    shutil.copyfile(__file__,OUT/"presentation_script.py")
    (OUT/"report_manifest.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        report=str(OUT/"detailed_report_ko.txt"),html=str(OUT/"detailed_report_ko.html"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),lines=len(lines),
        summary_sha256=hashlib.sha256((OUT/"summary.json").read_bytes()).hexdigest()),indent=2)+"\n")
    print(json.dumps(dict(report=str(OUT/"detailed_report_ko.txt"),probability_groups=probabilities,carbon_heavy_overflow_atoms=len(carbon),degree_le_4=degree_le4),indent=2))


if __name__=="__main__":main()
