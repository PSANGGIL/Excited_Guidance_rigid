# HP1KG Stage 7: 데이터 추가 없는 단계별 개선

작성일: 2026-10-20 UTC.
브랜치: `feature/molguidance-stage7-fixed-data-consistency`.
출발 커밋: `86ee936ff60fe18c09e01ec5a3513d6b59ea2145`.
이 기록은 구현과 검증 결과이며, 장기학습 결과나 생성 성능 향상 보고서가 아니다.

## 1. 보존한 것과 이번 범위

- HP1KG 총 1,581개: train 1,264 / validation 158 / test 159.
- 공식 최적화 heavy-atom XYZ, 원자 인덱스, S1/T1 및 EST=S1−T1, 기존 split 유지.
- 외부 DB, 다른 계산 수준의 EST label, pseudo-label, 외부 사전학습 추가 없음.
- 기존 checkpoint, 생성물, XTB/sTDA 결과 및 과거 감사 문서 수정 없음.
- 모델은 연속 좌표 flow + categorical masked CTMC이다. DDPM의 Gaussian
  noise-prediction 모델로 교체한 것이 아니다.
- 원소·형식전하·붙은 H 수의 joint state 37개와 fake state 1개를 유지한다.
  수소가 독립적인 좌표 노드로 바뀐 것은 아니다.
- 결합은 `NONE/SINGLE/DOUBLE/TRIPLE`의 Kekulé 4-state이며, 방향족은
  RDKit sanitization으로 판단한다. aromatic class 강제나 loss 증량은 하지 않았다.
- 이번 구현 대상은 현재 EST CFG 모델이다. MG의 categorical target 문제까지
  해결한 것으로 간주하면 안 된다.

DB 절대경로:

```text
/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz
```

## 2. 단계별 설정과 대조군

`configs/local/stage7/`의 `_base_`는 상대 YAML을 재귀적으로 병합한다.
순환 참조는 거부하고, 학습기는 병합된 설정을 저장한다.
아래는 구현·검증 순서이지, 모든 설정의 장기학습이 끝났다는 뜻이 아니다.

| 설정 | 이전 단계 대비 변경 |
|---|---|
| `00_stage6_control.yaml` | 기존 Stage 6 구조/목적함수 유지, 새 manifest와 평가만 연결 |
| `01_contracts.yaml` | masked sample-wise loss, O(3) 대칭 계약, 정규화·manifest 강제 |
| `01a_legacy_loss_control.yaml` | 01과 같은 구조에서 기존 factorized loss만 사용 |
| `01b_unweighted_loss_control.yaml` | 01과 같은 구조에서 categorical time weighting만 제거 |
| `02_feedback.yaml` | 이전 endpoint의 atom/H/charge ↔ bond chemistry feedback |
| `03_distribution.yaml` | 평균 valence/charge 항 대신 분포 호환성 항 |
| `04_correction.yaml` | 별도 off-path 복원 objective 추가 |
| `05_dropout.yaml` | GVP dropout 0 → 0.05 |
| `06_weight_decay.yaml` | weight decay 1e−12 → 1e−5 |
| `07_small_model.yaml` | 06에서 scalar/edge 256/128 → 128/64, update 8 → 4 |
| `08_condition_scale_control.yaml` | **04에서 분기**하여 조건 residual scale 1 → 0.5 |

00은 비교 대조군으로만 기존 `n_cp_feats=4`를 남긴다. 권장 대칭 설정은 01 이후의
`n_cp_feats=0`이다. 01a는 기존 Stage 6 전체의 복제본이 아니라 **loss만의 대조군**이다.
06/07/08의 개선은 아직 입증되지 않았다. 같은 split, seed, batch, optimizer-step 예산,
생성 count/조건 벡터로 비교해야 한다. loss 정의가 바뀌므로 서로 다른 행의 total loss
절대값을 성능 순위로 비교하면 안 된다.

## 3. 목적함수와 구현 계약

### 3.1 Categorical reduction

항목 i의 CE를 `l_i`, 명시적인 학습 mask를 `m_i`, 해당 항목의 time weight를
`w_i`라고 할 때 다음을 구분했다.

```text
legacy_factorized = mean(m_i l_i) · mean(w_i)
aligned_all       = sum(m_i w_i l_i) / N
masked_weighted   = sum(m_i w_i l_i) / max(sum(m_i), 1)   # Stage 7 기본
masked_unweighted = sum(m_i l_i) / max(sum(m_i), 1)
```

`CE==0`으로 mask를 추측하지 않는다. 분모는 weight 합이 아니므로 공통 time scale이
상쇄되지 않는다. graph가 전부 clean인 경우에도 NaN 대신 미분 가능한 0을 반환한다.
이는 명시적으로 선택한 weighted denoising objective이며, 자동으로 정확한 CTMC
likelihood/ELBO가 되는 것은 아니다. mask 수 정규화는 all-entry 평균과 다른 estimator다.
구현: `models/loss_utils.py`, `FlowMol.reduce_feature_loss`.
기존 checkpoint의 기본값은 `legacy_factorized`로 보존한다.

### 3.2 시간·대칭·조건 단위

좌표/graph 독립 time과 graph-clean branch는 유지하되 생성은 `t_x=t_graph`의
대각 경로임을 명시한다. graph-clean branch는 보조 denoising 정규화이지
구현되지 않은 graph-first 2-D reverse solver의 증거가 아니다.

비입체화학 표현에 맞춰 01 이후는 cross-product feature를 끄고 O(3) 계약을 택했다.
반사를 허용하는 Kabsch와 다층 GVP의 반사 등변성을 함께 검사한다.
`rigid_alignment`의 중심 복원식도 `centered_source @ R.T + target_center`로 바로잡았다.
SO(3) 전용 정렬은 `allow_reflection=False`로 가능하지만, chiral 모델 전체가 구현된
것은 아니다.

EST는 외부에서 raw eV, property embedder에는 train mean/std로 정규화된 값이 들어간다.
원자 수 조건 sampler에는 raw eV를 전달한다. Stage 7 checkpoint는 normalization을
자동 해석하고 다른 normalization 파일은 hash 검증으로 거부한다.
`condition_residual_scale=0.5`는 학습/샘플링 공통 residual 배율이지
“조건 기여율 50%”가 아니다. CFG scale과도 별개다.

### 3.3 원자/H/전하 ↔ 결합 feedback

`models/chemistry_feedback.py`는 detached 이전 endpoint에서 다음을 계산한다.

- incident bond-order 평균/분산/degree;
- train-supported heavy valence의 최소·최대, H, formal charge, fake 확률;
- bond valence와 atom state 사이의 차이 및 graph의 예상 net charge.

이를 node residual과 양 끝점을 대칭 처리한 edge residual로 전달한다.
추론 중 clean label, RDKit repair, EST 정답을 읽지 않는다.
마지막 linear를 0으로 초기화하여 초기 출력은 기존 self-conditioning과 동일하다.
같은 seed에서 공통 backbone 가중치가 달라지지 않게 선택 모듈 초기화 RNG도 격리한다.
1-step 뒤 residual이 실제 학습되는지 검사한다.

이는 MolDiff의 atom–bond inconsistency 문제와 JODO의 node/edge 상호작용 방향을
현재 CTMC/self-conditioning 구조에 적용한 **자체 구현**이다. 해당 논문의 모델을
그대로 복제하거나 동일 성능을 보장하는 것은 아니다.
[MolDiff](https://proceedings.mlr.press/v202/peng23b.html),
[JODO](https://arxiv.org/abs/2305.12347).

### 3.4 분포 기반 valence·net-charge 호환성

각 edge의 order 확률을 `p_ij(b)`, atom-state 확률을 `p_i(s)`,
state s에서 train이 관찰한 heavy valence 집합을 `V(s)`라고 한다.

```text
P_i(valid) = sum_s p_i(s) sum_{v in V(s)} P(sum_j B_ij = v)
P_Q(valid) = P(sum_i formal_charge(S_i) = Q_target)
L(P)      = -log((P + eps) / (1 + eps))
```

CTMC에서 이미 공개된 항목은 실제 관측 one-hot을 사용하고, masked 항목만 예측확률을
사용한다. fake state는 heavy valence 0/charge 0이다.
허용 valence는 단일 평균값으로 축약하지 않는다.
`P_i`는 incident edges와 node head의 factorization 아래 **국소** 확률이다.
`P_Q`는 node marginals 아래 전체 graph net-charge 확률이다.

`models/chemistry_distribution.py`의 DP는 다항식 convolution을 수행한다.
최대 valence 초과 확률을 overflow bin에 보존하며, 버리거나 다시 정규화하지 않는다.
charge는 정수 shift 후 convolution하고, 짧은 graph의 padding은 최소 charge가 아니라
**실제 charge 0**으로 채운다. 쯔비터이온은 net charge 0이면 허용된다.

벤젠 perimeter edge마다 S/D=0.5이면 기존 평균 valence loss는 0이지만,
각 탄소의 올바른 heavy valence 확률은 0.5여서 새 항은 약 `log(2)`다.
그렇다고 전역 alternating cycle 확률까지 정확하게 계산한 것은 아니다.
독립적인 6개 edge의 유효한 alternating 배치는 2/64이며 이 joint probability를
새 loss의 국소 곱과 동일시하면 안 된다.

03부터 기존 평균 valence/charge weight를 0으로 하고 새 weight를 0.02/0.01로 둔다.
graph time 0.7 이후 선형 ramp로만 적용한다. 이 weight는 실험 시작값이며 검증된 최적값이 아니다.
Semantic loss의 “제약을 만족하는 확률질량” 관점에서 착안했으나,
전역 논리식을 푸는 solver를 구현한 것은 아니다.
[Semantic Loss](https://proceedings.mlr.press/v80/xu18h.html).

### 3.5 별도 correction objective

04부터 train batch의 25%에서 late time 0.85 상태를 추가로 만들어,
node 10% / edge 2%의 replacement와 중심을 보존하는 좌표 표준편차 0.1 Å perturbation을 준다.
원본 endpoint/XYZ/EST는 바꾸지 않는다. 변경된 visible 항목과 기존 masked 항목을
원본으로 복원하고, 실제 원자 좌표에만 MSE를 준다.

```text
L_total = L_base + chemistry_terms
          + 0.1 · I(correction branch) ·
            (lambda_x L_corr_x + lambda_s L_corr_s + lambda_e L_corr_e)
```

branch 확률 0.25를 역확률 보정하지 않는다. 평균 기여도는 이 확률에도 영향을 받는다.
이 off-path objective는 **별도 보조항**이며, 변경한 입력을 원래 CTMC forward kernel에서
나온 것으로 간주하지 않는다. 새 molecule/EST label을 추가하는 증강도 아니다.
같은 CFG unconditional branch를 유지하며 validation에는 적용하지 않는다.

FlowMol3의 self-conditioning/fake atom/geometry distortion 방향을 참고했다.
이번 visible categorical corruption과 별도 objective는 그 논문의 동일 구현이 아니다.
[FlowMol3, §2.6](https://arxiv.org/html/2508.12629v1).
원래 discrete/continuous flow의 역할은 분리해서 유지한다.
[Discrete Flow Models](https://arxiv.org/abs/2402.04997).

## 4. 생성 평가·체크포인트·계산 결과 연결

`analysis/generation_callback.py`는 50 epoch마다 고정 validation cohort 최대 64개,
seed 17041, CFG=1, 100 time grid points로 평가한다. 100 points는 99 update이다.
validation에서만 조건과 원자 수를 가져온다. test는 조건 선택에 쓰지 않고,
전체 DB 신규성 검사에서만 참조한다. 이 패널은 EST=0 전용 평가가 아니다.

- whole-molecule RDKit sanitize: largest fragment만 살려서 성공 처리하지 않는다.
- raw valid / connected / neutral closed-shell / aromatic / topology warning /
  unoptimized geometry / novelty를 분리한다.
- 3·4원 및 9원 이상 ring은 warning이며, 존재만으로 화학적 불가능이라고 하지 않는다.
- geometry는 covalent-radius 상대 거리, nonbonded clash, 방향족 ring planarity의
  간단한 screen이다. PoseBusters 통과나 물리적 안정성 인증이 아니다.
- 결합이 없는 단원자는 raw valid일 수 있지만 geometry 평가 가능/성공으로 세지 않는다.
  요청 대비 실제 heavy-atom 수 차이와 retention도 기록한다.
- 좌표 재임베딩/최적화/수소 수 변경으로 성공률을 높이지 않는다.
- SDF에는 unrepaired RDKit-valid 출력만 저장하고, invalid endpoint는
  `raw_states.pt`에 보존한다. PNG는 DB에 없는 unique connected 출력의 RDKit 2D이다.
- Python/NumPy/Torch RNG와 모델 train/eval 상태를 복원한다.
- 같은 epoch 재평가도 새 attempt 폴더를 만들며 기존 결과를 덮어쓰지 않는다.
- 평가 중 수치/rollout 예외는 별도 JSON으로 남기고 이전 metric으로 checkpoint를
  선택하지 않는다. 학습 자체의 오류를 숨기는 예외 처리로 확장하지 않았다.

`val_total_loss` checkpoint와 `gen_geometry_eligible_rate` checkpoint를 별도로 저장한다.
하위 `checkpoints/generation_quality/`의 checkpoint도 부모 `config.yaml`을 찾아
재개할 수 있도록 run-directory 해석을 수정했다.
후자는 조건 충실도나 aromatic-low-EST 최적 모델이라는 뜻이 아니다. 조건을 무시하거나
단순 구조만 생성해서 유효율이 오르는 모델을 최종 승자로 선정하면 안 된다.
고정 EST=0, 동일 count/noise, 여러 seed의 비교와 XTB/sTDA 결과가 추가로 필요하다.

자동 평가에서 XTB를 실행하거나 surrogate 결과를 물리적 EST로 부르지 않는다.
실제 계산은 기존 diffusion-coordinate XTB/sTDA 절차로 수행한 뒤 다음 도구로 연결한다.

```text
tools/attach_stage7_condition_results.py
  --report <generation/report.json>
  --results <CSV: generation_index,status,s1_eV,t1_eV>
  --calculation-manifest <JSON: model_state_sha256,source_sdf_sha256,method>
  --output <새 결과 JSON>
  --tolerance 0.1
```

계산 provenance와 중복 index/NaN/비대상 구조를 검사한다.
성공 계산의 MAE와 전체 요청 분모의 coverage/hit/aromatic-hit yield를 분리한다.
결측 계산은 조건 적중으로 세지 않으며, DB↔XTB offset을 임의 적용하지 않는다.
자동 생성 JSON에는 물리적 조건 평가는 아직 미측정이라고 명시한다.

train/val/test processed tensor, normalization, state metadata, marginals,
count histogram, conversion report의 9개 SHA-256과 model-state hash를 checkpoint에 기록한다.
경로가 옮겨져도 내용이 같으면 허용하고, 내용 불일치는 중단한다.
이 기능은 source-code hash나 optimizer-state hash까지 검증하는 기능은 아니다.

## 5. 검증과 재현

저장 위치:

```text
<repository>/runs/stage7_fixed_data_verification_20261020_v6/verification.json
```

최종 v6에서 11개 설정 모두 CPU 소배치 forward/backward, finite gradient,
optimizer update, manifest 불일치 거부, checkpoint round-trip을 통과했다.
소배치에는 train의 최소/최대 크기(6/69 heavy atoms)와 다른 샘플을 포함한다.
설정별 smoke는 속도를 위해 network 폭을 축소했으므로 07의 capacity 효과 검증은 아니다.

실제 Lightning 학습 1-step → validation → generation callback → quality checkpoint
저장 → `load_from_checkpoint(strict=True)`도 통과했다.
이 checkpoint는 disposable smoke 결과이며 생산용 학습 결과가 아니다.
06의 전체 폭 모델 7,987,522 parameters에서도 실제 DB 4-graph 소배치
forward/backward를 별도로 통과했다. batch 640 GPU 검증은 아니다.

전체 DB의 **학습층을 통과하지 않는 codec** RDKit non-stereochemical identity:

| Split | 동일 복원 |
|---|---:|
| Train | 1,264 / 1,264 |
| Validation | 158 / 158 |
| Test | 159 / 159 |
| 합계 | 1,581 / 1,581 |

검증 전후 9개 DB 파일 hash가 동일하다.
기존 e1944 checkpoint도 새 코드의 legacy defaults로 strict load되고,
model-state SHA-256이 저장된 tensor와 동일함을 별도로 확인했다:
`b79d036485e24cbbeeab7828df96145d6f5e6a69edd443bf156def758cdfaeb2`.
그러나 **Stage 7 신규 구조로 e1944를 이어 학습한 것은 아니다**.
새 설정은 fresh training 대상이다.

전체 단위/회귀 테스트 **90/90 통과**. 기존 회귀 테스트에 다음을 추가했다.

- 서로 다른 weight에서 loss 값/gradient, empty mask, clean-graph weight 격리;
- Kabsch 중심/반사, 다층 field 반사 등변성;
- DP 완전 열거 일치·overflow 확률 보존·autograd gradcheck;
- 기대값 함정 반례, P/S 다중 valence, global charge와 zwitterion, padding;
- feedback 교차 영향·0 초기화·학습 가능성;
- correction target/EST 불변·edge 대칭·COM 보존;
- YAML 순환, data/weight hash, normalization hash, raw-eV train/sample 일치;
- generation RNG 복원·RDKit PNG·덮어쓰기 방지;
- 실패/오래된 generation metric의 checkpoint 선택 방지;
- 단원자 geometry의 공집합 성공 방지 및 nested checkpoint resume;
- 외부 S1/T1 결과의 provenance·분모·중복·NaN·경계 허용오차.

저장소 루트에서 재현:

```bash
PYTHONPATH=.:MolGuidance_guidance_selector_e987 OMP_NUM_THREADS=2 \
  /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python -m unittest discover \
  -s MolGuidance_guidance_selector_e987/tests -v

PYTHONPATH=.:MolGuidance_guidance_selector_e987 OMP_NUM_THREADS=2 \
  /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python \
  MolGuidance_guidance_selector_e987/tools/verify_stage7.py \
  --output runs/stage7_verification_NEW_UNIQUE_DIRECTORY
```

개발 중 v1은 새 timestep 명시화의 NameError, v2는 oracle 테스트의 `ue_mask` 누락으로
중단됐고 수정 후 v3부터 v6까지 통과했다. 기존 sampler는 이미 `**kwargs`로 timestep을 전달했다.
이를 기존 전달 누락 버그로 보고하지 않는다. 실패 결과도 삭제하지 않았다.

## 6. 아직 해결·입증하지 않은 것

1. 전역 Kekulé cycle/connectivity joint solver는 추가하지 않았다. 국소 valence
   probability가 높아도 모든 조건이 동시에 충족된다고 보장하지 않는다.
2. epsilon floor와 큰 graph에서 작은 compatibility probability는 gradient를 약하게
   만들 수 있다. late-time 적용/작은 weight는 완화책이지 수학적 제거가 아니다.
3. correction의 빈도·크기가 큰 경우 본래 endpoint objective와 충돌할 수 있다.
   03↔04 ablation과 실제 rollout을 봐야 한다.
4. 새 dropout/weight decay/작은 capacity/조건 약화는 일반화와 조건 충실도를
   악화시킬 수도 있다. train loss 감소만으로 채택하지 않는다.
5. batch 640은 기존 값만 상속했다. 새 DP/correction이 추가하는 GPU peak memory와
   시간은 아직 측정하지 않았으며, GPU 70% 사용을 달성했다고 주장하지 않는다.
6. 기존 `training.max_num_edges`는 실제 batch sampler에 연결되지 않았다.
   이 설정을 메모리 제한으로 믿으면 안 된다. 별도 edge-budget 작업이 남아 있다.
7. old `SampleAnalyzer`/생성 CLI의 느슨한 valid 정의를 모두 고친 것은 아니다.
   Stage 7 새 checkpoint 평가에서는 해당 정의 대신 strict callback을 사용한다.
8. 선택적 `campbell_rate_matrix`의 기존 guidance 한계는 미수정이다.
   새 평가에서는 검증한 `campbell`, CFG=1 경로를 명시한다.
9. Stage 7 correction 및 condition-residual ablation의 MG 확장은 지원하지 않아
   명시적으로 거부한다. 기존 기본 MG 경로를 수정 완료한 것으로 오해하면 안 된다.
10. 전체 DB oracle identity는 decoder의 무손실성 검사다. 모델 생성 성공률,
    독창성, 합성 가능성, 들뜬 상태 정확성의 증명은 아니다.

장기학습은 이번 코드 수정 요청에서 시작하지 않았다.
다음 실행은 01/01a/01b의 loss 대조부터 시작하고, 같은 optimizer-step 예산에서
02 → 03 → 04를 비교해야 한다. generation/geometry/외부 EST 결과가 나빠지면
해당 추가 항을 되돌릴 수 있도록 각각 독립 설정으로 보존했다.
