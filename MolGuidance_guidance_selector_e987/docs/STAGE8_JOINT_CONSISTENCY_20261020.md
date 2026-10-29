# Stage 8: 단계별 코드 개선 및 정합성 검증

날짜: 2026-10-20 UTC. 브랜치:
`feature/molguidance-stage8-joint-decoding-mg-budget`.
출발 commit: `05e87ef`. 이 문서는 Stage 7 결과를 덮어쓰지 않는다.
Stage 7의 600-step 실험 결과와 Stage 8의 동작 검증/장기학습은 서로 다르다.

## 1. 보존 범위와 판정의 의미

- HP1KG 1,581개, train/val/test=1,264/158/159를 그대로 사용한다.
- 공식 최적화 XYZ, 인덱스, EST=S1−T1, 원소/H/전하 state 및 split/normalization을 보존한다.
- 기존 DB/checkpoint/생성물/XTB 결과는 변경하지 않았다.
- H는 heavy atom에 붙은 정수 count이지 독립적인 좌표 node가 아니다.
- 37개 실제 joint state + fake state, Kekulé NONE/SINGLE/DOUBLE/TRIPLE을 유지한다.
- 모델은 coordinate endpoint flow + masked categorical CTMC이다.
  Gaussian-noise DDPM으로 새로 바꾼 것은 아니다.
- 외부 데이터, 합성 EST label, held-out support의 train 편입은 없다.
- 검증 통과는 알려진 계약·테스트 범위의 코드 정합성 판정이다.
  모든 입력에 대한 형식 증명, 생성 성능·화학 합성 가능성·조건 충실도 보장이 아니다.

DB:

```text
/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz
```

## 2. 단계와 상호작용

`configs/local/stage8/`의 설정을 순차 검증했다.

| 설정 | 핵심 변경 |
|---|---|
| 00_control | Stage 7 feedback 기반, 실제 edge budget, raw/solver 분리 평가 |
| 01_distribution | 기존 평균 valence/charge 대신 분포 호환성 loss |
| 01a_kekule_augmentation | 동일 분자의 전역 Kekulé 표현을 corruption 이전에 증강 |
| 02_correction | geometry-only correction, 지연 시작/ramp/별도 loss 크기 상한 |
| 03_dropout | GVP dropout 0.05 |
| 04_weight_decay | weight decay 1e−5 |
| 05_small_model | scalar/edge 128/64, update 4 |
| 06_condition_scale | 02에서 분기, condition residual scale 0.5 |
| 07_mg | 06 기반 MG, EMA/soft target/scale 0–2 |
| 08_longtrain_cfg | 05 + condition scale 0.5, LR 2.8e−4, 최대 10,000 epoch |

06/07은 작은 모델을 상속하지 않는다. 08은 CFG를 유지한 통합 학습 후보이다.
MG는 별도 경로로 검증했고, CFG를 MG로 바꿨다고 보고하지 않는다.
Stage 8은 기존 aromatic auxiliary weight를 올리지 않았다.
원래의 local aromatic auxiliary는 전역 Kekulé 유효성의 증명이 아니며,
새 global solver와 같은 것으로 해석해서는 안 된다.

## 3. 전역 최종 decoding

구현: `molguidance/analysis/joint_decode.py`.
최종 endpoint 확률에서만 선택적으로 실행한다. reverse 중간 상태를
projection으로 바꿔 forward kernel과 어긋나게 만들지 않는다.

이진 변수:

```text
z[i,s,v] = atom i가 joint state s와 train-supported heavy valence v를 선택
b[ij,k]  = edge ij가 bond order k∈{0,1,2,3}를 선택
```

목적함수는 endpoint의 factorized log probability 합을 최대화한다.

```text
maximize sum z[i,s,v] log p_i(s) + sum b[ij,k] log p_ij(k)
subject to:
  각 atom/edge에서 정확히 한 선택
  sum_j sum_k k b[ij,k] = sum_(s,v) v z[i,s,v]
  요청한 real atom count (주어진 경우), net charge
  super-root 단일 commodity flow에 의한 real atom 전체 연결성
```

fake는 valence 0이고 root는 real atom 중에서 선택한다. 첫 slot이 fake여도
root를 첫 atom에 고정하지 않는다. fused/spiro 구조의 공유 edge/atom에는
전역적으로 변수가 하나만 있으므로 고리별 독립 assignment 충돌을 피한다.
입력 확률이 0인 state/bond는 solver support 밖이다.

후보는 RDKit sanitize, H/charge 보존, connected/closed-shell,
global Kekulize를 통과해야 한다. 명시적 aromatic edge를 제공한 경우에만
그 edge들의 aromatic 실현도 요구한다. 일반 생성에는 정답 aromatic topology를
주입하지 않는다. 단순히 방향족 분자를 무조건 만들도록 강제한 것은 아니다.
실패한 이산 해에는 no-good cut을 넣되 시간/횟수 상한을 둔다.
시간제한에 걸린 해는 제약과 integrality를 따로 검증하고
`feasible_time_limit`으로 표시한다. `optimal`과 혼동하지 않는다.

raw `rdkit_mol`, 원래 slot state, 좌표는 변경하지 않는다.
`joint_rdkit_mol`/report/SDF/PNG를 별도로 붙인다. 이는 constrained MAP이므로
원래 CTMC의 unbiased sample이 아니고, valence를 통과했다고 geometry/EST까지
좋아졌다는 뜻도 아니다. solver formulation/RDKit 비용은 solver time limit 밖에도
발생할 수 있어 엄밀한 wall-clock deadline이라고 주장하지 않는다.
[SciPy MILP 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).

## 4. 전역 Kekulé 표현 증강: train–sample kernel 불일치 방지

구현: `data_processing/kekule_augmentation.py`, `dataset.py`.

고정된 원자/H/전하, 결합의 존재, 좌표를 유지한 채 aromatic component의
S/D 위치만 바꾼다. atom별 원래 incident aromatic double-bond 수를 demand로
삼고, demand=1인 vertex들에 weighted perfect matching을 푼다.
fused system은 개별 ring이 아니라 공유 atom/edge를 포함한 component 전체에서 푼다.
odd heterocycle의 pyrrolic N/O/S 등은 demand=0일 수 있으므로 모든 원자에
double 하나를 억지로 배정하지 않는다.

8회 random-weight matching 시도와 원래 assignment를 pool로 보존한다.
선택한 각 후보를 전체 분자의 RDKit canonical non-stereo identity로 검증한다.
non-S/D aromatic perimeter 또는 demand>1인 특수 component는 원본을 유지한다.
그것을 무조건 5/6원 benzene형으로 강제하지 않는다.

중요한 순서는 다음과 같다.

```text
원래 동일 분자 → 검증된 Kekulé 표현 선택 → clean target 확정
             → 원래 CTMC corruption → noisy input/target 학습
```

공개된 결합을 본 뒤 정답만 다른 배열로 바꾸지 않는다.
따라서 새로운 표현에 대해서도 noisy input과 target이 동일 forward kernel에서
연결된다. 추가된 선택 RNG는 격리하여 동일 paired sample의 좌표 prior를
대조군과 다르게 만들지 않는다. val/test target은 증강하지 않는다.

학습 1,264개 전체에서 pool 4,414개를 확인했다. 995개 분자는 둘 이상의
Kekulé 표현을 가지며 identity 불일치 0건이다. 특수 component 1건은 원본을 보존한다.
이는 **추가 4,414개 분자나 EST label**을 만들었다는 뜻이 아니다.
모든 유효 assignment를 균등하게 샘플링한다는 주장도 아니다.
matching은 지수적 cycle pattern 열거 대신 polynomial blossom 구현을 사용한다.
[NetworkX 공식 matching 문서](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html).

## 5. MG, EMA, CFG rate 경로

MG categorical target의 hard argmax를 제거했다. clean one-hot y,
EMA conditional/unconditional endpoint logits l_c/l_u, scale g에 대해:

```text
q_g = softmax((1−g) l_u + g l_c) on supported classes
rho = min(|g−1|, 1)
q_target = (1−rho)y + rho q_g
L = CE(q_target, student_logits) on active masked items
```

target은 detached이고 probability simplex 위에 있다. g=1이면 기존 clean CE와
정확히 같다. categorical feature는 coordinate time이 아니라 graph time으로
MG cutoff를 판단한다. 노드/edge pack은 각각 n, n(n−1)/2를 사용한다.
EMA 생성값으로 원래 clean graph tensor를 in-place 수정하던 경로도 제거했다.
좌표 teacher shift는 g−1 convention으로 맞춰 1=conditional을 유지한다.

이는 **probability-space distillation의 자체 categorical 확장**이다.
연속 MG 논문의 likelihood theorem이 자동으로 성립하거나 exact guided marginal을
학습한다고 주장하지 않는다.
[원래 MG 논문](https://arxiv.org/abs/2502.12154),
[Discrete Flow Models](https://arxiv.org/abs/2402.04997).

EMA는 실제 `optimizer_step` 뒤 업데이트하고 buffer도 복사한다.
teacher property embedder는 teacher 가중치를 사용하며 항상 eval 상태를 유지한다.
MG 학습/생성 scale 범위는 0–2, 기본값 1이다. 범위 밖 생성과 scale-aware가 아닌
모델의 variable-scale MG는 명시적으로 거부한다.
scale 0의 결과는 동일 noise/count에서 입력 property와 무관함을 검사했다.
correction branch는 원본 복원 objective이므로 MG pseudo target이 아니라
conditional/unconditional anchor 1/0을 사용한다.

선택형 `campbell_rate_matrix` 경로도 수정했다.

- log(0)에 eps를 넣어 금지된 transition을 되살리지 않는다.
- unsupported logits는 softmax 전에 차단한다.
- probability-space와 rate-space guidance의 정규화 위치를 구분한다.
- `I+dt Q`를 원소별 clip하지 않고 mask-star frozen generator의 정확한 전이를 쓴다.
- 마지막 단계는 요청한 temperature/guidance의 categorical sampling이다.
  임의 argmax로 바꾸지 않는다.

mask→class j rate r_j, class→mask rate η, λ=Σr_j인 고정 generator의
transition을 닫힌 형태로 구현하고 `torch.matrix_exp(Q·dt)`와 비교했다.
이는 **각 step에서 frozen한 generator**에 대한 정확한 전이이지,
시간에 따라 변하는 신경망 field의 연속 적분 전체를 정확히 푼다는 뜻은 아니다.
기본 production 평가의 `campbell` 경로는 이 별도 옵션과 구분된다.

## 6. 배치·정규화·correction

`EdgeBudgetBatchSampler`는 fake 최대치를 포함한 directed complete graph
`n_slots(n_slots−1)`를 계산한다. 단일 분자만으로 budget을 넘으면 사전에 거부한다.
한 epoch에서 누락/중복 없이 모든 train molecule을 사용한다.
fixed-cost packing으로 epoch별 batch 수가 바뀌어 scheduler가 오작동하지 않게 했다.
현재 single-device만 지원하며 DDP에서는 조용히 중복 학습하지 않고 거부한다.

correction은 200 step부터 시작, 400 step 동안 ramp, branch probability 0.25,
weight 0.02, 좌표 표준편차 0.05 Å, node/edge replacement 0을 사용한다.
가중 correction loss의 크기가 기본 loss의 10%를 넘지 않도록 detached scale을 둔다.
이는 **loss magnitude 상한**이지 gradient norm/각도 상한이 아니다.
기존 Stage 7 correction arm의 낮은 생성 성능 원인이 완전히 규명됐다는 뜻도 아니다.

dropout 0.05는 실제 GVP 모듈에 전달된다. validation에서는 eval 모드로 꺼진다.
weight decay 1e−5, 작은 모델 128/64×4, condition residual 0.5는
모두 실험 가능한 설정이며 생성 성능에 유리함이 아직 입증된 것은 아니다.
residual 0.5를 “조건 반영률 50%”라고 해석하면 안 된다.
EST raw eV는 기존 train normalization 후 256-d property embedding에 전달된다.
현재 condition Gaussian expansion은 꺼져 있다.

## 7. 검증 결과와 보존된 실패 기록

최종 통합 결과:

```text
runs/stage8_integration_20261020_v3/verification.json
```

| 검사 | 결과 |
|---|---:|
| pure codec train identity | 1,264/1,264 |
| pure codec val identity | 158/158 |
| pure codec test identity | 159/159 |
| train-supported solver oracle identity | 1,580/1,580 |
| unit/regression tests | 110/110 |
| 설정별 GPU optimizer/checkpoint/생성 연결 | 10/10 |
| 실제 optimizer steps | 각 설정 3 step |
| DB hash 전후 동일 | 통과 |

test index 82의 Si–H state는 train support 밖이다. pure codec은 정상 복원하지만
생성 solver의 oracle 통계에서는 따로 제외했다. test를 읽어 support를 넓히지 않았다.

```text
CC1=C(C)N(C)C(=[SiH][Si](C(C)(C)C)(C(C)(C)C)C(C)(C)C)N1C
```

검사에는 loss/gradient finite, clean target 불변, symmetry/center/반사 계약,
EMA 실제 변화, strict checkpoint tensor hash round-trip, MG scale 0/2,
CFG rate 경로 실제 sampling, train-only augmentation, 모든 graph의 budget/coverage,
zero support/infeasible solver, 작은 문제 완전 열거 최적해 비교,
odd heterocycle/fused/charged/zwitterionic/phosphorus/sulfur/작은 고리 사례가 포함된다.

GPU 통합검사는 full production width이나 batch 4, 3 step, 12 time points의
작은 검사이다. branch 실행을 위해 correction/MG 시작 시점만 즉시로 override했다.
실제 장기학습은 원래 schedule로 fresh start하며 전체 batch capacity probe를 별도로 한다.
생성 패널은 경로 검사이지 성능 추정이 아니다. 물리적 EST는 이 검사에서 미측정이다.

v1은 verifier에서 RDKit sanitization 전/후 SMILES를 비교한 검사기 오류로 실패했다.
수정 후 v2를 통과했고, paired-RNG 격리와 추가 scale/rate 검사까지 반영한 최종본은 v3다.
v1/v2 결과도 삭제하지 않았다. import 경로가 빠진 unittest 실행 실패도 있었으며,
아래 PYTHONPATH로 전체 테스트가 통과했다.

## 8. 실행·학습 기록

repository root에서:

```bash
PYTHONPATH=.:MolGuidance_guidance_selector_e987 OMP_NUM_THREADS=2 \
  /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python -m unittest discover \
  -s MolGuidance_guidance_selector_e987/tests -v
```

`tools/verify_stage8.py --output <새 폴더> --device cuda`는 세션 분리된 검증이다.
`tools/train_stage8.py`는 완료된 검증 JSON, 동일 code/config/data hash를 요구한다.
fresh 학습이 기본이며 `--resume`을 명시할 때만 checkpoint를 재개한다.
resume도 새 output directory를 요구하여 기존 run을 덮어쓰지 않는다.

이번 장기학습 요청의 실행 경로:

```text
/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/runs/molguidance_stage8_cfg_longtrain_20261020
```

설정: `08_longtrain_cfg.yaml`, 최대 10,000 epoch, FP32/TF32 off,
LR 2.8e−4(5 epoch warmup), 최대 batch 128, directed edge budget 40,000.
GPU allocator 상한 70%는 실제 메모리 점유율 70% 달성과 다른 뜻이다.
작은 데이터/작은 모델에 맞는 배치를 유지하며 미리 메모리를 채우지 않는다.

시작 시 largest planned batch의 forward/backward/optimizer capacity probe 모델은
버리고 seed를 다시 고정한 새 모델로 학습한다.
소스 snapshot, expanded config, manifest, `train.log`, `status.json`,
`capacity_probe.json`, CSV metrics, `epoch_history.jsonl`을 저장한다.
best val 5개 + last checkpoint, generation geometry 기준 3개를 별도 보존한다.
50 epoch마다 64개 strict validation 생성과 RDKit 2D PNG, raw/solver별 report를 저장한다.
XTB/sTDA 조건 충실도는 외부 계산 후 기존 provenance-aware 연결 도구로 평가한다.
train-only distribution loss나 condition shuffle gain을 물리적 조건 정확도로 부르지 않는다.

실제 시작 확인: PID **2849871**, PPID **1**, SID/PGID **2849871**.
새 session에서 stdin `/dev/null`, 파일 stdout/stderr로 실행되어 Codex 세션에 묶이지 않는다.
production capacity probe는 64 molecules / 실제 directed edges 31,842 /
최대 추정 40,000에서 finite forward/backward/update를 통과했다.
GPU peak allocated 3,156.7 MiB, reserved 3,236 MiB로 **70%를 점유한 것은 아니다**.
epoch당 43개 가변 크기 batch로 train 1,264개 전체를 사용한다.
확인 시점 epoch index 10 / optimizer step 440, 첫 val 5.6643,
epoch index 8의 val 2.9003이며 best/last checkpoint 저장을 확인했다.
초기 loss 감소는 생성 성공률 향상을 아직 증명하지 않는다.

## 9. 남은 실증적 판단

- raw connected/neutral/closed-shell/geometry와 projected 성공률을 분리한다.
- 검증을 통과해도 de novo 방향족 생성률이나 EST≈0 적중률은 낮을 수 있다.
- 작은 train set에서 overfitting, conditioned size bias, discretization error는 남을 수 있다.
- train-supported solver는 미관찰 valence를 제한하고 constrained MAP bias를 만든다.
- 파라미터 변경별 인과적 성능 비교에는 동일 budget/seed의 추가 ablation이 필요하다.
- 장기학습 시작 여부는 위 run의 `launch.json`/`status.json`/실제 PID로 판단한다.
  이 문서의 설정 설명만으로 “학습 중”이라고 판정하지 않는다.
