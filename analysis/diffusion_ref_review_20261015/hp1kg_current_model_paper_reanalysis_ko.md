# HP1KG EST 조건부 3D 분자 생성 모델: 문헌 재분석과 현재 코드 대조

작성 기준일: 2026-10-16 UTC

## 1. 결론 요약

현재 모델은 엄밀히 말해 단순한 Gaussian DDPM이 아니다. 좌표 `x`에는 centered-normal prior에서 데이터 좌표로 가는 continuous endpoint flow를, 원자 종류 `a`, 형식전하 `c`, 결합 `e`, 부착 수소 수 `h`에는 mask-state CTMC discrete flow를 쓰는 PropMolFlow 계열의 혼합 모델이다. EST 조건은 정규화된 단일 scalar를 MLP로 임베딩하고, 10% condition dropout으로 classifier-free guidance(CFG)를 학습한다.

문헌과 현재 코드의 대조에서 가장 중요한 결론은 다음과 같다.

1. 현재까지의 비정상 고리·방향족 붕괴·나쁜 생성 좌표는 flow/DDPM 자체가 분자를 못 만들어서라기보다, 학습과 추론 사이의 off-manifold drift 및 결합 표현의 전역 일관성 부족과 더 잘 맞는다.
2. FlowMol3가 제시한 `self-conditioning + fake atoms + train-time geometry distortion`은 지금 관찰한 실패 양상을 직접 겨냥한 개선이다. 단순 loss weight 조정이나 CFG scale 탐색보다 우선순위가 높다.
3. Megalodon이 지적한 단일 시간 변수 문제도 현재 코드에 그대로 존재한다. 현재는 좌표와 모든 이산 modality가 같은 `t`를 공유한다. discrete time과 continuous time을 독립적으로 샘플링해 “그래프가 비교적 깨끗하지만 좌표는 noisy”인 상태와 그 반대 상태를 모두 학습시키는 것이 필요하다.
4. 현재 데이터는 방향족이 부족하지 않다. train 1,264개 중 1,086개(85.9%)가 방향족이고, 실제 결합 36,379개 중 aromatic class가 17,333개다. 그럼에도 방향족이 생성되지 않는다면 데이터 존재 여부보다 aromatic edge를 각 pair의 독립 class로 예측하는 방식과 cycle 전체 일관성의 문제가 더 유력하다.
5. FlowMol3는 모든 분자를 Kekulé화하고 aromatic bond class를 별도로 생성하지 않는다. 현재 모델은 `NO_BOND/SINGLE/DOUBLE/TRIPLE/AROMATIC` 5-class를 직접 생성한다. 다음 재학습 branch에서는 single/double Kekulé target으로 통일하고 RDKit sanitization에서 aromaticity를 복원하는 대조 실험이 필요하다.
6. PropMolFlow와 FlowMol3의 강한 결과는 각각 약 100k QM9 분자 또는 243,480개 GEOM-Drugs 분자와 5,741,535 conformer를 사용했다. 현재 train은 1,264분자, 분자당 공식 S0 geometry 하나다. 7.16M parameter 모델에 이 데이터만 사용하므로, 구조 변경만으로 논문 수준의 일반화를 기대하기 어렵다. 외부 3D 데이터로 unconditional backbone을 사전학습하고 HP1KG의 EST condition을 fine-tune하는 것이 가장 큰 데이터 측 개선이다.
7. 현재 학습은 유지한 채 분석했으며, 설정된 2,000 epoch에 도달해 자연 종료됐다. 종료 직전 로그는 epoch 1999 train loss 0.490, val loss 0.542다. 저장된 최신 checkpoint는 epoch 1888/global step 3778이다.

## 2. 현재 실행 모델의 정확한 상태

### 2.1 데이터

- 전체 1,581분자: train 1,264 / validation 158 / test 159
- 좌표: PSDI 공식 S0 B3LYP/6-31G* 최적화 좌표, Å, float32
- heavy atom 수: 6–69
- 모델 graph에는 heavy atom만 node로 들어간다.
- 공식 HDF5에는 explicit-H 좌표가 보존돼 있지만, 현재 모델 view는 heavy atom 좌표와 각 heavy atom의 attached-H count 0–3만 쓴다.
- 조건: `EST = S1 - T1` 단일 scalar
- train EST 범위: 약 -1.133–2.655 eV, median 0.480 eV
- train에서 `|EST| <= 0.1 eV`는 181개, `|EST| <= 0.5 eV`는 607개다. 따라서 EST≈0 자체는 전역적으로 완전한 OOD가 아니다. 다만 특정 atom count·원소 조합·scaffold와 함께 주면 joint support가 희박할 수 있다.

원소별 train atom 수:

| 원소 | atom 수 |
|---|---:|
| C | 25,736 |
| O | 2,757 |
| N | 2,376 |
| S | 1,471 |
| F | 422 |
| P | 166 |
| Cl | 129 |
| Se | 111 |
| Br | 72 |
| B | 53 |
| Si | 52 |
| As | 1 |

어휘에 원소가 들어 있다고 그 원소의 화학 환경을 생성할 수 있는 것은 아니다. 특히 As 1개, B/Si 약 50개는 사실상 일반화 학습량이 아니다.

### 2.2 생성 상태

현재 joint state는 다음과 같다.

\[
G_t=(X_t,A_t,C_t,E_t,H_t),
\]

- `X`: heavy-atom Cartesian coordinate, continuous flow
- `A`: 12종 atom type + mask
- `C`: formal charge + mask
- `E`: 모든 atom pair의 `none/single/double/triple/aromatic` + mask
- `H`: attached-H count `0/1/2/3` + mask
- `c`: 정규화된 EST scalar

완전연결 directed graph에서 message passing을 수행하므로 edge 수와 memory는 대략 O(N²)이다. 원자 수는 reverse process가 생성하는 변수가 아니라 시작 전에 train의 heavy-atom-count marginal에서 샘플링하거나 사용자가 고정한다. 즉 현재 생성기는 fixed-size conditional generator이며, 잘못 잡은 node 수를 생성 중 삭제하거나 추가할 수 없다.

### 2.3 조건과 regularization

- EST scalar → 256차원 MLP embedding
- Gaussian/RBF expansion: 현재 `false`
- CFG condition dropout: 0.1
- GVP message/node dropout: 현재 vector-field 생성부에서 지정하지 않아 기본값 0.0
- batch size 1024, train molecule 1,264개이므로 epoch당 optimizer step은 2개다.
- 2,000 epoch라도 총 optimizer step은 약 4,000개뿐이다.

CFG dropout은 일반적인 network regularization dropout과 역할이 다르다. 전자는 conditional/unconditional branch를 동시에 학습하기 위한 것이며, GVP 자체의 overfitting을 직접 억제하지 않는다.

## 3. 직접 관련 논문 분석

### 3.1 PropMolFlow: 현재 모델의 직접 기반

PropMolFlow는 좌표, atom type, formal charge, bond order를 공동 생성한다. 좌표에는 continuous flow matching, 이산 변수에는 mask prior CTMC를 쓰고 총 loss는 modality별 loss의 가중합이다. 논문의 기본 가중치 `(x,a,c,e)=(3.0,0.4,1.0,2.0)`는 현재 모델에도 유지됐고 H-count loss 1.0이 추가됐다.

중요한 차이:

- 논문: explicit-H QM9, 약 100k generator training 규모, 최대 29 all atoms
- 현재: heavy-atom HP1KG 1,264개, 6–69 heavy atoms, H는 count class
- 논문: 데이터의 잘못된 valence/charge를 먼저 수정했을 때 생성 validity가 크게 향상
- 현재: graph identity는 검증됐지만 official coordinate–SMILES H mapping fallback이 61분자/164 atom에 존재한다.
- 논문: ID 평가 시 atom count를 먼저 뽑고 `P(property | atom count)`에서 조건을 뽑아 양립 가능한 pair를 사용
- 현재: 임의 EST와 atom-count marginal을 독립 조합할 수 있어 joint-support 밖 조합이 발생할 수 있다.

Property embedding도 “RBF면 항상 더 좋다”는 결과가 아니다. 논문에서 embedding interaction 선택에 따라 gap MAE가 391–551 meV로 40% 이상 달랐으며 property별 최적 방식도 다르다. 따라서 현재 MLP를 곧바로 RBF로 대체하기보다 동일 split/seed에서 다음을 비교해야 한다.

1. MLP + concatenate-sum(현재)
2. data-range Gaussian expansion + concatenate-sum
3. FiLM/AdaLN 방식의 layer-wise condition injection

논문의 OOD 실험도 99th percentile 목표에서 분포가 목표 방향으로 이동한 것이지 정확히 목표에 수렴한 것은 아니다. condition은 train support를 무시하는 장치가 아니다.

### 3.2 FlowMol3: 현재 실패 양상과 가장 직접적으로 일치

FlowMol3는 PropMolFlow와 같은 multi-modal flow 계열에 세 가지 self-correction을 추가한다.

1. Self-conditioning: 50%의 train step에서 첫 예측을 no-grad로 계산한 뒤, 그 예측을 두 번째 denoising 입력으로 사용한다. inference에서는 항상 사용한다.
2. Fake atom: 실제 원자 수의 최대 약 30%까지 fake atom을 anchor 주변에 추가해 잘못된 node 수와 국소 topology를 수정하는 능력을 학습한다.
3. Geometry distortion: 중간/후반 timestep의 일부 atom 좌표를 추가로 교란해, inference에서 학습 interpolation path를 벗어난 geometry에서 회복하도록 학습한다.

GEOM-Drugs explicit-H 평가에서 세 장치를 모두 제거하면 PoseBusters validity가 95.9%에서 78.1%로, median xTB relaxation energy가 4.50에서 14.75 kcal/mol로 악화됐다. geometry distortion은 validity와 OOD ring, self-conditioning은 relaxation energy, fake atom은 ring plausibility에 가장 큰 영향을 보였다.

이는 현재 프로젝트의 다음 현상과 직접 대응한다.

- 과도한 3·4원/거대 고리
- 생성 좌표의 큰 relaxation 변화
- fixed node count 때문에 불필요한 atom을 제거하지 못함
- 매 reverse step의 작은 오차가 누적돼 train path 밖으로 이탈

다만 강한 distortion은 rare valid motif를 덜 생성하는 mode concentration을 만들었다. validity만 최대로 하는 hyperparameter를 고르면 안 되고, ring-size JSD와 rare motif coverage를 함께 봐야 한다.

또한 FlowMol3는 GEOM-Drugs를 전부 Kekulé화하고 aromatic bond class를 별도로 모델링하지 않는다. 이 점은 현재 aromatic 5-class 방식과 가장 중요한 표현 차이다.

### 3.3 Megalodon: 시간 결합 문제와 topology/geometry 역할 분리

Megalodon은 invariant Transformer가 discrete atom/bond topology를, equivariant EGNN layer가 coordinates를 갱신한다. 핵심 발견은 shared time을 쓰는 기존 모델에서 초기 절반 동안 bond가 사실상 모두 사라져, 2D graph가 3D geometry를 안내하지 못했다는 것이다.

해결책은 구조와 이산 graph에 독립적인 시간 변수 `t_continuous`, `t_discrete`를 샘플링하는 것이다. 이렇게 하면 다음 혼합 난이도를 모두 보게 된다.

- clean-ish graph + noisy geometry
- noisy graph + clean-ish geometry
- 둘 다 noisy
- 둘 다 nearly clean

현재 코드는 batch molecule마다 하나의 `t`를 뽑아 x/a/c/e/h 모두에 사용하므로 이 문제가 재현될 가능성이 있다. modality별 schedule 계수만 바꾸는 것과 독립 time sampling은 동일하지 않다.

Megalodon 결과에서는 flow가 topology validity에서, diffusion이 fine geometry와 xTB relaxation energy에서 더 좋은 trade-off를 보였다. 따라서 sampler step 수를 줄이거나 flow로 바꾸는 것만으로 좌표가 개선된다고 가정하면 안 된다.

### 3.4 UAE-3D / UDM-3D: unified latent의 장점과 현재 데이터 한계

UAE-3D는 atom, bond, coordinate를 atom당 unified latent로 압축하고 일반 DiT가 latent를 diffusion한다. near-lossless codec을 먼저 확보해 modality mismatch를 latent 단계에서 줄이는 접근이다.

현재 프로젝트에서 장점은 분명하다. 하지만 QM9 130k와 GEOM-Drugs 수십만 규모에서 학습한 결과를 1,581개 HP1KG에 바로 옮기면 codec이 train molecule을 암기할 가능성이 높다. deterministic codec의 oracle identity 100%와, held-out molecule에서 learned latent reconstruction 100%는 다른 주장이다.

따라서 이 방향은 외부 대규모 3D 데이터로 codec을 사전학습할 수 있을 때 P2 후보이며, 현재 바로 갈아탈 P0 수정은 아니다.

### 3.5 DiTMC: graph를 고정한 conformer 생성기

DiTMC는 새 graph를 생성하지 않는다. atom/bond graph를 조건으로 coordinates/conformer ensemble만 생성한다. 모든 atom pair의 graph distance를 condition token으로 쓰는 것이 효과적이었다.

현재 joint model이 계속 topology와 geometry를 동시에 못 맞추면 다음의 안전한 2-stage baseline이 가능하다.

1. EST-conditioned Kekulé molecular graph generator
2. graph + EST conditioned conformer/coordinate generator

장점은 각 단계의 실패 원인이 분리된다는 것이다. 단점은 excited-state condition과 geometry의 상호작용이 두 단계 사이에서 약해질 수 있다는 점이다. EST를 graph stage와 coordinate stage 모두에 넣고, 최종 independent quantum calculation으로 재평가해야 한다.

### 3.6 SLDM

SLDM의 핵심은 거의 직선인 diffusion path와 적은 NFE다. 그러나 주 pipeline은 coordinates/atom type을 생성하고 bond를 거리 기반으로 복원한다. 빠른 sampling 연구로는 유용하지만, 현재 병목인 bond/ring/aromatic joint consistency를 직접 해결하지 않는다. 우선순위는 낮다.

### 3.7 CHEFNMR

CHEFNMR는 raw 1H/13C NMR spectrum과 정답 molecular formula가 주어진 상태에서 atom coordinates를 생성하고, RDKit으로 bond를 복원한다. 보고된 높은 성능은 exact formula/atom multiset이라는 강한 조건과 top-k sampling을 포함한다.

따라서 CHEFNMR 결과를 “scalar EST만으로 de novo graph와 coordinates를 모두 만들 수 있다”는 근거로 사용할 수 없다. 다만 향후 S1/T1 전체 spectrum을 조건으로 쓸 경우, 별도의 spectrum encoder, condition dropout, global coordinate loss와 local pair-distance loss를 같이 쓰는 설계는 참고할 수 있다.

## 4. topology·guidance 관련 논문이 주는 범위

### HOG-Diff

Cell-complex filtering으로 higher-order skeleton을 먼저 만들고 full adjacency를 복원하는 coarse-to-fine 2D graph 모델이다. “ring 정보를 사용한다”는 방향은 맞지만, 별도의 ring template과 atom/bond head를 독립 예측해 마지막에 우연히 맞추는 구조와는 다르다. 과거 계층형 모델의 실패를 피하려면 skeleton이 adjacency를 결정론적으로 제한하거나 bridge process의 중간 endpoint로 들어가야 한다.

현재 P0 수정으로 cell hierarchy를 다시 넣는 것은 권장하지 않는다. 먼저 joint Kekulé edge flow와 self-correction baseline이 정상 동작하는지 확인한 뒤, ring-size 분포가 여전히 무너지면 P2로 검토하는 것이 맞다.

### TreeDiff

Inference-time MCTS와 verifier로 reverse trajectory를 탐색한다. Base model이 어느 정도 valid graph를 만들 때 property reward를 더 잘 만족시키는 방법이지, base validity가 매우 낮은 모델을 화학적으로 고치는 방법은 아니다. verifier bias와 큰 inference cost도 있다.

### DemoDiff

대규모 discrete graph transformer가 demonstration molecule-score pair로 과제를 문맥 학습한다. 0.7B급 모델과 대규모 pretraining 전제가 있어 1,581개 데이터의 현재 실험에 직접 적용하기 어렵다.

### Mol-StrucTok

3D local geometry를 spherical-coordinate token으로 양자화해 autoregressive language model로 생성한다. sampling 속도와 discrete interface는 장점이지만, tokenizer reconstruction과 sequence ordering이라는 새 병목이 생긴다. 현재 joint flow의 원인을 분리하기 전 대체할 이유는 약하다.

### BADGER, TreeDiff, VIDD 및 reward guidance

이 계열은 유효한 base distribution이 먼저 필요하다. 현재 단계에서 guidance 종류나 scale을 늘리면 invalid topology를 target predictor가 선호하는 방향으로 더 강하게 밀 가능성이 있다. base chemistry를 고친 뒤 condition fidelity 미세 조정 단계에서 써야 한다.

## 5. 데이터 제한과 현재 실험 해석

### 5.1 규모 차이

| 모델/논문 | 주요 학습 규모 | 수소 | topology/geometry |
|---|---:|---|---|
| 현재 HP1KG | train 1,264, 공식 geometry 1개/분자 | attached-H count | joint CTMC/flow |
| PropMolFlow | QM9 약 100k | explicit H | joint CTMC/flow |
| FlowMol3 | 243,480 molecules, 5.74M conformers | explicit H | joint CTMC/flow |
| Megalodon | GEOM-Drugs, 낮은 에너지 conformer 5개/분자 | explicit H | joint diffusion/flow |
| UAE-3D | QM9/GEOM-Drugs 대규모 | dataset 설정 | learned unified latent |

현재 batch 1024는 GPU 활용률에는 유리하지만 데이터 다양성을 늘리지 않는다. 한 epoch이 2 optimizer step이므로 epoch 수만 보고 충분히 학습했다고 판단할 수도, 더 돌리면 좋아진다고 판단할 수도 없다. checkpoint는 optimizer step, held-out loss, 실제 generation metric으로 비교해야 한다.

### 5.2 공식 S0 좌표의 의미

공식 최적화 좌표를 사용한 것은 이전의 RDKit ETKDG 임의 conformer보다 명확히 개선이다. S1/T1가 S0 geometry에서 계산된 vertical excitation이라면 입력 geometry와 condition의 물리적 정의도 일치한다.

그러나 이 모델이 생성하는 것은 excited-state optimized geometry가 아니다. EST conditioning은 S0 geometry의 전자 여기 특성을 제어하는 것이다. 결과를 “S1 또는 T1 최적화 구조 생성”으로 해석하면 안 된다.

### 5.3 explicit H와 H-count

현재 H-count는 valence identity를 보존하면서 O(N²) 비용을 줄이는 실용적 선택이다. 다만 다음은 학습하지 못한다.

- X–H bond length와 angle
- proton transfer/tautomer geometry
- all-atom steric clash
- 생성 좌표 그대로의 all-atom xTB energy

공식 HDF5에는 explicit-H coordinates가 있으므로 별도 all-atom ablation은 가능하다. 하지만 node 수가 커져 complete graph memory가 크게 증가하므로 현재 batch 1024와 병행할 수는 없다. H-count 모델과 explicit-H 모델을 공통 sample/metric으로 비교해야 한다.

## 6. 다음 branch의 논리적 우선순위

현재 run의 checkpoint를 먼저 공통 evaluator로 측정한 뒤, 새 branch에서 다음 순서로 수정하는 것이 가장 타당하다.

### P0-A. 현재 checkpoint 평가 고정

동일한 EST target·atom-count 조건·seed로 다음을 모두 저장한다.

1. raw RDKit validity / connected validity / uniqueness / novelty
2. aromatic molecule 비율과 aromatic ring 수
3. ring-size 분포, 3·4원 비율, 7+ 및 9+ 비율, train 대비 JSD
4. FlowMol3식 OOD ring rate와 functional-group deviation
5. PoseBusters validity
6. 생성 좌표 그대로의 bond-length/angle/dihedral 분포
7. 생성 좌표에서 직접 시작한 xTB relaxation의 ΔE와 heavy/all-atom RMSD
8. 독립 XTB/STDA EST fidelity와 계산 실패율
9. target별 mean뿐 아니라 분포, hit rate, diversity, acceptance rate

RDKit ETKDG로 좌표를 다시 만든 뒤 xTB를 수행하면 graph 계산 가능성만 평가하고 coordinate generator의 품질은 평가하지 못한다. 두 pipeline은 반드시 분리 표기한다.

### P0-B. 다음 학습 branch

1. aromatic class 제거, deterministic Kekulé single/double target 사용
2. coordinate time과 categorical time 독립 샘플링
3. FlowMol3식 self-conditioning
4. fake atom state를 이용한 node-count correction
5. timestep 이후 일부 atom에만 적용하는 train-time geometry distortion
6. inference에서도 self-conditioning 사용
7. train-support ring/functional-group metric으로 distortion strength 선택

이 다섯 변경은 서로 상호작용하므로 한 번에 최종모델만 만들기보다 다음 ablation이 필요하다.

- B0: 현재 모델
- B1: Kekulé + independent times
- B2: B1 + self-conditioning
- B3: B2 + fake atoms
- B4: B3 + geometry distortion

각 모델은 같은 optimizer step 수와 3개 seed가 이상적이다. 데이터가 작아 단일 seed의 생성 성공률 차이는 우연일 수 있다.

### P1. 조건부 표현

Base validity가 확보된 뒤에만 다음을 비교한다.

- 현재 scalar MLP
- train min/max 또는 robust quantile 범위의 Gaussian expansion
- layer-wise FiLM/AdaLN
- CFG scale sweep

조건 성능은 random baseline뿐 아니라 atom-count-only, unconditional, shuffled-condition baseline과 비교해야 한다. 그래야 모델이 EST를 쓰는지, 단순히 DB prior를 재현하는지 분리할 수 있다.

### P1. 외부 사전학습

가장 큰 성능 상한 개선은 대규모 3D chemical pretraining이다.

1. GEOM-Drugs/QM9 등에서 explicit-H 또는 compatible heavy-atom backbone을 unconditional pretrain
2. HP1KG에서 atom/edge/coordinate denoising을 계속 학습
3. EST condition adapter 또는 embedding을 fine-tune
4. HP1KG에만 존재하는 희귀 원소는 충분한 외부 원소 데이터가 없으면 generation vocabulary에서 신뢰 구간을 별도로 표시

다만 GEOM과 HP1KG의 계산 수준·원소 분포·conformer 의미가 다르므로 coordinate head 전체를 무비판적으로 고정하면 안 된다.

### P2. 2-stage 또는 latent 대안

P0 branch에서도 joint generation이 실패할 때만 검토한다.

- graph generator + DiTMC식 coordinate generator
- 대규모 사전학습 UAE-3D codec + conditional latent diffusion
- HOG-Diff식 higher-order skeleton bridge

이들은 전면 재설계이므로 현재 원인의 가장 작은 검증 실험보다 먼저 할 이유가 없다.

## 7. 우선하지 말아야 할 변경

- 현재 checkpoint를 더 오래 학습하는 것만으로 ring/aromatic 문제를 해결한다고 기대하기
- aromatic/ring loss weight만 반복 조정하기
- hard projection으로 train process와 다른 상태를 매 reverse step 강제하기
- base validity가 낮은 상태에서 CFG/MG/AG scale만 키우기
- 1,581개만으로 더 큰 Transformer 또는 latent codec부터 학습하기
- RDKit valid만 checkpoint 선택 기준으로 사용하기
- ETKDG 재생성 좌표의 xTB 결과를 diffusion coordinate 품질로 보고하기

## 8. 전체 26편의 역할별 해석

직접 architecture 근거:

- PropMolFlow, FlowMol3, Megalodon, UAE-3D, SLDM, DiTMC

Topology/condition/guidance 아이디어:

- HOG-Diff, TreeDiff, DemoDiff, Mol-StrucTok, CHEFNMR, BADGER, VIDD

물리 검증 및 폐루프 교훈:

- DiffDock-Glide, guided superconductor discovery, FunDiff, photonic inverse design, alloy/amorphous-material inverse design

다른 domain의 구조 생성 교훈:

- CrystalDiT, protein surface diffusion bridge, Co-Diffuse, relational DB diffusion, PACT hybrid diffusion, materials reviews

CrystalDiT의 가장 관련성 높은 교훈은 데이터가 작은 과학 domain에서는 복잡한 multi-stream hierarchy가 단순한 unified architecture보다 쉽게 overfit한다는 점이다. 현재 HP1KG에서도 더 큰 hierarchy를 추가하기보다 joint representation을 유지하면서 학습 경로와 self-correction을 고치는 쪽이 먼저다.

## 9. 최종 판단

현재 run은 “공식 S0 좌표가 들어간 PropMolFlow-compatible heavy-atom/H-count baseline”으로 가치가 있다. 이 checkpoint를 폐기할 이유는 없으며 이후 모든 구조 변경의 control로 남겨야 한다.

다만 문헌 기준으로 현재 모델은 최신 FlowMol3/Megalodon의 핵심 안정화 장치를 빠뜨리고 있다.

- shared continuous/discrete time
- no self-conditioning
- no fake atom/count correction
- no train-time geometry distortion
- explicit aromatic edge class
- no GVP network dropout
- 1,264개라는 매우 작은 train set

따라서 다음 개선은 loss penalty를 더 얹는 방향보다 `Kekulé joint edge + independent modality times + self-correction`이 핵심이다. 그리고 논문 성능과의 가장 큰 차이는 architecture보다 데이터 규모이므로, 외부 3D pretraining 없이는 generated chemistry의 tail과 희귀 원소 일반화에 명확한 한계가 남는다.

## 10. 근거 파일

- 논문 PDF 폴더: `/home/jovyan/cpu-only-1-datavol-1/psg/generation/diffusion_ref`
- 기존 26편 통합 보고서: `diffusion_inverse_design_26_paper_report_ko.pdf`
- 추출 text/manifest: `analysis/diffusion_ref_review_20261015/`
- 현재 config: `MolGuidance_guidance_selector_e987/configs/local/guidance_est/hp1kg_est_cfg_hcount_official_xyz_bs1024_70pct.yaml`
- 데이터 manifest: `joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz/manifest.json`
- 변환 report: `molguidance_hp1kg_baseline/data_est_hcount_official_xyz/conversion_report.json`
- 학습 log: `runs/molguidance_hp1kg_official_xyz_cfg_bs1024_lr1p4e3_20261015/logs/train_expandable.log`
- checkpoint: `runs/molguidance_hp1kg_official_xyz_cfg_bs1024_lr1p4e3_20261015/None_c1ihkbur/checkpoints/`
