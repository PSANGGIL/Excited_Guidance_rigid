# Joint Kekulé Diffusion V8

V8은 HP1KG/QCDGE 조건부 분자 생성에서 반복된 비정상 고리, 방향족 붕괴, 원자가 위반과 좌표 불안정을 분리해 고치기 위한 모델이다. 원자 상태와 결합 상태를 공동 확산하고 reverse diffusion 뒤 하나의 전역 화학 제약 문제로 분자를 복원한다.

V8은 V7과 체크포인트 호환성이 없다. metadata schema는 13이며 데이터 manifest, 전이적으로 사용하는 코드 fingerprint 또는 Python/Torch/RDKit/SciPy/NumPy/NetworkX 버전이 다르면 재개와 생성을 중단한다.

## 1. 입력, 출력, diffusion

입력은 SMILES에서 RDKit graph를 만든 뒤 저장한 heavy-atom 순서의 tensor다. 좌표는 원본 양자화학 좌표가 아니라 SMILES에서 ETKDG로 만든 conformer이므로 보조 구조 정보이며 정답 양자기하로 해석하면 안 된다.

하나의 joint heavy-atom 노드 상태:

- element
- formal charge
- total valence
- aromatic role
- Cartesian coordinate

수소는 별도 노드나 독립 categorical state가 아니다. 최종 heavy-atom 결합차수와 joint node의 total valence 차이로 residual hydrogen을 결정한다. 현재 DB의 radical electron은 전부 0이므로 radical state는 schema 13 어휘에 넣지 않고, 생성 결과에 radical이 생기면 chemical-clean 평가에서 실패시킨다.

pair 상태는 edge 존재, bond order와 aromatic-system role을 하나의 joint class로 표현한다.

- no edge
- regular single, double, triple
- aromatic-system Kekulé single, double, triple

조건은 S1/T1 스칼라를 각각 RBF expansion한 값이다. train 평균/표준편차로 정규화하며 기본 범위는 -6 sigma에서 +6 sigma다. 범위 밖 condition은 명시적 override 없이는 거부한다.

좌표는 Gaussian DDPM, 원자와 결합 범주는 marginal categorical D3PM이다. 모델은 noisy state x_t, timestep t, S1/T1 condition을 받아 node x0 logits, joint-edge x0 logits와 coordinate noise를 공동 예측한다.

## 2. 감사에서 확인한 문제와 해결

### 2.1 canonical SMILES 재파싱에 의한 atom-index 파손

저장 tensor의 atom 순서와 canonical SMILES를 다시 parse한 atom 순서를 섞으면 ring/aromatic target이 다른 edge에 붙는다. train 1,264개 중 index adjacency 완전 일치는 264개, symbol 순서 일치는 340개뿐이었다. 기존 방식은 935개 분자에서 ring target이 실제 non-bond를 가리켰고, cyclic target 23,456개 중 7,025개와 aromatic target 17,333개 중 6,006개가 잘못 매핑됐다.

해결:

- index별 target은 저장 atom_types와 kekule_bonds에서만 계산한다.
- canonical SMILES는 분자 단위 통계와 identity 확인에만 쓴다.
- cyclic edge는 stored bond의 부분집합, aromatic edge는 cyclic edge의 부분집합이어야 하며 위반 데이터는 fail closed한다.
- chemistry support reference도 저장 tensor 순서로 graph를 decode해서 만든다.

### 2.2 spanning-tree closure target의 gauge 의존성

임의 spanning tree의 non-tree edge를 closure로 정하면 대칭 고리, fused/bridged graph에서 같은 분자가 다른 target을 가진다. 이전 oracle 반례 1,581개 중 1,518개에서 선택 tree가 원 graph에 없는 edge를 강제했고 false edge는 18,752개였다. fundamental-cycle size도 minimum ring과 달라 closure edge 5,435개 중 429개가 불일치했고 실제 5/6원 edge 234개가 9원 이상으로 표기됐다.

해결:

- 각 stored bond를 제거한 뒤 endpoint 사이 shortest replacement path를 구한다.
- path가 있으면 cyclic edge이며 minimum cycle size는 path length + 1이다.
- 최종 MILP에 특정 spanning tree를 강제하지 않는다.
- single-commodity flow로 선택 graph 전체 connectedness만 강제한다.
- ring size와 topology score는 solver 결과 graph에서 다시 계산한다.
- candidate graph를 연결하기 위한 maximum-spanning-tree 후보 추가는 허용하지만 solver가 그 edge를 선택하도록 고정하지 않는다.

### 2.3 ring histogram 단위 불일치와 SSSR 비결정성

과거 training target은 cyclic-edge count이고 evaluation은 RDKit ring count여서 loss와 checkpoint 지표의 단위가 달랐다.

RDKit SSSR ring list는 fused/bridged graph의 동등한 cycle basis와 atom renumbering에 따라 개수와 cycle 선택이 달라질 수 있었다. 따라서 SSSR 개별 ring을 학습 identity나 train-support key로 사용하면 같은 분자가 다른 target을 가질 수 있다.

해결:

- target/evaluation histogram은 graph minimum-cycle-basis의 size multiset이며 개수는 cycle rank와 일치한다.
- ring train-support는 개별 SSSR sequence 대신 endpoint node state, aromatic role, minimum-cycle-size class와 cyclic degree로 구성한 cyclic-bond signature를 쓴다.
- cyclic edge와 그 edge를 포함하는 minimum-cycle-size는 pair별 supervised target으로 학습한다.
- cyclic-edge mass/size로 근사한 global histogram은 fused/bridged graph의 exact cycle basis와 일치하지 않으므로 loss에서 사용하지 않는다. 전체 ring-size 분포는 decoded graph의 exact MCB와 train-only prior로 projection/checkpoint 단계에서 평가한다.
- 평가는 3/4원, 5/6원, 9원 이상과 ring-size JSD를 별도로 기록한다.

### 2.4 aromatic resonance phase 고정

특정 alternating single/double 배치를 정답으로 고정하면 동등한 Kekulé form이 서로 벌점을 받는다. 재-Kekulization 감사에서 aromatic system 4,743개 중 336개는 개별 double edge가 달랐지만 atom별 aromatic double incidence는 보존됐다.

해결:

- aromatic edge의 local single-vs-double CE로 특정 phase를 강제하지 않는다.
- aromatic bond가 single/double 허용 집합에 속하는지 학습한다.
- aromatic atom별 double incidence와 system total을 resonance-invariant loss로 사용한다.
- final graph는 RDKit sanitization과 aromatic perception을 통과해야 한다.
- proposed aromatic mask와 RDKit perceived aromatic bond mask가 정확히 같아야 solver 결과로 인정한다.
- non-single/double aromatic perimeter는 train support와 RDKit 판정이 있을 때만 허용한다.

### 2.5 edge existence와 bond type 독립 예측

독립 head는 no-edge인데 bond type이 있거나 edge인데 type이 없는 상태를 만든다.

해결:

- edge existence, bond order, aromatic role을 하나의 joint categorical edge state로 통합한다.
- reverse categorical posterior도 같은 joint state에서 계산한다.

### 2.6 clean-neighbor valence leakage

이전 valence loss는 clean target 이웃 mask만 사용해 false-positive edge를 원자가 합에서 숨겼다.

해결:

- 모든 pair의 predicted joint bond probability로 expected valence를 계산한다.
- formal charge, joint total valence, residual hydrogen과 bond order를 함께 사용한다.
- edge count, connectedness, aromatic degree와 global chemistry loss를 공동 적용한다.

### 2.7 topology prior가 condition을 무시

atom count만 조건으로 topology prior를 고르면 S1/T1이 요구하는 구조와 평균 topology가 충돌한다.

해결:

- train-only reference에 atom count와 normalized S1/T1을 함께 저장한다.
- P(topology | atom count, S1, T1)를 Gaussian kernel로 근사해 MILP 후보를 rerank한다.
- condition 이웃이 희소하면 atom-count conditional prior로 fallback한다.
- train-only reference와 held-out 평가는 완전히 분리한다.

### 2.8 generated coordinate가 final graph 선택에서 무시됨

과거 solver는 생성 Cartesian 좌표와 무관하게 graph를 선택했다.

해결:

- train-only bond type별 거리 평균/표준편차를 reference에 저장한다.
- solver objective에 generated distance의 robust z-score 비용을 넣는다.
- direct argmax graph도 bond zRMS와 steric clash 검사를 통과해야 한다.
- generated-coordinate geometry와 ETKDG 재구성 지표를 따로 기록한다.

### 2.9 padding/count/condition leakage

과거 padded relation이 real hidden을 바꾸거나 clean atom count가 사실상 oracle로 주어지는 경로가 있었다.

해결:

- 모든 node/pair attention과 loss에 node/pair padding mask를 적용한다.
- count는 별도 count head와 train count prior로 처리한다.
- sampler의 atom count가 train support 밖이면 기본 거부한다.
- condition dropout은 CFG용이며 일반 regularization dropout과 구분한다.
- residual/attention/MLP에는 일반 dropout을 별도로 둔다.

### 2.10 수치 안정성과 checkpoint 오선택

loss 감소만으로 chemical validity나 condition fidelity를 보장하지 않는다. 이전 모델에서는 validation divergence, 큰 gradient와 reverse-coordinate 폭주가 관찰됐다. TF32 사용도 strict FP32 요구와 달랐다.

해결:

- FP32가 기본이며 TF32는 명시적 flag로만 켠다.
- finite loss/gradient와 clipping 전 gradient norm을 기록한다.
- EMA model을 validation, rollout과 기본 sampling에 사용한다.
- checkpoint 우선순위는 raw joint validity Wilson lower bound, projection validity Wilson lower bound, chemical-clean Wilson lower bound, S1/T1 surrogate MAE, generated-coordinate zRMS, DB-supported clean Wilson lower bound, ring JSD 순이다.
- metadata/checkpoint를 atomic write하고 source hash와 DB manifest를 검사한다.
- non-empty run/output directory는 덮어쓰지 않는다.

### 2.11 reverse-coordinate kernel과 projection 후보 열거

기존 공용 sampler는 reverse 중간 좌표를 반경 25로 hard clip했다. forward Gaussian kernel에는 이 projection이 없으므로 학습과 생성 분포가 달라지는 문제였다. 또한 MILP no-good cut이 이미 선택된 binary variable만 사용해, 첫 tree 해의 edge superset인 ring 해를 다음 후보로 열거하지 못했다.

해결:

- Cartesian reverse는 표준 epsilon-parameterized DDPM posterior mean/variance를 그대로 사용하고 매 단계 center-of-mass만 제거한다.
- 좌표 한계를 넘으면 clip하지 않고 numerical failure로 중단한다.
- production schedule의 terminal alpha-bar가 1e-4보다 크면 prior에서 시작한다고 볼 수 없어 sampling을 거부한다.
- MILP no-good cut은 선택/비선택 binary를 모두 포함해 정확히 한 assignment만 제외한다.
- topology prior와 learned ring head가 켜져 있으면 여러 feasible candidate를 모두 점수화한 뒤 rerank한다.
- degree-3 cyclic-core를 bridgehead로 간주하던 soft surrogate는 정답 adjacency에서도 큰 편향이 있어 제거한다. 실제 RDKit bridgehead count를 graph topology에서 계산해 train-only topology reference와 최종 candidate rerank에 사용한다.
- matrix-tree log partition은 dense soft graph가 너무 쉽게 만족하고 spanning-tree 수가 많은 graph를 보상하므로 학습 loss에서 사용하지 않는다. predicted/clean adjacency의 normalized Laplacian algebraic connectivity(lambda_2)를 회귀해 disconnected graph와 과도하게 dense한 graph를 모두 벌점화하며 특정 spanning-tree gauge는 두지 않는다.

## 3. final joint projection

projection은 joint atom state, joint edge, valence, residual hydrogen, formal charge와 connectivity를 한 전역 문제에서 함께 결정한다.

제약:

1. 선택 edge는 선택 atom 사이에만 존재한다.
2. bond order와 formal charge를 포함한 valence 상한을 만족한다.
3. single-commodity flow로 모든 선택 atom이 하나의 connected component를 이룬다.
4. aromatic edge endpoint는 aromatic atom role이어야 한다.
5. aromatic atom degree와 Kekulé double incidence가 허용 범위여야 한다.
6. RDKit sanitization과 전역 aromatic perception을 최종 oracle로 사용한다.

특정 spanning tree, biconnected-component all-or-none aromatic 강제와 intermediate hard projection은 사용하지 않는다. forward corruption kernel에 없는 hard constraint를 reverse 중간 단계에 넣어 분포를 바꾸는 일을 피하기 위해서다.

MILP 후보는 다음 항으로 rerank한다.

- atom/edge model log probability
- condition-aware train topology likelihood
- learned cyclic-edge/minimum-cycle-size agreement
- train bond-length geometry likelihood
- local environment support

## 4. 학습 loss

총 loss 계열:

- joint heavy-atom state(element, charge, valence, aromatic role) categorical denoising
- joint-edge categorical denoising
- coordinate noise prediction
- atom count
- pairwise cyclic-edge/minimum-cycle-size (exact MCB histogram은 projection/evaluation 전용)
- resonance-invariant aromatic system
- valence, charge balance, edge count, connectedness
- cyclic-edge and aromatic-edge consistency
- cumulene safety surrogate와 projection용 exact bridgehead topology reference
- coordinate bond length and clash surrogate
- condition shuffle gain and held-out surrogate fidelity

희소 positive target은 class-balanced loss를 쓴다. molecule-level loss는 batch 및 유효 atom/pair 수로 정규화해 큰 분자가 loss를 독점하지 않게 한다.

## 5. 분리해서 기록할 평가 지표

- raw joint validity: projection 전 argmax graph
- projected RDKit validity
- connectedness
- local atom/bond support
- DB-supported chemical-clean validity
- 3/4-member ring molecule rate
- 5/6-member ring fraction
- 9+ member ring molecule rate
- ring-size distribution JSD
- ETKDG success와 structural strain
- generated-coordinate bond zRMS와 clashes
- S1/T1 held-out surrogate MAE와 condition-shuffle gain
- XTB/STDA fidelity: 비싼 독립 평가이며 매 epoch checkpoint 선택에는 사용하지 않음

raw와 projected metric을 분리해야 projection이 약한 model을 가리는 현상을 찾을 수 있다. rollout 표본이 적을 때 우연한 고득점을 막기 위해 Wilson lower bound를 사용하며 기본 rollout은 최소 32개다.

## 6. 장기학습 전 검증 게이트

다음을 모두 통과한 뒤에만 장기학습을 시작한다.

1. 전체 DB direct codec RDKit identity 100 percent
2. cyclic target subset stored bond 및 aromatic subset cyclic 100 percent
3. padding invariance
4. atom permutation equivariance
5. coordinate translation/rotation equivariance
6. forward/backward finite loss와 finite gradient
7. resonance-phase invariance
8. flow-MILP oracle reconstruction
9. conditioned topology reference serialization round-trip
10. checkpoint/metadata/source/DB manifest hash 일치
11. sampler out-of-support atom count와 condition fail closed
12. FP32 GPU smoke test와 peak memory 확인
13. 전체 DB atom renumbering에 대한 ring topology/support 불변성
14. D3PM posterior brute-force Bayes 대조와 Cartesian t=0 oracle 복원
15. 정확한 MILP no-good cut이 tree의 ring superset 후보를 허용하는지 확인
16. metadata 내 runtime dependency version lock 확인

검증:

    /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python validate_joint_kekule_v8.py

학습 예시:

    CUDA_VISIBLE_DEVICES=0 /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python train_joint_kekule_v8.py --run-dir runs/joint_kekule_v8 --epochs 10000 --batch-size 192 --precision fp32

생성 예시:

    CUDA_VISIBLE_DEVICES=0 /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python sample_joint_kekule_v8.py --checkpoint runs/joint_kekule_v8/checkpoints/best_quality.pt --metadata runs/joint_kekule_v8/metadata.json --output runs/generation_v8 --num 1000 --weights ema

## 7. 현재 한계와 다음 개선

- 입력 좌표는 SMILES 유래 conformer다. 양자기하를 목표로 하려면 원본 계산 좌표와 atom mapping을 DB에 저장해야 한다.
- cache row에는 개별 coordinate 생성 method가 보존되지 않아 현재 tensor만으로 ETKDG/MMFF, UFF, 2D fallback을 다시 구분할 수 없다. 이후 DB schema에서는 method와 optimization status를 각 row에 저장해야 한다.
- train 1,264개에 약 15M parameter는 과적합 위험이 크다. 데이터 확대, hidden/layer 축소, 일반 dropout, weight decay와 early stopping을 비교해야 한다.
- 과거 held-out condition surrogate는 S1 MAE 약 0.599 eV, T1 MAE 약 0.533 eV, R2 약 0.285/0.382로 약했다. 실제 fidelity는 XTB/STDA 또는 더 정확한 surrogate로 검증해야 한다.
- MILP는 양자화학 solver가 아니다. unusual valence, charge와 exotic aromaticity는 train support와 RDKit convention 밖에서 실패할 수 있다.
- fused/bridged graph의 global cycle histogram은 differentiable edge-mass 근사로 학습하지 않는다. pairwise minimum-cycle target을 학습하고 checkpoint에서는 decoded graph의 exact MCB 통계를 사용한다.
- stereochemistry는 현재 DB scope에서 제외했다. chiral/E-Z 데이터가 추가되면 새 schema가 필요하다.
- condition kernel prior는 희소 영역에서 평균 prior로 fallback한다. 데이터가 충분해지면 learned conditional topology model로 교체하는 편이 낫다.

## 8. 참고 문헌과 구현 근거

- Austin et al., Structured Denoising Diffusion Models in Discrete State-Spaces: https://arxiv.org/abs/2107.03006
- Vignac et al., DiGress: https://arxiv.org/abs/2209.14734
- Vignac et al., MiDi: https://arxiv.org/abs/2302.09048
- Hoogeboom et al., Equivariant Diffusion for Molecule Generation in 3D: https://arxiv.org/abs/2203.17003
- Peng et al., MolDiff: https://arxiv.org/abs/2305.07508
- Ho and Salimans, Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
- RDKit sanitization and aromaticity: https://www.rdkit.org/docs/RDKit_Book.html

핵심 원칙은 discrete atom/bond joint denoising, continuous coordinate denoising, SE(3) equivariance, CFG를 위한 condition dropout, 그리고 최종 전역 chemistry consistency를 분리하지 않는 것이다.

## 9. 호환성과 보존

V7 이하 checkpoint는 V8에서 resume/sampling할 수 없다. V8 내부에서도 metadata schema, source fingerprint, atom vocabulary, marginal, DB manifest와 RBF configuration 중 하나가 다르면 중단한다. 이전 결과와 checkpoint는 삭제하지 않고 새 run directory를 사용한다.

## 10. 2026-09-25 최종 검증 결과

추가 full-flow 감사에서 최초에는 1,581개 중 18개가 실패했다. 모두 protonated aromatic [nH+]를 neutral pyrrolic [nH]와 동일하게 취급해 aromatic multiple bond를 금지한 과도한 MILP 특례가 원인이었다. 이 특례를 제거하고 state의 formal charge, valence와 hydrogen 범위로 구분한 뒤 RDKit 전역 판정을 유지했다. 수정 후 flow-MILP exact oracle은 1,581/1,581을 통과했다.

무작위 초기화 loss는 valence 범위 밖 오차를 제곱해 단일 항이 총 loss를 독점했다. 범위 안 0, 경계 근처 quadratic, 큰 위반 linear인 smooth-L1 penalty로 바꿨다. 동일 smoke batch에서 총 loss는 약 577에서 47로, valence 항은 약 546에서 15.9로 감소했다. 검증은 총 초기 loss 100 미만과 단일 항 점유율 75 percent 미만을 강제한다.

독립 검증 결과:

- 전체 direct codec RDKit identity: 1,581/1,581
- 전체 flow-MILP exact oracle: 1,581/1,581
- 전체 atom-renumbering ring topology/support: 3,162/3,162
- 전체 alternative Kekulé phase incidence: 3,162/3,162, 실제 edge assignment 변경 496건
- D3PM posterior brute-force 최대 오차: 3.58e-7
- Cartesian t=0 oracle 복원 최대 오차: 1.55e-6
- production terminal alpha-bar: 2.93e-6
- exact no-good/rerank 반례: tree에서 ring edge superset 후보로 정상 전환
- 자동 flow 회귀 표본: 42/42, [nH+] 18개 전부 포함
- padding error: 0
- permutation error: 2.98e-7
- coordinate rotation error: 3.17e-8
- resonance phase loss difference: 0
- strict FP32 B200 optimizer smoke: 2 steps finite
- batch 192 peak allocated: 138.66 GiB / 178.35 GiB, 77.7 percent
- peak reserved: 146.61 GiB, 82.2 percent
- TF32: disabled

상세 결과:

- validation/v8_validation_report.json
- validation/v8_full_flow_oracle_report.json
- validation/v8_gpu_smoke_report.json
