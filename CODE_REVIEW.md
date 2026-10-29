# Excited Guidance Rigid 모델 코드리뷰

## 1. 리뷰 범위와 결론

이 문서는 `feature/chemical-validity-curriculum` 브랜치의 실제 데이터베이스, 실데이터 학습 경로, 조건부 생성, RDKit 구조 복원, validity scorer, xTB/sTDA 평가를 검토한 결과다. `src/polyring_diffusion/train.py`로 실행하는 합성 smoke 경로보다 현재 운용 중인 `train_rigid.py`와 `sample_rigid.py`를 중심으로 평가했다.

현재 구현은 원자별 좌표를 직접 생성하는 모델이 아니라 다음 상태를 생성하는 **고리 단위 계층형 diffusion 모델**이다.

```text
원본 SMILES + 3D 원자 좌표 + f_osc
  -> SSSR 고리 추출 및 canonical ring template 생성
  -> 고리 template / center / normal / connection / port 텐서
  -> 조건부 dense equivariant denoiser
  -> reverse continuous + absorbing-mask categorical diffusion
  -> 여러 connection tree 후보 생성
  -> RDKit sanitize + validity reward로 후보 선택
  -> 원자 그래프 복원 및 ETKDG/MMFF 또는 UFF 3D 생성
  -> xTB geometry optimization
  -> sTDA state 1-10 및 oscillator strength 계산
```

데이터 무결성과 구조 유효성 개선 경로는 비교적 잘 구축되어 있다. 반면 `f_osc` condition의 실제 기여를 강제하거나 측정하는 목적함수가 없고, CFG unconditional 경로가 학습 시 condition-drop 경로와 다르다. 따라서 현재 모델은 “condition이 입력되는 모델”이지만 “목표 oscillator strength가 강하게 제어되는 모델”이라고 보기는 어렵다.

가장 먼저 수정해야 할 항목은 다음 네 가지다.

1. validation metric을 샘플 수로 가중해 batch size에 독립적으로 만든다.
2. resume 시 기존 best loss와 patience를 복구한다.
3. CFG unconditional 호출을 실제 condition-drop 경로와 동일하게 만든다.
4. condition contribution과 목표-실측 property alignment를 학습·평가 지표에 추가한다.

## 2. 실제 데이터 입력

### 2.1 원천 데이터

`prepare_rigid_dataset.py`는 STGG-AL의 Kekulé 전처리 결과를 입력으로 받는다.

- 원천 경로: `/home/jovyan/cpu-only-1-datavol-1/psg/generation/STGG-AL/data/data_kekule`
- 전체 분자: 100,000개
- split: train 80,000 / validation 10,000 / test 10,000
- 물성 원본: `f_osc`
- 학습 물성: `log10(f_osc + 1e-6)`
- 좌표 원천: RDKit ETKDGv3
- 원자 종류: H, B, C, N, O, F, S, Cl, Br
- 결합 표현: Kekulé

`verification_report.json` 기준 CSV provenance는 100,000개 모두 확인됐으며 missing, SMILES mismatch, property mismatch가 0이다.

### 2.2 데이터 규모

| Split | 분자 | 원자 | 고리 | 고리 연결 |
|---|---:|---:|---:|---:|
| train | 80,000 | 7,176,576 | 1,020,485 | 1,240,993 |
| validation | 10,000 | 897,490 | 127,554 | 155,019 |
| test | 10,000 | 897,659 | 127,815 | 155,725 |

실데이터에서 발견된 canonical ring template는 48개이고, 한 분자의 최대 고리 수는 19개다. train property 통계는 mean `-0.8302586`, standard deviation `0.9663340`이다.

### 2.3 고리 추출과 canonicalization

전처리는 RDKit `BondRings()`를 순회하고 각 simple cycle을 양방향·모든 시작점으로 회전해 가장 작은 signature를 선택한다. 각 port signature는 다음 네 필드다.

```text
(atomic_number, formal_charge, is_aromatic, outgoing_bond_label)
```

이 방식은 시작 원자와 순회 방향에 독립적인 template를 만든다는 장점이 있다. SHA-256 앞 8 byte를 signed 63-bit 범위로 제한한 hash를 template key로 사용하며, 전처리 중 hash collision도 검사한다.

주의할 점은 RDKit SSSR 계열 ring decomposition 자체가 유일한 화학적 고리 분해는 아니라는 것이다. 현재 데이터에는 `COMPLEX` 연결이 0개지만, 다른 데이터베이스로 확장할 때 bridged/overlapping system의 표현 안정성을 다시 검증해야 한다.

### 2.4 기하 표현

각 고리는 다음 값을 저장한다.

- `center`: 고리 원자 좌표 평균, shape `[R, 3]`
- `normal`: 중심화 좌표의 SVD 최소 singular vector, shape `[R, 3]`
- `radius`: 고리 원자의 중심까지 평균 거리, shape `[R]`
- `ring_template_hashes`: canonical template 식별자
- `ring_atom_indices`: 원래 원자 그래프와의 membership

normal의 부호는 절댓값이 가장 큰 축 성분이 양수가 되도록 고정한다. 부호 ambiguity를 결정적으로 제거하는 방식이지만, 물리적으로 normal `n`과 `-n`이 같은 평면을 의미한다는 대칭을 loss가 명시적으로 처리하지는 않는다.

### 2.5 연결 표현

연결 레코드는 `[left_ring, right_ring, kind, left_port, right_port, reverse]`의 6개 정수다.

- `NONE=0`
- `SINGLE=1`: 서로 다른 두 고리 원자 사이 단일 결합
- `FUSED=2`: 두 인접 원자를 공유하는 fused edge
- `SPIRO=3`: 원자 하나를 공유
- `COMPLEX=4`: 현재 loader에서 학습 대상에서 제외

`reverse` orientation은 데이터에 저장되지만 현재 실데이터 모델의 loss와 출력 head에는 사용되지 않는다.

### 2.6 샤드 스키마

샤드는 분자 2,000개 단위 `.pt` 파일이다. 첫 train shard의 실제 주요 shape는 다음과 같다.

| Key | Shape | Dtype | 의미 |
|---|---:|---|---|
| `properties` | `[2000]` | float32 | 변환된 property |
| `raw_properties` | `[2000]` | float32 | 원본 f_osc |
| `atom_positions` | `[179147,3]` | float32 | 원자 좌표 |
| `atom_types` | `[179147,9]` | bool | 원자 one-hot |
| `atom_charges` | `[179147]` | int32 | formal charge |
| `bond_indices` | `[202640,2]` | int32 | 원자 결합 index |
| `ring_template_hashes` | `[25493]` | int64 | 고리 template hash |
| `ring_centers` | `[25493,3]` | float32 | 고리 중심 |
| `ring_normals` | `[25493,3]` | float32 | 고리 평면 normal |
| `ring_radii` | `[25493]` | float32 | 고리 반경 |
| `connections` | `[30919,6]` | int32 | 고리 연결 |

offset 배열로 분자별 variable-length 원자·고리·연결을 복원한다. 임시 파일 저장 후 `os.replace`를 사용하므로 샤드 생성의 원자성도 확보돼 있다.

## 3. 학습 입력 파이프라인

`RigidRingDataset`은 모든 샤드를 초기화 시 CPU 메모리에 로드하고, template hash를 연속 ID로 치환한다. 샤드 자체는 atom graph를 보존하지만 학습 loader가 실제 반환하는 것은 다음 고리 수준 데이터다.

- `ring_types [R]`
- `centers [R,3]`
- `normals [R,3]`
- `connection_types [R,R]`
- `ports [R,R]`
- `orientation [R,R]`
- standardized property `[1]`
- raw property `[1]`
- source row

collate 단계는 batch의 최대 고리 수까지 padding하고 `node_mask`를 만든다. center는 분자별 평균을 빼 translation invariant하게 만든다.

중요한 범위 제한이 있다. 데이터베이스는 비고리 치환기를 포함한 atom graph를 저장하지만, 현재 denoiser 입력에는 atom graph나 비고리 치환기가 들어가지 않는다. 따라서 README의 “데이터베이스가 비고리 치환기에 대해 lossless”라는 설명은 저장 형식에는 맞지만 **현재 생성 모델의 표현력에는 해당하지 않는다**. 생성 결과는 ring template들의 조립체이며 원본 비고리 치환기를 별도로 생성하지 않는다.

### 3.1 Property band sampler

현재 설정은 raw `f_osc`가 `[1.0, 2.0)`인 train sample을 epoch당 60%가 되도록 replacement sampling하고 나머지 영역을 40% 유지한다.

장점은 3.0 tail에 과도하게 집중하지 않고 1.x 영역의 노출을 늘리는 것이다. 단점은 실제 data likelihood를 왜곡하며, sample별 importance weight 보정이 없기 때문에 학습된 분포가 원래 데이터 분포와 달라진다는 것이다. 이것은 의도된 fine-tuning bias로 문서화해야 한다.

## 4. 모델 구조

### 4.1 전체 설정

현재 실학습 설정은 다음과 같다.

- hidden dimension: 192
- equivariant layer: 6
- time embedding: 64
- 최대 port: 8
- condition dimension: 1
- condition dropout: 0.1
- diffusion step: 200
- 파라미터 수: 약 1.93M

### 4.2 입력 embedding

각 ring node의 초기 hidden은 다음의 concat projection이다.

```text
ring token embedding (192)
+ sinusoidal time embedding (64)
+ template feature (4)
-> MLP -> hidden 192
```

실데이터 template feature는 다음 네 값이다.

1. ring size / 8
2. train split 평균 ring radius / 3
3. hetero atom fraction
4. aromatic atom fraction

condition은 scalar standardized log oscillator strength를 2-layer MLP로 192차원에 투영한 뒤 모든 ring node hidden에 동일하게 더한다.

### 4.3 Dense equivariant layer

모든 유효 ring pair에 대해 dense message를 계산한다. edge feature는 다음을 포함한다.

- left hidden
- right hidden
- noisy connection embedding
- radius-normalized squared distance
- 두 normal의 dot product

message를 degree-normalized mean aggregation한 뒤 node hidden을 residual update한다. center update는 상대 좌표 방향에 scalar weight를 곱해 구성하므로 rotation equivariant하다. normal update도 상대 normal 및 self normal의 scalar-weighted 합으로 구성된다.

복잡도는 ring 수 `R`에 대해 시간·메모리 모두 `O(B R² H)`다. 현재 최대 R=19라 dense 구현이 현실적이지만, ring 수를 크게 확장하면 가장 먼저 병목이 된다.

### 4.4 출력 head

- `center_noise [B,R,3]`
- `normal_noise [B,R,3]`
- `ring_logits [B,R,48]`
- `connection_logits [B,R,R,4]`
- `port_logits [B,R,R,8]`

connection logits는 transpose 평균으로 대칭화한다. port logits는 방향성이 있으므로 대칭화하지 않는다.

모델 파일에 `element_record`가 두 번 import되는 사소한 중복이 있다.

## 5. Diffusion과 학습 목적함수

### 5.1 Noise process

center와 normal에는 linear beta schedule의 Gaussian forward process를 적용한다. ring token과 connection token에는 같은 `alpha_bar` 확률을 이용한 absorbing-mask corruption을 적용한다.

continuous reverse는 DDPM `p_step`을 사용한다. categorical reverse는 엄밀한 posterior sampling이라기보다 각 step에서 `1/(step+1)` 확률로 mask를 현재 logits sample로 공개하는 heuristic이다. 즉 continuous 부분은 DDPM에 가깝지만 categorical 부분은 별도의 reveal schedule을 이용하는 혼합 구현이다.

### 5.2 현재 loss

```text
L = 1.0 L_center
  + 0.25 L_normal
  + 1.0 L_ring
  + 1.0 L_connection
  + 0.5 L_port
  + 0.5 L_valence
  + 0.5 L_kekule
  + 0.3 L_port_validity
```

- center/normal: 실제 Gaussian noise와 예측 noise의 MSE
- ring: clean ring template cross entropy
- connection: upper-triangle clean connection type cross entropy
- port: 실제 material connection에 대한 directional port cross entropy
- valence: soft connection/port 확률의 expected valence overflow
- Kekulé: fused edge 양쪽의 원자·결합 class compatibility
- port validity: 존재하지 않는 port 확률 질량 + port reuse

`NONE` class weight가 0.05이고 SINGLE/FUSED/SPIRO는 1.0이라 dense pair에서 NONE dominance를 줄인다.

### 5.3 Condition 관련 핵심 한계

현재 condition은 입력일 뿐 전용 loss가 없다.

- condition reconstruction loss 없음
- 생성 graph의 property prediction loss 없음
- condition을 shuffle했을 때의 contrastive loss 없음
- conditional/unconditional 차이 최소 크기 없음
- mutual information 또는 sensitivity regularization 없음
- 목표 `f_osc`와 실제 sTDA `f_osc` alignment loss 없음

따라서 모델이 condition을 일부 또는 대부분 무시해도 diffusion loss를 낮출 수 있다. 현재 loss 감소만으로 property controllability 향상을 결론 내릴 수 없다.

## 6. CFG 구현

학습 시 10% sample에 대해 condition projection 결과 전체를 0으로 만들어 unconditional branch를 학습한다.

생성은 다음 식을 사용한다.

```text
prediction = uncond + guidance_scale * (cond - uncond)
```

그러나 현재 `sample_rigid.py`의 unconditional 호출은 `condition=zeros`를 전달한다. 이때 model에는 drop mask가 전달되지 않으므로 `condition_projection(0)`의 bias가 살아 있다. 또한 standardized condition 0은 train property mean에 해당하는 정상적인 condition 값이기도 하다.

즉 현재 생성의 `uncond`는 학습에서 사용한 “projection을 완전히 제거한 branch”와 동일하지 않다. 올바른 구현은 unconditional 호출에 all-true `condition_drop_mask`를 전달하거나 명시적 null-condition embedding을 사용해야 한다. 이 차이는 guidance contribution과 목표 반응성을 약화하거나 왜곡할 수 있다.

현재 condition contribution norm, head별 cond-uncond delta, guidance-scale response는 기록되지 않는다.

## 7. Chemical constraint loss

chemical loss는 template signature에서 valid port, 예상 valence capacity, fused edge class, vertex incidence table을 미리 만든다. forward에서는 softmax 확률로 expected invalidity를 미분 가능하게 계산한다.

장점:

- RDKit을 GPU 학습 inner loop에 넣지 않는다.
- invalid port와 중복 port 사용을 직접 줄인다.
- fused atom/Kekulé compatibility를 학습 신호로 준다.
- logits까지 gradient가 정상 전달되는 테스트가 있다.

제한:

- 완전한 sanitization이나 전역 방향족 전자 배치를 보장하지 않는다.
- valence table이 제한된 원소 및 단순 formal-charge 규칙에 의존한다.
- ring strain, steric clash, 3D feasibility는 포함하지 않는다.
- soft expected penalty와 discrete decode validity 사이에 차이가 있다.

따라서 이 loss는 유효성 prior이지 exact constraint solver는 아니다.

## 8. 학습 루프와 체크포인트

학습은 AdamW, cosine annealing, BF16 autocast, gradient clipping을 사용한다. checkpoint에는 model, optimizer, scheduler, epoch, step, config, property normalization, template features, manifest, metrics가 포함된다. `.tmp`에 쓴 뒤 rename하므로 단일 writer 환경에서 중단 내성이 좋다.

### 8.1 중대한 resume best 문제

resume로 model/optimizer/scheduler/epoch/step은 복구하지만 `best`는 항상 `math.inf`, `patience`는 항상 0에서 시작한다. 그 결과 resume 후 첫 epoch가 과거 best보다 나빠도 `best.pt`를 덮어쓴다.

권장 수정:

- checkpoint에 `best_val_loss`와 `patience`를 저장한다.
- resume 시 두 값을 복구한다.
- 기존 `best.pt`의 metric과도 교차 확인한다.

### 8.2 validation 평균 편향

각 batch의 이미 평균된 loss를 더한 뒤 batch 수로 나눈다. 마지막 작은 batch도 full batch와 동일한 가중치를 가진다.

validation 10,000개에서 batch 17,408은 1 batch지만 batch 8,704는 8,704개 batch와 1,296개 batch를 동일 비중으로 평균한다. 실제 운영에서 batch를 절반으로 바꾼 직후 validation loss가 `약 3.96 -> 3.94`로 불연속적으로 변한 것은 이 집계 방식의 영향일 가능성이 높다.

권장 수정은 각 loss numerator와 유효 element/sample count를 함께 누적해 전 데이터 기준 weighted mean을 계산하는 것이다. 수정 전후의 validation loss는 직접 비교하면 안 된다.

### 8.3 Seed와 validation noise

매 epoch validation 전에 같은 seed를 재설정하므로 validation noise/timestep은 재현 가능하다. 이는 epoch 간 비교에는 좋다. 다만 batch size가 바뀌면 random draw grouping과 평균 편향이 달라져 동일 metric contract가 깨진다.

### 8.4 장기 10,000 epoch 설정

현재 80,000 sample, batch 8,704이므로 epoch당 약 10 optimizer step이다. 10,000 epoch는 약 100,000 step에 해당한다. epoch 숫자만 보면 과도해 보이지만 step 기준으로 함께 판단해야 한다. 그래도 condition alignment 지표 없이 diffusion validation loss만 장기간 최적화하면 property control이 개선되는지 알 수 없다.

## 9. 생성 과정

### 9.1 초기 상태와 reverse process

생성 시 ring 수는 학습된 분포에서 생성하지 않고 CLI `--num-rings`로 고정한다. 현재 실험은 13개다.

- ring token: 전부 MASK
- connection token: off-diagonal 전부 MASK
- center: centered Gaussian
- normal: normalized Gaussian
- condition: `log10(target+1e-6)` 후 checkpoint mean/std로 표준화

200 reverse step 동안 continuous pose와 categorical ring/connection을 갱신한다.

### 9.2 연결 후보 cache와 tree projection

최종 logits에서 모든 ring pair, SINGLE/FUSED/SPIRO, 양쪽 port 조합의 점수를 한 번 계산해 cache한다. 각 attempt는 같은 후보 점수에 Gumbel noise만 더해 정렬을 바꾸므로 과거의 전체 candidate table 재계산 병목을 제거했다.

projector는 union-find로 cycle을 만들지 않는 edge를 greedy 선택해 spanning tree를 만든다. `project()`에는 extra closure 기능이 있지만 `sample()` 호출은 기본 `max_extra=0`을 바꾸지 않는다.

결과적으로 현재 생성되는 **고리 인접 그래프는 tree**다. 원자 수준에서는 fused ring 때문에 polycycle일 수 있지만, ring-node topology 자체의 peri-condensed cycle은 만들지 못한다. closure 개수나 ring adjacency cyclomatic number를 조건/분포로 학습·샘플링하는 개선이 필요하다.

### 9.3 RDKit와 validity scorer 선택

각 분자당 기본 32개 connection tree를 탐색한다. 연결된 후보는 RDKit `Chem.SanitizeMol`을 실제로 실행해 valid 여부를 확인한다.

선택 순서는 다음과 같다.

1. RDKit-valid 후보가 하나라도 있으면 valid 후보만 남긴다.
2. conditional edge score + validity scorer log probability가 가장 큰 후보를 선택한다.
3. valid 후보가 없을 때만 전체 후보 중 joint reward 최고를 선택한다.

따라서 현재 높은 생성 성공률의 주된 원인은 learned validity scorer 단독이 아니라 **32개 후보에 대한 exact RDKit best-of-N filtering**이다. scorer는 valid 후보 사이의 ranking 또는 valid 후보가 전혀 없을 때의 fallback 역할이 크다.

### 9.4 성능 병목

diffusion은 GPU batch로 빠르게 계산되지만 connection search는 분자별 Python loop와 RDKit sanitize를 수행한다. 1,000개 × 32 후보는 프로세스당 약 32,000회 후보 projection/검사를 만든다. 실제 생성이 GPU 메모리보다 CPU single-process 병목을 보인 이유다.

개선 방향:

- candidate score tensorization
- tree state의 incremental update
- multiprocessing pool로 RDKit sanitize 분리
- early accept/upper-bound pruning
- 후보 수 adaptive scheduling
- identical topology/SMILES memoization

### 9.5 Property transform의 하드코딩

샘플러에 manifest transform을 읽는 것처럼 보이는 코드가 있지만 `if False` 때문에 결과적으로 항상 `log10`을 사용한다. 현재 DB에는 맞지만 다른 property transform으로 checkpoint를 만들면 조용히 잘못된 condition을 입력한다. manifest의 `target_transform`과 epsilon을 실제로 읽고 지원하지 않는 transform은 명시적으로 실패해야 한다.

### 9.6 출력 JSONL

각 생성 record는 다음을 포함한다.

- `target_f_osc`, `normalized_condition`
- `connected`
- rings: template ID, signature, generated center, normal
- connections: ring index, type, ports, log score
- connection search metadata
  - 요청/실제 후보 수
  - valid 후보 수
  - 선택 attempt
  - RDKit selected validity
  - conditional score
  - learned validity probability
  - joint reward

현재 summary JSON은 connected count와 template/connection histogram만 제공한다. RDKit-valid rate, uniqueness, novelty, target-response correlation도 표준 summary에 포함하는 것이 좋다.

## 10. 원자 그래프와 SDF 출력

`export_rigid_sdf.py`는 DSU로 공유 원자를 병합한다.

- SPIRO: 한 vertex 병합
- FUSED: compatible edge의 두 vertex 병합
- SINGLE: 원자를 병합하지 않고 단일 결합 추가

이후 RDKit sanitization, explicit hydrogen 추가, ETKDGv3 embedding, MMFF 또는 UFF optimization을 거쳐 SDF를 쓴다.

문제점:

- 생성한 ring center/normal은 SDF 3D embedding의 constraint로 사용되지 않는다. diffusion이 생성한 pose는 JSON에 남지만 최종 3D 구조는 RDKit이 새로 만든다.
- `--per-target` 종료 조건이 `(0.1, 0.8, 3.0)`으로 하드코딩돼 있다. 다른 target에서는 우연히 전체 입력을 끝까지 읽어 동작하지만 API contract가 잘못돼 있다.
- embedding seed가 전역 index 기반이어서 재현 가능하지만 seed별 provenance가 SDF property에 명시되지 않는다.
- 대량 export는 전체 JSONL을 `read_text().splitlines()`로 메모리에 읽는다.

## 11. Validity scorer와 property scorer

### 11.1 Validity scorer

24개 handcrafted feature를 사용하는 2-layer MLP binary classifier다. feature에는 edge score 통계, connection type 비율, hetero/aromatic/charge 비율, 원소 비율, ring size, port 사용률, valence capacity, ring/edge 수와 normalized condition이 포함된다.

현재 학습 결과는 12,288 candidate, positive rate 약 26.95%, validation accuracy 약 74.65%, precision 약 51.6%, recall 약 78.9%였다. 작은 데이터와 random row split이므로 같은 생성 분자에서 나온 서로 유사한 후보가 train/validation에 나뉘는 leakage 가능성이 있다. molecule/group 단위 split이 필요하다.

Validity 자체는 condition과 인과 관계가 없으므로 feature에 normalized condition을 넣으면 spurious correlation을 배울 가능성이 있다.

### 11.2 Property scorer

동일 feature를 입력받아 log10 oscillator strength를 예측하는 MLP다. 현재 validation correlation은 약 0.417, log10 MAE 약 0.668로 약하다. 또한 실제 샘플링 코드에는 property scorer reward가 통합돼 있지 않다. 따라서 현재 generation candidate selection은 목표 property alignment를 직접 평가하지 않는다.

## 12. xTB/sTDA 평가

후처리는 분자별로 다음을 실행한다.

1. `xtb input.sdf --opt loose --cycles 500`
2. `xtb4stda xtbopt.sdf`
3. `stda -xtb -e 10.0`
4. state 1-10의 energy, wavelength, oscillator strength 파싱

장점은 quantum-chemical 평가가 학습/GPU inner loop와 분리돼 있다는 것이다.

운영상 문제점:

- `xtb4stda` return code를 확인하지 않고 `stda`를 실행한다.
- `stda` return code도 parsed row가 있으면 `ok`가 될 수 있다.
- 완료된 분자 resume/skip 기능이 없어 재실행하면 계산을 덮어쓴다.
- 결과 JSON/CSV는 seed 전체 작업이 끝나야 작성된다. 중간 상태는 stdout log를 다시 파싱해야 한다.
- launcher의 OMP 설정과 무관하게 스크립트가 OMP/MKL thread를 4로 덮어쓴다.
- output directory가 이미 있을 때 stale file과 새로운 결과가 섞일 수 있다.
- input SDF의 target과 계산된 f_osc 간 오차/상관을 자동 집계하지 않는다.

권장 방식은 분자별 `result.json`을 원자적으로 저장하고, 성공 결과가 있으면 skip하며, 마지막에 aggregate하는 것이다.

## 13. 테스트와 품질

현재 표준 `unittest` 15개는 통과한다. 테스트 범위는 다음을 포함한다.

- chemical loss finite/gradient/invalid port
- curriculum sampler 확장
- fused/single/spiro decoder
- graph port reuse와 cycle rank
- 데이터 전처리 기본 동작
- projector tree/closure 및 disconnected failure
- property band sampler 비율/재현성
- real data collate

누락된 중요한 테스트:

- 학습 condition drop과 sampling unconditional의 동일성
- guidance scale별 cond-uncond delta
- resume 후 과거 best 보존
- batch size와 무관한 validation metric
- checkpoint interrupted write/reload
- 48개 template 전체 round-trip
- generated JSON -> RDKit -> canonical SMILES property-based test
- xTB stage별 return-code handling과 resume
- export의 arbitrary target 처리
- sampling output uniqueness/reproducibility

코드 스타일은 핵심 운영 스크립트가 여러 statement를 한 줄에 배치해 리뷰와 안전한 수정이 어렵다. Ruff line length 설정 100과도 크게 어긋난다. 과학적 로직을 변경하기 전에 formatter 적용과 함수 분리가 필요하다.

README도 현재 실데이터 condition, chemical loss, 48 templates, tree search, validity scorer의 실제 상태와 일부 불일치한다. 특히 “target property conditioning은 향후 단계”라는 문구는 이미 구현된 condition embedding과 모순된다.

## 14. 우선순위별 코드리뷰 Finding

### Critical

1. **Validation loss가 batch size에 종속됨**
   모델 선택과 실험 비교를 오염시킨다. sample/element count weighted aggregation으로 수정해야 한다.

2. **CFG unconditional branch 불일치**
   학습은 projected condition을 mask로 0화하지만 생성은 condition 값 0을 projection한다. true null branch로 통일해야 한다.

### High

3. **Resume가 historical best를 보존하지 않음**
   resume 직후 더 나쁜 checkpoint가 `best.pt`를 덮어쓸 수 있다.

4. **Condition 사용을 강제·측정하는 목적함수와 metric 부재**
   diffusion loss 감소와 property controllability를 구분할 수 없다.

5. **생성 topology가 기본적으로 ring-adjacency tree에 제한됨**
   `max_extra=0`이 고정되어 peri-condensed topology 표현을 제한한다.

6. **생성 pose가 최종 SDF에 전달되지 않음**
   RDKit ETKDG가 3D를 새로 만들기 때문에 center/normal diffusion의 실질적 활용이 약하다.

### Medium

7. property transform이 `log10`으로 하드코딩됨.
8. `orientation`과 instance `radii` 일부가 저장되지만 학습 출력에는 활용되지 않음.
9. validity classifier split leakage와 condition feature의 spurious correlation 위험.
10. xTB/sTDA return code, resume, incremental result 관리 부족.
11. export 종료 target이 0.1/0.8/3.0으로 하드코딩됨.
12. 고정 `num_rings`로만 생성하며 ring-count distribution을 생성하지 않음.

### Low

13. 중복 import와 과도하게 압축된 one-line 코드.
14. README 및 실행 문서가 현재 구현과 불일치.
15. dependency에서 RDKit/xTB 실행 환경이 완전하게 재현되지 않음.

## 15. 권장 개선 순서

### 1단계: 측정 신뢰성

1. validation weighted mean 수정
2. resume best/patience 복구
3. condition contribution logging
4. target별 sTDA correlation, MAE, calibration curve 자동 생성
5. 현재 checkpoint들을 수정 전 metric과 분리 표기

### 2단계: Condition 학습 강화

1. CFG null branch 수정
2. denoised hidden/graph에서 property를 예측하는 auxiliary head 추가
3. `L_property = SmoothL1(predicted_log_fosc, target_log_fosc)` 추가
4. shuffled-condition contrastive 또는 ranking loss 검토
5. guidance scale ablation을 고정 seed/동일 noise로 수행

Property auxiliary head는 실제 sTDA oracle을 대체하지 못하지만 condition을 hidden representation에 보존하도록 강제한다. 이후 xTB/sTDA 결과를 sparse reward dataset으로 누적해 surrogate를 반복 개선하는 편이 현실적이다.

### 3단계: 생성 구조 개선

1. tree 이후 closure 수를 학습 분포와 맞춰 허용
2. constrained beam/SMC로 여러 partial tree 유지
3. candidate cache의 tensor화 및 incremental constraint state
4. exact RDKit check를 multiprocessing으로 분리
5. generated center/normal을 RDKit distance/planarity constraint로 전달

### 4단계: 운영 안정성

1. xTB 분자별 atomic result + resume
2. seed/checkpoint hash/config/git commit을 모든 output manifest에 기록
3. launcher와 worker thread 설정 단일화
4. 대량 JSONL streaming I/O
5. integration test와 end-to-end smoke test 추가

## 16. 최종 평가

현재 코드는 “고리 보존형 분자 생성 + exact RDKit 후보 필터 + 비동기 quantum evaluation”이라는 방향은 타당하고, 100k provenance와 48개 template 데이터베이스도 검증돼 있다. 32개 connection 후보 탐색을 통해 RDKit/3D 성공률 80% 이상을 달성한 점은 실용적 성과다.

하지만 이 성공률은 condition 정확도가 아니라 구조 유효성 성공률이다. 현재 `f_osc` condition은 모델에 입력되지만 전용 loss가 없고 CFG null 구현에도 불일치가 있다. 따라서 다음 모델 수정의 핵심 목표는 epoch를 더 늘리는 것이 아니라 **condition이 실제 output을 변화시키는 정도를 측정하고, 그 변화가 sTDA oscillator strength와 정렬되도록 학습 신호를 추가하는 것**이어야 한다.
