# 3D 분자 생성의 화학적 유효성·방향족성: 논문 분석과 HP1KG Stage 8 적용 리포트

작성일: 2026-10-29 (Asia/Seoul)  
분석 저장소: `PSANGGIL/Excited_Guidance_rigid`  
대상 브랜치: `feature/molguidance-stage8-joint-decoding-mg-budget`  
코드 기준: `c8af84d6a382f76c8757ffbc149e771a992f6629`  
기존 분석 문서 기준: `06d5fdf476b79a1a1ab1d575201b1fe54ee592ab`  
문서 성격: 원문에 근거한 연구 검토 + 코드 정적 분석 + 후속 실험 제안. 모델 수정·재학습·신규 생성 실험의 결과 보고서가 아니다.

## 읽는 순서와 근거 구분

이 문서는 “3D 좌표로 분자를 생성할 때 화학적으로 유효하지 않은 구조가 나오는 문제, 특히 방향족 고리 문제를 기존 연구는 어떻게 다뤘는가?”를 현재 구현과 연결한다. 기본 diffusion 논문만 나열하지 않고, flow matching, 조각 조립, 역문제, 조합최적화, 다양체 학습까지 비교한다.

- 전체 판단과 우선순위: 1–3절, 9–10절.
- 개별 논문: 4–7절. 총 17편과 RDKit 공식 문서를 검토한다.
- 방향족성과 보정의 수학적 구조: 2절, 8절.
- 현재 코드의 구현·관측·가설 구분: 3절, 9절.
- 재현 가능한 실험 설계와 평가 계약: 10–12절.
- 원문·코드 링크 및 확인 범위: 13–14절.

본문의 **[논문]**은 해당 원문이 기술하거나 보고한 사실, **[코드]**는 기준 커밋의 구현, **[관측]**은 저장소에 기록된 생성 결과, **[해석/제안]**은 이 리포트의 추론과 미검증 설계다. 논문 성능은 데이터·수소 표현·후처리·분모가 다르면 직접 비교하지 않는다. 출판본과 preprint 수치도 섞지 않는다.

원문은 출판사·학회·arXiv·저자 제공 자료를 사용했다. 일부 출판사 페이지는 직접 열기가 제한되어 검색에 제공된 출판사 본문과 공개 preprint를 대조했다. 모든 보충자료와 저자 코드를 실행·감사한 것은 아니다. 특정 주장의 확인 위치와 문헌 버전은 13절에 명시한다.

## 1. 핵심 판단

현재 Stage 8에는 이미 다음이 구현돼 있다.

1. `(원소, formal charge, attached H count)`의 joint atom state.
2. `NONE/SINGLE/DOUBLE/TRIPLE`의 Kekulé 결합 표현.
3. 동일 분자의 대체 Kekulé assignment를 사용하는 표현 증강.
4. 원자가 **분포**와 총전하의 호환성 loss.
5. 방향족 결합 영역의 SINGLE/DOUBLE 확률과 원자별 이중결합 수를 다루는 보조 loss.
6. 원자가·연결성·총전하·선택적 원자 수 제약을 결합한 최종 MILP decoder.
7. RDKit sanitize, radical 검사, 전역 Kekulization을 이용한 최종 후보 검증.

따라서 “결합 정보를 추가하자”, “Kekulé로 바꾸자”, “전역 solver를 붙이자”는 이 코드의 새로운 해결책이 아니다. 해당 장치가 **어떤 후보를 입력받고, 어떤 점수로 선택하며, 무엇을 보장하는지**가 다음 연구 대상이다.

현재 관측을 설명하는 데 가장 직접적인 세 가지 문제는 다음과 같다.

- **Decoder에 전달되는 후보의 소실:** 낮은 temperature를 적용한 float32 확률의 수치적 0이 hard support처럼 사용된다.
- **생성과 decoder의 원자 수 계약 불일치:** FAKE/real 생성으로 달라진 실제 원자 수와 solver의 strict requested count가 충돌한다.
- **보정용 점수의 학습 불일치 가능성:** 주 학습에서 직접 감독하지 않는 visible 위치의 head 출력까지 최종 solver가 재선택 점수로 사용한다.

첫 두 항목은 기존 리포트의 필요조건 검사와 코드로 확인된다. 세 번째는 구현상 근거가 있는 가설이며 기여율을 입증한 실험은 없다. 이들을 해결해도 고리 조성 편향·3D 왜곡·전자구조 적합성은 별도로 평가해야 한다.

문헌에서 가장 직접적으로 가져올 교훈은 다음과 같다.

| 연구 계열 | 가져올 원리 | 현재 코드에 대한 판단 |
|---|---|---|
| MolDiff·MiDi | 원자·결합·좌표를 함께 학습하고 modality별 corruption을 설계 | 공동 생성은 이미 구현. 학습 경로와 실제 추론 경로의 차이를 조사 |
| GEOM-drugs revisited | 방향족 원자가 산정과 validity 평가 정의를 먼저 교정 | Kekulé는 이미 적용. 평가 일관성과 state 보존 검사가 중요 |
| FlowMol3 | 추론 중 분포 이탈에 견디도록 self-conditioning·FAKE·교란 학습 | 유사 요소가 존재하지만 논문과 동일한 학습 절차는 아님 |
| JT-VAE·FragGen | 화학적으로 유효한 조각의 내부를 보존하고 연결을 학습 | 고리 topology 자체의 과생성이 남을 경우 중장기 대안 |
| DIFUSCO | 연속적인 후보 점수를 유지하고 전역 제약으로 조합을 선택 | 현재 solver 앞의 확률 소실 문제와 특히 직접적으로 연결 |
| PDM·DPS·RFM | 제약 투영, 미분 가능한 유도, 적합한 상태공간 사용 | 화학적 이산 제약과 연속 기하 제약을 구분해 전이 |

## 2. ‘유효한 분자’와 ‘방향족 문제’를 분리해서 정의하기

### 2.1 다섯 수준의 유효성

| 수준 | 질문 | 예시 검사 | 이 검사만으로 보장하지 못하는 것 |
|---|---|---|---|
| 표현 | 객체를 읽고 일관된 그래프로 해석할 수 있는가? | parser, sanitize, atom/bond schema | 저에너지 구조·합성 가능성 |
| 전자수·국소 원자가 | 원소·전하·H·결합차수 조합이 허용되는가? | heavy bond-order sum, formal charge, radical flag | 고리 전체의 방향족성·3D 구조 |
| 전역 그래프 | 연결성·고리계·Kekulé assignment가 일관되는가? | component 검사, shared-atom constraints, global Kekulization | 해당 XYZ의 결합거리·입체 타당성 |
| 3D 기하 | 이 그래프를 이 좌표로 실현할 수 있는가? | 거리·각도·비결합 충돌·평면성·relaxation | 고수준 전자구조·반응 안정성 |
| 연구 목적 | 원하는 전자상태·물성·구조 다양성을 갖는가? | 동일 수준 전자구조 계산, 상태 분석, scaffold 분포 | 합성 경로·실험적 수명 전반 |

RDKit의 radical 수가 0이라는 것은 해당 toolkit 표현에서의 검사다. 실제 전자구조가 단일참조 singlet인지, diradical 성격이 없는지, Hessian이 최소점을 지지하는지를 증명하지 않는다. 본 프로젝트의 “중성·폐각” 필터도 그 운영상 정의와 양자화학적 증명을 구별해야 한다.

### 2.2 방향족성은 개별 edge label이 아니다

**[공식 문서]** RDKit은 원자 환경에 따른 π 전자 기여와 고리 또는 융합 고리계에 대한 규칙으로 방향족성을 인식한다. 두 aromatic atom 사이의 결합이 항상 aromatic인 것은 아니다. 개별 고리보다 융합계 전체를 봐야 하는 예도 있다. 따라서 원자 flag만으로 aromatic perimeter를 정의하면 다른 결합이 섞일 수 있다. [D01]

**[해석]** 생성 모델의 방향족 문제는 적어도 세 경우로 나뉜다.

1. 연결 구조는 적절하지만 H/전하/결합차수 선택이 서로 모순되는 경우.
2. 전자수 규칙에 맞는 고리계가 존재하지만, 원래 의도한 scaffold와 다른 그래프를 만든 경우.
3. 그래프는 방향족으로 해석되지만 XYZ가 심하게 뒤틀리거나 입체적으로 충돌하는 경우.

첫 경우에는 공동 이산 보정이 유효할 수 있다. 두 번째에는 topology 또는 motif 모델이 필요하다. 세 번째에는 기하 보정이 필요하다. “aromatic loss 증가” 한 가지로 세 문제를 동시에 해결했다고 주장할 수 없다.

### 2.3 Kekulé 표현이 해결하는 것과 남기는 것

Kekulé 표현은 공명계의 결합을 정수 SINGLE/DOUBLE로 나타내므로 결합차수 합을 일관되게 계산하기 쉽다. 그러나 벤젠에도 대체 assignment가 있으며, 융합계에서는 여러 고리가 공유 원자와 공유 결합을 가진다. 정수 표현으로 바꿔도 그 변수들을 동시에 일관되게 선택해야 한다.

예를 들어 중성 탄소에 대한 일반적인 폐각 예에서는

$$
H_i+\sum_{j\ne i}b_{ij}=4
$$

가 필요한 조건이다. 여기서 $b_{ij}$는 정수 결합차수이며 degree와 다르다. 이웃이 네 개 이하라도 결합차수 합은 5 이상일 수 있다. 모든 원소에 위 식의 우변 4를 적용해서는 안 되며, 실제 코드는 원소·전하·H state에 연결된 허용 heavy valence 집합을 사용한다.

고리 전체에서 $4n+2$를 세는 단순 규칙 역시 임의의 융합계·exocyclic 결합·heteroatom 환경을 완전히 대체하지 못한다. 단순 고리에서의 직관과 toolkit의 전역 인식 규칙을 구분하고, 최종 선택된 전체 그래프에서 검증해야 한다.

## 3. 현재 Stage 8의 기준선

### 3.1 표현과 학습 경로

**[코드]** 활성 분석 대상은 `MolGuidance_guidance_selector_e987/`의 HP1KG Stage 8 CFG 경로다. 저장소 루트의 PolyRing prototype과 그 생성 성능을 혼동하지 않는다. 루트 README는 별도 고리 단위 prototype의 존재를 설명하지만, 본 보고서의 1,000개 결과는 그 prototype 결과가 아니다.

분자를 다음처럼 표기한다.

$$
G=(X,S,B),\qquad X\in\mathbb R^{N_{slot}\times3},\quad
S_i=(Z_i,q_i,H_i)\text{ 또는 FAKE},\quad B_{ij}\in\{0,1,2,3\}.
$$

현재 설정은 12개 원소, FAKE를 포함한 38개 endpoint state를 사용하며 CTMC MASK는 corruption을 위한 별도 상태다. 수소는 독립 좌표 node가 아니라 heavy atom의 정수 H count다. 좌표는 endpoint flow, categorical 변수는 masked CTMC로 생성하므로 Gaussian DDPM으로만 설명하면 부정확하다. [C01], [C02]

좌표의 선형 조건부 경로와 endpoint parameterization의 핵심은

$$
X_t=(1-t_x)X_0+t_xX_1,
\qquad v_\theta(X_t,t_x)=\frac{\widehat X_1-X_t}{1-t_x}
$$

이다. 일반 schedule에서는 해당 $\alpha(t)$와 도함수로 표현해야 한다. 위 식은 현재 linear 좌표 schedule을 설명하는 식이다.

학습에서는 graph time과 coordinate time을 따로 주고 clean graph branch도 사용한다. 실제 추론은 대각선 $t_x=t_g$ 경로다. `bond_first_geometry`는 결합 feature가 좌표 갱신에 관여한다는 뜻이며 “완전한 유효 그래프를 먼저 확정한 뒤 좌표를 생성한다”는 별도 두 단계 모델을 뜻하지 않는다. [C02], [C03]

### 3.2 현재 실행 조건과 결과

**[관측]** HP1KG 1,581개를 train 1,264 / validation 158 / test 159로 나누었다. 분석 checkpoint는 validation loss 기준 e4690, optimizer step 201,713이다. 전체 학습은 10,000 epoch / 430,000 step이 기록돼 있다. 현재 목표는 $Q=S_1-2T_1=0$ eV이며 $\Delta E_{ST}=S_1-T_1$과 다르다. 생성은 1,000개, CFG 1, 100 timepoints, seed 20261027이다. [C10]

| 지표 | 개수 | 분모와 해석 |
|---|---:|---|
| raw RDKit valid | 574 | 전체 1,000개 |
| connected raw valid | 565 | 전체 1,000개 |
| 연결·중성·toolkit상 radical 없음, 계산 완료 | 452 | 전체 1,000개 |
| invalid graph로 계산 전 제외 | 426 | 판정 순서상 상호배타적 분류 |
| open-shell로 제외 | 98 | 앞 단계 통과 후 판정 |
| disconnected로 제외 | 9 | 앞 단계 통과 후 판정 |
| nonzero net charge로 제외 | 15 | 앞 단계 통과 후 판정 |
| requested count와 sampled real count 일치 | 119 | 전체 1,000개 |
| 별도 joint projection optimal | 93 | raw 결과를 대체하지 않은 별도 1,000회 |

426개 invalid 중 312개는 H를 0으로 해도 heavy bond-order 합이 초과한다. H-only 변경으로 RDKit valid가 되는 111개와 neutral/closed-shell가 되는 106개는 **반사실 진단**이며 새로운 생성 성공분이 아니다. [C10]

### 3.3 비교에서 유지할 조건

현재 HP1KG는 논문들의 대규모 GEOM-Drugs와 학습 분포·크기·원소·H 표현·목표가 다르다. 특히 explicit H를 개별 원자로 생성한 논문의 validity와, attached H count를 생성하는 현재 모델의 validity는 같은 실패 조건을 갖지 않는다. 기존 논문의 높은 수치를 현 checkpoint의 기대 성능으로 그대로 적용하지 않는다.

현재 452개 계산 대상의 전부가 XTB/sTDA를 완료했다는 사실도 구조 생성 성공률 100%를 뜻하지 않는다. 구조 필터가 먼저 작동했기 때문이다. NEVPT2 학습 라벨과 XTB/sTDA 평가값의 차이는 물성 조건 제어와 계산 방법 차이를 함께 포함하므로 단순 MAE 하나로 guidance의 원인을 분리할 수 없다. [C10]

## 4. 좌표·그래프 공동 생성과 flow matching 논문

### 4.1 EDM — Equivariant Diffusion for Molecule Generation in 3D (ICML 2022)

**[논문: P01]** Hoogeboom 등의 EDM은 atom type과 3D 좌표를 equivariant network로 함께 denoise한다. 좌표는 중심이 제거된 공간에서 다루고 회전·반사 등에 대한 대칭을 모델에 반영한다. 별도 결합차수 생성 변수를 중심에 둔 후속 joint graph 모델과 달리, 생성 원자·좌표에서 결합을 복원하는 설정이 핵심 비교점이다. QM9와 GEOM-Drugs를 사용해 이전 3D 생성 방법보다 개선된 결과를 제시했다.

가우시안 corruption의 개략식은

$$
z_t=\alpha_t z_0+\sigma_t\epsilon,\qquad
\mathcal L_{noise}=\mathbb E\|\epsilon-\epsilon_\theta(z_t,t)\|^2
$$

으로 쓸 수 있다. categorical feature의 연속 인코딩·좌표의 zero-center 조건 등은 실제 구현에서 별도로 처리된다.

**[해석/적용]** 이 논문의 중요한 한계는 equivariance가 화학적 유효성과 다른 조건이라는 점이다. 좌표 회전에 일관된 예측을 하는 모델도 모순된 결합 구조를 만들 수 있다. 현재 코드는 이미 bond-order와 joint state를 학습하므로 EDM형 모델에 단순 결합 후처리를 추가하는 단계보다 앞서 있다. EDM은 “좌표를 잘 생성하면 화학 그래프도 저절로 성립한다”는 전제를 검토하는 기준선으로 읽는 것이 적절하다.

### 4.2 GeoDiff — A Geometric Diffusion Model for Molecular Conformation Generation (ICLR 2022)

**[논문: P02]** Xu 등의 GeoDiff는 **주어진 분자 그래프에서 conformation을 생성**한다. equivariant Markov kernel을 이용해 회전·이동에 일관된 분포를 구성하고, 조건부 likelihood의 변분 목적을 학습한다. 입력 그래프가 이미 원자와 결합 관계를 제공한다는 점이 de novo graph generation과 다르다.

과제는

$$
p_\theta(X\mid G_{chem})
$$

이지, 결합 그래프 자체까지 포함한 $p_\theta(G_{chem},X)$ 전체가 아니다.

**[해석/적용]** 현재 문제에서 GeoDiff 계열은 화학적으로 일관된 그래프를 얻은 **이후**의 좌표 생성·복원 단계에 적합하다. 잘못된 N의 전하나 융합 고리의 모순된 bond order를 좌표 diffusion만으로 해결한다는 근거는 아니다. 두 단계 비교를 설계한다면 `joint decode → graph-conditioned coordinate refinement`를 별도 arm으로 둔다. 단, 원래 생성 좌표에서 얼마나 이동했는지와 refine 이전 품질을 반드시 남긴다. 그래프를 바꾼 뒤 전체 좌표를 재생성한 결과를 raw 3D generation 성능으로 보고하면 안 된다.

### 4.3 MolDiff — Addressing the Atom–Bond Inconsistency Problem (ICML 2023)

**[논문: P03]** Peng 등의 MolDiff는 atom type·좌표·bond type을 공동 생성한다. 결합 class에는 aromatic이 포함되며 H는 별도 원자로 생성하지 않는다. 두 단계 **forward noising**에서는 결합 정보를 먼저 빠르게 제거하고, 그동안 원자 type과 좌표는 약하게 교란한다. 이후 원자 정보를 더 교란한다. 별도로 학습한 bond predictor의 confidence gradient로 좌표 생성도 유도한다. 원문 Table 1의 MolDiff validity는 0.997, connectivity 0.996, success rate 0.993이다. 이는 해당 데이터 처리와 평가 조건의 결과다.

좌표 guidance의 구조는

$$
\mu'_{\theta,t}=\mu_{\theta,t}+s\nabla_{X_t}\log C_\phi(A_t,X_t,t)
$$

이다. $C_\phi$는 bond predictor에서 얻는 confidence이며 실험 물성 자체의 likelihood가 아니다.

**[해석/적용]** 가장 유용한 부분은 “bond 정보를 넣는다”보다 **교란된 graph와 geometry 사이에 어떤 학습 문제를 만들 것인가**다. 현재 코드의 독립 graph/coordinate time과 MolDiff의 두 단계 schedule은 동일하지 않다. 특히 ‘bond-first’라는 말은 forward noising 순서인지, network update 순서인지, reverse generation 순서인지 명시해야 한다. MolDiff가 정답 방향족 electron assignment를 hard constraint로 보장한다고 해석할 근거는 없다. 현재 H count 충돌은 MolDiff의 비명시적 H 설정과도 구별해 평가해야 한다.

### 4.4 MiDi — Mixed Graph and 3D Denoising Diffusion (2023)

**[논문: P04]** Vignac 등의 MiDi는 atom type·formal charge·bond type의 discrete diffusion과 좌표의 Gaussian diffusion을 결합한다. adaptive schedule은 atom type과 charge를 좌표·결합보다 빠르게 교란하여 reverse process에서 geometry/bond와 원자 identity의 복원 시점을 조절한다. 원문 GEOM-Drugs explicit-H Table 3에서 adaptive 모델의 molecular stability는 91.6%, RDKit validity는 77.8%, connected는 90.0%다. **91.6%를 validity로 바꾸어 인용하면 틀린다.**

**[해석/적용]** 여러 head를 동시에 예측하는 공동 모델도 최종 sample의 전역 제약을 자동으로 만족시키지는 않는다. 또한 후속 GEOM-drugs revisited가 aromatic valence 계산과 lookup table의 문제를 지적했으므로, 원문 stability 숫자는 당시 평가 정의의 수치로 보존해야 한다. 이 수치를 현재 57.4% raw RDKit validity와 단순 대조할 수 없다. 현재 코드는 원소·전하·H를 한 state로 묶어 **원자 내부** 조합을 제한하지만, 그 state와 이웃 결합들 사이의 **원자 간** 일관성은 여전히 학습·sampling·decoder가 공동으로 해결해야 한다.

### 4.5 FlowMol3 — Flow Matching for 3D De Novo Small-Molecule Generation (2025 preprint / 2026 출판)

**[논문: P05]** Dunn과 Koes는 연속 좌표 flow와 categorical CTMC에 self-conditioning, removable FAKE atoms, geometry distortion을 결합한다. 추론 중 분포 이탈에 대한 견고성은 저자들의 설명 가설이며, 이를 직접 측정한 정확한 marginal drift 값과 혼동하지 않는다. 출판본 Table 1은 GEOM-Drugs explicit-H에서 validity 100.0%, PB-validity 95.9%, median relaxation energy 4.50 kcal/mol, median RMSD 0.28 Å를 보고한다. 높은 validity에도 드문 유효 motif의 과소생성 가능성을 논의한다.

| 항목 | 2025 arXiv v1 | 2026 출판본 |
|---|---:|---:|
| distortion 적용 원자 확률 | 0.2 | 0.7 |
| distortion 시간 경계 | 0.5 | 0.25 |
| PB-validity | 91.9% | 95.9% |

**[해석/적용]** 현재 코드에 유사한 구성 요소가 있다고 해서 FlowMol3 재현 모델이라고 부를 수는 없다. 현재 correction은 별도 auxiliary pass이며 `time=0.85`, `coordinate_std=0.05`, 실행 probability 0.25다. 이 probability는 논문의 **원자별** distortion 확률과 의미도 다르다. FAKE는 삭제 자유도를 제공하지만 정확한 원자 수 조건과 충돌할 수 있다. 특히 구조 validity만 최대화하면 유효하지만 드문 고리·치환기 분포가 줄어들 수 있으므로, 이를 단순 성능 향상으로 해석하지 않는다. 본 프로젝트에는 correction의 강도보다 **교란 분포와 실제 residual error의 일치**를 먼저 점검하는 실험이 적합하다.

### 4.6 PropMolFlow — Property-guided Molecule Generation with Geometry-Complete Flow Matching (2026)

**[논문: P06]** Zeng 등의 PropMolFlow는 FlowMol 기반 공동 생성에 물성 embedding을 결합한다. 선택적 Gaussian expansion과 다섯 embedding 결합 방식을 비교하며, 생성 물성의 predictor 평가를 DFT와 대조한다. revised QM9와 explicit H·formal charge·bond order를 사용하고, closed-shell ratio와 구조 지표를 별도로 평가한다. 출판일은 2026-01-21이며 DOI 문자열에 `025`가 있다고 출판연도를 2025로 적지 않는다.

스칼라 조건의 Gaussian feature는 일반적으로

$$
\phi_k(y)=\exp\left[-\frac{(y-\mu_k)^2}{2\sigma_k^2}\right]
$$

같이 표현할 수 있고, 이후 MLP와 graph feature 결합을 적용한다.

**[해석/적용]** 본 프로젝트에서 중요한 교훈은 물성 predictor의 낮은 오차와 실제 전자구조 계산 결과를 구별하는 것이다. $S_1-2T_1$의 목표 오차가 작더라도 잘못된 graph 또는 과도하게 변형된 좌표라면 목표 분자 생성의 성공으로 보기 어렵다. 또한 같은 그래프의 coordinate refinement 전후에는 조건 물성 자체가 변할 수 있다. 구조 적합성 평가를 통과한 뒤, 생성에 쓴 조건과 동일한 물리량·계산 수준·상태 정의로 검증해야 한다. QM9에서 얻은 embedding 우열을 HP1KG의 작은 조건부 데이터에 그대로 이전할 수는 없다.

### 4.7 MolGuidance — Advanced Guidance Strategies for Conditional Molecular Generation with Flow Matching (2025 preprint)

**[논문: P07]** Jin 등의 공개 preprint는 vanilla conditioning, CFG, autoguidance, model guidance를 비교한다. 연속 modality는 velocity, 이산 modality는 probability/logit/rate 형태를 검토하고, modality별 guidance scale을 Bayesian optimization으로 조정한다. QM9·QMe14S에서 이산 변수의 guidance가 물성 제어에 미치는 영향을 강조한다. 본 절의 상세 근거 버전은 arXiv:2512.12198v1이며, 저장소 README의 저널 링크만으로 출판본의 모든 내용을 동일하다고 가정하지 않는다.

일반적인 log-CFG는

$$
\ell^{guid}_k=(1-w)\log p_{u,k}+w\log p_{c,k},\qquad
q_k=\operatorname{softmax}(\ell^{guid}/T)_k
$$

로 설명할 수 있다. linear probability extrapolation은 음의 성분이 생길 수 있으므로 별도 유효 확률 처리와 구별해야 한다.

**[해석/적용]** 현 생성은 CFG=1, `dfm_type=campbell`, linear probability 경로를 사용한다. branch 이름에 MG가 있다고 현재 1,000개를 MG 결과로 해석하면 안 된다. 또 원 논문에서 특정 guidance가 좋았다는 결과는 현 코드의 temperature 적용·support 보존이 올바르다는 검증이 아니다. 물성 제어의 강도를 높이기 전에 이산 점수의 수치적 상태와 화학적 feasible set을 확인해야 한다. atom/state와 bond의 서로 다른 scale은 화학적 trade-off도 바꾸므로 동일 원자 수·noise를 쓴 비교가 필요하다.

## 5. 방향족 표현과 구조 평가를 다시 검토한 연구

### 5.1 GEOM-drugs revisited — Toward More Chemically Accurate Benchmarks for 3D Molecule Generation (2025)

**[논문: P08]** Nikitin 등은 aromatic bond contribution의 정수화 오류, 부적절한 allowed-valence table, reference와 다른 수준의 geometry 평가를 지적했다. aromatic bond를 1 또는 1.5로 일괄 처리하는 대신 aromatic incidence와 나머지 bond-order 합을 구별하거나, Kekulé 데이터로 재학습하는 방안을 검토했다. GFN2-xTB 기반 평가도 제시했다. 일부 기존 수치가 수정된 것은 모델이 갑자기 나빠졌다는 뜻이 아니라 평가 정의가 달라졌다는 뜻이다.

**[해석/적용]** 이 논문은 “Kekulé를 사용하면 방향족 문제가 모두 해결된다”는 근거가 아니다. 현재 코드는 해당 방향을 이미 채택했다. 남은 질문은 다음과 같다.

- H/charge/state를 고정했을 때 결합차수 합이 정말 허용되는가?
- sanitize가 H나 charge를 바꿔 원래 모델 오류를 가리고 있지 않은가?
- 전역 Kekulization에 성공한 그래프와 원래 예측 XYZ가 서로 일치하는가?
- 학습 데이터에 드문 비정상 구조가 있다면 lookup table이 이를 일반적인 허용 상태로 확대하지 않는가?

현재 decoder는 H·formal charge 보존을 검사하고 global Kekulization을 수행하므로 이 부분의 방향은 적절하다. 그러나 생성 모델의 raw 결과는 별도로 평가해야 한다. 또한 train-supported state는 **이번 데이터 분포에서 허용하기로 한 상태**이지 모든 화학적 가능성의 완전한 목록은 아니다. 지원 밖 상태를 막는 것은 범위 제한이며, 수치적 underflow 때문에 지원 안 상태가 사라지는 현상과 다르다.

### 5.2 PoseBusters — AI-based Docking Methods Fail to Generate Physically Valid Poses or Generalise to Novel Sequences (2024)

**[논문: P13]** Buttenschoen 등은 docking에서 낮은 pose RMSD가 화학적·물리적 적합성을 보장하지 않음을 보였다. PoseBusters는 화학 구조·stereochemistry·결합거리·평면성·분자 내외 충돌 등을 검사한다. 논문의 출판 권호는 2024년이며 최초 온라인 공개는 2023년이다. 이는 원래 docking 평가 연구로, 생성 모델이나 방향족 복구 알고리즘 자체를 제안한 논문은 아니다.

**[해석/적용]** 현재 코드의 `geometry_screen`은 covalent-radius 비율과 aromatic plane RMS 등을 검사하는 자체 screening이다. 이를 PoseBusters 전체 통과율과 동일하게 부르면 안 된다. receptor가 없는 HP1KG에서는 protein–ligand clash 검사도 그대로 적용할 수 없다. 어떤 검사 집합을 실행했는지와 각 실패의 교집합을 기록해야 한다.

본 프로젝트의 RMSD는 raw XYZ와 XTB 최적화 후 XYZ의 변화량이다. 이는 결정구조 pose와의 RMSD도, strain energy도 아니다. 같은 RMSD라도 bond stretching인지 torsion 이동인지에 따라 에너지 의미가 다르므로 $\Delta E_{relax}$와 기하 항목을 함께 본다. aromatic plane 경고는 운영상 screening이며, 모든 방향족 구조가 무조건 하나의 평면에 놓여야 한다는 보편적 화학 명제로 확대하지 않는다.

## 6. 유효한 조각을 보존하거나 복원하는 연구

### 6.1 JT-VAE — Junction Tree Variational Autoencoder for Molecular Graph Generation (ICML 2018)

**[논문: P09]** Jin 등은 분자를 ring·bond 등의 substructure로 분해하고, 먼저 junction tree를 생성한 뒤 이를 연결하는 세부 graph를 조립한다. VAE latent도 tree와 graph 정보를 구분한다. 유효한 부분구조와 허용되는 assembly를 이용해 원자별 생성의 불완전한 aromatic 중간 상태를 피한다. 해당 실험에서 prior sample의 100% validity를 보고했으나, 이는 2D graph 생성의 결과다.

생성 구조를 개략적으로 쓰면

$$
p(G,T\mid z_T,z_G)=p(T\mid z_T)\,p(G\mid T,z_G)
$$

이다. substructure가 같은 종류라는 것과 어떤 원자들을 공유하여 결합하는지는 별개 변수다.

**[해석/적용]** 현재 고리의 내부 aromatic pattern을 매번 atom-wise로 재학습하는 부담을 줄이는 데 직접적인 참고가 된다. 다만 “모든 ring을 하나의 node로 바꾼다”와 JT-VAE의 decomposition은 동일하지 않다. 공유 원자가 많은 고리계는 큰 단위로 묶어야 할 수 있고, vocabulary가 커지면 작은 HP1KG train에서 support가 부족해진다. 또한 유효한 조각 두 개를 잘못된 fusion port로 붙이면 전체 분자가 여전히 모순될 수 있다. 3D orientation·steric fit·융합부의 공유 원자 좌표 일치는 별도 문제다.

### 6.2 FragGen — Towards 3D Geometry Reliable Fragment-based Molecular Generation (Chemical Science 2024)

**[논문: P10]** Zhang 등의 FragGen은 pocket-conditioned fragment-wise generation으로, 일반적인 atom-wise diffusion과 다른 계열이다. geometry handling을 cavity detection, bond linking, chemical initialization, dihedral handling 등으로 나누고, 상대 위치·화학적 국소 기하·torsion 예측을 결합한다. 본문은 docking뿐 아니라 relaxation energy와 optimization RMSD를 비교한다. Table 2에서 FragGen의 OptRMSD 0.878 Å는 ResGen 0.465 Å보다 크므로 “모든 기하 지표에서 최저”라고 쓰면 부정확하다. Relax E는 논문 정의상 $E_{end}-E_{initial}$로 음수 방향이다.

**[해석/적용]** 이 연구의 전이 대상은 pocket 점수보다 **fragment 내부와 fragment 사이의 자유도를 분리하는 설계**다. 방향족 core를 유지한 채 치환기 연결·회전만 생성하면 불필요한 내부 왜곡을 줄일 수 있다. 반면 새로운 융합 aromatic topology를 설계하려면 단일결합 linker assembly보다 훨씬 강한 port consistency가 필요하다. 또한 fragment 자체를 학습 DB에서 제한하면 새로운 ring system 탐색 범위가 줄어들 수 있으므로, validity 증가와 scaffold 다양성을 함께 측정해야 한다. relaxation energy의 부호는 본 프로젝트에서 정의할 $E_{raw}-E_{opt}$와 반대이므로 수치 비교 전에 통일한다.

### 6.3 DiffLinker — Equivariant 3D-conditional Diffusion Model for Molecular Linker Design (Nature Machine Intelligence 2024)

**[논문: P11]** Igashov 등은 주어진 fragment들의 3D 배치를 조건으로 linker atom의 수·type·좌표를 생성한다. 고정 길이 linker나 두 fragment만을 전제로 하지 않으며, pocket을 추가 조건으로 사용할 수 있다. fragment-conditioned point-cloud diffusion이라는 점이 핵심이고, 이미 주어진 core의 전체 화학 구조를 de novo로 다시 생성하는 과제와 다르다.

**[해석/적용]** 기존 aromatic core를 보존하면서 linker·주변 치환기를 탐색할 목적이면 유용하다. 그러나 현재 실패한 raw molecule에 임의의 fragment를 찾아 고정한다고 원래 의도한 분자가 복원되지는 않는다. 신뢰할 수 있는 부분과 오류 부분을 구분하는 기준이 필요하다. 적용 실험은 두 가지로 나누는 것이 좋다.

1. 알려진 valid graph를 일부 가려 복원하는 조건부 복구: 정답과 복구 정확도를 평가할 수 있다.
2. de novo sample에서 모순 부위를 다시 생성: 유효성·수정량·다양성은 평가할 수 있으나 “원래 정답 복원률”은 정의할 수 없다.

fragment 고정 때문에 증가한 성공률을 완전 자유 생성의 향상으로 보고하지 않는다.

### 6.4 LiGAN — Generating 3D Molecules Conditional on Receptor Binding Sites with Deep Generative Models (Chemical Science 2022)

**[논문: P12]** Ragoza 등은 receptor-conditioned atomic density grid를 CVAE로 생성하고, atom fitting과 bond inference를 통해 분자 구조로 변환한다. 이름에 GAN이 포함되어 있어도 이 논문에서 설명하는 핵심 생성 모델은 conditional VAE다. density에서 discrete atom·bond로 넘어가는 명시적인 reconstruction 단계와 후속 기하 평가가 특징이다.

**[해석/적용]** 신경망이 낸 연속적인 표현을 그대로 화학 구조라고 선언하지 않고, 별도의 decoding 계약을 둔 사례다. 하지만 density reconstruction의 존재만으로 모든 aromatic·charge·H 조합이 보장되는 것은 아니다. 현재 Stage 8도 이미 별도 joint decoder가 있으므로, LiGAN을 근거로 “후처리를 하나 붙이면 된다”고 제안하는 것은 불충분하다. 실제로 어떤 후보를 허용하고, 재구성 과정이 원자 identity와 좌표를 얼마나 바꾸며, 최종 유효성을 어떻게 확인하는지가 비교 대상이다. 본 검토에서는 LiGAN의 세부 bond-building 코드까지 실행 검증하지 않았으므로 특수 aromatic heuristic의 완전성은 주장하지 않는다.

## 7. 화학 밖에서 가져올 수 있는 보정 원리

### 7.1 Projected Diffusion Models — Constrained Synthesis with Projected Diffusion Models (NeurIPS 2024)

**[논문: P14]** Christopher 등은 denoising update 뒤에 제약 집합으로의 projection을 반복하는 PDM을 제안했다. 재료 구조, 사람 동작, 경로 계획, 물리적 운동 등에서 평가했다. 원문의 projection은

$$
\Pi_{\mathcal C}(x)=\arg\min_{y\in\mathcal C}\|y-x\|_2^2
$$

이고, sampling update의 개략형은

$$
x^+=\Pi_{\mathcal C}\left(x+\gamma s_\theta(x,t)+\sqrt{2\gamma}\epsilon\right)
$$

이다. convex set에 대한 수렴 논의와 nonconvex 실험을 구별해야 한다.

**[해석/적용]** 화학적 유효 집합은 원자 identity·bond order·전하가 바뀌는 이산적인 여러 영역의 합집합이다. 따라서 거리 제약을 convex projection하는 보장을 aromatic molecular graph 전체로 가져올 수 없다. 현재의 final MILP는 이산 상태에 대한 전역 제약 선택이며, 모든 sampling step에 적용되는 Euclidean projection과도 다르다.

실용적인 전이는 late-stage endpoint에 제한적으로 공동 보정을 적용하는 것이다. 아직 MASK가 많은 중간 $G_t$에 완성 분자 규칙을 그대로 강제하면 학습한 corruption 과정과 충돌할 수 있다. 먼저 final decoder의 수치 support를 고친 뒤, 최종 한 번과 마지막 몇 단계의 제한적 보정을 비교해야 한다. 매 step MILP는 계산량·mode concentration·수정 누적을 동시에 늘릴 수 있다.

### 7.2 DPS — Diffusion Posterior Sampling for General Noisy Inverse Problems (ICLR 2023)

**[논문: P15]** Chung 등은 noisy inverse problem에서 denoiser의 clean estimate를 사용해 measurement likelihood를 근사한다. 핵심은

$$
p(y\mid x_t)\approx p(y\mid\widehat x_0(x_t))
$$

이며, 이를 미분하여 sampling에 반영한다. 엄격한 measurement projection을 매번 강제하는 방법과 달리 noise를 고려한 soft correction이다. 정확한 posterior 계산이나 모든 제약의 hard satisfaction을 보장하는 방법으로 설명하면 안 된다.

**[해석/적용]** 분자에서는 거리·각도·평면성·비결합 충돌 같이 미분 가능한 잔차를 clean endpoint estimate에서 계산하는 원리로 전이할 수 있다. 예를 들어

$$
E_{geom}=\lambda_b E_{bond}+\lambda_a E_{angle}
+\lambda_p E_{plane}+\lambda_c E_{clash}
$$

의 gradient를 좌표에 적용하는 것은 가능하다. 그러나 RDKit의 discrete aromatic 판정을 그대로 좌표 gradient로 바꿀 수는 없다. topology가 잘못되어 있다면 좌표를 평탄화해도 올바른 π 전자계가 되지 않는다. 또한 현재 모델은 flow endpoint parameterization이므로 DDPM의 Tweedie 식을 그대로 복사하지 않고 해당 경로에 맞는 유도와 scale 검증이 필요하다.

### 7.3 DIFUSCO — Graph-based Diffusion Solvers for Combinatorial Optimization (NeurIPS 2023)

**[논문: P16]** Sun과 Yang은 TSP의 edge 선택이나 MIS의 node 선택을 discrete diffusion으로 학습한다. sample이 자동으로 feasible하지 않으므로 최종 score heatmap을 유지하고 task-specific decoding을 적용한다. hard quantization이 상대 confidence를 잃게 한다는 이유를 명시한다. TSP에서는 greedy decoding과 선택적인 2-opt 또는 MCTS를 사용하며, 이는 일반적인 정확한 MILP 해법과 다르다.

**[해석/적용]** 현재 코드의 문제에 가장 직접적으로 연결되는 외부 사례다. 올바른 전역 조합을 고르려면 첫 번째 선택뿐 아니라 **차선 후보의 상대 비용**이 필요하다. 최종 확률이 one-hot에 가깝게 붕괴해 support까지 0이 되면 solver가 아무리 강해도 대안을 고를 수 없다.

다만 heatmap을 보존했다는 것과 그 score가 화학 복구용으로 잘 학습됐다는 것은 다른 문제다. 현 모델의 visible head supervision 문제를 동시에 점검해야 한다. 또한 DIFUSCO는 고정된 문제 instance에서 좋은 feasible solution을 찾는 연구다. 분자 다양성을 보존하며 분포를 샘플링하는 목표는 하나의 최적 tour를 찾는 목적과 다르다. molecule마다 deterministic MAP 하나만 얻으면 diversity가 줄어드는지 별도로 확인한다.

### 7.4 Riemannian Flow Matching — Flow Matching on General Geometries (ICLR 2024)

**[논문: P17]** Chen과 Lipman은 Euclidean space 밖의 다양체에서 flow matching을 구성한다. vector field는 tangent space에 놓이고 학습 오차도 해당 metric으로 측정한다. 단순 기하에서는 geodesic을 이용한 조건부 경로를 구성하며 sphere·torus·mesh 등에서 평가했다.

$$
v_\theta(x,t)\in T_x\mathcal M,\qquad
\mathcal L=\mathbb E\|v_\theta(x_t,t)-u_t(x_t\mid x_1)\|_{g}^{2}.
$$

**[해석/적용]** 유효한 fragment 내부를 고정하고 위치 $r\in\mathbb R^3$, orientation $R\in SO(3)$, torsion $\phi\in S^1$ 등을 생성하는 구조에 자연스럽다. 링의 원자 좌표를 각각 움직이는 대신 허용되는 기하 자유도를 생성하므로 국소 왜곡을 줄일 수 있다.

하지만 모든 가능한 molecular graph를 하나의 매끄러운 고정 차원 manifold로 볼 수는 없다. 원자 수 변화, atom type, fusion topology는 이산 변수다. 실제 설계는 discrete graph 선택과 graph별 continuous geometry 공간을 결합해야 한다. 또한 normal vector 하나는 방향이 지정된 비대칭 heterocycle의 in-plane rotation을 완전히 표현하지 못한다. frame 또는 명시적인 내부 기준축과 symmetry 처리가 필요하다. 이것은 현재 저장소의 ring-level prototype을 검토할 때도 유용한 설계 점검 항목이며, 그 prototype 코드의 추가 감사 결과를 뜻하지 않는다.

## 8. 논문과 현재 코드를 연결하는 수학적 분석

이 절은 특정 논문의 결과를 재현한 것이 아니라, 현재 구현에서 무엇이 성립하고 무엇이 성립하지 않는지 설명하기 위한 독립적인 분석이다.

### 8.1 Joint model과 joint feasible sample은 다르다

공유 GNN으로 모든 원자·결합을 예측하면 graph context를 통해 의존성을 학습한다. 반복 생성 전체가 원자별 완전 독립이라는 뜻은 아니다. 하지만 한 시점의 출력 head를

$$
\widetilde p(S,B\mid G_t,y)=\prod_i p_i(S_i\mid G_t,y)
\prod_{i<j}p_{ij}(B_{ij}\mid G_t,y)
$$

처럼 사용하면, 해당 곱분포의 sample이 모든 제약을 동시에 만족한다는 보장은 없다. 현 decoder가 최적화하는 것도 이런 endpoint head score의 합이다. 이를 전체 생성 trajectory를 적분한 정확한 $\log p_\theta(G\mid y)$와 동일시하지 않는다.

### 8.2 현재 방향족 보조 loss의 의미와 반례

현재 구현의 핵심을 상수·reduction을 생략해 쓰면

$$
\mathcal L_{arom}\sim
-\sum_{e\in A_{true}}\log[p_e(1)+p_e(2)]
+\sum_i\left(\sum_{e\in A_{true}:i\in e}p_e(2)-d_i^{true}\right)^2.
$$

여기서 $A_{true}$는 학습 정답의 방향족 결합 mask이고, $d_i^{true}$는 정답 Kekulé 표현에서 원자 $i$에 연결된 aromatic double bond 수다. 실제 코드에서는 MASK 위치의 예측과 visible 위치의 one-hot을 결합한 effective probability와 graph weighting을 사용한다. [C04]

**간단한 반례:** 여섯 탄소의 고리와 각 C에 H 하나를 가정한다. 고리 결합이 모두 MASK인 입력에서 각 결합의 SINGLE/DOUBLE 예측 확률을 각각 0.5로 둔다. 각 원자에 incident double bond의 기댓값은 1이므로 위 aromatic 항은 0이 될 수 있다. 그러나 독립적인 six-edge categorical draw의 64개 배치 중 모든 탄소의 heavy valence가 3인 교대 배치는 2개뿐이다.

$$
P(\text{consistent alternating assignment})=\frac{2}{2^6}=\frac1{32}.
$$

이는 모델의 실제 생성 성공률이 3.125%라는 주장이 아니다. 또한 **전체 loss가 0이라는 주장도 아니다.** CE와 아래의 distribution-valence loss는 이 예를 별도로 벌점화한다. 반례는 기대 incident degree를 맞추는 aromatic 항만으로 개별 공동 배치를 보장할 수 없음을 보인다.

### 8.3 현재 distribution-valence loss는 평균보다 강하다

원자 $i$ 주변 결합차수 합의 분포는 head factorization 아래 다음 생성함수로 표현할 수 있다.

$$
F_i(z)=\prod_{j\ne i}\left(\sum_{b=0}^{3}p_{ij}(b)z^b\right),
\quad P_i(v)=[z^v]F_i(z).
$$

state $s$에 허용되는 heavy valence 집합을 $\mathcal V(s)$라 하면

$$
C_i=\sum_s p_i(s)\sum_{v\in\mathcal V(s)}P_i(v),
\qquad \mathcal L_{val}\sim-\sum_i\log(C_i+\epsilon).
$$

실제 구현은 convolution과 overflow bin을 사용해 초과 확률을 버리지 않는다. 총전하 분포도 정수 charge 확률을 convolution하여 target net charge의 확률을 계산한다. [C05]

앞선 benzene toy에서 한 원자의 두 incident bond 합이 3일 확률은 0.5다. 따라서 local compatibility loss는 $-\log 0.5$로 양수다. 현재 모델을 “원자가 평균만 맞추는 모델”로 설명하면 틀린다.

그럼에도 local $C_i$들은 서로 edge를 공유하므로 그 곱이 분자 전체 유효 확률은 아니다. finite loss weight와 불완전한 학습 아래 모든 endpoint가 제약을 만족한다는 보장은 남지 않는다. active time 구간과 visible/masked 비율에 따라 gradient가 들어가는 범위도 달라진다.

### 8.4 전역 Kekulé assignment의 정수계획 관점

고정된 aromatic candidate subgraph에서 각 edge를 single 기반으로 놓고 추가 double 여부를 $x_e\in\{0,1\}$로 쓰면, 각 원자의 residual double demand를 $d_i$로 두어

$$
\sum_{e\ni i}x_e=d_i
$$

를 동시에 만족시키는 matching 또는 b-matching 형태가 된다. 단순 탄소 고리에서는 $d_i=1$인 경우가 익숙하지만, heteroatom·H·charge·exocyclic 결합에 따라 demand가 달라진다. 일반적인 aromatic 판정 전체를 이 식 하나로 대체하지는 않는다.

융합 고리는 공유 원자와 결합을 **한 번만** 변수로 두어야 한다. 고리마다 따로 alternate pattern을 고른 뒤 합치면 공유부가 서로 모순될 수 있다. 현재 MILP는 전체 graph의 state와 bond 변수를 공유하므로 이 방향은 이미 반영돼 있다. 추가로 필요한 것은 생성 단계에서 어떤 고리계를 aromatic으로 요구할 것인지와 해당 assignment 후보가 support에 살아 있는지다.

### 8.5 Temperature와 수치적 support

현재 CFG=1의 해당 경로는 허용 class에 대하여 대략

$$
q_k=\frac{\max(p_k,10^{-9})^{1/T}}{\sum_l\max(p_l,10^{-9})^{1/T}}
$$

를 계산한다. $T=0.05$이면 지수가 20이다. $p=(0.999,0.001)$에서 작은 성분은 float64에서 약 $1.0202\times10^{-60}$이나 float32에서는 0이 된다. decoder가 `p > 0`을 hard support로 사용하면 feasible set이 수치 정밀도에 의존하게 된다. [C06], [C07]

**해결 원리:** 의도적인 allowed-class mask $m_k$와 비용 $c_k$를 분리한다.

$$
c_k=-\log p_k\quad\text{또는 logit 기반 동등 비용},
\qquad k\text{를 허용할지 여부는 }m_k\text{로 결정}.
$$

이미 0이 되어 저장된 확률에 `log`를 취하거나 float64로 캐스팅해도 원래 정보를 복구하지 못한다. 모든 0에 epsilon을 더하는 방법은 원래 금지한 state까지 되살릴 위험이 있다. 최초 softmax 전에 logits와 명시적 support mask를 보존하는 것이 더 명확하다. linear-CFG를 쓸 때는 simplex 처리의 0, unsupported mask의 0, floating-point underflow의 0도 구분한다.

### 8.6 고정 점수의 MAP에서 uniform temperature가 하는 일

고정된 head probability $p$, 동일한 $T>0$, 동일한 feasible set을 가정하면

$$
-\sum_j\log q_j(x_j)
=-\frac1T\sum_j\log p_j(x_j)+\sum_j\log Z_j(T).
$$

각 $Z_j(T)$는 선택된 class와 무관하므로 정확한 산술에서 argmin은 동일하다. 따라서 **같은 고정 endpoint 점수로 MAP만 계산할 때** 모든 항에 공통 temperature를 바꾸는 것 자체는 최적 조합을 바꾸지 않는다.

현실에서는 sampling trajectory가 달라지고, modality별 temperature가 다를 수 있고, underflow·clipping·추가 geometry/edit 비용이 있어 결과가 달라질 수 있다. 그래서 “sampler temperature 비교”와 “decoder numerical support 수정”은 서로 다른 실험이어야 한다. 후보 보존의 효과를 보여주려면 가능한 한 동일 endpoint logits를 사용해야 한다.

### 8.7 기하 평면성과 전자구조 방향족성은 별도 조건이다

고정된 고리 원자 집합 $R$에 대한 plane residual은

$$
E_{plane}(R)=\min_{\|n\|=1}\frac1{|R|}\sum_{i\in R}[n^\top(x_i-\bar x_R)]^2
$$

로 정의할 수 있다. centered coordinate covariance의 최소 고유값과 연결되며 SVD로 계산할 수 있다. 그러나 carbon·nitrogen state와 결합차수가 틀리면 이 값을 0으로 만들어도 aromatic graph가 되지 않는다. 융합계 전체를 하나의 plane에 강제할지, 개별 aromatic circuit에 적용할지도 구분해야 한다. 응용 영역에 따라 본질적으로 비평면적인 conjugated 구조가 있으므로 지나친 hard planarity는 학습·탐색 공간을 부당하게 줄일 수 있다.

## 9. 현재 코드와 논문의 대응: 확인된 문제와 남은 가설

### 9.1 구현 대응표

| 항목 | 코드 위치 | 현재 의미 | 아직 보장하지 않는 것 |
|---|---|---|---|
| Joint state codec | `molguidance_hp1kg_baseline/convert_hp1kg_joint_kekule.py` | 원소·H·charge를 공동 token으로 처리 | incident bond-order와의 공동 선택 |
| Kekulé augmentation | `data_processing/kekule_augmentation.py` | 정답 graph identity를 유지한 대체 배정 | 새 scaffold 생성, unknown raw graph 복구 |
| Aromatic auxiliary | `models/chemistry_auxiliary.py` | 정답 aromatic 영역에 대한 soft consistency | 임의 생성 고리계의 aromatic 보존 |
| Valence distribution | `models/chemistry_distribution.py` | local bond-order sum 분포와 state 호환성 | 모든 원자의 동시 유효 sample |
| Correction | `models/correction.py` | 교란된 late input의 원래 endpoint 복원 기능 | 현재 설정에서 visible categorical 오류 복원 학습 |
| Guided sampler | `models/classifier_free_guidance.py` | 실제 sampling과 endpoint score 저장 | 저장된 score의 보정용 calibration |
| Final decoder | `analysis/joint_decode.py` | joint discrete constraints와 최종 전체 검사 | 3D geometry 비용, 원본 identity 유지 |
| Geometry screen | `analysis/strict_generation.py` | 자체 기하 screening | PoseBusters 전체 또는 양자화학적 안정성 |

상대 경로는 codec을 제외하면 `MolGuidance_guidance_selector_e987/molguidance/` 기준이다. pinned 링크는 14절에 있다.

### 9.2 확인된 문제 A: 수치적 0을 금지 후보로 사용

**[코드]** `ctmc_vector_field.py` 기본 temperature는 0.05다. 실제 generation script는 `dfm_type="campbell"`로 실행하고 temperature를 별도로 넘기지 않는다. `classifier_free_guidance.py`는 temperature 적용 후 확률을 endpoint에 저장한다. `joint_decode.py`는 양의 확률만 변수로 만든다. [C06], [C07], [C08]

**[관측]** raw heavy-heavy 실제 결합 42,688개 중 41,059개, 즉 96.18%는 양의 확률 class가 하나다. 실제 원자 state도 85.26%가 단일 양의 support다. [C10]

**판정:** numerical support 문제가 현재 경로에 남아 있다는 근거가 충분하다. 그러나 모든 0이 underflow라는 뜻은 아니다. forbidden class, probability processing, 이미 뾰족한 예측 등 원인을 구분해야 한다. 기존 파일에는 temperature 전 logits가 없어 저장된 0을 역으로 복구할 수 없다.

**문헌 연결:** DIFUSCO의 confidence 보존 원리, PDM의 feasible projection 전제. projection은 입력 후보 공간이 이미 공집합이면 성공할 수 없다.

### 9.3 확인된 문제 B: FAKE/real count와 hard count의 충돌

각 slot의 support에서 real 선택 가능 여부를 $u_i$, real 선택이 필수인지 여부를 $l_i$라 두면

$$
N_{min}=\sum_i l_i,\qquad N_{max}=\sum_i u_i.
$$

요청한 $N^*$가 이 구간 밖이면 다른 화학 제약을 보기 전에도 infeasible이다. 이 구간 안이라고 해가 존재하는 것은 아니다. valence·charge·connectivity가 추가로 충돌할 수 있다.

**[관측]** 요청 count가 support 범위 밖인 경우 679개, local bond-order/valence 범위 충돌 179개, 둘의 합집합 730개다. 이는 projection infeasible 907개의 80.49%다. 요청 평균 34.224와 실제 평균 36.413도 다르다. [C10]

**판정:** sampling count를 size prior로 사용할지 엄격한 조건으로 사용할지 정해야 한다. actual count를 허용하면 decoder가 쉬워질 수 있지만, 이는 원래 count condition을 만족한 결과와 다른 과제다. 반대로 exact count를 반드시 유지하려면 FAKE 상태의 합에 관한 제약이 sampling 또는 decoding에 일관되게 작동해야 한다.

**문헌 연결:** FlowMol3의 removable atom은 correction 자유도이며 exact cardinality 보장과 동일하지 않다. count를 완화한 성능을 엄격한 count 모델의 향상으로 보고하면 안 된다.

### 9.4 설계상 위험 C: visible head가 보정 확률로 학습되는가?

**[코드]** 주 categorical CE는 MASK 위치에 적용하고 이미 visible한 위치는 ignore한다. chemistry auxiliary에서도 visible 위치에는 예측 head 대신 입력 one-hot을 넣는다. 그러나 endpoint 저장과 final decoding에서는 visible 여부에 관계없이 모든 위치의 head 확률을 사용한다. [C04], [C06], [C09]

**[코드]** correction 모듈은 일부 visible 상태를 잘못된 다른 상태로 바꾸고 이를 감독하는 기능을 갖고 있다. 하지만 현재 config의 `node_rate=0`, `edge_rate=0`이므로 해당 감독 경로에 실제 visible categorical corruption은 들어가지 않는다. [C11]

**가설:** late graph의 visible 위치에 대한 raw head score를 전체 구조 복구의 calibrated probability로 취급하기 어렵다. MASK 예측을 잘하는 모델이 잘못된 visible label을 수정하는 모델과 같지는 않다.

**필요한 구분:** 올바르게 관측된 visible label을 조건으로 보는 경우 그 값은 고정되어야 한다. 반면 생성 오류일 수 있는 visible label을 수정하려면 별도의 불확실성 모델 또는 error distribution이 필요하다. visible을 전부 one-hot으로 고정하면 학습 계약에는 가까워져도 복구할 자유도가 사라진다. 모든 head를 자유롭게 쓰면 복구 자유도는 생기지만 점수의 의미가 불명확해진다.

**검증 방법:** 동일한 오류 graph에서 (a) 현 full-head score, (b) 의심 부위만 remask 후 얻은 score, (c) visible 오류 복원 학습을 거친 score를 비교한다. 문헌의 “self-correction”이라는 명칭만으로 이 차이가 해결됐다고 간주하지 않는다.

### 9.5 확인된 범위 D: current decoder의 전역 검사는 존재한다

현재 solver는 원자별 `(state, allowed heavy valence)`와 pair별 bond order를 공동 선택한다. exactly-one 제약, 총전하, 선택적 원자 수, heavy valence balance, flow 기반 연결성 제약을 갖는다. 정수 후보가 RDKit 검사에 실패하면 해당 assignment를 제외하는 no-good cut을 넣고 시간·round budget 안에서 다시 푼다. [C07]

최종 molecule에는 다음을 확인한다.

- sanitize 성공.
- 선택한 H count·formal charge 보존.
- 하나의 연결 component.
- toolkit상 radical 없음.
- 명시적으로 요청된 aromatic edge가 있다면 aromatic으로 실현됨.
- 전체 분자의 Kekulization 가능.

그러나 production 호출은 `aromatic_edges`를 지정하지 않는다. 이 때문에 생성된 모든 ring을 aromatic으로 강제하지 않으며, 그렇게 해야 한다는 화학적 이유도 없다. 특정 scaffold를 유지하려면 신뢰 가능한 subgraph 또는 별도 조건을 제공해야 한다.

또한 `optimal`은 **현재 score·support·제약 아래**의 최적성이다. 전체 화학 공간에서의 최적성이나 정답 분자 복원을 뜻하지 않는다. time-limited feasible solution과 infeasible 증명, unresolved timeout도 구분해야 한다.

### 9.6 확인된 범위 E: coordinate–graph 공동 보정은 아직 아니다

현재 objective는 node/edge NLL의 합이며 좌표는 선택된 slot에서 그대로 복사한다. solver 내부에 bond-length penalty, angular penalty, planarity penalty, displacement cost, edit budget은 없다. 따라서 비슷한 score의 다른 topology를 선택하더라도 XYZ가 새 topology에 맞는지 목적함수는 알지 못한다. [C07]

**[제안]** 고정된 좌표에 대한 pair-distance compatibility 비용은 이산 decoder에 보조항으로 넣을 수 있다. 그러나 결합거리를 딱 한 값으로 제한하면 공명·결합 환경 차이를 무시하므로 허용 분포·불확실성을 사용해야 한다. H나 atom type을 바꾸는 경우 그에 따라 geometry 비용도 갱신해야 한다.

새 그래프에 대해 좌표를 refine한다면 raw 좌표, graph-only projection 좌표, refine 후 좌표, XTB 후 좌표를 따로 보존한다. 이 단계들을 섞으면 모델이 무엇을 생성했고 후처리가 무엇을 바꿨는지 알 수 없다.

### 9.7 평가상 문제 F: 평면성 정의와 구조 분포

**[코드/관측]** production은 ring의 모든 원자가 aromatic이면 plane 검사에 넣는다. 정정된 재분석은 ring의 모든 결합이 aromatic인 경우로 좁혔다. 후자의 정의로 post-opt warning은 102→89, 계산 대상의 input geometry 통과는 364→373으로 바뀌었다. 이는 평가 교정이며 모델 개선이 아니다. [C10], [C12]

| 구조 지표 | train 1,264 | 계산 완료 생성 452 |
|---|---:|---:|
| 방향족 포함 | 85.92% | 96.02% |
| 3·4원 고리 포함 | 2.37% | 16.81% |
| 7원 이상 고리 포함 | 5.22% | 51.33% |
| 평균 ring atom fraction | 63.68% | 82.98% |
| 고리 밖 heavy atom 수 | 8.847 | 3.836 |
| 5·6원 고리 중 hetero 포함 비율 | 39.50% | 16.58% |

위 결과는 계산 대상이라는 선택 편향을 포함한다. 조건 주변 train을 재가중하면 평균 ring count 차이는 줄어들므로, 전체 train 대비 ring count만으로 조건 무시를 결론 내릴 수 없다. 그러나 고리 밖 원자와 조성 차이는 별도로 남는다. 큰 고리나 작은 고리 자체를 모두 invalid로 분류하는 것은 해결책이 아니다. 452개가 모두 unique·novel이어도 유용한 scaffold 다양성까지 입증되지는 않는다. [C10]

### 9.8 데이터·checkpoint에 대한 추가 해석

train 1,264개는 대규모 공개 benchmark보다 매우 작은 조건부 학습 환경이다. validation loss 최저 epoch를 선택하는 것은 통계적 학습 기준이며, 생성 graph·ring·geometry 품질의 최적 checkpoint를 직접 고른 것은 아니다. 장기학습을 했다는 사실만으로 향후 epoch 연장이 구조 실패를 해결할 근거는 없다.

모델 크기, 데이터 크기, conditioning, support, decoder 변화가 동시에 들어간 결과를 하나의 새 화학 loss 효과로 해석해서도 안 된다. 먼저 inference-only 변경으로 원인을 분리하고, 그 다음 재학습 항목을 하나씩 비교하는 편이 정보 가치가 높다.

## 10. 권장 연구·수정 순서와 비교 실험

아래는 **미실행 제안**이다. 이 문서를 커밋하는 작업은 이들 모델 변경이나 생성 실험을 실행하지 않는다.

### 10.1 첫 단계: 같은 checkpoint에서 분리할 수 있는 문제

| ID | 변경 사항 | 고정할 것 | 주 평가 | 해석 가능한 결론 |
|---|---|---|---|---|
| E0 | 현 raw 생성과 현 projection 재현 | checkpoint, seed, counts, schedule | raw valid·eligible·projection status | 대조군 |
| E1 | pre-temperature log score와 allowed mask를 분리, decoder log-domain 비용 | 가능한 한 동일 endpoint·count 제약 | 후보 수, infeasible 원인, rescue, edits | 수치 support 소실의 영향 |
| E2a | E1에서 actual sampled count 유지 | E1 점수·geometry | feasibility, count error | 원본 count를 유지하는 복구 |
| E2b | E1에서 requested count 강제 | E1 점수·geometry | feasibility, atom additions/deletions | 정확한 count 조건의 비용 |
| E3 | 충돌 부분 remask 후 score 재계산 | 기존 weight, 정의된 재예측 budget | repair·identity·geometry | 현 모델의 조건부 복원 능력 |
| E4 | E1/E3에 geometry compatibility·edit budget 추가 | raw endpoint와 허용 범위 | valid와 geometry 동시 통과율 | 구조 변경의 기하적 비용 |

기존 endpoint 파일의 0을 untemper하여 E1을 만들 수는 없다. logits를 저장하도록 최소한의 instrumentation을 추가한 재생성이 필요하다. 그때 instrumentation이 RNG를 소비하거나 sampling 순서를 바꾸지 않게 하여 raw endpoint가 기준선과 일치하는지 확인한다. 완전 재현이 안 되면 그 원인을 기록하고 paired comparison의 강도를 낮춰 해석한다.

E2a와 E2b는 같은 성공 정의가 아니다. 양쪽을 모두 보고하면 FAKE correction과 exact count의 trade-off를 볼 수 있다. count를 자유롭게 두고 성공률만 높인 뒤 원래 count-conditional task를 해결했다고 쓰지 않는다.

### 10.2 두 번째 단계: 학습과 보정 점수를 일치시키기

| ID | 학습 변화 | 핵심 질문 | 필요한 대조 |
|---|---|---|---|
| T0 | 현 correction 유지 | 현재 geometry-only 수준의 효과 | 동일 데이터·학습 budget |
| T1 | selected visible atom/bond 오류 복원 감독 | 실제 생성 오류를 수정하는가? | 무작위 label flip과 실제 residual 오류의 차이 |
| T2 | late-time graph/geometry corruption 분포 조절 | 너무 강한 교란이 다양성을 줄이는가? | validity·FG·ring·count 함께 비교 |
| T3 | 실제 sampling 주변 상태를 이용한 복원 학습 | off-path robustness가 증가하는가? | target identity·label 일관성 검증 |
| T4 | aromatic component 단위 masked reconstruction | atom-wise보다 ring-system consistency가 좋아지는가? | 같은 가린 원자 수·compute budget |

T1의 `node_rate`나 `edge_rate`를 임의로 올린 뒤 개선을 단정하지 않는다. 균일 random corruption은 실제 오류인 과원자가 C, H/bond 충돌, 잘못된 fusion과 분포가 다를 수 있다. 오류 유형별 curriculum 또는 실제 residual 통계를 이용한 교란은 후보지만, validation/test 결과를 training corruption 설계의 정답 데이터로 섞지 않는다.

T3에서는 두 종류의 학습 예를 구분한다. 유효한 원본 분자를 의도적으로 손상시킨 경우에는 원래 endpoint가 명확하다. 반면 독립적인 de novo sample에는 유일한 정답 분자가 없다. 그 sample에 가까운 임의의 train molecule을 정답으로 붙이면 supervision의 의미가 바뀐다. 동일 부모 분자에서 출발한 교란/rollout을 이용하거나, 별도의 validity-aware objective로 설계해야 한다. 바뀐 graph에 원래 전자구조 라벨을 자동으로 승계해서도 안 된다.

### 10.3 세 번째 단계: 표현 자체를 바꿀 필요가 있는지 판단

| 대안 | 얻는 것 | 잃거나 새로 해결할 것 |
|---|---|---|
| graph-first 후 conformation 생성 | discrete chemistry와 geometry 문제 분리 | 두 모델의 조건 일관성, end-to-end 제어 |
| 고정 aromatic core + linker/치환기 생성 | core의 내부 유효성 보존 | 새 aromatic core 탐색 범위 감소 |
| ring-system token + assembly | 융합 고리의 공유 전자/결합 관계를 큰 단위로 보존 | vocabulary coverage, 큰 motif의 희소성 |
| atom–fragment 다중 해상도 모델 | 작은 단위의 신규성과 큰 단위의 일관성 절충 | representation alignment·fusion 제약·학습 복잡도 |

현재 저장소의 root README에 ring-level prototype이 있어도 그것이 Stage 8의 대체 해법으로 검증된 것은 아니다. 실제 HP1KG 분자를 동일 표현으로 losslessly encode/decode하는지, bridged/fused/spiro 구분이 정확한지, 범위 밖 구조를 어떻게 처리하는지부터 확인해야 한다.

추천 순서는 E1–E2로 기존 decoder가 작동할 조건을 확보하고, E3/T1로 복원 점수를 검증한 뒤, 여전히 고리 구성 편향이 남을 때 표현 변경을 비교하는 것이다. 다만 표현 비교용 데이터 codec 검증은 이와 독립적으로 진행할 수 있다.

### 10.4 통계와 sampling budget

- 기존 1,000개를 기준으로 하되, 시간·GPU 예산이 허용하면 여러 독립 seed의 재현을 계획한다. 특정 반복 수가 이미 실행된 것처럼 기록하지 않는다.
- 같은 noise/count/condition을 쓰는 paired arm에서는 각 분자의 실패→성공·성공→실패 전환을 기록한다.
- raw invalid rescue와 raw valid retention을 별도로 보고한다. “solver 성공률”만으로 복구 품질을 판정하지 않는다.
- confidence interval은 이항 비율과 seed 간 변동을 구분한다. 1,000개에서 얻은 표본오차가 dataset·checkpoint 변동을 대표하지는 않는다.
- molecule size 구간별로 나눈다. 현재 큰 분자에서의 생존 편향이 강하므로 전체 평균만 비교하면 안 된다.
- 여러 후보 중 best-of-K를 고르면 K와 전체 시도 수를 분모에 포함하고, wall time·solver 호출 수·NN evaluation 수도 남긴다.
- 조건 제어 실험은 같은 count 정책 아래 CFG=0·CFG=1·강한 guidance 및 다른 target을 비교해야 한다. 한 target·한 seed만으로 guidance의 인과효과를 결론 내리지 않는다.

## 11. 구현을 위한 구체적 설계안

### 11.1 Endpoint score 저장 계약

권장 저장 필드는 다음과 같다. 모두 신규 제안이며 현재 저장 포맷에 이미 있다는 뜻은 아니다.

| 필드 | 용도 |
|---|---|
| `raw_node_logits`, `raw_edge_logits` | softmax 이전 정보 보존 |
| `allowed_node_mask`, `allowed_edge_mask` | 의도적 금지 class와 수치적 0 분리 |
| `guided_log_scores` | 선택한 CFG 방식에 맞는 decoder 비용 |
| `sampling_probs` | 실제 CTMC가 사용한 확률 재현 |
| `visible_mask_at_scoring` | 해당 head 점수의 학습 범위 판단 |
| `raw_states`, `raw_bonds`, `raw_xyz` | sampling 결과 원본 |
| `requested_count`, `sampled_real_count`, `slot_count` | 세 가지 count 의미 구분 |
| `temperature`, `guidance_mode`, `guidance_scales` | 재현과 비용 해석 |
| `checkpoint_hash`, `source_commit`, `schema_version` | 결과–모델–코드 연결 |

CFG=1에서는 조건부 logits와 support mask를 직접 쓰는 경로가 가장 명확한 비교 기준이다. 일반 $w\ne1$에서 linear-probability guidance를 log score로 옮길 때는 수학적으로 다른 sampler가 되지 않게 해야 한다. sampler와 decoder의 점수를 의도적으로 다르게 쓰면 그 사실을 별도 옵션으로 기록한다.

### 11.2 Decoder의 성공·실패 계약

결과 status를 최소한 다음과 같이 분리한다.

- `optimal`: 정의된 surrogate score와 제약에서 최적성이 확인된 해.
- `feasible_time_limit`: 시간제한 내 검증된 feasible 해, 최적성 미확인.
- `infeasible`: 해당 support·constraints 아래 해 없음이 확인됨.
- `unresolved`: timeout, cut budget 소진 등으로 결론 미확정.
- `invalid_input` 또는 `exception`: 입력·수치·시스템 오류.

현재 decoder도 optimal/feasible/infeasible/unresolved를 구별하는 부분이 있다. 개선의 목적은 이를 생성 평가 전체에 일관되게 전달하고 count/valence/support 원인을 더 세분화하는 것이다. timeout을 화학적 infeasibility로 집계하지 않는다. 실패 시 원본으로 fallback한다면 `projection_success=false`를 유지하고 raw validity와 혼동하지 않는다.

### 11.3 제한적 공동 보정의 목적함수

향후 보정 문제는 다음처럼 구조화할 수 있다.

$$
\min_{S,B,X'}\left[
\sum_i c_i(S_i)+\sum_{i<j}c_{ij}(B_{ij})
+\lambda_{edit}d_{edit}((S,B),(S^{raw},B^{raw}))
+\lambda_x\|X'-X^{raw}\|_F^2
+\lambda_g E_{geom}(S,B,X')\right]
$$

$$
\text{subject to }(S,B)\in\mathcal C_{chem},\quad
N_{real}=N^*\ \text{또는 명시적인 별도 count 정책}.
$$

이는 현 MILP에 그대로 들어가는 선형식이 아니다. $X'$와 geometry 항까지 최적화하면 mixed discrete–continuous, 일반적으로 nonconvex 문제가 된다. 실용적으로는 다음처럼 분리할 수 있다.

1. 고정 $X^{raw}$에서 log-score·수정량·거리 호환성을 고려한 graph 후보 생성.
2. 각 후보의 전체 chemical validation.
3. 고정 graph에서 제한된 coordinate refinement.
4. graph identity·H·charge와 기하 재검증.
5. 여러 후보 중 정의한 기준으로 선택하되 전체 후보 budget 기록.

원본 XYZ에 과도하게 묶으면 원래 좌표 오류를 복구하지 못하고, 너무 자유롭게 움직이면 3D 생성 모델의 기여가 희미해진다. $\lambda_x$와 edit budget을 변화시키는 trade-off 곡선이 단일 성공률보다 유용하다. identity-preserving 모드와 chemistry-valid alternative 모드를 명시적으로 구분해야 한다.

### 11.4 Aromatic component 복원 시 주의점

신뢰할 수 있는 aromatic scaffold가 조건으로 주어지면 해당 subgraph의 atom identity·H·charge·attachment를 보존하는 global assignment를 요구할 수 있다. 반면 완전히 새로운 분자에서는 정답 aromatic edges를 알 수 없다. 다음 세 방식을 구별한다.

- **조건 보존:** 주어진 valid scaffold의 aromaticity를 유지.
- **후보 추론:** 생성 graph에서 candidate conjugated component를 찾아 여러 state/assignment를 공동 검토.
- **motif 생성:** aromatic ring system 자체를 discrete 단위로 생성.

후보 추론에서 단순히 가까운 원자를 고리로 연결하거나 모든 탄소 고리를 aromatic으로 강제하면 false positive를 늘릴 수 있다. heteroatom의 전자 기여, exocyclic bond, charge/H 선택을 함께 다루고 최종 전체 graph를 sanitize한다. 국소 완벽 matching의 존재만으로 물리적 aromaticity를 증명했다고 쓰지 않는다.

## 12. 평가와 재현을 위한 체크리스트

### 12.1 네 단계의 출력 보존

| 단계 | 보존할 자료 | 목적 |
|---|---|---|
| Raw | state·bond·heavy XYZ·score·count | 생성 모델 자체의 출력 평가 |
| Graph-corrected | graph, 변경 원자/결합 목록, 원본 slot mapping | 어떤 화학 구조가 바뀌었는지 추적 |
| Geometry-refined | 동일 graph의 XYZ, displacement·energy | 기하 보정의 기여 분리 |
| Quantum-relaxed | 입력·출력 좌표·로그·WBO·전자상태 | 물리 평가와 identity 변화 검토 |

원자 삭제·추가가 있으면 단순 atom-index RMSD는 정의가 달라진다. 공통 heavy atom mapping과 retained subset을 명시해야 한다. fragment 재조합 후 새 XYZ를 생성했다면 동일 molecule의 작은 relaxation처럼 표시하지 않는다.

### 12.2 반드시 따로 보고할 지표

1. 전체 시도 대비 raw validity, connected validity, neutral/radical-free 비율.
2. raw invalid 중 보정 성공률과 raw valid 중 유지율.
3. exact requested count 일치율, actual count 변화량, FAKE 선택 통계.
4. node/edge edit 수, 원소·H·charge 변화, aromatic subgraph 보존율.
5. raw/refined/post-opt 기하 통과율과 각 실패 유형.
6. ring size뿐 아니라 ring-system size, fused/spiro/bridged 분류, non-ring atom 비율.
7. 유효 집단에서 scaffold 빈도, effective scaffold count, pairwise similarity, hetero 위치 다양성.
8. 동일 계산 수준에서 target fidelity와 그 분모. 실패·제외도 전체 시도에 포함해 따로 보고.
9. inference 시간, solver 시간, timeout율, 후보 수, NN evaluation 횟수.
10. train·validation·test identity 분리, 조건·size가 맞는 reference와의 비교.

effective scaffold count는 scaffold 빈도 $p_k$에 대해 $\exp[-\sum_k p_k\log p_k]$ 같은 entropy 기반 지표를 사용할 수 있다. 이는 새로 제안하는 평가 예시이며 현재 리포트에 이미 계산돼 있다는 뜻은 아니다.

### 12.3 회귀 테스트가 다뤄야 할 반례

| 시험 | 검증 목표 |
|---|---|
| 작은 양의 후보가 float32에서 사라지는 입력 | numerical zero와 forbidden support 분리 |
| requested count가 feasible support 밖인 입력 | 즉시 설명 가능한 infeasible 판정 |
| valid raw인데 다른 head argmax를 가진 입력 | raw retention과 재선택 의미 확인 |
| benzene·pyrrole·pyridine·융합계 | heteroatom H·charge와 shared-atom consistency |
| nitro처럼 formal charge를 가진 중성 구조 | 형식전하 존재를 비정상으로 오판하지 않음 |
| aromatic atoms 사이 non-aromatic bond를 포함한 cycle | plane 평가 대상 정의 확인 |
| graph는 valid이지만 좌표가 충돌하는 입력 | graph와 geometry pass 분리 |
| unknown aromatic scaffold의 de novo 입력 | oracle aromatic edge 테스트와 실제 사용 구분 |
| solver timeout과 proven infeasible | status 혼동 방지 |
| 여러 Kekulé 표현의 동일 identity | augmentation이 화학종을 바꾸지 않음 |

이미 있는 테스트를 무조건 새로 중복 작성하자는 제안은 아니다. 현재 130개 중 해당 계약을 다루는 것을 재사용하고, 실제 수정으로 바뀌는 경계 조건만 추가해야 한다. 전체 unit test 통과는 생성 분포의 화학적 품질에 대한 증명이 아니므로 end-to-end sample evaluation이 별도로 필요하다.

### 12.4 이번 문서 작성에서 직접 확인한 것

- 원격 HEAD `06d5fdf`와 부모 `c8af84d` 확인.
- 활성 package·config·sampling·decoder·test·기존 보고서 정적 대조.
- 커밋된 CSV 1,000행의 status, raw valid 574, eligible 452, count 일치 119 재집계.
- NumPy로 temperature 수식의 float32/float64 차이 재현.
- six-edge toy의 일관된 교대 배치 2/64 직접 열거.
- 저장된 `unittest.log`의 130개 통과 기록 확인.

이 문서 작성 환경에는 PyTorch·DGL·RDKit이 설치돼 있지 않아 모델 unit test, 실제 RDKit 복구, checkpoint inference, XTB를 새로 실행하지 않았다. checkpoint weight와 원본 endpoint tensor·전체 양자화학 로그는 이 검토에서 내려받아 재현한 자료가 아니다. 기존 보고서의 64개 source hash 일치 및 904개 로그 재확인은 **기존 검증 기록**으로 인용하며 이번 실행의 성과로 바꾸어 쓰지 않는다.

### 12.5 독립적으로 재현할 수 있는 최소 수치 예

아래 코드는 논문의 benchmark를 재현하는 코드가 아니라 8절의 수치·조합 반례를 확인하는 예다. NumPy만 필요하다.

```python
import itertools
import numpy as np

for dtype in (np.float32, np.float64):
    p = np.array([0.999, 0.001], dtype=dtype)
    logits = np.log(np.maximum(p, dtype(1e-9))) / dtype(0.05)
    q = np.exp(logits - logits.max())
    q /= q.sum()
    print(dtype.__name__, q.tolist())

assignments = list(itertools.product((1, 2), repeat=6))
consistent = [
    b for b in assignments
    if all(b[i] + b[(i + 1) % 6] == 3 for i in range(6))
]
print(len(consistent), len(assignments))  # 2, 64
print((2 * 0.5 - 1) ** 2)              # aromatic degree term: 0
print(-np.log(0.5))                    # local valence loss: positive
```

## 13. 논문·공식 문서 목록과 확인 위치

아래 링크는 원문 접근점이다. 유료 또는 접근 제한이 있는 출판사 페이지는 공개 preprint를 함께 표시했다. 이 문서의 분석은 특정 원문의 전재가 아니라 방법의 요약·비교와 현재 코드에 대한 별도 분석이다.

| ID | 문헌 | 원문·확인 위치 | 이 보고서에서의 역할 |
|---|---|---|---|
| [P01] | Hoogeboom, E.; Satorras, V. G.; Vignac, C.; Welling, M. **Equivariant Diffusion for Molecule Generation in 3D.** ICML/PMLR 162, 8867–8887 (2022). | [학회 페이지](https://proceedings.mlr.press/v162/hoogeboom22a.html), [PDF](https://proceedings.mlr.press/v162/hoogeboom22a/hoogeboom22a.pdf). Abstract·Introduction과 생성 표현 확인. | 좌표 중심 equivariant generation의 기준 |
| [P02] | Xu, M.; Yu, L.; Song, Y.; Shi, C.; Ermon, S.; Tang, J. **GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation.** ICLR (2022). | [공개 원문](https://arxiv.org/abs/2203.02923), [PDF](https://arxiv.org/pdf/2203.02923). Abstract·Introduction의 graph-conditioned task 확인. | graph 생성과 conformer 생성 구분 |
| [P03] | Peng, X.; Guan, J.; Liu, Q.; Ma, J. **MolDiff: Addressing the Atom–Bond Inconsistency Problem in 3D Molecule Diffusion Generation.** ICML/PMLR 202, 27611–27629 (2023). | [학회 페이지](https://proceedings.mlr.press/v202/peng23b.html), [PDF](https://proceedings.mlr.press/v202/peng23b/peng23b.pdf). §3.1, §3.3–3.4, Table 1. | 공동 생성, noise schedule, bond-predictor guidance |
| [P04] | Vignac, C.; Osman, N.; Toni, L.; Frossard, P. **MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation.** (2023). | [arXiv:2302.09048](https://arxiv.org/abs/2302.09048), [검토 PDF](https://arxiv.org/pdf/2302.09048v2). Introduction·§5.3·Table 3. | Gaussian/discrete 결합, formal charge, 지표 해석 |
| [P05] | Dunn, I.; Koes, D. R. **FlowMol3: Flow Matching for 3D De Novo Small-Molecule Generation.** Digital Discovery 5, 2052–2066 (2026). DOI: **10.1039/D5DD00363F**. | [출판본](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00363f), [2025 v1](https://arxiv.org/pdf/2508.12629v1). 출판본 §2.6.3·§4·Tables 1–2와 v1 대조. | self-correction과 validity–diversity trade-off |
| [P06] | Zeng, C.; Jin, J.; Ambrose, C.; et al. **PropMolFlow: Property-guided Molecule Generation with Geometry-Complete Flow Matching.** Nature Computational Science 6, 233–242 (2026). DOI: **10.1038/s43588-025-00946-y**. | [출판사](https://www.nature.com/articles/s43588-025-00946-y), [arXiv v4](https://arxiv.org/pdf/2505.21469v4). Overview·property evaluation·closed-shell 평가. | 현재 코드 계보와 물성 검증 |
| [P07] | Jin, J.; Zeng, C.; et al. **MolGuidance: Advanced Guidance Strategies for Conditional Molecular Generation with Flow Matching.** arXiv:2512.12198v1 (2025). | [공개 원문](https://arxiv.org/abs/2512.12198), [v1 PDF](https://arxiv.org/pdf/2512.12198v1). Introduction·guidance 비교 및 modalities 설명. | guidance의 역할과 화학 validity의 구분 |
| [P08] | Nikitin, F.; Dunn, I.; Koes, D. R.; Isayev, O. **GEOM-drugs Revisited: Toward More Chemically Accurate Benchmarks for 3D Molecule Generation.** Digital Discovery 4, 3282–3291 (2025). DOI: **10.1039/D5DD00206K**. | [출판사](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00206k), [공개 v2](https://arxiv.org/pdf/2505.00169v2). Molecule Stability·Identified Issues·Solution. | aromatic valence와 benchmark 재검토 |
| [P09] | Jin, W.; Barzilay, R.; Jaakkola, T. **Junction Tree Variational Autoencoder for Molecular Graph Generation.** ICML/PMLR 80, 2323–2332 (2018). | [학회 페이지](https://proceedings.mlr.press/v80/jin18a.html), [PDF](https://proceedings.mlr.press/v80/jin18a/jin18a.pdf). §2.1 및 tree/graph 생성 설명. | 유효 motif와 assembly |
| [P10] | Zhang, O.; Huang, Y.; et al. **FragGen: Towards 3D Geometry Reliable Fragment-based Molecular Generation.** Chemical Science 15, 19452–19465 (2024). DOI: **10.1039/D4SC04620J**. | [출판사](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc04620j), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11575641/). Combined geometry strategy·Methods·Table 2. | fragment 내부와 연결 자유도의 분리 |
| [P11] | Igashov, I.; Stärk, H.; Vignac, C.; et al. **Equivariant 3D-conditional Diffusion Model for Molecular Linker Design.** Nature Machine Intelligence 6, 417–427 (2024). DOI: **10.1038/s42256-024-00815-9**. | [출판사](https://www.nature.com/articles/s42256-024-00815-9), [공개 preprint](https://arxiv.org/abs/2210.05274). 출판본 Main·Fig. 1 설명 확인. | fragment-conditioned linker 생성 |
| [P12] | Ragoza, M.; Masuda, T.; Koes, D. R. **Generating 3D Molecules Conditional on Receptor Binding Sites with Deep Generative Models.** Chemical Science 13, 2701–2713 (2022). DOI: **10.1039/D1SC05976A**. | [출판사](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05976a), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8890264/). CVAE·density-to-molecule workflow 확인. | 생성 표현과 discrete reconstruction 분리 |
| [P13] | Buttenschoen, M.; Morris, G. M.; Deane, C. M. **PoseBusters: AI-based Docking Methods Fail to Generate Physically Valid Poses or Generalise to Novel Sequences.** Chemical Science 15, 3130–3139 (2024). DOI: **10.1039/D3SC04185A**. | [출판사](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc04185a), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/). Abstract·Introduction·평가 목적 확인. | geometry screening의 필요성과 범위 |
| [P14] | Christopher, J. K.; Baek, S.; Fioretto, F. **Constrained Synthesis with Projected Diffusion Models.** NeurIPS (2024). | [공개 원문 v3](https://arxiv.org/html/2402.03559v3), [학회 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/a28af221f2f70be183afc16797a56b91-Paper-Conference.pdf). §4.2, Eqs. 7–8, convex 조건. | 단계별 projection과 보장 범위 |
| [P15] | Chung, H.; Kim, J.; McCann, M. T.; Klasky, M. L.; Ye, J. C. **Diffusion Posterior Sampling for General Noisy Inverse Problems.** ICLR (2023). | [학회 원문](https://openreview.net/pdf?id=OnD9zGAGT0k), [공개 v4](https://arxiv.org/html/2209.14687v4). §3.1, Eqs. 7–11. | clean estimate에서 soft constraint 유도 |
| [P16] | Sun, Z.; Yang, Y. **DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization.** NeurIPS (2023). | [학회 페이지](https://papers.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/0ba520d93c3df592c83a611961314c98-Paper-Conference.pdf). §3.5 Decoding Strategies. | score 보존과 feasible decoding |
| [P17] | Chen, R. T. Q.; Lipman, Y. **Flow Matching on General Geometries.** ICLR (2024). | [학회 페이지](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d1f9936d3be6997ffffab692977eebe6-Abstract-Conference.html), [공개 v3](https://arxiv.org/html/2302.03660v3). §2–3, manifold·tangent field 정의. | rigid frame·torsion 상태공간의 설계 |
| [D01] | **The RDKit Book — Aromaticity.** 공식 문서. | [문서](https://www.rdkit.org/docs/RDKit_Book.html#aromaticity). 확인 페이지는 2026.09.1 문서. | aromatic atom/bond 구분과 fused-system 규칙 |

RDKit 문서의 현재 버전과 프로젝트 실행 환경의 RDKit 버전은 같다고 가정하지 않는다. 실제 재현 실험에서는 기존 environment와 aromaticity model을 고정하고, 버전 변경은 별도 비교한다.

### 13.1 성능 수치를 읽을 때의 정정표

| 피해야 할 서술 | 정확한 서술 |
|---|---|
| MiDi validity가 91.6%다 | 원문 Table 3의 91.6%는 molecular stability이며 validity는 77.8% |
| FlowMol3는 2025년 논문이고 PB-valid는 91.9%다 | v1 preprint와 2026 출판본을 구별. 출판본은 95.9% |
| FlowMol3의 100%는 모든 미래 sample에 대한 보장이다 | 보고된 유한 표본에서 반올림된 실험 수치이며 보편적 정리는 아님 |
| FragGen은 RMSD도 항상 최저다 | 원문 Table 2에서 일부 atom-wise 방법의 OptRMSD가 더 낮음 |
| GeoDiff가 원자가·방향족 그래프를 새로 만들어 해결했다 | 주어진 graph에서 conformation을 생성하는 과제 |
| JT-VAE·LiGAN·FragGen은 모두 diffusion 모델이다 | 각각 VAE 기반 graph assembly, density CVAE, fragment-wise generation 계열로 구분 |
| PDM 보장이 임의의 화학 제약 전체에 바로 적용된다 | projection 계산 가능성·제약 형태·수렴 가정과 discrete chemistry를 구별 |
| RDKit sanitize 성공이면 안정한 singlet 분자다 | toolkit 표현 검사와 실제 전자구조 안정성은 다른 수준 |
| H-only 수정 가능한 수를 raw 성공률에 더할 수 있다 | 반사실 진단이며 실제 신규 생성·재계산 성공이 아님 |

## 14. 코드·기존 리포트 근거와 기록 원칙

아래 코드는 `c8af84d`에 고정하고, 기존 생성 리포트는 `06d5fdf`에 고정했다. 이후 branch가 바뀌어도 이 문서가 어떤 상태를 분석했는지 추적할 수 있다.

| ID | 위치 | 확인한 내용 |
|---|---|---|
| [C01] | `molguidance_hp1kg_baseline/convert_hp1kg_joint_kekule.py` | joint atom-state·Kekulé codec |
| [C02] | `configs/local/guidance_est/hp1kg_est_cfg_joint_kekule_stage6_independent_time_bond_first.yaml` | modality time, state 수, 원소, FAKE 설정의 base |
| [C03] | `configs/local/stage8/08_longtrain_cfg.yaml` | Stage 8 CFG production config의 상속 시작점 |
| [C04] | `molguidance/models/chemistry_auxiliary.py` | effective probabilities와 aromatic loss |
| [C05] | `molguidance/models/chemistry_distribution.py` | bounded convolution, overflow, valence/charge probability |
| [C06] | `molguidance/models/classifier_free_guidance.py` | CFG 확률, MASK-only CE, endpoint 저장 |
| [C07] | `molguidance/analysis/joint_decode.py` | support, objective, constraints, global validation |
| [C08] | `tools/generate_stage8_panel.py` | 실제 Campbell sampling·required count·decoder 호출 |
| [C09] | `molguidance/models/flowmol.py` | 공통 학습·auxiliary 처리 |
| [C10] | `docs/STRUCTURE_FAILURE_REPORT_20261029.txt` | 기존 1,000개 raw·projection·geometry·분포 분석 |
| [C11] | `configs/local/stage8/02_correction.yaml`, `molguidance/models/correction.py` | node/edge corruption 0 및 auxiliary 복원 구현 |
| [C12] | `molguidance/analysis/strict_generation.py` | production aromatic-plane 검사 |
| [C13] | `molguidance/data_processing/kekule_augmentation.py` | identity를 보존하는 aromatic component matching |
| [C14] | `tests/test_stage8_joint_decode.py` | oracle aromatic edge를 제공하는 복구 시험 |
| [C15] | `analysis/hp1kg_e4690_s1_minus_2t1_n1000_report_20261027/all_1000_structure_energy.csv` | raw·eligible·count 독립 재집계 자료 |
| [C16] | `analysis/stage8_commit_validation_20261029/unittest.log` | 기존 130개 통과 로그 |

현재 코드에 대한 해석은 다음 경계를 유지한다.

- **이미 있음:** joint state, Kekulé, representation augmentation, distribution loss, global decoder.
- **확인된 결함/불일치:** numerical support 처리, count 정책 충돌, production/reanalysis plane 정의 차이.
- **추가 검증할 가설:** visible head의 복원 점수 적합성, 학습 교란과 생성 residual의 차이, 작은 데이터에서의 고리계 분포 학습.
- **현재 미구현 제안:** log-score 저장 계약 개선, remask-based repair, geometry/edit-aware joint correction, ring-system 중심 대안의 실증 비교.

본 문서의 우선순위는 화학 loss를 계속 추가하는 것보다 **기존 모델과 decoder 사이의 정보·제약·학습 계약을 검증하는 것**이다. 그 이후에도 남는 topology와 geometry 문제를 motif·fragment·다양체 방법으로 분리해 실험하는 연구 경로를 제안한다.

[P01]: https://proceedings.mlr.press/v162/hoogeboom22a.html
[P02]: https://arxiv.org/abs/2203.02923
[P03]: https://proceedings.mlr.press/v202/peng23b.html
[P04]: https://arxiv.org/abs/2302.09048
[P05]: https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00363f
[P06]: https://www.nature.com/articles/s43588-025-00946-y
[P07]: https://arxiv.org/abs/2512.12198
[P08]: https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00206k
[P09]: https://proceedings.mlr.press/v80/jin18a.html
[P10]: https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc04620j
[P11]: https://www.nature.com/articles/s42256-024-00815-9
[P12]: https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05976a
[P13]: https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc04185a
[P14]: https://arxiv.org/html/2402.03559v3
[P15]: https://arxiv.org/html/2209.14687v4
[P16]: https://papers.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html
[P17]: https://arxiv.org/html/2302.03660v3
[D01]: https://www.rdkit.org/docs/RDKit_Book.html#aromaticity
[C01]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/molguidance_hp1kg_baseline/convert_hp1kg_joint_kekule.py
[C02]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/configs/local/guidance_est/hp1kg_est_cfg_joint_kekule_stage6_independent_time_bond_first.yaml
[C03]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/configs/local/stage8/08_longtrain_cfg.yaml
[C04]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/molguidance/models/chemistry_auxiliary.py#L107
[C05]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/molguidance/models/chemistry_distribution.py
[C06]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/molguidance/models/classifier_free_guidance.py#L76
[C07]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/molguidance/analysis/joint_decode.py#L49
[C08]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/tools/generate_stage8_panel.py#L328
[C09]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/molguidance/models/flowmol.py
[C10]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/06d5fdf476b79a1a1ab1d575201b1fe54ee592ab/MolGuidance_guidance_selector_e987/docs/STRUCTURE_FAILURE_REPORT_20261029.txt
[C11]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/configs/local/stage8/02_correction.yaml
[C12]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/molguidance/analysis/strict_generation.py#L48
[C13]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/molguidance/data_processing/kekule_augmentation.py
[C14]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/c8af84d6a382f76c8757ffbc149e771a992f6629/MolGuidance_guidance_selector_e987/tests/test_stage8_joint_decode.py#L47
[C15]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/06d5fdf476b79a1a1ab1d575201b1fe54ee592ab/analysis/hp1kg_e4690_s1_minus_2t1_n1000_report_20261027/all_1000_structure_energy.csv
[C16]: https://github.com/PSANGGIL/Excited_Guidance_rigid/blob/06d5fdf476b79a1a1ab1d575201b1fe54ee592ab/analysis/stage8_commit_validation_20261029/unittest.log
