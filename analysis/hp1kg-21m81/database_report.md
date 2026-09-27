# HP1KG BenchmarkSet1500 데이터베이스 분석 보고서

- 원본: `/home/jovyan/cpu-only-1-datavol-1/psg/generation/data/hp1kg-21m81/BenchmarkSet1500.csv`
- 분석일: 2026-09-17
- 라이선스: CC-BY-4.0
- 용도: 유기 반도체의 SA-CASSCF/NEVPT2 excited-state benchmark

## 1. 개요 및 선택 편향

데이터셋 이름은 BenchmarkSet1500이지만 실제 CSV에는 고유한 분자 1,581개가 있다. 약 40,000개의 CSD 기반 유기 반도체 후보 중 작은 singlet-triplet gap, 작은 S2-S1 간격, double-excitation 성격을 보이는 전자적으로 까다로운 후보를 중심으로 선별됐으며, 약 200개의 무작위 표본이 화학적 다양성을 위해 추가됐다. 따라서 일반 유기분자의 무작위 표본이 아니라 작은 ΔEST와 다중참조 후보가 강화된 데이터다.

포함 정보는 SMILES, InChI, 분자식, CCDC ID, DOI, SA-CASSCF/NEVPT2 S1·S2·T1·T2 에너지 및 S1·S2 oscillator strength다.

## 2. 구조 무결성

| 항목 | 결과 |
|---|---:|
| 전체 행 | 1,581 |
| RDKit 파싱 성공 | 1,581 |
| ID/SMILES/InChI 중복 | 0 |
| 다중 fragment | 0 |
| net formal charge 비영(非零) | 0 |
| 내부 formal charge 보유 | 429 |
| radical | 0 |
| 기존 QCDGE A+B와 중복 | 0 |
| RDKit codec identity round-trip | 1,581/1,581 |

429개는 전체 전하가 0인 charge-separated/zwitterionic 구조다. 주요 패턴은 N+/O− 309개, B−/N+ 25개, N−/N+ 20개, 복수 N+/O− 15개, C−/N+ 15개다.

원본 `#atoms` 열은 18개 행에서 RDKit heavy-atom 수와 불일치한다. 일부 행은 수소까지 포함한 전체 원자 수를 기록한 것으로 보인다. 전처리에서는 이 열을 쓰지 않고 SMILES에서 heavy-atom 수를 재계산한다.

## 3. 원소 분포

| 원소 | 포함 분자 수 | 전체 원자 수 |
|---|---:|---:|
| C | 1,581 | 32,535 |
| N | 1,219 | 2,965 |
| O | 1,121 | 3,381 |
| S | 594 | 1,886 |
| P | 147 | 212 |
| F | 115 | 566 |
| Cl | 95 | 152 |
| Se | 95 | 141 |
| Br | 66 | 94 |
| B | 61 | 70 |
| Si | 24 | 58 |
| As | 1 | 1 |

기존 C/N/O/F 어휘만으로 표현 가능한 분자는 663개(41.9%)이며, 918개는 신규 원소를 포함한다. 현재 HP1KG 모델 어휘에는 `B, C, N, O, F, Si, P, S, Cl, As, Se, Br` 12종을 포함했다. As와 Si는 표본이 극히 적어 생성 품질을 기대하기 어렵다.

## 4. 분자 크기

RDKit heavy-atom 기준 최소 6, 평균 26.6, 중앙값 25, 최대 69개다.

| 구간 | 결과 |
|---|---:|
| 정확히 10개 | 22 |
| 32개 이하 | 1,201 (76.0%) |
| 50개 이하 | 1,534 (97.0%) |
| 50개 초과 | 47 |
| 75 percentile | 32 |
| 90 percentile | 42 |
| 95 percentile | 47 |
| 99 percentile | 55 |

기존 QCDGE의 10원자 고정 분포와 달리 6~69개의 가변 크기를 제공하므로 큰 분자에서의 edge-density와 atom-count prior를 학습하는 데 유용하다.

## 5. 고리 및 방향족성

| 항목 | 결과 |
|---|---:|
| 평균 전체 ring | 3.46 |
| 중앙값 전체 ring | 3 |
| 최대 전체 ring | 14 |
| ringless | 63 |
| 방향족 포함 | 1,362 (86.1%) |
| 비방향족 | 219 (13.9%) |
| 평균 방향족 ring | 2.43 |
| 최대 방향족 ring | 13 |

방향족 ring 개수는 0개 219, 1개 305, 2개 451, 3개 226, 4개 177, 5개 83, 6개 이상 120개다. fused, bridged, heteroaromatic 및 polyaromatic 구조 학습에 유리하다.

## 6. Excited-state 에너지

### SA-CASSCF

| 값 | 평균 | 중앙값 | 최소 | 최대 |
|---|---:|---:|---:|---:|
| S1 | 4.518 | 4.620 | 1.045 | 8.676 |
| T1 | 3.282 | 3.253 | 0.862 | 8.236 |
| S1−T1 | 1.236 | 1.306 | 0.000 | 3.547 |

SA-CASSCF에서 S1<T1은 0개, S1=T1은 1개이며 S2<S1 및 T2<T1은 없다.

### NEVPT2

| 값 | 평균 | 중앙값 | 최소 | 최대 |
|---|---:|---:|---:|---:|
| S1 | 3.619 | 3.507 | 1.253 | 8.443 |
| T1 | 3.097 | 2.969 | 1.029 | 8.174 |
| S1−T1 | 0.522 | 0.463 | −1.133 | 2.655 |

NEVPT2에서 S1<T1은 144개(9.1%), S1=T1은 4개이며, ΔEST<0.275 eV는 603개다. 요청에 따라 역전 상태를 필터링하거나 별도 class로 취급하지 않고 그대로 학습한다.

현재 train split 정규화 값은 S1 평균 3.6400 eV/표준편차 1.0043 eV, T1 평균 3.1115 eV/표준편차 0.9701 eV다.

## 7. 계산법 간 차이

| 상태 | Pearson r | NEVPT2−CASSCF 평균 | 평균 절대 차이 |
|---|---:|---:|---:|
| S1 | 0.606 | −0.898 eV | 1.048 eV |
| S2 | 0.767 | −1.040 eV | 1.131 eV |
| T1 | 0.865 | −0.185 eV | 0.369 eV |
| T2 | 0.875 | −0.195 eV | 0.352 eV |

S1 방법 차이가 1 eV를 넘는 분자는 804개, 2 eV 초과 162개, 3 eV 초과 3개다. 따라서 SA-CASSCF와 NEVPT2를 동일 label source처럼 섞으면 안 된다. 현재 학습은 NEVPT2 S1/T1만 사용한다.

## 8. Oscillator strength

NEVPT2 f(S1)은 평균 0.204, 중앙값 0.018, 최대 2.771이며 f(S2)는 평균 0.255, 중앙값 0.110, 최대 1.710이다. 분포가 강하게 치우쳐 있어 추후 condition으로 추가할 때 raw scalar보다 `log1p(f)` 등의 변환이 적절하다. 현재 모델에는 oscillator strength를 사용하지 않는다.

## 9. Cartesian 좌표 처리

원본 CSV에는 실제 계산 Cartesian 좌표가 없다. 현재 전처리는 결정론적 RDKit conformer를 생성한다.

| 좌표 출처 | 분자 수 |
|---|---:|
| ETKDGv3 + MMFF | 1,423 |
| ETKDGv3 + UFF | 82 |
| ETKDGv3, 최적화 없음 | 74 |
| RDKit 2D fallback | 2 |

이 좌표는 실제 CCDC/Gaussian/NEVPT2 geometry가 아니므로 그래프·원소·결합·크기 학습과 geometry prior에는 사용할 수 있지만, 정밀한 geometry–excited-state 상관관계에는 오차가 있다.

## 10. 현재 학습용 캐스팅

| split | 분자 수 |
|---|---:|
| train | 1,264 |
| validation | 158 |
| test | 159 |

입력/target은 atom type, formal charge, explicit H, radical count, Kekulé bond, aromatic edge/cycle, heavy-atom 좌표, atom-count mask, NEVPT2 S1/T1이다. 사용하지 않는 원본 정보는 SA-CASSCF 전체, NEVPT2 S2/T2, oscillator strength, stereochemistry와 실제 계산 geometry다.

전처리 오류 0건, 좌표 NaN/Inf 0건이며 6~69원자 혼합 배치 optimizer smoke test에서 loss와 gradient가 모두 유한했다.

## 11. 종합 평가

장점은 다양한 원소, 6~69원의 폭넓은 크기, 풍부한 polyaromatic 구조, 많은 형식전하 구조, 중복·fragment·파싱 실패가 없는 점, 고정밀 다중참조 excited-state label이다.

한계는 전체 1,581개로 diffusion 학습에는 작고, As·Si 등 희소 원소가 있으며, 실제 계산 좌표가 없고, 일반 화학공간이 아닌 작은 ΔEST/다중참조 후보 중심의 selection bias가 있다는 점이다. 장기적으로는 큰 구조 DB로 graph/geometry를 선학습한 뒤 HP1KG의 NEVPT2 S1/T1으로 condition fine-tuning하는 방식이 가장 안정적이다.
