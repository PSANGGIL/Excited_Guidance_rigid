# MolGuidance 분석 메모 (2026-10-07)

## 핵심 결과

| 방법 | RDKit valid | connected | xTB/STDA 성공 | strict target-like |
|---|---:|---:|---:|---:|
| CFG | 527 | 514 | 437 | 65/1000 |
| MG | 705 | 339 | 259 | 29/1000 |
| CG | 351 | 277 | 237 | 25/1000 |
| AG | 549 | 532 | 466 | 82/1000 |

strict target-like는 |EST| <= 0.1, 방향족 포함, 5/6원 고리, chemical-clean을 동시에 만족하는 구조다.

## 병목

1. CFG/AG probability guidance가 probability에 다시 softmax(probability / temperature)를 적용해 log-space guidance와 불일치한다.
2. residual atom/charge/edge mask를 Se/+4/no-bond로 해석할 수 있다.
3. atom, charge, bond가 독립 예측되고 전역 valence/connectivity/Kekulé 제약이 없다.
4. NONE edge가 약 92.6%라 edge class imbalance가 크다.
5. xTB 성공은 diffusion 좌표 품질이 아니라 xTB 구조 복구를 포함한다.

## 우선 수정

1. guidance probability를 log-space에서 결합하고 temperature를 log probability에만 적용한다.
2. residual mask를 실제 원소/전하로 변환하지 않고 invalid 처리한다.
3. 기존 checkpoint로 temperature/guidance sweep을 먼저 수행한다.
4. 이후 Kekulé target, balanced edge loss, implicit-H/valence state, degree-aware normalization, global chemical decoder를 추가한다.

## 평가

chemical-clean, connectedness, |EST| hit, 5/6-ring fraction, 3/4-ring fraction, >=9-ring fraction, DB-supported local environment, coordinate RMSD/strain을 함께 기록한다.
