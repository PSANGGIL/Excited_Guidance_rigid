# HP1KG 공식 XYZ 재검증

검증일: 2026-10-16 UTC

## 결론

PSDI에서 받은 XYZ 파일과 canonical HDF5는 정상이다. 누락, 잘린 XYZ, checksum 불일치, 비유한 좌표, 중복 파일, 단위 오류는 발견되지 않았다.

다만 공식 좌표를 기존 SMILES-derived heavy-atom graph에 결합하는 모델 view에는 별도의 표현 불일치가 있다.

- 61분자, 164 heavy atom에서 official explicit-H attachment와 trusted SMILES H-count가 다르다.
- 6분자에서 XYZ 거리 기반 heavy connectivity와 trusted SMILES graph가 edge 1개만큼 다르다.
- 이 문제는 다운로드 손상이 아니라 좌표/SMILES tautomer·bond-perception·graph 정의 차이다.

## 다운로드 무결성

- 공식 API snapshot record: 1,581
- CSV record: 1,581
- 로컬 XYZ: 1,581
- API/CSV/manifest ID set: 완전 일치
- parse 성공: 1,581/1,581
- 공식 repository MD5 일치: 1,581/1,581
- 공식 API의 record ID, XYZ filename, size, checksum과 manifest·실제 파일의 전수 대조: 1,581/1,581 일치
- unique XYZ path: 1,581
- unique repository MD5: 1,581
- duplicate MD5 group: 0
- formula와 XYZ 원소 조성: 1,581/1,581 일치

CSV의 `number_of_atoms`는 1,563건에서는 heavy-atom count, 18건에서는 all-atom count로 기록돼 있지만, 18건 모두 공식 formula와 XYZ 총 atom count가 일치한다. 이는 CSV column 해석의 비일관성이지 XYZ 오류가 아니다.

## ORCA 원본 대조

테스트 record `NEFPAR`의 공식 ORCA `.out`도 repository MD5와 일치한다.

- ORCA normal termination 확인
- `.out` 최종 원자 순서와 XYZ 순서 일치
- `.out` 최종 좌표 대 XYZ RMSD: 0.0 Å
- 최대 좌표 차이: 0.0 Å
- SA-CASSCF/SC-NEVPT2 S1/S2/T1/T2 값이 CSV와 모두 정확히 일치

따라서 XYZ는 B3LYP/6-31G*로 최적화된 S0 좌표이며, 저장된 excited-state energy는 이 좌표에서 계산된 vertical 값이다. S1/T1 최적화 좌표는 아니다.

## HDF5 독립 재빌드

기존 파일을 변경하지 않고 `/tmp/hp1kg_xyz_recheck_20261016_01/rebuilt.h5`에 1,581개 XYZ를 다시 파싱해 HDF5를 구축했다.

- HDF5 dataset path: 24개, 동일
- 모든 dataset shape/dtype: 동일
- 모든 문자열·정수·float 값: exact match
- 좌표 최대 차이: 0.0 Å
- source XYZ byte stream: 1,581개 exact round-trip
- 차이가 난 유일한 HDF5 attribute: `created_at_utc`

따라서 재빌드 파일 SHA256은 생성시각 metadata 때문에 다르지만 실제 dataset 값은 모두 동일하다.

Canonical HDF5:

- SHA256: `fa6a60cdc9921a998b0e44b753ac21b39f067abf234727722fd160bd3c12cff5`
- molecule: 1,581
- all atoms: 73,235
- heavy atoms: 42,061
- hydrogen: 31,174
- coordinate dtype: float64
- unit: Å
- non-finite coordinates/properties: 0

## 좌표 물리 범위 점검

전체 공식 all-atom 좌표:

- 분자별 최소 pair distance 범위: 0.927–1.096 Å
- 0.5 Å 미만 atom collision: 0분자
- H에서 가장 가까운 heavy atom 거리 median: 1.091 Å
- H-heavy distance 99% quantile: 1.103 Å
- per-atom nearest-neighbor distance median: 1.094 Å

모델의 trusted graph edge에 대한 heavy-atom 거리:

- 전체 median: 1.406 Å
- aromatic median: 1.398 Å
- single median: 1.501 Å
- double median: 1.343 Å
- triple median: 1.165 Å

이는 Bohr/Å 변환 오류나 좌표 scale 손상이 없음을 보여 준다.

HDF5 float64 좌표를 현재 모델 split의 float32로 변환할 때:

- 최대 반올림 오차: `9.28e-7 Å`
- RMS 반올림 오차: `7.72e-8 Å`

학습에 영향을 줄 수준이 아니다.

## 모델용 heavy-atom mapping 재검증

재빌드 HDF5에서 model-ready split을 `/tmp`에 다시 생성했다. 기존 split과 SHA256까지 동일했다.

- train: `d9f4777afcc753180aedf7b2c0674278cf7cc28963ca681c86c0b575133e1896`
- validation: `e77e7fd2989918c20ba23bdca4e0fdf34e1881eb5b6bfd0088c69faadf5f1de2`
- test: `05d9f0500d620db1f11d18f5556ea326da6e757a26295011f00d1f28117fff7d`

Mapping mode:

- heavy graph + attached-H exact: 1,516
- heavy graph + element exact, H-count 불일치: 59
- XYZ graph가 trusted graph보다 edge 1개 많음: 4
- trusted graph가 XYZ graph보다 edge 1개 많음: 2

Edge 예외 6개:

| ID | mapping | 설명 |
|---|---|---|
| SEFVEE | xyz_superset_hcount | XYZ perception edge +1 |
| TIPTOB | xyz_superset_element | XYZ perception edge +1, H-count mismatch |
| AFAFAP | xyz_superset_element | XYZ perception edge +1, H-count mismatch |
| SODBIW | xyz_superset_hcount | XYZ perception edge +1 |
| HONFIA | target_superset_hcount | trusted graph edge +1 |
| ICAHUQ | target_superset_hcount | trusted graph edge +1 |

가능한 graph mapping을 전부 열거해 bond-distance 기준으로 다시 비교했지만 더 나은 atom-index mapping은 없었다. 즉 이 6건은 인덱스 선택 오류가 아니라 graph와 official geometry의 실제 정의 차이다.

추가 관찰:

- `BAQBEA01`: exact mapping이지만 S-S trusted bond가 2.548 Å다. 공식 좌표 자체와 checksum은 정상이며 unusual sulfur geometry에 해당한다.
- `HONFIA`: trusted N-C bond가 2.826 Å이고 XYZ-perceived graph에는 해당 edge가 없다. 가능한 다른 mapping으로도 개선되지 않는다.
- `ICAHUQ`: trusted graph가 XYZ-perceived graph보다 edge 1개 많다.

## 사용 판단

1. 공식 XYZ/HDF5 자체는 그대로 사용해도 된다.
2. 현재 heavy-atom coordinate training에도 전체적인 다운로드·단위·index 손상은 없다.
3. H-count head의 엄밀한 학습을 위해서는 61개 H-attachment mismatch를 별도 검토하거나 제외하는 것이 좋다.
4. graph-coordinate 완전 일관성이 필요한 다음 학습에서는 edge 예외 6개, 특히 `HONFIA`와 `ICAHUQ`를 제외하거나 공식 geometry와 일치하는 graph를 별도로 확정해야 한다.
5. explicit-H 모델을 만들 때는 HDF5의 공식 all-atom order를 직접 사용하면 heavy-atom remapping 자체를 피할 수 있다. 다만 SMILES tautomer/결합 정의와의 일치 여부는 여전히 검증해야 한다.

## 근거 파일

- `data_hp1kg_official_psdi/manifest.csv`
- `data_hp1kg_official_psdi/metadata/community_records.json`
- `data_hp1kg_official_psdi/validation_report.json`
- `data_hp1kg_official_psdi/hp1kg_benchmarkset1500_official_3d.h5`
- `data_hp1kg_official_psdi/hp1kg_benchmarkset1500_official_3d.validation.json`
- `joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz/manifest.json`
