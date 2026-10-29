#!/usr/bin/env python3
"""Recheck saved e4690 outputs; write separate reports without running chemistry."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "analysis/hp1kg_e4690_s1_minus_2t1_n1000_report_20261027"
DEEP = ROOT / "analysis/hp1kg_e4690_deep_diagnostics_20261028_verified"
GEN = ROOT / "runs/generation_stage8_hp1kg_best_e4690_s1_minus_2t1_0_n1000_cfg1_20261027"
XTB = GEN / "xtb_stda_diffusion_coords_24x1"
DEFAULT_OUT = ROOT / "analysis/hp1kg_e4690_result_reanalysis_20261029"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    def value(k, v):
        if k in ("q_exact", "s1_text", "t1_text"):
            return v
        if v == "":
            return None
        if v in ("True", "False"):
            return v == "True"
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return v
    with path.open() as f:
        return [{k: value(k, v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def stat(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), std_ddof0=float(a.std()),
                median=float(np.median(a)), minimum=float(a.min()), maximum=float(a.max()))


def hit(row):
    return row["calculation_status"] == "ok" and abs(Decimal(row["q_exact"])) <= Decimal("0.1")


def plane_metrics(mol, xyz):
    atoms_only, full_bonds, excluded = [], [], []
    for ring in mol.GetRingInfo().AtomRings():
        atom_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        bond_aromatic = all(mol.GetBondBetweenAtoms(ring[j], ring[(j + 1) % len(ring)]).GetIsAromatic()
                            for j in range(len(ring)))
        assert not bond_aromatic or atom_aromatic
        if not atom_aromatic:
            continue
        centered = xyz[list(ring)] - xyz[list(ring)].mean(0)
        normal = np.linalg.svd(centered, full_matrices=False)[-1][-1]
        rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
        atoms_only.append(rms)
        if bond_aromatic:
            full_bonds.append(rms)
        else:
            excluded.append(dict(atom_indices=list(ring), size=len(ring), rms_A=rms))
    return dict(atom_rule_max_A=max(atoms_only, default=0.0),
                bond_rule_max_A=max(full_bonds, default=0.0),
                bond_rule_ring_count=len(full_bonds), excluded_rings=excluded)


def render_candidates(rows, mols, output):
    from PIL import Image, ImageDraw, ImageFont
    width, height, cols = 580, 740, 3
    canvas = Image.new("RGB", (cols * width, ((len(rows) + cols - 1) // cols) * height), "white")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 27)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    draw = ImageDraw.Draw(canvas)
    for j, r in enumerate(rows):
        x, y = (j % cols) * width, (j // cols) * height
        index = r["generation_index"]
        labels = [f"#{index:06d}   {r['formula']}",
                  f"S1 {r['s1']:.3f}    T1 {r['t1']:.3f} eV",
                  f"Q {r['q']:+.3f} eV    f {r['s1_f']:.4f}",
                  f"Rings {','.join(map(str, sorted(r['ring_sizes'])))}"]
        for line, text in enumerate(labels):
            draw.text((x + 20, y + 18 + 52 * line), text, font=font, fill="#122438")
        mol = Chem.RemoveHs(Chem.Mol(mols[index]), sanitize=False)
        rdDepictor.Compute2DCoords(mol)
        renderer = rdMolDraw2D.MolDraw2DCairo(width - 30, 455)
        renderer.drawOptions().padding = 0.10
        renderer.DrawMolecule(mol)
        renderer.FinishDrawing()
        canvas.paste(Image.open(io.BytesIO(renderer.GetDrawingText())).convert("RGB"), (x + 15, y + 226))
        note = "Aromatic included" if r["aromatic"] else "No RDKit aromatic ring"
        if r["nitro_only_formal_charges"]:
            note += "; nitro charge notation"
        draw.text((x + 20, y + 694), note, font=small, fill="#36536b")
        draw.line((x + 8, y + height - 2, x + width - 8, y + height - 2), fill="#dce2e8", width=2)
    canvas.save(output / "near_zero_only_5_6_rings_12_rdkit2d.png")


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "analysis_script.py")
    status = output / "status.json"
    status.write_text(json.dumps(dict(phase="running", started_utc=datetime.now(timezone.utc).isoformat())))
    expected = {}
    for source in (PREV / "summary.json", DEEP / "summary.json"):
        for p, digest in json.loads(source.read_text())["source_sha256"].items():
            assert p not in expected or expected[p] == digest
            expected[p] = digest
    for p, digest in expected.items():
        assert sha(p) == digest, f"Changed source: {p}"

    rows = read_csv(PREV / "all_1000_structure_energy.csv")
    diagnostics = {r["generation_index"]: r for r in read_csv(DEEP / "endpoint_molecule_diagnostics_1000.csv")}
    raw = {r["generation_index"]: r for r in json.loads((GEN / "raw_analysis/records.json").read_text())}
    quantum = {r["generation_index"]: r for r in read_csv(XTB / "results.csv")}
    assert len(rows) == len(raw) == len(diagnostics) == 1000
    assert set(r["generation_index"] for r in rows) == set(range(1000))
    for r in rows:
        i = r["generation_index"]
        for key in ("raw_valid", "eligible", "aromatic", "requested_heavy_atoms", "sampled_real_atoms"):
            assert r[key] == raw[i][key], (i, key)
        for key in ("nonring_atoms", "ring_atom_fraction", "largest_ring_system"):
            r[key] = diagnostics[i].get(key)

    helper_path = ROOT / "analysis/analyze_hp1kg_e3911_n10000_20261023.py"
    spec = importlib.util.spec_from_file_location("previous_geometry_audit", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.GEN, helper.XTB = GEN, XTB
    success = [r for r in rows if r["calculation_status"] == "ok"]
    assert len(success) == len(quantum) == 452
    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    mols, rechecked, geometry_changes, input_hashes = {}, [], [], {}
    for r in success:
        i = r["generation_index"]
        qrow = quantum[i]
        q = Decimal(str(qrow["s1_energy_eV"])) - 2 * Decimal(str(qrow["t1_energy_eV"]))
        assert q == Decimal(r["q_exact"])
        assert Decimal(r["s1_text"]) == Decimal(str(qrow["s1_energy_eV"]))
        assert Decimal(r["t1_text"]) == Decimal(str(qrow["t1_energy_eV"]))
        assert qrow["status"] == "ok"
        check = helper.geometry_audit(r)
        assert not check["geometry_audit_error"], check
        assert check["energy_log_match"] and check["singlet_triplet_flags_verified"] and check["optimization_converged"]
        for key in ("heavy_rmsd_A", "postopt_aromatic_plane_rms_A"):
            assert abs(check[key] - r[key]) < 1e-10, (i, key)
        assert check["wbo_change_warning"] == r["wbo_change_warning"]
        work = XTB / "work" / f"generated_{i:06d}"
        inp_path = XTB / "inputs" / f"generated_{i:06d}.sdf"
        mol = Chem.MolFromMolFile(str(inp_path), removeHs=False, sanitize=True)
        opt = Chem.MolFromMolFile(str(work / "xtbopt.sdf"), removeHs=False, sanitize=False)
        assert mol is not None and opt is not None
        mols[i] = mol
        inp_plane = plane_metrics(mol, mol.GetConformer().GetPositions())
        opt_plane = plane_metrics(mol, opt.GetConformer().GetPositions())
        assert abs(opt_plane["atom_rule_max_A"] - r["postopt_aromatic_plane_rms_A"]) < 1e-10
        other_input_ok = r["geometry_evaluable"] and not any(r[k] for k in ("short_bonds", "long_bonds", "nonbonded_clashes"))
        assert bool(other_input_ok and inp_plane["atom_rule_max_A"] <= .15) == r["geometry_screen"]
        charges = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
        nitro_charges = {i for match in mol.GetSubstructMatches(nitro) for i in match
                         if mol.GetAtomWithIdx(i).GetFormalCharge()}
        oo = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
              if b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 8]
        r.update(formula=rdMolDescriptors.CalcMolFormula(mol), OO_bonds=oo,
                 nitro_only_formal_charges=bool(charges) and charges <= nitro_charges,
                 formal_charges_outside_nitro=len(charges - nitro_charges),
                 input_bond_aromatic_plane_rms_A=inp_plane["bond_rule_max_A"],
                 postopt_bond_aromatic_plane_rms_A=opt_plane["bond_rule_max_A"],
                 input_geometry_bond_aromatic=bool(other_input_ok and inp_plane["bond_rule_max_A"] <= .15),
                 postopt_bond_aromatic_plane_warning=opt_plane["bond_rule_max_A"] > .15,
                 bond_aromatic_ring_count=opt_plane["bond_rule_ring_count"],
                 all_local_environments_in_train=r["unsupported_atoms"] == r["unsupported_bonds"] == 0)
        check.update(input_plane=inp_plane, postopt_plane=opt_plane)
        rechecked.append(check)
        if (opt_plane["atom_rule_max_A"] > .15) != (opt_plane["bond_rule_max_A"] > .15):
            geometry_changes.append(dict(generation_index=i, old_A=opt_plane["atom_rule_max_A"],
                                         revised_A=opt_plane["bond_rule_max_A"], excluded=opt_plane["excluded_rings"]))
        for p in [inp_path, *[work / name for name in ("xtbopt.sdf", "wbo", "xtb_opt.log", "stda_singlet.log", "stda_triplet.log")]]:
            input_hashes[str(p)] = sha(p)

    near = [r for r in success if hit(r)]
    only56_near = sorted([r for r in near if r["only56"]], key=lambda r: abs(Decimal(r["q_exact"])))
    assert len(near) == 34 and len(only56_near) == 12
    by_id = {r["generation_index"]: r for r in success}
    assert by_id[830]["OO_bonds"] == []
    assert by_id[189]["nitro_only_formal_charges"] and by_id[189]["net_charge"] == 0

    group_defs = [("only_5_6", "5·6원 고리만", lambda r: r["only56"]),
                  ("small_without_7plus", "3·4원 포함, 7원 이상 없음", lambda r: r["small"] and not r["seven_plus"]),
                  ("7plus_without_small", "7원 이상 포함, 3·4원 없음", lambda r: r["seven_plus"] and not r["small"]),
                  ("small_and_7plus", "3·4원과 7원 이상 모두 포함", lambda r: r["small"] and r["seven_plus"])]
    groups = []
    for key, label, predicate in group_defs:
        part = [r for r in success if predicate(r)]
        groups.append(dict(key=key, label=label, n=len(part), target_hits=sum(hit(r) for r in part),
                           target_hit_rate=sum(hit(r) for r in part) / len(part),
                           mean_q=float(np.mean([r["q"] for r in part])),
                           mean_s1=float(np.mean([r["s1"] for r in part])),
                           mean_heavy_atoms=float(np.mean([r["heavy_atoms"] for r in part])),
                           mean_rmsd_A=float(np.mean([r["heavy_rmsd_A"] for r in part])),
                           postopt_plane_warnings=sum(r["postopt_bond_aromatic_plane_warning"] for r in part),
                           wbo_warnings=sum(r["wbo_change_warning"] for r in part)))
    assert sum(g["n"] for g in groups) == 452 and sum(g["target_hits"] for g in groups) == 34

    size_strata = []
    for lo, hi in [(16, 25), (26, 35), (36, 45), (46, 100)]:
        part = [r for r in rows if lo <= r["requested_heavy_atoms"] <= hi]
        size_strata.append(dict(requested_size=f"{lo}-{hi}", attempts=len(part),
                                raw_valid=sum(r["raw_valid"] for r in part),
                                xtb_ok=sum(r["calculation_status"] == "ok" for r in part),
                                target_hits=sum(hit(r) for r in part)))

    filters = dict(target_0p1=len(near), only_5_6=len(only56_near),
                   only_5_6_aromatic=sum(r["aromatic"] for r in only56_near),
                   only_5_6_aromatic_no_formal_charge=sum(r["aromatic"] and not r["has_atomic_charge"] for r in only56_near),
                   only_5_6_geometry_wbo_clear=sum(r["input_geometry_bond_aromatic"] and not r["postopt_bond_aromatic_plane_warning"] and not r["wbo_change_warning"] for r in only56_near),
                   only_5_6_all_local_supported=sum(r["all_local_environments_in_train"] for r in only56_near))
    preferred = [r["generation_index"] for r in only56_near if r["all_local_environments_in_train"]
                 and r["input_geometry_bond_aromatic"] and not r["postopt_bond_aromatic_plane_warning"]
                 and not r["wbo_change_warning"] and r["s1"] >= 1 and r["s1_f"] >= .01]
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), checkpoint_epoch=4690,
                   condition="S1 - 2*T1", target_eV=0, guidance_scale=1,
                   verification=dict(previous_source_files_checked=len(expected), source_mismatches=0,
                                     regenerated_geometry_audits=len(rechecked), matched_singlet_triplet_logs=904),
                   attempted=1000, raw_valid=sum(r["raw_valid"] for r in rows), xtb_success=len(success),
                   energy={k: stat([r[k] for r in success]) for k in ("s1", "t1", "q")},
                   energy_target_hits={k: stat([r[k] for r in near]) for k in ("s1", "t1", "q")},
                   filters=filters, ring_groups=groups, requested_size_strata=size_strata,
                   plane_warning_old=sum(r["postopt_aromatic_plane_rms_A"] > .15 for r in success),
                   plane_warning_bond_aromatic=sum(r["postopt_bond_aromatic_plane_warning"] for r in success),
                   plane_warning_changed_ids=[r["generation_index"] for r in geometry_changes],
                   input_geometry_old=sum(r["geometry_screen"] for r in success),
                   input_geometry_bond_aromatic=sum(r["input_geometry_bond_aromatic"] for r in success),
                   wbo_warnings=sum(r["wbo_change_warning"] for r in success),
                   near_wbo_warnings=sum(r["wbo_change_warning"] for r in near),
                   near_plane_warnings=sum(r["postopt_bond_aromatic_plane_warning"] for r in near),
                   near_S1_below1=sum(r["s1"] < 1 for r in near),
                   molecule830_OO_bonds=by_id[830]["OO_bonds"],
                   candidate_ids_optional_S1_ge1_f_ge0p01_supported=preferred,
                   eligibility_exclusions=dict(Counter(r["calculation_status"] for r in rows if r["calculation_status"] != "ok")),
                   source_sha256={**expected, **input_hashes})
    write_csv(output / "ring_group_comparison.csv", groups)
    write_csv(output / "rechecked_success_452.csv", success)
    write_csv(output / "target_hits_34.csv", near)
    write_csv(output / "target_hits_only56_12.csv", only56_near)
    write_csv(output / "geometry_recheck_452.csv", rechecked)
    write_csv(output / "corrected_aromatic_plane_cases.csv", geometry_changes)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    lines = ["HP1KG e4690 생성 결과 재분석 및 정정", summary["created_utc"],
             "대상: 2026-10-27 생성 1,000개. Q=S1−2T1=0 eV, CFG=1. 새 생성/재계산 없이 저장 결과를 재검사했다.",
             "기존 원자료 83개 해시 일치. 452개 최적화 좌표/WBO와 904개 singlet/triplet 로그를 재대조했다.", "",
             "이전 설명의 정정",
             "#830의 O−O 결합 지적은 잘못이었다. 원본 SDF의 O 세 개는 C와 결합하며 O−O edge는 없다. 분자식 C47H30O3.",
             "#189의 [N+](=O)[O−]는 nitro 표기다. net charge=0, radical=0이며 원자 형식전하만으로 비정상이라 판단할 수 없다.",
             "기존 방향족 평면성 검사는 모든 원자가 aromatic인 고리를 포함했다. 이번 지표는 고리 전체 결합이 aromatic인 경우에만 적용한다.",
             f"최적화 후 RMS plane>0.15 Å 경고: {summary['plane_warning_old']} → {summary['plane_warning_bond_aromatic']}개. 판정 변경 13개.",
             f"계산 성공 452개 중 입력 geometry 통과: 기존 {summary['input_geometry_old']}, 같은 결합 기준 재평가 {summary['input_geometry_bond_aromatic']}개.",
             "이는 평가 정의의 수정이며 모델 성능 개선이 아니다. 기존 모델·생성기·원본 산출물은 수정하지 않았다.", "",
             "성공률과 조건 반영",
             "1,000 시도 → RDKit valid 574 → 연결된 중성·폐각 및 XTB/sTDA 완료 452 → |Q|≤0.1 eV 34.",
             "XTB 계산에 투입된 452개는 모두 완료됐다. 548개는 계산 전 제외되었으며 계산 실패로 세지 않는다.",
             "34개 목표 근접은 전체 시도의 3.4%, 계산 완료분의 7.52%다. 모두 HP1KG 전체 DB와 중복 없는 생성 그래프다.",
             f"Q 평균 {summary['energy']['q']['mean']:.3f} ± {summary['energy']['q']['std_ddof0']:.3f} eV; 목표 MAE {np.mean([abs(r['q']) for r in success]):.3f} eV.",
             f"S1 {summary['energy']['s1']['mean']:.3f} ± {summary['energy']['s1']['std_ddof0']:.3f}; T1 {summary['energy']['t1']['mean']:.3f} ± {summary['energy']['t1']['std_ddof0']:.3f} eV.",
             "표준편차 ddof=0. 34개 중 S1<1 eV가 17개. Q 단일 조건은 개별 S1/T1 절대값이나 oscillator strength를 지정하지 않는다.", "",
             "고리 유형별 계산 결과: 서로 겹치지 않는 네 그룹",
             "유형 | 계산 완료 | |Q|≤0.1 | 적중률 | Q 평균 eV | heavy RMSD Å | 수정 평면 경고 | WBO 경고"]
    for g in groups:
        lines.append(f"{g['label']} | {g['n']} | {g['target_hits']} | {100*g['target_hit_rate']:.2f}% | {g['mean_q']:.3f} | {g['mean_rmsd_A']:.3f} | {g['postopt_plane_warnings']} | {g['wbo_warnings']}")
    lines += ["이 표는 같은 생성 표본의 기술 통계다. 크기·조성 차이가 있어 고리 크기의 인과효과로 해석하지 않는다.",
              "3/4원·7원 이상 고리가 함께 있는 그룹은 Q 오차가 작아 보이지만 좌표 및 WBO 경고를 함께 확인해야 한다.",
              "5/6원만 있는 그룹에도 12개 근접 후보가 있어 목표 달성에 다른 크기의 고리가 반드시 필요한 것은 아니다.", "",
              "기존 '선별 10개'의 의미",
              "34개 중 고리 크기가 모두 5/6인 것은 12개다. 12개 모두 입력 geometry, 수정된 최적화 후 평면성 및 WBO 기준을 통과했다.",
              "여기에 방향족 포함을 요구하면 11개(#638 제외), 모든 원자의 formal charge=0까지 요구하면 10개(#189 제외)다.",
              "방향족 포함 및 원자별 전하 없음은 추가 선호 조건이다. 따라서 10개를 화학적으로 가능한 분자의 총수로 해석하지 않는다.",
              f"12개 중 train에서 모든 local atom/bond 환경을 본 구조는 {filters['only_5_6_all_local_supported']}개다.",
              f"선택적으로 S1≥1 eV, f(S1)≥0.01, local support를 요구하면 {preferred}가 남는다. #638은 RDKit 비방향족이다.",
              "ID | 분자식 | S1 | T1 | Q | f(S1) | 고리 | 주석"]
    for r in only56_near:
        notes = []
        if not r["aromatic"]: notes.append("RDKit 비방향족")
        if r["nitro_only_formal_charges"]: notes.append("nitro 형식전하")
        if not r["all_local_environments_in_train"]: notes.append("train 미관찰 local 환경")
        if r["s1"] < 1: notes.append("S1<1 eV")
        if r["s1_f"] < .01: notes.append("f<0.01")
        lines.append(f"#{r['generation_index']:06d} | {r['formula']} | {r['s1']:.3f} | {r['t1']:.3f} | {r['q']:+.3f} | {r['s1_f']:.4f} | {r['ring_sizes']} | {', '.join(notes)}")
    lines += ["", "분자 크기와 전체 성공률", "요청 heavy atom 구간 | 시도 | RDKit valid | XTB 완료 | Q 근접"]
    for r in size_strata:
        lines.append(f"{r['requested_size']} | {r['attempts']} | {r['raw_valid']} | {r['xtb_ok']} | {r['target_hits']}")
    lines += ["큰 분자에서 계산 전 탈락이 많다. surviving molecule의 에너지 평균만 비교하면 전체 생성 성능을 과대평가할 수 있다.", "",
              "구조 분포와 해석",
              "방향족 포함은 434/452(96.02%). 7원 이상 포함 232/452(51.33%), 3/4원 포함 76/452(16.81%). 두 집계는 중복될 수 있다.",
              "train의 해당 비율은 각각 5.22%, 2.37%. 원자 수를 재가중한 train의 7원 이상 비율도 4.86%로 낮다.",
              "분자별 ring atom fraction 평균은 train 63.68%, 생성 82.98%; 고리 밖 원자 평균은 8.85→3.84.",
              "고리 수만 많은 것이 아니라 고리 밖 치환기/연결기와 헤테로 고리 구성이 줄어드는 편향이 남는다.",
              "원래 무효 426개 중 312개는 수소를 제거해도 heavy bond-order 합이 과도하다. H 후처리만으로 전체 실패를 해결할 수 없다.",
              "기존 joint projection의 확률 support와 requested count 충돌도 별도 문제다. projection 결과는 여기 XTB 결과에 포함하지 않았다.", "",
              "평가 한계",
              "Q=0은 HP1KG NEVPT2 라벨, 여기 평가는 XTB/sTDA다. 과거 DB의 Q≈0 11개도 XTB에서는 평균 −0.531 eV였다.",
              "과거 DB 재계산과 이번 생성은 초기 geometry protocol도 다르다. 전체 평균 offset으로 보정하거나 모델 오차를 분리 추정하지 않았다.",
              "RDKit, 최적화 수렴, WBO, 평면성은 각각 진단 기준이다. Hessian/진동수·다른 전자구조 방법·합성 가능성을 검증한 것은 아니다.",
              "이번 표본만으로 conditioning의 인과효과나 추가 학습의 성과를 판정할 수 없다. 동일 checkpoint/count/noise 대조가 필요하다.", "",
              f"재분석 CSV: {output / 'rechecked_success_452.csv'}",
              f"12개 RDKit 2D: {output / 'near_zero_only_5_6_rings_12_rdkit2d.png'}",
              f"수정된 평면성 사례: {output / 'corrected_aromatic_plane_cases.csv'}"]
    (output / "reanalysis_report_ko.txt").write_text("\n".join(lines) + "\n")
    (output / "ERRATA_previous_reports.txt").write_text("\n".join(lines[4:12]) + "\n")
    render_candidates(only56_near, mols, output)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    ax = axes[0]
    counts = [1000, 574, 452, 34, 12]
    ax.barh(range(5), counts, color=["#8297ab", "#4879aa", "#225c8e", "#cc7832", "#49816e"])
    ax.set_yticks(range(5), ["Attempted", "RDKit valid", "XTB/sTDA complete", "Within 0.1 eV", "Within 0.1 eV + only 5/6"])
    ax.invert_yaxis(); ax.set_xlim(0, 1140)
    for j, n in enumerate(counts): ax.text(n + 12, j, str(n), va="center")
    ax.set_title("Counts (denominator = 1,000)")
    ax = axes[1]
    bins = np.arange(-5.25, 1.501, .25)
    for pred, label, color in [(True, "Only 5/6 rings (n=195)", "#247a80"), (False, "Other ring sizes (n=257)", "#b66547")]:
        vals = [r["q"] for r in success if r["only56"] == pred]
        ax.hist(vals, bins=bins, weights=np.ones(len(vals)) / len(vals), histtype="step", linewidth=2, label=label, color=color)
    ax.axvspan(-.1, .1, color="#83ae76", alpha=.2); ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("S1 - 2*T1 (eV)"); ax.set_ylabel("Fraction per 0.25-eV bin")
    ax.set_title("Energy distribution by ring composition"); ax.legend(fontsize=8)
    ax = axes[2]
    labels = ["Only 5/6", "3/4, no 7+", "7+, no 3/4", "3/4 and 7+"]
    x = np.arange(4)
    ax.bar(x - .18, [100*g["target_hit_rate"] for g in groups], width=.36, label="Within 0.1 eV", color="#49816e")
    ax.bar(x + .18, [100*g["postopt_plane_warnings"]/g["n"] for g in groups], width=.36, label="Aromatic plane warning", color="#b66547")
    ax.set_xticks(x, labels, rotation=22, ha="right"); ax.set_ylabel("Within-group rate (%)")
    ax.set_title("Target agreement and geometry are distinct"); ax.legend(fontsize=8)
    fig.suptitle("HP1KG e4690 | target S1 - 2*T1 = 0 eV | CFG 1 | saved-run reanalysis", fontsize=13)
    fig.savefig(output / "reanalysis_overview.png", dpi=180)
    fig.savefig(output / "reanalysis_overview.pdf")
    plt.close(fig)
    status.write_text(json.dumps(dict(phase="complete", finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=0), indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().output.resolve())
