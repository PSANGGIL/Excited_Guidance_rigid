#!/usr/bin/env python3
"""Read-only analysis of saved generation/XTB artifacts; writes new reports only."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'runs/generation_stage8_hp1kg_best_e1321_est0_n1000_cfg1_20261021'
XTB = GEN / 'xtb_stda_diffusion_coords_24x1_20261021'
DB = ROOT / 'joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz'
OUT = ROOT / 'analysis/hp1kg_e1321_full1000_structure_energy_20261021'
sys.path.insert(0, str(ROOT / 'MolGuidance_guidance_selector_e987/tools'))
from analyze_structure_xtb_detail import atom_signature, bond_signature, scaffold_smiles


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()),
                q05=float(np.quantile(a, .05)), q95=float(np.quantile(a, .95)))


def features(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(smiles)
    rings = list(m.GetRingInfo().AtomRings())
    sizes = [len(r) for r in rings]
    charges = [a.GetFormalCharge() for a in m.GetAtoms()]
    return dict(
        smiles=Chem.MolToSmiles(m, isomericSmiles=False), mol=m,
        heavy=m.GetNumHeavyAtoms(), rings=len(rings), ring_sizes=sizes,
        aromatic=any(a.GetIsAromatic() for a in m.GetAtoms()),
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(m),
        ringless=not sizes, only56=bool(sizes) and all(s in (5, 6) for s in sizes),
        small=any(s <= 4 for s in sizes), seven_plus=any(s >= 7 for s in sizes),
        nine_plus=any(s >= 9 for s in sizes),
        bridgehead=rdMolDescriptors.CalcNumBridgeheadAtoms(m) > 0,
        spiro=rdMolDescriptors.CalcNumSpiroAtoms(m) > 0,
        cumulene=any(sum(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds()) >= 2 for a in m.GetAtoms()),
        net_charge=sum(charges), charged_atoms=any(charges),
        zwitterionic=any(c > 0 for c in charges) and any(c < 0 for c in charges),
        radicals=sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()),
        elements=Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() > 1),
        scaffold=scaffold_smiles(m),
    )


FLAGS = ['aromatic', 'ringless', 'only56', 'small', 'seven_plus', 'nine_plus',
         'bridgehead', 'spiro', 'cumulene', 'charged_atoms', 'zwitterionic']


def describe(rows):
    ring_hist = Counter(s for r in rows for s in r['ring_sizes'])
    elem_atoms, elem_mols = Counter(), Counter()
    for r in rows:
        elem_atoms.update(r['elements'])
        elem_mols.update(r['elements'].keys())
    d = dict(n=len(rows), counts={k:sum(bool(r[k]) for r in rows) for k in FLAGS},
             heavy_atoms=stats([r['heavy'] for r in rows]),
             ring_count=stats([r['rings'] for r in rows]),
             aromatic_ring_count=stats([r['aromatic_rings'] for r in rows]),
             ring_sizes=dict(sorted(ring_hist.items())),
             fraction56_rings=(ring_hist[5]+ring_hist[6])/max(1,sum(ring_hist.values())),
             element_atom_counts=dict(elem_atoms), element_molecule_counts=dict(elem_mols),
             unique_smiles=len({r['smiles'] for r in rows}),
             unique_nonempty_scaffolds=len({r['scaffold'] for r in rows if r['scaffold']}))
    energy_rows = [r for r in rows if 'est' in r]
    if energy_rows:
        d.update(energy_evaluated_n=len(energy_rows),est=stats([r['est'] for r in energy_rows]),
                 abs_est=stats([abs(r['est']) for r in energy_rows]),
                 s1=stats([r['s1'] for r in energy_rows]), t1=stats([r['t1'] for r in energy_rows]),
                 negative=sum(r['est'] < 0 for r in energy_rows),
                 hits={str(t):sum(abs(r['est_exact']) < Decimal(str(t)) for r in energy_rows)
                       for t in (.05,.1,.25,.5)},
                 inclusive_0p1=sum(abs(r['est_exact']) <= Decimal('.1') for r in energy_rows))
    oscillator_rows = [r for r in rows if 'f' in r]
    if oscillator_rows:
        d.update(s1_f=stats([r['f'] for r in oscillator_rows]),
                 f_zero=sum(r['f']==0 for r in oscillator_rows),
                 f_lt_0p001=sum(r['f']<.001 for r in oscillator_rows),
                 f_ge_0p01=sum(r['f']>=.01 for r in oscillator_rows),
                 s1_below_0p5=sum(r['s1']<.5 for r in oscillator_rows),
                 s1_below_1=sum(r['s1']<1 for r in oscillator_rows),
                 s1_1p5_to_4=sum(1.5<=r['s1']<=4 for r in oscillator_rows))
    return d


def geometry_audit(index):
    work = XTB / 'work' / f'generated_{index:06d}'
    inp = Chem.MolFromMolFile(str(XTB/'inputs'/f'generated_{index:06d}.sdf'), removeHs=False, sanitize=True)
    opt = Chem.MolFromMolFile(str(work/'xtbopt.sdf'), removeHs=False, sanitize=False)
    if inp is None or opt is None:
        raise ValueError(f'Cannot read geometry {index}')
    assert [a.GetAtomicNum() for a in inp.GetAtoms()] == [a.GetAtomicNum() for a in opt.GetAtoms()]
    heavy = [a.GetIdx() for a in inp.GetAtoms() if a.GetAtomicNum() > 1]
    x, y = inp.GetConformer().GetPositions()[heavy], opt.GetConformer().GetPositions()[heavy]
    xc, yc = x-x.mean(0), y-y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    correction = np.eye(3)
    correction[-1,-1] = np.linalg.det(u @ vt)
    displacement = xc @ (u @ correction @ vt) - yc
    pos = opt.GetConformer().GetPositions()
    all_edges = {tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in inp.GetBonds()}
    edges = {pair for pair in all_edges if all(inp.GetAtomWithIdx(i).GetAtomicNum()>1 for i in pair)}
    wbo = {}
    for line in (work/'wbo').read_text().splitlines():
        a,b,v = line.split()
        wbo[tuple(sorted((int(a)-1,int(b)-1)))] = float(v)
    weak = [pair for pair in edges if wbo.get(pair,0.) < .3]
    new = [pair for pair,v in wbo.items() if v > .5 and pair not in edges
           and all(inp.GetAtomWithIdx(i).GetAtomicNum()>1 for i in pair)]
    h_transfer = []
    for a in inp.GetAtoms():
        if a.GetAtomicNum()!=1 or a.GetDegree()!=1:
            continue
        old = tuple(sorted((a.GetIdx(), a.GetNeighbors()[0].GetIdx())))
        if wbo.get(old,0.) < .3:
            alternatives = [p for p,v in wbo.items() if a.GetIdx() in p and p != old and v > .5]
            if alternatives:
                h_transfer.append(a.GetIdx())
    periodic = Chem.GetPeriodicTable()
    ratios = [float(np.linalg.norm(pos[a]-pos[b])) /
              (periodic.GetRcovalent(inp.GetAtomWithIdx(a).GetAtomicNum())+
               periodic.GetRcovalent(inp.GetAtomWithIdx(b).GetAtomicNum())) for a,b in edges]
    planes = []
    for ring in inp.GetRingInfo().AtomRings():
        if all(inp.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            c = pos[list(ring)]-pos[list(ring)].mean(0)
            _,_,v = np.linalg.svd(c,full_matrices=False)
            planes.append(float(np.sqrt(np.mean((c @ v[-1])**2))))
    log = (work/'xtb_opt.log').read_text(errors='replace')
    convergence = re.search(r'GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS', log)
    triplet = bool(re.search(r'triplet\s*:\s*T', (work/'stda_triplet.log').read_text()))
    return dict(generation_index=index, optimization_converged=bool(convergence),
                optimization_iterations=int(convergence.group(1)) if convergence else None,
                triplet_flag_verified=triplet,
                heavy_aligned_rmsd_A=float(np.sqrt(np.mean(np.sum(displacement**2,axis=1)))),
                max_heavy_displacement_A=float(np.sqrt(np.sum(displacement**2,axis=1)).max()),
                weak_original_heavy_bonds=len(weak), new_wbo_heavy_contacts=len(new),
                hydrogen_transfer_warnings=len(h_transfer),
                weak_original_heavy_bond_pairs=weak,new_wbo_heavy_pairs=new,
                min_original_heavy_bond_wbo=min(wbo.get(p,0.) for p in edges),
                max_postopt_bond_radius_ratio=max(ratios),
                max_postopt_aromatic_plane_rms_A=max(planes,default=0.))


def csv_output(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('x', newline='') as handle:
        w = csv.DictWriter(handle,fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def main():
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    OUT.mkdir(exist_ok=True)
    for filename in ('summary.json','all_1000_structure_energy.csv','xtb_geometry_audit_659.csv','near_zero_novel_unique_43.csv'):
        if (OUT/filename).exists():
            raise FileExistsError(OUT/filename)
    raw = json.loads((GEN/'raw_analysis/records.json').read_text())
    assert len(raw)==1000 and len({r['generation_index'] for r in raw})==1000
    with (XTB/'results.csv').open() as handle:
        calculations = {int(r['generation_index']):r for r in csv.DictReader(handle)}
    train, all_db = [], []
    source_db = []
    for split in ('train','val','test'):
        data = torch.load(DB/f'{split}.pt',map_location='cpu',weights_only=False)
        for r in data:
            d = features(r['smiles'])
            est = Decimal(str(r['s1']))-Decimal(str(r['t1']))
            d.update(id=r['id'],split=split,s1=r['s1'],t1=r['t1'],est=float(est),est_exact=est)
            all_db.append(d)
            if split=='train':train.append(d)
        source_db.append(DB/f'{split}.pt')
    db_smiles = {d['smiles'] for d in all_db}
    supported_atoms = {atom_signature(a) for r in train for a in r['mol'].GetAtoms()}
    supported_bonds = {bond_signature(b) for r in train for b in r['mol'].GetBonds()}
    fps = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    train_fps = [fps.GetFingerprint(r['mol']) for r in train]
    valid, calculated, geometry_rows, ledgers = [], [], [], []
    for row in raw:
        index = row['generation_index']
        if not row['raw_valid']:
            ledgers.append(dict(row))
            continue
        d = features(row['smiles'])
        assert d['heavy']==row['heavy_atoms']
        assert d['smiles'] in db_smiles if row['db_member'] else d['smiles'] not in db_smiles
        d.update(index=index, eligible=row['eligible'], geometry_screen=row['geometry_screen'],
                 topology_screen=row['topology_screen'], novel_unique=row['novel_unique'],db_member=row['db_member'],
                 requested_heavy_atoms=row['requested_heavy_atoms'],
                 preopt_plane_rms_A=row.get('max_aromatic_plane_rms_angstrom',0.))
        d['unsupported_atom_count'] = sum(atom_signature(a) not in supported_atoms for a in d['mol'].GetAtoms())
        d['unsupported_bond_count'] = sum(bond_signature(b) not in supported_bonds for b in d['mol'].GetBonds())
        similarities = DataStructs.BulkTanimotoSimilarity(fps.GetFingerprint(d['mol']), train_fps)
        d['nearest_train_tanimoto'] = max(similarities)
        d['nearest_train_id'] = train[int(np.argmax(similarities))]['id']
        if index in calculations:
            c=calculations[index]
            assert c['status']=='ok' and row['eligible']
            est=Decimal(c['s1_energy_eV'])-Decimal(c['t1_energy_eV'])
            d.update(s1=float(c['s1_energy_eV']),t1=float(c['t1_energy_eV']),
                     est=float(est),est_exact=est,f=float(c['s1_f_osc']))
            audit=geometry_audit(index)
            geometry_rows.append(audit)
            d.update(audit)
            calculated.append(d)
        valid.append(d)
        ledgers.append({**row, **{k:v for k,v in d.items() if k!='mol'}})
    assert len(calculated)==659
    selected=[r for r in calculated if abs(r['est_exact'])<Decimal('.1')]
    unique_selected=[];seen=set()
    for r in sorted(selected,key=lambda r:(abs(r['est']),r['index'])):
        if not r['db_member'] and r['smiles'] not in seen:
            unique_selected.append(r);seen.add(r['smiles'])
    groups=dict(train_db=train,full_db=all_db,train_db_abs_est_lt_0p1=[r for r in train if abs(r['est_exact'])<Decimal('.1')],
                raw_valid=valid,raw_eligible=[r for r in valid if r['eligible']],
                calculated=calculated,near_zero_records=selected,near_zero_novel_unique=unique_selected)
    summaries={name:describe(rows) for name,rows in groups.items()}
    for name,rows in [('aromatic',[r for r in calculated if r['aromatic']]),
                      ('nonaromatic',[r for r in calculated if not r['aromatic']]),
                      ('small_ring',[r for r in calculated if r['small']]),
                      ('nine_plus_ring',[r for r in calculated if r['nine_plus']]),
                      ('only56',[r for r in calculated if r['only56']]),
                      ('zwitterionic',[r for r in calculated if r['zwitterionic']]),
                      ('no_atomic_charges',[r for r in calculated if not r['charged_atoms']]),
                      ('geometry_pass',[r for r in calculated if r['geometry_screen']]),
                      ('geometry_fail',[r for r in calculated if not r['geometry_screen']])]:
        summaries['calculated_'+name]=describe(rows)
    summary=dict(generation_manifest=json.loads((GEN/'manifest.json').read_text()),
                 raw_summary={k:v for k,v in json.loads((GEN/'raw_analysis/summary.json').read_text()).items()
                              if not isinstance(v,(dict,list))},groups=summaries)
    summary['failed_graph_errors']=dict(Counter(re.sub(r'atom # \d+', 'atom # INDEX',r['error']) for r in raw if not r['raw_valid']))
    summary['support']={name:dict(n=len(rows),all_atoms_supported=sum(r['unsupported_atom_count']==0 for r in rows),
                                  all_bonds_supported=sum(r['unsupported_bond_count']==0 for r in rows),
                                  unsupported_atom_fraction=stats([r['unsupported_atom_count']/r['heavy'] for r in rows]),
                                  nearest_train_similarity=stats([r['nearest_train_tanimoto'] for r in rows]))
                        for name,rows in [('valid',valid),('calculated',calculated),('near_zero_unique',unique_selected)]}
    summary['geometry_audit']=dict(
        count=len(geometry_rows),converged=sum(r['optimization_converged'] for r in geometry_rows),
        triplet_flag_verified=sum(r['triplet_flag_verified'] for r in geometry_rows),
        iterations=stats([r['optimization_iterations'] for r in geometry_rows if r['optimization_iterations'] is not None]),
        aligned_heavy_rmsd_A=stats([r['heavy_aligned_rmsd_A'] for r in geometry_rows]),
        rmsd_above_0p5=sum(r['heavy_aligned_rmsd_A']>.5 for r in geometry_rows),
        rmsd_above_1=sum(r['heavy_aligned_rmsd_A']>1 for r in geometry_rows),
        weak_original_bond_molecules=sum(r['weak_original_heavy_bonds']>0 for r in geometry_rows),
        new_heavy_contact_molecules=sum(r['new_wbo_heavy_contacts']>0 for r in geometry_rows),
        h_transfer_warning_molecules=sum(r['hydrogen_transfer_warnings']>0 for r in geometry_rows),
        postopt_original_bond_ratio_above_1p4=sum(r['max_postopt_bond_radius_ratio']>1.4 for r in geometry_rows),
        postopt_aromatic_plane_rms_above_0p15=sum(r['max_postopt_aromatic_plane_rms_A']>.15 for r in geometry_rows),
        caveat='WBO <0.3 on original heavy edges and >0.5 on new heavy contacts are heuristic warnings, not a chemical identity certificate.')
    summary['count_control']=dict(requested=stats([r['requested_heavy_atoms'] for r in raw]),
        sampled=stats([r['sampled_real_atoms'] for r in raw]),
        exact=sum(r['sampled_real_atoms']==r['requested_heavy_atoms'] for r in raw),
        mean_signed_difference=float(np.mean([r['sampled_real_atoms']-r['requested_heavy_atoms'] for r in raw])))
    summary['rounding_boundary_ids']=[r['index'] for r in calculated if abs(r['est_exact'])==Decimal('.1')]
    summary['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
       [GEN/'raw_analysis/records.json',XTB/'results.csv',*source_db]}
    summary['interpretation_caveats']=[
        'No new generation, XTB calculation, or training changes were performed.',
        'RDKit ring counts use its perceived ring basis, not all possible graph cycles.',
        'Structural warning thresholds are not absolute chemical impossibility tests.',
        'Train DB uses NEVPT2 labels; generated energies are uncalibrated XTB/sTDA.',
        'Loss of an original graph edge or appearance of a new WBO contact needs individual geometry verification.',
        'A small XTB/sTDA EST alone does not establish experimental inverted singlet-triplet ordering or fluorescence.',
        'Unique near-zero entries select the smallest absolute EST among repeated generated graphs; not independent unique draws.',
    ]
    csv_output(OUT/'all_1000_structure_energy.csv',ledgers)
    csv_output(OUT/'xtb_geometry_audit_659.csv',geometry_rows)
    csv_output(OUT/'near_zero_novel_unique_43.csv',[{k:v for k,v in r.items() if k!='mol'} for r in unique_selected])
    with (OUT/'summary.json').open('x') as handle:
        json.dump(summary,handle,indent=2,default=str)
    print(json.dumps({'output':str(OUT),'groups':{k:{f:v for f,v in d.items() if f in
          ['n','counts','heavy_atoms','ring_count','fraction56_rings','est','abs_est','negative','hits','inclusive_0p1','s1_f','f_zero','f_lt_0p001','f_ge_0p01','s1_below_0p5','s1_1p5_to_4']}
          for k,d in summaries.items()},'geometry_audit':summary['geometry_audit']},indent=2))


if __name__=='__main__':
    main()
