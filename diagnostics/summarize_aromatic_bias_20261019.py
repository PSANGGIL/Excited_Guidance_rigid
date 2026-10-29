"""Read-only model diagnostics: summarize independently stored experimental outputs.

No training, production decoding, or source results are changed. All counterfactual
structures are explicitly diagnostic and must not be counted as generation successes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor, rdMolDescriptors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'e1074': ROOT/'runs/cfg_paired_e1074_est0_n1000_seed20261019/cfg1',
    'e1713': ROOT/'runs/generation_stage6_best_e1713_est0_n1000_cfg1_seed20261019',
    'e1944': ROOT/'runs/generation_stage6_best_e1944_est0_n1000_cfg1_seed20261019',
}
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')


def json_save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')


def wilson(success, total):
    if not total:
        return (None, None)
    z = 1.959963984540054
    p = success/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return center-half, center+half


def aggregate(frame, groups, metrics):
    output = []
    for keys, part in frame.groupby(groups, dropna=False, sort=False):
        if len(groups) == 1:
            keys = (keys,) if not isinstance(keys, tuple) else keys
        row = dict(zip(groups, keys))
        row['n'] = len(part)
        for metric in metrics:
            if metric not in part:
                continue
            values = part[metric].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            row[metric+'_n'] = len(values)
            row[metric+'_mean'] = values.mean() if len(values) else np.nan
            if len(values) and set(values.unique()).issubset({0.,1.}):
                row[metric+'_count'] = int(values.sum())
                row[metric+'_ci_low'], row[metric+'_ci_high'] = wilson(int(values.sum()), len(values))
        output.append(row)
    return pd.DataFrame(output)


def category(mol):
    return 'aromatic' if rdMolDescriptors.CalcNumAromaticRings(mol) else ('ringless' if not rdMolDescriptors.CalcNumRings(mol) else 'nonaromatic_cyclic')


def sanitize(raw):
    if raw is None:
        return None, 'read_failed'
    mol = Chem.Mol(raw)
    try:
        Chem.SanitizeMol(mol)
        return mol, ''
    except Exception as error:
        return None, type(error).__name__


def existing(output):
    structure_rows, problem_rows, sensitivity_rows, energy_rows = [], [], [], []
    for label, run in RUNS.items():
        metadata = pd.read_csv(run/'analysis/molecules.csv').set_index('generation_index')
        pictures, legends = [], []
        for index, raw in enumerate(Chem.SDMolSupplier(str(run/'samples.sdf'), sanitize=False, removeHs=False, strictParsing=False)):
            mol, error = sanitize(raw)
            record = {'checkpoint':label, 'generation_index':index, 'valid':mol is not None, 'error':error}
            if raw is not None:
                raw.UpdatePropertyCache(strict=False)
                rings = list(Chem.GetSymmSSSR(raw))
                ring_atoms = set(atom for ring in rings for atom in ring)
                record.update(atoms=raw.GetNumHeavyAtoms(), raw_cyclic=bool(rings), raw_ring_sizes=';'.join(str(len(ring)) for ring in rings),
                              atom_bin='<=15' if raw.GetNumHeavyAtoms() <= 15 else ('16-25' if raw.GetNumHeavyAtoms() <= 25 else '>=26'))
                if mol is not None:
                    record['category'] = category(mol)
                else:
                    for problem in Chem.DetectChemistryProblems(raw):
                        if not hasattr(problem, 'GetAtomIdx'):
                            continue
                        atom_index = problem.GetAtomIdx()
                        atom = raw.GetAtomWithIdx(atom_index)
                        problem_rows.append({'checkpoint':label, 'generation_index':index, 'problem':problem.GetType(),
                            'atom_index':atom_index, 'element':atom.GetSymbol(), 'charge':atom.GetFormalCharge(),
                            'explicit_h':atom.GetNumExplicitHs(), 'bond_order_sum':sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()),
                            'ring_atom':atom_index in ring_atoms, 'rings_at_atom':';'.join(str(len(r)) for r in rings if atom_index in r),
                            'message':problem.Message()})
                    # Sensitivity only: relax serialized H-count constraints, retaining
                    # elements, formal charges and every heavy-atom edge/bond order.
                    relaxed = Chem.Mol(raw)
                    for atom in relaxed.GetAtoms():
                        atom.SetNumExplicitHs(0)
                        atom.SetNoImplicit(False)
                    rescued, rescue_error = sanitize(relaxed)
                    sensitivity_rows.append({'checkpoint':label, 'generation_index':index, 'valid_after_H_relaxation':rescued is not None,
                        'aromatic_after_H_relaxation':rescued is not None and category(rescued)=='aromatic',
                        'error_after_H_relaxation':rescue_error,
                        'diagnostic_only_not_generation_success':True,
                        'smiles_after_H_relaxation':Chem.MolToSmiles(rescued) if rescued is not None else ''})
                    if rescued is not None and len(pictures) < 12:
                        display = Chem.Mol(rescued)
                        rdDepictor.Compute2DCoords(display, clearConfs=True)
                        pictures.append(display)
                        legends.append(f'{label} id={index}: H-relaxed diagnostic ONLY')
            record['csv_valid_agrees'] = bool(metadata.loc[index, 'valid']) == record['valid']
            structure_rows.append(record)
        if pictures:
            Draw.MolsToGridImage(pictures, legends=legends, molsPerRow=3, subImgSize=(380,300)).save(output/f'{label}_H_relaxation_diagnostic_only.png')
        energy = pd.read_csv(run/'xtb_stda_diffusion_coords_24x1/work/results.csv')
        energy = energy[energy.status=='ok'].copy()
        energy['checkpoint'] = label
        energy['category'] = np.where(energy.aromatic_rings>0,'aromatic',np.where(energy.rings==0,'ringless','nonaromatic_cyclic'))
        energy['est'] = energy.s1_energy_eV-energy.t1_energy_eV
        energy['near_zero'] = energy.est.abs()<=.100001
        energy_rows.extend(energy[['checkpoint','generation_index','category','est','near_zero']].to_dict('records'))
    structures = pd.DataFrame(structure_rows)
    structures.to_csv(output/'original_structure_failures.csv',index=False)
    pd.DataFrame(problem_rows).to_csv(output/'original_valence_problem_atoms.csv',index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output/'serialized_H_sensitivity_DIAGNOSTIC_ONLY.csv',index=False)
    aggregate(sensitivity,['checkpoint'],['valid_after_H_relaxation','aromatic_after_H_relaxation']).to_csv(output/'serialized_H_sensitivity_summary.csv',index=False)
    aggregate(structures,['checkpoint','raw_cyclic'],['valid','csv_valid_agrees']).to_csv(output/'cyclic_validity.csv',index=False)
    aggregate(structures,['checkpoint','atom_bin','raw_cyclic'],['valid']).to_csv(output/'cyclic_validity_by_atom_count.csv',index=False)
    energy = pd.DataFrame(energy_rows)
    energy.to_csv(output/'existing_energy_by_ring_class.csv',index=False)
    energy_summary = aggregate(energy,['checkpoint','category'],['near_zero','est'])
    energy_summary.to_csv(output/'existing_target_hit_by_ring_class.csv',index=False)
    tests=[]
    for label, part in structures.groupby('checkpoint'):
        yes = part[part.raw_cyclic==True]
        no = part[part.raw_cyclic==False]
        table=[[int(yes.valid.sum()), int((~yes.valid).sum())], [int(no.valid.sum()),int((~no.valid).sum())]]
        result=fisher_exact(table)
        tests.append({'checkpoint':label, 'cyclic_vs_acyclic_validity_odds_ratio':float(result.statistic), 'p':float(result.pvalue), 'table':table,
                      'caution':'observational, not aromatic/nonaromatic; atom count is stratified in a separate CSV'})
    json_save(output/'cyclic_validity_fisher.json',tests)
    return structures, energy_summary


def database(source, output):
    rows=pd.read_csv(source/'source_records.csv')
    rows['category']=np.where(rows.source_aromatic,'aromatic',np.where(rows.source_ringless,'ringless','nonaromatic_cyclic'))
    old=pd.read_csv(ROOT/'runs/hp1kg_nevpt2_db_xtb_stda_all_20260919/results.csv')
    joined=rows.merge(old[['split','split_index','reference_s1_eV','reference_t1_eV','status','calculated_s1_eV','calculated_t1_eV']],on=['split','split_index'],validate='one_to_one')
    joined['reference_gap_check']=(joined.reference_s1_eV-joined.reference_t1_eV-joined.est).abs()
    if joined.reference_gap_check.max()>1e-6 or len(joined)!=1581:
        raise RuntimeError('Old DB calculation mapping does not match current labels')
    joined['xtb_est']=joined.calculated_s1_eV-joined.calculated_t1_eV
    joined['xtb_near_zero']=(joined.xtb_est.abs()<=.100001)&(joined.status=='ok')
    joined.to_csv(output/'database_ring_EST.csv',index=False)
    counts=[]
    for scope,part in [('whole',joined),('train',joined[joined.split=='train']),('heldout',joined[joined.split!='train'])]:
        for threshold in [.01,.05,.1,.5]:
            for level,column in [('reference','est'),('xtb','xtb_est')]:
                selected=part[part[column].abs()<=threshold+1e-6]
                counts.append({'scope':scope,'abs_est_max':threshold,'energy_level':level,'n':len(selected),**selected.category.value_counts().to_dict()})
    pd.DataFrame(counts).fillna(0).to_csv(output/'database_ring_EST_counts.csv',index=False)


def endpoints(source,output):
    paths=list(source.glob('endpoint_e*.csv'))
    if not paths:
        return None
    data=pd.concat([pd.read_csv(path,dtype={'condition':str},low_memory=False) for path in paths],ignore_index=True)
    matched=pd.read_csv(source/'matched_sources.csv')
    matched_ids=set(matched.record_id)
    matched_pairs=[]
    for i in range(0,len(matched),2):
        left,right=matched.iloc[i:i+2].to_dict('records')
        matched_pairs.append({'pair':i//2,'nonaromatic_record_id':left['record_id'],'aromatic_record_id':right['record_id'],
            'split':left['split'],'nonaromatic_atoms':left['source_atoms'],'aromatic_atoms':right['source_atoms'],
            'atom_count_difference':right['source_atoms']-left['source_atoms'],
            'nonaromatic_est':left['est'],'aromatic_est':right['est']})
    pair_frame=pd.DataFrame(matched_pairs)
    pair_frame.to_csv(output/'matched_pair_balance.csv',index=False)
    exact=pair_frame[pair_frame.atom_count_difference==0]
    exact_ids=set(exact.nonaromatic_record_id)|set(exact.aromatic_record_id)
    common=['checkpoint','split','source_aromatic','coordinate_time','graph_time','condition','intervention']
    metrics=['valid','connected','aromatic','ringless','identity','cycles_lost_fake_removal','fake_removed']
    for name,part in [('all',data),('near_zero',data[data.near_zero]),('matched',data[data.record_id.isin(matched_ids)]),('exact_count_matched',data[data.record_id.isin(exact_ids)])]:
        aggregate(part,common,metrics).to_csv(output/f'endpoint_{name}_summary.csv',index=False)
        aggregate(part,['checkpoint','source_aromatic','coordinate_time','graph_time','condition','intervention'],metrics).to_csv(output/f'endpoint_{name}_pooled.csv',index=False)
    keys=['checkpoint','record_id','coordinate_time','graph_time','condition']
    baseline=data[data.intervention=='masked_fill'][keys+['valid','aromatic','identity']].rename(columns={x:'baseline_'+x for x in ['valid','aromatic','identity']})
    paired=data.merge(baseline,on=keys,validate='many_to_one')
    contrasts=[]
    group=['checkpoint','source_aromatic','coordinate_time','graph_time','condition','intervention']
    for keys_,part in paired.groupby(group,sort=False):
        row=dict(zip(group,keys_));row['n']=len(part)
        for field in ['valid','aromatic','identity']:
            gained=(part[field]&~part['baseline_'+field]).sum()
            lost=(~part[field]&part['baseline_'+field]).sum()
            row[field+'_rescued']=int(gained);row[field+'_lost']=int(lost)
            row[field+'_paired_delta']=(gained-lost)/len(part)
            row[field+'_mcnemar_exact_p']=float(binomtest(int(gained),int(gained+lost)).pvalue) if gained+lost else 1.
        contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(output/'oracle_intervention_contrasts.csv',index=False)
    metric_paths=list(source.glob('endpoint_metrics_e*.csv'))
    if metric_paths:
        met=pd.concat([pd.read_csv(path,dtype={'condition':str},low_memory=False) for path in metric_paths],ignore_index=True)
        cols=['masked_aromatic_edges','aromatic_edge_missing','aromatic_edge_order_accuracy','masked_positive_edge_recall',
              'masked_edge_nll','masked_supported_state_nll','unsupported_target_nodes','aromatic_atom_deleted','true_atom_deleted_fraction']
        aggregate(met,['checkpoint','split','source_aromatic','coordinate_time','graph_time','condition'],cols).to_csv(output/'denoising_error_summary.csv',index=False)
    return data


def new_generations(source,output):
    free_path=source/'free_generation.csv'
    if not free_path.exists():
        return None
    free=pd.read_csv(free_path)
    summary=aggregate(free,['arm'],['valid','connected','aromatic','ringless','fake_removed','cycles_lost_fake_removal','atoms','charge','radicals'])
    summary.to_csv(output/'free_generation_summary_with_intervals.csv',index=False)
    base=free[free.arm=='baseline'][['seed','sample_index','valid','aromatic','ringless']].rename(columns={x:'base_'+x for x in ['valid','aromatic','ringless']})
    paired=free.merge(base,on=['seed','sample_index'],validate='many_to_one')
    contrasts=[]
    for arm,part in paired.groupby('arm'):
        row={'arm':arm,'n':len(part)}
        for field in ['valid','aromatic','ringless']:
            gains=int((part[field]&~part['base_'+field]).sum());losses=int((~part[field]&part['base_'+field]).sum())
            row[field+'_gained']=gains;row[field+'_lost']=losses
            row[field+'_delta']=(gains-losses)/len(part)
            row[field+'_mcnemar_exact_p']=float(binomtest(gains,gains+losses).pvalue) if gains+losses else 1.
        contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(output/'free_generation_paired_contrasts.csv',index=False)
    trace=pd.read_csv(source/'trajectory.csv')
    aggregate(trace,['arm','progress'],['valid','aromatic','ringless','cycles_lost_fake_removal','masked_states','masked_edges']).to_csv(output/'conditional_endpoint_trajectory_summary.csv',index=False)
    terminal=pd.read_csv(source/'terminal_decoding_counterfactuals.csv')
    aggregate(terminal,['arm','view'],['valid','aromatic','ringless','cycles_lost_fake_removal']).to_csv(output/'terminal_endpoint_summary.csv',index=False)
    png_dir=output/'rdkit_2d'
    png_dir.mkdir(exist_ok=True)
    for path in sorted((source/'free_generation').glob('*/*.sdf')):
        molecules=[];legends=[]
        for molecule in Chem.SDMolSupplier(str(path),removeHs=False):
            if molecule is None:
                continue
            rdDepictor.Compute2DCoords(molecule,clearConfs=True)
            molecules.append(molecule)
            legends.append(f'{path.parent.name} {molecule.GetProp("generation_index")}')
        if molecules:
            Draw.MolsToGridImage(molecules,legends=legends,molsPerRow=4,subImgSize=(320,270)).save(png_dir/f'{path.parent.name}_{path.stem}.png')
    return summary


def plots(output,structures,energy_summary,data):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    labels=list(RUNS)
    x=np.arange(len(labels))
    for offset,cyclic,label in [(-.18,True,'Raw cyclic'),(.18,False,'Raw ringless')]:
        rates=[]
        for checkpoint in labels:
            part=structures[(structures.checkpoint==checkpoint)&(structures.raw_cyclic==cyclic)]
            rates.append(part.valid.mean()*100)
        axes[0].bar(x+offset,rates,width=.35,label=label)
    axes[0].set_xticks(x,labels);axes[0].set_ylim(0,105);axes[0].set_ylabel('RDKit valid (%)');axes[0].legend()
    axes[0].set_title('Cyclic graph validity penalty (not aromatic-specific)')
    for offset,kind in [(-.24,'aromatic'),(0.,'nonaromatic_cyclic'),(.24,'ringless')]:
        rates=[energy_summary[(energy_summary.checkpoint==label)&(energy_summary.category==kind)].near_zero_mean.iloc[0]*100 for label in labels]
        axes[1].bar(x+offset,rates,width=.23,label=kind)
    axes[1].set_xticks(x,labels);axes[1].set_ylim(0,105);axes[1].set_ylabel('Within-class |calculated EST| <= 0.1 eV (%)')
    axes[1].set_title('Target-window selection among XTB successes');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'failure_and_target_selection.png',dpi=170);plt.close(fig)
    if data is not None:
        part=data[(data.checkpoint=='e1944')&(data.intervention=='masked_fill')&(data.condition=='truth')&(data.coordinate_time==data.graph_time)]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for aromatic,label in [(True,'Aromatic source'),(False,'Nonaromatic source')]:
            selected=part[part.source_aromatic==aromatic]
            grouped=selected.groupby('graph_time')[['valid','identity']].mean()
            for ax,metric in zip(axes,['valid','identity']):
                ax.plot(grouped.index,grouped[metric]*100,'o-',label=label)
                ax.set_xlabel('Flow time t (1 = clean, 0 = fully noisy)');ax.set_ylabel(metric+' (%)');ax.set_ylim(0,105);ax.legend()
        fig.suptitle('e1944: known DB molecules, masked-only reconstruction, all splits')
        fig.tight_layout();fig.savefig(output/'database_denoising_by_aromaticity.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    source=args.source.resolve()
    structures,energy=existing(output)
    database(source,output)
    data=endpoints(source,output)
    partial=source/'partial_reverse.csv'
    if partial.exists():
        aggregate(pd.read_csv(partial),['source_aromatic','start_time','condition'],['valid','aromatic','ringless','identity']).to_csv(output/'partial_reverse_pooled.csv',index=False)
    free=new_generations(source,output)
    plots(output,structures,energy,data)
    status=json.loads((source/'status.json').read_text())
    notes={
        'upstream_status':status,
        'sources':str(source),
        'observations':str(output),
        'validity_means':'RDKit sanitization, not physical stability or synthesis feasibility',
        'invalid_aromaticity':'Cannot infer intended aromaticity for an arbitrary invalid free sample; DB-corruption tests have known source aromaticity.',
        'H_sensitivity':'Counterfactual on serialized SDF, not original model tensors. H relaxation does not count as generation success and may not recover dropped serialization state.',
        'denoising':'Masked-fill keeps visible truth. Endpoint argmax replaces all values and is a different diagnostic. Partial reverse follows production stochastic steps.',
        'time':'0 is prior and 1 is clean, unlike common DDPM indexing.',
        'matching':'72 near-zero sources, 36 of each aromaticity; atom-count nearest-neighbour within split, not exact molecular match.',
        'exact_count_matching':'Separate exact-count panel: 15 pairs, 30 molecules; much smaller and not enough for definitive group claims.',
        'uncertainty':'Wilson intervals and exploratory paired McNemar tests. Many unadjusted tests and 32 samples/arm: not a definitive hyperparameter ranking.',
        'Kekule':'Exact bond-order accuracy is against one target assignment, not resonance-invariant; canonical RDKit identity is reported separately.',
        'trajectory':'Conditional endpoint argmax snapshots, not sanitization of partly masked states, nor the actual CFG-combined categorical sample.',
        'condition':'No new XTB calculated here. New free-generation arms test structure, not physical EST fidelity.',
        'threshold':'Inclusive 0.1 eV window with 1e-6 tolerance; e1074 hit count59 not earlier58; source DB231 not earlier227.',
        'model_changed':False,
    }
    json_save(output/'analysis_manifest.json',notes)
    lines=['AROMATIC FAILURE AND NEAR-ZERO EST BIAS DIAGNOSTICS',f'Experimental status: {status["state"]}','',
           'Raw cyclic versus ringless validity (cannot label invalid free graphs as aromatic):']
    for label,part in structures.groupby('checkpoint',sort=False):
        for cyclic in [True,False]:
            cohort=part[part.raw_cyclic==cyclic]
            lines.append(f'{label} raw_cyclic={cyclic}: {int(cohort.valid.sum())}/{len(cohort)} valid ({100*cohort.valid.mean():.1f}%)')
    lines.extend(['','XTB target hit rates by ring class:',energy[['checkpoint','category','n','near_zero_count','near_zero_mean']].to_string(index=False),
                  '', 'Interpretation: structural rejection and target-window selection are separate factors. Cyclic failures alone do not establish aromatic-specific failure.',
                  'Oracle input/output identity and same-initial-noise tests are saved alongside the experimental source. See the analysis manifest for assumptions and limitations.',
                  '', 'Experiments and all confidence intervals are in the CSVs. PNGs under rdkit_2d explicitly use RDKit 2D coordinates.'])
    (output/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':status['state'],'output':str(output),'endpoint_rows':len(data) if data is not None else 0,'free_arms':len(free) if free is not None else 0}))


if __name__=='__main__':
    main()
