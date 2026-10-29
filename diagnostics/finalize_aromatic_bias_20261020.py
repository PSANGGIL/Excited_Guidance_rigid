"""Consolidate frozen diagnostic outputs; no production data/model mutation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor
from scipy.stats import binomtest
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from summarize_aromatic_bias_20261019 import aggregate, ROOT, RUNS, json_save

sys.path.insert(0,str(ROOT/'MolGuidance_guidance_selector_e987'))
from molguidance.data_processing.conditional_count import ConditionalAtomCountSampler

RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')


def energetic_characterization(output):
    rows=[]
    for label,run in RUNS.items():
        frame=pd.read_csv(run/'xtb_stda_diffusion_coords_24x1/work/results.csv')
        frame=frame[frame.status=='ok'].copy()
        frame['checkpoint']=label
        frame['category']=np.where(frame.aromatic_rings>0,'aromatic',np.where(frame.rings==0,'ringless','nonaromatic_cyclic'))
        frame['est']=frame.s1_energy_eV-frame.t1_energy_eV
        frame['near_zero']=frame.est.abs()<=.100001
        for record in frame.to_dict('records'):
            molecule=Chem.MolFromSmiles(record['smiles'])
            charges=[atom.GetFormalCharge() for atom in molecule.GetAtoms()]
            record['zwitterionic']=any(x>0 for x in charges) and any(x<0 for x in charges)
            record['any_aromatic_atom']=any(atom.GetIsAromatic() for atom in molecule.GetAtoms())
            record['aromatic_definition_disagreement']=record['any_aromatic_atom']!=(record['aromatic_rings']>0)
            record['category_any_aromatic_atom']='aromatic' if record['any_aromatic_atom'] else ('ringless' if record['rings']==0 else 'nonaromatic_cyclic')
            record['achiral_smiles']=Chem.MolToSmiles(molecule,isomericSmiles=False)
            rows.append(record)
    energy=pd.DataFrame(rows)
    energy.to_csv(output/'xtb_structure_energy_rows.csv',index=False)
    aggregate(energy,['checkpoint','category_any_aromatic_atom'],['near_zero']).to_csv(output/'xtb_funnel_any_aromatic_atom.csv',index=False)
    energy[energy.aromatic_definition_disagreement].to_csv(output/'aromatic_definition_disagreements.csv',index=False)
    summary=energy[energy.near_zero].groupby(['checkpoint','category']).agg(
        n=('est','size'),achiral_unique=('achiral_smiles','nunique'),zwitterionic=('zwitterionic','sum'),
        s1_mean=('s1_energy_eV','mean'),s1_median=('s1_energy_eV','median'),t1_median=('t1_energy_eV','median'),
        est_mean=('est','mean'),est_median=('est','median'),fosc_median=('s1_f_osc','median'),
        heavy_atom_mean=('heavy_atoms','mean')).reset_index()
    summary.to_csv(output/'near_zero_structure_energy_summary.csv',index=False)
    return summary


def count_audit(base,output):
    sampler=ConditionalAtomCountSampler.from_processed_file(ROOT/'molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz/train_data_processed.pt')
    weights,diagnostics=sampler.probabilities([[0.]],.15,nearest_k=128)
    weights=weights[0]
    counts=np.load(ROOT/'runs/cfg_paired_e1074_est0_n1000_seed20261019/fixed_atom_counts.npy')
    previous=ROOT/'runs/generation_stage6_best_e1074_est0_n1000_cfg2_seed20261019/samples.sdf'
    molecules=list(Chem.SDMolSupplier(str(previous),sanitize=False,removeHs=False))
    decoded_counts=np.array([m.GetNumHeavyAtoms() if m is not None else -1 for m in molecules])
    result={
        'fixed_count_n':len(counts),'fixed_count_mean':float(counts.mean()),'fixed_count_min':int(counts.min()),'fixed_count_max':int(counts.max()),
        'matches_previous_generated_heavy_counts_exactly':bool(np.array_equal(counts,decoded_counts)),
        'previous_generation':str(previous),
        'fresh_conditional_mean':float((weights*sampler.atom_counts).sum()),
        'fresh_conditional_min':int(sampler.atom_counts[weights>0].min()),'fresh_conditional_max':int(sampler.atom_counts[weights>0].max()),
        'effective_reference_n':float(diagnostics.effective_sample_size[0]),
        'fixed_counts_absent_from_train':np.unique(counts[~np.isin(counts,sampler.atom_counts.numpy())]).tolist(),
        'interpretation':'Original checkpoint comparisons share the same input vector and are controlled relative comparisons. Their requested sizes do not represent fresh p(n|EST=0) from the current DB. A fixed --number_of_atoms overrides the nominal conditional-kernel option.',
    }
    json_save(output/'atom_count_provenance.json',result)
    source=pd.read_csv(base/'full_v2/source_records.csv')
    train=source[source.split=='train'].reset_index(drop=True)
    if not np.array_equal(train.split_index.to_numpy(),np.arange(len(weights))):
        raise RuntimeError('Train record order mismatch')
    train['count_kernel_weight']=weights.numpy()
    train['model_node_count']=sampler.atom_counts.numpy()
    train.to_csv(output/'train_count_kernel_reference.csv',index=False)
    return result


def replication(base,output):
    steps=base/'replication_steps_20261020'
    counts=base/'count_replication_20261020'
    statuses={name:json.loads((path/'status.json').read_text()) for name,path in [('steps',steps),('counts',counts)]}
    if any(item['state']!='complete' for item in statuses.values()):
        raise RuntimeError(f'Replication is not complete: {statuses}')
    a=pd.read_csv(steps/'molecules.csv')
    b=pd.read_csv(counts/'molecules.csv')
    all_rows=pd.concat([a,b],ignore_index=True)
    all_rows['any_aromatic_atom']=all_rows.smiles.fillna('').map(lambda value:bool(value) and any(atom.GetIsAromatic() for atom in Chem.MolFromSmiles(value).GetAtoms()))
    all_rows['neutral_closed_shell']=all_rows.valid & (all_rows.charge==0)&(all_rows.radicals==0)
    all_rows['neutral_aromatic']=all_rows.neutral_closed_shell & all_rows.any_aromatic_atom
    all_rows['eligible_connected']=all_rows.neutral_closed_shell & all_rows.connected
    summary=aggregate(all_rows,['arm'],['valid','connected','aromatic','any_aromatic_atom','ringless','neutral_closed_shell','neutral_aromatic','eligible_connected','requested_atoms','atoms','cycles_lost_fake_removal'])
    summary.to_csv(output/'replication_summary.csv',index=False)
    aggregate(all_rows,['arm','seed'],['valid','aromatic','ringless','neutral_closed_shell','neutral_aromatic','eligible_connected','requested_atoms','atoms']).to_csv(output/'replication_per_seed.csv',index=False)
    all_rows.to_csv(output/'replication_molecules.csv',index=False)
    baseline=all_rows[all_rows.arm=='steps100'].set_index(['seed','sample_index'])
    changed=all_rows[all_rows.arm=='steps250'].set_index(['seed','sample_index'])
    if not baseline.index.equals(changed.index):
        raise RuntimeError('Paired step comparison indices differ')
    if not np.array_equal(baseline.requested_atoms,changed.requested_atoms):
        raise RuntimeError('Paired step comparison count vectors differ')
    contrasts=[]
    generator=np.random.default_rng(20261020)
    for metric in ['valid','aromatic','ringless','neutral_closed_shell','neutral_aromatic','eligible_connected']:
        delta=changed[metric].astype(int).to_numpy()-baseline[metric].astype(int).to_numpy()
        gained=int((delta==1).sum());lost=int((delta==-1).sum())
        bootstrap=delta[generator.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
        contrasts.append({'metric':metric,'n':len(delta),'gained':gained,'lost':lost,'delta':float(delta.mean()),
                          'paired_bootstrap_ci_low':float(np.quantile(bootstrap,.025)),'paired_bootstrap_ci_high':float(np.quantile(bootstrap,.975)),
                          'mcnemar_exact_p':float(binomtest(gained,gained+lost).pvalue) if gained+lost else 1.})
    previous_adjusted=0.
    for rank,index in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]['mcnemar_exact_p'])):
        adjusted=max(previous_adjusted,min(1.,(len(contrasts)-rank)*contrasts[index]['mcnemar_exact_p']))
        contrasts[index]['holm_p_six_endpoints']=adjusted
        previous_adjusted=adjusted
    pd.DataFrame(contrasts).to_csv(output/'steps_paired_contrasts.csv',index=False)
    plot=summary.set_index('arm').loc[['steps100','steps250','db_conditional_counts_steps100']]
    fig,ax=plt.subplots(figsize=(9,4.7))
    locations=np.arange(len(plot))
    for offset,metric in [(-.25,'valid'),(0,'aromatic'),(.25,'ringless')]:
        values=plot[metric+'_mean'].to_numpy()
        interval=np.array([values-plot[metric+'_ci_low'].to_numpy(),plot[metric+'_ci_high'].to_numpy()-values])*100
        ax.bar(locations+offset,100*values,width=.24,yerr=interval,capsize=3,label=metric)
    ax.set_xticks(locations,['Legacy count / 100 steps','Legacy count / 250 steps','DB conditional count / 100 steps'])
    ax.set_ylim(0,105);ax.set_ylabel('Fraction of all requests (%)');ax.legend()
    ax.set_title('e1944, EST=0, CFG=1; 256 requests per setting')
    fig.tight_layout();fig.savefig(output/'replication_comparison.png',dpi=180);plt.close(fig)
    return summary,contrasts,statuses


def novel_pngs(base,output):
    """User-facing grids exclude DB molecules and within-setting duplicates.

    Diagnostic ALL-valid grids remain preserved in their original folders.
    Non-isomeric identity matches the user's constitutional-structure objective.
    """
    source=pd.read_csv(base/'full_v2/source_records.csv')
    database={Chem.MolToSmiles(Chem.MolFromSmiles(value),isomericSmiles=False) for value in source.source_smiles}
    destinations=output/'novel_unique_rdkit2d'
    destinations.mkdir(exist_ok=True)
    groups={}
    for path in sorted((base/'full_v2/free_generation').glob('*/*.sdf')):
        groups.setdefault('screen_'+path.parent.name,[]).append(path)
    for directory,prefix in [('replication_steps_20261020','replication_'),('count_replication_20261020','replication_')]:
        for path in sorted((base/directory).glob('*/*.sdf')):
            groups.setdefault(prefix+path.parent.name,[]).append(path)
    rows=[];summaries=[]
    for setting,paths in groups.items():
        molecules=[];legends=[];seen=set();read_failures=0;db_hits=0;duplicates=0
        for path in paths:
            supplier=Chem.SDMolSupplier(str(path),sanitize=False,removeHs=False,strictParsing=False)
            for position,raw in enumerate(supplier):
                identity={'setting':setting,'sdf':str(path),'sdf_record':position}
                try:
                    if raw is None:
                        raise ValueError('SDF read failed')
                    Chem.SanitizeMol(raw)
                    smiles=Chem.MolToSmiles(raw,isomericSmiles=False)
                    in_database=smiles in database
                    duplicate=smiles in seen
                    seen.add(smiles)
                    db_hits+=int(in_database);duplicates+=int(duplicate)
                    selected=not in_database and not duplicate
                    rows.append({**identity,'read_valid':True,'smiles':smiles,'in_database':in_database,'duplicate_in_setting':duplicate,'png_selected':selected})
                    if selected:
                        rdDepictor.Compute2DCoords(raw,clearConfs=True)
                        molecules.append(raw)
                        sample_id=raw.GetProp('sample_index') if raw.HasProp('sample_index') else raw.GetProp('generation_index')
                        legends.append(f'{setting} {path.stem} id={sample_id}')
                except Exception as error:
                    read_failures+=1
                    rows.append({**identity,'read_valid':False,'error':str(error),'png_selected':False})
        for start in range(0,len(molecules),16):
            Draw.MolsToGridImage(molecules[start:start+16],legends=legends[start:start+16],molsPerRow=4,subImgSize=(340,280)).save(destinations/f'{setting}_{start//16:03d}.png')
        summaries.append({'setting':setting,'novel_unique_png_molecules':len(molecules),'database_matches':db_hits,'duplicate_records':duplicates,'sdf_reread_failures':read_failures})
    pd.DataFrame(rows).to_csv(output/'novel_png_selection.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'novel_png_summary.csv',index=False)
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    base=args.base.resolve();output=base/'consolidated_20261020';output.mkdir(exist_ok=True)
    count=count_audit(base,output)
    energies=energetic_characterization(output)
    summary,contrasts,statuses=replication(base,output)
    novelty=novel_pngs(base,output)
    old=pd.read_csv(base/'analysis_final/database_ring_EST.csv')
    calibration=aggregate(old[old.near_zero],['category'],['xtb_near_zero','xtb_est'])
    calibration.to_csv(output/'db_reference_near_zero_xtb_by_class.csv',index=False)
    notes={
        'conclusion':'Aromatic/cyclic failures contribute, but do not alone explain enrichment of ringless near-zero-EST outputs.',
        'model_or_checkpoint_changed':False,'new_training':False,'new_xtb_calculations':False,
        'completed':statuses,
        'first_stage_counts':{'source_DB':1581,'source_aromatic':1362,'source_nonaromatic':219,'denoiser_forward_cases':15525,'intervention_decodes':93150,'partial_reverse':648,'free_samples':416},
        'replication_new_samples':768,
        'novel_pngs':novelty,
        'statistical_limits':'Per-setting N=256 across two seeds. Within paired steps, initial fixtures are identical but outcomes are different molecules. Bootstrap is over sample indices, not a large number of independent model seeds. Count-distribution intervention changes graph dimensions and cannot share identical noise tensors.',
        'physical_limits':'Only the existing 1000-sample checkpoint cohorts have XTB/sTDA results. New setting comparisons establish structural effects, not EST fidelity or physical feasibility.',
        'old_DB_calculation_caveat':'Historical DB XTB/sTDA started from ETKDG/MMFF conformers; generated cohorts started from diffusion coordinates. Do not attribute the whole DB/reference energy gap to electronic-structure method alone.',
        'threshold_correction':'Inclusive |EST|<=0.1 with 1e-6 tolerance: DB231 instead of227; e1074 calculated hits59 instead of58.',
        'references':[
            {'url':'https://www.rdkit.org/docs/RDKit_Book.html#aromaticity','use':'Aromaticity depends on complete atom/bond environment. Invalid raw cycles are not automatically labelled intended aromatic.'},
            {'url':'https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization','use':'Separating valence sanitization from aromatic perception; these checks do not establish physical stability.'},
            {'url':'https://arxiv.org/abs/2402.04997','use':'Continuous/discrete multimodal flow context; the local implementation uses masked CTMC for categorical features and coordinate flow, not Gaussian bond DDPM.'},
        ],
    }
    json_save(output/'conclusions.json',notes)
    table=summary[['arm','n','valid_count','aromatic_count','any_aromatic_atom_count','ringless_count','neutral_closed_shell_count','neutral_aromatic_count','requested_atoms_mean']].to_string(index=False)
    energy_table=energies[energies.checkpoint=='e1944'].to_string(index=False)
    text=f'''방향족 구조 실패 / EST≈0 생성 편향 진단 — 2026-10-20

결론
방향족 또는 고리 구조에서의 실패는 실제 존재한다. 그러나 고리 없는 후보 증가를 그것 하나로 설명할 수는 없다.
구분해서 검사한 항목은 (1) 큰 분자의 결합·원자/H 상태 불일치, (2) 생성 단계 수,
(3) 이전 생성 결과에서 가져온 크기 벡터, (4) 계산된 EST 범위로 선별할 때의 구조별 통과율 차이이다.
모든 조작의 효과가 검증된 것은 아니다. 아래 확대 재검증의 차이와 구간을 기준으로 판단해야 한다.
단순히 aromatic loss를 강화해야 한다거나 방향족 학습 자체가 불가능하다는 결론은 이 검사로 지지되지 않는다.

1. 완료한 검사와 보존 범위
- 전체 DB 1,581개를 실제 codec으로 복원: RDKit identity 1,581/1,581.
- e1944 전체 DB와 e1074/e1713 크기 근접 대조군: 15,525개 denoiser 조건, 6가지 출력 교체로 총 93,150회 구조 판정.
- 저 EST 원본 72개를 세 노이즈 수준·세 조건에서 실제 생성 경로로 재생성: 648개.
- 동일 초기 상태를 사용하는 13가지 생성 설정: 각 32개, 총 416개.
- 별도 시드로 100/250 단계와 DB 조건부 원자 수를 재검증: 각 256개, 총 768개.
- 생성된 대조군은 RDKit 2D PNG와 SDF로 보존. 재검증에는 실패 출력의 원시 그래프 텐서도 보존.
- 사용자 확인용 novel_unique_rdkit2d는 전체 DB를 제외하고, 설정별 중복을 제거한 RDKit 2D PNG다. 기존 all-valid 진단 PNG도 별도로 보존했다.
- PNG 선별은 비입체 canonical SMILES 기준이며, SDF 재독해 실패는 별도 기록했고 원래 모델 유효율을 덮어쓰지 않았다.
- 모델/체크포인트/기존 결과는 변경하지 않았으며 학습하지 않았다. 새 XTB 계산도 하지 않았다.

2. 구조 실패는 실제지만 방향족 자체의 취약성과 크기 효과를 구분해야 함
기존 e1944 1,000개: raw cyclic 669/891 valid=75.1%, ringless 107/109=98.2%.
그러나 ringless 109개는 모두 heavy atom<=15이다. 같은 <=15 범위의 cyclic은 56/59=94.9%.
이 작은 분자끼리의 비교 Fisher p=0.345로 방향족/고리 자체의 독립적인 효과를 입증하지 못한다.
cyclic 유효율은 16~25원자 270/306=88.2%, 26원자 이상 343/526=65.2%이다.
이 비교의 cyclic은 그래프 고리 존재이며 방향족과 동의어가 아니다.

기존 e1944 실패 224개는 모두 AtomValenceException이다. 실패 원자 375개 중 348개가 고리에 속한다.
대표적인 오류는 C의 heavy-bond order sum=5, 또는 heavy-bond sum=4인데 H=1을 함께 갖는 경우다.
직렬화된 SDF의 H 제약만 풀어보는 진단에서는 53/224개가 sanitize되고 48개가 방향족으로 판정된다.
이것은 일부 방향족 가능한 그래프가 원자/H 상태 불일치로 탈락한다는 증거다.
단, 이 53개는 새 성공 분자로 계산하지 않았으며, 수소를 지우는 후처리를 모델 해결책으로 채택하지 않았다.
나머지 171개는 이 조작으로 해결되지 않아 수소만의 문제로 볼 수 없다.

3. 학습된 모델의 복원 실험
시간 t=1이 clean, t=0이 fully noisy인 flow convention이다.
e1944에서 t=0.95의 masked-only 복원: 방향족 1,362/1,362 valid, 원본 identity 98.24%.
t=0.5: 방향족/비방향족 유효율이 각각 약79.0%로 거의 같다.
t=0.25의 단일 endpoint 복원은 양쪽 모두 크게 나빠진다. 단일 예측은 전체 생성 성공률과 다르다.
원자 수가 정확히 같은 15쌍의 t=0.5 복원은 방향족13/15, 비방향족11/15 valid로, 방향족만 특별히 깨진다는 증거는 없다.
부분적으로 훼손한 저 EST 방향족36개의 실제 reverse 생성(t=0.5 시작, EST=0)은 33개 valid, 30개 aromatic, 원본 identity14개다.
즉 방향족 유지 능력은 있으나, 원본 보존과 신규 생성 및 물성 fidelity는 서로 다른 평가다.

원자/H/전하를 정답으로 교체했을 때 방향족 t=0.5 validity:79.00%→79.59%.
결합을 정답으로 교체했을 때:79.00%→91.19%.
H만 정답으로 바꾸면 다른 예측 상태와 충돌하여71.51%로 내려간다.
따라서 개별 H 또는 방향족 라벨보다 결합과 원자 상태의 공동 일관성이 중요한 진단 지점이다.
좌표를 덜 훼손(t_x=.95, t_graph=.5)하면95.23% valid, 그래프를 덜 훼손(t_x=.5,t_graph=.95)하면99.78% valid다.
이는 noisy geometry와 graph의 상호 의존성을 보이지만 생성 좌표의 물리적 정확성을 인증하지는 않는다.

train/test에서 방향족 원본 identity(t=.5, true condition)는52.7%/27.0%로 차이가 있다.
true condition이 train에서 주는 이득은 held-out에서 일관되지 않았다. 조건을 받는 것과 좋은 물성 제어는 다르다.
test에만 있는 Si 관련 joint state 1개가 지원 마스크 밖에 있었다. EST=.634이며, 현재 near-zero 편향의 일반적 원인은 아니다.

4. 원자 수 출처: 추가 확인된 평가 조건
기존 fixed_atom_counts.npy가 앞선 e1074 CFG2 생성 결과의 최종 heavy-atom count와 1,000개 모두 정확히 일치한다.
기존 벡터 평균={count['fixed_count_mean']:.3f}, 범위={count['fixed_count_min']}~{count['fixed_count_max']}.
현재 train DB에서 EST=0, bandwidth=.15 eV, nearest_k=128 조건부 분포의 평균={count['fresh_conditional_mean']:.3f}, 범위={count['fresh_conditional_min']}~{count['fresh_conditional_max']}.
기존 벡터의 원자 수5는 현재 train count support에 없다.
--number_of_atoms가 주어지면 --atom_count_mode conditional_kernel보다 우선한다.
그러므로 기존 체크포인트간 상대 비교는 동일 조건이지만, 현재 DB의 조건부 크기 분포를 대표하는 테스트는 아니었다.

5. 확대 재검증 결과 (모두 e1944, EST=0, CFG=1)
{table}

단계 수 비교의 paired 결과:
{pd.DataFrame(contrasts).to_string(index=False)}

위 표의 분모는 각각256개의 생성 요청이다. aromatic/ringless는 sanitize 성공한 해당 구조의 수이며 invalid는 두 항목에 포함되지 않는다.
aromatic은 기존 분석과 같은 RDKit aromatic ring count>=1 기준이고, any_aromatic_atom은 융합계의 방향족 원자 존재를 직접 확인한 보완 기준이다.
단계수는 각각100/250개 time grid points이며 실제 업데이트는99/249회다.
count 대조군은 graph 차원이 바뀌므로 같은 seed여도 완전히 같은 초기 noise tensor는 아니다.
전체 유효율은205/256→208/256, 방향족은164→168로 개선을 확인하지 못했다.
다만 중성 closed-shell은138→163, 그중 방향족은112→136으로 증가했다(두 seed에서 같은 방향).
중성 방향족 증가의 paired p는 보정 전 약.032지만, 보고한6개 지표 Holm 보정 후에는 유의하지 않다.
따라서250단계를 완전히 무의미하다고 배제할 수도 없고, 효과가 확정됐다고 채택할 수도 없다.
이는 XTB 투입 가능 후보 수에 대한 신호이며 물성 fidelity 개선의 증거는 아니다.
최적 설정을 확정하거나 EST 성능 향상을 주장하려면 별도 XTB/sTDA 비교가 필요하다.

6. EST 선별 효과
기존 aromatic ring count 기준으로 e1944 XTB 성공509개 중 aromatic419, nonaromatic cyclic33, ringless57개였다.
|calculated EST|<=.1 eV 통과는 각각56/419=13.4%,25/33=75.8%,51/57=89.5%다.
따라서 XTB 입력에서 이미 방향족이82.3%인데도, 최종 EST 후보에서는42.4%로 감소한다.
방향족이 모두 구조 실패로 사라져서 생긴 현상은 아니다.
방향족 원자는 있지만 aromatic ring count=0인 융합 구조가 계산 cohort에2개 있어, any-aromatic-atom 기준으로는 방향족421개다.
두 개는 low-EST hit가 아니므로56개라는 hit 수는 동일하고 통과율은13.3%이다. DB의 두 방향족 기준 간 불일치는0개다.

저 EST 후보들의 절대 에너지/구조:
{energy_table}

특히 ringless 후보의 S1 중앙값은5.974 eV, 방향족은2.9905 eV다.
작은 EST만 요구하면 S1과 T1이 둘 다 높은 분자도 목표를 만족한다.
고리 없는51개 중35개는 +와- 형식전하를 동시에 가진다. 전체 DB의 reference-near-zero ringless10개도 모두 이 형태다.
고리 없는 후보가 무작위로만 생기는 것은 아니며 DB에서 관측되는 저 EST motif의 비중 확대 가능성이 있다.
입체표기를 지운 unique 수는 aromatic55/56, ringless50/51로, 중복만으로 이 차이를 설명하지는 못한다.

DB 자체도 reference EST≈0과 XTB EST≈0은 같지 않다:
aromatic86/195=44.1%, nonaromatic cyclic22/26=84.6%, ringless7/10=70.0%만 XTB 범위에 남는다.
이때도 최종 DB 후보115개 중86개가 aromatic이므로, 계산 수준/선별만으로 생성 분포 전체를 설명하지 못한다.
과거 DB XTB는 ETKDG/MMFF 출발이고 이번 생성 XTB는 diffusion 좌표 출발이므로, 이 차이를 전부 계산 방법의 보정치로 간주하면 안 된다.

7. 권장 다음 단계 (이번에는 생산 모델을 수정하지 않음)
- 새 평가에서는 DB 기반 count 분포와 기존 fixed-count benchmark를 구분하여 기록.
- 구조 성공률은 모든 요청 분모로, aromatic low-EST hit는 별도 분모/수치로 평가.
- 100/250 재검증을 바탕으로 sampler 설정을 고른 뒤 동일 조건 XTB로 확인. 유효율만으로 채택하지 않음.
- 모델 변경을 한다면 고리 클래스 loss 추가보다 noisy-state에서 결합/원자/H 상태의 일관성과 큰 분자의 오류 누적을 우선 검증.
- 타깃이 저에너지 방향족 후보라면 EST 외 S1 범위/방향족 여부를 평가 기준 또는 조건으로 명시해야 함. 현재 EST 스칼라만으로 그 요구는 표현되지 않음.
- RDKit aromaticity는 전체 원자/결합 환경의 판정이고 sanitization 성공은 합성 가능성이나 물리적 안정성 보장이 아님.

참고와 코드
RDKit aromaticity/sanitization: https://www.rdkit.org/docs/RDKit_Book.html
Discrete multimodal flow: https://arxiv.org/abs/2402.04997
실제 모델의 categorical 경로는 masked CTMC, 좌표는 continuous flow이다.
condition 입력 경로 오차는 반복 GPU 수치 오차 규모(약1e-5)였고, 진단 reverse와 실제 생성기는 categorical 결과가 같고 좌표 최대차2.15e-6이었다.

산출물
{base/'analysis_final'}
{base/'replication_steps_20261020'}
{base/'count_replication_20261020'}
{output}
'''
    (output/'REPORT_KO.txt').write_text(text)
    print(json.dumps({'state':'complete','output':str(output),'replication_samples':int(summary.n.sum())}))


if __name__=='__main__':
    main()
