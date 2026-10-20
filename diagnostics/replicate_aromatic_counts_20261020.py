"""Fresh train-DB conditional counts versus legacy output-derived fixed counts.

Compare to the steps100 arm of replicate_aromatic_steps_20261020.py. Seed,
checkpoint and all sampler settings match; graph dimensions/initial arrays
differ because this experiment changes the requested molecular-size distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import aromatic_bias_20261019 as diag
import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-seed', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.time()
    diag.save_json(output/'status.json', {'state':'running','pid':os.getpid(),'started':started})
    try:
        model = diag.ClassifierFreeGuidance.load_from_checkpoint(str(diag.CHECKPOINTS['e1944']), weights_only=False).to('cuda:0').eval()
        sampler = model.conditional_n_atoms_sampler
        targets = torch.zeros(args.per_seed,1)
        probabilities,count_diagnostics = sampler.probabilities(torch.zeros(1,1),.15,nearest_k=128)
        seeds = [20261022, 20261023]
        guide = {'x':1.,'s':1.,'e':1.}
        diag.save_json(output/'manifest.json', {
            'checkpoint':str(diag.CHECKPOINTS['e1944']),
            'checkpoint_sha256':hashlib.sha256(diag.CHECKPOINTS['e1944'].read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'shared_diagnostics_sha256':hashlib.sha256(Path(diag.__file__).read_bytes()).hexdigest(),
            'target_est':0., 'guide':guide, 'seeds':seeds, 'per_seed':args.per_seed,
            'integration_points':[100], 'updates':[99], 'batch_size':args.batch_size,
            'learned_weights_changed':False, 'sampling_settings_changed':['requested_atom_count_distribution'],
            'count_reference':model.atom_count_reference_file, 'count_reference_sha256':hashlib.sha256(Path(model.atom_count_reference_file).read_bytes()).hexdigest(),
            'count_bandwidth_eV':.15, 'count_nearest_k':128,
            'count_expected_mean':float((probabilities[0]*sampler.atom_counts).sum()),
            'count_ess':float(count_diagnostics.effective_sample_size[0]),
            'control':'Same seeds, batch size, checkpoint, CFG=1 and EST=0 as legacy steps100. Size changes graph dimension: initial noise tensors cannot be identical.',
            'limitations':'Structural comparison only; no new electronic structure calculation. This tests the count distribution, not aromaticity alone.',
        })
        all_rows = []
        with torch.inference_mode():
            for seed in seeds:
                counts = sampler.sample(targets,.15,nearest_k=128,generator=torch.Generator().manual_seed(seed+900000)).numpy()
                np.save(output/f'requested_counts_seed{seed}.npy',counts)
                for offset in range(0,args.per_seed,args.batch_size):
                    real_counts = torch.as_tensor(counts[offset:min(offset+args.batch_size,args.per_seed)],dtype=torch.long)
                    diag.seed_all(seed+offset)
                    slots = model.add_fake_atom_capacity(real_counts)
                    pieces = []
                    for count in slots.tolist():
                        edges = diag.build_edge_idxs(count)
                        pieces.append(dgl.graph((edges[0],edges[1]),num_nodes=count,device=model.device))
                    initial = dgl.batch(pieces)
                    node_batch,_ = diag.get_batch_idxs(initial)
                    initial = model.sample_prior(initial,node_batch,diag.get_upper_edge_mask(initial))
                    torch.save({'real_counts':real_counts,'slots':slots,'x0':initial.ndata['x_0'].cpu(),
                                's0':initial.ndata['s_0'].cpu(),'e0':initial.edata['e_0'].cpu()},
                               output/f'initial_seed{seed}_offset{offset}.pt')
                    for steps in [100]:
                        arm = 'db_conditional_counts_steps100'
                        arm_output = output/arm
                        arm_output.mkdir(exist_ok=True)
                        final = diag.reverse_loop(model,initial,seed=seed+100000+offset,target=0.,guide=guide,steps=steps)
                        final_cpu = final.to('cpu')
                        torch.save({'batch_nodes':final_cpu.batch_num_nodes(), 'src_dst':final_cpu.edges(),
                                    'state':final_cpu.ndata['s_1'].argmax(-1), 'edge':final_cpu.edata['e_1'].argmax(-1),
                                    'positions':final_cpu.ndata['x_1'], 'requested_atoms':real_counts},
                                   arm_output/f'final_seed{seed}_offset{offset}.pt')
                        writer = Chem.SDWriter(str(arm_output/f'seed{seed}_offset{offset}.sdf'))
                        writer.SetKekulize(False)
                        pictures,legends = [],[]
                        for local_index,graph in enumerate(dgl.unbatch(final_cpu)):
                            status,molecule = diag.decode(model,graph,graph.ndata['s_1'].argmax(-1),graph.edata['e_1'].argmax(-1),graph.ndata['x_1'])
                            identifier = {'arm':arm,'seed':seed,'sample_index':offset+local_index,'requested_atoms':int(real_counts[local_index]),'target_est':0.}
                            all_rows.append({**identifier,**status})
                            if molecule is not None:
                                molecule.SetIntProp('sample_index',offset+local_index)
                                molecule.SetIntProp('seed',seed)
                                writer.write(molecule)
                                display = Chem.Mol(molecule)
                                rdDepictor.Compute2DCoords(display,clearConfs=True)
                                pictures.append(display)
                                legends.append(f'{arm} seed={seed} id={offset+local_index}')
                        writer.close()
                        if pictures:
                            Draw.MolsToGridImage(pictures,legends=legends,molsPerRow=4,subImgSize=(320,270)).save(arm_output/f'seed{seed}_offset{offset}_rdkit2d.png')
                        diag.write_csv(output/'molecules.csv',all_rows)
                        diag.write_csv(output/'summary.csv',diag.summarize_rows(all_rows,['arm','seed'],['valid','connected','aromatic','ringless','cycles_lost_fake_removal','atoms','charge','radicals']))
                        diag.emit('replication_progress',seed=seed,offset=offset,arm=arm,rows=len(all_rows))
        diag.save_json(output/'status.json',{'state':'complete','pid':os.getpid(),'elapsed_seconds':time.time()-started,'samples':len(all_rows)})
        diag.emit('replication_complete',samples=len(all_rows),elapsed_seconds=time.time()-started)
    except Exception:
        diag.save_json(output/'status.json',{'state':'failed','pid':os.getpid(),'elapsed_seconds':time.time()-started,'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
