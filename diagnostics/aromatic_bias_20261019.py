from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(os.environ.get('AROMATIC_DIAG_ROOT', Path(__file__).resolve().parents[1]))
BUNDLE = ROOT / 'MolGuidance_guidance_selector_e987'
sys.path.insert(0, str(BUNDLE))
os.chdir(BUNDLE)

import dgl
import numpy as np
import torch
import torch.nn.functional as functional
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors

from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.data_processing.dataset import MoleculeDataset, collate
from molguidance.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask
from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance

RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
DB = ROOT / 'molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz'
RUNS = {
    'e1074': ROOT / 'runs/cfg_paired_e1074_est0_n1000_seed20261019/cfg1',
    'e1713': ROOT / 'runs/generation_stage6_best_e1713_est0_n1000_cfg1_seed20261019',
    'e1944': ROOT / 'runs/generation_stage6_best_e1944_est0_n1000_cfg1_seed20261019',
}
CHECKPOINTS = {
    'e1074': RUNS['e1074'].parent / 'checkpoint_snapshot/epoch=1074-step=2150.ckpt',
    'e1713': RUNS['e1713'] / 'checkpoint_snapshot/epoch=1713-step=3428.ckpt',
    'e1944': RUNS['e1944'] / 'checkpoint_snapshot/epoch=1944-step=3890.ckpt',
}


def emit(event, **values):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}, default=str), flush=True)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def clone_graph(graph):
    duplicate = graph.clone()
    for key in graph.ndata:
        duplicate.ndata[key] = graph.ndata[key].clone()
    for key in graph.edata:
        duplicate.edata[key] = graph.edata[key].clone()
    if hasattr(graph, 'prop'):
        duplicate.prop = graph.prop.clone()
    return duplicate


def chemical_status(molecule):
    result = {'valid': False, 'connected': False, 'aromatic': False, 'ringless': False, 'error': '', 'smiles': ''}
    if molecule is None:
        result['error'] = 'construction_failed'
        return result, None
    try:
        raw = Chem.Mol(molecule)
        raw.UpdatePropertyCache(strict=False)
        result['raw_cycle_rank'] = raw.GetNumBonds() - raw.GetNumAtoms() + len(Chem.GetMolFrags(raw))
        Chem.SanitizeMol(raw)
        result.update(valid=True, connected=len(Chem.GetMolFrags(raw)) == 1,
                      aromatic=rdMolDescriptors.CalcNumAromaticRings(raw) > 0,
                      ringless=rdMolDescriptors.CalcNumRings(raw) == 0,
                      aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(raw),
                      rings=rdMolDescriptors.CalcNumRings(raw),
                      atoms=raw.GetNumHeavyAtoms(),
                      charge=sum(atom.GetFormalCharge() for atom in raw.GetAtoms()),
                      radicals=sum(atom.GetNumRadicalElectrons() for atom in raw.GetAtoms()),
                      smiles=Chem.MolToSmiles(raw, canonical=True, isomericSmiles=False))
        return result, raw
    except Exception as error:
        result['error'] = type(error).__name__
        return result, None


def decode(model, graph, states, edges, positions=None, hydrogen_oracle=None):
    template = graph.to('cpu')
    states = states.detach().cpu().long()
    edges = edges.detach().cpu().long()
    endpoint = dgl.graph(template.edges(), num_nodes=template.num_nodes())
    endpoint.ndata['s_1'] = functional.one_hot(states, model.n_atom_states + 1).float()
    endpoint.ndata['x_1'] = (positions if positions is not None else template.ndata.get('x_1_true', template.ndata.get('x_t'))).detach().cpu()
    endpoint.edata['e_1'] = functional.one_hot(edges, model.n_bond_types + 1).float()
    endpoint.edata['ue_mask'] = get_upper_edge_mask(endpoint)
    keep = (states != model.fake_atom_state_index).nonzero().flatten().tolist()
    try:
        sample = SampledMolecule(endpoint, model.atom_type_map, ctmc_mol=True,
                                 atom_state_vocab=model.atom_state_vocab,
                                 build_xt_traj=False, build_ep_traj=False)
        molecule = sample.rdkit_mol
        if hydrogen_oracle is not None and molecule is not None:
            molecule = Chem.Mol(molecule)
            for atom, source_index in zip(molecule.GetAtoms(), keep):
                true_state = int(hydrogen_oracle[source_index])
                if true_state != model.fake_atom_state_index:
                    atom.SetNumExplicitHs(int(model.atom_state_vocab[true_state]['hydrogens']))
        result, molecule = chemical_status(molecule)
    except Exception as error:
        result, molecule = chemical_status(None)
        result['error'] = type(error).__name__
    result['slots'] = len(states)
    result['fake_removed'] = int((states == model.fake_atom_state_index).sum())
    result['residual_node_masks'] = int((states == model.n_atom_states).sum())
    result['residual_edge_masks'] = int((edges == model.n_bond_types).sum()) // 2
    sources, targets = template.edges()
    upper = get_upper_edge_mask(template)
    present = upper & (edges > 0) & (edges < model.n_bond_types)
    def cycle_rank(active):
        parents = list(range(len(states)))
        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        bond_count = 0
        for left, right in zip(sources[present].tolist(), targets[present].tolist()):
            if active[left] and active[right]:
                bond_count += 1
                parents[root(left)] = root(right)
        components = len({root(index) for index, selected in enumerate(active) if selected})
        return bond_count - sum(active) + components
    result['slot_cycle_rank'] = cycle_rank([True] * len(states))
    result['real_cycle_rank'] = cycle_rank((states != model.fake_atom_state_index).tolist())
    result['cycles_lost_fake_removal'] = result['slot_cycle_rank'] - result['real_cycle_rank']
    return result, molecule


def full_edges(upper_values, upper_mask):
    output = torch.zeros(len(upper_mask), dtype=upper_values.dtype, device=upper_values.device)
    output[upper_mask] = upper_values
    output[~upper_mask] = upper_values
    return output


def summarize_rows(rows, groups, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in groups)].append(row)
    output = []
    for values, selected in grouped.items():
        record = dict(zip(groups, values))
        record['n'] = len(selected)
        for field in fields:
            numbers = [float(row[field]) for row in selected if field in row and row[field] is not None and math.isfinite(float(row[field]))]
            record[field + '_mean'] = float(np.mean(numbers)) if numbers else None
        output.append(record)
    return output


def existing_funnel(output):
    results = {}
    invalid_rows = []
    for label, run in RUNS.items():
        rows = read_csv(run / 'analysis/molecules.csv')
        stages = {}
        for name, selected in (
            ('rdkit_valid', [row for row in rows if row['valid'] == 'True']),
            ('novel_connected', [row for row in rows if row.get('novel_unique_connected') == 'True']),
        ):
            counts = Counter('aromatic' if int(row['num_aromatic_rings']) > 0 else ('ringless' if int(row['num_rings']) == 0 else 'nonaromatic_cyclic') for row in selected)
            stages[name] = dict(n=len(selected), **counts)
        xtb = read_csv(run / 'xtb_stda_diffusion_coords_24x1/work/results.csv')
        for name, selected in (
            ('xtb_evaluated', [row for row in xtb if row['status'] == 'ok']),
            ('xtb_near_zero', [row for row in xtb if row['status'] == 'ok' and abs(float(row['s1_energy_eV']) - float(row['t1_energy_eV'])) <= 0.100001]),
        ):
            counts = Counter('aromatic' if int(row['aromatic_rings']) > 0 else ('ringless' if int(row['rings']) == 0 else 'nonaromatic_cyclic') for row in selected)
            stages[name] = dict(n=len(selected), **counts)
        supplier = Chem.SDMolSupplier(str(run / 'samples.sdf'), sanitize=False, removeHs=False, strictParsing=False)
        for index, molecule in enumerate(supplier):
            status, clean = chemical_status(molecule)
            if not status['valid']:
                invalid_rows.append({'checkpoint': label, 'generation_index': index, **status})
        stages['invalid'] = dict(Counter(row['error'] for row in invalid_rows if row['checkpoint'] == label))
        results[label] = stages
    save_json(output / 'existing_funnel.json', results)
    write_csv(output / 'existing_invalid.csv', invalid_rows)
    emit('existing_funnel_complete', result=results)


def load_data(checkpoint):
    settings = checkpoint['datamodule_hyper_parameters']
    graphs, records = [], []
    for split in ('train', 'val', 'test'):
        config = copy.deepcopy(settings['dataset_config'])
        config['processed_data_dir'] = str(DB)
        dataset = MoleculeDataset(split, config, copy.deepcopy(settings['dm_prior_config']))
        data = torch.load(DB / f'{split}_data_processed.pt', map_location='cpu', weights_only=False)
        for index, smiles in enumerate(data['smiles']):
            seed_all(771000 + len(records))
            graph = dataset[index]
            molecule = Chem.MolFromSmiles(smiles)
            records.append({'record_id': len(records), 'split': split, 'split_index': index,
                            'est': float(data['properties'][index]),
                            'near_zero': abs(float(data['properties'][index])) <= 0.100001,
                            'source_aromatic': rdMolDescriptors.CalcNumAromaticRings(molecule) > 0,
                            'source_ringless': rdMolDescriptors.CalcNumRings(molecule) == 0,
                            'source_atoms': molecule.GetNumHeavyAtoms(),
                            'source_smiles': Chem.MolToSmiles(molecule, isomericSmiles=False)})
            graphs.append(graph)
    return graphs, records


def oracle(model, graphs, records, output):
    rows = []
    for graph, record in zip(graphs, records):
        status, _ = decode(model, graph, graph.ndata['s_1_true'].argmax(-1), graph.edata['e_1_true'].argmax(-1))
        rows.append({**record, **status, 'identity': status['smiles'] == record['source_smiles']})
    write_csv(output / 'oracle.csv', rows)
    summary = {'n': len(rows), 'identity': sum(row['identity'] for row in rows), 'valid': sum(row['valid'] for row in rows)}
    save_json(output / 'oracle_summary.json', summary)
    emit('oracle_complete', **summary)
    if summary['identity'] != summary['n']:
        raise RuntimeError('Oracle identity gate failed; learned-model diagnosis stopped')


def predict(model, graph, coordinate_time, graph_time, condition='truth', self_condition=True):
    node_batch, _ = get_batch_idxs(graph)
    upper_mask = get_upper_edge_mask(graph)
    raw_property = graph.prop.reshape(-1, 1)
    if condition == 'unconditional':
        embedding = model.unconditional_embedder(raw_property)
    elif condition == 'shuffled':
        embedding = model.property_embedder(raw_property.roll(1, 0))
    elif condition == 'truth':
        embedding = model.property_embedder(raw_property)
    else:
        normalization = torch.load(DB / 'train_data_property_normalization.pt', weights_only=False)
        normalized = (float(condition) - float(normalization['mean'])) / float(normalization['std'])
        embedding = model.property_embedder(torch.full_like(raw_property, normalized))
    combined = torch.cat([coordinate_time[:, None], embedding], -1)
    return model.predict_endpoint(graph, combined, node_batch, upper_mask,
                                  self_conditioning=self_condition, graph_time=graph_time,
                                  apply_softmax=False, remove_com=True)


def condition_route_gate(model, source, output):
    graph = collate([clone_graph(source)]).to(model.device)
    graph.prop = source.prop.reshape(1).to(model.device)
    node_batch, edge_batch = get_batch_idxs(graph)
    upper = get_upper_edge_mask(graph)
    times = torch.full((1,), 0.5, device=model.device)
    model.vector_field.sample_conditional_path(graph, times, node_batch, edge_batch, upper,
        generator=torch.Generator(device=model.device).manual_seed(7471))
    results = {}
    for target in (0.0, 0.5, 1.0):
        configure_field(model, target)
        training_route = predict(model, graph, times, times, str(target), False)
        sampling_route = model.vector_field(graph, times, node_batch_idx=node_batch,
            upper_edge_mask=upper, apply_softmax=False, remove_com=True)
        repeated_route = model.vector_field(graph, times, node_batch_idx=node_batch,
            upper_edge_mask=upper, apply_softmax=False, remove_com=True)
        errors = {key: float((training_route[key]-sampling_route[key]).abs().max()) for key in training_route}
        repeated_errors = {key: float((repeated_route[key]-sampling_route[key]).abs().max()) for key in training_route}
        matches = {key: bool(torch.allclose(training_route[key],sampling_route[key],atol=3e-5,rtol=1e-5)) for key in training_route}
        results[str(target)] = {'route_error':errors,'repeat_error':repeated_errors,'allclose':matches}
        if not all(matches.values()):
            raise RuntimeError(f'Condition embedding train/sample route mismatch: {errors}')
    save_json(output/'condition_route_gate.json', results)
    emit('condition_route_gate_complete', result=results)


def endpoint_panel(model, label, graphs, records, indices, output, batch_size, smoke=False):
    rows, metric_rows = [], []
    configs = [(0.95, 0.95, 'truth'), (0.75, 0.75, 'truth'), (0.5, 0.5, 'truth'), (0.25, 0.25, 'truth'),
               (0.5, 0.95, 'truth'), (0.95, 0.5, 'truth'), (0.5, 0.5, 'unconditional'),
               (0.5, 0.5, 'shuffled'), (0.5, 0.5, '0.0')]
    if smoke:
        configs = [(0.95, 0.95, 'truth'), (0.5, 0.5, 'unconditional')]
    device = model.device
    for coordinate_value, graph_value, condition in configs:
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            pieces = []
            for index in batch_indices:
                piece = clone_graph(graphs[index])
                piece_nodes, piece_edges = get_batch_idxs(piece)
                piece_coordinate_time = torch.full((1,), coordinate_value)
                piece_graph_time = torch.full((1,), graph_value)
                model.vector_field.sample_conditional_path(piece,piece_coordinate_time,piece_nodes,piece_edges,
                    get_upper_edge_mask(piece),generator=torch.Generator().manual_seed(190000+index),graph_time=piece_graph_time)
                pieces.append(piece)
            batched = collate(pieces).to(device)
            batched.prop = torch.stack([graphs[index].prop for index in batch_indices]).to(device)
            node_batch, edge_batch = get_batch_idxs(batched)
            upper_mask = get_upper_edge_mask(batched)
            coordinate_time = torch.full((len(batch_indices),), coordinate_value, device=device)
            graph_time = torch.full_like(coordinate_time, graph_value)
            endpoint = predict(model, batched, coordinate_time, graph_time, condition)
            logits = endpoint['s']
            support = model.vector_field.get_category_support_mask('s')
            if support is not None:
                logits = logits.masked_fill(~support[None], -torch.inf)
            predicted_states = logits.argmax(-1)
            predicted_edges = full_edges(endpoint['e'].argmax(-1), upper_mask)
            batched.ndata['diagnostic_state'] = predicted_states
            batched.ndata['diagnostic_position'] = endpoint['x']
            batched.edata['diagnostic_edge'] = predicted_edges
            node_loss = functional.cross_entropy(logits, batched.ndata['s_1_true'].argmax(-1), reduction='none')
            edge_loss = functional.cross_entropy(endpoint['e'], batched.edata['e_1_true'][upper_mask].argmax(-1), reduction='none')
            batched.ndata['diagnostic_nll'] = node_loss
            batched.edata['diagnostic_nll'] = full_edges(edge_loss, upper_mask)
            for index, graph in zip(batch_indices, dgl.unbatch(batched.to('cpu'))):
                record = records[index]
                true_states = graph.ndata['s_1_true'].argmax(-1)
                true_edges = graph.edata['e_1_true'].argmax(-1)
                noisy_states = graph.ndata['s_t'].argmax(-1)
                noisy_edges = graph.edata['e_t'].argmax(-1)
                state_prediction = graph.ndata['diagnostic_state']
                edge_prediction = graph.edata['diagnostic_edge']
                filled_states = torch.where(noisy_states == model.n_atom_states, state_prediction, noisy_states)
                filled_edges = torch.where(noisy_edges == model.n_bond_types, edge_prediction, noisy_edges)
                info = {**record, 'checkpoint': label, 'coordinate_time': coordinate_value, 'graph_time': graph_value, 'condition': condition}
                upper = get_upper_edge_mask(graph)
                masked_edges = (noisy_edges == model.n_bond_types) & upper
                aromatic_edges = graph.edata['e_aromatic_true'] & masked_edges
                positive_edges = (true_edges > 0) & masked_edges
                masked_states = noisy_states == model.n_atom_states
                real = true_states != model.fake_atom_state_index
                aromatic_nodes = torch.zeros(len(true_states), dtype=torch.bool)
                sources, destinations = graph.edges()
                aromatic_nodes[sources[graph.edata['e_aromatic_true']]] = True
                supported = model.vector_field.get_category_support_mask('s').cpu()[true_states]
                metrics = {**info,
                    'masked_aromatic_edges': int(aromatic_edges.sum()),
                    'aromatic_edge_missing': float((edge_prediction[aromatic_edges] == 0).float().mean()) if aromatic_edges.any() else None,
                    'aromatic_edge_order_accuracy': float((edge_prediction[aromatic_edges] == true_edges[aromatic_edges]).float().mean()) if aromatic_edges.any() else None,
                    'masked_positive_edge_recall': float((edge_prediction[positive_edges] > 0).float().mean()) if positive_edges.any() else None,
                    'masked_edge_nll': float(graph.edata['diagnostic_nll'][masked_edges].mean()) if masked_edges.any() else None,
                    'masked_state_nll': float(graph.ndata['diagnostic_nll'][masked_states].mean()) if masked_states.any() else None,
                    'unsupported_target_nodes': int((~supported).sum()),
                    'masked_supported_state_nll': float(graph.ndata['diagnostic_nll'][masked_states & supported].mean()) if (masked_states & supported).any() else None,
                    'aromatic_atom_deleted': int(((filled_states == model.fake_atom_state_index) & aromatic_nodes).sum()),
                    'true_atom_deleted': int(((filled_states == model.fake_atom_state_index) & real).sum()),
                    'true_atom_deleted_fraction': float(((filled_states == model.fake_atom_state_index) & real).sum() / real.sum())}
                metric_rows.append(metrics)
                branches = [('endpoint_argmax', state_prediction, edge_prediction, None),
                            ('masked_fill', filled_states, filled_edges, None),
                            ('oracle_atoms', true_states, filled_edges, None),
                            ('oracle_bonds', filled_states, true_edges, None),
                            ('oracle_H_only', filled_states, filled_edges, true_states),
                            ('oracle_fake_membership', torch.where(real, torch.where(filled_states == model.fake_atom_state_index, true_states, filled_states), true_states), filled_edges, None)]
                for intervention, states, edges, hydrogen_oracle in branches:
                    status, _ = decode(model, graph, states, edges, graph.ndata['diagnostic_position'], hydrogen_oracle)
                    rows.append({**info, 'intervention': intervention, **status,
                                 'identity': status['smiles'] == record['source_smiles']})
            if start % (batch_size * 8) == 0:
                emit('endpoint_progress', checkpoint=label, coordinate_time=coordinate_value,
                     graph_time=graph_value, condition=condition, completed=min(start+batch_size, len(indices)), total=len(indices))
        write_csv(output / f'endpoint_{label}.csv', rows)
        write_csv(output / f'endpoint_metrics_{label}.csv', metric_rows)
        write_csv(output / f'endpoint_summary_{label}.csv', summarize_rows(rows,
            ['checkpoint', 'split', 'source_aromatic', 'near_zero', 'coordinate_time', 'graph_time', 'condition', 'intervention'],
            ['valid', 'connected', 'aromatic', 'ringless', 'identity', 'fake_removed']))
    return rows


def matched_sources(records):
    selected = []
    for split in ('train', 'val', 'test'):
        negatives = [row for row in records if row['split'] == split and row['near_zero'] and not row['source_aromatic']]
        candidates = [row for row in records if row['split'] == split and row['near_zero'] and row['source_aromatic']]
        for negative in negatives:
            partner = min(candidates, key=lambda row: abs(row['source_atoms'] - negative['source_atoms']))
            candidates.remove(partner)
            selected.extend([negative['record_id'], partner['record_id']])
    return selected


def configure_field(model, target):
    field = model.vector_field
    field.properties_for_sampling = None if isinstance(target, list) else float(target)
    field.multilple_values_to_one_property = target if isinstance(target, list) else None
    field.property_name = 'est'
    field.normalization_file_path = str(DB / 'train_data_property_normalization.pt')
    field.conditional_generation = True
    field.properties_handle_method = 'concatenate_sum'
    field.dataset_name = 'qme14s'


def reverse_loop(model, initial, *, seed, target, guide, steps, start_time=0.0, temperature=None,
                 stochasticity=None, self_condition=True, trace_callback=None):
    field = model.vector_field
    configure_field(model, target)
    graph = clone_graph(initial)
    node_batch, edge_batch = get_batch_idxs(graph)
    upper = get_upper_edge_mask(graph)
    for feature in model.canonical_feat_order:
        data = graph.edata if feature == 'e' else graph.ndata
        data[feature + '_t'] = data[feature + '_0'].clone()
    times = torch.linspace(start_time, 1, steps, device=model.device)
    alpha = field.interpolant_scheduler.alpha_t(times)
    derivative = field.interpolant_scheduler.alpha_t_prime(times)
    previous_unconditional = previous_conditional = None
    seed_all(seed)
    for index in range(1, steps):
        graph, next_unconditional, next_conditional = field.step_with_CFG_guidance(
            graph, times[index], times[index-1], alpha[index-1], alpha[index], derivative[index-1],
            node_batch, edge_batch, upper, guide_w=guide,
            cat_temp_func=field.cat_temp_func, forward_weight_func=field.forward_weight_func,
            dfm_type='campbell', stochasticity=field.eta if stochasticity is None else stochasticity,
            high_confidence_threshold=field.hc_thresh, last_step=index == steps-1,
            categorical_temperature=temperature, guidance_format='linear',
            categorical_probability_mode='corrected', return_endpoints=True,
            previous_unconditional_endpoint=previous_unconditional,
            previous_conditional_endpoint=previous_conditional)
        if self_condition:
            previous_unconditional, previous_conditional = next_unconditional, next_conditional
        if trace_callback is not None and (index in {1, steps//4, steps//2, 3*steps//4, steps-2, steps-1}):
            trace_callback(graph, float(times[index]), next_conditional)
    for feature in model.canonical_feat_order:
        data = graph.edata if feature == 'e' else graph.ndata
        data[feature + '_1'] = data[feature + '_t']
    return graph


def partial_reverse(model, graphs, records, indices, output, batch_size, smoke=False):
    rows = []
    for start_time in ([0.95] if smoke else [0.95, 0.75, 0.5]):
        for condition in (['truth'] if smoke else ['truth', 'zero', 'unconditional']):
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                pieces = []
                for index in batch_indices:
                    graph = clone_graph(graphs[index]).to(model.device)
                    node_batch, edge_batch = get_batch_idxs(graph)
                    upper = get_upper_edge_mask(graph)
                    times = torch.full((1,), start_time, device=model.device)
                    generator = torch.Generator(device=model.device).manual_seed(341000 + index)
                    model.vector_field.sample_conditional_path(graph, times, node_batch, edge_batch, upper, generator=generator)
                    for feature in model.canonical_feat_order:
                        data = graph.edata if feature == 'e' else graph.ndata
                        data[feature + '_0'] = data[feature + '_t'].clone()
                    pieces.append(graph)
                initial = dgl.batch(pieces)
                weight = 0.0 if condition == 'unconditional' else 1.0
                target = [records[index]['est'] for index in batch_indices] if condition == 'truth' else 0.0
                final = reverse_loop(model, initial, seed=742000+start, target=target,
                                     guide={'x':weight,'s':weight,'e':weight},
                                     steps=max(3, round(100 * (1-start_time))+1), start_time=start_time)
                for index, graph in zip(batch_indices, dgl.unbatch(final.to('cpu'))):
                    status, _ = decode(model, graph, graph.ndata['s_1'].argmax(-1), graph.edata['e_1'].argmax(-1), graph.ndata['x_1'])
                    rows.append({**records[index], 'start_time':start_time, 'condition':condition, **status,
                                 'identity':status['smiles']==records[index]['source_smiles']})
            write_csv(output/'partial_reverse.csv', rows)
            emit('partial_reverse_progress', start_time=start_time, condition=condition, records=len(rows))
    write_csv(output/'partial_reverse_summary.csv', summarize_rows(rows,
              ['split','source_aromatic','start_time','condition'], ['valid','connected','aromatic','ringless','identity']))


def free_generation(model, output, batch_size, smoke=False):
    arms = [
        ('baseline',0.0,{'x':1.,'s':1.,'e':1.},100,None,None,True),
        ('unconditional',0.0,{'x':0.,'s':0.,'e':0.},100,None,None,True),
        ('est_0p5',0.5,{'x':1.,'s':1.,'e':1.},100,None,None,True),
        ('est_1p0',1.0,{'x':1.,'s':1.,'e':1.},100,None,None,True),
        ('cfg2',0.0,{'x':2.,'s':2.,'e':2.},100,None,None,True),
        ('atom_guidance_only',0.0,{'x':0.,'s':1.,'e':0.},100,None,None,True),
        ('edge_guidance_only',0.0,{'x':0.,'s':0.,'e':1.},100,None,None,True),
        ('coordinate_guidance_only',0.0,{'x':1.,'s':0.,'e':0.},100,None,None,True),
        ('temperature_0p2',0.0,{'x':1.,'s':1.,'e':1.},100,0.2,None,True),
        ('temperature_1p0',0.0,{'x':1.,'s':1.,'e':1.},100,1.0,None,True),
        ('steps250',0.0,{'x':1.,'s':1.,'e':1.},250,None,None,True),
        ('stochasticity0',0.0,{'x':1.,'s':1.,'e':1.},100,None,0.0,True),
        ('self_conditioning_off',0.0,{'x':1.,'s':1.,'e':1.},100,None,None,False),
    ]
    if smoke:
        arms = arms[:2]
    counts = np.load(ROOT/'runs/cfg_paired_e1074_est0_n1000_seed20261019/fixed_atom_counts.npy')
    samples_per_seed = 2 if smoke else 16
    seeds = [20261020] if smoke else [20261020, 20261021]
    rows, trace_rows, terminal_rows = [], [], []
    for seed in seeds:
        for offset in range(0, samples_per_seed, batch_size):
            real_counts = torch.as_tensor(counts[offset: min(offset+batch_size, samples_per_seed)], dtype=torch.long)
            seed_all(seed+offset)
            slots = model.add_fake_atom_capacity(real_counts)
            pieces = []
            for count in slots.tolist():
                edges = build_edge_idxs(count)
                pieces.append(dgl.graph((edges[0],edges[1]),num_nodes=count,device=model.device))
            initial = dgl.batch(pieces)
            node_batch, _ = get_batch_idxs(initial)
            initial = model.sample_prior(initial,node_batch,get_upper_edge_mask(initial))
            fixture = {'seed':seed,'offset':offset,'real_counts':real_counts,'slots':slots,
                       'x0':initial.ndata['x_0'].cpu(),'s0':initial.ndata['s_0'].cpu(),'e0':initial.edata['e_0'].cpu()}
            torch.save(fixture,output/f'initial_seed{seed}_offset{offset}.pt')
            for arm,target,guide,steps,temperature,stochasticity,self_condition in arms:
                def trace(graph, progress, endpoint):
                    temp = clone_graph(graph).to('cpu')
                    temp.ndata['trace_s'] = endpoint['s'].argmax(-1).cpu()
                    temp.edata['trace_e'] = full_edges(endpoint['e'].argmax(-1),get_upper_edge_mask(graph)).cpu()
                    for local_index, molecule_graph in enumerate(dgl.unbatch(temp)):
                        status,_ = decode(model,molecule_graph,molecule_graph.ndata['trace_s'],molecule_graph.edata['trace_e'],molecule_graph.ndata['x_t'])
                        trace_rows.append({'arm':arm,'seed':seed,'sample_index':offset+local_index,'progress':progress,
                                           'masked_states':int((molecule_graph.ndata['s_t'].argmax(-1)==model.n_atom_states).sum()),
                                           'masked_edges':int((molecule_graph.edata['e_t'].argmax(-1)==model.n_bond_types).sum())//2,
                                           'trace_definition':'conditional_endpoint_argmax_not_intermediate_RDKIT_validity',**status})
                final = reverse_loop(model,initial,seed=seed+100000+offset,target=target,guide=guide,steps=steps,
                                     temperature=temperature,stochasticity=stochasticity,self_condition=self_condition,trace_callback=trace)
                if arm == 'baseline' and offset == 0 and seed == seeds[0]:
                    seed_all(seed+100000+offset)
                    official = model.vector_field.integrate_with_CFG_guidance(
                        clone_graph(initial),node_batch,get_upper_edge_mask(initial),steps,
                        guide_w=guide,dfm_type='campbell',stochasticity=None,high_confidence_threshold=None,
                        normalization_file_path=str(DB/'train_data_property_normalization.pt'),
                        conditional_generation=True,property_name='est',properties_for_sampling=0.0,
                        properties_handle_method='concatenate_sum',dataset_name='qme14s',
                        guidance_format='linear',categorical_probability_mode='corrected')
                    parity={'x_max_abs_error':float((official.ndata['x_1']-final.ndata['x_1']).abs().max()),
                            'state_equal':bool(torch.equal(official.ndata['s_1'],final.ndata['s_1'])),
                            'edge_equal':bool(torch.equal(official.edata['e_1'],final.edata['e_1']))}
                    save_json(output/'reverse_parity_gate.json',parity)
                    if not torch.allclose(official.ndata['x_1'],final.ndata['x_1'],atol=1e-4,rtol=1e-5) or not parity['state_equal'] or not parity['edge_equal']:
                        raise RuntimeError(f'Diagnostic reverse loop differs from production: {parity}')
                    emit('reverse_parity_gate_complete',**parity)
                arm_path=output/'free_generation'/arm
                arm_path.mkdir(parents=True,exist_ok=True)
                writer=Chem.SDWriter(str(arm_path/f'seed{seed}_offset{offset}.sdf'))
                writer.SetKekulize(False)
                pictures,legends=[],[]
                for local_index,graph in enumerate(dgl.unbatch(final.to('cpu'))):
                    status,molecule=decode(model,graph,graph.ndata['s_1'].argmax(-1),graph.edata['e_1'].argmax(-1),graph.ndata['x_1'])
                    identifier={'arm':arm,'seed':seed,'sample_index':offset+local_index,'target_est':target,'requested_atoms':int(real_counts[local_index]),'slots':int(slots[local_index])}
                    rows.append({**identifier,**status})
                    endpoint_status,_=decode(model,graph,graph.ndata['s_1_pred'].argmax(-1),graph.edata['e_1_pred'].argmax(-1),graph.ndata['x_1_pred'])
                    terminal_rows.extend([{**identifier,'view':'actual_terminal_state',**status},
                                          {**identifier,'view':'last_sampled_endpoint',**endpoint_status}])
                    if molecule is not None:
                        molecule.SetIntProp('generation_index',offset+local_index)
                        writer.write(molecule)
                        pictures.append(molecule)
                        legends.append(f'{arm} seed={seed} idx={offset+local_index}')
                writer.close()
                if pictures:
                    Draw.MolsToGridImage(pictures,legends=legends,molsPerRow=4,subImgSize=(320,260)).save(arm_path/f'seed{seed}_offset{offset}.png')
                write_csv(output/'free_generation.csv',rows)
                write_csv(output/'trajectory.csv',trace_rows)
                write_csv(output/'terminal_decoding_counterfactuals.csv',terminal_rows)
                emit('free_generation_progress',arm=arm,seed=seed,offset=offset,completed=len(rows))
    write_csv(output/'free_generation_summary.csv',summarize_rows(rows,['arm'],['valid','connected','aromatic','ringless','fake_removed','atoms','charge','radicals']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--skip-free',action='store_true')
    args=parser.parse_args()
    output=args.out.resolve()
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started=time.time()
    save_json(output/'status.json',{'state':'running','pid':os.getpid(),'started':time.time()})
    try:
        existing_funnel(output)
        checkpoint=torch.load(CHECKPOINTS['e1944'],map_location='cpu',weights_only=False)
        graphs,records=load_data(checkpoint)
        write_csv(output/'source_records.csv',records)
        matched=matched_sources(records)
        write_csv(output/'matched_sources.csv',[records[index] for index in matched])
        manifest={'checkpoints':{key:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in CHECKPOINTS.items()},
                  'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'data_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in DB.glob('*data_processed.pt')},
                  'time_convention':'0=prior/masked, 1=clean endpoint; higher t means weaker corruption',
                  'architecture':'continuous coordinate flow + masked categorical CTMC; not Gaussian DDPM for bonds',
                  'diagnostic_only':True,'weights_updated':False,'new_training':False,'smoke':args.smoke,
                  'controls':'paired initial slots/x0 and RNG; schedule/temperature/condition arms are interventions, not unchanged production',
                  'limits':'Invalid free samples cannot be labelled attempted aromatic solely from a failed sanitizer; high-noise identity is not guaranteed; no synthesis feasibility claim.'}
        save_json(output/'manifest.json',manifest)
        for label in (['e1944'] if args.smoke else ['e1944','e1713','e1074']):
            emit('model_loading',checkpoint=label)
            model=ClassifierFreeGuidance.load_from_checkpoint(str(CHECKPOINTS[label]),weights_only=False).to('cuda:0').eval()
            with torch.inference_mode():
                if label=='e1944':
                    condition_route_gate(model,graphs[matched[0]],output)
                    oracle(model,graphs[:8] if args.smoke else graphs,records[:8] if args.smoke else records,output)
                indices=matched[:4] if args.smoke else (list(range(len(records))) if label=='e1944' else matched)
                endpoint_panel(model,label,graphs,records,indices,output,args.batch_size,args.smoke)
                if label=='e1944':
                    partial_reverse(model,graphs,records,matched[:2] if args.smoke else matched,output,args.batch_size,args.smoke)
                    if not args.skip_free:
                        free_generation(model,output,args.batch_size,args.smoke)
            del model
            torch.cuda.empty_cache()
        save_json(output/'status.json',{'state':'complete','pid':os.getpid(),'elapsed_seconds':time.time()-started})
        emit('complete',elapsed_seconds=time.time()-started)
    except Exception:
        save_json(output/'status.json',{'state':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started})
        raise


if __name__=='__main__':
    main()
