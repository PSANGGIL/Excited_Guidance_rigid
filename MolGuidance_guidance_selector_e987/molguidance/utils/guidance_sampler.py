from .propmolflow_util import allowed_properties, load_model_from_ckpt
import math
import torch
import torch.nn.functional as F
import dgl

from molguidance.models.classifier_free_guidance import (
    combine_guided_endpoint_probabilities,
)

def sample_molecules_with_guidance(model,
                    n_mols, 
                    property_name,
                    properties_handle_method,
                    multiple_values_to_one_property=None,
                    number_of_atoms = None,
                    guide_w1=1.5,
                    guide_w2=1.5,
                    fixed_property_val = None,
                    normalization_file_path=None,
                    n_timesteps=100,
                    stochasticity=None, 
                    max_batch_size=128, 
                    xt_traj=False, 
                    ep_traj=False,
                    high_confidence_threshold=None,
                    training_mode=False,
                    dataset_name=None,
                    device='cuda'
                    ):
    assert property_name in allowed_properties, f"Property {property_name} is not supported."
    assert (fixed_property_val is None) ^ (multiple_values_to_one_property is None)      

    n_batches = math.ceil(n_mols / max_batch_size)
    molecules = []
    if model.__class__.__name__ == 'ClassifierFreeGuidance':
        guide_w = {}
        guide_w['x'] = guide_w1
        guide_w['a'] = guide_w2
        guide_w['s'] = guide_w2
        guide_w['c'] = guide_w2
        guide_w['e'] = guide_w2
    elif model.__class__.__name__ == 'ModelGuidance':
        if not isinstance(guide_w1, (int, float)) or not isinstance(guide_w2, (int, float)):
            raise ValueError(f"guide_w should be a number, but got {type(guide_w1)} and {type(guide_w2)}")
        assert guide_w1 == guide_w2, "guide_w1 and guide_w2 should be equal for ModelGuidance."
        guide_w = guide_w1
    for _ in range(n_batches):
        n_mols_needed = n_mols - len(molecules)
        batch_size = min(n_mols_needed, max_batch_size)
        batch_property = None
        batch_no_atoms = None
        if multiple_values_to_one_property is not None:
            batch_property = multiple_values_to_one_property[len(molecules): len(molecules) + batch_size]
        if number_of_atoms is not None:
            batch_no_atoms = number_of_atoms[len(molecules): len(molecules) + batch_size]
        batch_molecules = model.sample_random_sizes(
            batch_size,
            device=device,
            n_timesteps=n_timesteps,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            properties_for_sampling=None,
            training_mode = training_mode,
            property_name = property_name,
            normalization_file_path = normalization_file_path,
            properties_handle_method = properties_handle_method,
            multilple_values_to_one_property = batch_property,
            number_of_atoms = batch_no_atoms,
            guide_w=guide_w,
            dataset_name=dataset_name,
        )
        molecules.extend(batch_molecules)
    return molecules

def sample_molecules_with_auto_guidance(good_model,
                    bad_model,
                    n_mols, 
                    property_name,
                    properties_handle_method,
                    multiple_values_to_one_property=None,
                    number_of_atoms=None,
                    guide_w=None,
                    fixed_property_val=None,
                    normalization_file_path=None,
                    n_timesteps=100,
                    stochasticity=None, 
                    max_batch_size=128, 
                    xt_traj=False, 
                    ep_traj=False,
                    high_confidence_threshold=None,
                    training_mode=False,
                    device='cuda'):
    assert property_name in allowed_properties, f"Property {property_name} not supported"
    assert (fixed_property_val is None) ^ (multiple_values_to_one_property is None)

    # Load models
    # good_model = load_model_from_ckpt(good_model_ckpt)
    # bad_model = load_model_from_ckpt(bad_model_ckpt)
    # good_model.to(device).eval()
    # bad_model.to(device).eval()

    # Setup guidance weights
    if guide_w is None:
        guide_w = 2.0
    if isinstance(guide_w, (int, float)):
        guide_w = {feat: guide_w for feat in ['x', 'a', 'c', 's', 'e']}
    
    if not fixed_property_val:
        property_value = multiple_values_to_one_property[0]
    else:
        property_value = fixed_property_val

    n_batches = math.ceil(n_mols / max_batch_size)
    molecules = []

    for _ in range(n_batches):
        n_mols_needed = n_mols - len(molecules)
        batch_size = min(n_mols_needed, max_batch_size)
        batch_property = None
        batch_no_atoms = None
        
        if multiple_values_to_one_property is not None:
            batch_property = multiple_values_to_one_property[len(molecules):len(molecules) + batch_size]
        if number_of_atoms is not None:
            batch_no_atoms = number_of_atoms[len(molecules):len(molecules) + batch_size]

        real_atoms_per_molecule = good_model.resolve_atom_counts(
            batch_size,
            device,
            number_of_atoms=batch_no_atoms,
            properties_for_sampling=property_value,
        )
        atoms_per_molecule = good_model.add_fake_atom_capacity(
            real_atoms_per_molecule
        )

        batch_molecules = sample_with_auto_guidance(
            good_model=good_model,
            bad_model=bad_model,
            n_atoms=atoms_per_molecule,
            guide_w=guide_w,
            device=device,
            n_timesteps=n_timesteps,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            properties_for_sampling=property_value,
            property_name=property_name,
            normalization_file_path=normalization_file_path,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=batch_property,
            conditional_generation=True
        )
        molecules.extend(batch_molecules)

    return molecules

def _prepare_property_embedding(model, prop_value, batch_size, device, normalization_file_path=None, property_name=None):
    """Helper function to prepare property embeddings"""
    if normalization_file_path is not None and property_name is not None:
        norm_params = torch.load(normalization_file_path)
        property_idx = PROPERTY_MAP.get(property_name)
        mean = norm_params['mean'][property_idx].item()
        std = norm_params['std'][property_idx].item()
        prop_value = (prop_value - mean) / std
    
    prop_tensor = torch.full((batch_size, 1), prop_value, device=device)
    return model.property_embedder(prop_tensor)

def _handle_categorical_feature(feat, good_model, bad_model, good_pred, bad_pred, guide_w, g, 
                              t_i, alpha_t, alpha_t_prime, s_idx, dt, last_step,
                              node_batch_idx, edge_batch_idx, upper_edge_mask,
                              stochasticity=None, high_confidence_threshold=None):
    """Helper function to handle categorical features (atoms, charges, bonds)"""
    if feat == 'e':
        data_src = g.edata
    else:
        data_src = g.ndata
    
    xt = data_src[f'{feat}_t'].argmax(-1)
    if feat == 'e':
        xt = xt[upper_edge_mask]
    
    # Apply guidance in log space
    log_p_bad = torch.log(bad_pred[feat] + 1e-10)
    log_p_good = torch.log(good_pred[feat] + 1e-10)
    p_s_1 = torch.exp(log_p_bad + guide_w.get(feat, guide_w.get('a', 1.0)) * (log_p_good - log_p_bad))
    p_s_1 = F.softmax(torch.log(p_s_1)/good_model.vector_field.cat_temp_func(t_i), dim=-1)
    class_support_mask = good_model.vector_field.get_category_support_mask(feat)
    if class_support_mask is not None:
        p_s_1 = p_s_1.masked_fill(
            ~class_support_mask.unsqueeze(0), 0.0
        )
        p_s_1 = p_s_1 / p_s_1.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    
    # Campbell step
    if hasattr(good_model.vector_field, 'campbell_step'):
        batch_idx = edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx
        xt, x_1_sampled = good_model.vector_field.campbell_step(
            p_1_given_t=p_s_1, xt=xt,
            stochasticity=stochasticity or good_model.vector_field.eta,
            hc_thresh=high_confidence_threshold or good_model.vector_field.hc_thresh,
            alpha_t=alpha_t[s_idx - 1][good_model.canonical_feat_order.index(feat)],
            alpha_t_prime=alpha_t_prime[s_idx - 1][good_model.canonical_feat_order.index(feat)],
            dt=dt, batch_size=g.batch_size,
            batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
            n_classes=good_model.vector_field.n_cat_feats[feat]+1,
            mask_index=good_model.vector_field.mask_idxs[feat],
            last_step=last_step, batch_idx=batch_idx
        )
    
    # Handle edge features specially
    if feat == 'e':
        xt, x_1_sampled = _handle_edge_features(xt, x_1_sampled, g)
    
    return xt, x_1_sampled

from molguidance.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask
from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.models.ctmc_vector_field import PROPERTY_MAP

def _assert_sampling_compatible(reference_model, *other_models):
    for model in other_models:
        if reference_model.canonical_feat_order != model.canonical_feat_order:
            raise ValueError('guidance models use different feature schemas')
        if getattr(reference_model, 'atom_state_vocab', None) != getattr(
            model, 'atom_state_vocab', None
        ):
            raise ValueError(
                'guidance models use different atom-state codecs'
            )
        for feat in reference_model.canonical_feat_order:
            reference_support = (
                reference_model.vector_field.get_category_support_mask(feat)
            )
            model_support = model.vector_field.get_category_support_mask(feat)
            reference_signature = (
                None if reference_support is None
                else tuple(reference_support.detach().cpu().tolist())
            )
            model_signature = (
                None if model_support is None
                else tuple(model_support.detach().cpu().tolist())
            )
            if reference_signature != model_signature:
                raise ValueError(
                    'guidance models use different support masks'
                )


@torch.no_grad()
def sample_with_auto_guidance(good_model, bad_model, n_atoms, guide_w, **kwargs):
    """Sample molecules using auto-guidance with good and bad models"""
    _assert_sampling_compatible(good_model, bad_model)
    # Initialize graph
    g = _create_batched_graph(n_atoms, kwargs.get('device', 'cuda:0'))
    upper_edge_mask = get_upper_edge_mask(g)
    node_batch_idx, edge_batch_idx = get_batch_idxs(g)
    g = good_model.sample_prior(g, node_batch_idx, upper_edge_mask)
    
    # Setup integration
    t = torch.linspace(0, 1, kwargs.get('n_timesteps', 100), device=kwargs.get('device', 'cuda:0'))
    alpha_t = good_model.interpolant_scheduler.alpha_t(t)
    alpha_t_prime = good_model.interpolant_scheduler.alpha_t_prime(t)
    
    # Initialize features
    _initialize_features(g, good_model)
    
    previous_good_endpoint = None
    previous_bad_endpoint = None

    # Integration loop
    for s_idx in range(1, len(t)):
        # Get predictions from both models
        (
            good_pred,
            bad_pred,
            previous_good_endpoint,
            previous_bad_endpoint,
        ) = _get_model_predictions(
            good_model, bad_model, g, t, s_idx, 
            kwargs.get('properties_for_sampling'), 
            kwargs.get('normalization_file_path'),
            kwargs.get('property_name'),
            previous_good_endpoint,
            previous_bad_endpoint,
        )
        
        # Update positions
        _update_positions(g, good_pred, bad_pred, guide_w, t[s_idx] - t[s_idx-1])
        
        # Update categorical features
        for feat in good_model.canonical_feat_order:
            if feat != 'x':
                xt, x_1_sampled = _handle_categorical_feature(
                    feat, good_model, bad_model, good_pred, bad_pred, guide_w, g,
                    t[s_idx-1], alpha_t, alpha_t_prime, s_idx, 
                    t[s_idx] - t[s_idx-1], s_idx == len(t)-1,
                    *get_batch_idxs(g), get_upper_edge_mask(g),
                    kwargs.get('stochasticity'), kwargs.get('high_confidence_threshold')
                )
                if feat == 'e':
                    g.edata[f'{feat}_t'] = xt
                    g.edata[f'{feat}_1_pred'] = x_1_sampled
                else:
                    g.ndata[f'{feat}_t'] = xt
                    g.ndata[f'{feat}_1_pred'] = x_1_sampled
    
    return _build_molecules(g, good_model, kwargs.get('xt_traj', False), kwargs.get('ep_traj', False))

def _create_batched_graph(n_atoms, device):
    """Create a batched graph for molecule generation"""
    edge_idxs_dict = {}
    for n_atoms_i in torch.unique(n_atoms):
        edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

    g = []
    for n_atoms_i in n_atoms:
        edge_idxs = edge_idxs_dict[int(n_atoms_i)]
        g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
        g.append(g_i)

    return dgl.batch(g)

def _initialize_features(g, model):
    """Initialize graph features from prior"""
    for feat in model.canonical_feat_order:
        if feat == 'e':
            g.edata[f'{feat}_t'] = g.edata[f'{feat}_0']
        else:
            g.ndata[f'{feat}_t'] = g.ndata[f'{feat}_0']

def _get_model_predictions(
    good_model,
    bad_model,
    g,
    t,
    s_idx,
    props_sampling,
    norm_file,
    prop_name,
    previous_good_endpoint=None,
    previous_bad_endpoint=None,
):
    """Get predictions from both models"""
    t_i = t[s_idx - 1]
    batch_size, device = g.batch_size, g.device
    
    # Prepare property embeddings for both models
    prop_emb_good = _prepare_property_embedding(good_model, props_sampling, batch_size, device, norm_file, prop_name)
    prop_emb_bad = _prepare_property_embedding(bad_model, props_sampling, batch_size, device, norm_file, prop_name)
    
    # Combine time and property embeddings
    t_tensor = torch.full((batch_size, 1), t_i, device=device)
    t_combined_good = torch.cat([t_tensor, prop_emb_good], dim=-1)
    t_combined_bad = torch.cat([t_tensor, prop_emb_bad], dim=-1)
    
    node_batch_idx, edge_batch_idx = get_batch_idxs(g)
    upper_edge_mask = get_upper_edge_mask(g)
    
    # Get predictions from both models
    good_pred = good_model.vector_field(
        g,
        t_combined_good,
        node_batch_idx,
        upper_edge_mask,
        apply_softmax=True,
        remove_com=True,
        previous_endpoint=previous_good_endpoint,
    )
    bad_pred = bad_model.vector_field(
        g,
        t_combined_bad,
        node_batch_idx,
        upper_edge_mask,
        apply_softmax=True,
        remove_com=True,
        previous_endpoint=previous_bad_endpoint,
    )
    next_good_endpoint = (
        good_model.vector_field.prepare_self_conditioning_endpoint(
            good_pred, categorical_are_logits=False
        )
    )
    next_bad_endpoint = (
        bad_model.vector_field.prepare_self_conditioning_endpoint(
            bad_pred, categorical_are_logits=False
        )
    )
    
    return good_pred, bad_pred, next_good_endpoint, next_bad_endpoint


def _get_cfg_branch_predictions(
    cfg_model,
    g,
    t_i,
    node_batch_idx,
    upper_edge_mask,
    property_value,
    normalization_file_path=None,
    property_name=None,
    previous_unconditional_endpoint=None,
    previous_conditional_endpoint=None,
):
    """Predict CFG branches while preserving branch-local endpoint history."""
    batch_size, device = g.batch_size, g.device
    conditional_embedding = _prepare_property_embedding(
        cfg_model,
        property_value,
        batch_size,
        device,
        normalization_file_path,
        property_name,
    )
    unconditional_embedding = cfg_model.unconditional_embedder(
        batch_size, device=device
    )
    time_column = torch.full(
        (batch_size, 1), t_i, device=device, dtype=torch.float32
    )
    unconditional_time = torch.cat(
        (time_column, unconditional_embedding), dim=-1
    )
    conditional_time = torch.cat(
        (time_column, conditional_embedding), dim=-1
    )

    unconditional_prediction = cfg_model.vector_field(
        g,
        unconditional_time,
        node_batch_idx,
        upper_edge_mask,
        apply_softmax=False,
        remove_com=True,
        previous_endpoint=previous_unconditional_endpoint,
    )
    conditional_prediction = cfg_model.vector_field(
        g,
        conditional_time,
        node_batch_idx,
        upper_edge_mask,
        apply_softmax=False,
        remove_com=True,
        previous_endpoint=previous_conditional_endpoint,
    )
    next_unconditional_endpoint = (
        cfg_model.vector_field.prepare_self_conditioning_endpoint(
            unconditional_prediction, categorical_are_logits=True
        )
    )
    next_conditional_endpoint = (
        cfg_model.vector_field.prepare_self_conditioning_endpoint(
            conditional_prediction, categorical_are_logits=True
        )
    )
    return (
        unconditional_prediction,
        conditional_prediction,
        next_unconditional_endpoint,
        next_conditional_endpoint,
    )


def _advance_categorical_endpoint(
    g,
    feat,
    endpoint_probabilities,
    model,
    feat_idx,
    s_idx,
    t,
    alpha_t,
    alpha_t_prime,
    node_batch_idx,
    edge_batch_idx,
    upper_edge_mask,
    stochasticity=None,
    high_confidence_threshold=None,
):
    """Advance one categorical CTMC state from endpoint probabilities."""
    data_src = g.edata if feat == "e" else g.ndata
    xt = data_src[f"{feat}_t"].argmax(-1)
    if feat == "e":
        xt = xt[upper_edge_mask]

    actual_stochasticity = (
        model.vector_field.eta
        if stochasticity is None
        else stochasticity
    )
    actual_threshold = (
        model.vector_field.hc_thresh
        if high_confidence_threshold is None
        else high_confidence_threshold
    )
    xt, x_1_sampled = model.vector_field.campbell_step(
        p_1_given_t=endpoint_probabilities,
        xt=xt,
        stochasticity=actual_stochasticity,
        hc_thresh=actual_threshold,
        alpha_t=alpha_t[s_idx - 1][feat_idx],
        alpha_t_prime=alpha_t_prime[s_idx - 1][feat_idx],
        dt=t[s_idx] - t[s_idx - 1],
        batch_size=g.batch_size,
        batch_num_nodes=(
            g.batch_num_edges() // 2
            if feat == "e"
            else g.batch_num_nodes()
        ),
        n_classes=model.vector_field.n_cat_feats[feat] + 1,
        mask_index=model.vector_field.mask_idxs[feat],
        last_step=s_idx == len(t) - 1,
        batch_idx=(
            edge_batch_idx[upper_edge_mask]
            if feat == "e"
            else node_batch_idx
        ),
    )
    if feat == "e":
        xt, x_1_sampled = _handle_edge_features(xt, x_1_sampled, g)
    data_src[f"{feat}_t"] = xt
    data_src[f"{feat}_1_pred"] = x_1_sampled


def _update_positions(g, good_pred, bad_pred, guide_w, dt):
    """Update position features using guided predictions"""
    x_t = g.ndata['x_t']
    guide_w_pos = guide_w['x']
    
    # Apply guidance
    vf = bad_pred['x'] - x_t + guide_w_pos * (good_pred['x'] - bad_pred['x'])
    
    # Update positions
    g.ndata['x_t'] = x_t + dt * vf
    g.ndata['x_1_pred'] = bad_pred['x'] + guide_w_pos * (good_pred['x'] - bad_pred['x'])

def _handle_edge_features(xt, x_1_sampled, g):
    """Handle edge feature updates"""
    e_t = torch.zeros_like(g.edata['e_t'])
    e_1_sampled = torch.zeros_like(g.edata['e_t'])
    
    upper_edge_mask = get_upper_edge_mask(g)
    e_t[upper_edge_mask] = xt
    e_t[~upper_edge_mask] = xt
    e_1_sampled[upper_edge_mask] = x_1_sampled
    e_1_sampled[~upper_edge_mask] = x_1_sampled
    
    return e_t, e_1_sampled

def _build_molecules(g, model, xt_traj=False, ep_traj=False):
    """Build final molecule objects"""
    # Set final values
    for feat in model.canonical_feat_order:
        if feat == 'e':
            g.edata[f'{feat}_1'] = g.edata[f'{feat}_t']
        else:
            g.ndata[f'{feat}_1'] = g.ndata[f'{feat}_t']
    
    g.edata['ue_mask'] = get_upper_edge_mask(g)
    g = g.to('cpu')
    
    # Build molecules
    return [SampledMolecule(g_i, model.atom_type_map,
                           ctmc_mol=model.parameterization == 'ctmc',
                           build_xt_traj=xt_traj,
                           build_ep_traj=ep_traj,
                           exclude_charges=model.exclude_charges,
                           atom_state_vocab=getattr(
                               model, 'atom_state_vocab', None
                           ))
            for g_i in dgl.unbatch(g)]

@torch.no_grad()
def sample_with_hybrid_guidance(
    good_model,
    bad_model,
    cfg_model,
    n_atoms,
    auto_guide_w={"x": 2.0, "a": 1.0, "c": 1.0, "e": 1.0},
    cfg_guide_w={"x": 2.0, "a": 1.0, "c": 1.0, "e": 1.0},
    interpolation_weight=0.5,
    **kwargs,
):
    """Sample with a log-probability blend of auto-guidance and CFG.

    Each of the four denoising branches owns an independent previous-endpoint
    history. Feeding a guided endpoint back into a branch would change the
    model represented by that branch.
    """
    _assert_sampling_compatible(good_model, bad_model, cfg_model)
    if not 0.0 <= float(interpolation_weight) <= 1.0:
        raise ValueError("interpolation_weight must be in [0, 1]")

    device = kwargs.get("device", "cuda")
    g = _create_batched_graph(n_atoms, device)
    upper_edge_mask = get_upper_edge_mask(g)
    node_batch_idx, edge_batch_idx = get_batch_idxs(g)
    g = good_model.sample_prior(g, node_batch_idx, upper_edge_mask)
    _initialize_features(g, good_model)

    t = torch.linspace(0, 1, kwargs.get("n_timesteps", 100), device=device)
    alpha_t = good_model.interpolant_scheduler.alpha_t(t)
    alpha_t_prime = good_model.interpolant_scheduler.alpha_t_prime(t)

    previous_good_endpoint = None
    previous_bad_endpoint = None
    previous_cfg_unconditional_endpoint = None
    previous_cfg_conditional_endpoint = None

    for s_idx in range(1, len(t)):
        (
            good_pred,
            bad_pred,
            previous_good_endpoint,
            previous_bad_endpoint,
        ) = _get_model_predictions(
            good_model,
            bad_model,
            g,
            t,
            s_idx,
            kwargs.get("properties_for_sampling"),
            kwargs.get("normalization_file_path"),
            kwargs.get("property_name"),
            previous_good_endpoint,
            previous_bad_endpoint,
        )
        (
            cfg_unconditional_pred,
            cfg_conditional_pred,
            previous_cfg_unconditional_endpoint,
            previous_cfg_conditional_endpoint,
        ) = _get_cfg_branch_predictions(
            cfg_model,
            g,
            t[s_idx - 1],
            node_batch_idx,
            upper_edge_mask,
            kwargs.get("properties_for_sampling"),
            kwargs.get("normalization_file_path"),
            kwargs.get("property_name"),
            previous_cfg_unconditional_endpoint,
            previous_cfg_conditional_endpoint,
        )

        for feat_idx, feat in enumerate(good_model.canonical_feat_order):
            if feat == "x":
                x_t = g.ndata["x_t"]
                alpha = alpha_t[s_idx - 1][feat_idx]
                alpha_prime = alpha_t_prime[s_idx - 1][feat_idx]
                vf_bad = bad_model.vector_field.vector_field(
                    x_t, bad_pred["x"], alpha, alpha_prime
                )
                vf_good = good_model.vector_field.vector_field(
                    x_t, good_pred["x"], alpha, alpha_prime
                )
                vf_cfg_unconditional = cfg_model.vector_field.vector_field(
                    x_t, cfg_unconditional_pred["x"], alpha, alpha_prime
                )
                vf_cfg_conditional = cfg_model.vector_field.vector_field(
                    x_t, cfg_conditional_pred["x"], alpha, alpha_prime
                )
                vf_auto = vf_bad + auto_guide_w["x"] * (vf_good - vf_bad)
                vf_cfg = vf_cfg_unconditional + cfg_guide_w["x"] * (
                    vf_cfg_conditional - vf_cfg_unconditional
                )
                vf = (
                    (1.0 - interpolation_weight) * vf_cfg
                    + interpolation_weight * vf_auto
                )
                g.ndata["x_t"] = x_t + (t[s_idx] - t[s_idx - 1]) * vf

                x_1_auto = bad_pred["x"] + auto_guide_w["x"] * (
                    good_pred["x"] - bad_pred["x"]
                )
                x_1_cfg = cfg_unconditional_pred["x"] + cfg_guide_w["x"] * (
                    cfg_conditional_pred["x"]
                    - cfg_unconditional_pred["x"]
                )
                g.ndata["x_1_pred"] = (
                    (1.0 - interpolation_weight) * x_1_cfg
                    + interpolation_weight * x_1_auto
                ).detach().clone()
                continue

            auto_probability = combine_guided_endpoint_probabilities(
                bad_pred[feat],
                good_pred[feat],
                auto_guide_w.get(feat, auto_guide_w.get("a", 1.0)),
                temperature=1.0,
                guidance_format="log",
                probability_mode="corrected",
            )
            cfg_probability = combine_guided_endpoint_probabilities(
                F.softmax(cfg_unconditional_pred[feat], dim=-1),
                F.softmax(cfg_conditional_pred[feat], dim=-1),
                cfg_guide_w.get(feat, cfg_guide_w.get("a", 1.0)),
                temperature=1.0,
                guidance_format="log",
                probability_mode="corrected",
            )
            hybrid_log_probability = (
                (1.0 - interpolation_weight)
                * torch.log(cfg_probability.clamp_min(1e-9))
                + interpolation_weight
                * torch.log(auto_probability.clamp_min(1e-9))
            )
            temperature = good_model.vector_field.cat_temp_func(t[s_idx - 1])
            p_s_1 = F.softmax(
                hybrid_log_probability / temperature, dim=-1
            )
            class_support_mask = (
                good_model.vector_field.get_category_support_mask(feat)
            )
            if class_support_mask is not None:
                p_s_1 = p_s_1.masked_fill(
                    ~class_support_mask.unsqueeze(0), 0.0
                )
                p_s_1 = p_s_1 / p_s_1.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-9)

            _advance_categorical_endpoint(
                g,
                feat,
                p_s_1,
                good_model,
                feat_idx,
                s_idx,
                t,
                alpha_t,
                alpha_t_prime,
                node_batch_idx,
                edge_batch_idx,
                upper_edge_mask,
                kwargs.get("stochasticity"),
                kwargs.get("high_confidence_threshold"),
            )

    return _build_molecules(
        g,
        good_model,
        kwargs.get("xt_traj", False),
        kwargs.get("ep_traj", False),
    )
