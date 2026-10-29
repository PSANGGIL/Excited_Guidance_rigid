import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from typing import Union, Callable
import scipy

from molguidance.models.gvp import GVPConv, GVP, _rbf, _norm_no_nan
from molguidance.models.interpolant_scheduler import InterpolantScheduler
from molguidance.models.self_conditioning import SelfConditioningResidualLayer
from molguidance.models.chemistry_feedback import JointChemicalFeedback
from molguidance.utils.dirflow import DirichletConditionalFlow, simplex_proj

class EndpointVectorField(nn.Module):

    def __init__(self, n_atom_types: int,
                    canonical_feat_order: list,
                    interpolant_scheduler: InterpolantScheduler,
                    n_charges: int = 6,
                    n_bond_types: int = 5,
                    n_hydrogen_counts: int = 0,
                    n_atom_states: int = 0,
                    category_support_masks: dict | None = None,
                    n_vec_channels: int = 16,
                    n_cp_feats: int = 0, 
                    n_hidden_scalars: int = 64,
                    n_hidden_edge_feats: int = 64,
                    n_recycles: int = 1,
                    n_molecule_updates: int = 2, 
                    convs_per_update: int = 2,
                    n_message_gvps: int = 3, 
                    n_update_gvps: int = 3,
                    separate_mol_updaters: bool = False,
                    message_norm: Union[float, str] = 100,
                    update_edge_w_distance: bool = False,
                    rbf_dmax = 20,
                    rbf_dim = 16,
                    exclude_charges: bool = False,
                    continuous_inv_temp_schedule = None,
                    continuous_inv_temp_max: float = 10.0,
                    self_conditioning: bool = False,
                    independent_modality_time: bool = False,
                    bond_first_geometry: bool = False,
                    bond_coordinate_scale: float = 0.1,
                    dropout: float = 0.0,
                    chemistry_feedback: bool = False,
                    chemistry_state_metadata: list | None = None,
                    has_mask: bool = False # if we are using CTMC, input categorical features will have mask tokens,
                    # this means their one-hot representations will have an extra dimension,
                    # and the neural network instantiated by this method need to account for this
                    # it is definitely anti-pattern to have a parameter in parent class that is only needed for one sub-class (CTMCVectorField)
                    # however, this is the fastest way to get CTMCVectorField working right now, so we will be anti-pattern for the sake of time
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hydrogen_counts = int(n_hydrogen_counts)
        self.n_atom_states = int(n_atom_states)
        if self.n_atom_states < 0:
            raise ValueError('n_atom_states must be non-negative')
        self.use_joint_atom_state = self.n_atom_states > 0
        if self.use_joint_atom_state and self.n_hydrogen_counts > 0:
            raise ValueError(
                'joint atom state already contains charge and H count; '
                'n_hydrogen_counts must be zero'
            )
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.exclude_charges = exclude_charges
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = canonical_feat_order

        if self.exclude_charges:
            self.n_charges = 0

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.self_conditioning = bool(self_conditioning)
        self.independent_modality_time = bool(independent_modality_time)
        self.bond_first_geometry = bool(bond_first_geometry)
        self.bond_coordinate_scale = float(bond_coordinate_scale)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("GVP dropout must be in [0, 1)")
        self.dropout_rate = float(dropout)
        if (
            not torch.isfinite(torch.tensor(self.bond_coordinate_scale))
            or self.bond_coordinate_scale <= 0.0
        ):
            raise ValueError('bond_coordinate_scale must be finite and positive')

        assert n_vec_channels >= 3, 'n_vec_channels must be >= 3'

        self.continuous_inv_temp_schedule = continuous_inv_temp_schedule
        self.continouts_inv_temp_max = continuous_inv_temp_max
        self.continuous_inv_temp_func = self.build_continuous_inv_temp_func(self.continuous_inv_temp_schedule, self.continouts_inv_temp_max) 

        if self.use_joint_atom_state:
            self.node_categorical_feats = ['s']
            self.n_cat_feats = {
                's': self.n_atom_states,
                'e': self.n_bond_types,
            }
        else:
            self.node_categorical_feats = ['a']
            self.n_cat_feats = {
                'a': self.n_atom_types,
                'c': self.n_charges,
                'e': self.n_bond_types,
            }
            if not self.exclude_charges:
                self.node_categorical_feats.append('c')
            if self.n_hydrogen_counts > 0:
                self.node_categorical_feats.append('h')
                self.n_cat_feats['h'] = self.n_hydrogen_counts

        category_support_masks = category_support_masks or {}
        self._category_support_buffer_names = {}
        for feat, support in category_support_masks.items():
            if feat not in self.n_cat_feats or feat == 'e':
                raise ValueError(f'unsupported category-support feature: {feat}')
            support_tensor = torch.as_tensor(support, dtype=torch.bool)
            expected = self.n_cat_feats[feat]
            if support_tensor.shape != (expected,):
                raise ValueError(
                    f'{feat} support mask must have shape ({expected},), '
                    f'got {tuple(support_tensor.shape)}'
                )
            if not bool(support_tensor.any()):
                raise ValueError(f'{feat} support mask cannot exclude every class')
            buffer_name = f'_category_support_{feat}'
            self.register_buffer(buffer_name, support_tensor, persistent=False)
            self._category_support_buffer_names[feat] = buffer_name

        n_mask_feats = int(has_mask)
        self.n_mask_feats = n_mask_feats

        node_category_dim = sum(
            self.n_cat_feats[feat] for feat in self.node_categorical_feats
        )
        node_input_dim = (
            node_category_dim
            + 1
            + int(self.independent_modality_time)
            + len(self.node_categorical_feats) * n_mask_feats
        )
        self.node_input_dim = node_input_dim
        self.scalar_embedding = nn.Sequential(
            nn.Linear(node_input_dim, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types + n_mask_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats)
        )

        self.self_conditioner = None
        self.chemistry_feedback = None
        if chemistry_feedback:
            if not self.use_joint_atom_state or not self.self_conditioning or n_bond_types != 4:
                raise ValueError("chemistry feedback requires joint Kekule states and self-conditioning")
            # Optional modules must not shift common-backbone initialization
            # under the same seed in paired ablations.
            with torch.random.fork_rng(devices=[]):
                self.chemistry_feedback = JointChemicalFeedback(
                    chemistry_state_metadata, n_hidden_scalars, n_hidden_edge_feats
                )
        if self.self_conditioning:
            self.self_conditioner = SelfConditioningResidualLayer(
                node_feature_names=self.node_categorical_feats,
                node_feature_dims=[
                    self.n_cat_feats[feat] for feat in self.node_categorical_feats
                ],
                edge_probability_dim=self.n_bond_types,
                node_hidden_dim=self.n_hidden_scalars,
                edge_hidden_dim=self.n_hidden_edge_feats,
                rbf_dim=self.rbf_dim,
                rbf_dmax=self.rbf_dmax,
            )

        self.conv_layers = []
        for conv_idx in range(convs_per_update*n_molecule_updates):
            self.conv_layers.append(GVPConv(
                scalar_size=n_hidden_scalars,
                vector_size=n_vec_channels,
                n_cp_feats=n_cp_feats,
                edge_feat_size=n_hidden_edge_feats,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim,
                dropout=self.dropout_rate,
            )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        self.bond_coordinate_updaters = nn.ModuleList([])
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(NodePositionUpdate(n_hidden_scalars, n_vec_channels, n_gvps=3, n_cp_feats=n_cp_feats))
            self.edge_updaters.append(EdgeUpdate(n_hidden_scalars, n_hidden_edge_feats, update_edge_w_distance=update_edge_w_distance, rbf_dim=rbf_dim))
            if self.bond_first_geometry:
                self.bond_coordinate_updaters.append(
                    BondFirstCoordinateUpdate(
                        n_node_scalars=n_hidden_scalars,
                        n_edge_feats=n_hidden_edge_feats,
                        rbf_dim=rbf_dim,
                        coordinate_scale=self.bond_coordinate_scale,
                    )
                )


        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(
                n_hidden_scalars,
                node_category_dim,
            )
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types)
        )

    def build_node_scalar_inputs(
        self,
        g: dgl.DGLGraph,
        time_tensor: torch.Tensor,
        node_batch_idx: torch.Tensor,
        graph_time_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate noisy node states and modality times canonically.

        Legacy fields consume only the coordinate time.  Stage-6 fields also
        consume a separate graph time so the endpoint is identifiable when
        coordinates and categorical graph variables are corrupted at different
        points along their paths.
        """
        features = [g.ndata[f'{self.node_categorical_feats[0]}_t']]
        features.append(time_tensor[node_batch_idx].unsqueeze(-1))
        if self.independent_modality_time:
            if graph_time_tensor is None:
                graph_time_tensor = time_tensor
            if graph_time_tensor.shape != time_tensor.shape:
                raise ValueError(
                    'graph_time_tensor must have the same shape as time_tensor'
                )
            features.append(graph_time_tensor[node_batch_idx].unsqueeze(-1))
        features.extend(
            g.ndata[f'{feat}_t'] for feat in self.node_categorical_feats[1:]
        )
        return torch.cat(features, dim=-1)

    def update_molecule_state(
        self,
        g: dgl.DGLGraph,
        node_scalar_features: torch.Tensor,
        node_vec_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_positions: torch.Tensor,
        d: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        updater_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one topology/geometry update under an explicit causal order.

        The legacy route is deliberately unchanged.  In the Stage-6 route the
        invariant edge state is refreshed first, an antisymmetric pair update
        lets that refreshed topology move coordinates, and only then does the
        existing GVP coordinate head add its residual.  Distances are recomputed
        once from the final coordinates returned by the block.
        """
        if self.bond_first_geometry:
            edge_features = self.edge_updaters[updater_idx](
                g, node_scalar_features, edge_features, d=d
            )
            node_positions = self.bond_coordinate_updaters[updater_idx](
                g,
                node_scalar_features,
                edge_features,
                node_positions,
                d,
                upper_edge_mask,
            )
            node_positions = self.node_position_updaters[updater_idx](
                node_scalar_features, node_positions, node_vec_features
            )
        else:
            node_positions = self.node_position_updaters[updater_idx](
                node_scalar_features, node_positions, node_vec_features
            )
            x_diff, d = self.precompute_distances(g, node_positions)
            edge_features = self.edge_updaters[updater_idx](
                g, node_scalar_features, edge_features, d=d
            )
            return node_positions, edge_features, x_diff, d

        x_diff, d = self.precompute_distances(g, node_positions)
        return node_positions, edge_features, x_diff, d

    def molecule_update_index(self, conv_idx: int) -> int | None:
        """Return the updater index or ``None`` when no update is due.

        The released implementation skipped convolution zero.  That is kept for
        legacy checkpoints.  Bond-first Stage-6 fixes the off-by-one so
        ``convs_per_update=1`` really performs one molecule update per block.
        """
        if (conv_idx + 1) % self.convs_per_update != 0:
            return None
        if not self.bond_first_geometry and conv_idx == 0:
            return None
        if self.separate_mol_updaters:
            if self.bond_first_geometry:
                return (conv_idx + 1) // self.convs_per_update - 1
            return conv_idx // self.convs_per_update
        return 0

    def split_node_output(
        self,
        output: torch.Tensor,
        enforce_support: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Split the shared node head using the same feature specification."""
        result = {}
        offset = 0
        for feat in self.node_categorical_feats:
            width = self.n_cat_feats[feat]
            logits = output[:, offset:offset + width]
            result[feat] = (
                self.apply_category_support(logits, feat)
                if enforce_support else logits
            )
            offset += width
        if offset != output.shape[-1]:
            raise RuntimeError(
                f'node output width mismatch: consumed {offset}, got {output.shape[-1]}'
            )
        return result

    def get_category_support_mask(self, feat: str) -> torch.Tensor | None:
        """Return the train-supported endpoint classes for a feature."""
        buffer_name = self._category_support_buffer_names.get(feat)
        if buffer_name is None:
            return None
        return getattr(self, buffer_name)

    def apply_category_support(
        self, logits: torch.Tensor, feat: str
    ) -> torch.Tensor:
        """Prevent held-out-only codec states from being generated."""
        support = self.get_category_support_mask(feat)
        if support is None:
            return logits
        # A finite sentinel avoids the undefined 0 * (-inf) term in CFG at
        # guide_weight=1 while still underflowing to zero after softmax.
        return logits.masked_fill(~support.unsqueeze(0), -1.0e4)

    def prepare_self_conditioning_endpoint(
        self,
        endpoint: dict[str, torch.Tensor],
        *,
        categorical_are_logits: bool,
    ) -> dict[str, torch.Tensor] | None:
        """Detach and canonicalize an endpoint for the next denoising call.

        The residual layer consumes probabilities rather than raw logits. This
        keeps training and inference under one explicit endpoint contract and
        applies the train-vocabulary support mask before feedback.
        """
        if not self.self_conditioning:
            return None
        expected = {"x", "e", *self.node_categorical_feats}
        missing = expected - set(endpoint)
        if missing:
            raise KeyError(
                "endpoint is missing self-conditioning features: "
                + ", ".join(sorted(missing))
            )

        prepared = {"x": endpoint["x"].detach()}
        for feat in (*self.node_categorical_feats, "e"):
            value = endpoint[feat].detach()
            support = (
                None if feat == "e" else self.get_category_support_mask(feat)
            )
            if categorical_are_logits:
                value = torch.nan_to_num(
                    value, nan=0.0, posinf=1.0e4, neginf=-1.0e4
                )
                if support is not None:
                    value = value.masked_fill(~support.unsqueeze(0), -1.0e4)
                value = torch.softmax(value, dim=-1)
            else:
                value = torch.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp_min(0.0)
                if support is not None:
                    value = value.masked_fill(~support.unsqueeze(0), 0.0)
                row_mass = value.sum(dim=-1, keepdim=True)
                if support is None:
                    fallback = torch.full_like(
                        value, 1.0 / value.shape[-1]
                    )
                else:
                    fallback = support.to(value).expand_as(value)
                    fallback = fallback / fallback.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1.0)
                value = torch.where(
                    row_mass > 1.0e-12,
                    value / row_mass.clamp_min(1.0e-12),
                    fallback,
                )
            prepared[feat] = value.detach()
        return prepared

    def apply_self_conditioning(
        self,
        g: dgl.DGLGraph,
        node_scalar_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_positions: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        previous_endpoint: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_endpoint is None:
            return node_scalar_features, edge_features
        if self.self_conditioner is None:
            raise RuntimeError(
                "a previous endpoint was supplied but self-conditioning is disabled"
            )
        node_scalar_features, edge_features = self.self_conditioner(
            g,
            node_hidden=node_scalar_features,
            edge_hidden=edge_features,
            current_positions=node_positions,
            upper_edge_mask=upper_edge_mask,
            previous_endpoint=previous_endpoint,
        )
        if self.chemistry_feedback is not None:
            node_scalar_features, edge_features = self.chemistry_feedback(
                g, node_scalar_features, edge_features, upper_edge_mask,
                previous_endpoint,
            )
        return node_scalar_features, edge_features

    def build_continuous_inv_temp_func(self, schedule, max_inv_temp=None):

        if schedule is None:
            inv_temp_func = lambda t: 1.0
        elif schedule == 'linear':
            inv_temp_func = lambda t: max_inv_temp*(1 - t)
        elif callable(schedule):
            inv_temp_func = schedule
        else:
            raise ValueError(f'Invalid continuous_inv_temp_schedule: {schedule}')
        return inv_temp_func
        

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False,
                 previous_endpoint: dict[str, torch.Tensor] | None = None,
                 graph_time: torch.Tensor | None = None):
        """Predict x_1 (trajectory destination) given x_t"""
        device = g.device

        with g.local_scope():
            # gather node and edge features for input to convolutions
            node_scalar_features = self.build_node_scalar_inputs(
                g, t, node_batch_idx, graph_time_tensor=graph_time
            )
            node_scalar_features = self.scalar_embedding(node_scalar_features)

            node_positions = g.ndata['x_t']

            num_nodes = g.num_nodes()

            # initialize the vector features for every node to be zeros
            node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)
            # i thought setting the first three channels to the identity matrix would be a good idea,
            # but this actually breaks rotational equivariance
            # node_vec_features[:, :3, :] = torch.eye(3, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)

            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)

            node_scalar_features, edge_features = self.apply_self_conditioning(
                g,
                node_scalar_features,
                edge_features,
                node_positions,
                upper_edge_mask,
                previous_endpoint,
            )

            x_diff, d = self.precompute_distances(g)
            for recycle_idx in range(self.n_recycles):
                for conv_idx, conv in enumerate(self.conv_layers):

                    # perform a single convolution which updates node scalar and vector features (but not positions)
                    node_scalar_features, node_vec_features = conv(g, 
                            scalar_feats=node_scalar_features, 
                            coord_feats=node_positions,
                            vec_feats=node_vec_features,
                            edge_feats=edge_features,
                            x_diff=x_diff,
                            d=d
                    )

                    # every convs_per_update convolutions, update the node positions and edge features
                    updater_idx = self.molecule_update_index(conv_idx)
                    if updater_idx is not None:
                        node_positions, edge_features, x_diff, d = (
                            self.update_molecule_state(
                                g,
                                node_scalar_features,
                                node_vec_features,
                                edge_features,
                                node_positions,
                                d,
                                upper_edge_mask,
                                updater_idx,
                            )
                        )

            
            # predict final categorical node states
            node_logits = self.split_node_output(
                self.node_output_head(node_scalar_features),
                enforce_support=apply_softmax,
            )

            # predict the final edge logits
            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            # project node positions back into zero-COM subspace
            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {'x': node_positions, **node_logits, 'e': edge_logits}

        # apply softmax to categorical features, if requested
        # at training time, we don't want to apply softmax because we use cross-entropy loss which includes softmax
        # at inference time, we want to apply softmax to get a vector which lies on the simplex
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in self.n_cat_feats: # if this is a categorical feature
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1) # apply softmax to this feature

        return dst_dict
    
    def precompute_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            if node_positions is None:
                g.ndata['x_d'] = g.ndata['x_t']
            else:
                g.ndata['x_d'] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d
      
    def integrate(self, g: dgl.DGLGraph, 
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor, 
        n_timesteps: int, 
        visualize=False, **kwargs):
        """Integrate the trajectories of molecules along the vector field."""

        # get the timepoint for integration
        t = torch.linspace(0, 1, n_timesteps, device=g.device)

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']


        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]
                traj_frames[f'{feat}_1_pred'] = []
    
        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # compute next step and set x_t = x_s
            g = self.step(g, s_i, t_i, alpha_t_i, alpha_s_i, alpha_t_prime_i, node_batch_idx, upper_edge_mask, **kwargs)

            if visualize:
                for feat in self.canonical_feat_order:

                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    if feat == 'e':
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)


                    # record endpoint frame for visualization
                    ep_key = f'{feat}_1_pred'
                    if ep_key not in g_data_src: 
                        # the endpoint key wont be there for VectorField because
                        # i haven't dervived a method of obtaining intermediate xhats from the vector field
                        continue
                    ep_frame = g_data_src[ep_key].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[ep_key].append(ep_frame)

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames
        
        return g
    
    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
             inv_temp_func=None,
            **kwargs):
        
        if inv_temp_func is None:
            inv_temp_func = self.continuous_inv_temp_func
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == "e":
                data_src = g.edata
            else:
                data_src = g.ndata

            x_t = data_src[f'{feat}_t']
            x_1 = dst_dict[feat]

            if feat == "e":
                x_t = x_t[upper_edge_mask]

            # evaluate the vector field at the current timepoint
            vf = self.vector_field(x_t, x_1, alpha_t_i[feat_idx], alpha_t_prime_i[feat_idx])

            # apply temperature scaling
            vf = vf*inv_temp_func(t_i)

            # x1_weight = alpha_t_prime_i[feat_idx]*(s_i - t_i)/(1 - alpha_t_i[feat_idx])
            # xt_weight = 1 - x1_weight

            # apply euler integration step
            x_s = x_t + vf*(s_i - t_i)

            if feat == "e":

                # set the edge features so that corresponding upper and lower triangle edges have the same value
                e_s = torch.zeros_like(g.edata['e_0'])
                e_s[upper_edge_mask] = x_s
                e_s[~upper_edge_mask] = x_s
                x_s = e_s

                e_1 = torch.zeros_like(g.edata['e_0'])
                e_1[upper_edge_mask] = dst_dict[feat]
                e_1[~upper_edge_mask] = dst_dict[feat]
                x_1 = e_1

            # record predicted endoint, for visualization purposes
            data_src[f'{feat}_1_pred'] = x_1.detach().clone()

            # record updated feature in the graph
            data_src[f'{feat}_t'] = x_s

        return g


    def vector_field(self, x_t, x_1, alpha_t, alpha_t_prime):
        vf = alpha_t_prime/(1 - alpha_t) * (x_1 - x_t)
        return vf


    def sample_conditional_path(
        self,
        g,
        t,
        node_batch_idx,
        edge_batch_idx,
        upper_edge_mask,
        generator: torch.Generator | None = None,
    ):
        """Interpolate between the prior and true terminal state of the ligand."""
        # upper_edge_mask is not used here but it is needed for DirichletVectorField and we need to keep consistent
        # function signatures across vector field classes so that MolFM can use them interchangeably
        src_weights, dst_weights = self.interpolant_scheduler.interpolant_weights(t)

        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == 'e':
                continue

            src_weight, dst_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1), dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']

        e_idx = self.canonical_feat_order.index('e')
        src_weight, dst_weight = src_weights[:, e_idx][edge_batch_idx].unsqueeze(-1), dst_weights[:, e_idx][edge_batch_idx].unsqueeze(-1)
        g.edata[f'e_t'] = src_weight * g.edata[f'e_0'] + dst_weight * g.edata[f'e_1_true']

        return g


class VectorField(EndpointVectorField):

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False):
        
        dst_dict = super().forward(g, t, node_batch_idx, upper_edge_mask, apply_softmax, remove_com)
        dst_dict['x'] = dst_dict['x'] - g.ndata['x_t']
        return dst_dict
    
    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        
        # predict the destination of the trajectory given the current timepoint
        vec_field = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=False,
            remove_com=False
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == "e":
                x_t = g.edata[f'e_t'][upper_edge_mask]
            else:
                x_t = g.ndata[f'{feat}_t']

            # x_s = x_t + vec_field*(s - t)
            x_s = x_t + vec_field[feat]*(s_i - t_i)

            # set x_t = x_s
            if feat == "e":
                x_t = torch.zeros_like(g.edata['e_0'])
                x_t[upper_edge_mask] = x_s
                x_t[~upper_edge_mask] = x_s
                g.edata[f'{feat}_t'] = x_t
            else:
                x_t = x_s
                g.ndata[f'{feat}_t'] = x_t

        # remove COM from x_t
        g.ndata['x_t'] = g.ndata['x_t'] - dgl.readout_nodes(g, feat='x_t', op='mean')[node_batch_idx]

        return g


class DirichletVectorField(EndpointVectorField):

    def __init__(self, *args, w_max=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_max = w_max
        self.categorical_condflows = {}
        self.categorical_condflows['a'] = DirichletConditionalFlow(K=self.n_atom_types, alpha_min=0, alpha_max=w_max+2, alpha_spacing=0.01)
        self.categorical_condflows['c'] = DirichletConditionalFlow(K=self.n_charges, alpha_min=0, alpha_max=w_max+2, alpha_spacing=0.01)
        self.categorical_condflows['e'] = DirichletConditionalFlow(K=self.n_bond_types, alpha_min=0, alpha_max=w_max+2, alpha_spacing=0.01)


        self.n_cat_dict = {
            'a': self.n_atom_types,
            'c': self.n_charges,
            'e': self.n_bond_types
        }

    def alpha_to_w(self, alpha_t):
        return alpha_t*self.w_max + 1

    def sample_conditional_path(
        self,
        g,
        t,
        node_batch_idx,
        edge_batch_idx,
        upper_edge_mask,
        generator: torch.Generator | None = None,
    ):
        """Interpolate between the prior and true terminal state of the ligand."""
        # TODO: this computation could be made more efficient by concatenating node features and edge features into a single tensor and then interpolate them all at once before splitting them back up
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            # skip the bond orders, its too clunky to incorpoarte them into this loop due to upper/lower triangle edge indexing
            if feat == 'e':
                continue

            if feat == 'x':
                src_weights, dst_weights = 1 - alpha_t, alpha_t
                src_weight, dst_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1), dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
                g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']
            else: # now we are doing categorical node features (a,c)
                alpha_expanded = alpha_t[:, feat_idx][node_batch_idx].unsqueeze(-1)
                w_t = self.alpha_to_w(alpha_expanded)
                dirichlet_params = torch.ones_like(g.ndata[f'{feat}_1_true']) + w_t*g.ndata[f'{feat}_1_true']
                g.ndata[f'{feat}_t'] = torch.distributions.Dirichlet(dirichlet_params).sample()

        # sample condiitonal path for edge features
        e_idx = self.canonical_feat_order.index('e')
        alpha_expanded = alpha_t[:, e_idx][edge_batch_idx][upper_edge_mask].unsqueeze(-1)
        w_t = self.alpha_to_w(alpha_expanded)
        dirichlet_params = torch.ones_like(g.edata[f'e_1_true'][upper_edge_mask]) + w_t*(g.edata[f'e_1_true'][upper_edge_mask])
        ue_samples = torch.distributions.Dirichlet(dirichlet_params).sample()
        g.edata['e_t'] = torch.zeros_like(g.edata['e_1_true'])
        g.edata[f'e_t'][upper_edge_mask] = ue_samples
        g.edata[f'e_t'][~upper_edge_mask] = ue_samples

        return g
    
    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        
        # alpha_t_i has shape (n_feats,)
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        g.ndata['x_t'] = x_t + (s_i - t_i)*vf

        # record predicted endoint, for visualization purposes
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # convert alpha values to w
        w_t = self.alpha_to_w(alpha_t_i)
        w_s = self.alpha_to_w(alpha_s_i)


        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat not in ['a', 'c']:
                continue
            w_t_feat = w_t[feat_idx]
            w_s_feat = w_s[feat_idx]
            x_t = g.ndata[f'{feat}_t'] # has shape (n_nodes, n_cat)

            c_factor = self.categorical_condflows[feat].c_factor(
                x_t.cpu().numpy(),
                w_t_feat.item()
            )
            # c_factor has shape equal to x_t, which is (n_nodes, n_cat)
            c_factor = torch.from_numpy(c_factor).to(g.device).float()
            if torch.isnan(c_factor).any():
                # print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                print('NAN c_factor')
                c_factor = torch.nan_to_num(c_factor)

            # get possible endpoints as one-hot vectors
            eps = torch.eye(self.n_cat_dict[feat], device=g.device) # shape (n_cat, n_cat)

            # compute conditional vector fields for each possible endpoint
            cond_vec_fields = (eps[:, None, :] - x_t[None, :, :]) * c_factor.unsqueeze(0) # has shape (n_cat, n_nodes, n_cat)
            endpoint_probs = dst_dict[feat] # has shape (n_nodes, n_cat)
            endpoint_probs = endpoint_probs.transpose(0, 1).unsqueeze(-1) # has shape (n_cat, n_nodes, 1)
            marginal_vec_field = ( endpoint_probs * cond_vec_fields ).sum(dim=0)
            
            # take integration step
            x_s = x_t + marginal_vec_field*(w_s_feat - w_t_feat)

            # project onto simplex if necessary
            x_s = self.project_simplex(x_s)

            # set x_t = x_s
            g.ndata[f'{feat}_t'] = x_s

            # record predicted endoint, for visualization purposes
            g.ndata[f'{feat}_1_pred'] = dst_dict[feat].detach().clone()

        # now we compute marginal vector field and take a step for edge features
        e_idx = self.canonical_feat_order.index('e')
        w_t_e = w_t[e_idx]
        w_s_e = w_s[e_idx]
        x_t = g.edata['e_t'][upper_edge_mask] # has shape (n_edges, n_cat)
        c_factor = self.categorical_condflows['e'].c_factor(
            x_t.cpu().numpy(),
            w_t_e.item()
        )
        c_factor = torch.from_numpy(c_factor).to(g.device).float()
        if torch.isnan(c_factor).any():
            # print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
            print('NAN c_factor')
            c_factor = torch.nan_to_num(c_factor)
        # get possible endpoints as one-hot vectors
        eps = torch.eye(self.n_cat_dict['e'], device=g.device) # shape (n_cat, n_cat)

        # compute conditional vector fields for each possible endpoint
        cond_vec_fields = (eps[:, None, :] - x_t[None, :, :]) * c_factor.unsqueeze(0) # has shape (n_cat, n_edges, n_cat)
        endpoint_probs = dst_dict[feat] # has shape (n_edges, n_cat)
        endpoint_probs = endpoint_probs.transpose(0, 1).unsqueeze(-1) # has shape (n_cat, n_edges, 1)
        marginal_vec_field = ( endpoint_probs * cond_vec_fields ).sum(dim=0)
        x_s = x_t + marginal_vec_field*(w_s_e - w_t_e)
        g.edata['e_t'][upper_edge_mask] = x_s
        g.edata['e_t'][~upper_edge_mask] = x_s

        # record predicted endpoint for bond orders
        e_1_pred = torch.zeros_like(g.edata['e_0'])
        e_1_pred[upper_edge_mask] = dst_dict['e']
        e_1_pred[~upper_edge_mask] = dst_dict['e']
        g.edata['e_1_pred'] = e_1_pred
        
        return g
    
    def project_simplex(self, x_s: torch.Tensor):
        n, c = x_s.shape
        ref_sum = torch.ones(n, dtype=x_s.dtype, device=x_s.device)
        if not torch.allclose(x_s.sum(dim=-1), ref_sum, atol=1e-4) or not (x_s >= 0).all():
            # print(f'WARNING: x_t.min(): {x_s.min()}. Some values of x_s do not lie on the simplex. There are {(x_s<0).sum()} negative values in x_s of shape {x_s.shape} that are negative.')
            x_s = simplex_proj(x_s)
        return x_s

class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)


class BondFirstCoordinateUpdate(nn.Module):
    """Move coordinates with a refreshed, invariant pair representation.

    For every undirected pair ``{i,j}``, a scalar gate is predicted from a
    symmetric function of the two node states, the two directed edge states,
    and distance RBFs.  The resulting displacement is added as ``+delta`` to
    one endpoint and ``-delta`` to the other.  Consequently the layer is
    translation invariant, O(3)-equivariant, and conserves each graph's center
    of mass exactly (up to floating-point summation).
    """

    def __init__(
        self,
        n_node_scalars: int,
        n_edge_feats: int,
        rbf_dim: int,
        coordinate_scale: float = 0.1,
    ):
        super().__init__()
        self.coordinate_scale = float(coordinate_scale)
        input_dim = 2 * n_node_scalars + n_edge_feats + rbf_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, 1),
        )
        # Start from the proven legacy geometry route.  The new causal path is
        # live (and receives gradients) but cannot destabilize the first step.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_scalars: torch.Tensor,
        edge_feats: torch.Tensor,
        positions: torch.Tensor,
        distance_rbf: torch.Tensor,
        upper_edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        if upper_edge_mask.dtype != torch.bool:
            raise TypeError('upper_edge_mask must be boolean')
        if int(upper_edge_mask.sum()) * 2 != g.num_edges():
            raise ValueError(
                'bond-first update requires one aligned reverse edge per upper edge'
            )

        src, dst = g.edges()
        pair_src = src[upper_edge_mask]
        pair_dst = dst[upper_edge_mask]
        upper_edge = edge_feats[upper_edge_mask]
        lower_edge = edge_feats[~upper_edge_mask]
        upper_distance = distance_rbf[upper_edge_mask]
        lower_distance = distance_rbf[~upper_edge_mask]
        if upper_edge.shape != lower_edge.shape:
            raise ValueError('directed edge halves are not aligned')

        pair_inputs = torch.cat(
            (
                node_scalars[pair_src] + node_scalars[pair_dst],
                (node_scalars[pair_src] - node_scalars[pair_dst]).abs(),
                0.5 * (upper_edge + lower_edge),
                0.5 * (upper_distance + lower_distance),
            ),
            dim=-1,
        )
        gate = torch.tanh(self.gate(pair_inputs)) * self.coordinate_scale

        relative = positions[pair_src] - positions[pair_dst]
        unit_relative = relative / torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)

        degree = positions.new_zeros(positions.shape[0])
        ones = positions.new_ones(pair_src.shape[0])
        degree.index_add_(0, pair_src, ones)
        degree.index_add_(0, pair_dst, ones)
        pair_normalizer = (
            degree[pair_src].clamp_min(1.0)
            * degree[pair_dst].clamp_min(1.0)
        ).sqrt().unsqueeze(-1)
        pair_update = gate * unit_relative / pair_normalizer

        displacement = torch.zeros_like(positions)
        displacement.index_add_(0, pair_src, pair_update)
        displacement.index_add_(0, pair_dst, -pair_update)
        return positions + displacement

class EdgeUpdate(nn.Module):

    def __init__(self, n_node_scalars, n_edge_feats, update_edge_w_distance=False, rbf_dim=16):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars*2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )

        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d):
        

        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges()

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]

        if self.update_edge_w_distance:
            mlp_inputs.append(d)

        edge_feats = self.edge_norm(edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1)))
        return edge_feats
