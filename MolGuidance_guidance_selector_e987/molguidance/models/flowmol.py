from typing import Dict, List, Sequence
import json
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import torch.nn.functional as fn
from torch.distributions import Exponential
from pathlib import Path

from molguidance.models.lr_scheduler import LRScheduler
from molguidance.models.interpolant_scheduler import InterpolantScheduler
from molguidance.models.vector_field import EndpointVectorField, VectorField, DirichletVectorField
from molguidance.models.ctmc_vector_field import CTMCVectorField

from molguidance.data_processing.utils import build_edge_idxs, get_upper_edge_mask, get_batch_idxs
from molguidance.data_processing.priors import uniform_simplex_prior, biased_simplex_prior, batched_rigid_alignment, rigid_alignment
from molguidance.data_processing.priors import inference_prior_register, edge_prior
from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.analysis.metrics import SampleAnalyzer
from molguidance.data_processing.conditional_count import ConditionalAtomCountSampler
from einops import rearrange
from .property_embeddings import PropertyEmbedder
from .loss_utils import reduce_time_scaled_loss_equivalent
from .chemistry_auxiliary import JointChemistryAuxiliaryLoss


class FlowMol(pl.LightningModule):

    canonical_feat_order = ['x', 'a', 'c', 'e']
    node_feats = ['x', 'a', 'c']
    edge_feats = ['e']

    def __init__(self,
                 atom_type_map: List[str],               
                 n_atoms_hist_file: str,
                 marginal_dists_file: str,                 
                 n_atom_charges: int = 6,
                 n_bond_types: int = 5,
                 n_hydrogen_counts: int = 0,
                 n_atom_states: int = 0,
                 joint_atom_state_metadata_file: str | None = None,
                 sample_interval: float = 1.0, # how often to sample molecules from the model, measured in epochs
                 n_mols_to_sample: int = 64, # how many molecules to sample from the model during each sample/eval step during training
                 time_scaled_loss: bool = True,
                 exclude_charges: bool = False,
                 weight_ae: bool = False, # whether or not to apply weights to the atom and edge losses (infrequent categories given more weight)
                 target_blur: float = 0.0, # how much to blur the target distribution for categorical features
                 parameterization: str = 'ctmc', # how to parameterize the flow-matching objective, can be 'endpoint', 'vector-field', 'dirichlet' or 'ctmc'
                 total_loss_weights: Dict[str, float] = {}, 
                 lr_scheduler_config: dict = {},
                 interpolant_scheduler_config: dict = {},
                 vector_field_config: dict = {},
                 prior_config: dict = {},
                 default_n_timesteps: int = 250,
                 property_embedding_dim: int = 256,
                 conditional_generation: bool = True,
                 dataset_name: str = 'qm9',
                 gaussian_expansion: bool = False,
                 gaussian_start: float|None = None,
                 gaussian_stop: float|None = None,
                 n_gaussians: int = 5,
                 properties_handle_method: str = 'concatenate_sum', # choices ['concatenate', 'sum', 'multiply', 'concatenate_sum', 'concatenate_multiply']
                 validation_config: dict | None = None,
                 atom_count_conditioning: dict | None = None,
                 chemistry_auxiliary: dict | None = None,
                 self_conditioning_rate: float = 0.0,
                 ):
        super().__init__()

        # Keep optional feature sets local to each model instance. Mutating the
        # class lists leaks modalities into subsequently constructed models.
        self.canonical_feat_order = list(type(self).canonical_feat_order)
        self.node_feats = list(type(self).node_feats)
        self.edge_feats = list(type(self).edge_feats)

        self.lr_scheduler_config = lr_scheduler_config
        self.atom_type_map = atom_type_map
        self.n_atom_types = len(atom_type_map)
        self.n_atom_charges = n_atom_charges
        self.n_bond_types = n_bond_types
        self.n_hydrogen_counts = int(n_hydrogen_counts)
        if self.n_hydrogen_counts < 0:
            raise ValueError('n_hydrogen_counts must be non-negative')
        self.n_atom_states = int(n_atom_states)
        if self.n_atom_states < 0:
            raise ValueError('n_atom_states must be non-negative')
        self.use_joint_atom_state = self.n_atom_states > 0
        if self.use_joint_atom_state and self.n_hydrogen_counts > 0:
            raise ValueError(
                'joint atom state contains charge and H count; '
                'n_hydrogen_counts must be zero'
            )
        if self.use_joint_atom_state and exclude_charges:
            raise ValueError('exclude_charges is not meaningful for joint atom states')
        self.joint_atom_state_metadata_file = joint_atom_state_metadata_file
        self.atom_state_vocab = None
        self.category_support_masks = {}
        self.use_hydrogen_counts = self.n_hydrogen_counts > 0
        self.total_loss_weights = dict(total_loss_weights)
        self.time_scaled_loss = time_scaled_loss
        self.prior_config = prior_config
        self.exclude_charges = exclude_charges
        self.marginal_dists_file = marginal_dists_file
        self.parameterization = parameterization
        self.weight_ae = weight_ae
        self.target_blur = target_blur
        self.n_atoms_hist_file = n_atoms_hist_file
        self.default_n_timesteps = default_n_timesteps
        self.conditional_generation = conditional_generation
        self.properties_handle_method = properties_handle_method
        self.property_embedding_dim = property_embedding_dim
        self.dataset_name = dataset_name
        self.self_conditioning_rate = float(self_conditioning_rate)
        if (
            not torch.isfinite(torch.tensor(self.self_conditioning_rate))
            or not 0.0 <= self.self_conditioning_rate <= 1.0
        ):
            raise ValueError("self_conditioning_rate must be finite and in [0, 1]")

        chemistry_defaults = {
            'edge_existence_weight': 0.0,
            'valence_weight': 0.0,
            'charge_weight': 0.0,
            'connectivity_weight': 0.0,
            'aromatic_weight': 0.0,
        }
        chemistry_config = dict(chemistry_auxiliary or {})
        unknown_chemistry_keys = set(chemistry_config) - set(chemistry_defaults)
        if unknown_chemistry_keys:
            raise ValueError(
                'unknown chemistry auxiliary options: '
                + ', '.join(sorted(unknown_chemistry_keys))
            )
        chemistry_defaults.update(chemistry_config)
        for key, value in chemistry_defaults.items():
            value = float(value)
            if not torch.isfinite(torch.tensor(value)) or value < 0.0:
                raise ValueError(f'{key} must be finite and non-negative')
            chemistry_defaults[key] = value
        self.chemistry_auxiliary_config = chemistry_defaults
        self.chemistry_loss_weights = {
            'chem_edge_existence': chemistry_defaults['edge_existence_weight'],
            'chem_valence': chemistry_defaults['valence_weight'],
            'chem_charge': chemistry_defaults['charge_weight'],
            'chem_connectivity': chemistry_defaults['connectivity_weight'],
            'chem_aromatic': chemistry_defaults['aromatic_weight'],
        }
        self.chemistry_auxiliary_enabled = any(
            value > 0.0 for value in self.chemistry_loss_weights.values()
        )
        self.chemistry_auxiliary_loss = None
        if self.chemistry_auxiliary_enabled:
            if not self.use_joint_atom_state:
                raise ValueError(
                    'chemistry auxiliary losses require joint atom states'
                )
            if parameterization != 'ctmc' or n_bond_types != 4:
                raise ValueError(
                    'chemistry auxiliary losses require CTMC Kekule edge states '
                    'NONE/SINGLE/DOUBLE/TRIPLE'
                )

        validation_defaults = {
            'time_grid': [0.05, 0.25, 0.5, 0.75, 0.95],
            'seed': 17_041,
            'condition_shuffle': True,
        }
        validation_defaults.update(validation_config or {})
        self.validation_time_grid = tuple(
            float(value) for value in validation_defaults['time_grid']
        )
        if not self.validation_time_grid:
            raise ValueError('validation time_grid must contain at least one value')
        if any(not 0.0 < value < 1.0 for value in self.validation_time_grid):
            raise ValueError('validation time_grid values must be strictly between 0 and 1')
        self.validation_seed = int(validation_defaults['seed'])
        self.validation_condition_shuffle = bool(
            validation_defaults['condition_shuffle']
        )

        atom_count_defaults = {
            'mode': 'marginal',
            'bandwidth': 0.15,
            'nearest_k': 128,
            'reference_file': None,
        }
        atom_count_defaults.update(atom_count_conditioning or {})
        if atom_count_defaults['mode'] not in {'marginal', 'conditional_kernel'}:
            raise ValueError(
                "atom-count mode must be 'marginal' or 'conditional_kernel'"
            )
        self.atom_count_conditioning = atom_count_defaults

        if self.use_joint_atom_state:
            if parameterization != 'ctmc':
                raise NotImplementedError(
                    'joint atom state is currently supported only by CTMC'
                )
            self.canonical_feat_order = ['x', 's', 'e']
            self.node_feats = ['x', 's']
        elif self.use_hydrogen_counts:
            if parameterization != 'ctmc':
                raise NotImplementedError(
                    'hydrogen-count state is currently supported only by the CTMC parameterization'
                )
            self.canonical_feat_order.insert(-1, 'h')
            self.node_feats.append('h')
        self.categorical_feats = [
            feat for feat in self.canonical_feat_order if feat != 'x'
        ]

        # for conditional generation of molecules with a property
        self.property_embedder = nn.Sequential(
            nn.Linear(1, property_embedding_dim),
            nn.SiLU(),
            nn.Linear(property_embedding_dim, property_embedding_dim),
            nn.LayerNorm(property_embedding_dim)
        )
        
        # remember set normalization to false in GaussianExpansion, otherwise start and stop values are wrong
        if gaussian_expansion:
            self.property_embedder = PropertyEmbedder(input_dim=1, embedding_dim=property_embedding_dim,
                                                         start=gaussian_start, stop=gaussian_stop,
                                                         n_gaussians=n_gaussians, use_activation=True)

        if self.weight_ae and parameterization == 'vector-field':
            raise NotImplementedError('weighting the atom and edge losses is not yet implemented for the vector-field parameterization')
        
        if self.target_blur != 0.0 and parameterization == 'vector-field':
            raise NotImplementedError('blurring the target distribution is not yet implemented for the vector-field parameterization')
        
        if self.target_blur < 0.0:
            raise ValueError('target_blur must be non-negative')
        
        # if provided filepath to data dir does not exist, assume it is relative to the repo root
        processed_data_dir = Path(self.marginal_dists_file).parent
        if not processed_data_dir.exists():
            repo_root = Path(__file__).parent.parent.parent
            self.marginal_dists_file = repo_root / self.marginal_dists_file
            self.n_atoms_hist_file = repo_root / self.n_atoms_hist_file

        self.configure_joint_atom_state()
        self.configure_chemistry_auxiliary()

        # do some boring stuff regarding the prior distribution
        self.configure_prior()

        if self.exclude_charges:
            self.node_feats.remove('c')
            self.canonical_feat_order.remove('c')
            self.total_loss_weights.pop('c', None)
            self.categorical_feats.remove('c')

        # create a dictionary mapping feature -> number of categories
        if self.use_joint_atom_state:
            self.n_cat_dict = {'s': self.n_atom_states, 'e': n_bond_types}
        else:
            self.n_cat_dict = {
                'a': self.n_atom_types,
                'c': n_atom_charges,
                'e': n_bond_types,
            }
            if self.use_hydrogen_counts:
                self.n_cat_dict['h'] = self.n_hydrogen_counts

        for feat in self.canonical_feat_order:
            if feat not in total_loss_weights:
                self.total_loss_weights[feat] = 1.0

                # print warning if the user has not specified a loss weight for a feature
                print(f'WARNING: no loss weight specified for feature {feat}, using default of 1.0')

        self.exp_dist = Exponential(1.0)
        
        # construct histogram of number of atoms in each ligand
        self.build_n_atoms_dist(n_atoms_hist_file=self.n_atoms_hist_file)
        self.build_conditional_n_atoms_sampler()

        # create interpolant scheduler and vector field
        self.interpolant_scheduler = InterpolantScheduler(canonical_feat_order=self.canonical_feat_order, 
                                                          **interpolant_scheduler_config)
        
        # check that a valid parameterization was specified
        if self.parameterization not in ['endpoint', 'vector-field', 'dirichlet', 'ctmc']:
            raise ValueError(f'parameterization must be one of "endpoint", "vector-field", or "dirichlet", "ctmc", got {self.parameterization}')

        if self.parameterization == 'endpoint':
            vector_field_class = EndpointVectorField
        elif self.parameterization == 'vector-field':
            vector_field_class = VectorField
        elif self.parameterization == 'dirichlet':
            vector_field_class = DirichletVectorField
        elif self.parameterization == 'ctmc':
            vector_field_class = CTMCVectorField

        if self.parameterization == 'ctmc':
            self.vector_field = vector_field_class(n_atom_types=self.n_atom_types,
                                            canonical_feat_order=self.canonical_feat_order,
                                            interpolant_scheduler=self.interpolant_scheduler, 
                                            n_atom_states=self.n_atom_states,
                                            category_support_masks=self.category_support_masks,
                                            n_charges=n_atom_charges, 
                                            n_bond_types=n_bond_types,
                                            n_hydrogen_counts=self.n_hydrogen_counts,
                                            exclude_charges=self.exclude_charges,
                                            property_embedding_dim=property_embedding_dim,
                                            property_embedder=self.property_embedder,
                                            properties_handle_method=self.properties_handle_method,
                                            conditional_generation=self.conditional_generation,
                                            dataset_name=dataset_name,
                                            **vector_field_config)
        else:
            self.vector_field = vector_field_class(n_atom_types=self.n_atom_types,
                                            canonical_feat_order=self.canonical_feat_order,
                                            interpolant_scheduler=self.interpolant_scheduler, 
                                            n_atom_states=self.n_atom_states,
                                            category_support_masks=self.category_support_masks,
                                            n_charges=n_atom_charges, 
                                            n_bond_types=n_bond_types,
                                            n_hydrogen_counts=self.n_hydrogen_counts,
                                            exclude_charges=self.exclude_charges,
                                            **vector_field_config)

        if self.self_conditioning_rate > 0.0 and not self.vector_field.self_conditioning:
            raise ValueError(
                "self_conditioning_rate is positive but vector_field.self_conditioning "
                "is disabled"
            )
        if self.vector_field.self_conditioning:
            if self.parameterization != "ctmc":
                raise NotImplementedError(
                    "self-conditioning is currently implemented for CTMC only"
                )
            if self.self_conditioning_rate <= 0.0:
                raise ValueError(
                    "enabled vector-field self-conditioning requires a positive "
                    "self_conditioning_rate"
                )

        # remove charge loss function if necessary
        if self.exclude_charges:
            self.loss_fn_dict.pop('c')

        self.sample_interval = sample_interval # how often to sample molecules from the model, measured in epochs
        self.n_mols_to_sample = n_mols_to_sample # how many molecules to sample from the model during each sample/eval step during training
        self.last_sample_marker = 0 # this is the epoch_exact value of the last time we sampled molecules from the model
        self.sample_analyzer = SampleAnalyzer()


        # record the last epoch value for training steps -  this is really hacky but it lets me
        # align the validation losses with the correspoding training epoch value on W&B
        self.last_epoch_exact = 0

        self.save_hyperparameters()

    def configure_joint_atom_state(self):
        if not self.use_joint_atom_state:
            return

        metadata_path = self.joint_atom_state_metadata_file
        if metadata_path is None:
            metadata_path = Path(self.marginal_dists_file).parent / (
                'joint_atom_state_metadata.json'
            )
        else:
            metadata_path = Path(metadata_path)
            if not metadata_path.is_absolute():
                data_relative = Path(self.marginal_dists_file).parent / metadata_path
                if data_relative.exists():
                    metadata_path = data_relative
                else:
                    metadata_path = Path(__file__).parent.parent.parent / metadata_path
        if not metadata_path.exists():
            raise FileNotFoundError(
                f'joint atom-state metadata not found: {metadata_path}'
            )

        metadata = json.loads(metadata_path.read_text())
        states = metadata.get('states')
        support = metadata.get('train_supported')
        if not isinstance(states, list) or len(states) != self.n_atom_states:
            raise ValueError(
                'joint atom-state vocabulary length does not match n_atom_states'
            )
        if not isinstance(support, list) or len(support) != self.n_atom_states:
            raise ValueError(
                'joint atom-state train_supported mask has invalid length'
            )
        required_keys = {'symbol', 'formal_charge', 'hydrogens'}
        for idx, state in enumerate(states):
            if not isinstance(state, dict) or not required_keys <= state.keys():
                raise ValueError(f'invalid atom-state entry at index {idx}')
            if state['symbol'] not in self.atom_type_map:
                raise ValueError(
                    f"atom-state symbol {state['symbol']!r} is absent from atom_type_map"
                )

        self.atom_state_vocab = states
        self.category_support_masks = {'s': [bool(value) for value in support]}
        self.joint_atom_state_metadata_file = str(metadata_path.resolve())

    def configure_chemistry_auxiliary(self):
        if not self.chemistry_auxiliary_enabled:
            return
        if self.atom_state_vocab is None:
            raise RuntimeError("joint atom-state metadata was not configured")

        support = torch.tensor(
            self.category_support_masks["s"], dtype=torch.bool
        )
        charges = torch.tensor(
            [state["formal_charge"] for state in self.atom_state_vocab],
            dtype=torch.float32,
        )
        parsed_counts = []
        max_valence = 0
        for state_idx, state in enumerate(self.atom_state_vocab):
            raw_counts = state.get("train_heavy_valence_counts")
            if not isinstance(raw_counts, dict):
                raise ValueError(
                    "chemistry auxiliary losses require metadata version 2 "
                    "with train_heavy_valence_counts"
                )
            counts = {}
            for raw_valence, raw_count in raw_counts.items():
                try:
                    valence = int(raw_valence)
                    count = int(raw_count)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid valence count for atom state {state_idx}"
                    ) from exc
                if valence < 0 or count <= 0 or str(valence) != str(raw_valence):
                    raise ValueError(
                        f"invalid valence count for atom state {state_idx}"
                    )
                counts[valence] = count
                max_valence = max(max_valence, valence)
            if bool(support[state_idx]) and not counts:
                raise ValueError(
                    f"train-supported atom state {state_idx} has no valence support"
                )
            parsed_counts.append(counts)

        valence_probabilities = torch.zeros(
            (self.n_atom_states, max_valence + 1), dtype=torch.float32
        )
        for state_idx, counts in enumerate(parsed_counts):
            for valence, count in counts.items():
                valence_probabilities[state_idx, valence] = float(count)
            row_sum = valence_probabilities[state_idx].sum()
            if row_sum > 0:
                valence_probabilities[state_idx] /= row_sum

        self.chemistry_auxiliary_loss = JointChemistryAuxiliaryLoss(
            state_charges=charges,
            state_valence_probabilities=valence_probabilities,
            state_support=support,
        )

    def configure_prior(self):
        # load the marginal distributions of atom types, bond orders and the conditional distribution of charges given atom type
        p_a, p_c, p_e, p_c_given_a = torch.load(self.marginal_dists_file)
        self.p_a = p_a
        self.p_e = p_e
        self.p_s = None
        if self.use_joint_atom_state:
            state_marginal_file = Path(self.marginal_dists_file).with_name(
                'train_data_joint_state_marginal.pt'
            )
            self.p_s = torch.load(state_marginal_file)
            if self.p_s.shape != (self.n_atom_states,):
                raise ValueError('joint atom-state marginal has invalid shape')

        # add the marginal distributions as arguments to the prior sampling functions
        if 'a' in self.prior_config and self.prior_config['a']['type'] == 'marginal':
            self.prior_config['a']['kwargs']['p'] = p_a

        if self.prior_config['e']['type'] == 'marginal':
            self.prior_config['e']['kwargs']['p'] = p_e

        if 'c' in self.prior_config and self.prior_config['c']['type'] == 'marginal':
            self.prior_config['c']['kwargs']['p'] = p_c
        
        if 'c' in self.prior_config and self.prior_config['c']['type'] == 'c-given-a':
            self.prior_config['c']['kwargs']['p_c_given_a'] = p_c_given_a

        if 's' in self.prior_config and self.prior_config['s']['type'] == 'marginal':
            self.prior_config['s']['kwargs']['p'] = self.p_s

        if self.parameterization == 'dirichlet':
            for feat in self.categorical_feats:
                if self.prior_config[feat]['type'] != 'uniform-simplex':
                    raise ValueError('dirichlet parameterization requires that all categorical priors be uniform-simplex')

        if self.parameterization == 'ctmc':
            for feat in self.categorical_feats:
                if self.prior_config[feat]['type'] != 'ctmc':
                    raise ValueError('ctmc parameterization requires that all categorical priors be ctmc')
                
    def configure_loss_fns(self, device):    
        # instantiate loss functions
        if self.time_scaled_loss:
            reduction = 'none'
        else:
            reduction = 'mean'

        if self.parameterization in  ['endpoint', 'dirichlet', 'ctmc']:
            categorical_loss_fn = nn.CrossEntropyLoss
        elif self.parameterization == 'vector-field':
            categorical_loss_fn = nn.MSELoss


        if self.weight_ae:
            node_kwargs = {
                'weight': (
                    1 - (self.p_s if self.use_joint_atom_state else self.p_a)
                ).to(device)
            }
            e_kwargs = {'weight': (1 - self.p_e).to(device)}
        else:
            node_kwargs = {}
            e_kwargs = {}

        if self.parameterization == 'ctmc':
            cat_kwargs = {'ignore_index': -100}
        else:
            cat_kwargs = {}

        self.loss_fn_dict = {'x': nn.MSELoss(reduction=reduction)}
        if self.use_joint_atom_state:
            self.loss_fn_dict['s'] = categorical_loss_fn(
                reduction=reduction, **node_kwargs, **cat_kwargs
            )
        else:
            self.loss_fn_dict['a'] = categorical_loss_fn(
                reduction=reduction, **node_kwargs, **cat_kwargs
            )
            self.loss_fn_dict['c'] = categorical_loss_fn(
                reduction=reduction, **cat_kwargs
            )
            if self.use_hydrogen_counts:
                self.loss_fn_dict['h'] = categorical_loss_fn(
                    reduction=reduction, **cat_kwargs
                )
        self.loss_fn_dict['e'] = categorical_loss_fn(
            reduction=reduction, **e_kwargs, **cat_kwargs
        )

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):

        # check if self has the attribute batches_per_epoch
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)

        # compute epoch as a float
        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.last_epoch_exact = epoch_exact

        # update the learning rate
        self.lr_scheduler.step_lr(epoch_exact)

        # # sample and evaluate molecules if necessary
        # if epoch_exact - self.last_sample_marker >= self.sample_interval:
        #     self.last_sample_marker = epoch_exact
        #     self.eval()
        #     with torch.no_grad():
        #         property_targets = 0.3 # this is a dummy value, just for visualization purposes, won't affect loss computation
        #         sampled_molecules = self.sample_random_sizes(n_molecules=self.n_mols_to_sample, device=g.device,
        #                             properties_for_sampling=property_targets, 
        #                             properties_handle_method=self.properties_handle_method)
        #     self.train()
        #     sampled_mols_metrics = self.sample_analyzer.analyze(sampled_molecules, energy_div=False, functional_validity=True)
        #     self.log_dict(sampled_mols_metrics)

        # compute losses
        losses = self(g)

        # create a dictionary of values to log
        train_log_dict = {}
        train_log_dict['epoch_exact'] = epoch_exact

        for key in losses:
            train_log_dict[f'{key}_train_loss'] = losses[key]

        total_loss = self._total_from_feature_losses(losses)

        self.log_dict(train_log_dict, sync_dist=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, sync_dist=True)

        return total_loss

    @staticmethod
    def _seeded_generator(device: torch.device, seed: int) -> torch.Generator:
        generator_device = device if device.type == 'cuda' else torch.device('cpu')
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))
        return generator

    def _total_from_feature_losses(self, losses: Dict[str, torch.Tensor]):
        total = torch.zeros((), device=next(iter(losses.values())).device)
        for feat in self.canonical_feat_order:
            total = total + self.total_loss_weights[feat] * losses[feat]
        for loss_key, weight in self.chemistry_loss_weights.items():
            if weight <= 0.0:
                continue
            if loss_key not in losses:
                raise KeyError(f'enabled auxiliary loss {loss_key!r} is missing')
            total = total + weight * losses[loss_key]
        return total

    def compute_chemistry_auxiliary_losses(
        self,
        g: dgl.DGLGraph,
        vf_output: Dict[str, torch.Tensor],
        upper_edge_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.chemistry_auxiliary_enabled:
            return {}
        if self.chemistry_auxiliary_loss is None:
            raise RuntimeError('chemistry auxiliary module is not configured')
        graph_weights = vf_output['s'].new_ones(g.batch_size)
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
            state_idx = self.canonical_feat_order.index('s')
            edge_idx = self.canonical_feat_order.index('e')
            graph_weights = 0.5 * (
                time_weights[:, state_idx] + time_weights[:, edge_idx]
            ).to(vf_output['s'])
        return self.chemistry_auxiliary_loss(
            g,
            node_logits=vf_output['s'],
            edge_logits=vf_output['e'],
            upper_edge_mask=upper_edge_mask,
            graph_weights=graph_weights,
        )

    def _use_self_conditioning(
        self,
        vector_field: nn.Module,
        requested: bool | None,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> bool:
        """Choose one graph-level self-conditioning decision for this call."""
        if not getattr(vector_field, "self_conditioning", False):
            return False
        if requested is not None:
            return bool(requested)
        if not self.training:
            return True
        draw = torch.rand((), device=device, generator=generator)
        return bool(draw < self.self_conditioning_rate)

    def predict_endpoint(
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor,
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        *,
        corruption_generator: torch.Generator | None = None,
        self_conditioning: bool | None = None,
        vector_field: nn.Module | None = None,
        **vector_field_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run one endpoint pass or a detached first pass plus residual pass.

        Both passes consume exactly the same noisy graph, time, condition and
        condition-dropout choice. Only the second pass contributes gradients.
        """
        field = self.vector_field if vector_field is None else vector_field
        use_self_conditioning = self._use_self_conditioning(
            field, self_conditioning, corruption_generator, g.device
        )
        previous_endpoint = None
        if use_self_conditioning:
            with torch.no_grad():
                first_endpoint = field(
                    g,
                    t,
                    node_batch_idx=node_batch_idx,
                    upper_edge_mask=upper_edge_mask,
                    previous_endpoint=None,
                    **vector_field_kwargs,
                )
            previous_endpoint = field.prepare_self_conditioning_endpoint(
                first_endpoint,
                categorical_are_logits=not bool(
                    vector_field_kwargs.get("apply_softmax", False)
                ),
            )
        return field(
            g,
            t,
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            previous_endpoint=previous_endpoint,
            **vector_field_kwargs,
        )

    def deterministic_validation_losses(self, g: dgl.DGLGraph, batch_idx: int):
        """Evaluate a fixed time/noise grid with conditioning always enabled."""

        losses_by_time = []
        for grid_idx, time_value in enumerate(self.validation_time_grid):
            seed = self.validation_seed + batch_idx * len(self.validation_time_grid) + grid_idx
            generator = self._seeded_generator(g.device, seed)
            time = torch.full(
                (g.batch_size,), time_value, device=g.device, dtype=torch.float32
            )
            losses_by_time.append(
                self(
                    g,
                    t=time,
                    corruption_generator=generator,
                    condition_dropout=False,
                    self_conditioning=True,
                )
            )

        loss_keys = tuple(losses_by_time[0])
        if any(tuple(losses) != loss_keys for losses in losses_by_time[1:]):
            raise RuntimeError('validation loss keys changed across the time grid')
        mean_losses = {
            key: torch.stack([losses[key] for losses in losses_by_time]).mean()
            for key in loss_keys
        }
        diagnostics = {}

        if (
            self.validation_condition_shuffle
            and self.conditional_generation
            and hasattr(g, 'prop')
            and g.prop is not None
            and g.batch_size > 1
        ):
            midpoint_idx = min(
                range(len(self.validation_time_grid)),
                key=lambda idx: abs(self.validation_time_grid[idx] - 0.5),
            )
            time_value = self.validation_time_grid[midpoint_idx]
            seed = (
                self.validation_seed
                + batch_idx * len(self.validation_time_grid)
                + midpoint_idx
            )
            original_property = g.prop
            try:
                g.prop = original_property.roll(shifts=1, dims=0)
                shuffled_losses = self(
                    g,
                    t=torch.full(
                        (g.batch_size,), time_value, device=g.device,
                        dtype=torch.float32,
                    ),
                    corruption_generator=self._seeded_generator(g.device, seed),
                    condition_dropout=False,
                    self_conditioning=True,
                )
            finally:
                g.prop = original_property
            diagnostics['condition_shuffle_gain'] = (
                self._total_from_feature_losses(shuffled_losses)
                - self._total_from_feature_losses(losses_by_time[midpoint_idx])
            )

        return mean_losses, diagnostics

    def validation_step(self, g: dgl.DGLGraph, batch_idx: int):
        losses, diagnostics = self.deterministic_validation_losses(g, batch_idx)

        # create dictionary of values to log
        val_log_dict = {
            'epoch_exact': self.last_epoch_exact
        }

        for key in losses:
            val_log_dict[f'{key}_val_loss'] = losses[key]

        val_log_dict.update({f'val_{key}': value for key, value in diagnostics.items()})

        self.log_dict(val_log_dict, batch_size=g.batch_size, sync_dist=True)

        # combine individual losses into a total loss
        total_loss = self._total_from_feature_losses(losses)

        self.log('val_total_loss', total_loss, prog_bar=True, batch_size=g.batch_size, on_step=True, sync_dist=True)

        return total_loss
    
    def forward(
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor | None = None,
        corruption_generator: torch.Generator | None = None,
        condition_dropout: bool | None = None,
        self_conditioning: bool | None = None,
    ):
        
        batch_size = g.batch_size
        device = g.device

        # check if the attribute loss_fn_dict exists
        # it is necessary to do this here (as opposed to in __init__) beacause
        # to instantiate the loss function class-conditioned weights, the weight
        # tensors need to be on the same device as the graph...seems pretty dumb but that's how it is
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)

        # get batch indicies of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # get initial COM of each molecule and remove the COM from the atom positions
        # this step is now done in MoleculeDataset.__getitem__ method, a necessary adjustment to do OT alignment on
        # the prior during the __getitem__ method
        # init_coms = dgl.readout_nodes(g, feat='x_1_true', op='mean')
        # g.ndata['x_1_true'] = g.ndata['x_1_true'] - init_coms[node_batch_idx]

        # sample molecules from prior
        # we used to sample the prior in the forward pass at training time,
        # but now at training time we sample the prior in the __getitem__ method of MoleculeDataset
        # this is so that we can compute OT alignments in parallel (since they cannot be done in batch)
        # g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # sample timepoints for each molecule in the batch
        if t is None:
            t = torch.rand(
                batch_size, device=device, generator=corruption_generator
            ).float()
        else:
            t = torch.as_tensor(t, device=device, dtype=torch.float32)
            if t.ndim == 0:
                t = t.expand(batch_size)
            if t.shape != (batch_size,):
                raise ValueError(f't must have shape ({batch_size},), got {tuple(t.shape)}')

        # Get property embeddings if available
        if hasattr(g, 'prop') and self.conditional_generation:
            self.property_embedder = self.property_embedder.to(device) 
            g.prop = g.prop.unsqueeze(-1) if g.prop.dim() == 1 else g.prop
            g.prop = g.prop.to(device)
            prop_emb = self.property_embedder(g.prop)
            # Concatenate with time
            t_combined = torch.cat([t.unsqueeze(-1), prop_emb], dim=-1)
            # print("prop_emb\n", prop_emb.shape)
        else:
            t_combined = t
            print("no feature is used\n")

        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(
            g,
            t,
            node_batch_idx,
            edge_batch_idx,
            upper_edge_mask,
            generator=corruption_generator,
        )

        # forward pass for the vector field
        vf_output = self.predict_endpoint(
            g,
            t_combined,
            node_batch_idx,
            upper_edge_mask,
            corruption_generator=corruption_generator,
            self_conditioning=self_conditioning,
        )

        # get the target (label) for each feature
        targets = {}
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            # compute the target for endpoint parameterization
            if self.parameterization in ['endpoint', 'dirichlet', 'ctmc']:
                target = data_src[f'{feat}_1_true']
                if feat == "e":
                    target = target[upper_edge_mask]
                if feat in self.n_cat_dict:
                    if self.target_blur == 0.0:
                        target = target.argmax(dim=-1)
                    else:
                        target = target + torch.randn(
                            target.shape,
                            device=target.device,
                            dtype=target.dtype,
                            generator=corruption_generator,
                        ) * self.target_blur
                        target = fn.softmax(target, dim=-1)
            #  compute the target for vector-field parameterization
            elif self.parameterization == 'vector-field':
                alpha_t_prime_i = alpha_t_prime[:, feat_idx]
                x_1 = data_src[f'{feat}_1_true']
                x_0 = data_src[f'{feat}_0']

                if feat == 'e':
                    alpha_t_prime_i = alpha_t_prime_i[edge_batch_idx][upper_edge_mask].unsqueeze(-1)
                    x_1 = x_1[upper_edge_mask]
                    x_0 = x_0[upper_edge_mask]
                else:
                    alpha_t_prime_i = alpha_t_prime_i[node_batch_idx].unsqueeze(-1)

                target = alpha_t_prime_i*(x_1 - x_0)

            # for CTMC parameterization, we do not apply loss on already unmasked features
            if self.parameterization == 'ctmc' and feat in self.n_cat_dict:
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

                support = self.vector_field.get_category_support_mask(feat)
                if support is not None and target.ndim == 1:
                    unsupported_target = ~support[target.clamp_min(0)]
                    # Held-out-only lossless states remain decodable but are
                    # outside the train-supported generative vocabulary.
                    target[unsupported_target & (target != -100)] = -100

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
            
        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0

            # compute the losses
            target = targets[feat]
            losses[feat] = self.loss_fn_dict[feat](vf_output[feat], target)

            # Preserve the released reduction exactly: coordinates use row-wise weights,
            # while 1-D CE uses its factorized broadcast-equivalent.
            if self.time_scaled_loss:
                losses[feat] = reduce_time_scaled_loss_equivalent(
                    losses[feat], weight
                )

        losses.update(
            self.compute_chemistry_auxiliary_losses(
                g, vf_output, upper_edge_mask, t
            )
        )
        return losses
    
    def sample_prior(self, g, node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        """Sample from the prior distribution of the ligand."""
        # sample atom positions from prior
        # TODO: we should set the standard deviation of atom position prior to be like the average distance to the COM in the training set
        # or perhaps the average distance to COM for molecules with the same number of atoms
        num_nodes = g.num_nodes()
        device = g.device

        
        # sample the prior for node features
        for feat in self.node_feats:
            prior_type = self.prior_config[feat]['type']
            prior_fn = inference_prior_register[prior_type]
            # I tried to design consistent interface for prior functions, but it's not perfect
            # hence the need for the following two if statements
            if feat == 'x':
                args = [g, node_batch_idx,]
            else:
                args = [num_nodes, self.n_cat_dict[feat],]

            if feat == 'c' and self.prior_config[feat]['type'] == 'c-given-a':
                args.append(g.ndata['a_0'])

            kwargs = self.prior_config[feat]['kwargs']
            g.ndata[f'{feat}_0'] = prior_fn(*args, **kwargs).to(device)

        # sample the prior for edge features
        g.edata['e_0'] = edge_prior(
            upper_edge_mask,
            self.prior_config['e'],
            self.n_bond_types,
        ).to(device)
            
        return g

    def configure_optimizers(self):
        try:
            weight_decay = self.lr_scheduler_config['weight_decay']
        except KeyError:
            weight_decay = 0

        optimizer = optim.Adam(self.parameters(), lr=self.lr_scheduler_config['base_lr'], weight_decay=weight_decay)
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return optimizer

    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        """Builds the distribution of the number of atoms in a ligand."""
        n_atoms, n_atom_counts = torch.load(n_atoms_hist_file)
        n_atoms_prob = n_atom_counts / n_atom_counts.sum()
        self.n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
        self.n_atoms_map = n_atoms

    def build_conditional_n_atoms_sampler(self):
        reference_file = self.atom_count_conditioning.get('reference_file')
        if reference_file is None:
            reference_file = Path(self.n_atoms_hist_file).with_name(
                'train_data_processed.pt'
            )
        else:
            reference_file = Path(reference_file)
            if not reference_file.is_absolute():
                reference_file = Path(self.n_atoms_hist_file).parent / reference_file

        self.conditional_n_atoms_sampler = None
        self.atom_count_reference_file = str(reference_file)
        if reference_file.exists():
            self.conditional_n_atoms_sampler = (
                ConditionalAtomCountSampler.from_processed_file(reference_file)
            )
        elif self.atom_count_conditioning['mode'] == 'conditional_kernel':
            raise FileNotFoundError(
                f'conditional atom-count reference not found: {reference_file}'
            )

    def _prepare_count_targets(self, properties, n_molecules: int) -> torch.Tensor:
        if self.conditional_n_atoms_sampler is None:
            raise RuntimeError('conditional atom-count sampler is not available')
        condition_dim = self.conditional_n_atoms_sampler.condition_dim
        targets = torch.as_tensor(properties, dtype=torch.float32, device='cpu')
        if targets.ndim == 0:
            targets = targets.reshape(1, 1).expand(n_molecules, condition_dim)
        elif targets.ndim == 1:
            if condition_dim == 1 and targets.numel() == n_molecules:
                targets = targets.unsqueeze(-1)
            elif targets.numel() == condition_dim:
                targets = targets.reshape(1, condition_dim).expand(n_molecules, -1)
            else:
                raise ValueError(
                    '1-D count targets must contain one scalar per molecule or '
                    'one complete condition vector'
                )
        elif targets.ndim == 2:
            if targets.shape != (n_molecules, condition_dim):
                raise ValueError(
                    f'count targets must have shape ({n_molecules}, {condition_dim}), '
                    f'got {tuple(targets.shape)}'
                )
        else:
            raise ValueError('count targets must be scalar, 1-D, or 2-D')
        return targets

    def sample_n_atoms(
        self,
        n_molecules: int,
        properties=None,
        mode: str | None = None,
        bandwidth: float | Sequence[float] | None = None,
        nearest_k: int | None = None,
        generator: torch.Generator | None = None,
    ):
        """Draw heavy-atom counts from ``p(n)`` or empirical ``p(n | c)``."""

        mode = self.atom_count_conditioning['mode'] if mode in (None, 'model_default') else mode
        if mode == 'marginal':
            indices = torch.multinomial(
                self.n_atoms_dist.probs,
                num_samples=n_molecules,
                replacement=True,
                generator=generator,
            )
            self.last_atom_count_diagnostics = {'mode': 'marginal'}
            return self.n_atoms_map[indices]
        if mode != 'conditional_kernel':
            raise ValueError(f'unknown atom-count sampling mode: {mode}')
        if properties is None:
            raise ValueError(
                'conditional_kernel atom-count sampling requires property targets'
            )
        targets = self._prepare_count_targets(properties, n_molecules)
        bandwidth = (
            self.atom_count_conditioning['bandwidth']
            if bandwidth is None else bandwidth
        )
        nearest_k = (
            self.atom_count_conditioning.get('nearest_k')
            if nearest_k is None else nearest_k
        )
        samples, diagnostics = self.conditional_n_atoms_sampler.sample(
            targets,
            bandwidth,
            nearest_k=nearest_k,
            generator=generator,
            return_diagnostics=True,
        )
        self.last_atom_count_diagnostics = {
            'mode': mode,
            'effective_sample_size_mean': float(
                diagnostics.effective_sample_size.mean()
            ),
            'nearest_distance_max': float(diagnostics.nearest_distance.max()),
        }
        return samples

    def resolve_atom_counts(
        self,
        n_molecules: int,
        device,
        number_of_atoms=None,
        properties_for_sampling=None,
        multiple_properties=None,
        atom_count_mode: str | None = None,
        atom_count_bandwidth: float | Sequence[float] | None = None,
        atom_count_nearest_k: int | None = None,
        atom_count_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if number_of_atoms is not None:
            counts = torch.as_tensor(number_of_atoms, dtype=torch.long)
            if counts.ndim != 1 or counts.numel() != n_molecules:
                raise ValueError(f'number_of_atoms must contain {n_molecules} entries')
            if (counts <= 0).any():
                raise ValueError('number_of_atoms entries must be positive')
            return counts.to(device)

        targets = multiple_properties
        if targets is None:
            targets = properties_for_sampling
        return self.sample_n_atoms(
            n_molecules,
            properties=targets,
            mode=atom_count_mode,
            bandwidth=atom_count_bandwidth,
            nearest_k=atom_count_nearest_k,
            generator=atom_count_generator,
        ).to(device)

    def sample_random_sizes(self, n_molecules: int, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, 
        xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, # properties name is for finding normalizing vector for the property
        properties_for_sampling:int|float=None, 
        training_mode:bool=True,   
        properties_handle_method:str=None, # choices ['concatenate', 'sum', 'multiply', 'concatenate_sum', 'concatenate_multiply']
        multilple_values_to_one_property: List[float|int] | None = None,     
        number_of_atoms: List[int]|None = None, 
        atom_count_mode: str | None = None,
        atom_count_bandwidth: float | Sequence[float] | None = None,
        atom_count_nearest_k: int | None = None,
        atom_count_generator: torch.Generator | None = None,
        **kwargs):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        if multilple_values_to_one_property is not None and properties_for_sampling is not None:
            raise ValueError('You can not provide both multilple_values_to_one_property and properties_for_sampling, only one of them should be provided')
        
        atoms_per_molecule = self.resolve_atom_counts(
            n_molecules,
            device,
            number_of_atoms=number_of_atoms,
            properties_for_sampling=properties_for_sampling,
            multiple_properties=multilple_values_to_one_property,
            atom_count_mode=atom_count_mode,
            atom_count_bandwidth=atom_count_bandwidth,
            atom_count_nearest_k=atom_count_nearest_k,
            atom_count_generator=atom_count_generator,
        )

        if multilple_values_to_one_property is not None:
            assert len(atoms_per_molecule) == len(multilple_values_to_one_property), \
                f"{len(atoms_per_molecule)} != {len(multilple_values_to_one_property)}"

        return self.sample(atoms_per_molecule, 
            device=device,  
            stochasticity=stochasticity, 
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            normalization_file_path=normalization_file_path,
            conditional_generation=conditional_generation,
            property_name=property_name, # just for sampling process with normalizing 
            properties_for_sampling=properties_for_sampling,
            training_mode=training_mode,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=multilple_values_to_one_property,
            **kwargs)
    

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, properties_for_sampling:int|float=None, 
        training_mode:bool=True,   
        properties_handle_method:str='concatenate_sum', # choices ['concatenate', 'sum', 'multiply', 'concatenate_sum', 'concatenate_multiply']
        multilple_values_to_one_property: List[float|int] | None = None, 
         **kwargs):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        batch_size = n_atoms.shape[0]

        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)
            

        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # integrate trajectories
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'normalization_file_path': normalization_file_path,
            'conditional_generation': conditional_generation,
            'property_name': property_name,
            'properties_for_sampling': properties_for_sampling,
            'training_mode': training_mode,
            'properties_handle_method': properties_handle_method,
            'multilple_values_to_one_property': multilple_values_to_one_property
        }
        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold

        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        if self.parameterization == 'ctmc':
            ctmc_mol = True
        else:
            ctmc_mol = False


        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args, 
                ctmc_mol=ctmc_mol, 
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                atom_state_vocab=self.atom_state_vocab,
                exclude_charges=self.exclude_charges))

        return molecules
