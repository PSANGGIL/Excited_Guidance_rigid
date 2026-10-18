import math

import torch
from pathlib import Path
import dgl
from torch.nn.functional import one_hot
from molguidance.data_processing.priors import coupled_node_prior, edge_prior
from molguidance.models.ctmc_vector_field import PROPERTY_MAP


def collate(graphs):
    batched_graph = dgl.batch(graphs)
    if hasattr(graphs[0], 'prop'):
        # Stack properties and move to same device as graph
        props = torch.stack([g.prop for g in graphs])
        batched_graph.prop = props.to(batched_graph.device)
    return batched_graph

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super(MoleculeDataset, self).__init__()

        # unpack some configs regarding the prior
        self.prior_config = prior_config
        self.dataset_config = dataset_config
        self.split = split

        # get the condtional configuration
        self.conditioning = dataset_config.get('conditioning', {})
        self.conditioning_enabled = self.conditioning.get('enabled', False)
        self.target_property = self.conditioning.get('property', None)
        self.normalize_property = self.conditioning.get('normalize', True)
        self.use_hydrogen_counts = bool(dataset_config.get('use_hydrogen_counts', False))
        self.use_joint_atom_state = bool(
            dataset_config.get('use_joint_atom_state', False)
        )
        self.n_atom_states = int(dataset_config.get('n_atom_states', 0))
        self.n_bond_types = int(dataset_config.get('n_bond_types', 5))
        self.require_chemistry_auxiliary_labels = bool(
            dataset_config.get('require_chemistry_auxiliary_labels', False)
        )
        self.fake_atom_fraction = float(
            dataset_config.get('fake_atom_fraction', 0.0)
        )
        self.fake_atom_std = float(dataset_config.get('fake_atom_std', 1.0))
        self.fake_atom_validation_seed = int(
            dataset_config.get('fake_atom_validation_seed', 17_041)
        )
        if (
            not math.isfinite(self.fake_atom_fraction)
            or not 0.0 <= self.fake_atom_fraction <= 1.0
        ):
            raise ValueError('fake_atom_fraction must be finite and in [0, 1]')
        if not math.isfinite(self.fake_atom_std) or self.fake_atom_std <= 0.0:
            raise ValueError('fake_atom_std must be finite and positive')
        if self.fake_atom_validation_seed < 0:
            raise ValueError('fake_atom_validation_seed must be non-negative')
        self.use_fake_atoms = self.fake_atom_fraction > 0.0
        if self.use_joint_atom_state and self.n_atom_states <= 0:
            raise ValueError(
                'n_atom_states must be positive when joint atom state is enabled'
            )
        if self.use_joint_atom_state and self.use_hydrogen_counts:
            raise ValueError(
                'joint atom state already contains H count; disable separate H state'
            )
        if self.use_fake_atoms and not self.use_joint_atom_state:
            raise ValueError('fake atoms require the joint atom-state representation')
        if self.use_fake_atoms and self.n_atom_states < 2:
            raise ValueError(
                'fake atoms require at least one real state plus the fake state'
            )
        if self.n_bond_types <= 1:
            raise ValueError(
                'n_bond_types must include NONE and at least one bond'
            )
        if self.require_chemistry_auxiliary_labels and (
            not self.use_joint_atom_state or self.n_bond_types != 4
        ):
            raise ValueError(
                'chemistry auxiliary labels require joint atom states and '
                'four Kekule edge classes'
            )
        self.n_hydrogen_counts = int(dataset_config.get('n_hydrogen_counts', 0))
        if self.use_hydrogen_counts and self.n_hydrogen_counts <= 0:
            raise ValueError('n_hydrogen_counts must be positive when use_hydrogen_counts is enabled')
    
        # get the processed data directory
        processed_data_dir: Path = Path(dataset_config['processed_data_dir'])

        # if the processed data directory does not exist, check it relative to the root of flowmol repository
        if not processed_data_dir.exists():
            processed_data_dir = Path(__file__).parent.parent.parent / processed_data_dir
            if processed_data_dir.exists():
                dataset_config['processed_data_dir'] = str(processed_data_dir)
            else:
                raise FileNotFoundError(f"processed data directory {dataset_config['processed_data_dir']} not found.")
            
        self.processed_data_dir = processed_data_dir

        # load the marginal distributions of atom types and the conditional distribution of charges given atom type
        marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'
        p_a, p_c, p_e, p_c_given_a = torch.load(marginal_dists_file)

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
            p_s = torch.load(
                processed_data_dir / 'train_data_joint_state_marginal.pt'
            )
            self.prior_config['s']['kwargs']['p'] = p_s

        if dataset_config['dataset_name'] in ['geom', 'qm9', 'qme14s']:
            data_file = processed_data_dir / f'{split}_data_processed.pt'
        else:
            raise NotImplementedError('unsupported dataset_name')

        # load data from processed data directory
        data_dict = torch.load(data_file)

        self.positions = data_dict['positions']
        self.atom_types = data_dict['atom_types']
        self.atom_charges = data_dict['atom_charges']
        if self.use_joint_atom_state:
            if 'atom_states' not in data_dict:
                raise ValueError(
                    f'Joint atom state requested, but {data_file} has no atom_states tensor'
                )
            self.atom_states = data_dict['atom_states'].long()
            if self.atom_states.shape[0] != self.atom_types.shape[0]:
                raise ValueError('atom_states and atom_types have different lengths')
            if self.atom_states.numel() and (
                self.atom_states.min() < 0
                or self.atom_states.max() >= self.n_atom_states
            ):
                raise ValueError(
                    'atom_states contains a class outside n_atom_states'
                )
        if self.use_hydrogen_counts:
            if 'atom_hydrogens' not in data_dict:
                raise ValueError(
                    f"Hydrogen-count state requested, but {data_file} has no atom_hydrogens tensor"
                )
            self.atom_hydrogens = data_dict['atom_hydrogens'].long()
            if self.atom_hydrogens.shape[0] != self.atom_types.shape[0]:
                raise ValueError('atom_hydrogens and atom_types have different lengths')
            if self.atom_hydrogens.numel() and (
                self.atom_hydrogens.min() < 0
                or self.atom_hydrogens.max() >= self.n_hydrogen_counts
            ):
                raise ValueError('atom_hydrogens contains a class outside n_hydrogen_counts')
        self.bond_types = data_dict['bond_types']
        self.bond_idxs = data_dict['bond_idxs']
        self.bond_connectivity_weights = data_dict.get(
            'bond_connectivity_weights'
        )
        self.bond_aromatic_flags = data_dict.get('bond_aromatic_flags')
        if self.require_chemistry_auxiliary_labels and (
            self.bond_connectivity_weights is None
            or self.bond_aromatic_flags is None
        ):
            raise ValueError(
                f'Chemistry auxiliary labels requested, but {data_file} lacks '
                'bond_connectivity_weights or bond_aromatic_flags'
            )
        for name, values in (
            ('bond_connectivity_weights', self.bond_connectivity_weights),
            ('bond_aromatic_flags', self.bond_aromatic_flags),
        ):
            if values is not None and values.shape != self.bond_types.shape:
                raise ValueError(
                    f'{name} must align one-to-one with stored molecular bonds'
                )
        self.node_idx_array = data_dict['node_idx_array']
        self.edge_idx_array = data_dict['edge_idx_array']
        self.dataset = dataset_config['dataset_name']

        # Load and set up conditioning if enabled
        if self.conditioning_enabled and self.target_property:
            if 'properties' not in data_dict:
                raise ValueError("Properties not found in processed data")

            if self.dataset == "qm9":
                if self.target_property not in PROPERTY_MAP:
                    raise ValueError(f"Target property {self.target_property} not found in property map")
                self.property_idx = PROPERTY_MAP[self.target_property]

            self.properties = data_dict['properties']
            
            # Load normalization parameters if normalize is True
            if self.normalize_property:
                norm_params_file = processed_data_dir / 'train_data_property_normalization.pt'
                self.norm_params = torch.load(norm_params_file)
                

    def __len__(self):
        return self.node_idx_array.shape[0]

    def _fake_atom_generator(self, idx: int) -> torch.Generator | None:
        """Use worker RNG for training and a molecule-local RNG for evaluation."""
        if self.split in {'train', 'train_a', 'train_b'}:
            return None
        split_offset = {'val': 1_000_000, 'test': 2_000_000}.get(
            self.split, 3_000_000
        )
        generator = torch.Generator(device='cpu')
        generator.manual_seed(
            self.fake_atom_validation_seed + split_offset + int(idx)
        )
        return generator

    def _augment_with_fake_atoms(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        atom_charges: torch.Tensor,
        atom_states: torch.Tensor,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append FlowMol3-style anchored fake nodes before graph construction.

        The last joint atom-state class is reserved for the synthetic fake
        state. Fake nodes have no target bonds because the complete adjacency
        is built only after they have been appended.
        """
        n_real = int(positions.shape[0])
        if n_real <= 0:
            raise ValueError('fake-atom augmentation requires a non-empty molecule')
        generator = self._fake_atom_generator(idx)
        max_fake = int(math.ceil(n_real * self.fake_atom_fraction))
        n_fake = int(
            torch.randint(
                low=0,
                high=max_fake + 1,
                size=(1,),
                generator=generator,
            ).item()
        )
        is_real = torch.ones(n_real + n_fake, dtype=torch.bool)
        if n_fake == 0:
            return positions, atom_types, atom_charges, atom_states, is_real

        anchor_indices = torch.randint(
            low=0,
            high=n_real,
            size=(n_fake,),
            generator=generator,
        )
        offsets = torch.randn(
            (n_fake, positions.shape[-1]),
            generator=generator,
            dtype=positions.dtype,
        ) * self.fake_atom_std
        fake_positions = positions[anchor_indices] + offsets
        fake_atom_types = atom_types.new_zeros(
            (n_fake, atom_types.shape[-1])
        )
        fake_charges = atom_charges.new_zeros(n_fake)
        fake_states = atom_states.new_full(
            (n_fake,), self.n_atom_states - 1
        )
        is_real[n_real:] = False
        return (
            torch.cat((positions, fake_positions), dim=0),
            torch.cat((atom_types, fake_atom_types), dim=0),
            torch.cat((atom_charges, fake_charges), dim=0),
            torch.cat((atom_states, fake_states), dim=0),
            is_real,
        )
    
    def __getitem__(self, idx):
        node_start_idx = self.node_idx_array[idx, 0]
        node_end_idx = self.node_idx_array[idx, 1]
        edge_start_idx = self.edge_idx_array[idx, 0]
        edge_end_idx = self.edge_idx_array[idx, 1]
        
        # get data pertaining to nodes for this molecule
        positions = self.positions[node_start_idx:node_end_idx]
        atom_types = self.atom_types[node_start_idx:node_end_idx].float()
        atom_charges = self.atom_charges[node_start_idx:node_end_idx].long()
        if self.use_joint_atom_state:
            atom_states = self.atom_states[node_start_idx:node_end_idx]
        if self.use_hydrogen_counts:
            atom_hydrogens = self.atom_hydrogens[node_start_idx:node_end_idx]

        if self.use_fake_atoms:
            (
                positions,
                atom_types,
                atom_charges,
                atom_states,
                is_real,
            ) = self._augment_with_fake_atoms(
                positions, atom_types, atom_charges, atom_states, idx
            )

        # Remove the COM after fake-node augmentation so the entire state is
        # centered, matching the centered coordinate prior.
        positions = positions - positions.mean(dim=0, keepdim=True)

        # get data pertaining to edges for this molecule
        bond_types = self.bond_types[edge_start_idx:edge_end_idx].int()
        bond_idxs = self.bond_idxs[edge_start_idx:edge_end_idx].long()
        if self.bond_connectivity_weights is not None:
            bond_connectivity_weights = self.bond_connectivity_weights[
                edge_start_idx:edge_end_idx
            ].float()
        if self.bond_aromatic_flags is not None:
            bond_aromatic_flags = self.bond_aromatic_flags[
                edge_start_idx:edge_end_idx
            ].bool()

        # reconstruct adjacency matrix
        n_atoms = positions.shape[0]
        adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int32)

        # fill in the values of the adjacency matrix specified by bond_idxs
        adj[bond_idxs[:,0], bond_idxs[:,1]] = bond_types

        # get upper triangle of adjacency matrix
        upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1) # has shape (2, n_upper_edges)
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]

        # get lower triangle edges by swapping source and destination of upper_edge_idxs
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

        # one-hot encode edge labels and atom charges
        edge_labels = one_hot(
            edge_labels.to(torch.int64), num_classes=self.n_bond_types
        ).float()
        if not self.use_joint_atom_state:
            try:
                atom_charges = one_hot(atom_charges + 2, num_classes=6).float() # hard-coded legacy assumption that charges are in range [-2, 3]
            except Exception as e:
                print('an atom charge outside of the expected range was encountered')
                print(f'max atom charge: {atom_charges.max()}, min atom charge: {atom_charges.min()}')
                raise e

        # create a dgl graph
        g = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)

        # add edge features
        g.edata['e_1_true'] = edge_labels
        if self.bond_connectivity_weights is not None:
            connectivity_matrix = torch.zeros(
                (n_atoms, n_atoms), dtype=torch.float32
            )
            connectivity_matrix[bond_idxs[:, 0], bond_idxs[:, 1]] = (
                bond_connectivity_weights
            )
            upper_connectivity = connectivity_matrix[
                upper_edge_idxs[0], upper_edge_idxs[1]
            ]
            g.edata['e_connectivity_weight'] = torch.cat(
                (upper_connectivity, upper_connectivity)
            )
        if self.bond_aromatic_flags is not None:
            aromatic_matrix = torch.zeros(
                (n_atoms, n_atoms), dtype=torch.bool
            )
            aromatic_matrix[bond_idxs[:, 0], bond_idxs[:, 1]] = (
                bond_aromatic_flags
            )
            upper_aromatic = aromatic_matrix[
                upper_edge_idxs[0], upper_edge_idxs[1]
            ]
            g.edata['e_aromatic_true'] = torch.cat(
                (upper_aromatic, upper_aromatic)
            )

        # add node features
        g.ndata['x_1_true'] = positions
        if self.use_joint_atom_state:
            g.ndata['s_1_true'] = one_hot(
                atom_states, num_classes=self.n_atom_states
            ).float()
            if self.use_fake_atoms:
                g.ndata['is_real_1_true'] = is_real
        else:
            g.ndata['a_1_true'] = atom_types
            g.ndata['c_1_true'] = atom_charges
        if self.use_hydrogen_counts and not self.use_joint_atom_state:
            g.ndata['h_1_true'] = one_hot(
                atom_hydrogens, num_classes=self.n_hydrogen_counts
            ).float()

        # Add conditioning property if enabled
        if self.conditioning_enabled and self.target_property:
            if self.dataset == "qm9":
                property_value = self.properties[idx, self.property_idx].float()
            elif self.dataset == "qme14s":
                property_value = self.properties[idx].float()

            if self.normalize_property:
                if self.dataset == "qm9":
                    mean = self.norm_params['mean'][self.property_idx]
                    std = self.norm_params['std'][self.property_idx]
                elif self.dataset == "qme14s":
                    mean = self.norm_params['mean']
                    std = self.norm_params['std']
                property_value = (property_value - mean) / std
            g.prop = property_value.to(g.device)


        # sample prior for node features, coupled to the destination features
        if self.use_joint_atom_state:
            dst_dict = {'x': positions, 's': g.ndata['s_1_true']}
        else:
            dst_dict = {
                'x': positions,
                'a': atom_types,
                'c': atom_charges
            }
            if self.use_hydrogen_counts:
                dst_dict['h'] = g.ndata['h_1_true']
        prior_node_feats = coupled_node_prior(dst_dict=dst_dict, prior_config=self.prior_config)
        for feat in prior_node_feats:
            g.ndata[f'{feat}_0'] = prior_node_feats[feat]

        # sample the prior for the edge features    
        upper_edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        n_upper_edges = upper_edge_idxs.shape[1]
        upper_edge_mask[:n_upper_edges] = True
        g.edata['e_0'] = edge_prior(
            upper_edge_mask,
            self.prior_config['e'],
            self.n_bond_types,
        )

        return g
