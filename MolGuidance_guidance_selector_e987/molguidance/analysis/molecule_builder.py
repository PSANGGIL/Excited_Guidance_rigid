import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
import dgl
from typing import List, Dict
from molguidance.data_processing.priors import rigid_alignment
from molguidance.data_processing.utils import get_upper_edge_mask
from torch.nn.functional import one_hot

bond_type_map = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC, None] # last bond type is for masked bonds

bond_type_to_idx = { bond_type:idx for idx, bond_type in enumerate(bond_type_map)}
bond_type_to_idx[None] = 0


def fake_atom_state_index(atom_state_vocab: List[dict] | None) -> int | None:
    """Return the unique synthetic fake-state index, if the codec defines one."""
    if atom_state_vocab is None:
        return None
    indices = [
        idx for idx, state in enumerate(atom_state_vocab)
        if bool(state.get('is_fake', False))
    ]
    if len(indices) > 1:
        raise ValueError('atom_state_vocab contains more than one fake state')
    return None if not indices else indices[0]


def remove_fake_atoms_from_graph(
    g: dgl.DGLGraph,
    atom_state_vocab: List[dict] | None,
) -> dgl.DGLGraph:
    """Return the endpoint-induced real-atom subgraph with indices relabelled."""
    fake_index = fake_atom_state_index(atom_state_vocab)
    if fake_index is None:
        return g
    state_feature = 's_1' if 's_1' in g.ndata else 's_1_true'
    if state_feature not in g.ndata:
        raise ValueError('fake-state decoding requires s_1 or s_1_true')
    state_indices = g.ndata[state_feature].argmax(dim=-1)
    keep = state_indices != fake_index
    if bool(keep.all()):
        return g
    return dgl.node_subgraph(g, keep, store_ids=False)


class SampledMolecule:

    def __init__(self, g: dgl.DGLGraph, 
        atom_type_map: List[str], 
        traj_frames: Dict[str, torch.Tensor] = None,
        ctmc_mol: bool = False, # whether the molecule was sampled from a CTMC model. Important because one-hot encodings will contain a mask token.
        residual_mask_policy: str = "reject",
        exclude_charges: bool = False, 
        align_traj: bool = True,
        build_xt_traj=True,
        atom_state_vocab: List[dict] | None = None,
        build_ep_traj=True,):
        """Represents a molecule sampled from a model. Converts the DGL graph to an rdkit molecule and keeps all associated information."""

        atom_type_map = list(atom_type_map) # create a shallow copy of the atom type map so that we don't modify the original

        self.exclude_charges = exclude_charges
        self.align_traj = align_traj
        self.ctmc_mol = ctmc_mol
        self.atom_state_vocab = atom_state_vocab
        if residual_mask_policy not in {"reject", "allow"}:
            raise ValueError("residual_mask_policy must be 'reject' or 'allow'")
        self.residual_mask_policy = residual_mask_policy

        if ctmc_mol and residual_mask_policy == "reject":
            atom_type_map.append('__MASK_ATOM__')
        
        # Keep the full diffusion state for trajectories, but expose the
        # molecule graph after fake-node removal to all chemistry metrics.
        self.slot_graph = g
        self.num_slots = g.num_nodes()
        self.g = remove_fake_atoms_from_graph(g, self.atom_state_vocab)

        self.positions, self.atom_types, self.atom_charges, self.bond_types, self.bond_src_idxs, self.bond_dst_idxs = extract_moldata_from_graph(
            self.g,
            atom_type_map,
            exclude_charges=exclude_charges,
            ctmc_mol=self.ctmc_mol,
            atom_state_vocab=self.atom_state_vocab)
        self.atom_hydrogens = extract_hydrogen_counts_from_graph(
            self.g, ctmc_mol=self.ctmc_mol,
            atom_state_vocab=self.atom_state_vocab,
        )

        self.atom_type_map = atom_type_map
        self.num_atoms = self.g.num_nodes()
        self.num_atom_types = len(atom_type_map)

        # build rdkit molecule
        self.rdkit_mol = self.build_molecule()

        # compute valencies on every atom
        self.valencies = self.compute_valencies()

        # build trajectory molecules
        self.traj_frames = traj_frames
        if traj_frames is not None:

            if build_xt_traj:
                self.traj_mols = self.process_traj_frames(traj_frames) # convert frames into a list of rdkit molecules

            # construct molecules for endpoint trajectory
            if 'x_1_pred' in traj_frames and build_ep_traj:
                self.ep_traj_mols = self.process_traj_frames(traj_frames, ep_traj=True)

    @classmethod
    def from_rdkit_mol(cls, mol: Chem.Mol, atom_type_map: List[str] = None):
        """Creates a SampledMolecule from an rdkit molecule."""

        if atom_type_map is None:
            atom_types = [ atom.GetSymbol() for atom in mol.GetAtoms() ]
            atom_type_map = list(set(atom_types))

        atom_type_to_idx = {atom_type: i for i, atom_type in enumerate(atom_type_map)}

        # get positions
        positions = mol.GetConformer().GetPositions()
        positions = torch.from_numpy(positions)

        # get atom elements as a string
        # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        atom_types_idx = torch.zeros(mol.GetNumAtoms()).long()
        atom_charges = torch.zeros_like(atom_types_idx)
        atom_hydrogens = torch.zeros_like(atom_types_idx)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_types_idx[i] = atom_type_to_idx[atom.GetSymbol()]
            atom_charges[i] = atom.GetFormalCharge()
            atom_hydrogens[i] = atom.GetTotalNumHs(includeNeighbors=True)

        # get one-hot encoded of existing bonds only (no non-existing bonds)
        adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
        edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

        bond_types = adj[edge_index[:, 0], edge_index[:, 1]]
        bond_src_idxs = edge_index[:, 0]
        bond_dst_idxs = edge_index[:, 1]
        bond_types[bond_types == 1.5] = 4
        edge_attr = bond_types.type(torch.int32)

        # create graph
        g = dgl.graph((bond_src_idxs, bond_dst_idxs), num_nodes=mol.GetNumAtoms())

        # add data to the graph
        g.ndata['x_1'] = positions
        g.ndata['a_1'] = one_hot(atom_types_idx.long(), num_classes=len(atom_type_map)).float()
        g.ndata['c_1'] = one_hot(atom_charges.long() + 2, num_classes=6).float()
        n_hydrogen_classes = max(4, int(atom_hydrogens.max()) + 1)
        g.ndata['h_1'] = one_hot(
            atom_hydrogens.long(), num_classes=n_hydrogen_classes
        ).float()
        g.edata['e_1'] = one_hot(edge_attr.long(), num_classes=5).float()
        g.edata['ue_mask'] = torch.ones(g.num_edges()).bool()

        return cls(g, atom_type_map=atom_type_map)
    
    # this code is adapted from MiDi: https://github.com/cvignac/MiDi/blob/ba07fc5b1313855c047ba0b90e7aceae47e34e38/midi/analysis/rdkit_functions.py
    def build_molecule(self):
        mol = build_molecule(
            self.positions,
            self.atom_types,
            self.atom_charges,
            self.bond_src_idxs,
            self.bond_dst_idxs,
            self.bond_types,
            atom_hydrogens=self.atom_hydrogens,
        )
        return mol
    
    def compute_valencies(self):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""
        adj = torch.zeros((self.num_atoms, self.num_atoms))
        adjusted_bond_types = self.bond_types.clone()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adjusted_bond_types = adjusted_bond_types.float()
        adj[self.bond_src_idxs, self.bond_dst_idxs] = adjusted_bond_types
        adj[self.bond_dst_idxs, self.bond_src_idxs] = adjusted_bond_types
        valencies = torch.sum(adj, dim=-1).long()
        return valencies
    
    def process_traj_frames(self, traj_frames: Dict[str, torch.Tensor], ep_traj: bool = False):
        """Converts the trajectory frames to a list of rdkit molecules."""
        # convert the frames to a list of rdkit molecules
        g_dummy = copy_graph(self.slot_graph)

        if ep_traj:
            n_frames = traj_frames['x_1_pred'].shape[0]
        else:
            n_frames = traj_frames['x'].shape[0]

        traj_mols = []
        for frame_idx in range(n_frames):

            # put current frame data into graph
            for feat in traj_frames.keys():

                if '1_pred' in feat:
                    continue

                if feat == 'e':
                    data_src = g_dummy.edata
                else:
                    data_src = g_dummy.ndata

                if ep_traj:
                    traj_key = f'{feat}_1_pred'
                else:
                    traj_key = feat

                data_src[f'{feat}_1'] = traj_frames[traj_key][frame_idx].clone()

            frame_graph = remove_fake_atoms_from_graph(
                g_dummy, self.atom_state_vocab
            )

            # extract mol data from the current real-atom induced subgraph
            positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs = extract_moldata_from_graph(
                frame_graph,
                self.atom_type_map,
                ctmc_mol=self.ctmc_mol,
                atom_state_vocab=self.atom_state_vocab)

            atom_hydrogens = extract_hydrogen_counts_from_graph(
                frame_graph, ctmc_mol=self.ctmc_mol,
                atom_state_vocab=self.atom_state_vocab,
            )

            # align positions to final frame
            if (
                self.align_traj
                and positions.shape == self.positions.shape
                and positions.shape[0] > 0
            ):
                positions = rigid_alignment(positions, self.positions)

            # build rdkit molecule
            mol = build_molecule(
                positions,
                atom_types,
                atom_charges,
                bond_src_idxs,
                bond_dst_idxs,
                bond_types,
                atom_hydrogens=atom_hydrogens,
            )

            # add mol to list
            traj_mols.append(mol)

        traj_mols = [mol for mol in traj_mols if mol is not None]

        if len(traj_mols) < n_frames:
            print(f'WARNING: {n_frames - len(traj_mols)} frames were not converted to rdkit molecules')

        return traj_mols
    

def extract_moldata_from_graph(
    g: dgl.DGLGraph,
    atom_type_map: List[str],
    exclude_charges: bool = False,
    ctmc_mol: bool = False,
    atom_state_vocab: List[dict] | None = None,
):

    g = remove_fake_atoms_from_graph(g, atom_state_vocab)

    # extract node-level features
    positions = g.ndata['x_1']

    if 's_1' in g.ndata:
        if atom_state_vocab is None:
            raise ValueError('s_1 is present but atom_state_vocab was not provided')
        state_indices = g.ndata['s_1'].argmax(dim=1).tolist()
        atom_types = []
        atom_charges = []
        for state_idx in state_indices:
            if state_idx >= len(atom_state_vocab):
                atom_types.append('__MASK_ATOM__')
                atom_charges.append(0)
            else:
                state = atom_state_vocab[state_idx]
                atom_types.append(state['symbol'])
                atom_charges.append(int(state['formal_charge']))
        atom_charges = torch.tensor(atom_charges, dtype=torch.long)
    else:
        atom_types = g.ndata['a_1'].argmax(dim=1)
        atom_types = [atom_type_map[int(atom)] for atom in atom_types]
        if exclude_charges:
            atom_charges = None
        else:
            atom_charges = g.ndata['c_1'].argmax(dim=1) - 2


    # get bond types and atom indicies for every edge, convert types from simplex to integer
    bond_types = g.edata['e_1'].argmax(dim=1)
    if ctmc_mol:
        mask_index = g.edata['e_1'].shape[-1] - 1
        bond_types[bond_types == mask_index] = 0
    bond_src_idxs, bond_dst_idxs = g.edges()

    # get just the upper triangle of the adjacency matrix
    upper_edge_mask = g.edata['ue_mask']
    bond_types = bond_types[upper_edge_mask]
    bond_src_idxs = bond_src_idxs[upper_edge_mask]
    bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

    # get only non-zero bond types
    bond_mask = bond_types != 0
    bond_types = bond_types[bond_mask]
    bond_src_idxs = bond_src_idxs[bond_mask]
    bond_dst_idxs = bond_dst_idxs[bond_mask]


    return positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs



def extract_hydrogen_counts_from_graph(
    g: dgl.DGLGraph,
    ctmc_mol: bool = False,
    atom_state_vocab: List[dict] | None = None,
):
    g = remove_fake_atoms_from_graph(g, atom_state_vocab)
    # A residual CTMC mask is represented as -1 and rejected by build_molecule.
    if 's_1' in g.ndata:
        if atom_state_vocab is None:
            raise ValueError('s_1 is present but atom_state_vocab was not provided')
        state_indices = g.ndata['s_1'].argmax(dim=1).tolist()
        return torch.tensor(
            [
                -1
                if state_idx >= len(atom_state_vocab)
                else int(atom_state_vocab[state_idx]['hydrogens'])
                for state_idx in state_indices
            ],
            dtype=torch.long,
        )
    if 'h_1' not in g.ndata:
        return None
    hydrogen_state = g.ndata['h_1']
    hydrogen_counts = hydrogen_state.argmax(dim=1).long()
    if ctmc_mol:
        mask_index = hydrogen_state.shape[-1] - 1
        hydrogen_counts[hydrogen_counts == mask_index] = -1
    return hydrogen_counts


def build_molecule(positions, atom_types, atom_charges, bond_src_idxs, bond_dst_idxs, bond_types, atom_hydrogens=None):
    """Builds a rdkit molecule from the given atom and bond information."""
    if len(atom_types) == 0:
        return None
    # Residual CTMC masks are invalid states, not selenium or a charge class.
    if any(atom_type == '__MASK_ATOM__' for atom_type in atom_types):
        return None
    if atom_charges is None:
        atom_charges = [0] * len(atom_types)
    if atom_hydrogens is not None:
        if len(atom_hydrogens) != len(atom_types):
            raise ValueError('atom_hydrogens and atom_types have different lengths')
        if any(int(h) < 0 for h in atom_hydrogens):
            return None
    else:
        atom_hydrogens = [None] * len(atom_types)

    # create a rdkit molecule and add atoms to it
    mol = Chem.RWMol()
    for atom_type, charge, hydrogen_count in zip(
        atom_types, atom_charges, atom_hydrogens
    ):
        a = Chem.Atom(atom_type)
        if charge != 0:
            a.SetFormalCharge(int(charge))
        if hydrogen_count is not None:
            a.SetNumExplicitHs(int(hydrogen_count))
            a.SetNoImplicit(True)
        mol.AddAtom(a)

    # add bonds to rdkit molecule
    for bond_type, src_idx, dst_idx in zip(bond_types, bond_src_idxs, bond_dst_idxs):
        src_idx = int(src_idx)
        dst_idx = int(dst_idx)
        mol.AddBond(src_idx, dst_idx, bond_type_map[bond_type])

    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        return None

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = positions[i]
        x, y, z = float(x), float(y), float(z)
        conf.SetAtomPosition(i, Point3D(x,y,z))
    mol.AddConformer(conf)

    return mol


def copy_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    
    # get edges
    edges = g.edges(form='uv')

    # get number of nodes
    num_nodes = g.num_nodes()

    g_copy = dgl.graph(edges, num_nodes=num_nodes, device=g.device)

    # transfer over node features
    for nfeat in g.ndata.keys():
        g_copy.ndata[nfeat] = g.ndata[nfeat].detach().clone()

    # transfer over edge features
    for efeat in g.edata.keys():
        g_copy.edata[efeat] = g.edata[efeat].detach().clone()


    return g_copy

def dataset_mol_to_sampled_mol(
    g, atom_type_map, atom_state_vocab=None
) -> SampledMolecule:
    if 's_1_true' in g.ndata:
        features = ['x', 's', 'e']
    else:
        features = ['x', 'a', 'c', 'e']
        if 'h_1_true' in g.ndata:
            features.insert(-1, 'h')
    for feat in features:
        if feat == 'e':
            data_src = g.edata
        else:
            data_src = g.ndata
        data_src[f'{feat}_1'] = data_src[f'{feat}_1_true']

    g.edata['ue_mask'] = get_upper_edge_mask(g)
    return SampledMolecule(
        g, atom_type_map, atom_state_vocab=atom_state_vocab
    )

def dataset_mol_to_rdmol(g, atom_type_map, atom_state_vocab=None):
    return dataset_mol_to_sampled_mol(
        g, atom_type_map, atom_state_vocab=atom_state_vocab
    ).rdkit_mol
