"""Optional RDKit conversion kept outside the differentiable training path."""

from __future__ import annotations

from .decoder import AtomGraph


def to_rdkit(atom_graph: AtomGraph, sanitize: bool = True):
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'chem' extra to use RDKit export") from exc

    editable = Chem.RWMol()
    aromatic_atoms = {
        atom_index
        for bond in atom_graph.bonds
        if bond.kind == "AROMATIC"
        for atom_index in (bond.left, bond.right)
    }
    for atom in atom_graph.atoms:
        rd_atom = Chem.Atom(atom.symbol)
        rd_atom.SetFormalCharge(atom.formal_charge)
        rd_atom.SetIsAromatic(atom.index in aromatic_atoms)
        editable.AddAtom(rd_atom)
    bond_map = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }
    for bond in atom_graph.bonds:
        editable.AddBond(bond.left, bond.right, bond_map[bond.kind])
        if bond.kind == "AROMATIC":
            editable.GetBondBetweenAtoms(bond.left, bond.right).SetIsAromatic(True)
    molecule = editable.GetMol()
    if sanitize:
        Chem.SanitizeMol(molecule)
    return molecule


def canonical_smiles(atom_graph: AtomGraph) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(to_rdkit(atom_graph, sanitize=True), canonical=True)
