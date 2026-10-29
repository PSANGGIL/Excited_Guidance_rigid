"""Lossless RDKit codec with ordered ring slots and explicit junction mappings.

The hierarchy is a factorization of a complete molecular graph, not a template
library. Ring slots are merged by junction constraints; non-ring atoms are then
attached. Kekule bond orders are the generative representation and RDKit derives
aromaticity on sanitization.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
from rdkit import Chem

_BOND_TO_INT={Chem.BondType.SINGLE:1,Chem.BondType.DOUBLE:2,Chem.BondType.TRIPLE:3,Chem.BondType.AROMATIC:4}
_INT_TO_BOND={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}

@dataclass(frozen=True)
class AtomSpec:
    symbol:str; charge:int; hydrogens:int; radicals:int
@dataclass(frozen=True)
class BondSpec:
    left:int; right:int; order:int
@dataclass(frozen=True)
class RingSpec:
    slots:tuple[int,...]; aromatic:bool; unknown_size:bool=False
@dataclass(frozen=True)
class JunctionSpec:
    left_ring:int; right_ring:int; slot_pairs:tuple[tuple[int,int],...]
@dataclass(frozen=True)
class MoleculeSpec:
    atoms:tuple[AtomSpec,...]
    rings:tuple[RingSpec,...]
    junctions:tuple[JunctionSpec,...]
    perimeter_bonds:tuple[BondSpec,...]
    residual_bonds:tuple[BondSpec,...]
    scaffold_atoms:tuple[int,...]
    substituent_atoms:tuple[int,...]
    canonical_smiles:str
    def to_dict(self): return asdict(self)

class _UF:
    def __init__(self,n): self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[b]=a

def _normalize_cycle(cycle:Iterable[int])->tuple[int,...]:
    c=tuple(map(int,cycle)); n=len(c)
    variants=[]
    for seq in (c,tuple(reversed(c))):
        variants.extend(seq[i:]+seq[:i] for i in range(n))
    return min(variants)

def _ordered_cycle(mol, atoms):
    aset=set(map(int,atoms)); start=min(aset); path=[start]; previous=None
    while len(path)<len(aset):
        candidates=sorted(n.GetIdx() for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
            if n.GetIdx() in aset and n.GetIdx()!=previous and n.GetIdx() not in path)
        if not candidates: raise ValueError(f"ring atoms do not form one simple perimeter: {sorted(aset)}")
        previous,path_last=path[-1],candidates[0]; path.append(path_last)
    if mol.GetBondBetweenAtoms(path[-1],path[0]) is None: raise ValueError("open ring perimeter")
    return _normalize_cycle(path)

class RingSystemCodec:
    def __init__(self,min_ring_size=3,max_ring_size=10):
        self.min_ring_size=int(min_ring_size); self.max_ring_size=int(max_ring_size)
    def encode(self,mol:Chem.Mol)->MoleculeSpec:
        if mol is None: raise ValueError("mol is None")
        canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
        base=Chem.MolFromSmiles(canonical)
        if base is None: raise ValueError("RDKit cannot canonicalize molecule")
        aromatic_atoms=[a.GetIsAromatic() for a in base.GetAtoms()]
        aromatic_edges={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in base.GetBonds() if b.GetIsAromatic()}
        rings=[]
        for raw in Chem.GetSymmSSSR(base):
            slots=_ordered_cycle(base,list(raw)); edges={tuple(sorted((slots[i],slots[(i+1)%len(slots)]))) for i in range(len(slots))}
            rings.append(RingSpec(slots, bool(edges) and edges<=aromatic_edges,
                not self.min_ring_size<=len(slots)<=self.max_ring_size))
        rings=tuple(sorted(rings,key=lambda r:(r.slots,r.aromatic)))
        kek=Chem.Mol(base); Chem.Kekulize(kek,clearAromaticFlags=True)
        atoms=tuple(AtomSpec(a.GetSymbol(),a.GetFormalCharge(),a.GetTotalNumHs(),a.GetNumRadicalElectrons()) for a in base.GetAtoms())
        all_bonds={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):_BOND_TO_INT[b.GetBondType()] for b in kek.GetBonds()}
        perimeter=set()
        for ring in rings:
            perimeter|={tuple(sorted((ring.slots[i],ring.slots[(i+1)%len(ring.slots)]))) for i in range(len(ring.slots))}
        per=tuple(BondSpec(a,b,all_bonds[(a,b)]) for a,b in sorted(perimeter))
        residual=tuple(BondSpec(a,b,o) for (a,b),o in sorted(all_bonds.items()) if (a,b) not in perimeter)
        junctions=[]
        for i,left in enumerate(rings):
            for j in range(i+1,len(rings)):
                right=rings[j]; shared=set(left.slots)&set(right.slots)
                if shared:
                    pairs=tuple(sorted((left.slots.index(a),right.slots.index(a)) for a in shared))
                    junctions.append(JunctionSpec(i,j,pairs))
        scaffold=tuple(sorted(set().union(*(set(r.slots) for r in rings)) if rings else set()))
        substituent=tuple(i for i in range(len(atoms)) if i not in set(scaffold))
        return MoleculeSpec(atoms,rings,tuple(junctions),per,residual,scaffold,substituent,canonical)
    def decode(self,spec:MoleculeSpec)->Chem.Mol:
        rw=Chem.RWMol()
        for item in spec.atoms:
            atom=Chem.Atom(item.symbol); atom.SetFormalCharge(item.charge); atom.SetNumExplicitHs(item.hydrogens)
            atom.SetNumRadicalElectrons(item.radicals); atom.SetNoImplicit(True); rw.AddAtom(atom)
        seen=set()
        for bond in (*spec.perimeter_bonds,*spec.residual_bonds):
            edge=tuple(sorted((bond.left,bond.right)))
            if edge in seen: raise ValueError(f"duplicate bond {edge}")
            seen.add(edge); rw.AddBond(*edge,_INT_TO_BOND[bond.order])
        mol=rw.GetMol(); Chem.SanitizeMol(mol); return mol
    def canonical(self,mol): return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
    def roundtrip_equal(self,mol):
        spec=self.encode(mol); return self.canonical(self.decode(spec))==spec.canonical_smiles
    def validate_junctions(self,spec:MoleculeSpec):
        for j in spec.junctions:
            left,right=spec.rings[j.left_ring],spec.rings[j.right_ring]
            for a,b in j.slot_pairs:
                if left.slots[a]!=right.slots[b]: return False
        return True

    def validate_oracle_assembly(self, spec:MoleculeSpec):
        if not spec.rings: return len(spec.scaffold_atoms)==0
        import torch
        from .assembly import assemble_scaffold
        r=len(spec.rings); s=max(len(x.slots) for x in spec.rings); probability=torch.zeros(r,r,s,s)
        for junction in spec.junctions:
            for left,right in junction.slot_pairs:
                probability[junction.left_ring,junction.right_ring,left,right]=1
                probability[junction.right_ring,junction.left_ring,right,left]=1
        assembled=assemble_scaffold([len(x.slots) for x in spec.rings],[x.aromatic for x in spec.rings],probability)
        expected_edges={tuple(sorted((b.left,b.right))) for b in spec.perimeter_bonds}
        # Convert assembled atom ids back to the source atom ids represented by their slots.
        actual=set()
        for left,right in assembled.perimeter_edges:
            a=spec.rings[assembled.atom_slots[left][0][0]].slots[assembled.atom_slots[left][0][1]]
            b=spec.rings[assembled.atom_slots[right][0][0]].slots[assembled.atom_slots[right][0][1]]
            actual.add(tuple(sorted((a,b))))
        return assembled.valid and len(assembled.atom_slots)==len(spec.scaffold_atoms) and actual==expected_edges
