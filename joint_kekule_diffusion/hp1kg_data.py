"""HP1KG graph/NEVPT2 adapter with cached deterministic RDKit conformers."""
from __future__ import annotations
import csv,json,random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem,rdDepictor
ATOM_MAP=("B","C","N","O","F","Si","P","S","Cl","As","Se","Br")

def _coordinates(mol,seed):
 hmol=Chem.AddHs(Chem.Mol(mol));p=AllChem.ETKDGv3();p.randomSeed=int(seed);status=AllChem.EmbedMolecule(hmol,p);source="ETKDGv3"
 if status==0:
  try:
   if AllChem.MMFFHasAllMoleculeParams(hmol):AllChem.MMFFOptimizeMolecule(hmol,maxIters=500);source+="+MMFF"
   elif AllChem.UFFHasAllMoleculeParams(hmol):AllChem.UFFOptimizeMolecule(hmol,maxIters=500);source+="+UFF"
  except Exception:source+="+unoptimized"
 else:hmol=Chem.Mol(mol);rdDepictor.Compute2DCoords(hmol);source="RDKit2D_fallback"
 c=hmol.GetConformer();xyz=torch.tensor([[*c.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())],dtype=torch.float32);xyz-=xyz.mean(0,keepdim=True);return xyz,source

def _record(row,i,seed):
 mol=Chem.MolFromSmiles(row["SMILES"])
 if mol is None or len(Chem.GetMolFrags(mol))!=1:raise ValueError("invalid/fragmented SMILES")
 symbols=[a.GetSymbol() for a in mol.GetAtoms()];unknown=sorted(set(symbols)-set(ATOM_MAP))
 if unknown:raise ValueError(f"unsupported elements {unknown}")
 arom_atoms=torch.tensor([a.GetIsAromatic() for a in mol.GetAtoms()]);arom_edges=torch.zeros(len(symbols),len(symbols),dtype=torch.bool)
 for b in mol.GetBonds():
  if b.GetIsAromatic():u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx();arom_edges[u,v]=arom_edges[v,u]=1
 cycles=[]
 for ring in Chem.GetSymmSSSR(mol):
  ring=tuple(map(int,ring))
  if all(arom_atoms[j] for j in ring):cycles.append(ring)
 q=torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]);h=torch.tensor([a.GetTotalNumHs() for a in mol.GetAtoms()]);rad=torch.tensor([a.GetNumRadicalElectrons() for a in mol.GetAtoms()])
 if q.min() < -2 or q.max()>2 or h.max()>4 or rad.max()>2:raise ValueError("state outside categorical vocabulary")
 kek=Chem.Mol(mol);Chem.Kekulize(kek,clearAromaticFlags=True);bonds=torch.zeros(len(symbols),len(symbols),dtype=torch.long)
 for b in kek.GetBonds():u,v=b.GetBeginAtomIdx(),b.GetEndAtomIdx();bonds[u,v]=bonds[v,u]=int(round(b.GetBondTypeAsDouble()))
 xyz,src=_coordinates(mol,seed+i)
 return {"id":row["ID"],"smiles":Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False),"atom_types":torch.tensor([ATOM_MAP.index(x) for x in symbols]),"atom_charges":q,"atom_hydrogens":h,"atom_radicals":rad,"atom_positions":xyz,"kekule_bonds":bonds,"aromatic_edges":arom_edges,"aromatic_cycles":cycles,"s1":float(row["NEVPT2_E(S1)"]),"t1":float(row["NEVPT2_E(T1)"]),"geometry_source":src}

def build(csv_path,output,seed=73):
 RDLogger.DisableLog("rdApp.*");output=Path(output);output.mkdir(parents=True,exist_ok=True)
 with open(csv_path,newline="") as f:source=list(csv.DictReader(f))
 records=[];errors=[]
 for i,row in enumerate(source):
  try:records.append(_record(row,i,seed))
  except Exception as e:errors.append({"row":i,"id":row.get("ID"),"error":str(e)})
 random.Random(seed).shuffle(records);n=len(records);a,b=int(.8*n),int(.9*n);splits={"train":records[:a],"val":records[a:b],"test":records[b:]}
 s1=torch.tensor([x["s1"] for x in splits["train"]]);t1=torch.tensor([x["t1"] for x in splits["train"]]);stats={"mode":"s1_t1","s1_energy_mean":float(s1.mean()),"s1_energy_std":float(s1.std()),"t1_energy_mean":float(t1.mean()),"t1_energy_std":float(t1.std())}
 for k,v in splits.items():torch.save(v,output/f"{k}.pt")
 geo={}
 for x in records:geo[x["geometry_source"]]=geo.get(x["geometry_source"],0)+1
 meta={"schema_version":1,"dataset_type":"hp1kg_nevpt2","source_csv":str(Path(csv_path).resolve()),"atom_map":list(ATOM_MAP),"condition_stats":stats,"splits":{k:len(v) for k,v in splits.items()},"heavy_atom_min":min(len(x["atom_types"]) for x in records),"heavy_atom_max":max(len(x["atom_types"]) for x in records),"geometry_sources":geo,"errors":errors,"retain_inverted_s1_t1":True,"seed":seed};(output/"manifest.json").write_text(json.dumps(meta,indent=2));return meta

class HP1KGDataset(Dataset):
 def __init__(self,root,split,condition_stats=None,atom_map=None):
  root=Path(root);self.meta=json.loads((root/"manifest.json").read_text());self.rows=torch.load(root/f"{split}.pt",weights_only=False);self.atom_map=tuple(self.meta["atom_map"])
  if atom_map is not None and tuple(atom_map)!=self.atom_map:raise ValueError("atom map mismatch")
  self.condition_stats=condition_stats or self.meta["condition_stats"]
 def __len__(self):return len(self.rows)
 def heavy_count_at(self,i):return len(self.rows[i]["atom_types"])
 def __getitem__(self,i):
  x=self.rows[i];s=self.condition_stats;return {**x,"property":torch.tensor([(x["s1"]-s["s1_energy_mean"])/s["s1_energy_std"],(x["t1"]-s["t1_energy_mean"])/s["t1_energy_std"]])}

def collate_hp1kg(rows,max_atoms):
 B=len(rows);A=max_atoms;R=max(1,max(len(x["aromatic_cycles"]) for x in rows));z=lambda *s,dtype=torch.long:torch.zeros(*s,dtype=dtype)
 o={"condition":torch.stack([x["property"] for x in rows]),"ring_mask":z(B,R,dtype=torch.bool),"atom_mask":z(B,A,dtype=torch.bool),"atom_type":z(B,A),"atom_charge":z(B,A),"atom_hydrogen":z(B,A),"atom_radical":z(B,A),"atom_positions":z(B,A,3,dtype=torch.float32),"atom_bonds":z(B,A,A),"aromatic_cycle_edges":z(B,R,A,A,dtype=torch.bool),"aromatic_cycle_double_count":z(B,R,dtype=torch.float32),"aromatic_cycle_order":torch.full((B,R,A),-1,dtype=torch.long),"aromatic_cycle_size":z(B,R),"aromatic_edge_mask":z(B,A,A,dtype=torch.bool)}
 for b,x in enumerate(rows):
  n=len(x["atom_types"])
  if n>A:raise ValueError(f"{n} atoms exceeds max_atoms={A}")
  o["atom_mask"][b,:n]=1;o["atom_type"][b,:n]=x["atom_types"];o["atom_charge"][b,:n]=x["atom_charges"]+2;o["atom_hydrogen"][b,:n]=x["atom_hydrogens"];o["atom_radical"][b,:n]=x["atom_radicals"];o["atom_positions"][b,:n]=x["atom_positions"];o["atom_bonds"][b,:n,:n]=x["kekule_bonds"];o["aromatic_edge_mask"][b,:n,:n]=x["aromatic_edges"]
  for r,ring in enumerate(x["aromatic_cycles"]):
   o["ring_mask"][b,r]=1;o["aromatic_cycle_size"][b,r]=len(ring);o["aromatic_cycle_order"][b,r,:len(ring)]=torch.tensor(ring)
   for j,u in enumerate(ring):v=ring[(j+1)%len(ring)];o["aromatic_cycle_edges"][b,r,u,v]=o["aromatic_cycle_edges"][b,r,v,u]=1
   o["aromatic_cycle_double_count"][b,r]=sum(int(x["kekule_bonds"][ring[j],ring[(j+1)%len(ring)]])==2 for j in range(len(ring)))
 return o


def main():
 import argparse
 p=argparse.ArgumentParser(description="Build the HP1KG NEVPT2 training cache")
 p.add_argument("--csv",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=73);a=p.parse_args()
 print(json.dumps(build(a.csv,a.output,a.seed),indent=2))

if __name__=="__main__":main()
