"""Deterministic construction of an atom scaffold from ordered ring slots."""
from __future__ import annotations
from dataclasses import dataclass
import torch
@dataclass
class ScaffoldAssembly:
 slot_to_atom:torch.Tensor
 atom_slots:tuple[tuple[tuple[int,int],...],...]
 perimeter_edges:tuple[tuple[int,int],...]
 aromatic_edges:tuple[tuple[int,int],...]
 valid:bool
 errors:tuple[str,...]
class _UF:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[b]=a

def assemble_scaffold(ring_sizes,ring_aromatic,junction_probability,threshold=.5):
 """Merge ring slots with symmetric, one-to-one junction assignments.

 Junctions are selected greedily per ring pair. Two slots in the same ring may
 never collapse, which prevents degenerate cycles and inconsistent fused atoms.
 """
 sizes=list(map(int,ring_sizes)); r=len(sizes); smax=junction_probability.shape[-1]; offsets=[]; total=0
 for n in sizes: offsets.append(total);total+=n
 uf=_UF(total); errors=[]
 for a in range(r):
  for b in range(a+1,r):
   candidates=sorted(((float(junction_probability[a,b,i,j]),i,j) for i in range(sizes[a]) for j in range(sizes[b])),reverse=True)
   used_a=set();used_b=set()
   for score,i,j in candidates:
    if score<threshold:break
    if i in used_a or j in used_b:continue
    # Reject a transitive merge that identifies two distinct slots of one ring.
    x,y=offsets[a]+i,offsets[b]+j; rx,ry=uf.find(x),uf.find(y)
    conflict=False
    for k in range(r):
     roots=[uf.find(offsets[k]+q) for q in range(sizes[k])]
     if rx in roots and ry in roots and rx!=ry: conflict=True
    if conflict:continue
    uf.union(x,y);used_a.add(i);used_b.add(j)
 groups={}
 for ring,n in enumerate(sizes):
  for slot in range(n): groups.setdefault(uf.find(offsets[ring]+slot),[]).append((ring,slot))
 roots=sorted(groups); atom_id={root:i for i,root in enumerate(roots)}; mapping=torch.full((r,smax),-1,dtype=torch.long)
 for root,slots in groups.items():
  for ring,slot in slots:mapping[ring,slot]=atom_id[root]
 edges=set();aromatic=set()
 for ring,n in enumerate(sizes):
  for i in range(n):
   edge=tuple(sorted((int(mapping[ring,i]),int(mapping[ring,(i+1)%n]))))
   if edge[0]==edge[1]:errors.append(f'degenerate ring {ring}')
   else:edges.add(edge); aromatic.update([edge] if bool(ring_aromatic[ring]) else [])
 return ScaffoldAssembly(mapping,tuple(tuple(v) for _,v in sorted(groups.items())),tuple(sorted(edges)),tuple(sorted(aromatic)),not errors,tuple(errors))
