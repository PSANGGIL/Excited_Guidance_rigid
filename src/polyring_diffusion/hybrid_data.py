"""Hybrid rigid-ring plus free-heavy-atom views of the lossless database."""
from __future__ import annotations
import torch
from torch import Tensor
from .real_data import RigidRingDataset

class HybridRingAtomDataset(RigidRingDataset):
    """Keep rings rigid while exposing every non-ring heavy atom individually."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.atom_map=None
        for shard,rel in zip(self.shards,next(x for x in self.manifest['splits'] if x['name']==kwargs.get('split',args[1] if len(args)>1 else 'train'))['shards']):
            raw=torch.load(self.root/rel,map_location='cpu',weights_only=False)
            for key in ('atom_positions','atom_types','atom_charges','atom_offsets','bond_indices','bond_types','bond_offsets','ring_atom_indices','ring_atom_offsets','per_ring_atom_offsets'):
                shard[key]=raw[key]
            self.atom_map=tuple(raw['metadata']['atom_map'])
        self.num_atom_types=len(self.atom_map)
        self.max_free_atoms=max(self._free_count(s,i) for s in self.shards for i in range(s['length']))

    @staticmethod
    def _bounds(x,i):return map(int,x[i:i+2])
    def _free_count(self,s,i):
        a,b=self._bounds(s['atom_offsets'],i);ra,rb=self._bounds(s['ring_atom_offsets'],i)
        ring=set(map(int,s['ring_atom_indices'][ra:rb]));types=s['atom_types'][a:b].to(torch.uint8).argmax(-1)
        return sum(int(t)!=0 and j not in ring for j,t in enumerate(types))

    def __getitem__(self,index):
        out=super().__getitem__(index);si=__import__('bisect').bisect_right(self.ends,index);base=0 if si==0 else self.ends[si-1];i=index-base;s=self.shards[si]
        a,b=self._bounds(s['atom_offsets'],i);ba,bb=self._bounds(s['bond_offsets'],i);ra,rb=self._bounds(s['ring_atom_offsets'],i)
        types=s['atom_types'][a:b].to(torch.uint8).argmax(-1).long();ring_flat=s['ring_atom_indices'][ra:rb].long();ring_set=set(map(int,ring_flat));free_ids=[j for j,t in enumerate(types.tolist()) if t!=0 and j not in ring_set]
        lookup={old:new for new,old in enumerate(free_ids)};nf=len(free_ids);nr=len(out['ring_types'])
        atom_bonds=torch.zeros((nf,nf),dtype=torch.long);attachment=torch.zeros((nr,nf),dtype=torch.long);attachment_ports=torch.zeros((nr,nf),dtype=torch.long)
        atom_to_ports={}
        offsets=s['per_ring_atom_offsets'][i]
        for ring in range(nr):
            lo,hi=map(int,offsets[ring:ring+2])
            for port,atom in enumerate(ring_flat[lo:hi].tolist()):atom_to_ports.setdefault(int(atom),(ring,port))
        for edge,kind in zip(s['bond_indices'][ba:bb].long().tolist(),s['bond_types'][ba:bb].long().tolist()):
            u,v=map(int,edge);kind=int(kind)
            if u in lookup and v in lookup:atom_bonds[lookup[u],lookup[v]]=atom_bonds[lookup[v],lookup[u]]=kind
            elif u in lookup and v in atom_to_ports:
                ring,port=atom_to_ports[v];attachment[ring,lookup[u]]=kind;attachment_ports[ring,lookup[u]]=port
            elif v in lookup and u in atom_to_ports:
                ring,port=atom_to_ports[u];attachment[ring,lookup[v]]=kind;attachment_ports[ring,lookup[v]]=port
        ids=torch.tensor(free_ids,dtype=torch.long)
        out.update(free_atom_types=types[ids],free_atom_charges=s['atom_charges'][a:b][ids].long(),free_positions=s['atom_positions'][a:b][ids].float(),free_bond_types=atom_bonds,attachment_types=attachment,attachment_ports=attachment_ports)
        return out

def collate_hybrid(batch:list[dict[str,Tensor]])->dict[str,Tensor]:
    from .real_data import collate_rigid
    out=collate_rigid(batch);b=len(batch);na=max(1,max(len(x['free_atom_types']) for x in batch));nr=out['ring_types'].shape[1]
    out.update(free_atom_types=torch.zeros((b,na),dtype=torch.long),free_atom_charges=torch.zeros((b,na),dtype=torch.long),free_positions=torch.zeros((b,na,3)),free_atom_mask=torch.zeros((b,na),dtype=torch.bool),free_bond_types=torch.zeros((b,na,na),dtype=torch.long),attachment_types=torch.zeros((b,nr,na),dtype=torch.long),attachment_ports=torch.zeros((b,nr,na),dtype=torch.long),free_atom_count=torch.zeros(b,dtype=torch.long))
    for i,x in enumerate(batch):
        n=len(x['free_atom_types']);out['free_atom_count'][i]=n
        if not n:continue
        out['free_atom_types'][i,:n]=x['free_atom_types'];out['free_atom_charges'][i,:n]=x['free_atom_charges'];out['free_positions'][i,:n]=x['free_positions']-x['centers'].mean(0,keepdim=True);out['free_atom_mask'][i,:n]=True;out['free_bond_types'][i,:n,:n]=x['free_bond_types'];out['attachment_types'][i,:len(x['ring_types']),:n]=x['attachment_types'];out['attachment_ports'][i,:len(x['ring_types']),:n]=x['attachment_ports']
    return out
