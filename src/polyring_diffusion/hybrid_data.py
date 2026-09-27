"""Hybrid rigid-ring plus free-heavy-atom views of the lossless database."""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F
from .real_data import RigidRingDataset

class HybridRingAtomDataset(RigidRingDataset):
    """Keep rings rigid while exposing every non-ring heavy atom individually."""
    def __init__(self,*args,**kwargs):
        supplied_condition_stats=kwargs.pop('condition_stats',None)
        super().__init__(*args,**kwargs)
        self.atom_map=None
        for shard,rel in zip(self.shards,next(x for x in self.manifest['splits'] if x['name']==kwargs.get('split',args[1] if len(args)>1 else 'train'))['shards']):
            raw=torch.load(self.root/rel,map_location='cpu',weights_only=False)
            for key in ('atom_positions','atom_types','atom_charges','atom_offsets','bond_indices','bond_types','bond_offsets','ring_atom_indices','ring_atom_offsets','per_ring_atom_offsets','smiles','molecule_ids'):
                shard[key]=raw[key]
            for key in ('singlet_energies','singlet_oscillator_strengths','singlet_mask','triplet_energies','triplet_oscillator_strengths','triplet_mask'):
                if key in raw:shard[key]=raw[key]
            self.atom_map=tuple(raw['metadata']['atom_map'])
        self.num_atom_types=len(self.atom_map);self.h_index=self.atom_map.index('H') if 'H' in self.atom_map else None
        self.has_spectrum='singlet_energies' in self.shards[0]
        if self.has_spectrum:
            if supplied_condition_stats is None:
                supplied_condition_stats={}
                for prefix in ('singlet','triplet'):
                    mask=torch.cat([x[prefix+'_mask'] for x in self.shards]);energy=torch.cat([x[prefix+'_energies'] for x in self.shards])[mask];osc=torch.cat([x[prefix+'_oscillator_strengths'] for x in self.shards])[mask].clamp_min(0).add(1e-6).log10();supplied_condition_stats[prefix]={'energy_mean':float(energy.mean()),'energy_std':float(energy.std().clamp_min(1e-6)),'log_os_mean':float(osc.mean()),'log_os_std':float(osc.std().clamp_min(1e-6))}
            self.condition_stats=supplied_condition_stats;self.condition_dim=120
        else:self.condition_stats=None;self.condition_dim=1
        self.max_free_atoms=max(self._free_count(s,i) for s in self.shards for i in range(s['length']))

    @staticmethod
    def _bounds(x,i):return map(int,x[i:i+2])
    def _free_count(self,s,i):
        a,b=self._bounds(s['atom_offsets'],i);ra,rb=self._bounds(s['ring_atom_offsets'],i)
        ring=set(map(int,s['ring_atom_indices'][ra:rb]));types=s['atom_types'][a:b].to(torch.uint8).argmax(-1)
        return sum((self.h_index is None or int(t)!=self.h_index) and j not in ring for j,t in enumerate(types))

    def __getitem__(self,index):
        out=super().__getitem__(index);si=__import__('bisect').bisect_right(self.ends,index);base=0 if si==0 else self.ends[si-1];i=index-base;s=self.shards[si]
        a,b=self._bounds(s['atom_offsets'],i);ba,bb=self._bounds(s['bond_offsets'],i);ra,rb=self._bounds(s['ring_atom_offsets'],i)
        types=s['atom_types'][a:b].to(torch.uint8).argmax(-1).long();ring_flat=s['ring_atom_indices'][ra:rb].long();ring_set=set(map(int,ring_flat));free_ids=[j for j,t in enumerate(types.tolist()) if (self.h_index is None or t!=self.h_index) and j not in ring_set]
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
        positions=s['atom_positions'][a:b].float();tangents=[]
        for ring in range(nr):
            lo,hi=map(int,offsets[ring:ring+2]);normal=F.normalize(out['normals'][ring],dim=0)
            candidates=positions[ring_flat[lo:hi]]-out['centers'][ring]
            candidates=candidates-(candidates*normal).sum(-1,keepdim=True)*normal
            lengths=candidates.square().sum(-1)
            if bool((lengths>1e-8).any()):tangent=F.normalize(candidates[int(torch.where(lengths>1e-8)[0][0])],dim=0)
            else:
                axis=normal.new_tensor([1.,0.,0.]) if abs(float(normal[0]))<.9 else normal.new_tensor([0.,1.,0.])
                tangent=F.normalize(torch.linalg.cross(normal,axis),dim=0)
            tangents.append(tangent)
        ids=torch.tensor(free_ids,dtype=torch.long)
        out.update(ring_tangents=torch.stack(tangents) if tangents else positions.new_zeros((0,3)),free_atom_types=types[ids],free_atom_charges=s['atom_charges'][a:b][ids].long(),free_positions=positions[ids],free_bond_types=atom_bonds,attachment_types=attachment,attachment_ports=attachment_ports)
        if self.has_spectrum:
            chunks=[]
            for prefix in ('singlet','triplet'):
                mask=s[prefix+'_mask'][i].bool();stats=self.condition_stats[prefix];energy=(s[prefix+'_energies'][i].float()-stats['energy_mean'])/stats['energy_std'];log_os=(s[prefix+'_oscillator_strengths'][i].float().clamp_min(0)+1e-6).log10();log_os=(log_os-stats['log_os_mean'])/stats['log_os_std'];chunks.extend((energy*mask,log_os*mask,mask.float()))
            out['property']=torch.cat(chunks)
        return out

def collate_hybrid(batch:list[dict[str,Tensor]])->dict[str,Tensor]:
    from .real_data import collate_rigid
    out=collate_rigid(batch);b=len(batch);na=max(1,max(len(x['free_atom_types']) for x in batch));nr=out['ring_types'].shape[1]
    out.update(ring_tangents=torch.zeros((b,nr,3)),free_atom_types=torch.zeros((b,na),dtype=torch.long),free_atom_charges=torch.zeros((b,na),dtype=torch.long),free_positions=torch.zeros((b,na,3)),free_atom_mask=torch.zeros((b,na),dtype=torch.bool),free_bond_types=torch.zeros((b,na,na),dtype=torch.long),attachment_types=torch.zeros((b,nr,na),dtype=torch.long),attachment_ports=torch.zeros((b,nr,na),dtype=torch.long),free_atom_count=torch.zeros(b,dtype=torch.long))
    for i,x in enumerate(batch):
        out['ring_tangents'][i,:len(x['ring_types'])]=x['ring_tangents']
        n=len(x['free_atom_types']);out['free_atom_count'][i]=n
        if not n:continue
        center=x['centers'].mean(0,keepdim=True) if len(x['centers']) else x['free_positions'].mean(0,keepdim=True)
        out['free_atom_types'][i,:n]=x['free_atom_types'];out['free_atom_charges'][i,:n]=x['free_atom_charges'];out['free_positions'][i,:n]=x['free_positions']-center;out['free_atom_mask'][i,:n]=True;out['free_bond_types'][i,:n,:n]=x['free_bond_types'];out['attachment_types'][i,:len(x['ring_types']),:n]=x['attachment_types'];out['attachment_ports'][i,:len(x['ring_types']),:n]=x['attachment_ports']
    return out
