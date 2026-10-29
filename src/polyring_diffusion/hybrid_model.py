"""Joint diffusion over rigid rings and unconstrained non-ring heavy atoms."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .diffusion import SinusoidalTimeEmbedding
from .model import RingDenoiser

class HybridAtomLayer(nn.Module):
    def __init__(self,h,bond_classes=5):
        super().__init__();self.bond=nn.Embedding(bond_classes+1,h);self.msg=nn.Sequential(nn.Linear(3*h+1,h),nn.SiLU(),nn.Linear(h,h));self.up=nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,h));self.coord=nn.Linear(h,1,bias=False);self.norm=nn.LayerNorm(h)
    def forward(self,x,pos,bonds,mask,physical_radius,size_aware=False):
        x=torch.nan_to_num(x)*mask.unsqueeze(-1);pos=torch.nan_to_num(pos);n=x.shape[1];left=x[:,:,None,:].expand(-1,n,n,-1);right=x[:,None,:,:].expand(-1,n,n,-1);rel=pos[:,:,None,:]-pos[:,None,:,:];pm=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=x.device)[None];distance2=rel.square().sum(-1,keepdim=True)
        if size_aware:
            pair_scale=physical_radius[:,:,None,:]+physical_radius[:,None,:,:];distance2=distance2/pair_scale.square().clamp_min(1e-6)
        m=self.msg(torch.cat((left,right,self.bond(bonds),distance2),dim=-1))*pm.unsqueeze(-1);degree=pm.sum(2,keepdim=True).clamp_min(1).to(x.dtype);x=self.norm(x+self.up(torch.cat((x,m.sum(2)/degree),-1)))*mask.unsqueeze(-1);return x,pos+(rel*torch.tanh(self.coord(m))).sum(2)/degree*mask.unsqueeze(-1)

class HybridRingAtomDenoiser(nn.Module):
    def __init__(self,num_atom_types=9,max_free_atoms=64,atom_hidden_dim=192,atom_layers=4,charge_min=-2,charge_max=2,atom_radii=None,entity_size_aware=False,**ring_config):
        super().__init__();self.ring_model=RingDenoiser(**ring_config);self.num_atom_types=num_atom_types;self.atom_mask_token=num_atom_types;self.atom_pad_token=num_atom_types+1;self.charge_min=charge_min;self.charge_max=charge_max;self.num_charges=charge_max-charge_min+1;self.charge_mask_token=self.num_charges;self.max_free_atoms=max_free_atoms
        self.entity_size_aware=bool(entity_size_aware);radii=torch.ones(num_atom_types,dtype=torch.float32) if atom_radii is None else torch.as_tensor(atom_radii,dtype=torch.float32);mean=radii.mean().reshape(1);self.register_buffer('atom_radii',torch.cat((radii,mean,mean)),persistent=False)
        td=ring_config.get('time_dim',64);cd=ring_config.get('condition_dim',0);self.atom_type_embedding=nn.Embedding(num_atom_types+2,atom_hidden_dim);self.charge_embedding=nn.Embedding(self.num_charges+1,atom_hidden_dim);self.time_embedding=SinusoidalTimeEmbedding(td);self.atom_input=nn.Sequential(nn.Linear(atom_hidden_dim*2+td,atom_hidden_dim),nn.SiLU(),nn.Linear(atom_hidden_dim,atom_hidden_dim));self.condition_projection=nn.Sequential(nn.Linear(cd,atom_hidden_dim),nn.SiLU(),nn.Linear(atom_hidden_dim,atom_hidden_dim)) if cd else None
        cross_dim=atom_hidden_dim*3+(1 if self.entity_size_aware else 0);self.attachment_embedding=nn.Embedding(6,atom_hidden_dim);self.ring_to_atom=nn.Linear(ring_config.get('hidden_dim',192),atom_hidden_dim);self.cross_attention=nn.MultiheadAttention(atom_hidden_dim,4,batch_first=True);self.layers=nn.ModuleList(HybridAtomLayer(atom_hidden_dim) for _ in range(atom_layers));self.atom_type_head=nn.Linear(atom_hidden_dim,num_atom_types);self.charge_head=nn.Linear(atom_hidden_dim,self.num_charges);self.bond_head=nn.Sequential(nn.Linear(atom_hidden_dim*2,atom_hidden_dim),nn.SiLU(),nn.Linear(atom_hidden_dim,5));self.attach_head=nn.Sequential(nn.Linear(cross_dim,atom_hidden_dim),nn.SiLU(),nn.Linear(atom_hidden_dim,5));self.attach_port_head=nn.Sequential(nn.Linear(cross_dim,atom_hidden_dim),nn.SiLU(),nn.Linear(atom_hidden_dim,ring_config.get('max_ports',8)));self.count_head=nn.Sequential(nn.Linear(max(cd,1),atom_hidden_dim),nn.SiLU(),nn.Linear(atom_hidden_dim,max_free_atoms+1))
    def charge_index(self,q):return (q-self.charge_min).clamp(0,self.num_charges-1)
    def forward(self,ring_args,atom_types,atom_charges,atom_positions,atom_bonds,attachment_types,atom_mask,timesteps,condition=None,condition_drop_mask=None):
        if len(ring_args)==6:
            rt,rc,rn,rg,rct,rm=ring_args;ring=self.ring_model(rt,rc,rn,rct,rm,timesteps,condition,condition_drop_mask,tangents=rg)
        else:ring=self.ring_model(*ring_args,timesteps,condition,condition_drop_mask)
        initial=atom_positions;t=self.time_embedding(timesteps)[:,None,:].expand(-1,atom_types.shape[1],-1);x=self.atom_input(torch.cat((self.atom_type_embedding(atom_types),self.charge_embedding(atom_charges),t),-1));atom_radius=self.atom_radii[atom_types].unsqueeze(-1)
        if self.condition_projection is not None:
            c=self.condition_projection(condition.to(x.dtype));
            if condition_drop_mask is not None:c=c*(~condition_drop_mask).to(c.dtype).unsqueeze(-1)
            x=x+c[:,None,:]
        ring_mask=rm if len(ring_args)==6 else ring_args[4];projected_ring=self.ring_to_atom(ring['hidden']);context=torch.zeros_like(x);has_ring=ring_mask.any(1)
        if has_ring.any():context[has_ring],_=self.cross_attention(x[has_ring],torch.nan_to_num(projected_ring[has_ring]),torch.nan_to_num(projected_ring[has_ring]),key_padding_mask=~ring_mask[has_ring])
        x=torch.nan_to_num(x+context)*atom_mask.unsqueeze(-1)
        for layer in self.layers:x,atom_positions=layer(x,atom_positions,atom_bonds,atom_mask,atom_radius,self.entity_size_aware)
        n=x.shape[1];pair=torch.cat((x[:,:,None,:].expand(-1,n,n,-1),x[:,None,:,:].expand(-1,n,n,-1)),-1);bond=self.bond_head(pair);bond=.5*(bond+bond.transpose(1,2));rh=self.ring_to_atom(ring['hidden']);parts=[rh[:,:,None,:].expand(-1,-1,n,-1),x[:,None,:,:].expand(-1,rh.shape[1],-1,-1),self.attachment_embedding(attachment_types)]
        if self.entity_size_aware:parts.append((ring['centers'][:,:,None,:]-atom_positions[:,None,:,:]).square().sum(-1,keepdim=True)/(ring['physical_radius'][:,:,None,:]+atom_radius[:,None,:,:]).square().clamp_min(1e-6))
        cross=torch.cat(parts,-1)
        count_condition=condition if condition is not None else x.new_zeros((x.shape[0],1))
        if condition_drop_mask is not None:count_condition=count_condition*(~condition_drop_mask).to(x.dtype).unsqueeze(-1)
        ring.update(atom_coordinate_noise=atom_positions-initial,atom_type_logits=self.atom_type_head(x),atom_charge_logits=self.charge_head(x),atom_bond_logits=bond,attachment_logits=self.attach_head(cross),attachment_port_logits=self.attach_port_head(cross),free_atom_count_logits=self.count_head(count_condition),atom_hidden=x)
        return ring
