"""SE(3)-equivariant heavy-atom/FPA-cell diffusion denoiser (v2)."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .diffusion import SinusoidalTimeEmbedding
from .fpa_diffusion_v2 import masked_center


class EntityBlock(nn.Module):
    def __init__(self, h: int, relations: int = 10):
        super().__init__(); self.rel=nn.Embedding(relations,h)
        self.msg=nn.Sequential(nn.Linear(3*h+8,h),nn.SiLU(),nn.Linear(h,h));self.up=nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,h));self.norm=nn.LayerNorm(h)
    def forward(self,h,pos,frame,radius,kind,relation,mask):
        n=h.shape[1];pair=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=h.device)[None]
        delta=pos[:,:,None]-pos[:,None];dist=delta.norm(dim=-1,keepdim=True).clamp_min(1e-6);rs=(radius[:,:,None]+radius[:,None]).clamp_min(1e-4)
        normal=frame[...,2];tangent=frame[...,0];src_cell=kind[:,:,None,None].to(pos.dtype);dst_cell=kind[:,None,:,None].to(pos.dtype)
        axial=(delta*normal[:,:,None]).sum(-1,keepdim=True).abs()/(.18*radius[:,:,None]+.1);planar=torch.sqrt((dist.square()-(delta*normal[:,:,None]).sum(-1,keepdim=True).square()).clamp_min(0))/radius[:,:,None].clamp_min(.1)
        geo=(dist/rs,(dist-rs)/rs,torch.log((radius[:,:,None]+1e-4)/(radius[:,None]+1e-4)),(normal[:,:,None]*normal[:,None]).sum(-1,keepdim=True),(tangent[:,:,None]*tangent[:,None]).sum(-1,keepdim=True),axial*src_cell,planar*src_cell,src_cell-dst_cell)
        left=h[:,:,None].expand(-1,n,n,-1);right=h[:,None].expand(-1,n,n,-1)
        m=self.msg(torch.cat((left,right,self.rel(relation),*geo),-1))*pair.unsqueeze(-1);degree=pair.sum(2,keepdim=True).clamp_min(1).to(h.dtype)
        return self.norm(h+self.up(torch.cat((h,m.sum(2)/degree),-1)))*mask.unsqueeze(-1)


class FPAEntityDenoiserV2(nn.Module):
    def __init__(self,num_atom_types,num_cell_types,atom_radii,hidden_dim=192,num_layers=8,time_dim=64,condition_dim=120,condition_dropout=.2,max_cell_ports=8,max_atoms=128,max_cells=32,charge_min=-2,charge_max=2):
        super().__init__();self.num_atom_types=num_atom_types;self.num_cell_types=num_cell_types;self.max_cell_ports=max_cell_ports;self.max_atoms=max_atoms;self.max_cells=max_cells;self.condition_dim=condition_dim;self.condition_dropout=condition_dropout;self.charge_min=charge_min;self.num_charges=charge_max-charge_min+1
        self.atom_mask_token=num_atom_types;self.atom_pad_token=num_atom_types+1;self.cell_mask_token=num_cell_types;self.cell_pad_token=num_cell_types+1;self.charge_mask_token=self.num_charges;self.size_mask_token=max_cell_ports+1
        h=hidden_dim;self.atom_emb=nn.Embedding(num_atom_types+2,h);self.cell_emb=nn.Embedding(num_cell_types+2,h);self.charge_emb=nn.Embedding(self.num_charges+1,h);self.size_emb=nn.Embedding(max_cell_ports+2,h);self.kind_emb=nn.Embedding(2,h);self.time=SinusoidalTimeEmbedding(time_dim)
        self.atom_in=nn.Sequential(nn.Linear(3*h+time_dim+1,h),nn.SiLU(),nn.Linear(h,h));self.cell_in=nn.Sequential(nn.Linear(3*h+time_dim+1,h),nn.SiLU(),nn.Linear(h,h));self.cond=nn.Sequential(nn.Linear(condition_dim,h),nn.SiLU(),nn.Linear(h,h));self.blocks=nn.ModuleList(EntityBlock(h) for _ in range(num_layers))
        self.atom_type=nn.Linear(h,num_atom_types);self.charge=nn.Linear(h,self.num_charges);self.cell_type=nn.Linear(h,num_cell_types);self.cell_size=nn.Linear(h,max_cell_ports+1);self.radius_eps=nn.Linear(h,1);self.omega_coeff=nn.Linear(h,3)
        self.coord_pair=nn.Sequential(nn.Linear(2*h+2,h),nn.SiLU(),nn.Linear(h,1,bias=False));self.cell_coord_pair=nn.Sequential(nn.Linear(2*h+2,h),nn.SiLU(),nn.Linear(h,1,bias=False))
        self.bond=nn.Sequential(nn.Linear(2*h+2,h),nn.SiLU(),nn.Linear(h,5));self.fusion=nn.Sequential(nn.Linear(2*h+2,h),nn.SiLU(),nn.Linear(h,2));self.member=nn.Sequential(nn.Linear(2*h+2,h),nn.SiLU(),nn.Linear(h,2))
        self.atom_count=nn.Sequential(nn.Linear(condition_dim,h),nn.SiLU(),nn.Linear(h,max_atoms+1));self.cell_count=nn.Sequential(nn.Linear(condition_dim,h),nn.SiLU(),nn.Linear(h,max_cells+1))
        radii=torch.as_tensor(atom_radii,dtype=torch.float32);self.register_buffer('atom_radii',torch.cat((radii,radii.mean()[None],radii.mean()[None])),persistent=False)
    def charge_index(self,q):return (q-self.charge_min).clamp(0,self.num_charges-1)
    @staticmethod
    def pair(h1,h2,p1,p2):
        d=(p1[:,:,None]-p2[:,None]).norm(dim=-1,keepdim=True);return torch.cat((h1[:,:,None].expand(-1,-1,h2.shape[1],-1),h2[:,None].expand(-1,h1.shape[1],-1,-1),d,d.square()),-1)
    def forward(self,atom_types,charges,atom_pos,atom_bonds,atom_mask,cell_types,cell_sizes,cell_pos,cell_frames,cell_radius,fusion,membership,cell_mask,t,condition,drop=None):
        b,na=atom_types.shape;nc=cell_types.shape[1];te=self.time(t);ar=self.atom_radii[atom_types].unsqueeze(-1)
        ah=self.atom_in(torch.cat((self.atom_emb(atom_types),self.charge_emb(charges),self.kind_emb.weight[0][None,None].expand(b,na,-1),te[:,None].expand(-1,na,-1),ar),-1));ch=self.cell_in(torch.cat((self.cell_emb(cell_types),self.size_emb(cell_sizes),self.kind_emb.weight[1][None,None].expand(b,nc,-1),te[:,None].expand(-1,nc,-1),cell_radius.unsqueeze(-1)),-1))
        c=self.cond(condition.to(ah.dtype));c=c*((~drop).to(c.dtype).unsqueeze(-1) if drop is not None else 1);ah=ah+c[:,None];ch=ch+c[:,None]
        h=torch.cat((ah*atom_mask.unsqueeze(-1),ch*cell_mask.unsqueeze(-1)),1);pos=torch.cat((atom_pos,cell_pos),1);atom_frame=torch.zeros((b,na,3,3),device=h.device,dtype=atom_pos.dtype);frame=torch.cat((atom_frame,cell_frames),1);physical_cell_radius=cell_radius.exp().clamp(.1,10.);radius=torch.cat((ar,physical_cell_radius.unsqueeze(-1)),1);kind=torch.cat((torch.zeros_like(atom_mask),torch.ones_like(cell_mask)),1);mask=torch.cat((atom_mask,cell_mask),1)
        rel=atom_bonds.new_zeros((b,na+nc,na+nc));rel[:,:na,:na]=atom_bonds;rel[:,na:,na:]=torch.where(fusion.eq(1),8,torch.where(fusion.eq(2),9,0));rel[:,na:,:na]=torch.where(membership.eq(1),6,torch.where(membership.eq(2),7,0));rel[:,:na,na:]=rel[:,na:,:na].transpose(1,2)
        for block in self.blocks:h=block(h,pos,frame,radius,kind,rel,mask)
        ah,ch=h[:,:na],h[:,na:];aa=self.pair(ah,ah,atom_pos,atom_pos);cc=self.pair(ch,ch,cell_pos,cell_pos);ca=self.pair(ch,ah,cell_pos,atom_pos)
        bond=self.bond(aa);bond=(bond+bond.transpose(1,2))/2;fusion_logits=self.fusion(cc);fusion_logits=(fusion_logits+fusion_logits.transpose(1,2))/2
        ad=atom_pos[:,:,None]-atom_pos[:,None];aw=self.coord_pair(aa);aw=(aw+aw.transpose(1,2))/2;aeps=(ad*aw).sum(2);aeps=masked_center(aeps,atom_mask)
        cd=cell_pos[:,:,None]-torch.cat((atom_pos,cell_pos),1)[:,None];cx=self.pair(ch,h,cell_pos,pos);ceps=(cd*self.cell_coord_pair(cx)).sum(2)/(mask.sum(1)[:,None,None].clamp_min(1));ceps=ceps*cell_mask.unsqueeze(-1)
        coeff=self.omega_coeff(ch);omega=(cell_frames*coeff[..., :, None]).sum(-2)*cell_mask.unsqueeze(-1)
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return {'atom_eps':aeps,'cell_eps':ceps,'radius_eps':self.radius_eps(ch).squeeze(-1)*cell_mask,'rotation_score':omega,'atom_type_logits':self.atom_type(ah),'charge_logits':self.charge(ah),'bond_logits':bond,'cell_type_logits':self.cell_type(ch),'cell_size_logits':self.cell_size(ch),'fusion_logits':fusion_logits,'membership_logits':self.member(ca),'atom_count_logits':self.atom_count(condition),'cell_count_logits':self.cell_count(condition)}
