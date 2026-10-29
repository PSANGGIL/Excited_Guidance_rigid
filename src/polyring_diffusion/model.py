"""Dense equivariant denoiser over a padded ring graph."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from .chem import element_record
from .diffusion import SinusoidalTimeEmbedding
from .graph import ConnectionType
from .templates import DEFAULT_LIBRARY, TemplateLibrary

def template_feature_table(library:TemplateLibrary=DEFAULT_LIBRARY)->Tensor:
    rows=[]
    for template in library:
        vdw=[element_record(s).vdw_radius for s in template.atoms]; cov=[element_record(s).covalent_radius for s in template.atoms]
        rows.append([template.size/8.0,sum(vdw)/len(vdw)/2.0,sum(cov)/len(cov)/1.2,sum(s!='C' for s in template.atoms)/template.size])
    return torch.tensor(rows,dtype=torch.float32)

class DenseEquivariantLayer(nn.Module):
    def __init__(self,hidden_dim:int,full_frame_orientation:bool=False):
        super().__init__();self.full_frame_orientation=bool(full_frame_orientation);edge_dim=hidden_dim*3+(3 if self.full_frame_orientation else 2)
        self.message=nn.Sequential(nn.Linear(edge_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU())
        self.node_update=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        self.center_weight=nn.Linear(hidden_dim,1,bias=False); self.normal_relative_weight=nn.Linear(hidden_dim,1,bias=False); self.normal_self_weight=nn.Linear(hidden_dim,1,bias=False); self.tangent_relative_weight=nn.Linear(hidden_dim,1,bias=False) if self.full_frame_orientation else None; self.tangent_self_weight=nn.Linear(hidden_dim,1,bias=False) if self.full_frame_orientation else None; self.norm=nn.LayerNorm(hidden_dim)
    def forward(self,hidden,centers,normals,tangents,connection_embedding,physical_radius,node_mask):
        batch,nodes,_=hidden.shape; left=hidden[:,:,None,:].expand(batch,nodes,nodes,-1); right=hidden[:,None,:,:].expand(batch,nodes,nodes,-1)
        pair_mask=node_mask[:,:,None]&node_mask[:,None,:]; pair_mask &= ~torch.eye(nodes,dtype=torch.bool,device=hidden.device)[None]
        pair_float=pair_mask.unsqueeze(-1).to(centers.dtype)
        relative=(centers[:,:,None,:]-centers[:,None,:,:])*pair_float; distance2=relative.square().sum(-1,keepdim=True)
        pair_scale=physical_radius[:,:,None,:]+physical_radius[:,None,:,:]; pair_scale=torch.where(pair_mask.unsqueeze(-1),pair_scale,torch.ones_like(pair_scale))
        nd2=distance2/pair_scale.square().clamp_min(1e-6); normal_dot=(normals[:,:,None,:]*normals[:,None,:,:]).sum(-1,keepdim=True)*pair_float;tangent_dot=(tangents[:,:,None,:]*tangents[:,None,:,:]).sum(-1,keepdim=True)*pair_float
        geometry=[nd2,normal_dot,tangent_dot] if self.full_frame_orientation else [nd2,normal_dot]
        message=self.message(torch.cat([left,right,connection_embedding,*geometry],-1))*pair_mask.unsqueeze(-1)
        degree=pair_mask.sum(2,keepdim=True).clamp_min(1).to(hidden.dtype); aggregated=message.sum(2)/degree
        hidden=self.norm(hidden+self.node_update(torch.cat([hidden,aggregated],-1)))*node_mask.unsqueeze(-1)
        relative_unit=relative/torch.sqrt(distance2+1.0)
        center_delta=0.5*(relative_unit*torch.tanh(self.center_weight(message))).sum(2)/degree
        nrel=normals[:,:,None,:]-normals[:,None,:,:]
        normal_delta=0.5*(nrel*torch.tanh(self.normal_relative_weight(message))+normals[:,:,None,:]*torch.tanh(self.normal_self_weight(message))).sum(2)/degree
        if self.full_frame_orientation:
            trel=tangents[:,:,None,:]-tangents[:,None,:,:]
            tangent_delta=0.5*(trel*torch.tanh(self.tangent_relative_weight(message))+tangents[:,:,None,:]*torch.tanh(self.tangent_self_weight(message))).sum(2)/degree
        else:tangent_delta=torch.zeros_like(tangents)
        return hidden,centers+center_delta*node_mask.unsqueeze(-1),normals+normal_delta*node_mask.unsqueeze(-1),tangents+tangent_delta*node_mask.unsqueeze(-1),message

class RingDenoiser(nn.Module):
    """Predict clean categorical states and continuous noise for ring poses."""
    def __init__(self,hidden_dim:int=192,num_layers:int=6,time_dim:int=64,max_ports:int=8,
                 library:TemplateLibrary=DEFAULT_LIBRARY,num_templates:int|None=None,
                 template_features:list[list[float]]|Tensor|None=None,condition_dim:int=0,condition_dropout:float=0.1,
                 property_head:bool=False,full_frame_orientation:bool=False):
        super().__init__()
        if template_features is None:
            base=template_feature_table(library); resolved=len(library) if num_templates is None else int(num_templates)
            if resolved!=len(library): raise ValueError('num_templates without template_features must match library')
            self.template_names=library.names
        else:
            base=torch.as_tensor(template_features,dtype=torch.float32); resolved=base.shape[0] if num_templates is None else int(num_templates)
            if base.ndim!=2 or base.shape[0]!=resolved: raise ValueError('template feature rows must equal num_templates')
            self.template_names=tuple(f'template_{i}' for i in range(resolved))
        self.num_templates=resolved; self.ring_mask_token=resolved; self.ring_pad_token=resolved+1; self.connection_mask_token=len(ConnectionType); self.max_ports=max_ports
        self.ring_embedding=nn.Embedding(resolved+2,hidden_dim); self.connection_embedding=nn.Embedding(len(ConnectionType)+1,hidden_dim)
        mean=base.mean(0,keepdim=True); self.register_buffer('template_features',torch.cat([base,mean,torch.zeros_like(mean)]))
        self.time_embedding=SinusoidalTimeEmbedding(time_dim); self.input_projection=nn.Sequential(nn.Linear(hidden_dim+time_dim+base.shape[1],hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        self.condition_dim=int(condition_dim); self.condition_dropout=float(condition_dropout); self.has_property_head=bool(property_head);self.full_frame_orientation=bool(full_frame_orientation)
        self.condition_projection=nn.Sequential(nn.Linear(self.condition_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim)) if self.condition_dim else None
        self.property_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,self.condition_dim)) if self.has_property_head and self.condition_dim else None
        self.layers=nn.ModuleList(DenseEquivariantLayer(hidden_dim,self.full_frame_orientation) for _ in range(num_layers)); self.ring_head=nn.Linear(hidden_dim,resolved); pair_dim=hidden_dim*3+(3 if self.full_frame_orientation else 2)
        self.connection_head=nn.Sequential(nn.Linear(pair_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,len(ConnectionType)))
        self.port_head=nn.Sequential(nn.Linear(pair_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,max_ports)); self.junction_port_head=nn.Sequential(nn.Linear(pair_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,max_ports))
    def forward(self,ring_types,centers,normals,connection_types,node_mask,timesteps,property_condition:Tensor|None=None,condition_drop_mask:Tensor|None=None,tangents:Tensor|None=None):
        initial_centers=centers; initial_normals=normals
        if tangents is None:
            axis=torch.zeros_like(normals);axis[...,0]=1;alternate=torch.zeros_like(normals);alternate[...,1]=1;axis=torch.where((normals[...,0].abs()>.9).unsqueeze(-1),alternate,axis);tangents=torch.linalg.cross(normals,axis,dim=-1)
        tangents=torch.nn.functional.normalize(tangents-(tangents*normals).sum(-1,keepdim=True)*normals,dim=-1)*node_mask.unsqueeze(-1);initial_tangents=tangents
        time=self.time_embedding(timesteps)[:,None,:].expand(-1,ring_types.shape[1],-1); features=self.template_features[ring_types]; physical_radius=features[...,1:2]
        hidden=self.input_projection(torch.cat([self.ring_embedding(ring_types),features,time],-1))
        if self.condition_projection is not None:
            if property_condition is None: property_condition=torch.zeros((ring_types.shape[0],self.condition_dim),device=ring_types.device,dtype=hidden.dtype)
            cond=self.condition_projection(property_condition.to(hidden.dtype))
            if condition_drop_mask is not None: cond=cond*(~condition_drop_mask).to(cond.dtype).unsqueeze(-1)
            hidden=hidden+cond[:,None,:]
        hidden=hidden*node_mask.unsqueeze(-1); edge_embedding=self.connection_embedding(connection_types); messages=None
        for layer in self.layers:
            hidden,centers,normals,tangents,messages=layer(hidden,centers,normals,tangents,edge_embedding,physical_radius,node_mask)
            normals=torch.nn.functional.normalize(normals,dim=-1)*node_mask.unsqueeze(-1)
            tangents=torch.nn.functional.normalize(tangents-(tangents*normals).sum(-1,keepdim=True)*normals,dim=-1)*node_mask.unsqueeze(-1)
        left=hidden[:,:,None,:].expand(-1,-1,hidden.shape[1],-1); right=hidden[:,None,:,:].expand(-1,hidden.shape[1],-1,-1)
        valid_pair=node_mask[:,:,None]&node_mask[:,None,:]; pair_float=valid_pair.unsqueeze(-1).to(centers.dtype)
        relative=(centers[:,:,None,:]-centers[:,None,:,:])*pair_float; distance2=relative.square().sum(-1,keepdim=True)
        pair_scale=physical_radius[:,:,None,:]+physical_radius[:,None,:,:]; pair_scale=torch.where(valid_pair.unsqueeze(-1),pair_scale,torch.ones_like(pair_scale))
        nd2=distance2/pair_scale.square().clamp_min(1e-6); normal_dot=(normals[:,:,None,:]*normals[:,None,:,:]).sum(-1,keepdim=True)*pair_float;tangent_dot=(tangents[:,:,None,:]*tangents[:,None,:,:]).sum(-1,keepdim=True)*pair_float
        geometry=[nd2,normal_dot,tangent_dot] if self.full_frame_orientation else [nd2,normal_dot]
        pair_features=torch.cat([left,right,messages,*geometry],-1); connection_logits=self.connection_head(pair_features); connection_logits=.5*(connection_logits+connection_logits.transpose(1,2))
        output={'center_noise':centers-initial_centers,'normal_noise':normals-initial_normals,'tangent_noise':tangents-initial_tangents,'ring_logits':self.ring_head(hidden),'connection_logits':connection_logits,'port_logits':self.port_head(pair_features),'junction_port_logits':self.junction_port_head(pair_features),'hidden':hidden,'centers':centers,'normals':normals,'tangents':tangents,'physical_radius':physical_radius}
        if self.property_head is not None:
            denom=node_mask.sum(1,keepdim=True).clamp_min(1).to(hidden.dtype)
            output['property_pred']=self.property_head((hidden*node_mask.unsqueeze(-1)).sum(1)/denom)
        return output
