"""Differentiable chemical constraints for ring connection and port predictions."""
from __future__ import annotations
from collections import defaultdict
import torch
from torch import Tensor,nn

_DEFAULT_VALENCE={5:3.0,6:4.0,7:3.0,8:2.0,9:1.0,15:5.0,16:6.0,17:1.0,35:1.0}

def _bond_order(label:int)->float:
    return {1:1.0,2:2.0,3:3.0,4:1.5}.get(int(label),0.0)

class ChemicalConstraintLoss(nn.Module):
    """Expected invalid-port, valence/port-reuse, and fused-Kekule penalties."""
    def __init__(self,signatures:list[list[list[int]]],max_ports:int=8):
        super().__init__();t=len(signatures); valid=torch.zeros(t+2,max_ports); capacity=torch.zeros(t+2,max_ports); incidence=torch.zeros(t+2,max_ports,max_ports); edge_keys=[]
        raw_edges=[]
        for ti,sig in enumerate(signatures):
            n=len(sig)
            for p,row in enumerate(sig):
                if p>=max_ports:raise ValueError(f'template {ti} has {n} ports > max_ports={max_ports}')
                valid[ti,p]=1;incidence[ti,p,p]=1;incidence[ti,p,(p+1)%n]=1
                z,charge=int(row[0]),int(row[1]); internal=_bond_order(sig[p-1][3])+_bond_order(row[3]); vmax=_DEFAULT_VALENCE.get(z,4.0)+max(charge,0)
                # Aromatic valence conventions are fractional; every valid ring atom
                # still receives at least one external/fusion attachment slot.
                capacity[ti,p]=max(1.0,vmax-internal)
                a=(int(row[0]),int(row[1])); b=(int(sig[(p+1)%n][0]),int(sig[(p+1)%n][1])); key=(min(a,b),max(a,b),int(row[3]));edge_keys.append(key);raw_edges.append((ti,p,key))
        key_to_id={k:i for i,k in enumerate(sorted(set(edge_keys)))}; edge=torch.zeros(t+2,max_ports,len(key_to_id))
        for ti,p,key in raw_edges:edge[ti,p,key_to_id[key]]=1
        self.register_buffer('valid_ports',valid);self.register_buffer('valence_capacity',capacity);self.register_buffer('edge_classes',edge);self.register_buffer('edge_incidence',incidence)

    def forward(self,connection_logits:Tensor,port_logits:Tensor,ring_types:Tensor,node_mask:Tensor)->dict[str,Tensor]:
        b,n,_,p=port_logits.shape; pair=node_mask[:,:,None]&node_mask[:,None,:];pair&=~torch.eye(n,dtype=torch.bool,device=node_mask.device)[None];pf=pair.unsqueeze(-1).to(port_logits.dtype)
        conn=connection_logits.softmax(-1); raw_port=port_logits.softmax(-1); valid=self.valid_ports[ring_types]
        valid_dir=valid[:,:,None,:].expand(-1,-1,n,-1); invalid_mass=(raw_port*(1-valid_dir)*pf).sum()/pf.sum().clamp_min(1)
        port=raw_port*valid_dir;port=port/port.sum(-1,keepdim=True).clamp_min(1e-8);port=port*pf
        vertex_prob=(conn[...,1]+conn[...,3]).unsqueeze(-1)*port
        fused_start=conn[...,2].unsqueeze(-1)*port
        # Edge p consumes vertices p and p+1; roll by one maps previous edge to vertex p.
        fused_vertex=torch.einsum('bijp,bipv->bijv',fused_start,self.edge_incidence[ring_types])
        occupancy=(vertex_prob+fused_vertex).sum(2)
        reuse=torch.relu(occupancy-1).square();reuse=(reuse*valid).sum()/valid.sum().clamp_min(1)
        # Spiro merges two ring atoms and is more demanding than a single attachment.
        valence_use=((conn[...,1]+2*conn[...,3]).unsqueeze(-1)*port+fused_vertex).sum(2)
        overflow=torch.relu(valence_use-self.valence_capacity[ring_types]).square();valence=(overflow*valid).sum()/valid.sum().clamp_min(1)
        edge_table=self.edge_classes[ring_types];edge_dist=torch.einsum('bijp,bipc->bijc',port,edge_table);compatible=(edge_dist*edge_dist.transpose(1,2)).sum(-1)
        upper=pair&torch.triu(torch.ones(n,n,dtype=torch.bool,device=node_mask.device),1)[None]
        kekule=(conn[...,2]*(1-compatible))[upper].mean() if upper.any() else conn.new_zeros(())
        return {'valence':valence,'kekule':kekule,'port_validity':invalid_mass+reuse,'port_invalid_mass':invalid_mass.detach(),'port_reuse':reuse.detach()}
