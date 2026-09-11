"""Constrained projection from noisy FPA predictions to a consistent molecule."""
from __future__ import annotations
import math
import torch
from .chem import element_record


def _components(n, edges):
    parent=list(range(n))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    for a,b in edges:union(a,b)
    return parent,find,union


@torch.no_grad()
def project_fpa_batch(sample:dict,prediction:dict,atom_map:list[str]|tuple[str,...]):
    """Project membership, geometry and bonds; returns per-sample diagnostics."""
    b=len(sample['atom_types']);valid=[];repairs=[];reasons=[]
    mscore=prediction['membership_logits'].softmax(-1)[...,1];bprob=prediction['bond_logits'].softmax(-1)
    for row in range(b):
        na=int(sample['atom_mask'][row].sum());nc=int(sample['cell_mask'][row].sum());changed=0;reason=2 if nc and na<3 else 0
        pos=sample['positions'][row,:na];centers=sample['cell_centers'][row,:nc];frames=sample['cell_frames'][row,:nc];radii=sample['cell_radii'][row,:nc]
        fusion=sample['fusion'][row,:nc,:nc];sizes=sample['cell_sizes'][row,:nc].clamp(1,na);score=mscore[row,:nc,:na]
        chosen=[set(torch.topk(score[c],int(sizes[c])).indices.tolist()) for c in range(nc)]
        required=[set() for _ in range(nc)]
        for left in range(nc):
            for right in range(left+1,nc):
                if int(fusion[left,right])!=1:continue
                pair=torch.topk(score[left]+score[right],min(2,na)).indices.tolist();required[left].update(pair);required[right].update(pair)
        for c in range(nc):
            k=int(sizes[c]);pool=required[c]|chosen[c]
            ranked=sorted(pool-required[c],key=lambda a:float(score[c,a]),reverse=True);chosen[c]=set(required[c])|set(ranked[:max(0,k-len(required[c]))])
            if len(chosen[c])>k:reason|=1
        # Non-fused cells may share at most one atom.
        for left in range(nc):
            for right in range(left+1,nc):
                if int(fusion[left,right])==1:continue
                overlap=chosen[left]&chosen[right]
                for atom in list(overlap)[1:]:
                    loser=left if float(score[left,atom])<float(score[right,atom]) else right
                    if atom not in required[loser]:chosen[loser].remove(atom);changed+=1
        member=torch.zeros((nc,na),dtype=torch.long,device=pos.device)
        for c,atoms in enumerate(chosen):
            if atoms:member[c,torch.tensor(sorted(atoms),device=pos.device)]=1
        sample['membership'][row,:nc,:na]=member
        # Alternating projections onto all incident ring disks.
        for _ in range(3):
            targets=[[] for _ in range(na)]
            for c,atoms in enumerate(chosen):
                tangent,bitangent,normal=frames[c,0],frames[c,1],frames[c,2]
                for slot,atom in enumerate(sorted(atoms)):
                    rel=pos[atom]-centers[c];u=(rel*tangent).sum();v=(rel*bitangent).sum();norm=torch.sqrt(u*u+v*v)
                    if float(norm)<1e-5:angle=2*math.pi*slot/max(1,len(atoms));direction=math.cos(angle)*tangent+math.sin(angle)*bitangent
                    else:direction=(u*tangent+v*bitangent)/norm
                    targets[atom].append(centers[c]+radii[c]*direction)
            for atom,values in enumerate(targets):
                if values:pos[atom]=torch.stack(values).mean(0);changed+=1
        sample['positions'][row,:na]=pos
        # Ring cycles and fused shared edges are mandatory aromatic bonds.
        mandatory=set()
        for c,atoms in enumerate(chosen):
            atoms=list(atoms)
            if len(atoms)<3:reason|=2;continue
            tangent,bitangent=frames[c,0],frames[c,1];atoms.sort(key=lambda a:math.atan2(float(((pos[a]-centers[c])*bitangent).sum()),float(((pos[a]-centers[c])*tangent).sum())))
            mandatory.update(tuple(sorted((atoms[i],atoms[(i+1)%len(atoms)]))) for i in range(len(atoms)))
        projected=torch.zeros((na,na),dtype=torch.long,device=pos.device);used=torch.zeros(na,device=pos.device)
        limits=torch.tensor([max(float(element_record(atom_map[int(t)]).default_valence),4.5 if atom_map[int(t)] in ('C','N') else 0.) for t in sample['atom_types'][row,:na]],device=pos.device)
        edges=[]
        for i,j in mandatory:projected[i,j]=projected[j,i]=4;used[i]+=1.5;used[j]+=1.5;edges.append((i,j))
        parent,find,union=_components(na,edges)
        for i,j in edges:union(i,j)
        candidates=[]
        for i in range(na):
            for j in range(i+1,na):
                if (i,j) in mandatory:continue
                kind=int(torch.argmax(bprob[row,i,j]));order=(0.,1.,2.,3.,1.5)[kind];candidates.append((float(1-bprob[row,i,j,0]),i,j,kind,order))
        candidates.sort(reverse=True)
        # First connect components, then add other confident bonds.
        for connect_only in (True,False):
            for confidence,i,j,kind,order in candidates:
                if kind==0 or (connect_only and find(i)==find(j)) or (not connect_only and confidence<.5):continue
                if used[i]+order>limits[i] or used[j]+order>limits[j]:continue
                projected[i,j]=projected[j,i]=kind;used[i]+=order;used[j]+=order;union(i,j);changed+=1
        if len({find(i) for i in range(na)})>1:reason|=4
        if bool((used>limits+1e-5).any()):reason|=8
        sample['bonds'][row,:na,:na]=projected;valid.append(reason==0);repairs.append(changed);reasons.append(reason)
    sample['projection_valid']=torch.tensor(valid,device=sample['atom_types'].device);sample['projection_repairs']=torch.tensor(repairs,device=sample['atom_types'].device);sample['projection_reason']=torch.tensor(reasons,device=sample['atom_types'].device)
    return sample
