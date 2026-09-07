"""Loader for the sharded hierarchical rigid-ring database."""
from __future__ import annotations
import bisect, json
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset

class RigidRingDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, root: str|Path, split: str, property_mean: float|None=None, property_std: float|None=None):
        self.root=Path(root); self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.template_hash_to_id={int(x['hash']):int(x['id']) for x in self.manifest['templates']}
        info=next(x for x in self.manifest['splits'] if x['name']==split)
        self.shards=[]; self.ends=[]; total=0; all_props=[]
        radius_sum=torch.zeros(len(self.template_hash_to_id)); radius_count=torch.zeros_like(radius_sum)
        for rel in info['shards']:
            raw=torch.load(self.root/rel,map_location='cpu',weights_only=False)
            hashes=raw['ring_template_hashes']; ids=torch.empty_like(hashes)
            for h in torch.unique(hashes).tolist(): ids[hashes==h]=self.template_hash_to_id[int(h)]
            shard={k:raw[k] for k in ('ring_centers','ring_normals','ring_radii','ring_offsets','connections','connection_offsets','properties','raw_properties','source_rows')}
            shard['ring_types']=ids; n=len(raw['smiles']); shard['length']=n
            self.shards.append(shard); total+=n; self.ends.append(total); all_props.append(raw['properties'].float())
            radius_sum.scatter_add_(0,ids,raw['ring_radii'].float()); radius_count.scatter_add_(0,ids,torch.ones_like(raw['ring_radii'].float()))
        props=torch.cat(all_props)
        self.property_mean=float(props.mean()) if property_mean is None else float(property_mean)
        self.property_std=float(props.std().clamp_min(1e-8)) if property_std is None else float(property_std)
        self.template_features=self._features(radius_sum/radius_count.clamp_min(1))
        self.max_rings=max(int((s['ring_offsets'][1:]-s['ring_offsets'][:-1]).max()) for s in self.shards)

    def _features(self, mean_radii: Tensor)->Tensor:
        features=[]
        by_id=sorted(self.manifest['templates'],key=lambda x:x['id'])
        for item in by_id:
            sig=item['signature']; z=[row[0] for row in sig]
            features.append([len(sig)/8.0,float(mean_radii[item['id']])/3.0,sum(v!=6 for v in z)/len(z),sum(row[2] for row in sig)/len(sig)])
        return torch.tensor(features,dtype=torch.float32)

    def __len__(self): return self.ends[-1]
    def __getitem__(self,index):
        si=bisect.bisect_right(self.ends,index); base=0 if si==0 else self.ends[si-1]; i=index-base; s=self.shards[si]
        rs,re=map(int,s['ring_offsets'][i:i+2]); es,ee=map(int,s['connection_offsets'][i:i+2]); n=re-rs
        conn=torch.zeros((n,n),dtype=torch.long); ports=torch.zeros((n,n),dtype=torch.long)
        orientation=torch.zeros((n,n),dtype=torch.long)
        for left,right,kind,lp,rp,reverse in s['connections'][es:ee].tolist():
            if kind>3: continue
            conn[left,right]=conn[right,left]=kind; ports[left,right]=lp; ports[right,left]=rp
            orientation[left,right]=orientation[right,left]=reverse
        return {'ring_types':s['ring_types'][rs:re].long(),'centers':s['ring_centers'][rs:re].float(),
          'normals':s['ring_normals'][rs:re].float(),'radii':s['ring_radii'][rs:re].float(),
          'connection_types':conn,'ports':ports,'orientation':orientation,
          'property':((s['properties'][i].float()-self.property_mean)/self.property_std).reshape(1),
          'raw_property':s['raw_properties'][i].float().reshape(1),'source_row':torch.tensor(int(s['source_rows'][i]))}

def collate_rigid(batch:list[dict[str,Tensor]])->dict[str,Tensor]:
    b=len(batch); n=max(len(x['ring_types']) for x in batch)
    out={'ring_types':torch.zeros((b,n),dtype=torch.long),'node_mask':torch.zeros((b,n),dtype=torch.bool),
      'centers':torch.zeros((b,n,3)),'normals':torch.zeros((b,n,3)),
      'connection_types':torch.zeros((b,n,n),dtype=torch.long),'ports':torch.zeros((b,n,n),dtype=torch.long),
      'orientation':torch.zeros((b,n,n),dtype=torch.long),
      'property':torch.stack([x['property'] for x in batch]),'raw_property':torch.stack([x['raw_property'] for x in batch]),
      'source_row':torch.stack([x['source_row'] for x in batch])}
    for i,x in enumerate(batch):
        m=len(x['ring_types']); out['ring_types'][i,:m]=x['ring_types']; out['node_mask'][i,:m]=True
        out['centers'][i,:m]=x['centers']-x['centers'].mean(0,keepdim=True); out['normals'][i,:m]=x['normals']
        for key in ('connection_types','ports','orientation'): out[key][i,:m,:m]=x[key]
    return out

class ValidityCurriculumSampler(torch.utils.data.Sampler[int]):
    """Sample easy valid assemblies first, then expand to the full training set."""
    def __init__(self,dataset:RigidRingDataset,fractions=(0.4,0.7,1.0),milestones=(0,10,20),seed=42):
        if len(fractions)!=len(milestones) or sorted(fractions)!=list(fractions) or fractions[-1]!=1.0:raise ValueError('fractions must increase and end at 1.0')
        self.dataset=dataset;self.fractions=tuple(float(x) for x in fractions);self.milestones=tuple(int(x) for x in milestones);self.seed=int(seed);self.epoch=0
        scores=[]
        for shard in dataset.shards:
            for i in range(shard['length']):
                rs,re=map(int,shard['ring_offsets'][i:i+2]);es,ee=map(int,shard['connection_offsets'][i:i+2]);kinds=shard['connections'][es:ee,2]
                score=(re-rs)+.5*(ee-es)+.25*int((kinds==2).sum())+1.0*int((kinds==3).sum())
                scores.append(score)
        self.ordered=torch.argsort(torch.tensor(scores),stable=True)
    def set_epoch(self,epoch:int):self.epoch=int(epoch)
    def fraction(self):
        result=self.fractions[0]
        for milestone,fraction in zip(self.milestones,self.fractions):
            if self.epoch>=milestone:result=fraction
        return result
    def __len__(self):return len(self.dataset)
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch);count=max(1,int(len(self.dataset)*self.fraction()));eligible=self.ordered[:count]
        if count==len(self.dataset):indices=eligible[torch.randperm(count,generator=g)]
        else:
            repeats=(len(self.dataset)+count-1)//count
            indices=torch.cat([eligible[torch.randperm(count,generator=g)] for _ in range(repeats)])[:len(self.dataset)]
        return iter(indices.tolist())
