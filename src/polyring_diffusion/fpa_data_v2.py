"""Gauge-augmented FPA entity data for the v2 diffusion model."""
from __future__ import annotations
import math
import torch
from .fpa_data import FPAEntityDataset, collate_fpa


class FPAEntityDatasetV2(FPAEntityDataset):
    """Adds an orthonormal frame; ports are represented by shared membership."""
    def __init__(self,*args,frame_augmentation=True,**kwargs):super().__init__(*args,**kwargs);self.frame_augmentation=frame_augmentation
    def __getitem__(self,index):
        out=super().__getitem__(index);n=len(out['cell_types'])
        if not n:out['cell_frames']=out['cell_centers'].new_zeros((0,3,3));return out
        normal=out['cell_normals'];tangent=out['cell_tangents'];bitangent=torch.linalg.cross(normal,tangent,dim=-1)
        frames=torch.stack((tangent,bitangent,normal),-2)
        if self.frame_augmentation:
            # D_n gauge augmentation: rotations/reflections encode the same ring.
            for i,size in enumerate(out['cell_sizes'].tolist()):
                k=int(torch.randint(max(1,size),(1,)));angle=2*math.pi*k/max(1,size);c,s=math.cos(angle),math.sin(angle)
                gauge=frames.new_tensor([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
                if bool(torch.randint(2,(1,))):gauge=frames.new_tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])@gauge
                frames[i]=gauge@frames[i]
        out['cell_frames']=frames;return out


def collate_fpa_v2(batch):
    out=collate_fpa(batch);b,n=out['cell_mask'].shape;out['cell_frames']=torch.eye(3).reshape(1,1,3,3).repeat(b,n,1,1)
    for i,x in enumerate(batch):out['cell_frames'][i,:len(x['cell_types'])]=x['cell_frames']
    return out
