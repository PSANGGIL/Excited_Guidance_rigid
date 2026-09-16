import torch
from joint_kekule_diffusion.model import JointDiffusion,JointKekuleDDPM,aromatic_cycle_loss
from joint_kekule_diffusion.sample import sample_batch

def test_bond_corruption_is_symmetric():
 d=JointDiffusion(20);x=torch.randint(4,(3,8,8));x=torch.triu(x,1);x=x+x.transpose(1,2);valid=torch.ones_like(x,dtype=torch.bool);valid.diagonal(dim1=1,dim2=2).zero_();y=d.q_bond_token(x,torch.tensor([3,9,19]),4,valid);assert torch.equal(y,y.transpose(1,2));assert not (y.diagonal(dim1=1,dim2=2)==4).any()

def _cycle_loss(doubles):
 n=8;cy=torch.zeros(1,1,n,n,dtype=torch.bool);ae=torch.zeros(1,n,n,dtype=torch.bool);order=torch.full((1,1,n),-1,dtype=torch.long);order[0,0,:6]=torch.arange(6)
 z=torch.full((1,n,n,4),-8.);z[...,0]=0
 for i in range(6):
  j=(i+1)%6;cy[0,0,i,j]=cy[0,0,j,i]=1;ae[0,i,j]=ae[0,j,i]=1;k=2 if i in doubles else 1;z[0,i,j]=z[0,j,i]=-8;z[0,i,j,k]=z[0,j,i,k]=8
 return aromatic_cycle_loss(z,torch.zeros(1,n,n),cy,torch.tensor([[3.]]),torch.ones(1,1,dtype=torch.bool),order,torch.tensor([[6]]),ae,torch.ones(1,n,n,dtype=torch.bool))

def test_kekule_marginal_rejects_adjacent_double_bonds():
 assert _cycle_loss({0,2,4})+4<_cycle_loss({0,1,2})

def test_padding_cannot_change_real_outputs():
 torch.manual_seed(3);m=JointKekuleDDPM(32,12,4,2,0).eval();mask=torch.zeros(1,12,dtype=torch.bool);mask[:,:6]=1;s={'atom_mask':mask,'atom_type':torch.randint(5,(1,12)),'charge':torch.randint(6,(1,12)),'hydrogen':torch.randint(6,(1,12)),'radical':torch.randint(4,(1,12)),'bond':torch.randint(5,(1,12,12)),'positions':torch.randn(1,12,3)};s['bond']=(s['bond']+s['bond'].transpose(1,2))//2;y=torch.zeros(1,2);t=torch.tensor([7]);a=m(y,t,s);s2={k:v.clone() for k,v in s.items()};s2['positions'][:,6:]=1e4;s2['atom_type'][:,6:]=torch.randint(5,(1,6));s2['bond'][:,6:]=torch.randint(5,(1,6,12));s2['bond'][:,:,6:]=torch.randint(5,(1,12,6));b=m(y,t,s2)
 for key in ('atom_logits','charge_logits','hydrogen_logits','radical_logits','coordinate_eps'):assert torch.equal(a[key][:,:6],b[key][:,:6])
 assert torch.equal(a['bond_logits'][:,:6,:6],b['bond_logits'][:,:6,:6]);assert torch.equal(a['count_logits'],b['count_logits'])

def test_sampler_rejects_invalid_counts_and_cfg():
 m=JointKekuleDDPM(32,12,4,2,0).eval();d=JointDiffusion(3);y=torch.zeros(1,2)
 for atoms,cfg in ((0,1),(13,1),(6,float('nan'))):
  try:sample_batch(m,d,y,1,atoms,cfg)
  except ValueError:continue
  raise AssertionError((atoms,cfg))
