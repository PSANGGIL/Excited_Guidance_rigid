import torch
from joint_kekule_diffusion.model import ConditionEmbedding,JointDiffusion,JointKekuleDDPM,aromatic_cycle_loss,_cycle_log_partition
from joint_kekule_diffusion.sample import sample_batch,chemistry_projection,reverse_coords

def test_bond_corruption_is_symmetric():
 d=JointDiffusion(20);x=torch.randint(4,(3,8,8));x=torch.triu(x,1);x=x+x.transpose(1,2);valid=torch.ones_like(x,dtype=torch.bool);valid.diagonal(dim1=1,dim2=2).zero_();y=d.q_token(x,torch.tensor([3,9,19]),4,valid,True);assert torch.equal(y,y.transpose(1,2));assert (y.diagonal(dim1=1,dim2=2)==0).all()

def _cycle_loss(doubles):
 n=8;cy=torch.zeros(1,1,n,n,dtype=torch.bool);ae=torch.zeros(1,n,n,dtype=torch.bool);order=torch.full((1,1,n),-1,dtype=torch.long);order[0,0,:6]=torch.arange(6)
 z=torch.full((1,n,n,4),-8.);z[...,0]=0
 for i in range(6):
  j=(i+1)%6;cy[0,0,i,j]=cy[0,0,j,i]=1;ae[0,i,j]=ae[0,j,i]=1;k=2 if i in doubles else 1;z[0,i,j]=z[0,j,i]=-8;z[0,i,j,k]=z[0,j,i,k]=8
 return aromatic_cycle_loss(z,torch.zeros(1,n,n),cy,torch.tensor([[3.]]),torch.ones(1,1,dtype=torch.bool),order,torch.tensor([[6]]),ae,torch.ones(1,n,n,dtype=torch.bool))

def test_kekule_marginal_rejects_adjacent_double_bonds():
 assert _cycle_loss({0,2,4})+4<_cycle_loss({0,1,2})

def test_padding_cannot_change_real_outputs():
 torch.manual_seed(3);m=JointKekuleDDPM(32,12,4,2,0).eval();mask=torch.zeros(1,12,dtype=torch.bool);mask[:,:6]=1;s={'atom_mask':mask,'atom_type':torch.randint(4,(1,12)),'charge':torch.randint(5,(1,12)),'hydrogen':torch.randint(5,(1,12)),'radical':torch.randint(3,(1,12)),'edge_exists':torch.randint(2,(1,12,12)),'bond_type':torch.randint(3,(1,12,12)),'positions':torch.randn(1,12,3)};s['edge_exists']=(s['edge_exists']+s['edge_exists'].transpose(1,2))//2;s['bond_type']=(s['bond_type']+s['bond_type'].transpose(1,2))//2;y=torch.zeros(1,2);t=torch.tensor([7]);a=m(y,t,s);s2={k:v.clone() for k,v in s.items()};s2['positions'][:,6:]=1e4;s2['atom_type'][:,6:]=torch.randint(4,(1,6));s2['edge_exists'][:,6:]=torch.randint(2,(1,6,12));s2['edge_exists'][:,:,6:]=torch.randint(2,(1,12,6));s2['bond_type'][:,6:]=torch.randint(3,(1,6,12));s2['bond_type'][:,:,6:]=torch.randint(3,(1,12,6));b=m(y,t,s2)
 for key in ('atom_logits','charge_logits','hydrogen_logits','radical_logits','coordinate_eps'):assert torch.equal(a[key][:,:6],b[key][:,:6])
 assert torch.equal(a['bond_logits'][:,:6,:6],b['bond_logits'][:,:6,:6]);assert torch.equal(a['count_logits'],b['count_logits'])

def test_sampler_rejects_invalid_counts_and_cfg():
 m=JointKekuleDDPM(32,12,4,2,0).eval();d=JointDiffusion(3);y=torch.zeros(1,2)
 for atoms,cfg in ((-1,1),(13,1),(6,float('nan'))):
  try:sample_batch(m,d,y,1,atoms,cfg)
  except ValueError:continue
  raise AssertionError((atoms,cfg))


def test_scalar_conditions_use_separate_configurable_rbfs():
 e=ConditionEmbedding(32,s1_rbf_bins=7,t1_rbf_bins=11,rbf_min=-3,rbf_max=4,rbf_width_scale=.75)
 assert e.net[0].in_features==18
 x=torch.tensor([[0.,0.],[1.,0.],[0.,1.]],requires_grad=True);y=e(x)
 assert y.shape==(3,32) and torch.isfinite(y).all()
 assert not torch.allclose(y[0],y[1]) and not torch.allclose(y[0],y[2])
 y.square().mean().backward();assert torch.isfinite(x.grad).all()


def test_cycle_dp_matches_enumeration_and_has_finite_gradients():
 import itertools
 torch.manual_seed(21);length=8;doubles=3;lp=torch.randn(2,length,4,requires_grad=True);got=_cycle_log_partition(lp,doubles)
 refs=[]
 for b in range(2):
  scores=[sum(lp[b,i,bits[i]+1] for i in range(length)) for bits in itertools.product((0,1),repeat=length) if sum(bits)==doubles and all(not(bits[i] and bits[(i+1)%length]) for i in range(length))]
  refs.append(torch.logsumexp(torch.stack(scores),0))
 assert torch.allclose(got,torch.stack(refs),atol=2e-6);got.sum().backward();assert torch.isfinite(lp.grad).all()

def test_spanning_projection_upgrades_without_double_counting_valence():
 mask=torch.ones(1,2,dtype=torch.bool);state={"edge_exists":torch.tensor([[[0,1],[1,0]]]),"bond_type":torch.tensor([[[0,2],[2,0]]]),"atom_type":torch.zeros(1,2,dtype=torch.long),"charge":torch.full((1,2),2),"hydrogen":torch.ones(1,2,dtype=torch.long)};outputs={"edge_logits":torch.tensor([[[[8.,-8.],[-8.,8.]],[[-8.,8.],[8.,-8.]]]]),"bond_type_logits":torch.zeros(1,2,2,3),"aromatic_edge_logits":torch.full((1,2,2),-8.)};outputs["bond_type_logits"][...,2]=8.;result=chemistry_projection(state,outputs,mask,[4],True,False,["C"]);bonds=result["edge_exists"]*(result["bond_type"]+1);assert int(bonds[0,0,1])==3 and int(result["hydrogen"][0,0])==1

def test_reverse_coordinate_clip_is_finite_and_centered():
 diffusion=JointDiffusion(20);mask=torch.ones(2,7,dtype=torch.bool);x=torch.randn(2,7,3)*1e6;eps=torch.randn_like(x)*1e6;out=reverse_coords(x,eps,19,diffusion,mask,25.);assert torch.isfinite(out).all();assert out.norm(dim=-1).max()<=25.0001;assert out.mean(1).abs().max()<1e-5
