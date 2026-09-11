import math,unittest
import torch
from polyring_diffusion.fpa_projection import project_fpa_batch

class ProjectionTests(unittest.TestCase):
 def make(self,cells=1,atoms=6):
  angle=torch.arange(atoms)*2*math.pi/atoms;pos=torch.stack((angle.cos(),angle.sin(),torch.ones(atoms)*.3),-1)[None];sample={'atom_types':torch.zeros(1,atoms,dtype=torch.long),'charges':torch.full((1,atoms),2),'positions':pos,'bonds':torch.zeros(1,atoms,atoms,dtype=torch.long),'cell_types':torch.zeros(1,cells,dtype=torch.long),'cell_sizes':torch.full((1,cells),min(6,atoms)),'cell_centers':torch.zeros(1,cells,3),'cell_frames':torch.eye(3).reshape(1,1,3,3).repeat(1,cells,1,1),'cell_radii':torch.ones(1,cells),'fusion':torch.zeros(1,cells,cells,dtype=torch.long),'membership':torch.zeros(1,cells,atoms,dtype=torch.long),'atom_mask':torch.ones(1,atoms,dtype=torch.bool),'cell_mask':torch.ones(1,cells,dtype=torch.bool)};ml=torch.zeros(1,cells,atoms,2);ml[...,1]=5;bl=torch.zeros(1,atoms,atoms,5);bl[...,0]=3;return sample,{'membership_logits':ml,'bond_logits':bl}
 def test_single_cell_projects_cycle_and_plane(self):
  sample,pred=self.make();out=project_fpa_batch(sample,pred,['C']);self.assertEqual(int(out['membership'].sum()),6);self.assertEqual(int((out['bonds']==4).sum()),12);self.assertLess(float(out['positions'][0,:,2].abs().max()),1e-6);self.assertTrue(bool(out['projection_valid'][0]))
 def test_fused_cells_share_two_atoms(self):
  sample,pred=self.make(2,10);sample['fusion'][0,0,1]=sample['fusion'][0,1,0]=1;pred['membership_logits'][0,0,:,1]=torch.tensor([9,8,7,6,5,4,0,0,0,0]);pred['membership_logits'][0,1,:,1]=torch.tensor([9,8,0,0,0,0,7,6,5,4]);out=project_fpa_batch(sample,pred,['C']);shared=(out['membership'][0,0].bool()&out['membership'][0,1].bool());self.assertEqual(int(shared.sum()),2);self.assertEqual(int(out['membership'][0,0].sum()),6);self.assertEqual(int(out['membership'][0,1].sum()),6);ids=torch.where(shared)[0];self.assertEqual(int(out['bonds'][0,ids[0],ids[1]]),4)
if __name__=='__main__':unittest.main()
