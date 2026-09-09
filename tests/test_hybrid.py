import unittest
import torch
from polyring_diffusion.hybrid_data import collate_hybrid
from polyring_diffusion.hybrid_model import HybridRingAtomDenoiser

class HybridTests(unittest.TestCase):
 def record(self,n=2):
  return {'ring_types':torch.tensor([0]),'centers':torch.zeros(1,3),'normals':torch.tensor([[0.,0.,1.]]),'radii':torch.ones(1),'connection_types':torch.zeros(1,1,dtype=torch.long),'ports':torch.zeros(1,1,dtype=torch.long),'orientation':torch.zeros(1,1,dtype=torch.long),'property':torch.tensor([.2]),'raw_property':torch.tensor([.5]),'source_row':torch.tensor(1),'free_atom_types':torch.tensor([2,8][:n]),'free_atom_charges':torch.zeros(n,dtype=torch.long),'free_positions':torch.randn(n,3),'free_bond_types':torch.tensor([[0,1],[1,0]],dtype=torch.long)[:n,:n],'attachment_types':torch.tensor([[1,0]],dtype=torch.long)[:,:n],'attachment_ports':torch.tensor([[2,0]],dtype=torch.long)[:,:n]}
 def test_collate_preserves_free_atoms_and_attachment(self):
  b=collate_hybrid([self.record(2),self.record(1)]);self.assertEqual(tuple(b['free_atom_types'].shape),(2,2));self.assertEqual(b['free_atom_count'].tolist(),[2,1]);self.assertEqual(int(b['attachment_types'][0,0,0]),1);self.assertFalse(bool(b['free_atom_mask'][1,1]))
 def test_joint_forward_and_gradients(self):
  b=collate_hybrid([self.record(2),self.record(1)]);m=HybridRingAtomDenoiser(num_atom_types=9,max_free_atoms=8,atom_hidden_dim=16,atom_layers=1,hidden_dim=16,num_layers=1,time_dim=8,max_ports=8,condition_dim=1,property_head=True);t=torch.zeros(2,dtype=torch.long);q=m.charge_index(b['free_atom_charges']);o=m((b['ring_types'],b['centers'],b['normals'],b['connection_types'],b['node_mask']),b['free_atom_types'],q,b['free_positions'],b['free_bond_types'],b['attachment_types'],b['free_atom_mask'],t,b['property']);self.assertEqual(tuple(o['atom_type_logits'].shape),(2,2,9));self.assertEqual(tuple(o['attachment_logits'].shape),(2,1,2,5));self.assertEqual(tuple(o['free_atom_count_logits'].shape),(2,9));(o['atom_type_logits'].sum()+o['attachment_logits'].sum()).backward();self.assertIsNotNone(m.atom_type_head.weight.grad)
if __name__=='__main__':unittest.main()
