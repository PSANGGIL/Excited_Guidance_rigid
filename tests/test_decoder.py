import unittest

from polyring_diffusion.decoder import DecodeError, decode_ring_graph
from polyring_diffusion.graph import ConnectionType, RingConnection, RingGraph, RingNode


class DecoderTests(unittest.TestCase):
    def test_fused_benzenes_share_exactly_one_edge(self) -> None:
        graph = RingGraph(
            nodes=[RingNode("benzene"), RingNode("benzene")],
            connections=[RingConnection(0, 1, ConnectionType.FUSED, 0, 0)],
        )

        atoms = decode_ring_graph(graph)

        self.assertEqual(len(atoms.atoms), 10)
        self.assertEqual(len(atoms.bonds), 11)
        self.assertEqual(sum(len(atom.ring_membership) == 2 for atom in atoms.atoms), 2)


    def test_single_link_does_not_merge_ring_atoms(self) -> None:
        graph = RingGraph(
            nodes=[RingNode("benzene"), RingNode("benzene")],
            connections=[RingConnection(0, 1, ConnectionType.SINGLE, 0, 0)],
        )

        atoms = decode_ring_graph(graph)

        self.assertEqual(len(atoms.atoms), 12)
        self.assertEqual(len(atoms.bonds), 13)


    def test_spiro_link_shares_one_atom(self) -> None:
        graph = RingGraph(
            nodes=[RingNode("benzene"), RingNode("benzene")],
            connections=[RingConnection(0, 1, ConnectionType.SPIRO, 0, 0)],
        )

        atoms = decode_ring_graph(graph)

        self.assertEqual(len(atoms.atoms), 11)
        self.assertEqual(len(atoms.bonds), 12)
        self.assertEqual(sum(len(atom.ring_membership) == 2 for atom in atoms.atoms), 1)


    def test_incompatible_heteroatom_fusion_is_rejected(self) -> None:
        graph = RingGraph(
            nodes=[RingNode("pyridine"), RingNode("benzene")],
            # Pyridine edge 0 is N-C, while every benzene edge is C-C.
            connections=[RingConnection(0, 1, ConnectionType.FUSED, 0, 0)],
        )

        with self.assertRaisesRegex(DecodeError, "matching atom types"):
            decode_ring_graph(graph)


if __name__ == "__main__":
    unittest.main()
