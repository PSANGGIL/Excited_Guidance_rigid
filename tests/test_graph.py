import unittest

from polyring_diffusion.graph import ConnectionType, RingConnection, RingGraph, RingNode


class GraphTests(unittest.TestCase):
    def test_port_cannot_be_consumed_twice(self) -> None:
        graph = RingGraph(
            nodes=[RingNode("benzene") for _ in range(3)],
            connections=[
                RingConnection(0, 1, ConnectionType.FUSED, 0, 0),
                RingConnection(0, 2, ConnectionType.FUSED, 0, 1),
            ],
        )

        with self.assertRaisesRegex(ValueError, "cannot be used twice"):
            graph.validate()


    def test_ring_adjacency_cycle_rank(self) -> None:
        graph = RingGraph(
            nodes=[RingNode("benzene") for _ in range(3)],
            connections=[
                RingConnection(0, 1, ConnectionType.SINGLE, 0, 0),
                RingConnection(1, 2, ConnectionType.SINGLE, 1, 0),
                RingConnection(0, 2, ConnectionType.SINGLE, 1, 1),
            ],
        )

        graph.validate()
        self.assertTrue(graph.is_connected())
        self.assertEqual(graph.cyclomatic_number, 1)


if __name__ == "__main__":
    unittest.main()
