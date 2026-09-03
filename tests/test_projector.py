import unittest

from polyring_diffusion.decoder import DecodeError, decode_ring_graph
from polyring_diffusion.graph import ConnectionType, RingConnection, RingNode
from polyring_diffusion.projector import ConnectionCandidate, RingConstraintProjector


def candidate(score: float, left: int, right: int, left_port: int, right_port: int):
    return ConnectionCandidate(
        score,
        RingConnection(left, right, ConnectionType.SINGLE, left_port, right_port),
    )


class ProjectorTests(unittest.TestCase):
    def test_projector_builds_tree_then_one_closure(self) -> None:
        nodes = [RingNode("benzene") for _ in range(3)]
        candidates = [
            candidate(10.0, 0, 1, 0, 0),
            candidate(9.0, 1, 2, 1, 0),
            candidate(8.0, 0, 2, 1, 1),
        ]

        graph = RingConstraintProjector(max_extra_fusions=1).project(nodes, candidates)

        self.assertTrue(graph.is_connected())
        self.assertEqual(len(graph.connections), 3)
        self.assertEqual(graph.cyclomatic_number, 1)
        decode_ring_graph(graph)


    def test_projector_reports_disconnected_candidate_set(self) -> None:
        nodes = [RingNode("benzene") for _ in range(3)]
        candidates = [candidate(10.0, 0, 1, 0, 0)]

        with self.assertRaisesRegex(DecodeError, "connected"):
            RingConstraintProjector().project(nodes, candidates)


if __name__ == "__main__":
    unittest.main()
