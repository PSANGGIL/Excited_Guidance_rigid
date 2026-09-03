"""Hierarchical diffusion utilities for polycyclic molecular graphs.

The chemistry/data modules intentionally avoid importing PyTorch so that graph
decoding and validation can be used in lightweight preprocessing environments.
"""

from .decoder import AtomGraph, decode_ring_graph
from .graph import ConnectionType, RingConnection, RingGraph, RingNode
from .templates import DEFAULT_LIBRARY, RingTemplate, TemplateLibrary

__all__ = [
    "AtomGraph",
    "ConnectionType",
    "DEFAULT_LIBRARY",
    "RingConnection",
    "RingGraph",
    "RingNode",
    "RingTemplate",
    "TemplateLibrary",
    "decode_ring_graph",
]

