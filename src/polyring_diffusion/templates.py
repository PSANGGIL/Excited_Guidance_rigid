"""Canonical aromatic ring templates with indexed vertex and edge ports."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

from .chem import element_record


@dataclass(frozen=True)
class RingTemplate:
    name: str
    atoms: tuple[str, ...]
    formal_charges: tuple[int, ...] = ()
    aromatic: bool = True

    def __post_init__(self) -> None:
        if len(self.atoms) < 3:
            raise ValueError("A ring template requires at least three atoms")
        if self.formal_charges and len(self.formal_charges) != len(self.atoms):
            raise ValueError("formal_charges must match the number of atoms")
        for symbol in self.atoms:
            element_record(symbol)

    @property
    def size(self) -> int:
        return len(self.atoms)

    @property
    def charges(self) -> tuple[int, ...]:
        return self.formal_charges or (0,) * self.size

    @property
    def vertex_ports(self) -> range:
        return range(self.size)

    @property
    def edge_ports(self) -> range:
        return range(self.size)

    def edge_atoms(self, port: int) -> tuple[int, int]:
        self.check_edge_port(port)
        return port, (port + 1) % self.size

    def check_vertex_port(self, port: int) -> None:
        if port < 0 or port >= self.size:
            raise ValueError(f"Vertex port {port} is invalid for {self.name}")

    def check_edge_port(self, port: int) -> None:
        if port < 0 or port >= self.size:
            raise ValueError(f"Edge port {port} is invalid for {self.name}")

    def compatible_edge_ports(self, other: "RingTemplate") -> list[tuple[int, int]]:
        """Return fusion ports whose reversed atom types can be merged."""

        pairs: list[tuple[int, int]] = []
        for left in self.edge_ports:
            la, lb = self.edge_atoms(left)
            for right in other.edge_ports:
                ra, rb = other.edge_atoms(right)
                if self.atoms[la] == other.atoms[rb] and self.atoms[lb] == other.atoms[ra]:
                    pairs.append((left, right))
        return pairs

    def ideal_coordinates(self, bond_length: float = 1.40) -> tuple[tuple[float, float, float], ...]:
        """Planar regular polygon used only as a deterministic decode seed."""

        radius = bond_length / (2.0 * sin(pi / self.size))
        return tuple(
            (radius * cos(2.0 * pi * i / self.size), radius * sin(2.0 * pi * i / self.size), 0.0)
            for i in range(self.size)
        )


class TemplateLibrary:
    def __init__(self, templates: Iterable[RingTemplate]) -> None:
        items = tuple(templates)
        self._by_name = {template.name: template for template in items}
        if len(self._by_name) != len(items):
            raise ValueError("Ring template names must be unique")

    def __len__(self) -> int:
        return len(self._by_name)

    def __iter__(self):
        return iter(self._by_name.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def get(self, name: str) -> RingTemplate:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ValueError(f"Unknown ring template: {name}") from exc


DEFAULT_LIBRARY = TemplateLibrary(
    [
        RingTemplate("benzene", ("C",) * 6),
        RingTemplate("pyridine", ("N", "C", "C", "C", "C", "C")),
        RingTemplate("pyrazine", ("N", "C", "C", "N", "C", "C")),
        RingTemplate("pyrrole", ("N", "C", "C", "C", "C")),
        RingTemplate("furan", ("O", "C", "C", "C", "C")),
        RingTemplate("thiophene", ("S", "C", "C", "C", "C")),
        RingTemplate("borinine", ("B", "C", "C", "C", "C", "C")),
    ]
)

