"""Small chemistry tables used as explicit physical inductive biases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementRecord:
    atomic_number: int
    vdw_radius: float
    covalent_radius: float
    default_valence: int
    electronegativity: float


# Radii are compact, model-facing defaults in angstrom. Production work should
# use one consistent parameterization throughout preprocessing and evaluation.
ELEMENTS: dict[str, ElementRecord] = {
    "B": ElementRecord(5, 1.92, 0.84, 3, 2.04),
    "C": ElementRecord(6, 1.70, 0.76, 4, 2.55),
    "N": ElementRecord(7, 1.55, 0.71, 3, 3.04),
    "O": ElementRecord(8, 1.52, 0.66, 2, 3.44),
    "F": ElementRecord(9, 1.47, 0.57, 1, 3.98),
    "P": ElementRecord(15, 1.80, 1.07, 3, 2.19),
    "S": ElementRecord(16, 1.80, 1.05, 2, 2.58),
    "Cl": ElementRecord(17, 1.75, 1.02, 1, 3.16),
}


def element_record(symbol: str) -> ElementRecord:
    try:
        return ELEMENTS[symbol]
    except KeyError as exc:
        raise ValueError(f"Unsupported element: {symbol}") from exc


def normalized_atom_features(symbol: str, formal_charge: int = 0) -> tuple[float, ...]:
    """Return compact scalar atom features suitable for an equivariant GNN."""

    record = element_record(symbol)
    return (
        record.atomic_number / 20.0,
        float(formal_charge) / 3.0,
        record.vdw_radius / 2.0,
        record.covalent_radius / 1.2,
        record.default_valence / 6.0,
        record.electronegativity / 4.0,
    )

