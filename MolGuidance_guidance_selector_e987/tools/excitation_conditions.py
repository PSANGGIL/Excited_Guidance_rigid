"""Shared scalar-condition definitions for generation and quantum evaluation."""
from decimal import Decimal
import math

FORMULAS = {"est": "S1 - T1", "s1": "S1", "s1_minus_2t1": "S1 - 2*T1"}


def target_metadata(name, target):
    if name not in FORMULAS or not math.isfinite(float(target)):
        raise ValueError("invalid scalar excitation condition")
    return {"condition_name": name, "target_value_eV": float(target),
            f"target_{name}_eV": float(target)}


def energy_values(s1, t1):
    s1, t1 = Decimal(str(s1)), Decimal(str(t1))
    if not s1.is_finite() or not t1.is_finite():
        raise ValueError("nonfinite excitation energies")
    return {"s1": s1, "est": s1-t1, "s1_minus_2t1": s1-2*t1}


def require_property(expected, stored):
    if expected not in FORMULAS or stored != expected:
        raise ValueError(f"requested condition {expected!r} differs from stored labels {stored!r}")
