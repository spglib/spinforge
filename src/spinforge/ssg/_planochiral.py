from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from spinforge.irreps import RealIrrepType

SpinPlanochiral: TypeAlias = bool


def is_spin_planochiral(real_irrep_types: list[RealIrrepType]) -> SpinPlanochiral:
    return all(rit.dim % 2 == 0 for rit in real_irrep_types)
