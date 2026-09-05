from __future__ import annotations

from dataclasses import dataclass

from spgrep.utils import NDArrayFloat, NDArrayInt

from spinforge.utils.group_theory import GroupElementIndex

from ._sublattice import Sublattice


@dataclass
class NormalSpaceSubgroup:
    # Normal space subgroup (H)
    point_subgroup: list[GroupElementIndex]
    """Indices of the normal point subgroup."""
    translations: NDArrayFloat
    """Translation parts."""
    # Intermediate t-subgroup (M)
    sublattice: Sublattice
    """Invariant sublattice."""
    # Coset representatives (G/M)
    coset_representatives: list[GroupElementIndex]
    quotient_table: NDArrayInt
    """Cayley table of the finite quotient ``G/H``."""

    def __post_init__(self):
        if len(self.translations) != len(self.point_subgroup):
            raise ValueError(
                f"Translation count ({len(self.translations)}) must match "
                f"point subgroup size ({len(self.point_subgroup)})"
            )
        quotient_order = self.t_index * self.k_index
        if self.quotient_table.shape != (quotient_order, quotient_order):
            raise ValueError(
                f"Quotient table shape {self.quotient_table.shape} must be "
                f"({quotient_order}, {quotient_order})"
            )

    @property
    def k_index(self) -> int:
        return self.sublattice.order

    @property
    def t_index(self) -> int:
        return len(self.coset_representatives)
