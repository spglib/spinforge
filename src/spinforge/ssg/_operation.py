from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spinforge.space_group import Sublattice


@dataclass
class SpinSymmetryOperations:
    rotations: np.ndarray[tuple[int, int, int], np.dtype[np.int_]]
    """(n, 3, 3)"""
    translations: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    """(n, 3)"""
    spin_rotations: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    """(n, 3, 3)"""

    def __post_init__(self):
        size = len(self.rotations)
        if len(self.translations) != size:
            raise ValueError(
                f"Translation count ({len(self.translations)}) must match rotation count ({size})"
            )
        if len(self.spin_rotations) != size:
            raise ValueError(
                f"Spin rotation count ({len(self.spin_rotations)}) must match "
                f"rotation count ({size})"
            )

    @property
    def size(self) -> int:
        return len(self.rotations)

    def transform_to_parent(
        self,
        sublattice: Sublattice,
        *,
        atol: float = 1e-5,
    ) -> SpinSymmetryOperations:
        """Transform spatial parts from ``sublattice`` coordinates to the parent."""
        rotations, translations = sublattice.transform_operations_to_parent(
            self.rotations, self.translations, atol=atol
        )
        return SpinSymmetryOperations(
            rotations=rotations,
            translations=translations,
            spin_rotations=self.spin_rotations,
        )

    def __str__(self):
        return f"SpinSymmetryOperations(\nrotations={self.rotations},\ntranslations~={np.around(self.translations, 3)},\nspin_rotations~={np.around(self.spin_rotations, 3)})"
