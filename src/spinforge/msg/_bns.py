from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from moyopy import (
    Cell,
    MagneticSpaceGroupType,
    MoyoNonCollinearMagneticDataset,
    NonCollinearMagneticCell,
)
from spgrep.utils import NDArrayFloat

if TYPE_CHECKING:
    from spinforge.configuration import Supercell


def bns_from_moments(
    supercell: Cell | Supercell,
    moments: NDArrayFloat,
    *,
    atol: float = 1e-5,
    symprec: float | None = None,
    mag_symprec: float | None = None,
) -> str:
    """BNS number of ``supercell`` carrying the Cartesian ``moments``.

    The moments are normalized to unit max |m| before the magnetic symmetry
    search, so ``mag_symprec`` keeps its meaning regardless of the pattern's
    scale; a pattern whose max |m| is below ``atol`` is treated as zero and
    left unnormalized. ``symprec`` / ``mag_symprec`` override moyopy's search
    tolerances; ``None`` keeps its defaults.
    """
    moments = np.asarray(moments, dtype=float)
    norm = float(np.linalg.norm(moments, axis=1).max())
    if norm > atol:
        moments = moments / norm
    cell = NonCollinearMagneticCell(
        basis=np.array(supercell.basis).tolist(),
        positions=np.array(supercell.positions).tolist(),
        numbers=[int(n) for n in supercell.numbers],
        magnetic_moments=moments.tolist(),
    )
    if symprec is None:
        dataset = MoyoNonCollinearMagneticDataset(
            cell, rotate_basis=False, mag_symprec=mag_symprec
        )
    else:
        dataset = MoyoNonCollinearMagneticDataset(
            cell, rotate_basis=False, symprec=symprec, mag_symprec=mag_symprec
        )
    return MagneticSpaceGroupType(dataset.uni_number).bns_number
