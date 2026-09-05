from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from spgrep.utils import is_integer_array

if TYPE_CHECKING:
    from moyopy import NonCollinearMagneticCell


def match_mag_cells_with_same_basis(
    mag_cell_1: NonCollinearMagneticCell,
    mag_cell_2: NonCollinearMagneticCell,
    *,
    symprec: float,
) -> tuple[list[int] | None, np.ndarray | None]:
    """Try to match to magnetic cells with the same basis vectors.

    Returned site_mapping and origin_shift satisfy:
        mag_cell_1.positions[i] + origin_shift = mag_cell_2.positions[site_mapping[i]]
    up to lattice translations.

    Parameters
    ----------
    mag_cell_1 : NonCollinearMagneticCell
        The first magnetic cell.
    mag_cell_2 : NonCollinearMagneticCell
        The second magnetic cell.
    symprec : float

    Returns
    -------
    site_mapping : list[int] | None
        The site mapping from `mag_cell_1` to `mag_cell_2
    origin_shift : np.ndarray | None
        The origin shift applied to `mag_cell_1` to match `mag_cell_2`.
    """
    assert np.allclose(mag_cell_1.basis, mag_cell_2.basis, atol=symprec)
    assert mag_cell_1.num_atoms == mag_cell_2.num_atoms
    epsilon = symprec / (np.abs(np.linalg.det(mag_cell_1.basis)) ** (1 / 3))

    def _match(origin_shift: np.ndarray) -> list[int] | None:
        positions1 = np.array(mag_cell_1.positions) + origin_shift[None, :]
        site_mapping = [-1 for _ in range(mag_cell_1.num_atoms)]
        visited = set()
        for i, (pos1, num1) in enumerate(zip(positions1, mag_cell_1.numbers)):
            found = False
            for j, (pos2, num2) in enumerate(zip(mag_cell_2.positions, mag_cell_2.numbers)):
                if j in visited:
                    continue
                if num1 != num2:
                    continue
                diff = np.array(pos1) - np.array(pos2)
                diff -= np.rint(diff)
                if is_integer_array(diff, atol=epsilon):
                    site_mapping[i] = j
                    visited.add(j)
                    found = True
                    break
            if not found:
                return None
        assert len(visited) == mag_cell_1.num_atoms
        return site_mapping

    # Match positions and numbers
    candidate_origin_shifts = []
    for pos2, num2 in zip(mag_cell_2.positions, mag_cell_2.numbers):
        if num2 != mag_cell_1.numbers[0]:
            continue
        origin_shift = np.array(pos2) - np.array(mag_cell_1.positions[0])
        candidate_origin_shifts.append(origin_shift)

    for origin_shift in candidate_origin_shifts:
        site_mapping = _match(origin_shift)
        if site_mapping is not None:
            logger.trace(f"origin_shift: {np.around(origin_shift, 6)}")
            return site_mapping, origin_shift
    return None, None
