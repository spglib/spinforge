"""Finite subgroup selections with an unchanged spin assignment."""

from __future__ import annotations

import numpy as np
from spgrep.utils import NDArrayInt
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from ._spin_space_group import NontrivialSpinSpaceGroup


def _fixed_group_table(
    nssg: NontrivialSpinSpaceGroup, spin_only_group: SpinOnlyGroup, atol: float
) -> NDArrayInt:
    """Snapshot the quotient table after checking the assigned spin cosets.

    The spin-only group fixes the active spin subspace pointwise: a line for
    collinear input, a plane for coplanar input, and all of R^3 otherwise.
    Equality on this subspace therefore tests equality modulo the spin-only
    group without enumerating its possibly continuous elements.
    """
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be positive and finite")
    kind = spin_only_group.spin_only_group_type
    if kind not in (
        SpinOnlyGroupType.COLLINEAR,
        SpinOnlyGroupType.COPLANAR,
        SpinOnlyGroupType.NONCOPLANAR,
    ):
        raise ValueError("Fixed-SSG enumeration requires a magnetic spin-only group")

    projector = np.eye(3)
    if kind in (SpinOnlyGroupType.COLLINEAR, SpinOnlyGroupType.COPLANAR):
        axis = np.array(spin_only_group.axis, dtype=float, copy=True)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.any(axis):
            raise ValueError("spin_only_group.axis must be a finite nonzero three-vector")
        axis /= np.max(np.abs(axis))
        axis /= np.linalg.norm(axis)
        projector = np.outer(axis, axis)
        if kind == SpinOnlyGroupType.COPLANAR:
            projector = np.eye(3) - projector

    operations, table = nssg.get_full_operations_and_table()
    spins = operations.spin_rotations
    if spins.shape != (len(table), 3, 3) or not np.isfinite(spins).all():
        raise ValueError("spin_rotations must be finite 3-by-3 matrices")
    if not np.allclose(spins @ spins.transpose(0, 2, 1), np.eye(3), atol=atol, rtol=0):
        raise ValueError("spin_rotations must be orthogonal")
    active_spins = spins @ projector
    if not np.allclose(active_spins @ spins.transpose(0, 2, 1), projector, atol=atol, rtol=0):
        raise ValueError("spin_rotations must normalize the spin-only group")
    for i, spin in enumerate(spins):
        if not np.allclose(spin @ active_spins, active_spins[table[i]], atol=atol, rtol=0):
            raise ValueError(
                f"Spin assignments do not close modulo the spin-only group at operation {i}"
            )
    return table
