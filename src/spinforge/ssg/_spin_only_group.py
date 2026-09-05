from __future__ import annotations

import numpy as np
from spgrep.utils import NDArrayFloat
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

SPIN_ONLY_GROUP_AXIS = np.array([0, 0, 1], dtype=float)  # Choose z-axis as our convention


def get_spin_only_group(spin_only_group_type: SpinOnlyGroupType) -> SpinOnlyGroup:
    if spin_only_group_type == SpinOnlyGroupType.NONMAGNETIC:
        raise ValueError("Do not use NONMAGNETIC for spin space group enumeration.")
    elif spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
        spin_only_group = SpinOnlyGroup.collinear(axis=SPIN_ONLY_GROUP_AXIS)
    elif spin_only_group_type == SpinOnlyGroupType.COPLANAR:
        spin_only_group = SpinOnlyGroup.coplanar(axis=SPIN_ONLY_GROUP_AXIS)
    elif spin_only_group_type == SpinOnlyGroupType.NONCOPLANAR:
        spin_only_group = SpinOnlyGroup.noncoplanar()
    else:
        raise ValueError("unreachable!")

    return spin_only_group


def embed_real_representation(
    spin_only_group_type: SpinOnlyGroupType,
    rep: NDArrayFloat,
) -> NDArrayFloat:
    if spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
        spin_point_group = np.tile(np.eye(3), reps=(len(rep), 1, 1))
        spin_point_group[:, 2, 2] = rep[:, 0, 0]  # Parallel to z-axis
    elif spin_only_group_type == SpinOnlyGroupType.COPLANAR:
        spin_point_group = np.tile(np.eye(3), reps=(len(rep), 1, 1))
        spin_point_group[:, :2, :2] = rep  # Perpendicular to z-axis
    elif spin_only_group_type == SpinOnlyGroupType.NONCOPLANAR:
        spin_point_group = rep
    else:
        raise ValueError(f"Unsupported spin only group type: {spin_only_group_type}")

    return spin_point_group


def extract_active_block(
    spin_only_group_type: SpinOnlyGroupType,
    embedded_rep: NDArrayFloat,
) -> NDArrayFloat:
    """Return the ``d x d`` active block of a stack of spin rotations embedded
    in ``O(3)``.

    The active block is the one-sided inverse of
    :func:`embed_real_representation`: for each ``SpinOnlyGroupType`` it
    extracts exactly the sub-block that is meaningful modulo the spin-only
    group ``B_so``. The remaining ``(3 - d)`` perpendicular directions are
    the free directions of ``B_so``, so two embedded spin rotations are equal
    modulo ``B_so`` iff their active blocks agree.

    Parameters
    ----------
    spin_only_group_type:
        Type of the spin-only group determining ``d``.
    embedded_rep: (order, 3, 3)
        Stack of spin rotations embedded into ``O(3)`` via
        :func:`embed_real_representation`.

    Returns
    -------
    active: (order, d, d)
        ``d = 1`` for COLLINEAR, ``d = 2`` for COPLANAR, ``d = 3`` for
        NONCOPLANAR.
    """
    if spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
        return embedded_rep[:, 2:3, 2:3]
    if spin_only_group_type == SpinOnlyGroupType.COPLANAR:
        return embedded_rep[:, :2, :2]
    if spin_only_group_type == SpinOnlyGroupType.NONCOPLANAR:
        return embedded_rep
    raise ValueError(f"Unsupported spin only group type: {spin_only_group_type}")
