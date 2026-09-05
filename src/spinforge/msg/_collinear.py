from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.pointgroup import _get_rotation_type

from ._msg import (
    ConstructType,
    MagneticSpaceSubgroup,
    classify_construct_type,
    filter_conjugacy_magnetic_space_subgroups,
)
from ._utils import _align_cart_rotations, collect_cart_rotation_axes

if TYPE_CHECKING:
    from spinforge.ssg import SpinSymmetryOperations


def enumerate_magnetic_space_subgroups_collinear(
    cart_rotations: NDArrayFloat,
    operations: SpinSymmetryOperations,
    table: NDArrayInt,
    collinear_axis: NDArrayFloat,
    *,
    atol: float,
    extra_cart_rotations: NDArrayFloat | None = None,
):
    cart_rotation_axes = collect_cart_rotation_axes(
        cart_rotations, extra_cart_rotations, atol=atol
    )

    # Align spin-only group for collinear
    list_maximal_msg = []
    for axis in cart_rotation_axes:
        U0, aligned_cart_rotations = _align_cart_rotations(
            cart_rotations, src_axis=axis, dst_axis=collinear_axis
        )
        xsg, fsg, msg_type = _get_magnetic_space_subgroup_collinear(
            cart_rotations=aligned_cart_rotations,
            spin_rotations=operations.spin_rotations,
            atol=atol,
        )
        msg = MagneticSpaceSubgroup(
            xsg=xsg,
            fsg=fsg,
            msg_type=msg_type,
            Q=U0.T,
        )
        list_maximal_msg.append(msg)

    list_conjugacy_maximal_msg = filter_conjugacy_magnetic_space_subgroups(list_maximal_msg, table)
    return list_conjugacy_maximal_msg


def _get_magnetic_space_subgroup_collinear(
    cart_rotations: NDArrayFloat,
    spin_rotations: NDArrayFloat,
    *,
    atol: float,
) -> tuple[list[int], list[int], ConstructType]:
    spatial_rotation_types = np.array([_get_rotation_type(r) for r in cart_rotations])

    xsg = []
    fsg = []
    for i, (r, u) in enumerate(zip(cart_rotations, spin_rotations)):
        if (not np.allclose(r[:2, 2], 0, atol=atol)) or (not np.allclose(r[2, :2], 0, atol=atol)):
            continue
        det_r = np.around(np.linalg.det(r)).astype(int)
        if np.isclose(r[2, 2], u[2, 2], atol=atol):
            # r ~= u @ B_so
            det_adjusted_u = det_r
        elif np.isclose(r[2, 2], -u[2, 2], atol=atol):
            # -r ~= u @ B_so
            det_adjusted_u = -det_r
        else:
            raise ValueError(
                "Spatial rotation and spin rotation determinants are inconsistent. "
                f"r[2,2]={r[2, 2]:.4f}, u[2,2]={u[2, 2]:.4f}"
            )

        fsg.append(i)
        if det_adjusted_u == 1:
            xsg.append(i)

    msg_type = classify_construct_type(xsg, fsg, spatial_rotation_types == 1)

    return xsg, fsg, msg_type
