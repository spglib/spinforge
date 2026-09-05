from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from spgrep.rep.group import get_identity_index
from spgrep.symmetry.subgroup import enumerate_point_subgroup
from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.pointgroup import _get_rotation_type

from spinforge.irreps import get_real_intertwiner

from ._msg import (
    ConstructType,
    MagneticSpaceSubgroup,
    _assemble_maximal_magnetic_space_subgroups,
    classify_construct_type,
    filter_conjugacy_magnetic_space_subgroups,
)
from ._utils import determinant_signs

if TYPE_CHECKING:
    from spinforge.ssg import SpinSymmetryOperations


def enumerate_magnetic_space_subgroups_noncoplanar(
    cart_rotations: NDArrayFloat,
    operations: SpinSymmetryOperations,
    table: NDArrayInt,
    *,
    atol: float,
) -> list[MagneticSpaceSubgroup]:
    spatial_parities = determinant_signs(cart_rotations)  # 1 for identity, -1 for inversion
    time_reversals = determinant_signs(
        operations.spin_rotations
    )  # 1 for identity, -1 for time reversal

    # Find candidate XSG
    spatial_rotation_types = np.array([_get_rotation_type(r) for r in cart_rotations])
    spin_rotation_types = np.array([_get_rotation_type(u) for u in operations.spin_rotations])

    subgroups = enumerate_point_subgroup(
        table,
        # Consider only same-rotation-type pair
        preserve_sublattice=np.logical_and(
            np.abs(spatial_rotation_types) == spin_rotation_types,
            time_reversals == 1,
        ),
        return_conjugacy_class=False,
    )

    list_xsg: list[tuple[list[int], NDArrayFloat]] = []
    for xsg in subgroups:
        Q = _orthogonal_intertwiner(
            cart_rotations[xsg] * spatial_parities[xsg][:, None, None],
            operations.spin_rotations[xsg],
            atol=atol,
        )
        if Q is None:
            continue
        list_xsg.append((xsg, Q))

    identity = get_identity_index(table)

    def _is_antiunitary_generator(g: int, xsg: list[int]) -> bool:
        return time_reversals[g] == -1 and table[g, g] == identity

    def _find_fsg_intertwiner(fsg: list[int], conj_xsg: list[int]) -> NDArrayFloat | None:
        return _orthogonal_intertwiner(
            cart_rotations[fsg] * spatial_parities[fsg][:, None, None],
            operations.spin_rotations[fsg] * time_reversals[fsg][:, None, None],
            atol=atol,
        )

    def _classify(xsg: list[int], fsg: list[int], conj_xsg: list[int]) -> ConstructType:
        return classify_construct_type(xsg, fsg, spatial_rotation_types == 1)

    list_maximal_msg = _assemble_maximal_magnetic_space_subgroups(
        table,
        list_xsg,
        is_antiunitary_generator=_is_antiunitary_generator,
        find_fsg_intertwiner=_find_fsg_intertwiner,
        classify=_classify,
    )
    list_conjugacy_maximal_msg = filter_conjugacy_magnetic_space_subgroups(list_maximal_msg, table)
    return list_conjugacy_maximal_msg


def _orthogonal_intertwiner(
    spatial_subgroup: NDArrayFloat,
    spin_subgroup: NDArrayFloat,
    *,
    atol: float,
) -> NDArrayFloat | None:
    """Return Q with ``{ (det R) R } == Q @ { (det U) U } @ Q^-1``, or None if none exists."""
    Q = get_real_intertwiner(spatial_subgroup, spin_subgroup, atol=atol)
    if Q is None:
        return None
    assert np.allclose(Q @ Q.T, np.eye(3), atol=atol)
    return Q
