from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from spgrep.utils import NDArrayFloat
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from spinforge.utils.rotation_utils import to_cartesian_rotations

from ._collinear import enumerate_magnetic_space_subgroups_collinear
from ._coplanar import enumerate_magnetic_space_subgroups_coplanar
from ._msg import MagneticSpaceSubgroup, classify_construct_type
from ._noncoplanar import enumerate_magnetic_space_subgroups_noncoplanar

if TYPE_CHECKING:
    from spinforge.ssg import NontrivialSpinSpaceGroup, SpinSymmetryOperations


class OrientedSpinSpaceGroupEnumerator:
    """Enumerate oriented SSGs up to spatial conjugacy by their family group ``G``.

    ``preserve_spin_planochirality=True`` retains distinct spin-planochiral
    enantiomorphs. ``False`` identifies enantiomorphs related by an improper
    spin rotation.
    """

    def __init__(
        self,
        lattice: NDArrayFloat,
        spin_only_group: SpinOnlyGroup,
        spin_space_group: NontrivialSpinSpaceGroup,
        *,
        atol: float = 1e-5,
    ) -> None:
        self._lattice = lattice
        self._spin_only_group = spin_only_group
        self._spin_space_group = spin_space_group
        self._atol = atol

    def enumerate(
        self,
        *,
        preserve_spin_planochirality: bool = True,
    ) -> list[MagneticSpaceSubgroup]:
        return _enumerate_oriented_spin_space_groups(
            lattice=self._lattice,
            spin_only_group=self._spin_only_group,
            spin_space_group=self._spin_space_group,
            preserve_spin_planochirality=preserve_spin_planochirality,
            atol=self._atol,
        )


def _enumerate_oriented_spin_space_groups(
    lattice: NDArrayFloat,
    spin_only_group: SpinOnlyGroup,
    spin_space_group: NontrivialSpinSpaceGroup,
    *,
    preserve_spin_planochirality: bool,
    atol: float,
    extra_cart_rotations: NDArrayFloat | None = None,
) -> list[MagneticSpaceSubgroup]:
    operations, table = spin_space_group.get_full_operations_and_table()
    cart_rotations = to_cartesian_rotations(lattice, operations.rotations)

    # Triclinic case
    if all(
        np.logical_or(
            [np.allclose(r, np.eye(3)) for r in operations.rotations],
            [np.allclose(r, -np.eye(3)) for r in operations.rotations],
        )
    ):
        return [
            _get_triclinic_magnetic_space_subgroups(
                operations=operations,
                spin_only_group_type=spin_only_group.spin_only_group_type,
                atol=atol,
            )
        ]

    if spin_only_group.spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
        assert spin_only_group.axis is not None
        return enumerate_magnetic_space_subgroups_collinear(
            cart_rotations=cart_rotations,
            operations=operations,
            table=table,
            collinear_axis=spin_only_group.axis,
            atol=atol,
            extra_cart_rotations=extra_cart_rotations,
        )
    elif spin_only_group.spin_only_group_type == SpinOnlyGroupType.COPLANAR:
        assert spin_only_group.axis is not None
        return enumerate_magnetic_space_subgroups_coplanar(
            cart_rotations=cart_rotations,
            operations=operations,
            nssg=spin_space_group,
            table=table,
            coplanar_axis=spin_only_group.axis,
            preserve_spin_planochirality=preserve_spin_planochirality,
            atol=atol,
            extra_cart_rotations=extra_cart_rotations,
        )
    elif spin_only_group.spin_only_group_type == SpinOnlyGroupType.NONCOPLANAR:
        return enumerate_magnetic_space_subgroups_noncoplanar(
            cart_rotations=cart_rotations,
            operations=operations,
            table=table,
            atol=atol,
        )
    else:
        raise ValueError(
            f"Unsupported spin-only group type: {spin_only_group.spin_only_group_type}"
        )


def _get_triclinic_magnetic_space_subgroups(
    operations: SpinSymmetryOperations,
    spin_only_group_type: SpinOnlyGroupType,
    *,
    atol: float,
) -> MagneticSpaceSubgroup:
    def _match(u: np.ndarray) -> int | None:
        # Find (det u') u' = E, u' = u @ u0
        det = round(float(np.linalg.det(u)))
        if np.allclose(u, np.eye(3), atol=atol) or np.allclose(u, -np.eye(3), atol=atol):
            return det
        if spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
            if (not np.allclose(u[0, 2:], 0, atol=atol)) or (
                not np.allclose(u[2:, 0], 0, atol=atol)
            ):
                return None
            if np.isclose(u[2, 2], 1, atol=atol):
                return det
            elif np.isclose(u[2, 2], -1, atol=atol):
                return -det
        elif spin_only_group_type == SpinOnlyGroupType.COPLANAR:
            u2 = u @ np.diag([1, 1, -1])
            if np.allclose(u2, np.eye(3), atol=atol) or np.allclose(u2, -np.eye(3), atol=atol):
                return -det

        return None

    xsg = []
    fsg = []
    for i, W in enumerate(operations.spin_rotations):
        sign = _match(W)
        if sign is None:
            continue
        fsg.append(i)
        if sign == 1:
            xsg.append(i)

    is_identity_rotation = [
        np.allclose(rotation, np.eye(3), atol=atol) for rotation in operations.rotations
    ]
    msg_type = classify_construct_type(xsg, fsg, is_identity_rotation)

    return MagneticSpaceSubgroup(
        xsg=xsg,
        fsg=fsg,
        msg_type=msg_type,
        Q=np.eye(3),
    )
