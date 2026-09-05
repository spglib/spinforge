from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from spgrep.symmetry.subgroup import enumerate_point_subgroup
from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.group import _get_mirror_along_axis
from spinspg.pointgroup import _get_rotation_type

from spinforge.irreps import get_orientation_preserving_real_intertwiner, get_real_intertwiner

from ._msg import (
    ConstructType,
    MagneticSpaceSubgroup,
    _assemble_maximal_magnetic_space_subgroups,
    filter_conjugacy_magnetic_space_subgroups,
)
from ._utils import _align_cart_rotations, collect_cart_rotation_axes, determinant_signs

if TYPE_CHECKING:
    from spinforge.ssg import NontrivialSpinSpaceGroup, SpinSymmetryOperations


def enumerate_magnetic_space_subgroups_coplanar(
    cart_rotations: NDArrayFloat,
    operations: SpinSymmetryOperations,
    nssg: NontrivialSpinSpaceGroup,
    table: NDArrayInt,
    coplanar_axis: NDArrayFloat,
    *,
    preserve_spin_planochirality: bool,
    atol: float,
    extra_cart_rotations: NDArrayFloat | None = None,
):
    cart_rotation_axes = collect_cart_rotation_axes(
        cart_rotations, extra_cart_rotations, atol=atol
    )
    assert np.allclose(np.cross(coplanar_axis, [0, 0, 1]), 0)  # Convention

    # For chiral SSG
    mx = np.diag([-1.0, 1.0, 1.0])
    mirrored_spin_rotations = np.array([mx @ U @ mx.T for U in operations.spin_rotations])

    coplanar_mirror = _get_mirror_along_axis(coplanar_axis)

    list_maximal_msg = []
    for axis in cart_rotation_axes:
        # Align `axis` to z-axis of spin-only group
        U0, aligned_cart_rotations = _align_cart_rotations(
            cart_rotations, src_axis=axis, dst_axis=coplanar_axis
        )

        if preserve_spin_planochirality and nssg.spin_planochiral:
            for trial_spin_rotations, Q0 in [
                (operations.spin_rotations, np.eye(3)),
                (mirrored_spin_rotations, mx),
            ]:
                list_maximal_msg_trial = _enumerate_magnetic_space_subgroups_coplanar_with_axis(
                    cart_rotations=aligned_cart_rotations,
                    spin_rotations=trial_spin_rotations,
                    coplanar_mirror=coplanar_mirror,
                    table=table,
                    preserve_spin_planochirality=preserve_spin_planochirality,
                    atol=atol,
                )
                for msg in list_maximal_msg_trial:
                    # Q @ trial_spin_rotations[fsg] @ Q.T ~= aligned_cart_rotations[fsg]
                    # -> U0.T @ Q @ Q0 @ spin_rotations[fsg] @ (U0.T @ Q @ Q0).T ~= cart_rotations[fsg]
                    list_maximal_msg.append(
                        MagneticSpaceSubgroup(
                            xsg=msg.xsg,
                            fsg=msg.fsg,
                            msg_type=msg.msg_type,
                            Q=U0.T @ msg.Q @ Q0,
                        )
                    )
        else:
            list_maximal_msg_trial = _enumerate_magnetic_space_subgroups_coplanar_with_axis(
                cart_rotations=aligned_cart_rotations,
                spin_rotations=operations.spin_rotations,
                coplanar_mirror=coplanar_mirror,
                table=table,
                preserve_spin_planochirality=preserve_spin_planochirality,
                atol=atol,
            )
            for msg in list_maximal_msg_trial:
                # Q @ spin_rotations[fsg] @ Q.T ~= aligned_cart_rotations[fsg] (up to spin-only group)
                # -> U0.T @ Q @ spin_rotations[fsg] @ (U0.T @ Q).T ~= cart_rotations[fsg]
                list_maximal_msg.append(
                    MagneticSpaceSubgroup(
                        xsg=msg.xsg,
                        fsg=msg.fsg,
                        msg_type=msg.msg_type,
                        Q=U0.T @ msg.Q,
                    )
                )

    list_conjugacy_maximal_msg = filter_conjugacy_magnetic_space_subgroups(list_maximal_msg, table)
    return list_conjugacy_maximal_msg


def _enumerate_magnetic_space_subgroups_coplanar_with_axis(
    cart_rotations: NDArrayFloat,
    spin_rotations: NDArrayFloat,
    coplanar_mirror: NDArrayFloat,
    table: NDArrayInt,
    *,
    preserve_spin_planochirality: bool,
    atol: float,
) -> list[MagneticSpaceSubgroup]:
    def _adjust_time_reversal_parts(
        spin_rotations: NDArrayFloat,
        original_time_reversals: NDArrayInt,
        target_time_reversals: NDArrayInt,
    ) -> NDArrayFloat:
        """Apply mirror to spin_rotations to match target_time_reversals."""
        assert np.all(np.abs(original_time_reversals) == 1)
        apply_mirrors = target_time_reversals != original_time_reversals
        ret = spin_rotations.copy()
        ret[apply_mirrors] = ret[apply_mirrors] @ coplanar_mirror
        return ret * target_time_reversals[:, None, None]

    def _search_intertwiner_2d(
        spatial_rotations: NDArrayFloat,
        spin_rotations: NDArrayFloat,
        subgroup: list[int],
    ) -> NDArrayFloat | None:
        # Project to xy-plane
        if not np.allclose(spatial_rotations[:, 2, 2], spin_rotations[:, 2, 2], atol=atol):
            return None
        spatial_rotations_2d = spatial_rotations[:, :2, :2]
        spin_rotations_2d = spin_rotations[:, :2, :2]

        mapping = {g: i for i, g in enumerate(subgroup)}
        subtable = table[subgroup][:, subgroup]

        # Q_2d^-1 @ spatial_rotations_2d @ Q_2d == spin_rotations_2d
        if preserve_spin_planochirality:
            Q_2d = get_orientation_preserving_real_intertwiner(
                spatial_rotations_2d,
                spin_rotations_2d,
                table=np.array([[mapping[g] for g in row] for row in subtable]),
                atol=atol,
            )
        else:
            Q_2d = get_real_intertwiner(
                spatial_rotations_2d,
                spin_rotations_2d,
                atol=atol,
            )
        if Q_2d is None:
            return None
        assert np.allclose(Q_2d @ Q_2d.T, np.eye(2), atol=atol)
        assert np.allclose(
            [r @ Q_2d for r in spatial_rotations_2d],
            [Q_2d @ u for u in spin_rotations_2d],
        )

        Q = np.eye(3)
        Q[:2, :2] = Q_2d
        return Q

    is_coplanar_spatial_rotations = np.logical_and(
        np.all(np.isclose(cart_rotations[:, :2, 2], 0, atol=atol), axis=1),
        np.all(np.isclose(cart_rotations[:, 2, :2], 0, atol=atol), axis=1),
    )
    spatial_parities = determinant_signs(cart_rotations)  # 1 for identity, -1 for inversion
    original_time_reversals = determinant_signs(
        spin_rotations
    )  # 1 for identity, -1 for time reversal

    # Find candidate XSG
    spatial_rotation_types = [_get_rotation_type(r) for r in cart_rotations]
    spin_rotation_types = [_get_rotation_type(u) for u in spin_rotations]

    coplanar_mirrored_spin_rotations = np.array([u @ coplanar_mirror for u in spin_rotations])
    coplanar_mirrored_spin_rotation_types = [
        _get_rotation_type(u) for u in coplanar_mirrored_spin_rotations
    ]

    subgroups = enumerate_point_subgroup(
        table,
        preserve_sublattice=np.logical_and(
            is_coplanar_spatial_rotations,
            np.logical_or(
                np.logical_and(
                    np.abs(spatial_rotation_types) == spin_rotation_types,
                    original_time_reversals == 1,
                ),
                np.logical_and(
                    np.abs(spatial_rotation_types) == coplanar_mirrored_spin_rotation_types,
                    original_time_reversals == -1,
                ),
            ),
        ),
        return_conjugacy_class=False,
    )

    list_xsg: list[tuple[list[int], NDArrayFloat]] = []
    for xsg in subgroups:
        spatial_subgroup = cart_rotations[xsg] * spatial_parities[xsg][:, None, None]
        spin_subgroup = _adjust_time_reversal_parts(
            spin_rotations=spin_rotations[xsg],
            original_time_reversals=original_time_reversals[xsg],
            target_time_reversals=np.ones(len(xsg), dtype=int),  # XSG
        )
        Q = _search_intertwiner_2d(
            spatial_rotations=spatial_subgroup,
            spin_rotations=spin_subgroup,
            subgroup=xsg,
        )
        if Q is None:
            continue
        list_xsg.append((xsg, Q))

    def _is_antiunitary_generator(g: int, xsg: list[int]) -> bool:
        # Unlike the noncoplanar case, no time-reversal condition can be tested
        # here: the time-reversal pattern is imposed afterwards by
        # _adjust_time_reversal_parts (target -1 on the g-coset), and whether g
        # truly acts anti-unitarily is decided by the intertwiner search. Only
        # the group-extension condition g^2 in XSG remains, which also admits
        # higher-order generators (g^2 need not be the identity).
        return table[g, g] in xsg

    def _find_fsg_intertwiner(fsg: list[int], conj_xsg: list[int]) -> NDArrayFloat | None:
        spatial_subgroup = cart_rotations[fsg] * spatial_parities[fsg][:, None, None]
        target_time_reversals = np.ones(len(original_time_reversals), dtype=int)
        target_time_reversals[conj_xsg] = -1
        spin_subgroup = _adjust_time_reversal_parts(
            spin_rotations=spin_rotations[fsg],
            original_time_reversals=original_time_reversals[fsg],
            target_time_reversals=target_time_reversals[fsg],
        )
        return _search_intertwiner_2d(
            spatial_rotations=spatial_subgroup,
            spin_rotations=spin_subgroup,
            subgroup=fsg,
        )

    def _classify(xsg: list[int], fsg: list[int], conj_xsg: list[int]) -> ConstructType:
        # An anti-translation must act as -E in spin space *after* undoing the
        # coplanar mirror adjustment, so the Type-4 test depends on whether the
        # operation's time-reversal part was mirrored to reach its target.
        for h in fsg:
            if not np.allclose(cart_rotations[h], np.eye(3), atol=atol):
                continue
            target_time_reversal = -1 if h in conj_xsg else 1
            if original_time_reversals[h] == target_time_reversal:
                if np.allclose(spin_rotations[h], -np.eye(3), atol=atol):
                    return ConstructType.TYPE4
            elif np.allclose(coplanar_mirrored_spin_rotations[h], -np.eye(3), atol=atol):
                return ConstructType.TYPE4
        return ConstructType.TYPE3

    return _assemble_maximal_magnetic_space_subgroups(
        table,
        list_xsg,
        is_antiunitary_generator=_is_antiunitary_generator,
        find_fsg_intertwiner=_find_fsg_intertwiner,
        classify=_classify,
    )
