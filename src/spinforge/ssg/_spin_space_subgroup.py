"""Spin space groups over a family subgroup ``G'`` of a parent group ``G``."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from spinforge.space_group import FamilySpaceSubgroup, NormalSpaceSubgroup
from spinforge.space_group._normalizer import (
    classify_normal_space_subgroup_conjugacy_classes,
)

from ._normalizer_equivalence import deduplicate_spin_rotation_assignments
from ._spin_only_group import get_spin_only_group
from ._spin_space_group import NontrivialSpinSpaceGroup, SpinSpaceGroupEnumerator


class SpinSpaceSubgroupEnumerator:
    """Enumerate SSGs over a family subgroup ``G' <= G``.

    ``SpinSpaceGroupEnumerator`` supplies compatible spin-rotation parts for the
    fixed family group ``G'``, up to a global spin-frame transformation in
    ``O(3)``. This class additionally uses the inclusion ``G' <= G`` carried by
    ``FamilySpaceSubgroup``. By default it returns representatives under
    ``N_G(G') x O(3)``; callers may disable the ``N_G(G')`` conjugacy reduction.

    Returned operations are expressed in the translation-lattice basis of
    ``G'``. Use
    ``NontrivialSpinSpaceGroup.transform_to_parent`` with
    ``family_space_subgroup.translation_sublattice`` when parent coordinates
    are required.
    """

    def __init__(
        self,
        family_space_subgroup: FamilySpaceSubgroup,
        *,
        atol: float = 1e-5,
    ) -> None:
        self._family_space_subgroup = family_space_subgroup
        self._atol = atol
        self._spin_space_group_enumerator = SpinSpaceGroupEnumerator(
            prim_rotations=family_space_subgroup.rotations.tolist(),
            prim_translations=family_space_subgroup.translations.tolist(),
            sublattice=family_space_subgroup.relative_sublattice,
            atol=atol,
        )
        action = family_space_subgroup.normalizer_action
        self._normalizer_rotations, self._normalizer_translations = (
            family_space_subgroup.translation_sublattice.transform_operations(
                action.rotations,
                action.translations,
                atol=atol,
            )
        )

    @property
    def family_space_subgroup(self) -> FamilySpaceSubgroup:
        return self._family_space_subgroup

    @property
    def prim_rotations(self) -> NDArrayInt:
        return self._spin_space_group_enumerator.prim_rotations

    @property
    def prim_translations(self) -> NDArrayFloat:
        return self._spin_space_group_enumerator.prim_translations

    def enumerate_spin_space_groups(
        self,
        spin_only_group_type: SpinOnlyGroupType,
        k_index: int,
        *,
        up_to_parent_normalizer_conjugacy: bool = True,
    ) -> tuple[SpinOnlyGroup, list[NontrivialSpinSpaceGroup]]:
        """Enumerate spin space subgroups, optionally modulo ``N_G(G')``.

        ``up_to_parent_normalizer_conjugacy=True`` returns representatives under
        ``N_G(G') x O(3)``. ``False`` skips the spatial ``N_G(G')`` reduction
        and returns every fixed-family result; the intrinsic ``O(3)`` spin-frame
        equivalence remains applied.
        """
        spin_only_group, grouped_spin_space_groups = (
            self.enumerate_spin_space_groups_by_invariant_subgroup(
                spin_only_group_type,
                k_index,
                up_to_parent_normalizer_conjugacy=up_to_parent_normalizer_conjugacy,
            )
        )
        return spin_only_group, [
            spin_space_group
            for _, spin_space_groups in grouped_spin_space_groups
            for spin_space_group in spin_space_groups
        ]

    def enumerate_spin_space_groups_by_invariant_subgroup(
        self,
        spin_only_group_type: SpinOnlyGroupType,
        k_index: int,
        *,
        up_to_parent_normalizer_conjugacy: bool = True,
        invariant_subgroup_filter: Callable[[NormalSpaceSubgroup], bool] | None = None,
    ) -> tuple[
        SpinOnlyGroup,
        list[tuple[NormalSpaceSubgroup, list[NontrivialSpinSpaceGroup]]],
    ]:
        """Enumerate results grouped by invariant subgroup.

        When ``up_to_parent_normalizer_conjugacy`` is true, invariant subgroups
        and their spin-rotation assignments are reduced under the applicable
        ``N_G(G')`` actions. When false, the fixed-family groups are returned
        without either spatial conjugacy reduction. In both cases, spin frames
        remain equivalent under ``O(3)``.

        ``invariant_subgroup_filter`` is applied after the spatial conjugacy
        reduction and before spin-rotation assignments are enumerated. It lets
        callers reject invariant subgroups using properties that do not depend
        on those assignments.
        """
        spin_only_group = get_spin_only_group(spin_only_group_type)
        if (spin_only_group_type == SpinOnlyGroupType.COLLINEAR) and (k_index >= 3):
            return spin_only_group, []

        invariant_subgroups = self._spin_space_group_enumerator.enumerate_normal_space_subgroups(
            k_index
        )
        action = self._family_space_subgroup.normalizer_action
        if up_to_parent_normalizer_conjugacy:
            invariant_subgroups_with_stabilizers: list[
                tuple[NormalSpaceSubgroup, Sequence[int] | None]
            ] = [
                (equivalence_class.representative, equivalence_class.stabilizer)
                for equivalence_class in classify_normal_space_subgroup_conjugacy_classes(
                    list_nss=invariant_subgroups,
                    family_rotations=self.prim_rotations,
                    normalizer_rotations=self._normalizer_rotations,
                    normalizer_translations=self._normalizer_translations,
                    normalizer_permutations=action.permutations,
                    atol=self._atol,
                )
            ]
        else:
            invariant_subgroups_with_stabilizers = [
                (invariant_subgroup, None) for invariant_subgroup in invariant_subgroups
            ]

        grouped_spin_space_groups = []
        for invariant_subgroup, stabilizer in invariant_subgroups_with_stabilizers:
            if invariant_subgroup_filter is not None and not invariant_subgroup_filter(
                invariant_subgroup
            ):
                continue
            _, spin_space_groups = (
                self._spin_space_group_enumerator.enumerate_spin_space_groups_with_normal_space_subgroup(
                    spin_only_group_type=spin_only_group_type,
                    normal_space_subgroup=invariant_subgroup,
                )
            )
            if stabilizer is not None:
                spin_space_groups = deduplicate_spin_rotation_assignments(
                    list_nssg=spin_space_groups,
                    nss=invariant_subgroup,
                    stabilizer=stabilizer,
                    spin_only_group_type=spin_only_group_type,
                    family_table=self._spin_space_group_enumerator.table,
                    normalizer_permutations=action.permutations,
                    atol=self._atol,
                )
            grouped_spin_space_groups.append((invariant_subgroup, spin_space_groups))

        return spin_only_group, grouped_spin_space_groups
