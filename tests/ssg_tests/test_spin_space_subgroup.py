from __future__ import annotations

import numpy as np
import pytest
from moyopy import MoyoDataset, SpaceGroup
from spinspg.spin import SpinOnlyGroupType

from spinforge.space_group import FamilySpaceSubgroup, FamilySpaceSubgroupEnumerator
from spinforge.ssg import SpinSpaceGroupEnumerator, SpinSpaceSubgroupEnumerator
from spinforge.testing import load_prim_MnS2


@pytest.fixture(scope="module")
def mns2_family_subgroups() -> list[FamilySpaceSubgroup]:
    cell = load_prim_MnS2()
    dataset = MoyoDataset(cell, rotate_basis=False)
    rotations = np.asarray(dataset.operations.rotations, dtype=np.int64)
    translations = np.asarray(dataset.operations.translations, dtype=float)
    epsilon = dataset.symprec / np.abs(np.linalg.det(np.asarray(cell.basis))) ** (1 / 3)
    enumerator = FamilySpaceSubgroupEnumerator(
        rotations,
        translations,
        epsilon=epsilon,
        target_sublattice=None,
        atol=1e-5,
    )
    return enumerator.enumerate(
        k_index=2,
        up_to_parent_conjugacy=True,
        max_depth=None,
    )


@pytest.fixture(scope="module")
def pca21_family_subgroup(
    mns2_family_subgroups: list[FamilySpaceSubgroup],
) -> FamilySpaceSubgroup:
    """Return the translationengleiche Pca2_1 family subgroup of MnS2 (Pa-3)."""
    return next(
        family
        for family in mns2_family_subgroups
        if family.is_translationengleiche
        and SpaceGroup(
            prim_rotations=family.rotations.tolist(),
            prim_translations=family.translations.tolist(),
        ).number
        == 29
    )


@pytest.fixture(scope="module")
def bounded_r3_family_subgroup(
    mns2_family_subgroups: list[FamilySpaceSubgroup],
) -> FamilySpaceSubgroup:
    """Return the bounded R3 family subgroup of MnS2 (Pa-3)."""
    return next(
        family
        for family in mns2_family_subgroups
        if not family.is_translationengleiche
        and SpaceGroup(
            prim_rotations=family.rotations.tolist(),
            prim_translations=family.translations.tolist(),
        ).number
        == 146
    )


def test_parent_normalizer_reduces_fixed_family_representatives(
    pca21_family_subgroup: FamilySpaceSubgroup,
):
    fixed_family_enumerator = SpinSpaceGroupEnumerator(
        prim_rotations=pca21_family_subgroup.rotations.tolist(),
        prim_translations=pca21_family_subgroup.translations.tolist(),
        sublattice=pca21_family_subgroup.relative_sublattice,
    )
    subgroup_enumerator = SpinSpaceSubgroupEnumerator(pca21_family_subgroup)

    _, fixed_family_groups = fixed_family_enumerator.enumerate_spin_space_groups(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
    )
    fixed_family_representatives = [
        spin_space_group
        for _, spin_space_groups in fixed_family_groups
        for spin_space_group in spin_space_groups
    ]
    _, parent_normalizer_representatives = subgroup_enumerator.enumerate_spin_space_groups(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
    )
    _, unreduced_representatives = subgroup_enumerator.enumerate_spin_space_groups(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
        up_to_parent_normalizer_conjugacy=False,
    )

    assert len(fixed_family_representatives) == 4
    assert len(unreduced_representatives) == len(fixed_family_representatives)
    assert len(parent_normalizer_representatives) == 2


def test_parent_normalizer_reduction_is_candidate_order_invariant(
    pca21_family_subgroup: FamilySpaceSubgroup,
    monkeypatch: pytest.MonkeyPatch,
):
    enumerator = SpinSpaceSubgroupEnumerator(pca21_family_subgroup)
    _, expected = enumerator.enumerate_spin_space_groups(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
    )
    fixed_family_enumerator = enumerator._spin_space_group_enumerator
    original_enumerate = fixed_family_enumerator.enumerate_normal_space_subgroups

    def enumerate_in_reverse_order(*args, **kwargs):
        return list(reversed(original_enumerate(*args, **kwargs)))

    monkeypatch.setattr(
        fixed_family_enumerator,
        "enumerate_normal_space_subgroups",
        enumerate_in_reverse_order,
    )
    _, actual = enumerator.enumerate_spin_space_groups(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
    )

    assert len(actual) == len(expected)


def test_parent_normalizer_reduces_before_spin_assignment_enumeration(
    pca21_family_subgroup: FamilySpaceSubgroup,
    monkeypatch: pytest.MonkeyPatch,
):
    enumerator = SpinSpaceSubgroupEnumerator(pca21_family_subgroup)
    fixed_family_enumerator = enumerator._spin_space_group_enumerator
    invariant_subgroups = fixed_family_enumerator.enumerate_normal_space_subgroups(k_index=2)
    original_enumerate = (
        fixed_family_enumerator.enumerate_spin_space_groups_with_normal_space_subgroup
    )
    enumerated_invariant_subgroups = []

    def capture_enumeration(*args, **kwargs):
        enumerated_invariant_subgroups.append(kwargs["normal_space_subgroup"])
        return original_enumerate(*args, **kwargs)

    monkeypatch.setattr(
        fixed_family_enumerator,
        "enumerate_spin_space_groups_with_normal_space_subgroup",
        capture_enumeration,
    )
    _, grouped = enumerator.enumerate_spin_space_groups_by_invariant_subgroup(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
    )

    assert len(grouped) < len(invariant_subgroups)
    assert len(enumerated_invariant_subgroups) == len(grouped)


def test_invariant_subgroup_filter_runs_before_spin_assignment_enumeration(
    pca21_family_subgroup: FamilySpaceSubgroup,
    monkeypatch: pytest.MonkeyPatch,
):
    enumerator = SpinSpaceSubgroupEnumerator(pca21_family_subgroup)
    fixed_family_enumerator = enumerator._spin_space_group_enumerator
    filtered_invariant_subgroups = []

    def reject_invariant_subgroup(invariant_subgroup):
        filtered_invariant_subgroups.append(invariant_subgroup)
        return False

    def fail_if_enumerated(*args, **kwargs):
        raise AssertionError("spin assignments were enumerated for a rejected subgroup")

    monkeypatch.setattr(
        fixed_family_enumerator,
        "enumerate_spin_space_groups_with_normal_space_subgroup",
        fail_if_enumerated,
    )
    _, grouped = enumerator.enumerate_spin_space_groups_by_invariant_subgroup(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
        invariant_subgroup_filter=reject_invariant_subgroup,
    )

    assert filtered_invariant_subgroups
    assert grouped == []


def test_bounded_family_normalizer_uses_family_coordinates(
    bounded_r3_family_subgroup: FamilySpaceSubgroup,
):
    enumerator = SpinSpaceSubgroupEnumerator(bounded_r3_family_subgroup)
    translation_sublattice = bounded_r3_family_subgroup.translation_sublattice
    action = bounded_r3_family_subgroup.normalizer_action

    assert not np.array_equal(translation_sublattice.transformation, np.eye(3, dtype=int))
    parent_rotations, parent_translations = translation_sublattice.transform_operations_to_parent(
        enumerator._normalizer_rotations,
        enumerator._normalizer_translations,
    )
    np.testing.assert_array_equal(parent_rotations, action.rotations)
    np.testing.assert_allclose(parent_translations, action.translations)

    relative_k_index = bounded_r3_family_subgroup.relative_k_index
    assert relative_k_index is not None
    _, representatives = enumerator.enumerate_spin_space_groups(
        SpinOnlyGroupType.NONCOPLANAR,
        k_index=relative_k_index,
    )
    assert len(representatives) == 2
