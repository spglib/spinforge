from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from _family_subgroup_helpers import (
    _cubic_parent,
    _enumerator,
    _rotated_translation_coset_subgroup,
    _tetragonal_parent,
    _translation_coset_subgroup,
)

from spinforge.space_group import Sublattice, enumerate_sublattices
from spinforge.space_group._family_subgroup import (
    _are_conjugate_subgroups,
    _are_conjugate_sublattices,
)


def test_general_subgroups_are_k_subgroups_of_hermann_t_subgroups():
    enumerator = _enumerator()

    subgroups = enumerator.enumerate(
        k_index=2,
        up_to_parent_conjugacy=True,
        max_depth=None,
    )
    bounded = [subgroup for subgroup in subgroups if not subgroup.is_translationengleiche]
    parent_point_order = len(enumerator.parent_rotations)

    assert len({subgroup.translation_key for subgroup in bounded}) == 7
    assert any(len(subgroup.hermann_subgroup) == parent_point_order for subgroup in bounded)
    assert any(len(subgroup.hermann_subgroup) < parent_point_order for subgroup in bounded)
    assert all(len(subgroup.rotations) == len(subgroup.hermann_subgroup) for subgroup in bounded)
    assert all(subgroup.relative_k_index == 1 for subgroup in bounded)
    for subgroup in bounded:
        assert all(
            subgroup.translation_sublattice.is_normal(rotation)
            for rotation in subgroup.normalizer_action.rotations
        )
        for local_index, parent_index in enumerate(subgroup.operation_indices):
            np.testing.assert_array_equal(
                subgroup.parent_rotations[local_index],
                enumerator.parent_rotations[parent_index],
            )
            translation_difference = (
                subgroup.parent_translations[local_index]
                - enumerator.parent_translations[parent_index]
            )
            np.testing.assert_allclose(
                translation_difference,
                np.rint(translation_difference),
            )
    translationengleiche = {
        subgroup.hermann_subgroup: subgroup
        for subgroup in subgroups
        if subgroup.is_translationengleiche
    }
    assert all(
        subgroup.is_subgroup_of(
            translationengleiche[subgroup.hermann_subgroup],
            atol=1e-5,
        )
        for subgroup in bounded
    )
    assert all(
        not translationengleiche[subgroup.hermann_subgroup].is_subgroup_of(
            subgroup,
            atol=1e-5,
        )
        for subgroup in bounded
    )


def test_up_to_parent_conjugacy_is_true_by_default():
    enumerator = _enumerator(parent=_tetragonal_parent())

    default = enumerator.enumerate(k_index=2, max_depth=None)
    explicit = enumerator.enumerate(
        k_index=2,
        up_to_parent_conjugacy=True,
        max_depth=None,
    )

    assert default == explicit


def test_affine_containment_checks_operations_after_lattice_containment():
    subgroup = next(
        subgroup
        for subgroup in _enumerator().translationengleiche_subgroups
        if len(subgroup.parent_rotations) > 1
    )
    missing_rotation = replace(
        subgroup,
        parent_rotations=subgroup.parent_rotations[:1],
        parent_translations=subgroup.parent_translations[:1],
    )
    shifted_translations = replace(
        subgroup,
        parent_translations=subgroup.parent_translations + np.array([0.5, 0.0, 0.0]),
    )

    assert subgroup.is_subgroup_of(subgroup, atol=1e-5)
    assert not subgroup.is_subgroup_of(missing_rotation, atol=1e-5)
    assert not subgroup.is_subgroup_of(shifted_translations, atol=1e-5)


def test_family_space_subgroup_equality_uses_identity():
    subgroup = _enumerator().translationengleiche_subgroups[0]
    equivalent_fields = replace(subgroup)

    assert subgroup != equivalent_fields
    assert len({subgroup, equivalent_fields}) == 2


def test_k_index_restricts_family_and_relative_translation_indices():
    bounded = [
        subgroup
        for subgroup in _enumerator().enumerate(
            k_index=4,
            up_to_parent_conjugacy=True,
            max_depth=0,
        )
        if not subgroup.is_translationengleiche
    ]

    assert {subgroup.translation_sublattice.order for subgroup in bounded} == {2, 4}
    assert {subgroup.relative_k_index for subgroup in bounded} == {1, 2}
    for subgroup in bounded:
        assert subgroup.relative_k_index is not None
        assert subgroup.translation_sublattice.order * subgroup.relative_k_index == 4


def test_target_sublattice_restricts_family_translation_lattices():
    target = Sublattice(np.diag([2, 1, 1]))
    bounded = [
        subgroup
        for subgroup in _enumerator(sublattice=target).enumerate(
            k_index=2,
            up_to_parent_conjugacy=True,
            max_depth=None,
        )
        if not subgroup.is_translationengleiche
    ]

    assert {subgroup.translation_key for subgroup in bounded} == {
        tuple(target.transformation.ravel().tolist())
    }


def test_target_sublattice_restricts_hermann_conjugacy_reduction():
    target = Sublattice(np.diag([2, 1, 1]))
    enumerator = _enumerator(sublattice=target, parent=_cubic_parent())
    reduced = [
        subgroup
        for subgroup in enumerator.enumerate(
            k_index=2,
            up_to_parent_conjugacy=True,
            max_depth=None,
        )
        if subgroup.is_translationengleiche
    ]
    unreduced = [
        subgroup
        for subgroup in enumerator.enumerate(
            k_index=2,
            up_to_parent_conjugacy=False,
            max_depth=None,
        )
        if subgroup.is_translationengleiche
    ]
    stabilizer = [
        index
        for index, rotation in enumerate(enumerator.parent_rotations)
        if target.is_normal(rotation)
    ]

    assert len(reduced) < len(unreduced)
    assert all(
        any(
            _are_conjugate_subgroups(
                subgroup.hermann_subgroup,
                representative.hermann_subgroup,
                stabilizer,
                enumerator._parent_table,
            )
            for representative in reduced
        )
        for subgroup in unreduced
    )


def test_family_lattices_are_reduced_by_hermann_normalizer():
    enumerator = _enumerator(parent=_cubic_parent())
    trivial_hermann_group = next(
        subgroup
        for subgroup in enumerator.translationengleiche_subgroups
        if len(subgroup.hermann_subgroup) == 1
    )
    family_sublattices = [(sublattice, None) for sublattice in enumerate_sublattices(2)]
    representatives = enumerator._reduce_family_sublattices(
        family_sublattices,
        trivial_hermann_group.hermann_normalizer,
    )

    assert len(representatives) == 3


def test_parent_conjugacy_reduces_family_lattices_by_hermann_normalizer():
    def trivial_bounded_subgroups(*, up_to_parent_conjugacy: bool):
        enumerator = _enumerator(parent=_tetragonal_parent())
        subgroups = enumerator.enumerate(
            k_index=2,
            up_to_parent_conjugacy=up_to_parent_conjugacy,
            max_depth=None,
        )
        return enumerator, [
            subgroup
            for subgroup in subgroups
            if not subgroup.is_translationengleiche and len(subgroup.hermann_subgroup) == 1
        ]

    reduced_enumerator, reduced = trivial_bounded_subgroups(up_to_parent_conjugacy=True)
    _, unreduced = trivial_bounded_subgroups(up_to_parent_conjugacy=False)

    assert len(reduced) == 5
    assert len(unreduced) == 7
    assert {subgroup.translation_key for subgroup in reduced} < {
        subgroup.translation_key for subgroup in unreduced
    }
    assert all(
        any(
            _are_conjugate_sublattices(
                subgroup.translation_sublattice,
                representative.translation_sublattice,
                reduced_enumerator.parent_rotations,
                atol=1e-5,
            )
            for representative in reduced
        )
        for subgroup in unreduced
    )


def test_bounded_subgroups_inherit_hermann_depth():
    subgroups = _enumerator().enumerate(
        k_index=2,
        up_to_parent_conjugacy=True,
        max_depth=None,
    )
    translationengleiche_depths = {
        subgroup.hermann_subgroup: subgroup.depth
        for subgroup in subgroups
        if subgroup.is_translationengleiche
    }

    assert all(
        subgroup.depth == translationengleiche_depths[subgroup.hermann_subgroup]
        for subgroup in subgroups
        if not subgroup.is_translationengleiche
    )


def test_max_depth_one_retains_parent_and_maximal_hermann_subgroups():
    for k_index in [1, 2]:
        subgroups = _enumerator().enumerate(
            k_index=k_index,
            up_to_parent_conjugacy=True,
            max_depth=1,
        )

        assert all(subgroup.depth <= 1 for subgroup in subgroups)
        assert any(subgroup.depth == 1 for subgroup in subgroups)


def test_bounded_normal_listing_uses_full_parent_normalizer():
    enumerator = _enumerator(parent=_tetragonal_parent())
    candidates = enumerator.enumerate(
        k_index=2,
        up_to_parent_conjugacy=True,
        max_depth=1,
    )

    normal = enumerator.enumerate_normal(
        k_index=2,
        max_depth=1,
    )

    def parent_index(subgroup):
        point_index = len(enumerator.parent_rotations) // len(subgroup.parent_rotations)
        return point_index * subgroup.translation_sublattice.order

    excluded = next(
        subgroup
        for subgroup in candidates
        if len(subgroup.normalizer_action) < parent_index(subgroup)
    )
    assert excluded not in normal
    assert all(len(subgroup.normalizer_action) == parent_index(subgroup) for subgroup in normal)
    assert any(
        not subgroup.is_translationengleiche
        and len(subgroup.hermann_subgroup) < len(enumerator.parent_rotations)
        for subgroup in normal
    )
    assert all(subgroup.depth <= 1 for subgroup in normal)


@pytest.mark.parametrize("k_index", [0, -1])
def test_normal_listing_rejects_nonpositive_k_index(k_index):
    with pytest.raises(ValueError, match=f"k_index must be >= 1, got {k_index}"):
        _enumerator().enumerate_normal(k_index=k_index)


def test_enumeration_orders_translationengleiche_before_bounded_subgroups():
    subgroups = _enumerator().enumerate(
        k_index=4,
        up_to_parent_conjugacy=True,
        max_depth=None,
    )
    first_bounded = next(
        index for index, subgroup in enumerate(subgroups) if not subgroup.is_translationengleiche
    )
    bounded = subgroups[first_bounded:]

    assert all(subgroup.is_translationengleiche for subgroup in subgroups[:first_bounded])
    assert all(not subgroup.is_translationengleiche for subgroup in bounded)
    assert bounded == sorted(
        bounded,
        key=lambda subgroup: (
            subgroup.translation_sublattice.order,
            -len(subgroup.hermann_subgroup),
        ),
    )


def test_negative_max_depth_raises():
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        _enumerator().enumerate(
            up_to_parent_conjugacy=True,
            max_depth=-1,
        )


def test_parent_normalizer_includes_operations_outside_hermann_group():
    _, subgroup = _translation_coset_subgroup()

    action = subgroup.normalizer_action
    assert len(action) == 4
    assert action.permutations == ((0,), (0,), (0,), (0,))
    assert any(np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in action.rotations)
    assert any(np.allclose(translation, [1.0, 0.0, 0.0]) for translation in action.translations)


def test_translationengleiche_normalizer_uses_hermann_coset_representatives():
    enumerator = _enumerator(parent=_cubic_parent())

    for subgroup in enumerator.translationengleiche_subgroups:
        representative_indices = list(subgroup.hermann_normalizer)
        np.testing.assert_array_equal(
            subgroup.normalizer_action.rotations,
            enumerator.parent_rotations[representative_indices],
        )
        np.testing.assert_allclose(
            subgroup.normalizer_action.translations,
            enumerator.parent_translations[representative_indices],
        )


def test_parent_normalizer_handles_rotated_translation_cosets():
    _, subgroup = _rotated_translation_coset_subgroup()

    assert len(subgroup.normalizer_action) == 8
    assert subgroup.normalizer_action.permutations == ((0, 1),) * 8
