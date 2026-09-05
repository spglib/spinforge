from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from moyopy import Cell
from spinspg.spin import SpinOnlyGroupType

from spinforge.configuration import SSAGenerator


def _triclinic_parent() -> Cell:
    cell = Cell(
        [[2.1, 0.0, 0.0], [0.2, 2.7, 0.0], [0.3, 0.4, 3.2]],
        [[0.0, 0.0, 0.0]],
        [25],
    )
    return cell


def test_propagation_vector_fixes_one_subgroup_bound():
    cell = _triclinic_parent()
    generator = SSAGenerator.with_propagation_vectors(
        cell,
        [np.array([0.5, 0.0, 0.0])],
        [0],
        multiplicity_preserving=False,
    )

    candidates = [
        candidate
        for candidate in generator._family_subgroups(
            k_index=2,
            up_to_parent_conjugacy=True,
            max_depth=None,
        )
        if not candidate.is_translationengleiche
    ]

    assert len({candidate.translation_key for candidate in candidates}) == 1
    with pytest.raises(ValueError, match="does not match"):
        generator.enumerate(SpinOnlyGroupType.COLLINEAR, k_index=3)


def test_general_family_results_are_returned_in_parent_coordinates():
    cell = _triclinic_parent()
    generator = SSAGenerator(
        cell,
        [0],
        multiplicity_preserving=False,
    )

    results = generator.enumerate(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
        max_depth=None,
    )
    general_results = [
        (nssg, structure) for _, nssg, structure in results if nssg.family_k_index > 1
    ]

    assert general_results
    for nssg, structure in general_results:
        assert nssg.family_k_index == 2
        assert nssg.k_index == 2
        assert nssg.relative_k_index == 1
        assert nssg.sublattice.order == 2
        assert structure.supercell.sublattice.order == 2
        assert nssg.get_invariant_space_subgroup_type().number in {1, 2}
        assert nssg.get_family_space_group_type().number in {1, 2}
        operations, table = nssg.get_full_operations_and_table()
        assert table.shape == (operations.size, operations.size)


def test_default_depth_bound_keeps_bounded_families_separate_from_t_families():
    cell = _triclinic_parent()
    generator = SSAGenerator(
        cell,
        [0],
        multiplicity_preserving=False,
    )

    results = generator.enumerate(SpinOnlyGroupType.COLLINEAR, k_index=2)

    assert any(nssg.family_k_index > 1 for _, nssg, _ in results)


def test_all_families_use_their_own_coordinate_transformation(monkeypatch):
    cell = _triclinic_parent()
    generator = SSAGenerator(cell, [0], multiplicity_preserving=False)
    calls = []

    def capture_call(
        ssgse,
        spin_only_group_type,
        k_index,
        *,
        coordinate_transformation,
    ):
        calls.append(round(abs(np.linalg.det(coordinate_transformation))))
        return []

    monkeypatch.setattr(generator, "_enumerate_with_ssgse", capture_call)

    generator.enumerate(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
        max_depth=None,
    )

    assert any(index == 1 for index in calls)
    assert any(index > 1 for index in calls)


def test_equal_sized_bounded_candidates_are_retained(monkeypatch):
    cell = _triclinic_parent()
    generator = SSAGenerator(cell, [0], multiplicity_preserving=False)
    bounded = next(
        candidate
        for candidate in generator._family_subgroups(
            k_index=2,
            up_to_parent_conjugacy=False,
            max_depth=None,
        )
        if not candidate.is_translationengleiche
    )
    equal_sized_conjugate = replace(bounded)

    def candidates(*, k_index, up_to_parent_conjugacy, max_depth):
        assert not up_to_parent_conjugacy
        assert max_depth is None
        return [] if k_index == 1 else [bounded, equal_sized_conjugate]

    monkeypatch.setattr(generator, "_family_subgroups", candidates)
    monkeypatch.setattr(generator, "_enumerate_with_ssgse", lambda *args, **kwargs: [None])

    results = generator.enumerate(
        SpinOnlyGroupType.COLLINEAR,
        k_index=2,
        up_to_parent_conjugacy=False,
        max_depth=None,
    )

    assert len(results) == 2
