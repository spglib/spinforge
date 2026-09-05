from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import pytest
from moyopy import Cell, MoyoDataset, operations_from_number
from spinspg.spin import SpinOnlyGroupType

from spinforge.space_group import Sublattice
from spinforge.ssg import (
    NontrivialSpinSpaceGroup,
    SpinSpaceGroupEnumerator,
    SpinSymmetryOperations,
)
from spinforge.testing import load_prim_CoTa3S6, load_prim_Mn3Sn


def validate_nontrivial_spin_space_group(nssg: NontrivialSpinSpaceGroup, atol: float = 1e-5):
    """Validate an enumerated SSG against the actual operation algebra.

    `get_full_operations_and_table` builds the table from actual matrix products, so
    checking the spin assignment against it verifies the spin rotations form a
    homomorphism with respect to real space products.
    """
    operations, table = nssg.get_full_operations_and_table()
    size = operations.size

    # The table is a valid multiplication table (Latin square)
    for i in range(size):
        assert len(set(table[i, :].tolist())) == size
        assert len(set(table[:, i].tolist())) == size

    # Spin rotations form a homomorphism w.r.t. actual space products
    for i, j in product(range(size), repeat=2):
        k = int(table[i, j])
        assert np.allclose(
            operations.spin_rotations[i] @ operations.spin_rotations[j],
            operations.spin_rotations[k],
            atol=atol,
        )


def _flatten_spin_space_groups(grouped_spin_space_groups):
    return [
        spin_space_group
        for _, spin_space_groups in grouped_spin_space_groups
        for spin_space_group in spin_space_groups
    ]


@pytest.mark.parametrize(
    ("load_prim_cell", "spin_only_group_type", "k_index"),
    [
        pytest.param(
            load_prim_Mn3Sn,
            SpinOnlyGroupType.COPLANAR,
            1,
            id="Mn3Sn-coplanar-k1",
        ),
        pytest.param(
            load_prim_CoTa3S6,
            SpinOnlyGroupType.NONCOPLANAR,
            4,
            id="CoTa3S6-noncoplanar-k4",
        ),
    ],
)
def test_ssg_enumerator_material_regression(
    load_prim_cell: Callable[[], Cell],
    spin_only_group_type: SpinOnlyGroupType,
    k_index: int,
):
    prim_cell = load_prim_cell()
    prim_dataset = MoyoDataset(prim_cell)

    ssge = SpinSpaceGroupEnumerator(
        prim_rotations=np.asarray(prim_dataset.operations.rotations).tolist(),
        prim_translations=np.asarray(prim_dataset.operations.translations).tolist(),
    )

    _, grouped_spin_space_groups = ssge.enumerate_spin_space_groups(
        spin_only_group_type=spin_only_group_type,
        k_index=k_index,
    )
    nontrivial_spin_space_groups = _flatten_spin_space_groups(grouped_spin_space_groups)
    assert len(nontrivial_spin_space_groups) > 0
    for nssg in nontrivial_spin_space_groups:
        validate_nontrivial_spin_space_group(nssg)


def test_ssg_enumerator_groups_results_by_invariant_subgroup():
    operations = operations_from_number(75, primitive=True)
    enumerator = SpinSpaceGroupEnumerator(
        prim_rotations=np.asarray(operations.rotations, dtype=np.int64).tolist(),
        prim_translations=np.asarray(operations.translations, dtype=float).tolist(),
    )

    _, grouped = enumerator.enumerate_spin_space_groups(
        SpinOnlyGroupType.NONCOPLANAR,
        k_index=2,
    )

    assert grouped
    assert all(spin_space_groups for _, spin_space_groups in grouped)
    for invariant_subgroup, spin_space_groups in grouped:
        for spin_space_group in spin_space_groups:
            assert spin_space_group.sublattice == invariant_subgroup.sublattice
            np.testing.assert_array_equal(
                spin_space_group.invariant_rotations,
                enumerator.prim_rotations[invariant_subgroup.point_subgroup],
            )
            np.testing.assert_allclose(
                spin_space_group.invariant_translations,
                invariant_subgroup.translations,
            )


def test_invariant_space_subgroup_type_k_index_2():
    # Regression test for sublattice-basis transformation: for P4 (75) with k_index=2,
    # the adjusted shift (0, 0, 1) becomes (0, 0, 1/2) in the c-doubled sublattice basis,
    # so the invariant space subgroup is P4_2 (77); the body-centered sublattice yields
    # I4 (79). Without the basis transformation both were misidentified as P4 (75).
    prim_operations = operations_from_number(75, primitive=True)
    ssge = SpinSpaceGroupEnumerator(
        prim_rotations=np.array(prim_operations.rotations).tolist(),
        prim_translations=np.array(prim_operations.translations).tolist(),
    )
    _, grouped_spin_space_groups = ssge.enumerate_spin_space_groups(
        SpinOnlyGroupType.NONCOPLANAR,
        k_index=2,
    )
    nontrivial_spin_space_groups = _flatten_spin_space_groups(grouped_spin_space_groups)
    numbers = {
        nssg.get_invariant_space_subgroup_type().number for nssg in nontrivial_spin_space_groups
    }
    assert 77 in numbers
    assert 79 in numbers
    # Every candidate is a t-family here: the family space group is the parent P4 itself.
    assert all(
        nssg.get_family_space_group_type().number == 75 for nssg in nontrivial_spin_space_groups
    )


def test_family_space_group_type_uses_family_lattice():
    # Regression test for family-sublattice transformation: a P4_2 (77) family group
    # enumerated in its own (c-doubled) basis keeps its type after transform_to_parent.
    # The 4_2 screw translation (0, 0, 1/2) becomes (0, 0, 1) in the parent basis, so
    # identifying the parent-basis operations without transforming into the family
    # lattice would misread the screw as a lattice translation and return P4 (75).
    family_operations = operations_from_number(77, primitive=True)
    rotations = np.array(family_operations.rotations)
    translations = np.array(family_operations.translations)
    identity_op = SpinSymmetryOperations(
        rotations=np.eye(3, dtype=np.int64)[None],
        translations=np.zeros((1, 3)),
        spin_rotations=np.eye(3)[None],
    )
    nssg = NontrivialSpinSpaceGroup(
        invariant_rotations=np.eye(3, dtype=np.int64)[None],
        invariant_translations=np.zeros((1, 3)),
        sublattice=Sublattice(np.eye(3, dtype=np.int64)),
        spin_translation_coset=identity_op,
        nontrivial_coset=SpinSymmetryOperations(
            rotations=rotations,
            translations=translations,
            spin_rotations=np.array([np.eye(3)] * len(rotations)),
        ),
        spin_planochiral=False,
    )
    assert nssg.get_family_space_group_type().number == 77

    transformed = nssg.transform_to_parent(Sublattice(np.diag([1, 1, 2])))
    assert transformed.family_k_index == 2
    assert transformed.get_family_space_group_type().number == 77


def test_full_operations_are_independent_between_calls():
    family_operations = operations_from_number(77, primitive=True)
    rotations = np.array(family_operations.rotations)
    translations = np.array(family_operations.translations)
    identity_op = SpinSymmetryOperations(
        rotations=np.eye(3, dtype=np.int64)[None],
        translations=np.zeros((1, 3)),
        spin_rotations=np.eye(3)[None],
    )
    nssg = NontrivialSpinSpaceGroup(
        invariant_rotations=np.eye(3, dtype=np.int64)[None],
        invariant_translations=np.zeros((1, 3)),
        sublattice=Sublattice(np.eye(3, dtype=np.int64)),
        spin_translation_coset=identity_op,
        nontrivial_coset=SpinSymmetryOperations(
            rotations=rotations,
            translations=translations,
            spin_rotations=np.array([np.eye(3)] * len(rotations)),
        ),
        spin_planochiral=False,
    )

    first_operations, first_table = nssg.get_full_operations_and_table()
    second_operations, second_table = nssg.get_full_operations_and_table()

    np.testing.assert_array_equal(first_operations.rotations, second_operations.rotations)
    np.testing.assert_array_equal(first_operations.translations, second_operations.translations)
    np.testing.assert_array_equal(
        first_operations.spin_rotations, second_operations.spin_rotations
    )
    np.testing.assert_array_equal(first_table, second_table)
    assert not np.shares_memory(first_operations.rotations, second_operations.rotations)
    assert not np.shares_memory(first_operations.translations, second_operations.translations)
    assert not np.shares_memory(first_operations.spin_rotations, second_operations.spin_rotations)
    assert not np.shares_memory(first_table, second_table)


@pytest.mark.parametrize(
    "number",
    list(range(1, 230 + 1)),
)
@pytest.mark.parametrize(
    "spin_only_group_type",
    [
        SpinOnlyGroupType.COLLINEAR,
        SpinOnlyGroupType.COPLANAR,
        SpinOnlyGroupType.NONCOPLANAR,
    ],
)
@pytest.mark.slow
def test_ssg_enumerator_k_index_1(number: int, spin_only_group_type: SpinOnlyGroupType):
    prim_operations = operations_from_number(number, primitive=True)
    ssge = SpinSpaceGroupEnumerator(
        prim_rotations=np.array(prim_operations.rotations).tolist(),
        prim_translations=np.array(prim_operations.translations).tolist(),
    )
    _, grouped_spin_space_groups = ssge.enumerate_spin_space_groups(
        spin_only_group_type,
        k_index=1,
    )
    nontrivial_spin_space_groups = _flatten_spin_space_groups(grouped_spin_space_groups)
    assert len(nontrivial_spin_space_groups) > 0
    for nssg in nontrivial_spin_space_groups:
        validate_nontrivial_spin_space_group(nssg)


def test_transform_spin_space_group_to_parent_includes_spatial_parts():
    family_to_parent = Sublattice(np.array([[1, 0, 0], [0, 2, 0], [0, 1, 2]]))
    family_rotations = np.array([[[-1, 0, 0], [0, -1, 0], [0, 1, 1]]])
    family_translations = np.array([[0.25, 0.5, 0.75]])
    spin_rotations = np.array([np.diag([-1.0, -1.0, 1.0])])
    operations = SpinSymmetryOperations(
        rotations=family_rotations,
        translations=family_translations,
        spin_rotations=spin_rotations,
    )
    spin_space_group = NontrivialSpinSpaceGroup(
        invariant_rotations=family_rotations,
        invariant_translations=family_translations,
        sublattice=Sublattice(np.diag([1, 1, 2])),
        spin_translation_coset=operations,
        nontrivial_coset=operations,
        spin_planochiral=True,
    )

    assert spin_space_group.family_sublattice == Sublattice(np.eye(3, dtype=np.int64))

    transformed_operations = operations.transform_to_parent(family_to_parent)
    transformed_spin_space_group = spin_space_group.transform_to_parent(family_to_parent)

    expected_rotations = np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
    expected_translations = np.array([[0.25, 1.0, 2.0]])
    np.testing.assert_array_equal(transformed_operations.rotations, expected_rotations)
    np.testing.assert_allclose(transformed_operations.translations, expected_translations)
    np.testing.assert_array_equal(transformed_operations.spin_rotations, spin_rotations)
    np.testing.assert_array_equal(
        transformed_spin_space_group.invariant_rotations, expected_rotations
    )
    np.testing.assert_allclose(
        transformed_spin_space_group.invariant_translations, expected_translations
    )
    np.testing.assert_array_equal(
        transformed_spin_space_group.spin_translation_coset.rotations,
        expected_rotations,
    )
    np.testing.assert_array_equal(
        transformed_spin_space_group.nontrivial_coset.rotations,
        expected_rotations,
    )
    np.testing.assert_array_equal(
        transformed_spin_space_group.sublattice.transformation,
        family_to_parent.transformation @ np.diag([1, 1, 2]),
    )
    assert transformed_spin_space_group.family_sublattice == family_to_parent
    assert transformed_spin_space_group.spin_planochiral
