from __future__ import annotations

from itertools import permutations

import numpy as np
from spgrep.rep.group import get_inverse_index
from spgrep.symmetry.group import get_cayley_table
from spinspg.spin import SpinOnlyGroupType

from spinforge.msg import (
    ConstructType,
    MagneticSpaceSubgroup,
    _get_triclinic_magnetic_space_subgroups,
)
from spinforge.msg._msg import classify_magnetic_space_subgroup_conjugacy_classes
from spinforge.ssg import SpinSymmetryOperations


def test_orient_moments_basis_rotates_each_field() -> None:
    # 90-degree rotation about z; moment rows transform as m' = m @ Q^T = Q m.
    q = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    msg = MagneticSpaceSubgroup(xsg=[0], fsg=[0], msg_type=ConstructType.TYPE1, Q=q)
    basis = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]),
    ]

    oriented = msg.orient_moments_basis(basis)

    assert len(oriented) == len(basis)
    np.testing.assert_allclose(oriented[0], [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], atol=1e-12)
    # Moments along the rotation axis are unchanged.
    np.testing.assert_allclose(oriented[1], basis[1], atol=1e-12)


def test_triclinic_msg_rounds_spin_rotation_determinant() -> None:
    operations = SpinSymmetryOperations(
        rotations=np.eye(3, dtype=int)[None, :, :],
        translations=np.zeros((1, 3)),
        spin_rotations=((1 - 1e-15) * np.eye(3))[None, :, :],
    )

    msg = _get_triclinic_magnetic_space_subgroups(
        operations,
        SpinOnlyGroupType.COPLANAR,
        atol=1e-5,
    )

    assert msg.xsg == [0]
    assert msg.fsg == [0]
    assert msg.msg_type == ConstructType.TYPE1


def test_msg_conjugacy_classes_retain_family_group_transformations() -> None:
    family_operations = []
    for permutation in permutations(range(3)):
        operation = np.zeros((3, 3), dtype=int)
        operation[range(3), permutation] = 1
        family_operations.append(operation)
    family_operations = np.asarray(family_operations)
    table = get_cayley_table(family_operations)
    reflection_indices = [index for index in range(1, len(table)) if table[index, index] == 0]
    assert len(reflection_indices) == 3
    candidates = [
        MagneticSpaceSubgroup(
            xsg=[0, reflection_index],
            fsg=[0, reflection_index],
            msg_type=ConstructType.TYPE1,
            Q=np.eye(3),
        )
        for reflection_index in reflection_indices
    ]

    equivalence_classes = classify_magnetic_space_subgroup_conjugacy_classes(candidates, table)

    assert len(equivalence_classes) == 1
    equivalence_class = equivalence_classes[0]
    assert equivalence_class.representative is candidates[0]
    assert {id(obj) for obj in equivalence_class.equivalent_objects} == {
        id(candidate) for candidate in candidates
    }
    assert sum(
        len(candidate_class.equivalent_objects) for candidate_class in equivalence_classes
    ) == len(candidates)
    assert all(
        sum(
            obj is candidate
            for candidate_class in equivalence_classes
            for obj in candidate_class.equivalent_objects
        )
        == 1
        for candidate in candidates
    )
    assert len(equivalence_class.stabilizer) == 2
    representative_key = (
        frozenset(equivalence_class.representative.xsg),
        frozenset(equivalence_class.representative.fsg),
    )
    for equivalent_object, family_operation_index in zip(
        equivalence_class.equivalent_objects,
        equivalence_class.transformations_to_representative,
    ):
        inverse = get_inverse_index(table, family_operation_index)
        conjugated = (
            frozenset(
                int(table[table[family_operation_index, h], inverse])
                for h in equivalent_object.xsg
            ),
            frozenset(
                int(table[table[family_operation_index, h], inverse])
                for h in equivalent_object.fsg
            ),
        )
        assert conjugated == representative_key
    for family_operation_index in equivalence_class.stabilizer:
        inverse = get_inverse_index(table, family_operation_index)
        conjugated = (
            frozenset(
                int(table[table[family_operation_index, h], inverse])
                for h in equivalence_class.representative.xsg
            ),
            frozenset(
                int(table[table[family_operation_index, h], inverse])
                for h in equivalence_class.representative.fsg
            ),
        )
        assert conjugated == representative_key
