from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from spinforge.space_group import Sublattice
from spinforge.ssg import (
    NontrivialSpinSpaceGroup,
    SpinSpaceSubgroupEnumerator,
    SpinSymmetryOperations,
)


def _point_group(name="D4", permutation=None):
    """Build an exact integer point-group SSG with the same spin representation."""
    rotations = np.rint(Rotation.create_group(name).as_matrix()).astype(int)
    if permutation is not None:
        rotations = rotations[permutation]
    identity = np.eye(3, dtype=int)[None]
    return NontrivialSpinSpaceGroup(
        invariant_rotations=identity,
        invariant_translations=np.zeros((1, 3)),
        sublattice=Sublattice(np.eye(3, dtype=int)),
        spin_translation_coset=SpinSymmetryOperations(identity, np.zeros((1, 3)), identity),
        nontrivial_coset=SpinSymmetryOperations(
            rotations, np.zeros((len(rotations), 3)), rotations.astype(float)
        ),
        spin_planochiral=False,
    )


def _oracle_subgroups(rotations):
    """Exhaust every subset, checking closure with independent spatial products."""
    identity = next(i for i, matrix in enumerate(rotations) if np.array_equal(matrix, np.eye(3)))
    products = {
        (i, j): next(
            k for k, matrix in enumerate(rotations) if np.array_equal(left @ right, matrix)
        )
        for i, left in enumerate(rotations)
        for j, right in enumerate(rotations)
    }
    others = set(range(len(rotations))) - {identity}
    found = set()
    for size in range(len(others) + 1):
        for selection in combinations(sorted(others), size):
            subgroup = frozenset((identity, *selection))
            if all(products[i, j] in subgroup for i, j in product(subgroup, repeat=2)):
                found.add(subgroup)
    return found


@pytest.mark.parametrize("name", ["C4", "D4", "T"])
def test_fixed_subgroups_match_exhaustive_spatial_oracle(name):
    group = _point_group(name)
    operations, _ = group.get_full_operations_and_table()
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(
        group, SpinOnlyGroup.noncoplanar()
    )
    actual = enumerator.enumerate_subgroups()
    assert set(actual) == _oracle_subgroups(operations.rotations)
    assert all(isinstance(subgroup, frozenset) for subgroup in actual)
    assert actual == sorted(actual, key=lambda subgroup: (len(subgroup), tuple(sorted(subgroup))))
    assert len(actual) == len(set(actual))
    # Returned lists are independent and do not mutate subsequent enumeration.
    expected = actual.copy()
    actual.clear()
    assert enumerator.enumerate_subgroups() == expected


def test_all_subgroup_intervals_including_nonnormal_bounds():
    group = _point_group()
    operations, table = group.get_full_operations_and_table()
    all_subgroups = _oracle_subgroups(operations.rotations)
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(
        group, SpinOnlyGroup.noncoplanar()
    )
    # This oracle includes every conjugate order-two subgroup of D4 separately.
    for lower, upper in product(all_subgroups, repeat=2):
        if lower <= upper:
            actual = enumerator.enumerate_subgroups(containing=lower, contained_in=upper)
            assert set(actual) == {
                subgroup for subgroup in all_subgroups if lower <= subgroup <= upper
            }
        else:
            with pytest.raises(ValueError, match="subset"):
                enumerator.enumerate_subgroups(containing=lower, contained_in=upper)
    identity = next(i for i, row in enumerate(table) if np.array_equal(row, np.arange(len(table))))
    assert enumerator.enumerate_subgroups(contained_in={identity}) == [frozenset({identity})]


def test_original_indices_survive_reordering_and_nonzero_identity():
    permutation = np.array([2, 0, 5, 7, 1, 3, 6, 4])
    original = _point_group()
    reordered = _point_group(permutation=permutation)
    original_operations, _ = original.get_full_operations_and_table()
    reordered_operations, _ = reordered.get_full_operations_and_table()
    identity = next(
        i
        for i, matrix in enumerate(reordered_operations.rotations)
        if np.array_equal(matrix, np.eye(3))
    )
    assert identity != 0
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(
        reordered, SpinOnlyGroup.noncoplanar()
    )
    transported = {
        frozenset(int(permutation[i]) for i in subgroup)
        for subgroup in enumerator.enumerate_subgroups()
    }
    assert transported == _oracle_subgroups(original_operations.rotations)
    bound = frozenset(
        i
        for i, matrix in enumerate(reordered_operations.rotations)
        if np.allclose(matrix @ [0, 0, 1], [0, 0, 1])
    )
    assert enumerator.enumerate_subgroups(containing=bound, contained_in=bound) == [bound]


@pytest.mark.parametrize("kind", [SpinOnlyGroupType.COLLINEAR, SpinOnlyGroupType.COPLANAR])
def test_spin_cosets_allow_nonhomomorphic_lifts_in_arbitrary_frame(kind):
    group = _point_group()
    operations, table = group.get_full_operations_and_table()
    if kind == SpinOnlyGroupType.COLLINEAR:
        lifts = Rotation.from_euler(
            "z", np.linspace(0.2, 1.3, operations.size)[:, None]
        ).as_matrix()
    else:
        lifts = np.tile(np.eye(3), (operations.size, 1, 1))
        lifts[::2, 2, 2] = -1
    frame = Rotation.from_rotvec([0.2, -0.4, 0.6]).as_matrix()
    spins = frame @ (operations.spin_rotations @ lifts) @ frame.T
    modified = replace(
        group, nontrivial_coset=replace(group.nontrivial_coset, spin_rotations=spins)
    )
    axis = frame @ np.array([0.0, 0.0, 2.0])
    spin_only_group = SpinOnlyGroup(kind, axis)
    assert any(
        not np.allclose(spins[i] @ spins[j], spins[table[i, j]])
        for i, j in product(range(operations.size), repeat=2)
    )
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(modified, spin_only_group)
    assert set(enumerator.enumerate_subgroups()) == _oracle_subgroups(operations.rotations)
    np.testing.assert_array_equal(modified.nontrivial_coset.spin_rotations, spins)
    np.testing.assert_array_equal(spin_only_group.axis, axis)


def test_translation_cosets_and_parent_coordinate_transport():
    group = _point_group("C2")
    identity = np.eye(3, dtype=int)[None]
    group = replace(
        group,
        sublattice=Sublattice(np.diag([1, 1, 2])),
        spin_translation_coset=SpinSymmetryOperations(
            np.repeat(identity, 2, axis=0),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([np.eye(3), -np.eye(3)]),
        ),
    )
    # C2 point operations times C2 translation cosets give four spatial labels.
    operations, table = group.get_full_operations_and_table()
    assert operations.size == 4
    assert len({tuple(matrix.flat) for matrix in operations.rotations}) == 2
    spin_only_group = SpinOnlyGroup.noncoplanar()
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(group, spin_only_group)
    subgroups = enumerator.enumerate_subgroups()
    assert len(subgroups) == 5  # C2 x C2, including its three order-two subgroups.
    for subgroup in subgroups:
        for i, j in product(subgroup, repeat=2):
            k = int(table[i, j])
            assert k in subgroup
            delta = (
                operations.rotations[i] @ operations.translations[j]
                + operations.translations[i]
                - operations.translations[k]
            )
            fractional = np.linalg.solve(group.sublattice.transformation, delta)
            np.testing.assert_allclose(fractional, np.rint(fractional))
            np.testing.assert_allclose(
                operations.spin_rotations[i] @ operations.spin_rotations[j],
                operations.spin_rotations[k],
            )
    parent_group = group.transform_to_parent(Sublattice(np.diag([2, 3, 1])))
    assert (
        SpinSpaceSubgroupEnumerator.from_spin_space_group(
            parent_group, spin_only_group
        ).enumerate_subgroups()
        == subgroups
    )


@pytest.mark.parametrize("bound", ["containing", "contained_in"])
@pytest.mark.parametrize("bad", [[], [-1], [4], [0.0], [True], "not_closed"])
def test_invalid_bounds(bound, bad):
    group = _point_group("C4")
    if bad == "not_closed":
        operations, _ = group.get_full_operations_and_table()
        identity = next(
            i
            for i, rotation in enumerate(operations.rotations)
            if np.array_equal(rotation, np.eye(3))
        )
        quarter_turn = next(
            i
            for i, rotation in enumerate(operations.rotations)
            if not np.array_equal(rotation @ rotation, np.eye(3))
        )
        bad = [identity, quarter_turn]
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(
        group, SpinOnlyGroup.noncoplanar()
    )
    with pytest.raises(ValueError, match=bound):
        enumerator.enumerate_subgroups(**{bound: bad})


@pytest.mark.parametrize("atol", [0, -1, np.inf, np.nan])
def test_invalid_tolerance(atol):
    with pytest.raises(ValueError, match="atol"):
        SpinSpaceSubgroupEnumerator.from_spin_space_group(
            _point_group(), SpinOnlyGroup.noncoplanar(), atol=atol
        )


@pytest.mark.parametrize("axis", [None, np.zeros(3), np.ones(2), [0, 0, np.inf]])
def test_invalid_spin_only_axis(axis):
    with pytest.raises(ValueError, match="axis"):
        SpinSpaceSubgroupEnumerator.from_spin_space_group(
            _point_group(), SpinOnlyGroup.coplanar(axis)
        )


def test_invalid_spin_assignment_and_spin_only_group():
    group = _point_group()
    with pytest.raises(ValueError, match="magnetic"):
        SpinSpaceSubgroupEnumerator.from_spin_space_group(group, SpinOnlyGroup.nonmagnetic())
    with pytest.raises(ValueError, match="normalize"):
        SpinSpaceSubgroupEnumerator.from_spin_space_group(
            group, SpinOnlyGroup.coplanar(np.ones(3))
        )
    rotations = group.nontrivial_coset.rotations
    nonidentity = next(
        i for i, rotation in enumerate(rotations) if not np.array_equal(rotation, np.eye(3))
    )
    for bad, message in [
        (np.ones((3, 3)), "orthogonal"),
        (np.full((3, 3), np.nan), "finite"),
        (Rotation.from_rotvec([0.13, 0.27, 0.39]).as_matrix(), "close"),
    ]:
        spins = group.nontrivial_coset.spin_rotations.copy()
        spins[nonidentity] = bad
        invalid = replace(
            group, nontrivial_coset=replace(group.nontrivial_coset, spin_rotations=spins)
        )
        with pytest.raises(ValueError, match=message):
            SpinSpaceSubgroupEnumerator.from_spin_space_group(invalid, SpinOnlyGroup.noncoplanar())


def test_fixed_mode_rejects_family_methods_and_properties():
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(
        _point_group(), SpinOnlyGroup.noncoplanar()
    )
    for name in ("family_space_subgroup", "prim_rotations", "prim_translations"):
        with pytest.raises(ValueError, match="FamilySpaceSubgroup"):
            getattr(enumerator, name)
    for name in (
        "enumerate_spin_space_groups",
        "enumerate_spin_space_groups_by_invariant_subgroup",
    ):
        with pytest.raises(ValueError, match="FamilySpaceSubgroup"):
            getattr(enumerator, name)(SpinOnlyGroupType.NONCOPLANAR, k_index=1)


def test_nonsymmorphic_screw_and_half_translation_keep_distinct_labels():
    group = _point_group("C2")
    translations = np.array(
        [
            np.zeros(3) if np.array_equal(rotation, np.eye(3)) else [0.0, 0.0, 0.5]
            for rotation in group.nontrivial_coset.rotations
        ]
    )
    group = replace(
        group, nontrivial_coset=replace(group.nontrivial_coset, translations=translations)
    )
    operations, _ = group.get_full_operations_and_table()
    identity = next(
        i for i, rotation in enumerate(operations.rotations) if np.array_equal(rotation, np.eye(3))
    )
    screw = next(
        i
        for i, rotation in enumerate(operations.rotations)
        if not np.array_equal(rotation, np.eye(3))
    )
    np.testing.assert_allclose(
        operations.rotations[screw] @ operations.translations[screw]
        + operations.translations[screw],
        [0, 0, 1],
    )
    enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(
        group, SpinOnlyGroup.noncoplanar()
    )
    assert enumerator.enumerate_subgroups() == [
        frozenset({identity}),
        frozenset({identity, screw}),
    ]


def test_spin_coset_validation_uses_absolute_tolerance():
    group = _point_group("C2")
    spins = group.nontrivial_coset.spin_rotations.copy()
    nonidentity = next(
        i
        for i, rotation in enumerate(group.nontrivial_coset.rotations)
        if not np.array_equal(rotation, np.eye(3))
    )
    spins[nonidentity, 0, 0] += 2e-6
    perturbed = replace(
        group, nontrivial_coset=replace(group.nontrivial_coset, spin_rotations=spins)
    )
    accepted = SpinSpaceSubgroupEnumerator.from_spin_space_group(
        perturbed, SpinOnlyGroup.noncoplanar(), atol=1e-5
    )
    assert len(accepted.enumerate_subgroups()) == 2
    with pytest.raises(ValueError, match="orthogonal"):
        SpinSpaceSubgroupEnumerator.from_spin_space_group(
            perturbed, SpinOnlyGroup.noncoplanar(), atol=1e-7
        )
