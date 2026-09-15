from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spgrep.symmetry.group import get_cayley_table

from spinforge.utils.group_theory import enumerate_subgroups


def _point_group(name="D4", permutation=None):
    """Return integer rotation matrices and their multiplication table."""
    rotations = np.rint(Rotation.create_group(name).as_matrix()).astype(int)
    if permutation is not None:
        rotations = rotations[permutation]
    return rotations, get_cayley_table(rotations)


def _oracle_subgroups(rotations):
    """Exhaust every subset, checking closure with independent matrix products."""
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


@pytest.mark.parametrize("name", ["C1", "C4", "D4", "T"])
def test_subgroups_match_exhaustive_matrix_oracle(name):
    rotations, table = _point_group(name)
    actual = enumerate_subgroups(table)
    assert set(actual) == _oracle_subgroups(rotations)
    assert all(isinstance(subgroup, frozenset) for subgroup in actual)
    assert actual == sorted(actual, key=lambda subgroup: (len(subgroup), tuple(sorted(subgroup))))
    assert len(actual) == len(set(actual))
    # Returned lists are independent and do not mutate subsequent enumeration.
    expected = actual.copy()
    actual.clear()
    assert enumerate_subgroups(table) == expected


def test_all_subgroup_intervals_including_nonnormal_bounds():
    rotations, table = _point_group()
    all_subgroups = _oracle_subgroups(rotations)
    # This oracle includes every conjugate order-two subgroup of D4 separately.
    for lower, upper in product(all_subgroups, repeat=2):
        if lower <= upper:
            actual = enumerate_subgroups(table, containing=lower, contained_in=upper)
            assert set(actual) == {
                subgroup for subgroup in all_subgroups if lower <= subgroup <= upper
            }
        else:
            with pytest.raises(ValueError, match="subset"):
                enumerate_subgroups(table, containing=lower, contained_in=upper)
    for bound in all_subgroups:
        assert set(enumerate_subgroups(table, containing=bound)) == {
            subgroup for subgroup in all_subgroups if bound <= subgroup
        }
        assert set(enumerate_subgroups(table, contained_in=bound)) == {
            subgroup for subgroup in all_subgroups if subgroup <= bound
        }


def test_original_indices_survive_reordering_and_nonzero_identity():
    permutation = np.array([2, 0, 5, 7, 1, 3, 6, 4])
    original, _ = _point_group()
    reordered, table = _point_group(permutation=permutation)
    identity = next(i for i, matrix in enumerate(reordered) if np.array_equal(matrix, np.eye(3)))
    assert identity != 0
    transported = {
        frozenset(int(permutation[i]) for i in subgroup) for subgroup in enumerate_subgroups(table)
    }
    assert transported == _oracle_subgroups(original)
    bound = frozenset(
        i for i, matrix in enumerate(reordered) if np.array_equal(matrix @ [0, 0, 1], [0, 0, 1])
    )
    assert enumerate_subgroups(table, containing=bound, contained_in=bound) == [bound]
    assert enumerate_subgroups(table, contained_in={identity}) == [frozenset({identity})]


@pytest.mark.parametrize("bound", ["containing", "contained_in"])
@pytest.mark.parametrize(
    "bad", [[], [-1], [4], [0.0], [True], [np.bool_(True)], "not_closed", "missing_identity"]
)
def test_invalid_bounds(bound, bad):
    rotations, table = _point_group("C4")
    if isinstance(bad, str):
        identity = next(
            i for i, rotation in enumerate(rotations) if np.array_equal(rotation, np.eye(3))
        )
        quarter_turn = next(
            i
            for i, rotation in enumerate(rotations)
            if not np.array_equal(rotation @ rotation, np.eye(3))
        )
        bad = [identity, quarter_turn] if bad == "not_closed" else [quarter_turn]
    with pytest.raises(ValueError, match=bound):
        enumerate_subgroups(table, **{bound: bad})


def test_readonly_table_and_numpy_integer_bounds_are_unchanged():
    rotations, table = _point_group()
    upper = np.array(
        [i for i, matrix in enumerate(rotations) if np.array_equal(matrix @ [0, 0, 1], [0, 0, 1])]
    )
    lower = np.array([i for i in upper if np.array_equal(rotations[i] @ rotations[i], np.eye(3))])
    originals = [value.copy() for value in (table, lower, upper)]
    for value in (table, lower, upper):
        value.flags.writeable = False
    actual = enumerate_subgroups(table, containing=lower, contained_in=upper)
    assert set(actual) == {
        subgroup
        for subgroup in _oracle_subgroups(rotations)
        if set(lower) <= subgroup <= set(upper)
    }
    for value, original in zip((table, lower, upper), originals, strict=True):
        np.testing.assert_array_equal(value, original)
