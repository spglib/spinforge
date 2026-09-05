import numpy as np
import pytest

from spinforge.space_group import Sublattice, enumerate_sublattices


def test_sublattice():
    sublattice = Sublattice(
        np.array(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
    )
    assert len(sublattice.generators) == 2
    assert len(sublattice.lattice_points) == 4


def test_sublattice_2():
    sublattice = Sublattice(
        np.array(
            [
                [1, 0, 0],
                [0, 2, 0],
                [0, 1, 2],
            ]
        )
    )
    assert len(sublattice.generators) == 1  # SNF[1, 1, 4]
    assert len(sublattice.lattice_points) == 4
    assert sublattice.convert_to_factor((0, -3, 0)) == (0, 0, 3)


def test_enumerate_sublattices_index_two():
    sublattices = enumerate_sublattices(2)

    assert len(sublattices) == 7
    assert all(sublattice.order == 2 for sublattice in sublattices)
    assert len({tuple(sublattice.transformation.ravel()) for sublattice in sublattices}) == 7


def test_enumerate_sublattices_rejects_nonpositive_index():
    with pytest.raises(ValueError, match="must be >= 1"):
        enumerate_sublattices(0)


def test_relative_sublattice_requires_containment():
    lattice = Sublattice(np.diag([2, 1, 1]))
    contained = Sublattice(np.diag([4, 1, 1]))
    not_contained = Sublattice(np.diag([1, 2, 1]))

    relative = lattice.relative_sublattice(contained)

    assert lattice.contains(contained)
    assert not lattice.contains(not_contained)
    assert relative == Sublattice(np.diag([2, 1, 1]))
    assert lattice.relative_sublattice(not_contained) is None


def test_relative_sublattice_preserves_noncommuting_operand_order():
    transformation = np.array([[1, 0, 0], [0, 2, 0], [0, 1, 2]])
    expected_relative = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]])
    lattice = Sublattice(transformation)
    contained = Sublattice(transformation @ expected_relative)

    relative = lattice.relative_sublattice(contained)

    assert relative is not None
    assert relative == Sublattice(expected_relative)
    np.testing.assert_array_equal(
        lattice.transformation @ relative.transformation,
        contained.transformation,
    )


def test_is_normal_respects_configured_tolerance():
    sublattice = Sublattice(np.eye(3, dtype=int))
    sublattice._inverse_transformation[0, 1] = 5e-4
    identity = np.eye(3, dtype=np.int64)

    assert not sublattice.is_normal(identity)
    assert sublattice.is_normal(identity, atol=1e-3)


def test_transform_operations_to_parent_preserves_noncommuting_operand_order():
    transformation = np.array([[1, 0, 0], [0, 2, 0], [0, 1, 2]])
    sublattice = Sublattice(transformation)
    rotations = np.array([[[-1, 0, 0], [0, -1, 0], [0, 1, 1]]])
    translations = np.array([[0.25, 0.5, 0.75]])

    transformed_rotations, transformed_translations = sublattice.transform_operations_to_parent(
        rotations, translations
    )

    np.testing.assert_array_equal(
        transformed_rotations,
        np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]),
    )
    np.testing.assert_allclose(transformed_translations, [[0.25, 1.0, 2.0]])
