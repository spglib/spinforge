from __future__ import annotations

import numpy as np
import pytest
from moyopy import Cell, operations_from_number

from spinforge.configuration import Supercell
from spinforge.space_group import Sublattice


def test_supercell_fcc():
    prim_cell = Cell(
        basis=[
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        positions=[
            [0.0, 0.0, 0.0],
        ],
        numbers=[0],
    )
    sublattice = Sublattice(
        transformation=np.array(
            [
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
    )
    supercell = Supercell(prim_cell, sublattice)

    assert supercell.num_supercell_sites == 4
    assert np.allclose(
        supercell.basis,
        2.0 * np.eye(3),
    )

    assert supercell.map_sites([0]) == [0, 1, 2, 3]

    operations = operations_from_number(225, primitive=True)  # Fm-3m
    automorphism = set()
    for rotation, translation in zip(operations.rotations, operations.translations):
        permutation = supercell.act_operation(
            sub_sites=[0, 1, 2, 3],
            prim_rotation=np.array(rotation),
            prim_translation=np.array(translation),
        )
        assert permutation is not None
        automorphism.add(tuple(sorted(list(permutation.values()))))
    assert len(automorphism) > 0
    assert len(operations) % len(automorphism) == 0

    permutations = supercell.site_permutations(
        [np.array(rotation) for rotation in operations.rotations],
        [np.array(translation) for translation in operations.translations],
    )
    for (rotation, translation), permutation in zip(
        zip(operations.rotations, operations.translations), permutations
    ):
        expected = supercell.act_operation(
            sub_sites=[0, 1, 2, 3],
            prim_rotation=np.array(rotation),
            prim_translation=np.array(translation),
        )
        assert expected is not None
        assert [expected[i] for i in range(4)] == list(permutation)


def test_site_permutations_prim_frame_tolerance():
    # In a diag(1, 1, 5) supercell, a primitive-frame z error of 3 * symprec
    # shrinks to 0.6 * symprec in supercell coordinates; validating there
    # would wrongly accept it. The acceptance must match act_operation's
    # primitive-frame per-component tolerance.
    symprec = 0.01
    prim_cell = Cell(basis=np.eye(3).tolist(), positions=[[0.0, 0.0, 0.0]], numbers=[0])
    supercell = Supercell(
        prim_cell, Sublattice(transformation=np.diag([1, 1, 5])), symprec=symprec
    )

    good = np.array([0.0, 0.0, 0.3 * symprec])
    (permutation,) = supercell.site_permutations([np.eye(3, dtype=int)], [good])
    assert list(permutation) == list(range(5))

    bad = np.array([0.0, 0.0, 3 * symprec])
    assert supercell.act_operation(list(range(5)), np.eye(3, dtype=int), bad) is None
    with pytest.raises(ValueError, match="does not permute"):
        supercell.site_permutations([np.eye(3, dtype=int)], [bad])


def test_site_permutations_nearest_neighbor_decoy():
    # The Euclidean-nearest site (D, at 0.011) fails the per-component box
    # test while a farther site (A', at 0.0124) passes it; matching must
    # use the box test, not Euclidean proximity.
    symprec = 0.01
    a = [0.35, 0.1, 0.0]
    a_image = [0.642, 0.892, 0.995]  # C2z image of A, off by (0.008, 0.008, 0.005)
    d = [0.661, 0.9, 0.0]  # decoy: 0.011 from the exact C2z image of A
    d_image = [0.339, 0.1, 0.0]  # exact C2z image of D
    prim_cell = Cell(
        basis=np.eye(3).tolist(), positions=[a, a_image, d, d_image], numbers=[0, 0, 0, 0]
    )
    supercell = Supercell(prim_cell, Sublattice(transformation=np.eye(3, dtype=int)), symprec)

    c2z = np.diag([-1, -1, 1])
    (permutation,) = supercell.site_permutations([c2z], [np.zeros(3)])
    assert list(permutation) == [1, 0, 3, 2]

    expected = supercell.act_operation([0, 1, 2, 3], c2z, np.zeros(3))
    assert expected is not None
    assert [expected[i] for i in range(4)] == list(permutation)


def test_site_permutations_ambiguous_match_rejected():
    # X's image lies within tolerance of both X and Y (the sites are only
    # 1.5 * symprec apart): symprec is too coarse for this structure, so the
    # ambiguity is treated as failure rather than resolved to a bijection.
    symprec = 0.01
    prim_cell = Cell(
        basis=np.eye(3).tolist(),
        positions=[[0.0, 0.0, 0.0], [0.015, 0.0, 0.0]],
        numbers=[0, 0],
    )
    supercell = Supercell(prim_cell, Sublattice(transformation=np.eye(3, dtype=int)), symprec)

    translation = np.array([0.008, 0.0, 0.0])
    assert supercell.act_operation([0, 1], np.eye(3, dtype=int), translation) is None
    with pytest.raises(ValueError, match="does not permute"):
        supercell.site_permutations([np.eye(3, dtype=int)], [translation])


def test_image_indices_edge_cases():
    prim_cell = Cell(basis=np.eye(3).tolist(), positions=[[0.0, 0.0, 0.0]], numbers=[0])
    supercell = Supercell(prim_cell, Sublattice(transformation=np.eye(3, dtype=int)))

    assert supercell.act_operation([], np.eye(3, dtype=int), np.zeros(3)) == {}

    singular = np.zeros((3, 3), dtype=int)
    with pytest.raises(ValueError, match="det"):
        supercell.site_permutations([singular], [np.zeros(3)])


def test_site_permutations_skew_transformation():
    # For a strongly non-reduced transformation the nearest supercell-frame
    # lattice image is not the one minimizing the primitive-frame residual:
    # here the residual (0, 0.008, 0) wraps to supercell delta (0.2, 0.008, 0)
    # whose naive primitive lift is (1.0, 0.008, 0). The matcher must probe
    # neighboring lattice offsets instead of rejecting the valid match.
    symprec = 0.01
    shear = np.array([[1, 100, 0], [0, 1, 0], [0, 0, 1]])
    prim_cell = Cell(basis=np.eye(3).tolist(), positions=[[0.0, 0.0, 0.0]], numbers=[0])
    supercell = Supercell(prim_cell, Sublattice(transformation=shear), symprec=symprec)

    translation = np.array([0.0, 0.008, 0.0])
    (permutation,) = supercell.site_permutations([np.eye(3, dtype=int)], [translation])
    assert list(permutation) == [0]
