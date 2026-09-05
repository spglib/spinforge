"""Unit tests for the space group normalizer module."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from spgrep.rep.group import get_inverse_index
from spgrep.symmetry.group import get_cayley_table
from spgrep.utils import ndarray2d_to_integer_tuple

from spinforge.space_group import NormalSpaceSubgroup, Sublattice
from spinforge.space_group._normalizer import (
    _are_conjugated_nss_equal,
    _conjugate_nss,
    classify_normal_space_subgroup_conjugacy_classes,
    space_group_normalizer,
)


def _build_octahedral_rotations() -> np.ndarray:
    """Build the 24 proper rotations of the octahedral group O (432)."""
    rots = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for signs in product([1, -1], repeat=3):
            mat = np.zeros((3, 3), dtype=int)
            for i in range(3):
                mat[i, perm[i]] = signs[i]
            if np.linalg.det(mat) == 1:
                rots.append(mat)
    return np.array(rots)


class TestNormalizerP222SubgroupsUnderCubic:
    """The three order-2 subgroups <2x>, <2y>, <2z> of the 222 point group
    should be conjugate under the cubic supergroup O (432, order 24)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cubic_rots = _build_octahedral_rotations()
        assert len(self.cubic_rots) == 24
        self.table = get_cayley_table(self.cubic_rots)

        idx = {}
        for i, r in enumerate(self.cubic_rots):
            idx[ndarray2d_to_integer_tuple(r)] = i

        E = np.eye(3, dtype=int)
        C2x = np.diag([1, -1, -1])
        C2y = np.diag([-1, 1, -1])
        C2z = np.diag([-1, -1, 1])

        self.sg_2x = frozenset([idx[ndarray2d_to_integer_tuple(r)] for r in [E, C2x]])
        self.sg_2y = frozenset([idx[ndarray2d_to_integer_tuple(r)] for r in [E, C2y]])
        self.sg_2z = frozenset([idx[ndarray2d_to_integer_tuple(r)] for r in [E, C2z]])

    def test_three_subgroups_are_conjugate(self):
        """Some element of O maps <2x> to <2y> and another maps <2x> to <2z>."""
        found_y = False
        found_z = False
        for h in range(len(self.table)):
            h_inv = get_inverse_index(self.table, h)
            conj = frozenset(int(self.table[self.table[h, g], h_inv]) for g in self.sg_2x)
            if conj == self.sg_2y:
                found_y = True
            if conj == self.sg_2z:
                found_z = True
        assert found_y, "<2x> and <2y> should be conjugate under O"
        assert found_z, "<2x> and <2z> should be conjugate under O"

    def test_normalizer_coset_reps(self):
        """N_O(<2x>)/<2x> should have more than 1 coset representative since
        <2x> is not normal in O."""
        coset_reps = space_group_normalizer(self.table, self.sg_2x)
        assert len(coset_reps) == 4

    def test_normalizers_have_same_size(self):
        """All three conjugate subgroups have isomorphic normalizers."""
        norm_x = space_group_normalizer(self.table, self.sg_2x)
        norm_y = space_group_normalizer(self.table, self.sg_2y)
        norm_z = space_group_normalizer(self.table, self.sg_2z)
        assert len(norm_x) == len(norm_y) == len(norm_z)


def test_affine_translation_lift_classifies_normal_space_subgroup_equivalence_class():
    identity = np.eye(3, dtype=np.int64)
    inversion = -identity
    family_rotations = np.asarray([identity, inversion])
    sublattice = Sublattice(np.diag([2, 1, 1]))
    quotient_table = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    reference = NormalSpaceSubgroup(
        point_subgroup=[0, 1],
        translations=np.zeros((2, 3)),
        sublattice=sublattice,
        coset_representatives=[0],
        quotient_table=quotient_table,
    )
    translated = NormalSpaceSubgroup(
        point_subgroup=[0, 1],
        translations=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        sublattice=sublattice,
        coset_representatives=[0],
        quotient_table=quotient_table,
    )

    normalizer_rotations = np.asarray([identity, identity])
    normalizer_translations = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    normalizer_permutations = ((0, 1), (0, 1))
    candidates = [reference, translated]

    equivalence_classes = classify_normal_space_subgroup_conjugacy_classes(
        candidates,
        family_rotations=family_rotations,
        normalizer_rotations=normalizer_rotations,
        normalizer_translations=normalizer_translations,
        normalizer_permutations=normalizer_permutations,
    )
    assert len(equivalence_classes) == 1
    equivalence_class = equivalence_classes[0]
    assert equivalence_class.representative is reference
    assert equivalence_class.equivalent_objects == (reference, translated)
    assert equivalence_class.transformations_to_representative == (0, 1)
    assert equivalence_class.stabilizer == (0,)
    assert sum(
        len(candidate_class.equivalent_objects) for candidate_class in equivalence_classes
    ) == len(candidates)
    assert {id(obj) for obj in equivalence_class.equivalent_objects} == {
        id(candidate) for candidate in candidates
    }
    assert all(
        sum(
            obj is candidate
            for candidate_class in equivalence_classes
            for obj in candidate_class.equivalent_objects
        )
        == 1
        for candidate in candidates
    )
    for equivalent_object, normalizer_index in zip(
        equivalence_class.equivalent_objects,
        equivalence_class.transformations_to_representative,
    ):
        conjugated = _conjugate_nss(
            equivalent_object,
            family_rotations,
            normalizer_rotations[normalizer_index],
            normalizer_translations[normalizer_index],
            normalizer_permutations[normalizer_index],
        )
        assert _are_conjugated_nss_equal(*conjugated, reference, atol=1e-5)
    for normalizer_index in equivalence_class.stabilizer:
        conjugated = _conjugate_nss(
            reference,
            family_rotations,
            normalizer_rotations[normalizer_index],
            normalizer_translations[normalizer_index],
            normalizer_permutations[normalizer_index],
        )
        assert _are_conjugated_nss_equal(*conjugated, reference, atol=1e-5)
