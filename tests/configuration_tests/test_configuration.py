from __future__ import annotations

import numpy as np
import pytest
from loguru import logger
from moyopy import Cell, MoyoDataset, enumerate_translationengleiche_subgroups
from spinspg.spin import SpinOnlyGroupType

from spinforge.configuration import SSAGenerator
from spinforge.configuration._configuration import (
    _determine_real_order_parameter_directions,
    _varimax_rotation,
)
from spinforge.configuration._magnetic_multiplicity import MagneticMultiplicityClassifier
from spinforge.space_group import Sublattice
from spinforge.testing import (
    load_prim_CoTa3S6,
    load_prim_Mn3Sn,
    load_prim_MnTe,
    load_prim_pyrochlore,
    load_prim_Ta2FeO6,
)
from spinforge.utils.group_theory import SubgroupIndices


def test_magnetic_multiplicity_classifier_classifies_t_subgroups(
    pyrochlore_magnetic_site_indices: list[int],
):
    prim_cell = load_prim_pyrochlore()
    dataset = MoyoDataset(prim_cell)
    classifier = MagneticMultiplicityClassifier(
        prim_cell, dataset, pyrochlore_magnetic_site_indices
    )
    epsilon = dataset.symprec / abs(np.linalg.det(np.asarray(prim_cell.basis))) ** (1 / 3)
    classes = enumerate_translationengleiche_subgroups(
        classifier.prim_rotations.tolist(),
        classifier.prim_translations.tolist(),
        epsilon=epsilon,
    )
    flags: dict[SubgroupIndices, bool] = {}
    for item in classes:
        for conjugate in item.conjugates:
            subgroup = SubgroupIndices(conjugate.subgroup.operation_indices)
            flags[subgroup] = classifier.preserves_translationengleiche_multiplicity(subgroup)

    assert len(flags) == 98
    assert sum(flags.values()) == 34
    for item in classes:
        assert (
            len(
                {
                    flags[SubgroupIndices(conjugate.subgroup.operation_indices)]
                    for conjugate in item.conjugates
                }
            )
            == 1
        )


def test_conjugate_family_subgroups_retain_normalizers(
    pyrochlore_magnetic_site_indices: list[int],
):
    generator = SSAGenerator(
        load_prim_pyrochlore(),
        pyrochlore_magnetic_site_indices,
        multiplicity_preserving=False,
    )
    enumerator = generator._family_space_subgroup_enumerator
    representatives = enumerator.representative_subgroups
    conjugate_subgroups = [
        subgroup
        for subgroup in enumerator.translationengleiche_subgroups
        if subgroup.hermann_subgroup not in representatives
    ]

    assert conjugate_subgroups
    assert all(subgroup.hermann_normalizer for subgroup in conjugate_subgroups)

    subgroups_by_indices = {
        subgroup.hermann_subgroup: subgroup
        for subgroup in enumerator.translationengleiche_subgroups
    }
    for representative in representatives:
        subgroup = subgroups_by_indices[representative]
        assert {
            len(subgroups_by_indices[conjugate].hermann_normalizer)
            for conjugate in subgroup.hermann_conjugates
        } == {len(subgroup.hermann_normalizer)}


def test_spin_space_subgroup_enumerator_is_cached_per_family_subgroup():
    generator = SSAGenerator(
        Cell(np.eye(3).tolist(), [[0.0, 0.0, 0.0]], [1]),
        [0],
        multiplicity_preserving=False,
    )
    family_subgroup = generator._family_subgroups(
        k_index=1,
        up_to_parent_conjugacy=True,
        max_depth=0,
    )[0]

    repeated_family_subgroup = generator._family_subgroups(
        k_index=1,
        up_to_parent_conjugacy=True,
        max_depth=0,
    )[0]
    first = generator._spin_space_subgroup_enumerator(family_subgroup)
    second = generator._spin_space_subgroup_enumerator(repeated_family_subgroup)

    assert family_subgroup is repeated_family_subgroup
    assert first is second
    assert len(generator._spin_space_subgroup_enumerator_cache) == 1


def test_affine_multiplicity_classifier_accepts_and_rejects_concrete_subgroups(
    pyrochlore_magnetic_site_indices: list[int],
):
    prim_cell = load_prim_pyrochlore()
    dataset = MoyoDataset(prim_cell, rotate_basis=False)
    classifier = MagneticMultiplicityClassifier(
        prim_cell, dataset, pyrochlore_magnetic_site_indices
    )
    primitive_lattice = Sublattice(np.eye(3, dtype=np.int64))
    identity_index = next(
        index
        for index, (rotation, translation) in enumerate(
            zip(classifier.prim_rotations, classifier.prim_translations)
        )
        if np.array_equal(rotation, np.eye(3, dtype=np.int64))
        and np.allclose(translation, np.rint(translation))
    )

    assert classifier.preserves_magnetic_multiplicity(
        primitive_lattice,
        classifier.prim_rotations,
        classifier.prim_translations,
    )
    assert not classifier.preserves_magnetic_multiplicity(
        primitive_lattice,
        classifier.prim_rotations[[identity_index]],
        classifier.prim_translations[[identity_index]],
    )


@pytest.mark.parametrize(
    "spin_only_group_type,num_expects",
    [
        (SpinOnlyGroupType.COLLINEAR, 2),  # At least, FM and AFM
        (SpinOnlyGroupType.COPLANAR, 2),  # At least, chiral and inverse-chiral
    ],
)
def test_spin_configurations_Mn3Sn(
    Mn3Sn_ssa_generator: SSAGenerator,
    spin_only_group_type: SpinOnlyGroupType,
    num_expects: int,
):
    magnetic_structures = Mn3Sn_ssa_generator.enumerate(
        spin_only_group_type=spin_only_group_type, k_index=1
    )
    assert len(magnetic_structures) >= num_expects


def test_generate_high_symmetry_spin_structures_type3_collinear(
    Mn3Sn_ssa_generator: SSAGenerator,
):
    # type-3 collinear structure
    magnetic_structures = Mn3Sn_ssa_generator.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
        k_index=1,
    )
    assert len(magnetic_structures) > 0

    has_afm = False
    for sog, nssg, sas in magnetic_structures:
        ms = sas.generate()
        total_magmom = np.sum(ms.site_properties["magmom"], axis=0)
        if np.allclose(total_magmom, 0):
            has_afm = True
            break
    assert has_afm


class TestVarimaxRotation:
    def test_single_vector_unchanged(self):
        basis = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = _varimax_rotation(basis)
        np.testing.assert_allclose(result, basis)

    def test_orthonormality_preserved(self):
        # Create a 2D orthonormal basis rotated by 30 degrees from axes
        theta = np.pi / 6
        basis = np.array(
            [
                [np.cos(theta), np.sin(theta), 0.0, 0.0],
                [-np.sin(theta), np.cos(theta), 0.0, 0.0],
            ]
        )
        result = _varimax_rotation(basis)
        gram = result @ result.T
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-12)

    def test_sparsity_increases(self):
        # Create a 2D basis with overlapping nonzero entries
        theta = np.pi / 4
        basis = np.array(
            [
                [np.cos(theta), np.sin(theta), 0.0, 0.0],
                [-np.sin(theta), np.cos(theta), 0.0, 0.0],
            ]
        )
        result = _varimax_rotation(basis)
        # Varimax should rotate toward axis-aligned (sparser) solution
        # Count near-zero entries as a proxy for sparsity
        original_zeros = np.sum(np.abs(basis) < 1e-6)
        rotated_zeros = np.sum(np.abs(result) < 1e-6)
        assert rotated_zeros >= original_zeros

    def test_3d_orthonormality_preserved(self):
        rng = np.random.default_rng(42)
        q, _ = np.linalg.qr(rng.standard_normal((6, 3)).T)
        basis = q.T[:3]
        result = _varimax_rotation(basis)
        gram = result @ result.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)


class TestDetermineRealOrderParameterDirections:
    def test_degenerate_complex_pair_yields_orthonormal_real_basis(self):
        # Regression for the null-basis-vector bug. On a real Reynolds
        # operator with a degenerate eigenvalue-1 block, np.linalg.eig can
        # return a complex-conjugate eigenvector pair (eigenvalues 1 +- i*eps
        # within tolerance). The real/imag split then double-counts the
        # pair's 2D real span, and the varimax rotation turns the duplicate
        # rows into one sqrt(2)-norm vector plus one exactly-zero vector.
        # The tiny skew coupling below forces that eigendecomposition
        # deterministically.
        eps = 1e-9
        reynolds = np.zeros((4, 4))
        reynolds[0, 0] = reynolds[1, 1] = reynolds[2, 2] = 1.0
        reynolds[0, 1] = eps
        reynolds[1, 0] = -eps

        bases = _determine_real_order_parameter_directions(reynolds, atol=1e-5)

        stacked = np.array(bases)
        assert len(bases) == 3
        np.testing.assert_allclose(stacked @ stacked.T, np.eye(3), atol=1e-6)
        # Every returned vector lies in the eigenvalue-1 subspace.
        np.testing.assert_allclose(stacked @ reynolds.T, stacked, atol=1e-6)

    def test_real_degenerate_projector_unchanged(self):
        reynolds = np.diag([1.0, 1.0, 0.0, 0.0])
        bases = _determine_real_order_parameter_directions(reynolds, atol=1e-5)
        stacked = np.array(bases)
        assert len(bases) == 2
        np.testing.assert_allclose(stacked @ stacked.T, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(stacked[:, 2:], 0.0, atol=1e-10)

    def test_basis_independent_of_eigenvector_phases(self):
        # The extraction must not depend on the (arbitrary) phases or
        # orientations np.linalg.eig assigns inside a degenerate eigenspace:
        # conjugating the skewed projector by any orthogonal matrix changes
        # those choices but not the correct answer (roborev #713). The
        # count, orthonormality, and invariance must hold for every frame.
        eps = 1e-9
        reynolds = np.zeros((5, 5))
        reynolds[0, 0] = reynolds[1, 1] = reynolds[2, 2] = 1.0
        reynolds[0, 1] = eps
        reynolds[1, 0] = -eps
        rng = np.random.default_rng(7)
        for _ in range(5):
            rotation = np.linalg.qr(rng.standard_normal((5, 5)))[0]
            conjugated = rotation @ reynolds @ rotation.T
            bases = _determine_real_order_parameter_directions(conjugated, atol=1e-5)
            stacked = np.array(bases)
            assert len(bases) == 3
            np.testing.assert_allclose(stacked @ stacked.T, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(stacked @ conjugated.T, stacked, atol=1e-6)


class TestGroupedMagneticMomentsBasis:
    def test_groups_cover_all_bases(
        self,
        Mn3Sn_ssa_generator: SSAGenerator,
    ):
        """All basis indices appear exactly once across groups."""
        results = Mn3Sn_ssa_generator.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COPLANAR, k_index=1
        )
        assert len(results) > 0
        for _sog, _nssg, sas in results:
            grouped = sas.grouped_magnetic_moments_basis
            all_indices = sorted(idx for indices in grouped.values() for idx in indices)
            assert all_indices == list(range(sas.dim))

    def test_keys_are_subsets_of_magnetic_sites(
        self,
        Mn3Sn_ssa_generator: SSAGenerator,
    ):
        """Nonzero-site keys must be subsets of the supercell magnetic sites."""
        results = Mn3Sn_ssa_generator.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COPLANAR, k_index=1
        )
        assert len(results) > 0
        for _sog, _nssg, sas in results:
            grouped = sas.grouped_magnetic_moments_basis
            # Each basis vector should only be nonzero on magnetic sites
            magnetic_sites = frozenset(
                int(i)
                for i in range(sas.supercell.num_supercell_sites)
                if np.linalg.norm(sas.magnetic_moments_basis[0][i]) > 1e-8
                or any(np.linalg.norm(b[i]) > 1e-8 for b in sas.magnetic_moments_basis)
            )
            for sites in grouped:
                assert sites.issubset(magnetic_sites)

    def test_dim1_has_single_group(
        self,
        Mn3Sn_ssa_generator: SSAGenerator,
    ):
        """A dim=1 SSA structure should have exactly one group."""
        results = Mn3Sn_ssa_generator.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COPLANAR, k_index=1
        )
        for _sog, _nssg, sas in results:
            if sas.dim == 1:
                grouped = sas.grouped_magnetic_moments_basis
                assert len(grouped) == 1
                assert list(grouped.values()) == [[0]]
                break
        else:
            pytest.skip("No dim=1 structure found for this material/group type")


class TestMaxDepth:
    def test_max_depth_none_includes_all(
        self,
        Mn3Sn_ssa_generator: SSAGenerator,
    ):
        """max_depth=None returns all Hermann depths."""
        results_all = Mn3Sn_ssa_generator.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
            k_index=1,
            max_depth=None,
        )
        results_depth1 = Mn3Sn_ssa_generator.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
            k_index=1,
            max_depth=1,
        )
        assert len(results_all) >= len(results_depth1)

    def test_max_depth_monotonic(
        self,
        Mn3Sn_ssa_generator: SSAGenerator,
    ):
        """Increasing max_depth yields monotonically more results."""
        counts = []
        for depth in [0, 1, 2, 3]:
            results = Mn3Sn_ssa_generator.enumerate(
                spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
                k_index=1,
                max_depth=depth,
            )
            counts.append(len(results))
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1]

    def test_max_depth_negative_raises(
        self,
        Mn3Sn_ssa_generator: SSAGenerator,
    ):
        """max_depth < 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_depth must be >= 0"):
            Mn3Sn_ssa_generator.enumerate(
                spin_only_group_type=SpinOnlyGroupType.COLLINEAR, k_index=1, max_depth=-1
            )


class TestPropagationVectorFrame:
    """Regression tests for k-vector frame handling (issue #75).

    A k-vector is interpreted in the setting of the cell passed to
    ``with_propagation_vectors`` and transformed into moyopy's primitive
    standardized cell internally. The same physical k expressed in two
    different cell settings must give identical enumerations.
    """

    # Rows give the new basis vectors in terms of the old ones (a' = c,
    # b' = a + b, c' = -a): a hexagonal axis permutation of the kind that
    # bit CsCrF4 and Ho2BaNiO5.
    PERMUTATION = np.array([[0, 0, 1], [1, 1, 0], [-1, 0, 0]])

    @staticmethod
    def _repermuted_cell(cell: Cell, permutation: np.ndarray) -> Cell:
        """Re-express ``cell`` in the basis ``permutation @ basis``."""
        basis_old = np.array(cell.basis)
        basis_new = permutation @ basis_old
        positions_new = (np.array(cell.positions) @ basis_old) @ np.linalg.inv(basis_new)
        return Cell(basis_new.tolist(), (positions_new % 1.0).tolist(), cell.numbers)

    def test_same_physical_k_in_two_settings(
        self,
        MnTe_magnetic_site_indices: list[int],
    ):
        """The same physical k in a permuted setting gives the same result."""
        prim_cell = load_prim_MnTe()
        k = np.array([0.0, 0.0, 0.5])

        sag_input = SSAGenerator.with_propagation_vectors(
            prim_cell, [k], MnTe_magnetic_site_indices
        )
        results_input = sag_input.enumerate(SpinOnlyGroupType.COLLINEAR)
        assert len(results_input) > 0

        permuted_cell = self._repermuted_cell(prim_cell, self.PERMUTATION)
        # For an integer basis change (rows = new basis in old), a fractional
        # reciprocal vector transforms by the same matrix.
        k_permuted = self.PERMUTATION @ k
        sag_permuted = SSAGenerator.with_propagation_vectors(
            permuted_cell, [k_permuted], MnTe_magnetic_site_indices
        )
        results_permuted = sag_permuted.enumerate(SpinOnlyGroupType.COLLINEAR)

        assert len(results_permuted) == len(results_input)
        assert sag_input.propagation_vectors is not None
        assert sag_permuted.propagation_vectors is not None
        np.testing.assert_allclose(
            sag_permuted.propagation_vectors[0], sag_input.propagation_vectors[0], atol=1e-8
        )

    def test_prim_std_frame_skips_transform(
        self,
        MnTe_magnetic_site_indices: list[int],
    ):
        """k_frame="prim_std" takes the vectors as already standardized."""
        prim_cell = load_prim_MnTe()
        k = np.array([0.0, 0.0, 0.5])

        sag_input = SSAGenerator.with_propagation_vectors(
            prim_cell, [k], MnTe_magnetic_site_indices
        )
        assert sag_input.propagation_vectors is not None
        k_std = sag_input.propagation_vectors[0]

        permuted_cell = self._repermuted_cell(prim_cell, self.PERMUTATION)
        sag_std = SSAGenerator.with_propagation_vectors(
            permuted_cell, [k_std], MnTe_magnetic_site_indices, k_frame="prim_std"
        )
        assert sag_std.propagation_vectors is not None
        np.testing.assert_allclose(sag_std.propagation_vectors[0], k_std, atol=1e-8)
        results_std = sag_std.enumerate(SpinOnlyGroupType.COLLINEAR)
        results_input = sag_input.enumerate(SpinOnlyGroupType.COLLINEAR)
        assert len(results_std) == len(results_input)

    def test_invalid_k_frame_raises(
        self,
        MnTe_magnetic_site_indices: list[int],
    ):
        prim_cell = load_prim_MnTe()
        with pytest.raises(ValueError, match="k_frame"):
            SSAGenerator.with_propagation_vectors(
                prim_cell,
                [np.array([0.0, 0.0, 0.5])],
                MnTe_magnetic_site_indices,
                k_frame="bogus",  # ty: ignore[invalid-argument-type]
            )

    def test_propagation_vectors_none_without_k(
        self,
        Mn3Sn_ssa_generator: SSAGenerator,
    ):
        assert Mn3Sn_ssa_generator.propagation_vectors is None

    def test_zero_candidates_warns_frame_mismatch(
        self,
        Mn3Sn_magnetic_site_indices: list[int],
    ):
        """Zero candidates for a supplied k emits a loud hint naming both the
        spin-type incompatibility and frame-mismatch causes."""
        sag = SSAGenerator.with_propagation_vectors(
            load_prim_Mn3Sn(),
            [np.array([0.5, 0.0, 0.0])],
            Mn3Sn_magnetic_site_indices,
        )
        messages: list[str] = []
        sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            results = sag.enumerate(SpinOnlyGroupType.COLLINEAR)
        finally:
            logger.remove(sink_id)
        assert len(results) == 0
        assert any("frame mismatch" in message.lower() for message in messages)
        assert any("spin-type incompatibility" in message.lower() for message in messages)


class TestDefaultEnumerationRegression:
    """Pin the Hermann depth-one default for the three paper materials.

    The default now includes descendants of each depth-one Hermann subgroup,
    so these totals intentionally exceed the parent-only counts shown in the
    paper examples.
    """

    def test_collinear_MnTe(self, MnTe_magnetic_site_indices: list[int]):
        scg = SSAGenerator(
            prim_cell=load_prim_MnTe(),
            magnetic_site_indices=MnTe_magnetic_site_indices,
        )

        results = scg.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
            k_index=1,
        )
        assert len(results) == 16

    def test_coplanar_Mn3Sn(self, Mn3Sn_ssa_generator: SSAGenerator):
        results = Mn3Sn_ssa_generator.enumerate(
            spin_only_group_type=SpinOnlyGroupType.COPLANAR,
            k_index=1,
        )
        assert len(results) == 34

    @pytest.mark.slow
    def test_noncoplanar_CoTa3S6(self, CoTa3S6_magnetic_site_indices: list[int]):
        scg = SSAGenerator(
            prim_cell=load_prim_CoTa3S6(),
            magnetic_site_indices=CoTa3S6_magnetic_site_indices,
        )

        results = scg.enumerate(
            spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR,
            k_index=4,
        )
        assert len(results) == 212


def test_oriented_ssg_enumerator_coplanar_Ta2FeO6(
    Ta2FeO6_magnetic_site_indices: list[int],
):
    prim_cell = load_prim_Ta2FeO6()
    magnetic_site_indices = Ta2FeO6_magnetic_site_indices

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COPLANAR,
        k_index=4,
    )
    assert len(all_magnetic_structures) > 0
