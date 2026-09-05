from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Final, Literal, Self

import numpy as np
from loguru import logger
from moyopy import Cell, MoyoDataset, SpaceGroup, SpaceGroupType
from pymatgen.core import Element, Structure
from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType, get_spin_only_group

from spinforge.msg import MagneticSpaceSubgroup
from spinforge.msg._enumerator import _enumerate_oriented_spin_space_groups
from spinforge.space_group import (
    FamilySpaceSubgroup,
    FamilySpaceSubgroupEnumerator,
    NormalSpaceSubgroup,
    Sublattice,
)
from spinforge.ssg import (
    NontrivialSpinSpaceGroup,
    SpinSpaceSubgroupEnumerator,
    SpinSymmetryOperations,
)
from spinforge.utils.rotation_utils import to_cartesian_rotations

from ._magnetic_multiplicity import MagneticMultiplicityClassifier
from ._propagation import get_commensurate_sublattice
from ._representation import get_site_representation, get_spin_only_group_reynolds_operator
from ._supercell import Supercell, SupercellSiteIndex


@dataclass
class SpinSymmetryAdaptedStructure:
    supercell: Supercell
    magnetic_moments_basis: list[NDArrayFloat]  # (dim, num_supercell_sites, 3)
    atol: float = field(default=1e-8, repr=False)

    @property
    def dim(self) -> int:
        return len(self.magnetic_moments_basis)

    @cached_property
    def grouped_magnetic_moments_basis(
        self,
    ) -> dict[frozenset[SupercellSiteIndex], list[int]]:
        """Group basis vectors by the supercell sites they act on.

        Returns a dict mapping frozensets of supercell site indices (where the
        basis vector has nonzero rows) to lists of indices into
        ``magnetic_moments_basis`` sharing that site pattern.
        """
        groups: dict[frozenset[SupercellSiteIndex], list[int]] = defaultdict(list)
        for idx, basis in enumerate(self.magnetic_moments_basis):
            nonzero_sites: frozenset[SupercellSiteIndex] = frozenset(
                int(i) for i in range(basis.shape[0]) if np.linalg.norm(basis[i]) > self.atol
            )
            groups[nonzero_sites].append(idx)
        return dict(groups)

    def generate(self, rng: np.random.Generator | None = None) -> Structure:
        magnetic_moments = self._sample_magnetic_moments(rng=rng)
        return self.generate_with_magnetic_moments(magnetic_moments)

    def generate_oriented(
        self,
        spin_only_group: SpinOnlyGroup,
        nontrivial_spin_space_group: NontrivialSpinSpaceGroup,
        *,
        preserve_spin_planochirality: bool = True,
        parent_prim_rotations: NDArrayInt | None = None,
        rng: np.random.Generator | None = None,
        atol: float = 1e-5,
    ) -> list[tuple[Structure, MagneticSpaceSubgroup]]:
        magnetic_moments = self._sample_magnetic_moments(rng=rng)

        lattice = np.array(self.supercell.prim_cell.basis)
        extra_cart_rotations: NDArrayFloat | None = None
        if parent_prim_rotations is not None:
            extra_cart_rotations = to_cartesian_rotations(lattice, parent_prim_rotations)
        list_maximal_msg = _enumerate_oriented_spin_space_groups(
            lattice=lattice,
            spin_only_group=spin_only_group,
            spin_space_group=nontrivial_spin_space_group,
            preserve_spin_planochirality=preserve_spin_planochirality,
            atol=atol,
            extra_cart_rotations=extra_cart_rotations,
        )

        oriented_structures = []
        for msg in list_maximal_msg:
            new_magnetic_moments = magnetic_moments @ msg.Q.T
            new_ms = self.generate_with_magnetic_moments(new_magnetic_moments)
            oriented_structures.append((new_ms, msg))

        return oriented_structures

    def generate_with_magnetic_moments(self, magnetic_moments: NDArrayFloat) -> Structure:
        ms = Structure(
            lattice=self.supercell.basis,
            coords=self.supercell.positions,
            species=[Element.from_Z(number) for number in self.supercell.numbers],
            site_properties={"magmom": magnetic_moments},
        )
        return ms

    def _sample_magnetic_moments(self, rng: np.random.Generator | None = None) -> NDArrayFloat:
        if self.dim == 1:
            return self.magnetic_moments_basis[0]
        else:
            if rng is None:
                rng = np.random.default_rng()
            coeffs = _sample_on_unit_sphere(rng, n=self.dim, size=1)[0]  # (dim, )
            magnetic_moments = np.sum(
                coeffs[:, None, None] * np.array(self.magnetic_moments_basis), axis=0
            )
            return magnetic_moments


DEFAULT_SYMPREC: Final[float] = 1e-3


class SSAGenerator:
    def __init__(
        self,
        prim_cell: Cell,
        magnetic_site_indices: Sequence[int],
        *,
        sublattice: Sublattice | None = None,
        multiplicity_preserving: bool = True,
        symprec: float = DEFAULT_SYMPREC,
        atol: float = 1e-5,
    ):
        self._prim_cell = prim_cell
        self._sublattice = sublattice
        self._atol = atol
        # Set by with_propagation_vectors; used to hint at a possible k-vector
        # frame mismatch when enumeration finds zero candidates (issue #75).
        self._propagation_vectors: list[NDArrayFloat] | None = None

        # Check primitive cell
        self._prim_dataset = MoyoDataset(prim_cell, symprec=symprec, rotate_basis=False)
        if self._prim_dataset.prim_std_cell.num_atoms != prim_cell.num_atoms:
            raise ValueError(f"Given prim_cell is not a primitive cell with {symprec=}.")

        self._magnetic_site_indices = magnetic_site_indices
        self._multiplicity_classifier = MagneticMultiplicityClassifier(
            prim_cell=self._prim_cell,
            prim_dataset=self._prim_dataset,
            magnetic_site_indices=self._magnetic_site_indices,
        )
        parent_rotations = np.asarray(self._prim_dataset.operations.rotations, dtype=np.int64)
        parent_translations = np.asarray(self._prim_dataset.operations.translations, dtype=float)
        epsilon = self._prim_dataset.symprec / np.abs(
            np.linalg.det(np.asarray(self._prim_cell.basis))
        ) ** (1 / 3)
        self._family_space_subgroup_enumerator = FamilySpaceSubgroupEnumerator(
            parent_rotations,
            parent_translations,
            epsilon=epsilon,
            target_sublattice=self._sublattice,
            atol=atol,
        )
        self._spin_space_subgroup_enumerator_cache: dict[
            FamilySpaceSubgroup, SpinSpaceSubgroupEnumerator
        ] = {}
        self._multiplicity_preserving = multiplicity_preserving
        self._prim_rotations = self._family_space_subgroup_enumerator.parent_rotations
        self._prim_translations = self._family_space_subgroup_enumerator.parent_translations

    @classmethod
    def with_propagation_vectors(
        cls: type[Self],
        cell: Cell,
        propagation_vectors: Sequence[NDArrayFloat],
        magnetic_site_indices: Sequence[int],
        *,
        k_frame: Literal["input", "prim_std"] = "input",
        multiplicity_preserving: bool = True,
        symprec: float = DEFAULT_SYMPREC,
        atol: float = 1e-5,
    ) -> Self:
        """Create a generator from propagation vectors given in a named frame.

        The internal enumeration works in moyopy's primitive standardized cell,
        whose setting may differ from that of ``cell`` (e.g. hexagonal axis
        permutations). The propagation vectors are transformed into that frame
        explicitly, and the accepted vectors are echoed back in both settings
        at INFO level so a frame mismatch can be caught before enumeration. The
        resulting commensurate translation lattice is also used as the single
        bound for t-, k-, and general-family subgroup enumeration.

        Parameters
        ----------
        cell:
            Input cell, e.g. in the CIF / experimental setting. Do not
            pre-standardize it unless the propagation vectors are given in the
            same standardized setting (then use ``k_frame="prim_std"``).
        propagation_vectors:
            Propagation vectors in fractional reciprocal coordinates of the
            frame named by ``k_frame``.
        magnetic_site_indices:
            Indices of magnetic sites in ``cell``.
        k_frame:
            Frame in which ``propagation_vectors`` are given. ``"input"``
            (default) is the setting of ``cell``. ``"prim_std"`` is moyopy's
            primitive standardized cell; the vectors are used as-is.
        """
        dataset = MoyoDataset(cell, symprec=symprec, rotate_basis=False)
        prim_cell = dataset.prim_std_cell
        prim_magnetic_site_indices = list(
            frozenset([dataset.mapping_std_prim[i] for i in magnetic_site_indices])
        )
        if k_frame == "input":
            # A fractional reciprocal vector transforms with the transpose of the
            # (input -> primitive standardized) linear part, which keeps the
            # physical reciprocal vector unchanged.
            prim_propagation_vectors = [
                np.transpose(dataset.prim_std_linear) @ np.asarray(pv, dtype=float)
                for pv in propagation_vectors
            ]
        elif k_frame == "prim_std":
            prim_propagation_vectors = [np.asarray(pv, dtype=float) for pv in propagation_vectors]
        else:
            raise ValueError(f'k_frame must be "input" or "prim_std", got {k_frame!r}')

        for pv, prim_pv in zip(propagation_vectors, prim_propagation_vectors):
            logger.info(
                f"Propagation vector k={np.asarray(pv, dtype=float)} ({k_frame} setting) "
                f"is used as k={prim_pv} in the primitive standardized setting."
            )

        transformation = get_commensurate_sublattice(prim_propagation_vectors)
        sublattice = Sublattice(np.transpose(transformation))
        logger.info(f"Setting commensurate sublattice: {sublattice}")
        obj = cls(
            prim_cell=prim_cell,
            magnetic_site_indices=prim_magnetic_site_indices,
            sublattice=sublattice,
            multiplicity_preserving=multiplicity_preserving,
            symprec=symprec,
            atol=atol,
        )
        obj._propagation_vectors = prim_propagation_vectors
        return obj

    @property
    def prim_dataset(self) -> MoyoDataset:
        return self._prim_dataset

    @property
    def propagation_vectors(self) -> list[NDArrayFloat] | None:
        """Propagation vectors in the primitive standardized setting, or
        ``None`` if the generator was not constructed with
        :meth:`with_propagation_vectors`."""
        return self._propagation_vectors

    @property
    def prim_rotations(self) -> NDArrayInt:
        return self._prim_rotations

    @property
    def prim_translations(self) -> NDArrayFloat:
        return self._prim_translations

    def generate_oriented(
        self,
        sas: SpinSymmetryAdaptedStructure,
        spin_only_group: SpinOnlyGroup,
        nontrivial_spin_space_group: NontrivialSpinSpaceGroup,
        *,
        preserve_spin_planochirality: bool = True,
        rng: np.random.Generator | None = None,
        atol: float = 1e-5,
    ) -> list[tuple[Structure, MagneticSpaceSubgroup]]:
        """Generate oriented structures using only the SSG family group's axes."""
        return sas.generate_oriented(
            spin_only_group=spin_only_group,
            nontrivial_spin_space_group=nontrivial_spin_space_group,
            preserve_spin_planochirality=preserve_spin_planochirality,
            rng=rng,
            atol=atol,
        )

    def _family_subgroups(
        self,
        *,
        k_index: int,
        up_to_parent_conjugacy: bool,
        max_depth: int | None,
    ) -> list[FamilySpaceSubgroup]:
        return [
            subgroup
            for subgroup in self._family_space_subgroup_enumerator.enumerate(
                k_index=k_index,
                up_to_parent_conjugacy=up_to_parent_conjugacy,
                max_depth=max_depth,
            )
            if self._preserves_magnetic_multiplicity(subgroup)
        ]

    def _spin_space_subgroup_enumerator(
        self,
        subgroup: FamilySpaceSubgroup,
    ) -> SpinSpaceSubgroupEnumerator:
        enumerator = self._spin_space_subgroup_enumerator_cache.get(subgroup)
        if enumerator is None:
            enumerator = SpinSpaceSubgroupEnumerator(
                subgroup,
                atol=self._atol,
            )
            self._spin_space_subgroup_enumerator_cache[subgroup] = enumerator
        return enumerator

    def _preserves_magnetic_multiplicity(
        self,
        subgroup: FamilySpaceSubgroup,
    ) -> bool:
        if not self._multiplicity_preserving:
            return True
        if subgroup.is_translationengleiche:
            return self._multiplicity_classifier.preserves_translationengleiche_multiplicity(
                subgroup.hermann_subgroup
            )
        return self._multiplicity_classifier.preserves_magnetic_multiplicity(
            subgroup.translation_sublattice,
            subgroup.parent_rotations,
            subgroup.parent_translations,
        )

    def enumerate(
        self,
        spin_only_group_type: SpinOnlyGroupType,
        *,
        k_index: int | None = None,
        max_depth: int | None = 1,
        up_to_parent_conjugacy: bool = True,
    ) -> list[tuple[SpinOnlyGroup, NontrivialSpinSpaceGroup, SpinSymmetryAdaptedStructure]]:
        """Enumerate spin symmetry adapted (SSA) structures.

        For every retained family space subgroup ``G' <= G``, spin space groups
        are classified under ``N_G(G') x O(3)``. ``up_to_parent_conjugacy``
        independently controls whether ``G``-conjugate family subgroups are
        retained separately.

        Parameters
        ----------
        spin_only_group_type:
            Type of the spin-only group to enumerate.
        k_index:
            Global index of the invariant translation sublattice used to
            bound family-subgroup enumeration. Without propagation vectors,
            every index-``k_index`` sublattice is considered. With propagation
            vectors, their commensurate sublattice fixes the bound and its
            index is used automatically.
        max_depth:
            Maximum Hermann translationengleiche subgroup depth from the
            parent group. ``0`` retains the parent as the only Hermann group
            while still including its bounded klassengleiche descendants.
            ``1`` retains the parent and maximal proper Hermann subgroups
            (default); ``None`` retains all Hermann subgroups. Bounded
            klassengleiche descendants inherit their Hermann group depth.
        up_to_parent_conjugacy:
            Classify family space subgroups up to conjugation by the
            crystallographic parent. When ``False``, retain every enumerated
            t-group, family-lattice, and finite-quotient complement conjugate.
            This does not change the per-family ``N_G(G') x O(3)`` relation.
        """
        if k_index is None:
            if self._sublattice is None:
                raise ValueError("k_index must be specified when sublattice is None.")
            k_index = self._sublattice.order
        elif k_index < 1:
            raise ValueError(f"k_index must be >= 1, got {k_index}")
        elif self._sublattice is not None and k_index != self._sublattice.order:
            raise ValueError(
                f"k_index={k_index} does not match the propagation-vector sublattice "
                f"index {self._sublattice.order}."
            )
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be >= 0 or None, got {max_depth}")

        results = []
        family_subgroups = self._family_subgroups(
            k_index=k_index,
            up_to_parent_conjugacy=up_to_parent_conjugacy,
            max_depth=max_depth,
        )
        for family_subgroup in family_subgroups:
            subgroup = family_subgroup.hermann_subgroup

            ssgse = self._spin_space_subgroup_enumerator(family_subgroup)
            space_subgroup = SpaceGroup(
                prim_rotations=ssgse.prim_rotations.tolist(),
                prim_translations=ssgse.prim_translations.tolist(),
                basis=(
                    family_subgroup.translation_sublattice.transformation.T
                    @ np.asarray(self._prim_cell.basis)
                ).tolist(),
            )
            space_subgroup_type = SpaceGroupType(space_subgroup.number)
            logger.info(
                f"Enumerating spin space groups with space subgroup: {space_subgroup_type.hm_short} (No. {space_subgroup_type.number})"
            )
            logger.trace(subgroup)

            results_with_ssgse = self._enumerate_with_ssgse(
                ssgse,
                spin_only_group_type,
                (
                    family_subgroup.relative_k_index
                    if family_subgroup.relative_k_index is not None
                    else k_index
                ),
                coordinate_transformation=family_subgroup.translation_sublattice.transformation,
            )
            logger.info(f"Enumerated SSG-adapted magnetic structures: {len(results_with_ssgse)} ")

            results.extend(results_with_ssgse)

        if len(results) == 0 and self._propagation_vectors is not None:
            logger.warning(
                "Enumeration found zero candidates for propagation vector(s) "
                f"{[pv.tolist() for pv in self._propagation_vectors]} in the primitive "
                f"standardized setting with spin-only group {spin_only_group_type.name}. "
                "This has two common causes. (1) Spin-type incompatibility: the k-forced "
                "translation quotient may have an order that this spin-only group cannot "
                "realize (a COLLINEAR group can only realize a quotient of order <= 2, the "
                "spin flip), in which case zero candidates is the correct physical answer "
                "-- retry with a less restrictive spin-only group type (COPLANAR or "
                "NONCOPLANAR). (2) Frame mismatch: if the vectors were supplied in a "
                "different cell setting than the cell passed to with_propagation_vectors, "
                "compare the round-tripped k logged at construction with the intended one."
            )

        return results

    def _enumerate_with_ssgse(
        self,
        ssgse: SpinSpaceSubgroupEnumerator,
        spin_only_group_type: SpinOnlyGroupType,
        k_index: int,
        *,
        coordinate_transformation: NDArrayInt | None = None,
    ) -> list[tuple[SpinOnlyGroup, NontrivialSpinSpaceGroup, SpinSymmetryAdaptedStructure]]:
        results = []

        if coordinate_transformation is None:
            coordinate_transformation = np.eye(3, dtype=np.int64)

        coordinate_sublattice = Sublattice(coordinate_transformation)
        invariant_subgroup_context: dict[
            int,
            tuple[Supercell, Sequence[SupercellSiteIndex], NDArrayFloat],
        ] = {}

        def retain_invariant_subgroup(normal_space_subgroup: NormalSpaceSubgroup) -> bool:
            parent_sublattice = Sublattice(
                coordinate_sublattice.transformation
                @ normal_space_subgroup.sublattice.transformation
            )
            supercell = Supercell(prim_cell=self._prim_cell, sublattice=parent_sublattice)
            sub_sites = supercell.map_sites(self._magnetic_site_indices)

            invariant_rotations, invariant_translations = (
                coordinate_sublattice.transform_operations_to_parent(
                    ssgse.prim_rotations[normal_space_subgroup.point_subgroup],
                    normal_space_subgroup.translations,
                    atol=self._atol,
                )
            )
            invariant_space_subgroup_reynolds = self._compute_invariant_space_subgroup_reynolds(
                prim_rotations=invariant_rotations,
                prim_translations=invariant_translations,
                supercell=supercell,
                sub_sites=sub_sites,
            )
            if invariant_space_subgroup_reynolds is None:
                return False

            invariant_subgroup_context[id(normal_space_subgroup)] = (
                supercell,
                sub_sites,
                invariant_space_subgroup_reynolds,
            )
            return True

        spin_only_group, grouped_spin_space_groups = (
            ssgse.enumerate_spin_space_groups_by_invariant_subgroup(
                spin_only_group_type,
                k_index,
                invariant_subgroup_filter=retain_invariant_subgroup,
            )
        )
        spin_only_group_reynolds = get_spin_only_group_reynolds_operator(spin_only_group)  # (3, 3)
        for normal_space_subgroup, list_spin_space_groups in grouped_spin_space_groups:
            supercell, sub_sites, invariant_space_subgroup_reynolds = invariant_subgroup_context[
                id(normal_space_subgroup)
            ]
            for nssg in list_spin_space_groups:
                nssg = nssg.transform_to_parent(coordinate_sublattice, atol=self._atol)
                ssa = self._build_ssa_structure(
                    nssg=nssg,
                    supercell=supercell,
                    sub_sites=sub_sites,
                    invariant_space_subgroup_reynolds=invariant_space_subgroup_reynolds,
                    spin_only_group=spin_only_group,
                    spin_only_group_reynolds=spin_only_group_reynolds,
                )
                if ssa is None:
                    continue
                results.append((spin_only_group, nssg, ssa))

        return results

    def _compute_invariant_space_subgroup_reynolds(
        self,
        prim_rotations: NDArrayInt,
        prim_translations: NDArrayFloat,
        supercell: Supercell,
        sub_sites: Sequence[SupercellSiteIndex],
    ) -> NDArrayFloat | None:
        """Return the Reynolds operator of the invariant space subgroup on the
        magnetic sites, or ``None`` if the subgroup does not admit a nontrivial
        invariant representation (site representation missing, or all-zero
        Reynolds eigenvalues).
        """
        invariant_space_subgroup_rep = get_site_representation(
            supercell=supercell,
            sub_sites=sub_sites,
            prim_rotations=prim_rotations,
            prim_translations=prim_translations,
        )
        if invariant_space_subgroup_rep is None:
            return None

        invariant_space_subgroup_reynolds = np.mean(
            invariant_space_subgroup_rep, axis=0
        )  # (num_sub_sites, num_sub_sites)
        invariant_space_subgroup_reynolds_eigvals = np.linalg.eigvals(
            invariant_space_subgroup_reynolds
        )

        if not self._validate_invariant_subgroup_reynolds(
            invariant_space_subgroup_reynolds_eigvals, self._atol
        ):
            logger.debug("    Skip invariant space subgroup with zero eigenvalues.")
            return None

        return invariant_space_subgroup_reynolds

    def _build_ssa_structure(
        self,
        nssg: NontrivialSpinSpaceGroup,
        supercell: Supercell,
        sub_sites: Sequence[SupercellSiteIndex],
        invariant_space_subgroup_reynolds: NDArrayFloat,
        spin_only_group: SpinOnlyGroup,
        spin_only_group_reynolds: NDArrayFloat,
    ) -> SpinSymmetryAdaptedStructure | None:
        """Build a spin-symmetry-adapted structure for a single NSSG.

        Returns ``None`` if the NSSG does not admit a nontrivial SSA structure
        (e.g. the spin translation group is ill-defined, no invariant basis
        exists, or the resulting spin-only group type does not match the
        requested one).
        """
        num_sub_sites = len(sub_sites)

        # Spin translation group
        spin_translation_group_factor_rep = _get_spin_space_group_representation(
            supercell=supercell,
            sub_sites=sub_sites,
            spin_symmetry_operations=nssg.spin_translation_coset,
        )
        if spin_translation_group_factor_rep is None:
            return None
        spin_translation_group_rep = np.einsum(
            "ipaqb,qr,bc->iparc",
            spin_translation_group_factor_rep,
            invariant_space_subgroup_reynolds,
            spin_only_group_reynolds,
            optimize="greedy",
        )
        spin_translation_group_reynolds = np.mean(
            spin_translation_group_rep, axis=0
        )  # (num_sub_sites, 3, num_sub_sites, 3)
        spin_translation_group_reynolds_eigvals = np.linalg.eigvals(
            spin_translation_group_reynolds.reshape(num_sub_sites * 3, num_sub_sites * 3)
        )

        # Check identity representation from spin translation group
        if not np.allclose(
            spin_translation_group_reynolds_eigvals
            * (spin_translation_group_reynolds_eigvals - 1),
            0,
            atol=self._atol,
        ):
            logger.warning("Given spin translation group is ill-defined.")
            return None
        if np.allclose(spin_translation_group_reynolds_eigvals, 0, atol=self._atol):
            logger.trace("    Skip spin translation group with zero eigenvalues.")
            return None

        # Spin space group
        spin_space_group_factor_rep = _get_spin_space_group_representation(
            supercell=supercell,
            sub_sites=sub_sites,
            spin_symmetry_operations=nssg.nontrivial_coset,
        )
        if spin_space_group_factor_rep is None:
            return None
        spin_space_group_rep = np.einsum(
            "ipaqb,qbrc->iparc",
            spin_space_group_factor_rep,
            spin_translation_group_reynolds,
            optimize="greedy",
        )
        spin_space_group_reynolds = np.mean(spin_space_group_rep, axis=0)

        try:
            bases = _determine_real_order_parameter_directions(
                spin_space_group_reynolds.reshape(num_sub_sites * 3, num_sub_sites * 3),
                atol=self._atol,
            )
        except AssertionError:
            logger.warning("Failed to generate basis.")
            return None
        if len(bases) == 0:
            return None

        # Generate magnetic structure
        list_magmoms = self._assemble_magnetic_moments(
            bases, sub_sites, supercell.num_supercell_sites
        )

        # TODO: We cannot judge whether a final magnetic configuration satisfies the given spin-only group type when len(list_magmoms) >= 2 so far.
        # For example, two collinear magmom bases forms a coplanar structure.
        if (len(list_magmoms) == 1) and (
            get_spin_only_group(list_magmoms[0], self._atol).spin_only_group_type
            != spin_only_group.spin_only_group_type
        ):
            logger.trace("    skip because spin only group type does not match.")
            return None

        return SpinSymmetryAdaptedStructure(
            supercell=supercell, magnetic_moments_basis=list_magmoms
        )

    @staticmethod
    def _validate_invariant_subgroup_reynolds(
        reynolds_eigvals: NDArrayFloat,
        atol: float,
    ) -> bool:
        """
        Validate Reynolds operator eigenvalues are 0 or 1.

        Returns True if valid and non-zero, False if all zero.
        Raises ValueError if eigenvalues are invalid.
        """
        eigval_product = reynolds_eigvals * (reynolds_eigvals - 1)
        if not np.allclose(eigval_product, 0, atol=atol):
            raise ValueError(
                "Reynolds operator eigenvalues must be 0 or 1. "
                f"Got eigenvalues: {reynolds_eigvals}"
            )
        if np.allclose(reynolds_eigvals, 0, atol=atol):
            return False
        return True

    @staticmethod
    def _assemble_magnetic_moments(
        bases: list[NDArrayFloat],
        sub_sites: Sequence[SupercellSiteIndex],
        num_supercell_sites: int,
    ) -> list[NDArrayFloat]:
        """Assemble magnetic moments from basis vectors."""
        list_magmoms = []
        for basis in bases:
            magmoms = np.zeros((num_supercell_sites, 3), dtype=np.float64)
            magmoms[sub_sites] = basis.reshape(-1, 3)
            list_magmoms.append(magmoms)
        return list_magmoms


def _get_spin_space_group_representation(
    supercell: Supercell,
    sub_sites: Sequence[SupercellSiteIndex],
    spin_symmetry_operations: SpinSymmetryOperations,
) -> NDArrayFloat | None:
    """Return representation of spin space group by configuration."""
    site_rep = get_site_representation(
        supercell=supercell,
        sub_sites=sub_sites,
        prim_rotations=spin_symmetry_operations.rotations,
        prim_translations=spin_symmetry_operations.translations,
    )
    if site_rep is None:
        return None

    rep = (
        site_rep[:, :, None, :, None]
        * spin_symmetry_operations.spin_rotations[:, None, :, None, :]
    )
    return rep


def _determine_real_order_parameter_directions(
    reynolds: NDArrayFloat,
    atol: float,
) -> list[NDArrayFloat]:
    eigvals = np.linalg.eigvals(reynolds)
    assert np.allclose(eigvals**2, eigvals, atol=atol), (
        f"Eigenvalues are not 0 or 1: {np.around(eigvals, 6)}"
    )
    num_invariant = int(np.sum(np.isclose(eigvals, 1, atol=atol)))
    if num_invariant == 0:
        return []

    # The invariant subspace {x : P x = x} is the null space of (P - I),
    # extracted with a rank-revealing SVD of the real matrix directly. Do
    # NOT extract it from np.linalg.eig eigenvectors: inside a degenerate
    # eigenvalue-1 block, eig may return complex-conjugate pairs with
    # arbitrary phases, and any real/imag splitting of those is
    # phase-dependent -- it can duplicate a 2D real subspace (which the
    # varimax rotation below then turns into one sqrt(2)-norm vector plus
    # one exactly-zero vector, overcounting `dim`) or drop a direction
    # hidden in a small component. The right singular vectors are real,
    # orthonormal, and phase-free. The threshold separates cleanly: (I - P)
    # is itself a projector, whose nonzero singular values are >= 1, while
    # the null-space singular values sit at the numerical-noise floor.
    singular_values, vh = np.linalg.svd(reynolds - np.eye(reynolds.shape[0]))[1:]
    bases = list(vh[singular_values < atol])
    if len(bases) != num_invariant:
        logger.warning(
            f"Invariant-subspace rank ({len(bases)}) disagrees with the eigenvalue-1 "
            f"count ({num_invariant}); returning the rank-revealed basis."
        )
    if len(bases) == 0:
        return []

    # Varimax rotation to maximize sparsity of real-valued basis vectors
    if len(bases) >= 2:
        stacked = np.array(bases)
        bases = list(_varimax_rotation(stacked))

    return bases


def _varimax_rotation(
    basis: NDArrayFloat,
    max_iter: int = 1000,
    atol: float = 1e-9,
) -> NDArrayFloat:
    """Apply varimax rotation to maximize sparsity of basis vectors.

    Uses Kaiser's varimax criterion: maximize sum of variances of squared loadings.
    Iterates pairwise Givens rotations until convergence.

    Parameters
    ----------
    basis : (dim, n) float array
        Orthonormal basis vectors as rows.
    max_iter : int
        Maximum number of full sweeps.
    atol : float
        Convergence tolerance on rotation angle.

    Returns
    -------
    (dim, n) float array
        Rotated orthonormal basis vectors.
    """
    dim = basis.shape[0]
    if dim < 2:
        return basis.copy()

    n = basis.shape[1]
    rotated = basis.copy()
    for _ in range(max_iter):
        max_angle = 0.0
        for i in range(dim):
            for j in range(i + 1, dim):
                u = rotated[i] ** 2 - rotated[j] ** 2
                v = 2.0 * rotated[i] * rotated[j]
                A = np.sum(v)
                B = np.sum(u)
                C = np.sum(u**2 - v**2)
                D = 2.0 * np.sum(u * v)
                angle = 0.25 * np.arctan2(D - 2.0 * A * B / n, C - (A**2 - B**2) / n)
                max_angle = max(max_angle, abs(angle))
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                ri = cos_a * rotated[i] + sin_a * rotated[j]
                rj = -sin_a * rotated[i] + cos_a * rotated[j]
                rotated[i] = ri
                rotated[j] = rj
        if max_angle < atol:
            break

    return rotated


def _sample_on_unit_sphere(rng: np.random.Generator, n: int, size: int = 1) -> NDArrayFloat:
    """Return random points from a surface of n-dimensional unit sphere.

    Ref: M. E. Muller, Communications of the ACM 2.4, 19-20 (1959).

    Parameters
    ----------
    rng: numpy's random generator
    n: int
        Number of variables
    size: int, default=1
        Number of points to be sampled

    Returns
    -------
    points: array, (size, n)
    """
    points = rng.standard_normal((size, n))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points
