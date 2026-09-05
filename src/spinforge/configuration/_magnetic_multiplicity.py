from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from moyopy import Cell, MoyoDataset
from spgrep.utils import NDArrayFloat, NDArrayInt

from spinforge.space_group import Sublattice
from spinforge.utils.group_theory import SubgroupIndices

from ._supercell import Supercell


class MagneticMultiplicityClassifier:
    """Apply the magnetic-site multiplicity policy to family subgroups.

    The classifier does not enumerate or classify subgroup conjugacy. Given a
    parent primitive cell and selected magnetic sites, it identifies the
    complete parent crystallographic orbits containing those sites and accepts
    a family space subgroup only when every resulting subgroup orbit has the
    same multiplicity as its parent primitive-cell orbit.

    Translationengleiche subgroups use the orbit-stabilizer relation in the
    finite quotient ``G / T``. General subgroups use their full affine
    operations on the supercell defined by their translation sublattice; this
    includes nonsymmorphic translations and distinct lattice-point copies. The
    two public predicates are optimized implementations of the same policy.

    Attributes
    ----------
    _translationengleiche_contexts : list[tuple[tuple[bool, ...], int]]
        One entry per distinct magnetic parent-space-group orbit. The boolean
        tuple marks which parent operations fix a representative site, and the
        integer is that orbit's original multiplicity. These cached values make
        repeated t-subgroup checks finite-index calculations.

    Parameters
    ----------
    prim_cell : Cell
        Primitive cell.
    prim_dataset : MoyoDataset
        Symmetry dataset for the primitive cell.
    magnetic_site_indices : Sequence[int]
        Indices of magnetic sites in the primitive cell.
    """

    def __init__(
        self,
        prim_cell: Cell,
        prim_dataset: MoyoDataset,
        magnetic_site_indices: Sequence[int],
    ):
        if len(magnetic_site_indices) == 0:
            raise ValueError("No magnetic site is found.")

        self._prim_rotations = np.array(prim_dataset.operations.rotations, dtype=np.int64)
        self._prim_translations = np.array(prim_dataset.operations.translations)
        self._prim_cell = prim_cell
        self._prim_dataset = prim_dataset
        self._magnetic_site_indices = list(magnetic_site_indices)
        orbits = np.asarray(prim_dataset.orbits)
        self._translationengleiche_contexts: list[tuple[tuple[bool, ...], int]] = []
        for orbit in set(orbits[magnetic_site_indices]):
            idx = int(np.where(orbits == orbit)[0][0])
            position = np.asarray(prim_cell.positions[idx], dtype=float)
            is_site_symmetry = tuple(
                self._get_site_symmetry_indicators(
                    prim_cell,
                    self._prim_rotations,
                    self._prim_translations,
                    position,
                    prim_dataset.symprec,
                )
            )
            original_multiplicity = int(np.count_nonzero(orbits == orbit))
            site_symmetry_count = sum(is_site_symmetry)
            if len(self._prim_rotations) % site_symmetry_count != 0:
                raise ValueError(
                    f"Primitive rotations count ({len(self._prim_rotations)}) must be "
                    f"divisible by site symmetry count ({site_symmetry_count})"
                )
            computed_multiplicity = len(self._prim_rotations) // site_symmetry_count
            if computed_multiplicity != original_multiplicity:
                raise ValueError(
                    f"Computed multiplicity ({computed_multiplicity}) does not match "
                    f"original multiplicity ({original_multiplicity})"
                )
            self._translationengleiche_contexts.append((is_site_symmetry, original_multiplicity))

    @property
    def prim_rotations(self) -> NDArrayInt:
        return self._prim_rotations

    @property
    def prim_translations(self) -> NDArrayFloat:
        return self._prim_translations

    def preserves_translationengleiche_multiplicity(
        self,
        subgroup: SubgroupIndices,
    ) -> bool:
        """Whether a t-subgroup preserves every magnetic-site multiplicity."""
        for is_site_symmetry, target in self._translationengleiche_contexts:
            site_symmetry_count = sum(is_site_symmetry[index] for index in subgroup)
            if len(subgroup) % site_symmetry_count != 0:
                raise ValueError(
                    f"Subgroup size ({len(subgroup)}) must be divisible by "
                    f"site symmetry subgroup size ({site_symmetry_count})"
                )
            if len(subgroup) // site_symmetry_count != target:
                return False
        return True

    def preserves_magnetic_multiplicity(
        self,
        sublattice: Sublattice,
        rotations: NDArrayInt,
        translations: NDArrayFloat,
    ) -> bool:
        """Whether a space subgroup preserves every magnetic-site multiplicity."""
        family_cell = Supercell(
            self._prim_cell,
            sublattice,
            symprec=self._prim_dataset.symprec,
        )
        parent_orbits = np.asarray(self._prim_dataset.orbits)
        magnetic_parent_orbits = set(parent_orbits[self._magnetic_site_indices])
        magnetic_family_orbits = tuple(
            (
                set(family_cell.map_sites(np.where(parent_orbits == orbit)[0].tolist())),
                int(np.count_nonzero(parent_orbits == orbit)),
            )
            for orbit in magnetic_parent_orbits
        )
        permutations = family_cell.site_permutations(list(rotations), list(translations))
        return all(
            _all_orbits_have_size(
                family_sites,
                permutations,
                target_size=target_size,
            )
            for family_sites, target_size in magnetic_family_orbits
        )

    @staticmethod
    def _get_site_symmetry_indicators(
        prim_cell: Cell,
        prim_rotations: NDArrayInt,
        prim_translations: NDArrayFloat,
        position: NDArrayFloat,
        symprec: float,
    ) -> list[bool]:
        """Compute which symmetry operations fix the given position."""
        is_site_symmetry = [False for _ in range(len(prim_rotations))]
        for i, (rotation, translation) in enumerate(zip(prim_rotations, prim_translations)):
            new_position = rotation @ position + translation
            diff = new_position - position
            diff -= np.rint(diff)
            if np.linalg.norm(np.array(prim_cell.basis).T @ diff) < symprec:
                is_site_symmetry[i] = True
        return is_site_symmetry


def _all_orbits_have_size(
    sites: set[int],
    permutations: list[NDArrayInt],
    *,
    target_size: int,
) -> bool:
    """Check orbit sizes for permutations known to form a subgroup action.

    Every subgroup of the parent space group must preserve each parent orbit.
    An escaping image therefore signals inconsistent operations or site mapping.
    """
    unseen = set(sites)
    while unseen:
        seed = min(unseen)
        orbit = {seed}
        frontier = [seed]
        while frontier:
            site = frontier.pop()
            for permutation in permutations:
                image = int(permutation[site])
                if image not in sites:
                    raise ValueError(
                        "Space-subgroup operation escaped its parent-space-group orbit."
                    )
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        unseen -= orbit
        if len(orbit) != target_size:
            return False
    return True
