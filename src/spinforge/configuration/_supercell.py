import itertools
from collections.abc import Sequence
from typing import Annotated, NamedTuple

import numpy as np
from moyopy import Cell
from spgrep.utils import NDArrayFloat, NDArrayInt

from spinforge.space_group import Sublattice

PrimSiteIndex = Annotated[int, "Index of site in the primitive cell"]
SupercellSiteIndex = Annotated[int, "Index of site in the supercell"]


class PrimSiteImage(NamedTuple):
    """One supercell site seen from the primitive cell."""

    prim_site: PrimSiteIndex
    prim_frac: NDArrayFloat
    """Unwrapped fractional coordinates in the *primitive* basis."""


class Supercell:
    def __init__(self, prim_cell: Cell, sublattice: Sublattice, symprec: float = 1e-4) -> None:
        self._prim_cell = prim_cell
        self._sublattice = sublattice
        self._epsilon = symprec / (np.abs(np.linalg.det(self._prim_cell.basis)) ** (1 / 3))

        self._supercell_basis: NDArrayFloat = sublattice.transformation.T @ np.array(
            prim_cell.basis
        )  # row major
        supercell_positions = []
        supercell_numbers = []
        prim_fractional_coordinates = []

        for image, _, factor in sublattice.lattice_points:
            for kappa, (position_kappa, number_kappa) in enumerate(
                zip(prim_cell.positions, prim_cell.numbers)
            ):
                fc = np.array(position_kappa) + np.array(image)
                new_position = np.remainder(sublattice.inverse_transformation @ fc, 1)
                prim_fractional_coordinates.append(PrimSiteImage(kappa, fc))
                supercell_positions.append(new_position)
                supercell_numbers.append(number_kappa)
        self._supercell_positions = np.array(supercell_positions)
        self._supercell_numbers = supercell_numbers
        self._prim_fractional_coordinates: list[PrimSiteImage] = prim_fractional_coordinates

    @property
    def prim_cell(self) -> Cell:
        return self._prim_cell

    @property
    def sublattice(self) -> Sublattice:
        return self._sublattice

    @property
    def basis(self) -> NDArrayFloat:
        return self._supercell_basis

    @property
    def positions(self) -> NDArrayFloat:
        """Return fractional coordinates of sites in the *supercell*."""
        return self._supercell_positions

    @property
    def numbers(self) -> list[int]:
        return self._supercell_numbers

    @property
    def num_supercell_sites(self) -> int:
        return len(self._supercell_positions)

    def map_sites(self, prim_site_indices: Sequence[PrimSiteIndex]) -> list[SupercellSiteIndex]:
        ret = []
        for i, entry in enumerate(self._prim_fractional_coordinates):
            if entry.prim_site in prim_site_indices:
                ret.append(i)
        return ret

    def site_permutations(
        self,
        rotations: Sequence[NDArrayInt],
        translations: Sequence[NDArrayFloat],
    ) -> list[NDArrayInt]:
        """Supercell-site permutations under primitive-frame operations.

        For each ``(rotation, translation)`` pair, returns the array mapping
        each supercell site index to its image site index (brute-force
        O(n^2) matching). Raises if an operation does not permute the sites
        within the symprec tolerance.
        """
        permutations = []
        for rotation, translation in zip(rotations, translations):
            indices = self._image_indices(rotation, translation)
            if indices is None:
                raise ValueError(
                    f"Operation ({rotation}, {translation}) does not permute the sites"
                )
            permutations.append(indices)
        return permutations

    def act_operation(
        self,
        sub_sites: Sequence[SupercellSiteIndex],
        prim_rotation: NDArrayInt,
        prim_translation: NDArrayFloat,
    ) -> dict[SupercellSiteIndex, SupercellSiteIndex] | None:
        """Permutation of ``sub_sites`` under one primitive-frame operation.

        Returns the mapping from each site in ``sub_sites`` to its image
        site, or None when the operation does not permute ``sub_sites``
        within the symprec tolerance.
        """
        indices = self._image_indices(prim_rotation, prim_translation, sub_sites=sub_sites)
        if indices is None:
            return None
        return dict(zip(sub_sites, indices))

    def _image_indices(
        self,
        rotation: NDArrayInt,
        translation: NDArrayFloat,
        sub_sites: Sequence[SupercellSiteIndex] | None = None,
    ) -> NDArrayInt | None:
        """Image site index per site in ``sub_sites`` (all sites when None),
        or None when the operation does not permute them within tolerance.

        Brute-force O(n^2) matching: every pair is tested with the
        primitive-frame per-component modular tolerance, and each image must
        match exactly one site (an ambiguous match means two sites lie
        inside one tolerance box, i.e. symprec is too coarse for the
        structure; that is treated as failure rather than resolved).
        """
        rotation = np.asarray(rotation, dtype=float)
        if not np.isclose(np.abs(np.linalg.det(rotation)), 1.0, rtol=0, atol=1e-8):
            raise ValueError(f"Rotation must have |det| = 1, got {rotation}")
        selected = (
            np.arange(self.num_supercell_sites)
            if sub_sites is None
            else np.asarray(list(sub_sites), dtype=int)
        )
        if len(selected) == 0:
            return np.empty(0, dtype=int)
        prim_coords = np.array([self._prim_fractional_coordinates[i].prim_frac for i in selected])
        transformation = np.array(self._sublattice.transformation, dtype=float)
        inverse_transformation = np.array(self._sublattice.inverse_transformation, dtype=float)
        acted = prim_coords @ rotation.T + translation
        images = np.remainder(acted @ inverse_transformation.T, 1.0)

        delta = images[:, None, :] - self._supercell_positions[selected][None, :, :]
        delta -= np.rint(delta)  # wrap to the nearest supercell lattice image
        # For a skew transformation the nearest image in supercell coordinates
        # is not always the one minimizing the primitive-frame residual; probe
        # every lattice offset that could still land inside the epsilon box
        # (reach is 0 unless the transformation is strongly non-reduced).
        reach = int(np.sum(np.abs(inverse_transformation), axis=1).max() * self._epsilon + 0.5)
        within = np.zeros((len(selected), len(selected)), dtype=bool)
        for offset in itertools.product(range(-reach, reach + 1), repeat=3):
            shifted = delta - np.asarray(offset, dtype=float)
            within |= np.all(np.abs(shifted @ transformation.T) <= self._epsilon, axis=2)
        if np.any(np.sum(within, axis=1) != 1):
            return None
        indices = selected[np.argmax(within, axis=1)]
        if len(np.unique(indices)) != len(selected):
            return None
        return indices
