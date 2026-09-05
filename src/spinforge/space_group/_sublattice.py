from __future__ import annotations

from itertools import product
from typing import Annotated

import numpy as np
from hsnf.Z_module import smith_normal_form
from spgrep.utils import NDArrayFloat, NDArrayInt, is_integer_array

Image = Annotated[tuple[int, int, int], "Image"]
Factor = Annotated[tuple[int, int, int], "Factor"]


class Sublattice:
    def __init__(self, transformation: NDArrayInt):
        det = np.linalg.det(transformation)
        if det <= 0:
            raise ValueError(
                f"Sublattice transformation must have a positive determinant, got {det}"
            )
        self._transformation = transformation
        self._inverse_transformation = np.linalg.inv(transformation)

        snf, L, _ = smith_normal_form(transformation)  # snf = L @ transformation @ R
        self._L = L
        Linv = np.linalg.inv(L)

        # Factors of SNF correspond to generators of the sublattice
        self._invariant_factors = tuple(snf.diagonal().tolist())
        generators: list[tuple[int, Image, Factor]] = []
        for i, factor in enumerate(self._invariant_factors):
            if factor == 1:
                continue
            onehot = np.zeros(3, dtype=int)
            onehot[i] = 1
            translation = np.around(Linv @ onehot).astype(int)
            generators.append((int(factor), tuple(translation.tolist()), tuple(onehot.tolist())))
        self._generators = generators

        generator_orders = [order for order, _, _ in generators]
        lattice_points: list[tuple[Image, list[int], Factor]] = []
        for counts in product(*[range(order) for order in generator_orders]):
            factor = np.zeros(3, dtype=int)
            for (_, _, generator_factor), count in zip(generators, counts):
                factor += count * np.array(generator_factor)
            image = np.around(Linv @ factor).astype(int)
            lattice_points.append((tuple(image.tolist()), list(counts), tuple(factor.tolist())))
        self._lattice_points = lattice_points

    @property
    def transformation(self) -> NDArrayInt:
        return self._transformation

    @property
    def inverse_transformation(self) -> NDArrayFloat:
        return self._inverse_transformation

    @property
    def order(self) -> int:
        return len(self.lattice_points)

    @property
    def generators(self) -> list[tuple[int, Image, Factor]]:
        return self._generators

    @property
    def lattice_points(self) -> list[tuple[Image, list[int], Factor]]:
        return self._lattice_points

    def convert_to_factor(self, image: Image) -> Factor:
        factor = np.remainder(self._L @ np.array(image), self._invariant_factors)
        return tuple(np.around(factor).astype(int).tolist())

    def try_convert_to_factor(self, translation: NDArrayFloat, atol: float = 1e-5) -> Factor:
        if not is_integer_array(np.asarray(translation), atol=atol):
            raise ValueError(
                f"Translation must be an integer array within tolerance {atol}. Got: {translation}"
            )
        image = tuple(np.around(translation).astype(int).tolist())
        return self.convert_to_factor(image)

    def transform_operations(
        self,
        rotations: NDArrayInt,
        translations: NDArrayFloat,
        atol: float = 1e-5,
    ) -> tuple[NDArrayInt, NDArrayFloat]:
        """Change operations into the sublattice basis P (ITA convention).

        W' = P^-1 @ W @ P, w' = P^-1 @ w. Raises if a transformed rotation is
        not integral in the sublattice basis.
        """
        transformation = self.transformation
        inverse_transformation = self.inverse_transformation
        new_rotations = []
        new_translations = []
        for rotation, translation in zip(rotations, translations):
            new_rotation = inverse_transformation @ rotation @ transformation
            if not is_integer_array(new_rotation, atol=atol):
                raise ValueError(
                    f"Rotation {np.asarray(rotation).tolist()} is not integral in the "
                    f"sublattice basis {transformation.tolist()}: got {new_rotation.tolist()}"
                )
            new_rotations.append(np.around(new_rotation).astype(int))
            new_translations.append(inverse_transformation @ translation)
        return np.array(new_rotations), np.array(new_translations)

    def transform_operations_to_parent(
        self,
        rotations: NDArrayInt,
        translations: NDArrayFloat,
        atol: float = 1e-5,
    ) -> tuple[NDArrayInt, NDArrayFloat]:
        """Change operations from the sublattice basis P to the parent basis.

        ``W = P @ W' @ P^-1`` and ``w = P @ w'``. Raises if a transformed
        rotation is not integral in the parent basis.
        """
        new_rotations = []
        new_translations = []
        for rotation, translation in zip(rotations, translations):
            new_rotation = self.transformation @ rotation @ self.inverse_transformation
            if not is_integer_array(new_rotation, atol=atol):
                raise ValueError(
                    f"Rotation {np.asarray(rotation).tolist()} is not integral in the "
                    f"parent basis of {self.transformation.tolist()}: "
                    f"got {new_rotation.tolist()}"
                )
            new_rotations.append(np.around(new_rotation).astype(int))
            new_translations.append(self.transformation @ translation)
        return np.array(new_rotations), np.array(new_translations)

    def __repr__(self) -> str:
        return f"Sublattice(transformation={self.transformation.tolist()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sublattice):
            raise ValueError("Comparison is only supported between Sublattice instances.")
        return np.array_equal(self.transformation, other.transformation)

    def is_normal(self, prim_rotation: NDArrayInt, *, atol: float = 1e-5) -> bool:
        return _is_compatible(
            self.transformation,
            self.inverse_transformation,
            prim_rotation,
            atol=atol,
        )

    def contains(self, sublattice: Sublattice, *, atol: float = 1e-5) -> bool:
        """Return whether this lattice contains ``sublattice``."""
        relative = self.inverse_transformation @ sublattice.transformation
        return is_integer_array(relative, atol=atol)

    def relative_sublattice(
        self,
        sublattice: Sublattice,
        *,
        atol: float = 1e-5,
    ) -> Sublattice | None:
        """Express a contained ``sublattice`` in this lattice's basis.

        Return ``None`` when ``sublattice`` is not contained in this lattice.
        """
        if not self.contains(sublattice, atol=atol):
            return None
        relative = self.inverse_transformation @ sublattice.transformation
        return Sublattice(np.rint(relative).astype(np.int64))


def enumerate_normal_sublattices(prim_rotations: NDArrayInt, index: int) -> list[Sublattice]:
    all_sublattices = enumerate_sublattices(index)

    normal_sublattices = []
    for sublattice in all_sublattices:
        is_normal = True
        for rotation in prim_rotations:
            if not sublattice.is_normal(rotation):
                is_normal = False
                break
        if is_normal:
            normal_sublattices.append(sublattice)

    return normal_sublattices


def enumerate_sublattices(index: int) -> list[Sublattice]:
    """Enumerate index-``index`` sublattices in Hermite normal form."""
    if index < 1:
        raise ValueError(f"Sublattice index must be >= 1, got {index}")

    def make_HNF(a, b, c, d, e, f):
        arr = np.zeros((3, 3), dtype=int)
        arr[np.tril_indices(3)] = np.array([a, b, c, d, e, f])
        return arr

    list_HNF = []
    for a in range(1, index + 1):
        if index % a != 0:
            continue
        for c in range(1, index // a + 1):
            if (index % c != 0) or (index % (a * c) != 0):
                continue
            f = index // (a * c)
            list_HNF.extend(
                [make_HNF(a, b, c, d, e, f) for b in range(c) for d in range(f) for e in range(f)]
            )

    return [Sublattice(transformation) for transformation in list_HNF]


def _is_unimodular(matrix: NDArrayFloat, atol: float = 1e-5) -> bool:
    if not np.isclose(np.abs(np.linalg.det(matrix)), 1, atol=atol):
        return False
    return is_integer_array(matrix, atol=atol)


def _is_compatible(
    transformation: NDArrayInt,
    inverse_transformation: NDArrayFloat,
    rotation: NDArrayInt,
    atol: float = 1e-5,
) -> bool:
    rotated = inverse_transformation @ rotation @ transformation
    return _is_unimodular(rotated, atol=atol)
