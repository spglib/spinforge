from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import sqrtm
from spgrep._constants import ATOL, MAX_NUM_RANDOM_GENERATIONS
from spgrep.rep.irreps import frobenius_schur_indicator, is_equivalent_irrep
from spgrep.rep.pir import get_physically_irrep
from spgrep.rep.representation import get_character
from spgrep.symmetry.enumerate import enumerate_unitary_irreps_from_regular_representation
from spgrep.utils import NDArrayComplex, NDArrayFloat, NDArrayInt

from ._representation import get_regular_representation


@dataclass(frozen=True)
class RealIrrepType:
    dim: int


def get_real_irreps(irreps: list[NDArrayComplex]) -> list[tuple[NDArrayFloat, RealIrrepType]]:
    conjugated_pairs = []
    visited = [False for _ in range(len(irreps))]
    characters = [get_character(irrep) for irrep in irreps]
    for i, ci in enumerate(characters):
        if visited[i]:
            continue
        visited[i] = True
        inequivalent = False
        for j, cj in enumerate(characters):
            if visited[j]:
                continue
            if is_equivalent_irrep(np.conj(ci), cj):
                conjugated_pairs.append((i, j))
                visited[j] = True
                inequivalent = True
                break
        if not inequivalent:
            conjugated_pairs.append((i, i))

    real_irreps = []
    for conj_pair in conjugated_pairs:
        irrep = irreps[conj_pair[0]]
        indicator = frobenius_schur_indicator(irrep)
        real_irrep = get_physically_irrep(irrep, indicator)
        real_irreps.append((real_irrep, RealIrrepType(dim=real_irrep.shape[1])))

    return real_irreps


def get_real_intertwiner(
    rep1: NDArrayFloat,
    rep2: NDArrayFloat,
    *,
    atol: float = ATOL,
    max_num_random_generations: int = MAX_NUM_RANDOM_GENERATIONS,
) -> NDArrayFloat | None:
    """Calculate intertwiner matrix between ``rep1`` and ``rep2`` such that ``rep1 @ matrix == matrix @ rep2`` if they are equivalent.

    Returns
    -------
    matrix: array, (dim, dim)
    """
    assert rep1.shape == rep2.shape
    dim = rep1.shape[1]

    # Check by characters
    if not np.allclose(
        get_character(rep1.astype(np.complex128)),
        get_character(rep2.astype(np.complex128)),
        atol=atol,
    ):
        return None

    rng = np.random.default_rng(0)
    for _ in range(max_num_random_generations):
        random = rng.random((dim, dim))
        matrix: NDArrayFloat = np.einsum("kil,lm,kjm->ij", rep1, random, rep2, optimize="greedy")

        if np.linalg.norm(matrix) < atol:
            continue
        orthogonal_intertwiner = matrix @ np.linalg.inv(sqrtm(matrix.T @ matrix))
        if dim % 2 == 1:
            # Ensure det = 1 for odd dimension cases
            orthogonal_intertwiner *= np.linalg.det(orthogonal_intertwiner)
        return orthogonal_intertwiner

    return None


def get_orientation_preserving_real_intertwiner(
    rep1: NDArrayFloat,
    rep2: NDArrayFloat,
    table: NDArrayInt,
    *,
    atol: float = ATOL,
    max_num_random_generations: int = MAX_NUM_RANDOM_GENERATIONS,
) -> NDArrayFloat | None:
    """Return an orientation-preserving intertwiner matrix such that ``rep1 @ matrix == matrix @ rep2`` if they are equivalent and such a matrix exists."""
    assert rep1.shape == rep2.shape

    orthogonal_intertwiner = get_real_intertwiner(
        rep1, rep2, atol=atol, max_num_random_generations=max_num_random_generations
    )
    if orthogonal_intertwiner is None:
        return None

    # Now, rep1 and rep2 are equivalent in R
    reflection = _try_reflection_in_centralizer(
        rep1, table, atol=atol, max_num_random_generations=max_num_random_generations
    )
    if np.linalg.det(orthogonal_intertwiner) > 0:
        return orthogonal_intertwiner
    else:
        if reflection is not None:
            # rep1 is direct sum of two 1d real irreps
            return reflection @ orthogonal_intertwiner
        else:
            # No way to make it orientation-preserving
            return None


def _try_reflection_in_centralizer(
    rep: NDArrayFloat,
    table: NDArrayInt,
    *,
    atol: float,
    max_num_random_generations: int,
) -> NDArrayFloat | None:
    order = len(table)
    dim = rep.shape[1]
    character = np.real(get_character(rep.astype(np.complex128)))

    reg = get_regular_representation(table)
    irreps: list[NDArrayComplex] = enumerate_unitary_irreps_from_regular_representation(
        reg.astype(np.complex128), max_num_random_generations=max_num_random_generations
    )

    # Choose active 1D real irrep if exists
    for irrep in irreps:
        if (irrep.shape[1] != 1) or (not np.allclose(irrep, np.conj(irrep), atol=atol)):
            continue

        character_1d = np.real(get_character(irrep))
        count = np.real(np.sum(character_1d * character)) / order
        count = round(count)
        if count == 0:
            continue

        projector = np.sum(character_1d[:, None, None] * rep, axis=0) / order
        reflection = 2 * projector - np.eye(dim)
        if np.allclose(reflection, np.eye(dim), atol=atol):
            # Choose anything det=-1
            ret = np.eye(dim)
            ret[0, 0] = -1
            return ret
        assert np.isclose(np.linalg.det(reflection), -1)
        return reflection

    return None
