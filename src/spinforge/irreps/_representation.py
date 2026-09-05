from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from spgrep.utils import NDArrayFloat, NDArrayInt

if TYPE_CHECKING:
    from ._real import RealIrrepType


def get_regular_representation(table: NDArrayInt) -> NDArrayInt:
    n = len(table)
    reg = np.zeros((n, n, n), dtype=int)
    for k, j in product(range(n), repeat=2):
        reg[k, table[k, j], j] = 1

    return reg


def multi_direct_sum(
    representations: list[tuple[NDArrayFloat, RealIrrepType]],
    combination: list[int],
) -> tuple[NDArrayFloat, list[RealIrrepType]]:
    direct_sum_rep, _ = representations[combination[0]]
    for idx in combination[1:]:
        rep, real_irrep_type = representations[idx]
        direct_sum_rep = _direct_sum(direct_sum_rep, rep)

    real_irrep_types = []
    for idx in combination:
        _, real_irrep_type = representations[idx]
        real_irrep_types.append(real_irrep_type)

    return direct_sum_rep, real_irrep_types


def _direct_sum(rep1: NDArrayFloat, rep2: NDArrayFloat) -> NDArrayFloat:
    assert len(rep1) == len(rep2)
    dim1 = rep1.shape[1]
    dim2 = rep2.shape[1]
    rep = np.zeros((len(rep1), dim1 + dim2, dim1 + dim2), dtype=rep1.dtype)
    rep[:, :dim1, :dim1] = rep1
    rep[:, dim1:, dim1:] = rep2
    return rep
