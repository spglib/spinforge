from __future__ import annotations

from itertools import product
from typing import Annotated

import numpy as np
from spgrep.rep.group import get_identity_index, get_inverse_index
from spgrep.utils import NDArrayInt

FactorGroupElementIndex = Annotated[int, "Index of factor group element"]
GroupElementIndex = Annotated[int, "Index of group element"]
SubgroupIndices = frozenset[GroupElementIndex]


def all_inverses(table: NDArrayInt) -> list[GroupElementIndex]:
    """Return the inverse of each group element, indexed by the Cayley table."""
    return [get_inverse_index(table, g) for g in range(len(table))]


def get_coset_representatives(
    normal_subgroup: list[GroupElementIndex],
    table: NDArrayInt,
) -> tuple[list[GroupElementIndex], NDArrayInt]:
    representatives = []
    identity = get_identity_index(table)
    representatives.append(identity)
    visited = set(normal_subgroup)

    # Coset decomposition
    order = len(table)
    for g in range(order):
        if g in visited:
            continue
        representatives.append(g)
        for h in normal_subgroup:
            visited.add(int(table[g, h]))
    num_cosets = len(representatives)
    expected_size = num_cosets * len(normal_subgroup)
    if expected_size != len(visited):
        raise ValueError(
            f"Coset decomposition failed: expected {expected_size} elements "
            f"({num_cosets} cosets × {len(normal_subgroup)} normal subgroup size), "
            f"but visited {len(visited)} elements"
        )

    # Factor group
    factor_group_table = np.zeros((num_cosets, num_cosets), dtype=int)
    for (i, gi), (j, gj) in product(enumerate(representatives), repeat=2):
        gk: GroupElementIndex = int(table[gi, gj])
        gH = [int(table[gk, h]) for h in normal_subgroup]
        for k in range(num_cosets):
            if representatives[k] in gH:
                factor_group_table[i, j] = k
                break

    return representatives, factor_group_table
