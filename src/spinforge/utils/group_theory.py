from __future__ import annotations

import itertools
from collections.abc import Collection
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
    for (i, gi), (j, gj) in itertools.product(enumerate(representatives), repeat=2):
        gk: GroupElementIndex = int(table[gi, gj])
        gH = [int(table[gk, h]) for h in normal_subgroup]
        for k in range(num_cosets):
            if representatives[k] in gH:
                factor_group_table[i, j] = k
                break

    return representatives, factor_group_table


def _subgroup_bound(
    values: Collection[GroupElementIndex], table: NDArrayInt, name: str
) -> SubgroupIndices:
    """Validate one bound as a subgroup using the input table indices."""
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or not 0 <= index < len(table)
        for index in values
    ):
        raise ValueError(f"{name} must contain integer indices in [0, {len(table)})")
    indices = frozenset(int(index) for index in values)
    selected = sorted(indices)
    if (
        get_identity_index(table) not in indices
        or not set(table[np.ix_(selected, selected)].flat) <= indices
    ):
        raise ValueError(
            f"{name} must be a subgroup containing identity and closed under products"
        )
    return indices


def _extend_subgroup(
    table: NDArrayInt, subgroup: SubgroupIndices, generator: GroupElementIndex
) -> SubgroupIndices:
    """Close a finite subgroup together with one added generator under products."""
    generators = [*sorted(subgroup), generator]
    reached = set(subgroup)
    pending = list(subgroup)
    while pending:
        element = pending.pop()
        for product in table[element, generators]:
            product = int(product)
            if product not in reached:
                reached.add(product)
                pending.append(product)
    return frozenset(reached)


def enumerate_subgroups(
    table: NDArrayInt,
    *,
    containing: Collection[GroupElementIndex] | None = None,
    contained_in: Collection[GroupElementIndex] | None = None,
) -> list[SubgroupIndices]:
    """Enumerate every subgroup between inclusive bounds in a finite group.

    Parameters
    ----------
    table : array, shape (order, order)
        Cayley table of a finite group: ``table[i, j]`` is the index of the
        product of elements ``i`` and ``j``. A valid group table is required;
        group axioms are not checked. The identity need not have index zero.
    containing : collection of int, optional
        Lower subgroup bound. Defaults to the identity subgroup.
    contained_in : collection of int, optional
        Upper subgroup bound. Defaults to the whole group.

    Returns
    -------
    list of frozenset of int
        All subgroups between the bounds, ordered by size and then sorted
        indices. Indices retain the input table's numbering. Conjugate
        subgroups remain separate. Inputs are not modified.

    Raises
    ------
    ValueError
        If either bound is not a subgroup of the supplied group, contains
        invalid indices, or ``containing`` is not a subset of ``contained_in``.

    Notes
    -----
    The search starts at the lower bound and stays inside the upper bound;
    neither bound needs to be normal. Every proper inclusion can be reached
    by adjoining an element and closing under products. Elements in the same
    right coset of the current subgroup generate the same extension, so only
    one per coset is needed.
    """
    lower = _subgroup_bound(
        (get_identity_index(table),) if containing is None else containing, table, "containing"
    )
    upper = _subgroup_bound(
        range(len(table)) if contained_in is None else contained_in, table, "contained_in"
    )
    if not lower <= upper:
        raise ValueError("containing must be a subset of contained_in")

    found = {lower}
    pending = [lower]
    while pending:
        subgroup = pending.pop()
        remaining = set(upper - subgroup)
        while remaining:
            generator = min(remaining)
            remaining.difference_update(int(table[h, generator]) for h in subgroup)
            extension = _extend_subgroup(table, subgroup, generator)
            if extension not in found:
                found.add(extension)
                pending.append(extension)
    return sorted(found, key=lambda subgroup: (len(subgroup), tuple(sorted(subgroup))))
