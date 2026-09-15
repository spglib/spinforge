"""Finite subgroup selections with an unchanged spin assignment."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
from spgrep.rep.group import get_identity_index
from spgrep.utils import NDArrayInt
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from spinforge.utils.group_theory import GroupElementIndex, SubgroupIndices

from ._spin_space_group import NontrivialSpinSpaceGroup


def _fixed_group_table(
    nssg: NontrivialSpinSpaceGroup, spin_only_group: SpinOnlyGroup, atol: float
) -> NDArrayInt:
    """Snapshot the quotient table after checking the assigned spin cosets.

    The spin-only group fixes the active spin subspace pointwise: a line for
    collinear input, a plane for coplanar input, and all of R^3 otherwise.
    Equality on this subspace therefore tests equality modulo the spin-only
    group without enumerating its possibly continuous elements.
    """
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be positive and finite")
    kind = spin_only_group.spin_only_group_type
    if kind not in (
        SpinOnlyGroupType.COLLINEAR,
        SpinOnlyGroupType.COPLANAR,
        SpinOnlyGroupType.NONCOPLANAR,
    ):
        raise ValueError("Fixed-SSG enumeration requires a magnetic spin-only group")

    projector = np.eye(3)
    if kind in (SpinOnlyGroupType.COLLINEAR, SpinOnlyGroupType.COPLANAR):
        axis = np.array(spin_only_group.axis, dtype=float, copy=True)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.any(axis):
            raise ValueError("spin_only_group.axis must be a finite nonzero three-vector")
        axis /= np.max(np.abs(axis))
        axis /= np.linalg.norm(axis)
        projector = np.outer(axis, axis)
        if kind == SpinOnlyGroupType.COPLANAR:
            projector = np.eye(3) - projector

    operations, table = nssg.get_full_operations_and_table()
    spins = operations.spin_rotations
    if spins.shape != (len(table), 3, 3) or not np.isfinite(spins).all():
        raise ValueError("spin_rotations must be finite 3-by-3 matrices")
    if not np.allclose(spins @ spins.transpose(0, 2, 1), np.eye(3), atol=atol, rtol=0):
        raise ValueError("spin_rotations must be orthogonal")
    active_spins = spins @ projector
    if not np.allclose(active_spins @ spins.transpose(0, 2, 1), projector, atol=atol, rtol=0):
        raise ValueError("spin_rotations must normalize the spin-only group")
    for i, spin in enumerate(spins):
        if not np.allclose(spin @ active_spins, active_spins[table[i]], atol=atol, rtol=0):
            raise ValueError(
                f"Spin assignments do not close modulo the spin-only group at operation {i}"
            )
    return table


def _subgroup_bound(
    values: Collection[GroupElementIndex], table: NDArrayInt, name: str
) -> SubgroupIndices:
    """Validate one bound as a subgroup in the original quotient-table order."""
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


def _enumerate_subgroups(
    table: NDArrayInt,
    containing: Collection[GroupElementIndex] | None,
    contained_in: Collection[GroupElementIndex] | None,
) -> list[SubgroupIndices]:
    """Enumerate the subgroup interval by adjoining generators to its lower bound.

    Every proper inclusion can be reached by adding one element and closing.
    Elements in the same right coset of the current subgroup generate the same
    extension, so only one per coset is needed. Bounds need not be normal.
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
