from __future__ import annotations

import numpy as np
from moyopy import Cell, MoyoDataset

from spinforge.space_group import (
    FamilySpaceSubgroup,
    FamilySpaceSubgroupEnumerator,
    Sublattice,
)


def _parent_cell(basis: list[list[float]]) -> tuple[Cell, MoyoDataset]:
    cell = Cell(
        basis,
        [[0.0, 0.0, 0.0]],
        [25],
    )
    return cell, MoyoDataset(cell, rotate_basis=False)


def _triclinic_parent() -> tuple[Cell, MoyoDataset]:
    return _parent_cell([[2.1, 0.0, 0.0], [0.2, 2.7, 0.0], [0.3, 0.4, 3.2]])


def _cubic_parent() -> tuple[Cell, MoyoDataset]:
    return _parent_cell(np.eye(3).tolist())


def _tetragonal_parent() -> tuple[Cell, MoyoDataset]:
    return _parent_cell(np.diag([1.0, 1.0, 2.0]).tolist())


def _enumerator(
    *,
    sublattice: Sublattice | None = None,
    parent: tuple[Cell, MoyoDataset] | None = None,
) -> FamilySpaceSubgroupEnumerator:
    cell, dataset = parent or _triclinic_parent()
    rotations = np.asarray(dataset.operations.rotations, dtype=np.int64)
    translations = np.asarray(dataset.operations.translations, dtype=float)
    epsilon = dataset.symprec / np.abs(np.linalg.det(np.asarray(cell.basis))) ** (1 / 3)
    return FamilySpaceSubgroupEnumerator(
        rotations,
        translations,
        epsilon=epsilon,
        target_sublattice=sublattice,
        atol=1e-5,
    )


def _translation_coset_subgroup() -> tuple[FamilySpaceSubgroupEnumerator, FamilySpaceSubgroup]:
    target = Sublattice(np.diag([2, 1, 1]))
    enumerator = _enumerator(sublattice=target)
    subgroup = next(
        subgroup
        for subgroup in enumerator.enumerate(
            k_index=2,
            up_to_parent_conjugacy=True,
            max_depth=None,
        )
        if not subgroup.is_translationengleiche and len(subgroup.hermann_subgroup) == 1
    )
    return enumerator, subgroup


def _rotated_translation_coset_subgroup() -> tuple[
    FamilySpaceSubgroupEnumerator,
    FamilySpaceSubgroup,
]:
    target = Sublattice(np.diag([3, 1, 1]))
    enumerator = _enumerator(sublattice=target, parent=_cubic_parent())
    subgroup = next(
        subgroup
        for subgroup in enumerator.enumerate(
            k_index=3,
            up_to_parent_conjugacy=True,
            max_depth=None,
        )
        if not subgroup.is_translationengleiche
        and len(subgroup.rotations) == 2
        and any(np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in subgroup.rotations)
    )
    return enumerator, subgroup
