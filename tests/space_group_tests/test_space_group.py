from __future__ import annotations

import numpy as np
import pytest
from moyopy import SpaceGroup, operations_from_number
from spgrep.symmetry.group import get_cayley_table

from spinforge.space_group import (
    NormalSpaceSubgroupEnumerator,
    Sublattice,
    enumerate_normal_groups,
)


def test_enumerate_normal_space_subgroup_28():
    operations = operations_from_number(28)  # Pma2
    prim_rotations = np.array(operations.rotations)
    prim_translations = np.array(operations.translations)
    table = get_cayley_table(prim_rotations)

    enumerator = NormalSpaceSubgroupEnumerator(
        prim_rotations=prim_rotations,
        prim_translations=prim_translations,
        table=table,
    )
    list_normal_space_subgroups_1 = enumerator.enumerate(
        normal_point_subgroup=[0, 1, 2, 3],  # t_index=1
        k_index=1,
    )
    # Trivial case
    assert len(list_normal_space_subgroups_1) == 1

    list_normal_space_subgroups_2 = enumerator.enumerate(
        normal_point_subgroup=[0, 1, 2, 3],  # t_index=1
        k_index=2,
    )
    # Ref: ITA1 maximal subgroups of space group 28
    assert len(list_normal_space_subgroups_2) == 12


@pytest.mark.parametrize(
    "sublattice,expected",
    [
        (Sublattice(np.diag([1, 1, 2])), 4),
        (Sublattice(np.diag([1, 2, 1])), 4),
        (Sublattice(np.array([[1, 0, 0], [0, 1, 0], [0, 1, 2]])), 4),
    ],
)
def test_enumerate_normal_space_subgroup_28_with_sublattice(sublattice: Sublattice, expected: int):
    operations = operations_from_number(28)  # Pma2
    prim_rotations = np.array(operations.rotations)
    prim_translations = np.array(operations.translations)
    table = get_cayley_table(prim_rotations)

    enumerator = NormalSpaceSubgroupEnumerator(
        prim_rotations=prim_rotations,
        prim_translations=prim_translations,
        table=table,
        sublattice=sublattice,
    )
    list_normal_space_subgroups = enumerator.enumerate(
        normal_point_subgroup=[0, 1, 2, 3],  # t_index=1
        k_index=2,
    )
    assert len(list_normal_space_subgroups) == expected


def test_explicit_sublattice_preserves_all_affine_embeddings_and_quotients():
    operations = operations_from_number(28)  # Pma2
    prim_rotations = np.array(operations.rotations)
    prim_translations = np.array(operations.translations)
    sublattice = Sublattice(np.diag([1, 1, 2]))
    enumerator = NormalSpaceSubgroupEnumerator(
        prim_rotations=prim_rotations,
        prim_translations=prim_translations,
        table=get_cayley_table(prim_rotations),
        sublattice=sublattice,
    )

    normal_space_subgroups = enumerator.enumerate(
        normal_point_subgroup=[0, 1, 2, 3],
        k_index=2,
    )

    expected_translation_keys = {
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 1.0), (0.5, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (0.5, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (0.5, 0.0, 0.0)),
    }
    assert len(normal_space_subgroups) == len(expected_translation_keys)
    subgroups_by_translation = {
        tuple(tuple(row) for row in subgroup.translations.tolist()): subgroup
        for subgroup in normal_space_subgroups
    }
    assert set(subgroups_by_translation) == expected_translation_keys
    for subgroup in subgroups_by_translation.values():
        assert subgroup.point_subgroup == [0, 1, 2, 3]
        assert subgroup.sublattice == sublattice
        np.testing.assert_array_equal(subgroup.quotient_table, [[0, 1], [1, 0]])


def test_enumerate_normal_space_subgroup_194():
    operations = operations_from_number(194)  # P6_3/mmc
    prim_rotations = np.array(operations.rotations)
    prim_translations = np.array(operations.translations)
    table = get_cayley_table(prim_rotations)

    enumerator = NormalSpaceSubgroupEnumerator(
        prim_rotations=prim_rotations,
        prim_translations=prim_translations,
        table=table,
    )
    list_normal_space_subgroups = enumerator.enumerate(
        normal_point_subgroup=list(range(len(prim_rotations))),  # t_index=1
        k_index=1,
    )

    # Trivial case
    assert len(list_normal_space_subgroups) == 1


@pytest.mark.parametrize(
    "number,k_index,normal_point_subgroup,expected_subgroups",
    [
        (176, 3, [0, 2, 4, 6, 8, 10], 173),  #  P 6_3/m -> P 6_3
        (191, 3, [0, 2, 6, 12, 15, 20], 168),  # P 6/m m m -> P 6
        (192, 3, [0, 2, 6, 12, 15, 20], 168),  # P 6/m c c -> P 6
        (193, 3, [0, 2, 6, 12, 15, 20], 173),  # P 6_3/m c m -> P 6_3
        (194, 3, [0, 2, 6, 12, 15, 20], 173),  # P 6_3/m m c -> P 6_3
        (194, 3, [0, 6, 10, 11, 20, 23], 156),  # P 6_3/m m c -> P 3 m 1
    ],
)
def test_enumerate_normal_space_subgroup(
    number: int,
    k_index: int,
    normal_point_subgroup: list[int],
    expected_subgroups: int,
):
    operations = operations_from_number(number)
    prim_rotations = np.array(operations.rotations)
    prim_translations = np.array(operations.translations)
    table = get_cayley_table(prim_rotations)

    enumerator = NormalSpaceSubgroupEnumerator(
        prim_rotations=prim_rotations,
        prim_translations=prim_translations,
        table=table,
    )
    list_normal_space_subgroups = enumerator.enumerate(
        normal_point_subgroup=normal_point_subgroup,
        k_index=k_index,
    )
    assert len(list_normal_space_subgroups) > 0

    ok = False
    for nss in list_normal_space_subgroups:
        space_group = SpaceGroup(
            prim_rotations=prim_rotations[nss.point_subgroup].tolist(),
            prim_translations=nss.translations.tolist(),
        )
        if space_group.number == expected_subgroups:
            ok = True
    assert ok


@pytest.mark.parametrize("k_index", [1, 2])
def test_enumerate_normal_space_subgroup_48(k_index: int):
    prim_operations = operations_from_number(number=48, primitive=True)
    prim_rotations = np.array(prim_operations.rotations)
    table = get_cayley_table(prim_rotations)
    enumerator = NormalSpaceSubgroupEnumerator(
        prim_rotations=prim_rotations,
        prim_translations=np.array(prim_operations.translations),
        table=table,
    )
    for normal_point_subgroup in enumerate_normal_groups(table):
        _list_normal_space_subgroups = enumerator.enumerate(
            normal_point_subgroup=normal_point_subgroup,
            k_index=k_index,
        )
