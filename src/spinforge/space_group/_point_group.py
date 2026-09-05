from __future__ import annotations

import numpy as np
from spgrep.symmetry.pointgroup import pg_dataset
from spgrep.symmetry.subgroup import enumerate_point_subgroup
from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.pointgroup import POINT_GROUP_REPRESENTATIVES

from spinforge.utils.group_theory import GroupElementIndex, all_inverses
from spinforge.utils.rotation_utils import to_cartesian_rotations


def enumerate_normal_groups(table: NDArrayInt) -> list[list[GroupElementIndex]]:
    """Enumerate normal point groups up to conjugacy classes."""
    order = len(table)
    inverses = all_inverses(table)
    subgroups = enumerate_point_subgroup(table, [True] * order, return_conjugacy_class=True)

    # Check if the subgroup is normal
    normal_subgroups = []
    for subgroup in subgroups:
        is_normal = True
        subgroup_set = frozenset(subgroup)

        for g in range(order):
            if not is_normal:
                break
            g_inv = inverses[g]
            for n in subgroup:
                conjugate = table[g_inv, table[n, g]]
                if conjugate not in subgroup_set:
                    is_normal = False
                    break
        if is_normal:
            normal_subgroups.append(sorted(subgroup))

    return normal_subgroups


def get_cartesian_point_group_representative(symbol: str) -> NDArrayFloat:
    if symbol in ["3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm"]:
        # Hexagonal
        lattice = np.array([[1.0, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 1.0]])
    else:
        lattice = np.eye(3)

    rotations = np.array(pg_dataset[symbol][POINT_GROUP_REPRESENTATIVES[symbol]])
    return to_cartesian_rotations(lattice, rotations)
