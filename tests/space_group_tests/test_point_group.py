from collections import Counter

import numpy as np
from moyopy import operations_from_number
from spgrep.symmetry.group import get_cayley_table

from spinforge.space_group import enumerate_normal_groups


def test_enumerate_normal_point_groups_hexagonal():
    operations = operations_from_number(191)  # P6/mmm
    table = get_cayley_table(np.array(operations.rotations))

    list_normal_subgroups = enumerate_normal_groups(table)

    counter = Counter([len(point_group) for point_group in list_normal_subgroups])
    assert counter[24] == 1  # 6/mmm
    assert counter[12] == 7  # 622, -62m, -6m2, 6mm, 6/m, -3m1, -31m
    assert counter[6] == 7  # 6, -6, 321, 312, 31m, 3m1, -3
    assert counter[3] == 1  # 3
    assert counter[1] == 1  # 1


def test_enumerate_normal_point_groups_cubic():
    operations = operations_from_number(221)  # Pm-3m
    table = get_cayley_table(np.array(operations.rotations))

    list_normal_subgroups = enumerate_normal_groups(table)

    counter = Counter([len(point_group) for point_group in list_normal_subgroups])
    assert counter[48] == 1  # m-3m
    assert counter[24] == 3  # 432, -43m, m-3
    assert counter[12] == 1  # 23
    assert counter[8] == 1  # mmm
    assert counter[4] == 1  # 222
    assert counter[2] == 1  # -1
    assert counter[1] == 1  # 1
