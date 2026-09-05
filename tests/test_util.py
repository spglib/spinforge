from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spgrep.symmetry.group import get_cayley_table
from spgrep.utils import NDArrayFloat
from spinspg.pointgroup import get_pointgroup_representative_from_symbol

from spinforge.space_group import enumerate_normal_groups
from spinforge.utils import (
    combine_dimensions,
    find_rotation_conjugator,
    get_coset_representatives,
)


def test_get_coset_representatives():
    rotations = get_pointgroup_representative_from_symbol("6/mmm")
    table = get_cayley_table(np.array(rotations))
    list_normal_subgroups = enumerate_normal_groups(table)
    for normal_subgroup in list_normal_subgroups:
        representatives, factor_group_table = get_coset_representatives(normal_subgroup, table)
        for row in factor_group_table:
            assert len(set(row)) == len(representatives)


def test_combine_dimensions():
    dimensions = [1, 1, 2, 3]
    combined_with_dims = combine_dimensions(dimensions, max_dim=3)
    assert combined_with_dims == {
        1: [[0], [1]],
        2: [
            [0, 0],
            [0, 1],
            [1, 1],
            [2],
        ],
        3: [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 2],
            [1, 2],
            [3],
        ],
    }


@pytest.mark.parametrize(
    "src_axis,dst_axis",
    [
        (np.array([1.0, 0, 0]), np.array([1.0, 0, 0])),
        (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])),
        (np.array([1.0, 0, 0]), np.array([1.0, 1.0, 0]) / np.sqrt(2)),
    ],
)
def test_find_rotation_conjugator(src_axis: NDArrayFloat, dst_axis: NDArrayFloat):
    angle = np.pi / 6
    src_rotation = Rotation.from_rotvec(angle * src_axis).as_matrix()
    dst_rotation = Rotation.from_rotvec(angle * dst_axis).as_matrix()
    conjugator = find_rotation_conjugator(src_axis, dst_axis)
    assert np.allclose(conjugator @ src_rotation @ np.linalg.inv(conjugator), dst_rotation)
