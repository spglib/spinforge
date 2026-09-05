from __future__ import annotations

import numpy as np
import pytest
from spgrep.utils import NDArrayFloat, NDArrayInt

from spinforge.configuration import (
    get_commensurate_sublattice,
)
from spinforge.configuration._propagation import _get_commensurate_sublattice_from_qpoint


@pytest.mark.parametrize(
    "qpoint,expected",
    [
        pytest.param(
            np.array([0.0, 0.0, 0.0]),
            np.diag([1, 1, 1]),
            id="(000)",
        ),
        pytest.param(
            np.array([0.5, 0.0, 0.0]),
            np.diag([2, 1, 1]),
            id="(100)/2",
        ),
        pytest.param(
            np.array([0.0, 0.5, 0.0]),
            np.diag([1, 2, 1]),
            id="(010)/2",
        ),
        pytest.param(
            np.array([0.0, 0.0, 0.5]),
            np.diag([1, 1, 2]),
            id="(001)/2",
        ),
        pytest.param(
            np.array([0.0, 0.5, 0.5]),
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 2],
                ]
            ),
            id="(011)/2",
        ),
        pytest.param(
            np.array([0.5, 0.0, 0.5]),
            np.array(
                [
                    [1, 0, 1],
                    [0, 1, 0],
                    [0, 0, 2],
                ]
            ),
            id="(101)/2",
        ),
        pytest.param(
            np.array([0.5, 0.5, 0.0]),
            np.array(
                [
                    [1, 1, 0],
                    [0, 2, 0],
                    [0, 0, 1],
                ]
            ),
            id="(110)/2",
        ),
        pytest.param(
            np.array([0.5, 0.5, 0.5]),
            np.array(
                [
                    [1, 0, 1],
                    [0, 1, 1],
                    [0, 0, 2],
                ]
            ),
            id="(111)/2",
        ),
        pytest.param(
            np.array([1 / 3, 0.0, 0.0]),
            np.diag([3, 1, 1]),
            id="(100)/3",
        ),
        pytest.param(
            np.array([2 / 3, 0.0, 0.0]),
            np.diag([3, 1, 1]),
            id="(200)/3",
        ),
        pytest.param(
            np.array([1 / 3, 1 / 3, 0.0]),
            np.array(
                [
                    [1, 2, 0],
                    [0, 3, 0],
                    [0, 0, 1],
                ]
            ),
            id="(110)/3",
        ),
        pytest.param(
            np.array([2 / 3, 2 / 3, 0.0]),
            np.array(
                [
                    [1, 2, 0],
                    [0, 3, 0],
                    [0, 0, 1],
                ]
            ),
            id="(220)/3",
        ),
        pytest.param(
            np.array([1 / 3, 2 / 3, 0.0]),
            np.array(
                [
                    [1, 1, 0],
                    [0, 3, 0],
                    [0, 0, 1],
                ]
            ),
            id="(120)/3",
        ),
        pytest.param(
            np.array([1 / 3, 1 / 3, 1 / 3]),
            np.array(
                [
                    [1, 0, 2],
                    [0, 1, 2],
                    [0, 0, 3],
                ]
            ),
            id="(111)/3",
        ),
        pytest.param(
            np.array([1 / 3, 1 / 3, 2 / 3]),
            np.array(
                [
                    [1, 0, 1],
                    [0, 1, 1],
                    [0, 0, 3],
                ]
            ),
            id="(112)/3",
        ),
        pytest.param(
            np.array([1 / 2, 1 / 3, 2 / 3]),
            np.array(
                [
                    [2, 0, 0],
                    [0, 1, 1],
                    [0, 0, 3],
                ]
            ),
            id="(324)/6",
        ),
    ],
)
def test_get_commensurate_sublattice_from_qpoint(qpoint: NDArrayFloat, expected: NDArrayInt):
    transformation = _get_commensurate_sublattice_from_qpoint(np.array(qpoint))
    assert np.array_equal(transformation, expected)


@pytest.mark.parametrize(
    "qpoints,expected",
    [
        pytest.param(
            [
                np.array([0.5, 0.0, 0.0]),
                np.array([0.0, 0.5, 0.0]),
                np.array([0.0, 0.0, 0.5]),
            ],
            np.diag([2, 2, 2]),
            id="(100)",
        ),
        pytest.param(
            [
                np.array([0.0, 0.5, 0.5]),
                np.array([0.5, 0.0, 0.5]),
                np.array([0.5, 0.5, 0.0]),
            ],
            np.array(
                [
                    [1, 1, 1],
                    [0, 2, 0],
                    [0, 0, 2],
                ]
            ),
            id="(110)/2",
        ),
        pytest.param(
            [
                np.array([0.5, 0.5, 0.5]),
            ],
            np.array(
                [
                    [1, 0, 1],
                    [0, 1, 1],
                    [0, 0, 2],
                ]
            ),
            id="(111)/2",
        ),
    ],
)
def test_get_commensurate_sublattice(qpoints: list[np.ndarray], expected: np.ndarray):
    transformation = get_commensurate_sublattice(qpoints)
    assert np.array_equal(transformation, expected)
