from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Final

import numpy as np
from hsnf import row_style_hermite_normal_form
from hsnf.integer_system import solve_integer_linear_system
from hsnf.lattice import compute_intersection
from spgrep.utils import NDArrayFloat

DEFAULT_MAX_DENOMINATOR: Final[int] = 100


def get_commensurate_sublattice(
    qpoints: list[NDArrayFloat], *, max_denominator: int = DEFAULT_MAX_DENOMINATOR
):
    ret = np.eye(3, dtype=int)
    for qpoint in qpoints:
        transformation = _get_commensurate_sublattice_from_qpoint(
            qpoint, max_denominator=max_denominator
        )
        ret = compute_intersection(ret, transformation, row_wise=True)
    ret, _ = row_style_hermite_normal_form(ret)
    return ret


# Adapted from spgrep-modulation
def _get_commensurate_sublattice_from_qpoint(
    qpoint: NDArrayFloat, *, max_denominator: int = DEFAULT_MAX_DENOMINATOR
):
    """Return transformation matrix of the following sublattice.

    Let ``t`` be a lattice point of the returned sublattice. Then, ``np.dot(t, qpoint)`` is integer.

    Parameters
    ----------
    qpoint: array, (3, )
        Reciprocal point
    max_denominator: int, default=100
        Maximal value to infer denominators of ``qpoint``

    Returns
    -------
    transformation: array, (3, 3)
        ``transformation @ qpoint`` is integer vector.
    """
    qpoint = np.array(qpoint, dtype=float)
    if np.allclose(qpoint, 0):
        # GAMMA point
        return np.diag([1, 1, 1])

    # Basis vectors of a sublattice formed by translation that preserve order parameter
    elements = [Fraction(qi).limit_denominator(max_denominator).denominator for qi in qpoint]
    lcm = _lcm_on_list(elements)
    numerators = np.around(qpoint * lcm).astype(int)
    g = _gcd_on_list(numerators)
    A = np.array([num // g for num in numerators])[None, :]  # Now, GCD(A) = 1

    # Solve `A @ t = lcm`
    # Since GCD of A is 1, this integer linear system always has a special solution.
    basis_and_special = solve_integer_linear_system(A, np.array([lcm]))
    # transformation @ mathbb{Z}^{3} forms sublattice
    transformation, _ = row_style_hermite_normal_form(np.vstack(basis_and_special))

    if np.linalg.det(transformation) < 0:
        transformation[0, :] *= -1

    return transformation


def _gcd_on_list(elements: list[int]) -> int:
    """Return greatest common devisor for list of integers."""
    g = elements[0]
    for e in elements[1:]:
        g = gcd(g, e)
    return g


def _lcm_on_list(elements: list[int]) -> int:
    """Return least common square for list of integers."""
    lcm = elements[0]
    for e in elements[1:]:
        lcm = lcm * e // gcd(lcm, e)
    return lcm
