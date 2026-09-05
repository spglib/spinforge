"""Value formatting for spinCIF tags: numbers, xyzt strings, uvw expressions,
and symmetry-restricted moment forms."""

from __future__ import annotations

import numpy as np
from pymatgen.core.operations import MagSymmOp
from spgrep.utils import NDArrayFloat
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType


def _fmt(value: float, *, decimals: int = 6) -> str:
    """Format with fixed decimals, avoiding a ``-0.000000`` output.

    The zero-flush threshold is half a unit in the last printed place, i.e.
    exactly the rounding boundary of the format itself, so no tolerance beyond
    the output precision is introduced.
    """
    v = 0.0 if abs(value) < 0.5 * 10.0**-decimals else float(value)
    return f"{v:.{decimals}f}"


def _spinframe_p_abc(frame: NDArrayFloat, lattice: NDArrayFloat, *, atol: float) -> str:
    """The spin basis declared as linear combinations of the lattice vectors.

    Declares the orthonormal spin frame through
    ``_space_group_spin.transform_spinframe_P_abc`` (the tag every COMCIFS
    reference file uses; readers tuned to those files may not implement the
    equivalent ``spinframe_orientation_cartn`` route). The basis itself is
    unchanged: each expression reproduces one unit vector of the frame.
    """
    inv_lattice = np.linalg.inv(lattice)
    expressions = [
        _linear_expr(frame[:, i] @ inv_lattice, ("a", "b", "c"), atol=atol) for i in range(3)
    ]
    return "'" + ",".join(expressions) + "'"


def _collinear_direction_xyz(
    spin_only_group: SpinOnlyGroup, q_mat: NDArrayFloat, lattice: NDArrayFloat
) -> str:
    """``_space_group_spin.collinear_direction_xyz`` value, "." if not collinear.

    ``spin_only_group.axis`` is the collinear moment direction (Cartesian, in
    the enumeration frame); ``q_mat`` is the rotation already applied to the
    written moments. Per the spec, the direction is "specified with respect
    to the lattice unit cell" (not the spin basis): components (x, y, z) of
    m = x a + y b + z c.
    """
    if spin_only_group.spin_only_group_type != SpinOnlyGroupType.COLLINEAR:
        return "."
    axis = q_mat @ np.asarray(spin_only_group.axis, dtype=float)
    direction = axis @ np.linalg.inv(lattice)
    direction = direction / np.max(np.abs(direction))
    return "'" + ",".join(_fmt(x) for x in direction) + "'"


def _coplanar_perp_uvw(
    spin_only_group: SpinOnlyGroup, q_mat: NDArrayFloat, frame: NDArrayFloat
) -> str:
    """``_space_group_spin.coplanar_perp_uvw`` value, "." if not coplanar.

    ``spin_only_group.axis`` is the spin-plane normal (Cartesian, in the
    enumeration frame); the tag is expressed in the spin basis ``frame``.
    """
    if spin_only_group.spin_only_group_type != SpinOnlyGroupType.COPLANAR:
        return "."
    normal = frame.T @ q_mat @ np.asarray(spin_only_group.axis, dtype=float)
    return "'" + ",".join(_fmt(x) for x in normal) + "'"


def _xyzt(w: np.ndarray, t: np.ndarray, u: np.ndarray) -> str:
    time_reversal = -1 if np.linalg.det(u) < 0 else 1
    op = MagSymmOp.from_rotation_and_translation_and_time_reversal(
        rotation_matrix=np.array(w),
        translation_vec=np.remainder(t, 1.0),
        time_reversal=time_reversal,
    )
    # CIF loop columns are whitespace-delimited; xyzt must be one token
    return op.as_xyzt_str().replace(" ", "")


# Exact coefficient strings for common rational values; anything else
# (e.g. sqrt(3)/2) is written as a plain 15-significant-digit number to
# keep the expression grammar free of math functions.
_EXACT_COEFFICIENTS = [
    (1.0, "1"),
    (1.0 / 2, "1/2"),
    (1.0 / 3, "1/3"),
    (2.0 / 3, "2/3"),
    (1.0 / 4, "1/4"),
    (3.0 / 4, "3/4"),
    (1.0 / 6, "1/6"),
    (5.0 / 6, "5/6"),
]


def _coefficient_str(value: float, *, atol: float) -> str:
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for ref, text in _EXACT_COEFFICIENTS:
        if np.isclose(magnitude, ref, rtol=0, atol=atol):
            return sign + text
    return f"{value:.15g}"


def _linear_expr(coefficients: NDArrayFloat, variables: tuple[str, ...], *, atol: float) -> str:
    """Format one row of a linear map as e.g. ``-1/2u+0.866025403784439v``."""
    terms = []
    for value, var in zip(coefficients, variables):
        if abs(value) <= atol:
            continue
        text = _coefficient_str(float(value), atol=atol)
        if text == "1":
            term = var
        elif text == "-1":
            term = "-" + var
        else:
            term = text + var
        terms.append(term)
    if not terms:
        return "0"
    expr = terms[0]
    for term in terms[1:]:
        expr += term if term.startswith("-") else "+" + term
    return expr


def _uvw_expr(u: np.ndarray, *, atol: float) -> str:
    return ",".join(_linear_expr(row, ("u", "v", "w"), atol=atol) for row in u)


def _symmform(subspace: list[np.ndarray], *, atol: float) -> str:
    """Symmetry-restricted moment form at a site, from its allowed subspace.

    ``subspace`` holds the values of each SSA basis vector at the site (spin
    frame). Row-reducing them yields the free parameters: pivot columns get
    the letters u/v/w, and each moment component is a linear expression in
    them (e.g. ``u,2u,0``).
    """
    rows = [v for v in subspace if np.linalg.norm(v) > atol]
    if not rows:
        return "0,0,0"
    reduced, pivots = _rref(np.array(rows), atol=atol)
    letters = tuple("uvw"[col] for col in pivots)
    return ",".join(_linear_expr(reduced[:, col], letters, atol=atol) for col in range(3))


def _rref(matrix: NDArrayFloat, *, atol: float) -> tuple[NDArrayFloat, list[int]]:
    """Reduced row echelon form with tolerance; returns (rows, pivot columns)."""
    m = matrix.astype(float).copy()
    pivots = []
    row = 0
    for col in range(m.shape[1]):
        if row >= len(m):
            break
        pivot = row + int(np.argmax(np.abs(m[row:, col])))
        if abs(m[pivot, col]) <= atol:
            continue
        m[[row, pivot]] = m[[pivot, row]]
        m[row] = m[row] / m[row, col]
        for other in range(len(m)):
            if other != row:
                m[other] = m[other] - m[other, col] * m[row]
        pivots.append(col)
        row += 1
    return m[: len(pivots)], pivots
