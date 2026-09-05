from __future__ import annotations

import numpy as np
from spgrep.utils import NDArrayFloat, NDArrayInt

from spinforge.utils.rotation_utils import find_rotation_conjugator, get_rotation_axis


def _align_cart_rotations(
    cart_rotations: NDArrayFloat,
    src_axis: NDArrayFloat,
    dst_axis: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Align Cartesian rotations so ``src_axis`` maps onto ``dst_axis``."""
    conjugator = find_rotation_conjugator(src_axis=src_axis, dst_axis=dst_axis)
    aligned = np.array([conjugator @ rotation @ conjugator.T for rotation in cart_rotations])
    return conjugator, aligned


def get_rotation_axes(
    prim_cart_rotations: NDArrayFloat,
    atol: float = 1e-5,
) -> list[NDArrayFloat]:
    cart_rotation_axes: list[NDArrayFloat] = []
    for cart_rotation in prim_cart_rotations:
        axis = get_rotation_axis(cart_rotation)
        assert axis is not False
        if axis is True:
            continue

        unique = True
        for other in cart_rotation_axes:
            if np.allclose(np.cross(other, axis), 0, atol=atol):
                unique = False
                break
        if unique:
            cart_rotation_axes.append(axis)
    return cart_rotation_axes


def merge_rotation_axes(
    axes_a: list[NDArrayFloat],
    axes_b: list[NDArrayFloat],
    atol: float = 1e-5,
) -> list[NDArrayFloat]:
    """Return the union of two axis lists, deduplicating parallel axes."""
    merged: list[NDArrayFloat] = list(axes_a)
    for axis in axes_b:
        if any(np.allclose(np.cross(other, axis), 0, atol=atol) for other in merged):
            continue
        merged.append(axis)
    return merged


def collect_cart_rotation_axes(
    cart_rotations: NDArrayFloat,
    extra_cart_rotations: NDArrayFloat | None,
    *,
    atol: float,
) -> list[NDArrayFloat]:
    """Collect candidate Cartesian rotation axes, optionally merged with extra rotations."""
    cart_rotation_axes = get_rotation_axes(cart_rotations, atol=atol)
    if extra_cart_rotations is not None:
        cart_rotation_axes = merge_rotation_axes(
            cart_rotation_axes,
            get_rotation_axes(extra_cart_rotations, atol=atol),
            atol=atol,
        )
    return cart_rotation_axes


def determinant_signs(matrices: NDArrayFloat) -> NDArrayInt:
    """Return the sign (+1 or -1) of the determinant of each matrix."""
    return np.around(np.linalg.det(matrices)).astype(int)
