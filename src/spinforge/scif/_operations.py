"""Geometry and group operations behind the spinCIF writer.

Composes the SSG coset representatives and transforms them from the
primitive input cell to the supercell frame. Site matching lives on
:meth:`spinforge.configuration.Supercell.site_permutations`; correctness of
the written operations against the moments is checked by the round-trip
tests, not at write time.
"""

from __future__ import annotations

import numpy as np
from spgrep.utils import NDArrayFloat

from spinforge.configuration import SpinSymmetryAdaptedStructure

_FLOAT_NOISE_ATOL = float(np.sqrt(np.finfo(float).eps))
"""Tolerance for pure floating-point roundoff in exact group data.

The supercell transforms below consume exact rationals (group rotations and
translations) carrying only accumulated roundoff, so their integrality,
identity, and grid-snapping checks use this machine-precision-derived bound
(~1.5e-8) rather than the writer's user-facing ``atol``: relaxing the
spin-space tolerance must not relax the structural checks.
"""


def _crystal_cartesian_frame(lattice: NDArrayFloat) -> NDArrayFloat:
    """Orthonormal frame with x || a and z || c*, as columns (Cartesian)."""
    e1 = lattice[0] / np.linalg.norm(lattice[0])
    e3 = np.cross(lattice[0], lattice[1])
    e3 = e3 / np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return np.column_stack([e1, e2, e3])


def _nontrivial_operations(
    nontrivial_coset,
    invariant_rotations,
    invariant_translations,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Primitive-frame coset representatives ``(W, t, U)`` of the SSG over
    its spin translation group: the nontrivial coset composed with the
    invariant space-subgroup representatives."""
    ops = []
    for r1, t1, u1 in zip(
        nontrivial_coset.rotations,
        nontrivial_coset.translations,
        nontrivial_coset.spin_rotations,
    ):
        for r3, t3 in zip(invariant_rotations, invariant_translations):
            w = np.array(r1, dtype=float) @ np.array(r3, dtype=float)
            t = np.array(r1, dtype=float) @ np.array(t3, dtype=float) + np.array(t1, dtype=float)
            ops.append((w, t, np.array(u1, dtype=float)))
    return ops


def _to_supercell_frame(
    sas: SpinSymmetryAdaptedStructure,
    operations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Transform primitive-frame operations ``(W, t, U)`` to the supercell frame.

    A primitive-frame operation ``(W, t)`` maps to ``(T^-1 W T, T^-1 t)``
    with ``T`` the sublattice transformation; the spin part is untouched
    (it lives in Cartesian spin space).
    """
    t_mat = np.array(sas.supercell.sublattice.transformation, dtype=float)
    t_inv = np.linalg.inv(t_mat)
    denom = _supercell_translation_denominator(t_mat)
    ops = []
    for w, t, u in operations:
        w_s = t_inv @ w @ t_mat
        if not np.allclose(w_s, np.rint(w_s), rtol=0, atol=_FLOAT_NOISE_ATOL):
            raise ValueError(f"Rotation not integer in supercell frame: {w_s}")
        w_s_int = np.rint(w_s).astype(int)
        if round(abs(np.linalg.det(w_s_int))) != 1:
            raise ValueError(f"Rotation must be unimodular in the supercell frame: {w_s_int}")
        t_s = _snap_translation(t_inv @ t, denom=denom, atol=_FLOAT_NOISE_ATOL)
        ops.append((w_s_int, t_s, u))
    return ops


def _supercell_spin_translations(
    sas: SpinSymmetryAdaptedStructure, spin_translation_coset
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Spin translation coset in the supercell frame; identity row first."""
    t_mat = np.array(sas.supercell.sublattice.transformation, dtype=float)
    t_inv = np.linalg.inv(t_mat)
    denom = _supercell_translation_denominator(t_mat)
    entries = []
    for t2, u2 in zip(spin_translation_coset.translations, spin_translation_coset.spin_rotations):
        t_s = _snap_translation(
            t_inv @ np.array(t2, dtype=float), denom=denom, atol=_FLOAT_NOISE_ATOL
        )
        entries.append((t_s, np.array(u2, dtype=float)))
    entries.sort(key=lambda e: float(np.linalg.norm(e[0])))
    if np.linalg.norm(entries[0][0]) > _FLOAT_NOISE_ATOL or not np.allclose(
        entries[0][1], np.eye(3), rtol=0, atol=_FLOAT_NOISE_ATOL
    ):
        entries.insert(0, (np.zeros(3), np.eye(3)))
    return entries


def _supercell_translation_denominator(t_mat: NDArrayFloat) -> int:
    """Grid denominator for fractional translations in the supercell frame.

    Primitive-frame crystallographic translations live on the 1/24 grid
    (multiples of 1/2, 1/3, 1/4, 1/6) plus integer lattice vectors. Applying
    ``T^-1 = adj(T) / det(T)`` puts the supercell-frame components on the
    ``1 / (24 |det T|)`` grid, e.g. a primitive translation (0, 0, 1) in a
    diag(1, 1, 5) supercell becomes 1/5 of the supercell c axis.
    """
    return 24 * round(abs(float(np.linalg.det(t_mat))))


def _snap_translation(trans: NDArrayFloat, *, denom: int, atol: float) -> np.ndarray:
    """Snap a fractional translation to the nearest k/denom when within atol.

    ``denom`` must be the exact grid of the frame the translation lives in
    (see :func:`_supercell_translation_denominator`); snapping then only
    removes float noise without altering a genuine translation.
    """
    t = np.asarray(trans, dtype=float) % 1.0
    snapped = np.round(t * denom) / denom
    return np.where(np.abs(snapped - t) <= atol, snapped, t) % 1.0


def _orbit_partition(num_sites: int, permutations: list[np.ndarray]) -> list[tuple[int, int]]:
    """Partition all sites into orbits; return (representative, multiplicity)."""
    remaining = set(range(num_sites))
    orbits = []
    while remaining:
        seed = min(remaining)
        orbit = {int(permutation[seed]) for permutation in permutations}
        orbits.append((seed, len(orbit)))
        remaining -= orbit
    return orbits
