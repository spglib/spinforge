"""Write magnetic structures as mcif files (BNS/MAGNDATA convention).

Reading needs no local code: ``pymatgen.core.Structure.from_file`` handles
mcif, and its symmetry expansion of ``_atom_site_moment.crystalaxis_*``
components is correct as of pymatgen-core v2026.7.16
(materialsproject/pymatgen-core#76): moments are converted to Cartesian before
applying an operation, so |m| is preserved even when an operation mixes
unequal or non-orthogonal axes (e.g. hexagonal settings; see
materialsproject/pymatgen-core#76 for the old failure mode).

:class:`MCifWriter` has no upstream equivalent (``CifWriter(write_magmoms=True)``
dumps every atom in P1 with no magnetic operations).
"""

from __future__ import annotations

import numpy as np
from moyopy import (
    MagneticSpaceGroupType,
    MoyoNonCollinearMagneticDataset,
    NonCollinearMagneticCell,
)
from pymatgen.core import Structure
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom

from spinforge._cif import _ATOM_SITE_LOOP_HEADER

__all__ = [
    "MCifWriter",
]

_OPERATION_TAG = "_space_group_symop_magn_operation.xyz"
_CENTERING_TAG = "_space_group_symop_magn_centering.xyz"


class MCifWriter:
    """A pymatgen-style writer for *symmetrized* magnetic CIF (BNS/MAGNDATA).

    Mirrors :class:`pymatgen.io.cif.CifWriter`: ``str(writer)`` returns the mcif
    text and ``writer.write_file(path)`` writes it. Unlike
    ``CifWriter(write_magmoms=True)`` -- which dumps every atom in P1 with no
    magnetic operations -- this finds the magnetic space group of ``struct`` with
    moyopy and writes the asymmetric unit plus the
    ``_space_group_symop_magn_operation.xyz`` /
    ``_space_group_symop_magn_centering.xyz`` loops, so the file round-trips
    through ``pymatgen.core.Structure.from_file``.

    The structure is symmetrized in its own input setting (the basis is not
    standardized). ``struct`` must carry a ``magmom`` site property (pymatgen
    ``Magmom`` or a Cartesian 3-vector per site). Moment components are written as
    ``_atom_site_moment.crystalaxis_*`` in the unit-axis convention read back by
    ``CifParser`` (``m = mx*a_hat + my*b_hat + mz*c_hat``).

    Partial occupancy is preserved: each site's fractional occupancy enters the
    moyopy equivalence label (so a 0.79-occupied site is never folded into a fully
    occupied orbit) and is written as an ``_atom_site_occupancy`` column. Only
    single-species sites are supported; a mixed solid-solution site (two species
    sharing one position) raises ``NotImplementedError``, since moyopy cannot
    symmetrize it.

    Parameters
    ----------
    struct
        Magnetic structure with a ``magmom`` site property.
    symprec, mag_symprec
        moyopy symmetry-search tolerances (basis units / moment units).
    """

    def __init__(
        self,
        struct: Structure,
        symprec: float = 1e-4,
        *,
        mag_symprec: float | None = None,
    ) -> None:
        if "magmom" not in struct.site_properties:
            raise ValueError("structure must have a 'magmom' site property to write an mcif")
        basis = struct.lattice.matrix
        moments = np.array([Magmom(m).global_moment for m in struct.site_properties["magmom"]])
        cell = NonCollinearMagneticCell(
            basis=basis.tolist(),
            positions=struct.frac_coords.tolist(),
            numbers=self._composition_numbers(struct),
            magnetic_moments=moments.tolist(),
        )
        dataset = MoyoNonCollinearMagneticDataset(
            cell, symprec=symprec, mag_symprec=mag_symprec, is_axial=True, rotate_basis=False
        )
        ops = dataset.magnetic_operations
        op_strings = [
            MagSymmOp.from_rotation_and_translation_and_time_reversal(
                rotation_matrix=np.array(rot),
                translation_vec=self._snap_translation(trans),
                time_reversal=-1 if rev else 1,
            )
            .as_xyzt_str()
            .replace(" ", "")  # CIF loop columns are whitespace-delimited; xyzt must be one token
            for rot, trans, rev in zip(ops.rotations, ops.translations, ops.time_reversals)
        ]

        reps = sorted(set(dataset.orbits))  # one representative atom index per orbit
        unit_axes = basis / np.linalg.norm(basis, axis=1)[:, None]
        inv_unit_axes = np.linalg.inv(unit_axes)  # crystalaxis comps = m_cart @ inv(unit_axes)

        bns = MagneticSpaceGroupType(dataset.uni_number).bns_number
        self._mcif_str = self._format_mcif(struct, reps, op_strings, moments, inv_unit_axes, bns)

    def __str__(self) -> str:
        """The symmetrized mcif as a string."""
        return self._mcif_str

    def write_file(self, filename: str, mode: str = "wt") -> None:
        """Write the symmetrized mcif to ``filename``."""
        with open(filename, mode) as f:
            f.write(self._mcif_str)

    @staticmethod
    def _format_mcif(
        structure: Structure,
        reps: list[int],
        op_strings: list[str],
        moments: np.ndarray,
        inv_unit_axes: np.ndarray,
        bns: str,
        *,
        atol: float = 1e-6,
    ) -> str:
        """Assemble symmetrized-mcif text from the asymmetric unit and magnetic operations."""
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        lines = [
            "# Symmetrized magnetic CIF written by spinforge.mcif.MCifWriter",
            "data_magnetic_structure",
            f"_space_group_magn.number_BNS {bns}",
            f"_cell_length_a {a:.6f}",
            f"_cell_length_b {b:.6f}",
            f"_cell_length_c {c:.6f}",
            f"_cell_angle_alpha {alpha:.6f}",
            f"_cell_angle_beta {beta:.6f}",
            f"_cell_angle_gamma {gamma:.6f}",
            "",
            "loop_",
            "_space_group_symop_magn_operation.id",
            f"{_OPERATION_TAG}",
        ]
        lines += [f"{i + 1} {s}" for i, s in enumerate(op_strings)]
        lines += [
            "",
            "loop_",
            "_space_group_symop_magn_centering.id",
            f"{_CENTERING_TAG}",
            "1 x,y,z,+1",
            *_ATOM_SITE_LOOP_HEADER,
        ]
        labels: dict[int, str] = {}
        counts: dict[str, int] = {}
        for idx in reps:
            symbol, occ = MCifWriter._single_species(structure[idx])
            counts[symbol] = counts.get(symbol, 0) + 1
            label = f"{symbol}{counts[symbol]}"
            labels[idx] = label
            fx, fy, fz = structure[idx].frac_coords
            lines.append(f"{label} {symbol} {fx:.6f} {fy:.6f} {fz:.6f} {occ:.6f}")

        moment_lines = []
        for idx in reps:
            comps = moments[idx] @ inv_unit_axes
            if np.linalg.norm(comps) <= atol:
                continue
            moment_lines.append(f"{labels[idx]} {comps[0]:.6f} {comps[1]:.6f} {comps[2]:.6f}")
        if moment_lines:
            lines += [
                "",
                "loop_",
                "_atom_site_moment.label",
                "_atom_site_moment.crystalaxis_x",
                "_atom_site_moment.crystalaxis_y",
                "_atom_site_moment.crystalaxis_z",
            ]
            lines += moment_lines
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _composition_numbers(structure: Structure) -> list[int]:
        """Equivalence labels for moyopy that fold occupancy into the site identity.

        moyopy treats ``numbers`` as opaque per-site equivalence classes. Encoding the
        full ``(Z, occupancy)`` signature means a magnetic operation may only map a site
        onto another with the *same* species and occupancy, so partially occupied sites
        are never merged into a fully occupied orbit.
        """
        signatures: dict[tuple, int] = {}
        numbers = []
        for site in structure:
            signature = tuple(sorted((sp.Z, round(occ, 6)) for sp, occ in site.species.items()))
            numbers.append(signatures.setdefault(signature, len(signatures) + 1))
        return numbers

    @staticmethod
    def _single_species(site) -> tuple[str, float]:
        """Return the ``(symbol, occupancy)`` of a single-species site.

        A mixed solid-solution site (two species sharing one position) cannot be
        symmetrized by moyopy nor written as one mcif ``_atom_site_*`` row, so it is
        rejected explicitly rather than silently mangled.
        """
        items = list(site.species.items())
        if len(items) != 1:
            raise NotImplementedError(
                f"Mixed-occupancy site {site.frac_coords} has species {site.species}; "
                "only single-species (optionally partial) occupancy is supported"
            )
        element, occ = items[0]
        return element.symbol, float(occ)

    @staticmethod
    def _snap_translation(
        trans: list[float], *, denom: int = 24, atol: float = 1e-3
    ) -> np.ndarray:
        """Snap a fractional translation to the nearest k/denom when within atol.

        moyopy's tolerant symmetry search can return translations carrying ~1e-6 noise
        on a fitted (numerically imperfect) structure. The crystallographic translations
        here are multiples of 1/2, 1/3, 1/4, 1/6 (all on the 1/24 grid), so snapping
        removes the noise without altering a genuine translation.
        """
        t = np.array(trans, dtype=float) % 1.0
        snapped = np.round(t * denom) / denom
        return np.where(np.abs(snapped - t) <= atol, snapped, t)
