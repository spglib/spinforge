"""Write magnetic structures as mcif files (BNS/MAGNDATA convention).

For single-species sites, ``pymatgen.core.Structure.from_file`` handles mcif,
and its symmetry expansion of ``_atom_site_moment.crystalaxis_*``
components is correct as of pymatgen-core v2026.7.16
(materialsproject/pymatgen-core#76): moments are converted to Cartesian before
applying an operation, so |m| is preserved even when an operation mixes
unequal or non-orthogonal axes (e.g. hexagonal settings; see
materialsproject/pymatgen-core#76 for the old failure mode).
Mixed-species files use constituent atom rows, which pymatgen's magnetic
structure reader does not currently support.

:class:`MCifWriter` has no upstream equivalent (``CifWriter(write_magmoms=True)``
dumps every atom in P1 with no magnetic operations).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from moyopy import (
    MagneticSpaceGroupType,
    MoyoNonCollinearMagneticDataset,
    NonCollinearMagneticCell,
)
from pymatgen.core import Structure
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom

from spinforge._cif import (
    _ATOM_SITE_LOOP_HEADER,
    _atom_type_lines,
    _normalize_site_species,
    _SiteConstituent,
)

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
    ``_space_group_symop_magn_centering.xyz`` loops.

    The structure is symmetrized in its own input setting (the basis is not
    standardized). ``struct`` must carry a ``magmom`` site property (pymatgen
    ``Magmom`` or a Cartesian 3-vector per site). Moment components are written as
    ``_atom_site_moment.crystalaxis_*`` in the unit-axis convention read back by
    ``CifParser`` (``m = mx*a_hat + my*b_hat + mz*c_hat``).

    Each constituent is written as an atom row with its occupancy and full species
    identity, including oxidation state. Compositions and magnetic selections enter
    the moyopy equivalence labels, so symmetry cannot exchange chemically distinct
    sites. Selected constituents share the site's occupied-atom moment; unselected
    constituents have zero moment. Occupancy never scales the written moment.

    Parameters
    ----------
    struct
        Magnetic structure with a ``magmom`` site property.
    symprec, mag_symprec
        moyopy symmetry-search tolerances (basis units / moment units).
    magnetic_species
        Species identities selected at each site, such as ``[("Fe2+",), ()]``.
        Defaults to all constituents. An empty selection requires a zero moment.
        Selected constituents retain moment rows even when their moment is zero.
    """

    def __init__(
        self,
        struct: Structure,
        symprec: float = 1e-4,
        *,
        mag_symprec: float | None = None,
        magnetic_species: Sequence[Sequence[str]] | None = None,
    ) -> None:
        if "magmom" not in struct.site_properties:
            raise ValueError("structure must have a 'magmom' site property to write an mcif")
        basis = struct.lattice.matrix
        moments = np.array([Magmom(m).global_moment for m in struct.site_properties["magmom"]])
        site_species = _normalize_site_species(
            [site.species for site in struct], magnetic_species, moments
        )
        cell = NonCollinearMagneticCell(
            basis=basis.tolist(),
            positions=struct.frac_coords.tolist(),
            numbers=self._composition_numbers(site_species),
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
        self._mcif_str = self._format_mcif(
            struct, site_species, reps, op_strings, moments, inv_unit_axes, bns
        )

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
        site_species: tuple[tuple[_SiteConstituent, ...], ...],
        reps: list[int],
        op_strings: list[str],
        moments: np.ndarray,
        inv_unit_axes: np.ndarray,
        bns: str,
    ) -> str:
        """Assemble symmetrized-mcif text from the asymmetric unit and magnetic operations."""
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        lines = [
            "# Symmetrized magnetic CIF written by spinforge.mcif.MCifWriter",
            "data_magnetic_structure",
            f"_chemical_formula_sum '{structure.composition.element_composition.formula}'",
            f"_space_group_magn.number_BNS {bns}",
            f"_cell_length_a {a:.6f}",
            f"_cell_length_b {b:.6f}",
            f"_cell_length_c {c:.6f}",
            f"_cell_angle_alpha {alpha:.6f}",
            f"_cell_angle_beta {beta:.6f}",
            f"_cell_angle_gamma {gamma:.6f}",
            *_atom_type_lines(site_species),
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
        counts: dict[str, int] = {}
        moment_lines = []
        for idx in reps:
            fx, fy, fz = structure[idx].frac_coords
            comps = moments[idx] @ inv_unit_axes
            for constituent in site_species[idx]:
                symbol = constituent.species.symbol
                counts[symbol] = counts.get(symbol, 0) + 1
                label = f"{symbol}{counts[symbol]}"
                lines.append(
                    f"{label} {constituent.species} {fx:.6f} {fy:.6f} {fz:.6f} "
                    f"{constituent.occupancy:.12g}"
                )
                if constituent.magnetic:
                    moment_lines.append(f"{label} {comps[0]:.6f} {comps[1]:.6f} {comps[2]:.6f}")
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
    def _composition_numbers(site_species: tuple[tuple[_SiteConstituent, ...], ...]) -> list[int]:
        """moyopy equivalence labels for complete compositions and magnetic selections."""
        signatures: dict[tuple, int] = {}
        numbers = []
        for site in site_species:
            signature = tuple((str(row.species), row.occupancy, row.magnetic) for row in site)
            numbers.append(signatures.setdefault(signature, len(signatures) + 1))
        return numbers

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
