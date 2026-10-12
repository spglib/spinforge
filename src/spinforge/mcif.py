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

import numpy as np
from moyopy import (
    MagneticSpaceGroupType,
    MoyoNonCollinearMagneticDataset,
    NonCollinearMagneticCell,
)
from pymatgen.core import Structure

from spinforge._cif import (
    _ATOM_SITE_LOOP_HEADER,
    _atom_type_lines,
    _number,
    _physical_sites,
    _SiteConstituent,
    _species_token,
    _xyzt_string,
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
    identity, including oxidation state. Compositions and magnetic flags enter
    the moyopy equivalence labels, so symmetry cannot exchange chemically distinct
    sites. Selected constituents share the site's occupied-atom moment; unselected
    constituents have zero moment. Occupancy never scales the written moment.
    The optional boolean ``magnetic`` site property defaults to true. A false
    flag requires a zero moment; a true flag retains a moment row even for zero.
    A mixed site applies its moment and flag to all its species. Use coincident
    constituent rows for partial magnetic selection. Their selected moments must
    agree within ``mag_symprec`` (which defaults to ``symprec`` as in moyopy).

    Parameters
    ----------
    struct
        Magnetic structure with a ``magmom`` site property.
    symprec, mag_symprec
        moyopy symmetry-search tolerances (basis units / moment units).
        ``symprec`` also bounds the Cartesian displacement introduced when
        snapping translations to the crystallographic fraction grid.
    """

    def __init__(
        self,
        struct: Structure,
        symprec: float = 1e-4,
        *,
        mag_symprec: float | None = None,
    ) -> None:
        physical = _physical_sites(
            struct, symprec=symprec, mag_symprec=symprec if mag_symprec is None else mag_symprec
        )
        struct = physical.structure
        basis = struct.lattice.matrix
        moments = np.asarray(struct.site_properties["magmom"])
        site_species = physical.constituents
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
            _xyzt_string(
                np.array(rot),
                self._snap_translation(trans, basis, symprec=symprec),
                -1 if rev else 1,
            )
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
            f"_cell_length_a {_number(a)}",
            f"_cell_length_b {_number(b)}",
            f"_cell_length_c {_number(c)}",
            f"_cell_angle_alpha {_number(alpha)}",
            f"_cell_angle_beta {_number(beta)}",
            f"_cell_angle_gamma {_number(gamma)}",
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
                    f"{label} {_species_token(constituent.species)} "
                    f"{_number(fx)} {_number(fy)} {_number(fz)} {_number(constituent.occupancy)}"
                )
                if constituent.magnetic:
                    moment_lines.append(f"{label} {' '.join(_number(value) for value in comps)}")
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
            signature = tuple((row.species, row.occupancy, row.magnetic) for row in site)
            numbers.append(signatures.setdefault(signature, len(signatures) + 1))
        return numbers

    @staticmethod
    def _snap_translation(
        trans: list[float], basis: np.ndarray, *, symprec: float, denom: int = 24
    ) -> np.ndarray:
        """Snap a fractional translation when its Cartesian change is within symprec.

        The 1/24 grid includes usual crystallographic fractions. General-origin
        translations need not lie on that grid, so preserve the complete vector
        unless snapping it respects the caller's Cartesian symmetry tolerance.
        """
        t = np.array(trans, dtype=float) % 1.0
        snapped = np.round(t * denom) / denom
        return snapped if np.linalg.norm((snapped - t) @ basis) <= symprec else t
