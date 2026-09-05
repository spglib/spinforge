"""The spinCIF writer: assembles the .scif text from a spin-symmetry-adapted
structure and its nontrivial spin space group."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from moyopy import SpaceGroupType
from pymatgen.core import Element, Lattice, Structure
from spgrep.utils import NDArrayFloat
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from spinforge._cif import _ATOM_SITE_LOOP_HEADER
from spinforge.configuration import SpinSymmetryAdaptedStructure
from spinforge.msg import MagneticSpaceSubgroup
from spinforge.ssg import NontrivialSpinSpaceGroup

from ._format import (
    _collinear_direction_xyz,
    _coplanar_perp_uvw,
    _fmt,
    _spinframe_p_abc,
    _symmform,
    _uvw_expr,
    _xyzt,
)
from ._operations import (
    _crystal_cartesian_frame,
    _nontrivial_operations,
    _orbit_partition,
    _supercell_spin_translations,
    _to_supercell_frame,
)

SPINCIF_REVISION = "2026-08-06 (COMCIFS/spinCIF@4142eb935045)"
"""Revision of the draft spinCIF dictionary this writer targets.

The upstream dictionary is preliminary and may change; bump this together
with any format adjustments. The revision is stamped into every written file.
"""


class SpinCifWriter:
    """A pymatgen-style writer for spinCIF (draft COMCIFS ``core_spin.dic``).

    Mirrors :class:`spinforge.mcif.MCifWriter`: ``str(writer)`` returns the
    spinCIF text and ``writer.write_file(path)`` writes it.

    Parameters
    ----------
    sas
        Spin-symmetry-adapted structure providing the supercell and the
        moment basis (used for the ``symmform_uvw`` restrictions).
    nssg
        The nontrivial spin space group the structure was generated from,
        in the primitive input-cell setting.
    spin_only_group
        The trivial spin-only group the structure was enumerated with; its
        type and axis fill ``_space_group_spin.collinear_direction_xyz`` /
        ``coplanar_perp_uvw``.
    magnetic_moments
        Cartesian magnetic moments per supercell site, shape
        ``(num_supercell_sites, 3)``, exactly as passed to
        :meth:`SpinSymmetryAdaptedStructure.generate_with_magnetic_moments`.
        For structures from ``generate_oriented`` use :meth:`from_oriented`,
        which also conjugates the spin operations by the MSG rotation.
    site_occupancies
        Fractional occupancy per supercell site (``None`` = fully occupied).
        Written as ``_atom_site_occupancy``; must be invariant under the SSG
        operations, since one asymmetric-unit row represents each orbit.
    data_name
        CIF data block name.
    parent_space_group_type
        Parent (nonmagnetic) space-group type; written as
        ``_parent_space_group`` tags when given.
    atol
        Tolerance for spin-space and translation checks (coefficient
        snapping, orthogonality). Site matching uses the supercell's own
        symprec via :meth:`Supercell.site_permutations`.
    """

    def __init__(
        self,
        sas: SpinSymmetryAdaptedStructure,
        nssg: NontrivialSpinSpaceGroup,
        magnetic_moments: NDArrayFloat,
        *,
        spin_only_group: SpinOnlyGroup,
        data_name: str = "spinforge",
        parent_space_group_type: SpaceGroupType | None = None,
        site_occupancies: Sequence[float] | NDArrayFloat | None = None,
        atol: float = 1e-5,
    ) -> None:
        self._build(
            sas,
            nssg,
            np.asarray(magnetic_moments, dtype=float),
            spin_only_group=spin_only_group,
            q_mat=np.eye(3),
            data_name=data_name,
            parent_space_group_type=parent_space_group_type,
            site_occupancies=site_occupancies,
            atol=atol,
        )

    @classmethod
    def from_oriented(
        cls,
        sas: SpinSymmetryAdaptedStructure,
        nssg: NontrivialSpinSpaceGroup,
        structure: Structure,
        msg: MagneticSpaceSubgroup,
        *,
        spin_only_group: SpinOnlyGroup,
        data_name: str = "spinforge",
        parent_space_group_type: SpaceGroupType | None = None,
        site_occupancies: Sequence[float] | NDArrayFloat | None = None,
        atol: float = 1e-5,
    ) -> SpinCifWriter:
        """Writer for one ``(structure, msg)`` pair from ``generate_oriented``.

        Takes the moments from the structure's ``magmom`` site property and
        conjugates the spin parts of the SSG operations by the MSG rotation
        ``msg.Q`` that ``generate_oriented`` applied to them.
        """
        writer = cls.__new__(cls)
        moments = np.array(
            [np.asarray(m) for m in structure.site_properties["magmom"]], dtype=float
        )
        writer._build(
            sas,
            nssg,
            moments,
            spin_only_group=spin_only_group,
            q_mat=np.asarray(msg.Q, dtype=float),
            data_name=data_name,
            parent_space_group_type=parent_space_group_type,
            site_occupancies=site_occupancies,
            atol=atol,
        )
        return writer

    def _build(
        self,
        sas: SpinSymmetryAdaptedStructure,
        nssg: NontrivialSpinSpaceGroup,
        moments: NDArrayFloat,
        *,
        spin_only_group: SpinOnlyGroup,
        q_mat: NDArrayFloat,
        data_name: str,
        parent_space_group_type: SpaceGroupType | None,
        site_occupancies: Sequence[float] | NDArrayFloat | None = None,
        atol: float,
    ) -> None:
        positions = np.asarray(sas.supercell.positions)
        if moments.shape != (len(positions), 3):
            raise ValueError(
                f"magnetic_moments shape {moments.shape} does not match "
                f"({len(positions)}, 3) supercell sites"
            )
        if not np.allclose(q_mat @ q_mat.T, np.eye(3), rtol=0, atol=atol):
            raise ValueError(f"MSG rotation Q must be orthogonal, got {q_mat}")

        lattice = np.asarray(sas.supercell.basis)
        frame = _crystal_cartesian_frame(lattice)  # columns = spin basis vectors (Cartesian)

        oriented_basis = [q_mat @ np.asarray(b).T for b in sas.magnetic_moments_basis]
        _validate_moments(moments, oriented_basis, spin_only_group, q_mat, atol=atol)

        prim_operations = _nontrivial_operations(
            nssg.nontrivial_coset, nssg.invariant_rotations, nssg.invariant_translations
        )
        operations = _to_supercell_frame(sas, prim_operations)
        spin_lattice = _supercell_spin_translations(sas, nssg.spin_translation_coset)
        # Spin parts: conjugate into the oriented frame, then express in the spin basis.
        operations = [(w, t, frame.T @ q_mat @ u @ q_mat.T @ frame) for w, t, u in operations]
        spin_lattice = [(t, frame.T @ q_mat @ u @ q_mat.T @ frame) for t, u in spin_lattice]

        # The asymmetric unit comes from the site orbits under the full SSG
        # (coset representatives composed with the spin translations); the
        # supercell owns the site matching. Moment equivariance of the written
        # operations is exercised by the round-trip tests, not here.
        spin_translations = [np.array(t2) for t2 in nssg.spin_translation_coset.translations]
        permutations = sas.supercell.site_permutations(
            [w for w, _, _ in prim_operations for _ in spin_translations],
            [t + t2 for _, t, _ in prim_operations for t2 in spin_translations],
        )
        orbits = _orbit_partition(len(positions), permutations)
        occupancies = _validated_occupancies(
            site_occupancies, len(positions), permutations, atol=atol
        )
        collinear_xyz = _collinear_direction_xyz(spin_only_group, q_mat, lattice)
        coplanar_uvw = _coplanar_perp_uvw(spin_only_group, q_mat, frame)

        self._scif_str = _format_scif(
            data_name=data_name,
            collinear_xyz=collinear_xyz,
            coplanar_uvw=coplanar_uvw,
            lattice=lattice,
            frame=frame,
            numbers=sas.supercell.numbers,
            positions=positions,
            moments=moments,
            occupancies=occupancies,
            orbits=orbits,
            operations=operations,
            spin_lattice=spin_lattice,
            site_moment_subspaces=[
                [frame.T @ b[:, rep] for b in oriented_basis] for rep, _ in orbits
            ],
            parent_space_group_type=parent_space_group_type,
            atol=atol,
        )

    def __str__(self) -> str:
        """The spinCIF as a string."""
        return self._scif_str

    def write_file(self, filename: str, mode: str = "wt") -> None:
        """Write the spinCIF to ``filename``."""
        with open(filename, mode) as f:
            f.write(self._scif_str)


def _validate_moments(
    moments: NDArrayFloat,
    oriented_basis: list[np.ndarray],
    spin_only_group,
    q_mat: NDArrayFloat,
    *,
    atol: float,
) -> None:
    """Fail fast on caller mismatches, in O(n).

    The moments must be a linear combination of the (rotated) SSA basis and
    compatible with the declared spin-only group axis; both break when the
    caller passes moments from a different SSG, feeds oriented moments to
    the plain constructor instead of ``from_oriented``, or supplies an
    inconsistent ``spin_only_group``.
    This intentionally replaces the per-operation equivariance validation,
    which is exercised by the round-trip tests instead of the write path.
    A sample whose moments span a lower rank than the spin-only group allows
    (including all zeros) is valid: the tags describe the SSG model.
    """
    target = moments.reshape(-1)
    scale = max(1.0, float(np.linalg.norm(target)))
    basis_matrix = np.stack([b.T.reshape(-1) for b in oriented_basis], axis=1)
    coefficients, *_ = np.linalg.lstsq(basis_matrix, target, rcond=None)
    residual = np.linalg.norm(basis_matrix @ coefficients - target)
    if residual > atol * scale:
        raise ValueError(
            f"magnetic_moments are not a combination of the SSA basis "
            f"(residual {residual:.3e}); check that they come from this "
            f"SpinSymmetryAdaptedStructure, and use from_oriented for "
            f"structures from generate_oriented"
        )

    if spin_only_group.axis is None:
        return
    axis = q_mat @ np.asarray(spin_only_group.axis, dtype=float)
    if spin_only_group.spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
        deviation = np.linalg.norm(np.cross(moments, axis), axis=1)  # parallel to axis
    elif spin_only_group.spin_only_group_type == SpinOnlyGroupType.COPLANAR:
        deviation = np.abs(moments @ axis)  # in the plane normal to axis
    else:
        return
    if np.any(deviation > atol * scale):
        raise ValueError(
            f"magnetic_moments are incompatible with the "
            f"{spin_only_group.spin_only_group_type} spin-only axis {axis}"
        )


def _validated_occupancies(
    site_occupancies: Sequence[float] | NDArrayFloat | None,
    num_sites: int,
    permutations: list,
    *,
    atol: float,
) -> NDArrayFloat:
    """Per-site occupancies, checked against the SSG's site permutations.

    One asymmetric-unit row represents each orbit, so an occupancy pattern
    that is not invariant under the operations cannot be written faithfully.
    """
    if site_occupancies is None:
        return np.ones(num_sites)
    occupancies = np.asarray(site_occupancies, dtype=float)
    if occupancies.shape != (num_sites,):
        raise ValueError(
            f"site_occupancies shape {occupancies.shape} does not match "
            f"({num_sites},) supercell sites"
        )
    if np.any(occupancies <= 0.0) or np.any(occupancies > 1.0 + atol):
        raise ValueError(f"site_occupancies must lie in (0, 1], got {occupancies.tolist()}")
    for permutation in permutations:
        permuted = occupancies[np.asarray(permutation, dtype=int)]
        if not np.allclose(permuted, occupancies, atol=atol):
            raise ValueError(
                "site_occupancies are not invariant under the SSG operations; "
                "symmetry-related sites must share one occupancy."
            )
    return occupancies


def _format_scif(
    *,
    data_name: str,
    collinear_xyz: str,
    coplanar_uvw: str,
    lattice: NDArrayFloat,
    frame: NDArrayFloat,
    numbers: list[int],
    positions: NDArrayFloat,
    moments: NDArrayFloat,
    occupancies: NDArrayFloat,
    orbits: list[tuple[int, int]],
    operations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    spin_lattice: list[tuple[np.ndarray, np.ndarray]],
    site_moment_subspaces: list[list[np.ndarray]],
    parent_space_group_type: SpaceGroupType | None,
    atol: float,
) -> str:
    lines = [
        "#\\#CIF_2.0",
        "# spinCIF written by spinforge.scif.SpinCifWriter",
        "# draft spinCIF dictionary, https://github.com/COMCIFS/spinCIF",
        f"# dictionary revision: {SPINCIF_REVISION}",
        f"data_{data_name}",
        f"_space_group_spin.transform_spinframe_P_abc  {_spinframe_p_abc(frame, lattice, atol=atol)}",
        f"_space_group_spin.collinear_direction_xyz  {collinear_xyz}",
        f"_space_group_spin.coplanar_perp_uvw  {coplanar_uvw}",
        "_space_group_spin.rotation_axis_xyz  .",
        "_space_group_spin.rotation_angle  .",
    ]
    if parent_space_group_type is not None:
        hm = parent_space_group_type.hm_short.replace(" ", "")
        lines += [
            f'_parent_space_group.name_H-M_alt  "{hm}"',
            f"_parent_space_group.IT_number  {parent_space_group_type.number}",
        ]

    a, b, c, alpha, beta, gamma = Lattice(lattice).parameters
    lines += [
        "",
        f"_cell_length_a    {a:.6f}",
        f"_cell_length_b    {b:.6f}",
        f"_cell_length_c    {c:.6f}",
        f"_cell_angle_alpha {alpha:.6f}",
        f"_cell_angle_beta  {beta:.6f}",
        f"_cell_angle_gamma {gamma:.6f}",
        "",
        "loop_",
        "_space_group_symop_spin_operation.id",
        "_space_group_symop_spin_operation.xyzt",
        "_space_group_symop_spin_operation.uvw",
    ]
    for i, (w, t, u) in enumerate(operations):
        lines.append(f"{i + 1} {_xyzt(w, t, u)}  {_uvw_expr(u, atol=atol)}")

    lines += [
        "",
        "loop_",
        "_space_group_symop_spin_lattice.id",
        "_space_group_symop_spin_lattice.xyzt",
        "_space_group_symop_spin_lattice.uvw",
    ]
    for i, (t, u) in enumerate(spin_lattice):
        lines.append(f"{i + 1} {_xyzt(np.eye(3, dtype=int), t, u)}  {_uvw_expr(u, atol=atol)}")

    symbols = [Element.from_Z(z).symbol for z in numbers]
    lines += ["", "loop_", "_atom_type_symbol"]
    lines += list(dict.fromkeys(symbols))

    lines += (*_ATOM_SITE_LOOP_HEADER, "_atom_site_symmetry_multiplicity")
    labels = []
    counts: dict[str, int] = {}
    for rep, multiplicity in orbits:
        symbol = symbols[rep]
        counts[symbol] = counts.get(symbol, 0) + 1
        label = f"{symbol}{counts[symbol]}"
        labels.append(label)
        fx, fy, fz = positions[rep]
        lines.append(
            f"{label} {symbol} {_fmt(fx)} {_fmt(fy)} {_fmt(fz)} "
            f"{_fmt(float(occupancies[rep]))} {multiplicity}"
        )

    moment_lines = []
    for (rep, _), label, subspace in zip(orbits, labels, site_moment_subspaces):
        m_uvw = frame.T @ moments[rep]
        components = [_fmt(x) for x in m_uvw]
        # The reader treats an omitted row as zero, so omit one only when the
        # written values are all zero anyway. The decision is read off the
        # formatted text rather than a cutoff, so weak moments that still print
        # a nonzero digit are never silently dropped.
        if all(float(component) == 0.0 for component in components):
            continue
        symmform = _symmform(subspace, atol=atol)
        moment_lines.append(
            f"{label} {' '.join(components)} {symmform} {_fmt(float(np.linalg.norm(m_uvw)))}"
        )
    if moment_lines:
        lines += [
            "",
            "loop_",
            "_atom_site_spin_moment.label",
            "_atom_site_spin_moment.axis_u",
            "_atom_site_spin_moment.axis_v",
            "_atom_site_spin_moment.axis_w",
            "_atom_site_spin_moment.symmform_uvw",
            "_atom_site_spin_moment.magnitude",
        ]
        lines += moment_lines
    lines.append("")
    return "\n".join(lines)
