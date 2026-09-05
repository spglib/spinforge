"""Minimal spinCIF reader for files produced by :class:`SpinCifWriter`.

CIF tokenization (tags, quoting, loops) is delegated to pymatgen's
:class:`~pymatgen.io.cif.CifFile`; this module supplies the spinCIF
semantics on top: the decoupled ``uvw`` spin operations, the spin-frame
declaration, and the expansion of the asymmetric unit into the full
magnetic structure. It targets the subset of the draft dictionary the
writer emits (see ``SPINCIF_REVISION``) plus the FINDSPINGROUP reference
files; it is not a general spinCIF reader.
"""

from __future__ import annotations

import ast
import re
from os import PathLike

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import MagSymmOp
from pymatgen.io.cif import CifBlock, CifFile
from spgrep.utils import NDArrayFloat, is_integer_array

from ._operations import _crystal_cartesian_frame

__all__ = [
    "SpinCifReader",
]

SymmetryOperation = tuple[np.ndarray, np.ndarray, int, np.ndarray]
"""(rotation, translation, time_reversal, spin_matrix)"""


class SpinCifReader:
    """Read a spinCIF string written by :class:`SpinCifWriter`.

    Attributes
    ----------
    operations, spin_lattice
        Rows of the ``_space_group_symop_spin_operation`` /
        ``_space_group_symop_spin_lattice`` loops as
        ``(rotation, translation, time_reversal, spin_matrix)`` tuples, with
        the spin matrix in the declared spin basis.
    sites
        ``(label, symbol, frac_position, moment_uvw, multiplicity)`` per
        asymmetric-unit site (zero moment when the site has no
        ``_atom_site_spin_moment`` row).
    lattice
        :class:`pymatgen.core.Lattice` from the cell tags.
    spin_basis
        Unit vectors of the declared spin basis as rows (Cartesian, in the
        lattice's own frame); moments expand as ``components @ spin_basis``.
    collinear_direction_xyz, coplanar_perp_uvw
        The spin-only tag values, ``"."`` when not applicable.

    Parameters
    ----------
    text
        The spinCIF content.
    symprec
        Tolerance (fractional units) for merging symmetry-equivalent
        positions and for moment consistency when expanding the orbits.
    """

    def __init__(self, text: str, *, symprec: float = 1e-4) -> None:
        self._symprec = symprec
        blocks = CifFile.from_str(text).data
        if len(blocks) != 1:
            raise ValueError(f"Expected a single data block, found {len(blocks)}")
        (block,) = blocks.values()

        self.operations = _read_symop_loop(block, "_space_group_symop_spin_operation")
        if "_space_group_symop_spin_lattice.xyzt" in block.data:
            self.spin_lattice = _read_symop_loop(block, "_space_group_symop_spin_lattice")
        else:  # the loop is optional in the dictionary
            self.spin_lattice = [(np.eye(3), np.zeros(3), 1, np.eye(3))]

        moments = {
            label: np.array([float(u), float(v), float(w)])
            for label, u, v, w in zip(
                *(
                    block.data.get(f"_atom_site_spin_moment.{column}", [])
                    for column in ("label", "axis_u", "axis_v", "axis_w")
                )
            )
        }
        self.sites = [
            (
                label,
                symbol,
                np.array([float(fx), float(fy), float(fz)]),
                moments.get(label, np.zeros(3)),
                int(multiplicity),
            )
            for label, symbol, fx, fy, fz, multiplicity in zip(
                block.data["_atom_site_label"],
                block.data["_atom_site_type_symbol"],
                block.data["_atom_site_fract_x"],
                block.data["_atom_site_fract_y"],
                block.data["_atom_site_fract_z"],
                block.data["_atom_site_symmetry_multiplicity"],
            )
        ]

        self.lattice = Lattice.from_parameters(
            *(float(block.data[f"_cell_length_{name}"]) for name in "abc"),
            *(float(block.data[f"_cell_angle_{name}"]) for name in ("alpha", "beta", "gamma")),
        )
        self.spin_basis = _read_spin_basis(block, text, np.array(self.lattice.matrix))
        self.collinear_direction_xyz = block.data["_space_group_spin.collinear_direction_xyz"]
        self.coplanar_perp_uvw = block.data["_space_group_spin.coplanar_perp_uvw"]

    @classmethod
    def from_file(cls, filename: str | PathLike[str], *, symprec: float = 1e-4) -> SpinCifReader:
        with open(filename) as f:
            return cls(f.read(), symprec=symprec)

    def expanded_operations(self) -> list[SymmetryOperation]:
        """Products of the spin-lattice and operation loops (the full listed group)."""
        return [
            (w, (t + tau) % 1.0, tr_op * tr_latt, u_latt @ u_op)
            for w, t, tr_op, u_op in self.operations
            for _, tau, tr_latt, u_latt in self.spin_lattice
        ]

    def get_structure(self) -> Structure:
        """Expand the asymmetric unit into the full magnetic structure.

        Moments are returned as Cartesian ``magmom`` site properties in the
        lattice's own Cartesian frame. Raises if an orbit disagrees with its
        ``_atom_site_symmetry_multiplicity`` or transforms inconsistently.
        """
        species, coords, magmoms = [], [], []
        for label, symbol, frac, moment_uvw, multiplicity in self.sites:
            orbit: list[tuple[np.ndarray, np.ndarray]] = []
            for w, t, _tr, u in self.expanded_operations():
                position = (w @ frac + t) % 1.0
                moment = u @ moment_uvw
                for known_position, known_moment in orbit:
                    if is_integer_array(position - known_position, atol=self._symprec):
                        if not np.allclose(moment, known_moment, rtol=0, atol=self._symprec):
                            raise ValueError(
                                f"Inconsistent moments for {label} at {position}: "
                                f"{moment} vs {known_moment}"
                            )
                        break
                else:
                    orbit.append((position, moment))
            if len(orbit) != multiplicity:
                raise ValueError(
                    f"Orbit of {label} has {len(orbit)} sites, multiplicity says {multiplicity}"
                )
            for position, moment in orbit:
                species.append(symbol)
                coords.append(position)
                magmoms.append(moment @ self.spin_basis)
        return Structure(self.lattice, species, coords, site_properties={"magmom": magmoms})


def _read_symop_loop(block: CifBlock, category: str) -> list[SymmetryOperation]:
    ops = []
    for xyzt, uvw in zip(block.data[f"{category}.xyzt"], block.data[f"{category}.uvw"]):
        op = MagSymmOp.from_xyzt_str(xyzt)
        ops.append(
            (
                np.array(op.rotation_matrix),
                np.array(op.translation_vector),
                op.time_reversal,
                _uvw_matrix(uvw),
            )
        )
    return ops


def _uvw_matrix(expr: str) -> np.ndarray:
    return _linear_matrix(expr, "uvw")


def _linear_matrix(expr: str, variables: str) -> np.ndarray:
    """Evaluate comma-separated linear expressions in the given variables.

    Handles the spinCIF expression grammar (implicit multiplication with the
    numeric factor preceding the variable, fractions), e.g.
    ``-1/2u+0.866025403784439v,...`` for operations or ``0.17a,...`` for the
    spin frame. Expressions are parsed with :mod:`ast` and reduced to their
    linear form by an interpreter restricted to that grammar, never with
    ``eval``; anything that is not a homogeneous linear map is rejected.
    """
    components = expr.split(",")
    if len(components) != len(variables):
        raise ValueError(f"Bad {variables} expression: {expr}")
    matrix = np.zeros((len(variables), len(variables)))
    implicit_multiplication = re.compile(rf"(?<=[\d\)])(?=[{variables}])")
    for row, component in enumerate(components):
        try:
            parsed = ast.parse(implicit_multiplication.sub("*", component), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Bad {variables} expression: {component}") from exc
        try:
            constant, coefficients = _linear_form(parsed.body, variables)
        except ZeroDivisionError as exc:
            raise ValueError(f"Bad {variables} expression: {component}") from exc
        if constant != 0.0:
            raise ValueError(f"Expression is not linear in {variables}: {component}")
        matrix[row] = coefficients
    return matrix


def _linear_form(node: ast.expr, variables: str) -> tuple[float, NDArrayFloat]:
    """The ``(constant, coefficients)`` of a spinCIF expression.

    The grammar is numbers, the variables in ``variables``, and ``+ - * /``
    (unary and binary). Anything else (attributes, subscripts, calls, other
    constants) raises ``ValueError``, so untrusted file content cannot execute
    code. Linearity is enforced structurally: both operands of a product, and
    the divisor of a quotient, must be free of variables. Sampling the
    expression instead would misread nonlinear input that happens to agree
    with a linear map at the sample points.
    """
    coefficients = np.zeros(len(variables))
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return float(node.value), coefficients
    elif isinstance(node, ast.Name):
        if node.id in variables:
            coefficients[variables.index(node.id)] = 1.0
            return 0.0, coefficients
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        constant, coefficients = _linear_form(node.operand, variables)
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return sign * constant, sign * coefficients
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, ast.Add | ast.Sub | ast.Mult | ast.Div
    ):
        left_constant, left_coefficients = _linear_form(node.left, variables)
        right_constant, right_coefficients = _linear_form(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left_constant + right_constant, left_coefficients + right_coefficients
        if isinstance(node.op, ast.Sub):
            return left_constant - right_constant, left_coefficients - right_coefficients
        if isinstance(node.op, ast.Mult):
            if np.any(left_coefficients != 0.0) and np.any(right_coefficients != 0.0):
                raise ValueError(f"Expression is not linear in {variables}: {ast.unparse(node)}")
            return (
                left_constant * right_constant,
                left_constant * right_coefficients + right_constant * left_coefficients,
            )
        if np.any(right_coefficients != 0.0):
            raise ValueError(f"Expression is not linear in {variables}: {ast.unparse(node)}")
        return left_constant / right_constant, left_coefficients / right_constant
    raise ValueError(f"Unsupported token in spinCIF expression: {ast.unparse(node)}")


def _read_spin_basis(block: CifBlock, text: str, lattice: np.ndarray) -> np.ndarray:
    """Unit spin-basis vectors (rows, Cartesian) from the frame declaration.

    Supports the ``transform_spinframe_P_abc`` route the writer emits (basis
    vectors as linear combinations of the lattice vectors) and, for files
    written by earlier versions, ``spinframe_orientation_cartn [0 0 0]``
    (spin basis = crystal-Cartesian frame x || a, z || c*; a CIF 2.0 list
    value, matched on the raw text since CifFile parses CIF 1.1). Rows are
    normalized: moment components count along the unit axis directions.
    """
    expression = block.data.get("_space_group_spin.transform_spinframe_P_abc")
    if expression is not None:
        basis = _linear_matrix(expression, "abc") @ lattice
        norms = np.linalg.norm(basis, axis=1)
        if not np.all(np.isfinite(basis)) or np.any(norms < 1e-8):
            raise ValueError(f"Degenerate spin-frame axis in declaration: {expression}")
        basis = basis / norms[:, None]
        if not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-3):
            raise ValueError(f"Spin-frame axes are not orthogonal: {expression}")
        return basis
    match = re.search(
        r"^_space_group_spin\.spinframe_orientation_cartn\s+\[0 0 0\]", text, flags=re.MULTILINE
    )
    if match is not None:
        return _crystal_cartesian_frame(lattice).T
    raise ValueError("No supported spin-frame declaration found")
