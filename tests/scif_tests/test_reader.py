"""Reader-side behaviour: external files, grammar limits, and legacy tags."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pymatgen.core import Composition
from spinspg.spin import SpinOnlyGroupType

from spinforge.scif import SpinCifReader, SpinCifWriter
from spinforge.scif._operations import _crystal_cartesian_frame
from spinforge.scif._reader import _linear_matrix


def test_reader_external_style_columns(external_scif):
    reader = SpinCifReader(external_scif)
    assert [label for label, *_ in reader.sites] == ["Mn1"]
    structure = reader.get_structure()
    assert len(structure) == 2
    assert np.allclose(structure.site_properties["magmom"], [[0, 0, 3.0], [0, 0, 3.0]])


def test_reader_rejects_duplicate_moment_labels(external_scif):
    text = external_scif + "Mn1 0.0 0.0 3.0 0,0,w 0,0,w 3.0\n"
    with pytest.raises(ValueError, match="Spin-moment labels must be unique"):
        SpinCifReader(text)


@pytest.mark.parametrize(
    "columns",
    [
        "loop_\n_atom_site_spin_moment.label\nMn1 Mn2\n"
        "loop_\n_atom_site_spin_moment.axis_u\n_atom_site_spin_moment.axis_v\n"
        "_atom_site_spin_moment.axis_w\n0 0 3\n",
        "loop_\n_atom_site_spin_moment.label\n_atom_site_spin_moment.axis_u\n"
        "_atom_site_spin_moment.axis_v\n_atom_site_spin_moment.axis_w\nMn1 0 0 3\n"
        "loop_\n_atom_site_spin_moment.symmform_uvw\n0,0,w 0,0,w\n",
        "_atom_site_spin_moment.label Mn1\n_atom_site_spin_moment.axis_u 0\n"
        "_atom_site_spin_moment.axis_v 0\n_atom_site_spin_moment.axis_w 3\n",
    ],
)
def test_reader_rejects_malformed_moment_columns(external_scif, columns):
    text = external_scif.split("loop_\n_atom_site_spin_moment.label")[0] + columns
    with pytest.raises(ValueError, match="equal-length loop columns"):
        SpinCifReader(text)


@pytest.mark.parametrize(
    ("type_rows", "message"),
    [
        ("Mn2+ 3", "contradicts its explicit species identity"),
        ("Mn2+ 2\nMn2+ 2", "Atom-type symbols must be unique"),
        ("Mn2+ nan", "Oxidation states must be finite"),
        ("Mn2+ inf", "Oxidation states must be finite"),
    ],
)
def test_reader_rejects_invalid_oxidation_metadata(external_scif, type_rows, message):
    text = external_scif.replace("Mn1 Mn ", "Mn1 Mn2+ ")
    text += "loop_\n_atom_type_symbol\n_atom_type_oxidation_number\n" + type_rows + "\n"
    with pytest.raises(ValueError, match=message):
        SpinCifReader(text)


def test_reader_rejects_mismatched_atom_type_columns(external_scif):
    text = external_scif + (
        "loop_\n_atom_type_symbol\nMn2+ Fe2+\nloop_\n_atom_type_oxidation_number\n2\n"
    )
    with pytest.raises(ValueError, match="equal-length loop columns"):
        SpinCifReader(text)


def test_reader_rejects_non_grammar_expressions(external_scif):
    # uvw values from the file must never reach code execution; anything
    # outside the linear-expression grammar is rejected while parsing.
    for payload in ("u.__class__,v,w", "2**u,v,w", "1j,v,w"):
        with pytest.raises(ValueError, match="Unsupported token"):
            SpinCifReader(external_scif.replace("u,v,w", payload, 1))


def test_reader_rejects_nonlinear_expressions(external_scif):
    # Reading coefficients off unit vectors would silently misread u*u as u
    # and u*v as 0; nonlinear or inhomogeneous input must be rejected.
    for payload in ("u*u,v,w", "u*v,v,w", "1/2,v,w", "1/u,v,w"):
        with pytest.raises(ValueError, match="not linear|Bad uvw"):
            SpinCifReader(external_scif.replace("u,v,w", payload, 1))


def test_linear_matrix_rejects_nonlinear_expressions_agreeing_at_sample_points():
    # Checking linearity by sampling is not enough: these agree with a linear
    # map at zero, at the unit vectors, and at any single further probe point,
    # yet decode to a different operation. Parentheses keep them out of the
    # CIF tokenizer, so the grammar is exercised directly.
    for expr in (
        "u*v*(u-0.375),v,w",  # vanishes wherever a probe is likely to look
        "u+u*(u-1)*(u-0.375),v,w",  # reads as u at 0, at e_u, and at the probe
        "u*u,v,w",
        "1/u,v,w",
    ):
        with pytest.raises(ValueError, match="not linear"):
            _linear_matrix(expr, "uvw")

    # Products and quotients by variable-free factors stay legal.
    assert np.allclose(_linear_matrix("2*u/4+0.5*v,v,w", "uvw")[0], [0.5, 0.5, 0.0])


def test_reader_rejects_degenerate_spin_frame(external_scif):
    with pytest.raises(ValueError, match="Degenerate spin-frame"):
        SpinCifReader(external_scif.replace("'a,b,c'", "'0,b,c'"))
    with pytest.raises(ValueError, match="not orthogonal"):
        SpinCifReader(external_scif.replace("'a,b,c'", "'a,a,c'"))


@pytest.mark.parametrize("from_file", [False, True])
def test_reader_spin_frame_uses_symprec(external_scif, tmp_path, from_file):
    text = external_scif.replace("'a,b,c'", "'a,b+0.0005*a,c'")
    source = text
    read = SpinCifReader
    if from_file:
        path = tmp_path / "frame.scif"
        path.write_text(text)
        source = str(path)
        read = SpinCifReader.from_file
    with pytest.raises(ValueError, match="not orthogonal"):
        read(source, symprec=1e-6)
    structure = read(source, symprec=1e-2).get_structure()
    assert len(structure) == 2
    assert np.allclose(structure.site_properties["magmom"], [[0, 0, 3], [0, 0, 3]])


@pytest.mark.parametrize("scale", [1e-200, 1e-10, 1e200])
def test_reader_normalizes_nonzero_spin_axes(external_scif, scale):
    text = external_scif.replace("'a,b,c'", f"'{scale:.200f}*a,b,c'")
    reader = SpinCifReader(text, symprec=1e-6)
    assert np.allclose(reader.spin_basis, np.eye(3), atol=1e-6, rtol=0)
    assert np.allclose(reader.get_structure().site_properties["magmom"], [[0, 0, 3], [0, 0, 3]])


@pytest.mark.parametrize("mixed", [False, True])
def test_reader_occupancy_uses_symprec_without_normalizing(external_scif, mixed):
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\n",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n",
    ).replace(
        "Mn1 Mn 0.25 0.0 0.0 2",
        "Mn1 Mn 0.25 0.0 0.0 2 0.5\nFe1 Fe 0.25 0.0 0.0 2 0.500000005"
        if mixed
        else "Mn1 Mn 0.25 0.0 0.0 2 1.000000005",
    )
    with pytest.raises(ValueError, match="occupanc"):
        SpinCifReader(text, symprec=1e-10).get_structure()
    structure = SpinCifReader(text, symprec=1e-6).get_structure()
    expected = {"Mn": 0.5, "Fe": 0.500000005} if mixed else {"Mn": 1.000000005}
    assert len(structure) == 2
    assert all(site.species.as_dict() == expected for site in structure)


def test_reader_occupancy_respects_structure_limit(external_scif):
    occupancy = 0.5 + 2 * Composition.amount_tolerance
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\n",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n",
    ).replace(
        "Mn1 Mn 0.25 0.0 0.0 2",
        f"Mn1 Mn 0.25 0.0 0.0 2 0.5\nFe1 Fe 0.25 0.0 0.0 2 {occupancy}",
    )
    with pytest.raises(ValueError, match="occupanc"):
        SpinCifReader(text, symprec=1e-2).get_structure()


def test_reader_orbit_positions_use_absolute_symprec(external_scif):
    text = external_scif.replace("Mn1 Mn 0.25", "Mn1 Mn 0.000002")
    structure = SpinCifReader(text, symprec=1e-7).get_structure()
    assert sorted(site.frac_coords[0] for site in structure) == [0.000002, 0.999998]
    with pytest.raises(ValueError, match="multiplicity"):
        SpinCifReader(text, symprec=1e-3).get_structure()


def test_reader_constituent_positions_use_absolute_symprec(external_scif):
    text = external_scif.replace("2 -x,-y,z,+1 u,v,w\n", "").replace(
        "Mn1 Mn 0.25 0.0 0.0 2",
        "Mn1 Mn 0.000002 0.0 0.0 1\nFe1 Fe 0.999998 0.0 0.0 1",
    )
    structure = SpinCifReader(text, symprec=1e-7).get_structure()
    assert {site.species_string: site.frac_coords[0] for site in structure} == {
        "Mn": 0.000002,
        "Fe": 0.999998,
    }
    with pytest.raises(ValueError, match="occupanc"):
        SpinCifReader(text, symprec=1e-3).get_structure()


def test_reader_legacy_cartn_frame_fallback(enumerated):
    # Files written before the P_abc switch declare the frame as
    # spinframe_orientation_cartn [0 0 0]; both routes must decode to the
    # same crystal-Cartesian spin basis and identical moments.
    sog, nssg, sas = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1)[0]
    moments = sas._sample_magnetic_moments(rng=np.random.default_rng(0))
    text = str(SpinCifWriter(sas, nssg, moments, spin_only_group=sog))
    legacy = re.sub(
        r"_space_group_spin\.transform_spinframe_P_abc\s+\S+",
        "_space_group_spin.spinframe_orientation_cartn  [0 0 0]",
        text,
    )
    assert legacy != text
    reader = SpinCifReader(legacy)
    assert np.allclose(
        reader.spin_basis, _crystal_cartesian_frame(np.array(reader.lattice.matrix)).T, atol=1e-8
    )
    reference = SpinCifReader(text).get_structure()
    structure = reader.get_structure()
    assert np.allclose(structure.frac_coords, reference.frac_coords, atol=1e-8)
    assert np.allclose(
        structure.site_properties["magmom"], reference.site_properties["magmom"], atol=1e-6
    )
