"""Reader-side behaviour: external files, grammar limits, and legacy tags."""

from __future__ import annotations

import re

import numpy as np
import pytest
from pymatgen.core import Element, Species
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
    "short_column",
    ["label", "type_symbol", "fract_x", "fract_y", "fract_z", "symmetry_multiplicity"],
)
def test_reader_rejects_mismatched_atom_site_columns(external_scif, short_column):
    columns = {
        "label": ["Mn1", "Mn2"],
        "type_symbol": ["Mn", "Mn"],
        "fract_x": ["0.25", "0.75"],
        "fract_y": ["0", "0"],
        "fract_z": ["0", "0"],
        "symmetry_multiplicity": ["2", "2"],
    }
    columns[short_column].pop()
    before, rest = external_scif.split("loop_\n_atom_site_label")
    _, after = rest.split("loop_\n_atom_site_spin_moment.label")
    text = (
        before
        + "".join(
            f"loop_\n_atom_site_{column}\n{' '.join(values)}\n"
            for column, values in columns.items()
        )
        + "loop_\n_atom_site_spin_moment.label"
        + after
    )
    with pytest.raises(ValueError, match="equal-length loop columns"):
        SpinCifReader(text)


@pytest.mark.parametrize("category", ["operation", "lattice"])
@pytest.mark.parametrize("missing_position", [False, True])
def test_reader_rejects_mismatched_symmetry_columns(external_scif, category, missing_position):
    columns = (
        f"loop_\n_space_group_symop_spin_{category}.xyzt\nx,y,z,+1 -x,-y,z,+1\n"
        f"loop_\n_space_group_symop_spin_{category}.uvw\nu,v,w\n"
    )
    if missing_position:
        columns = f"loop_\n_space_group_symop_spin_{category}.uvw\nu,v,w\n"
    if category == "operation":
        before, rest = external_scif.split("loop_\n_space_group_symop_spin_operation.id")
        _, after = rest.split("loop_\n_atom_site_label")
        text = before + columns + "loop_\n_atom_site_label" + after
    else:
        text = external_scif + columns
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


@pytest.mark.parametrize("scale", [1e-200, 1e-10, 1e200, 1e308])
def test_reader_normalizes_nonzero_spin_axes(external_scif, scale):
    text = external_scif.replace("'a,b,c'", f"'{scale:.200f}*a,b,c'")
    reader = SpinCifReader(text, symprec=1e-6)
    assert np.allclose(reader.spin_basis, np.eye(3), atol=1e-6, rtol=0)
    assert np.allclose(reader.get_structure().site_properties["magmom"], [[0, 0, 3], [0, 0, 3]])


@pytest.mark.parametrize("symprec", [1e-10, 1e-2])
def test_reader_occupancy_does_not_depend_on_geometric_tolerance(external_scif, symprec):
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\n",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n",
    ).replace(
        "Mn1 Mn 0.25 0.0 0.0 2",
        "Mn1 Mn 0.25 0.0 0.0 2 1.000000005",
    )
    structure = SpinCifReader(text, symprec=symprec).get_structure()
    assert len(structure) == 2
    assert all(dict(site.species.items()) == {Element("Mn"): 1.000000005} for site in structure)


@pytest.mark.parametrize("occupancy", [1.01, 1e-10])
def test_reader_rejects_occupancy_that_pymatgen_cannot_preserve(external_scif, occupancy):
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\n",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n",
    ).replace(
        "Mn1 Mn 0.25 0.0 0.0 2",
        f"Mn1 Mn 0.25 0.0 0.0 2 {occupancy}",
    )
    with pytest.raises(ValueError, match="occupanc"):
        SpinCifReader(text, symprec=1e-2).get_structure()


def test_reader_orbit_positions_use_absolute_symprec(external_scif):
    text = external_scif.replace("Mn1 Mn 0.25", "Mn1 Mn 0.000002")
    structure = SpinCifReader(text, symprec=1e-7).get_structure()
    assert sorted(site.frac_coords[0] for site in structure) == [0.000002, 0.999998]
    with pytest.raises(ValueError, match="multiplicity"):
        SpinCifReader(text, symprec=1e-3).get_structure()


@pytest.mark.parametrize("symprec", [1e-7, 1e-3])
def test_reader_preserves_distinct_constituent_rows(external_scif, symprec):
    text = external_scif.replace("2 -x,-y,z,+1 u,v,w\n", "").replace(
        "Mn1 Mn 0.25 0.0 0.0 2",
        "Mn1 Mn 0.000002 0.0 0.0 1\nFe1 Fe 0.999998 0.0 0.0 1",
    )
    structure = SpinCifReader(text, symprec=symprec).get_structure()
    assert {site.species_string: site.frac_coords[0] for site in structure} == {
        "Mn": 0.000002,
        "Fe": 0.999998,
    }
    assert structure.site_properties["magnetic"] == [True, False]


@pytest.mark.parametrize("from_file", [False, True])
def test_reader_frame_and_moment_tolerances_are_independent(external_scif, tmp_path, from_file):
    text = (
        external_scif.replace("'a,b,c'", "'a,b+0.0005*a,c'")
        .replace("Mn1 Mn 0.25 0.0 0.0 2", "Mn1 Mn 0.0 0.0 0.0 1")
        .replace("2 -x,-y,z,+1 u,v,w", "2 -x,-y,z,+1 u,v,1.0001*w")
    )
    source = text
    read = SpinCifReader
    if from_file:
        path = tmp_path / "tolerances.scif"
        path.write_text(text)
        source = str(path)
        read = SpinCifReader.from_file
    with pytest.raises(ValueError, match="not orthogonal"):
        read(source, symprec=1e-2, mag_symprec=1e-2, atol=1e-6)
    with pytest.raises(ValueError, match="Inconsistent moments"):
        read(source, symprec=1e-2, mag_symprec=1e-6, atol=1e-2).get_structure()
    with pytest.raises(ValueError, match="Inconsistent moments"):
        read(source, symprec=1e-6, atol=1e-2).get_structure()
    assert len(read(source, symprec=1e-2).get_structure()) == 1
    structure = read(source, symprec=1e-6, mag_symprec=1e-2, atol=1e-2).get_structure()
    assert len(structure) == 1
    np.testing.assert_allclose(structure.site_properties["magmom"], [[0, 0, 3]])


def test_reader_preserves_fractional_oxidation_identities(external_scif):
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\nMn1 Mn 0.25 0.0 0.0 2",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n"
        "Mn1 Mn0.3333333333333333+ 0.25 0.0 0.0 2 0.5\n"
        "Mn2 Mn0.3333333333333334+ 0.25 0.0 0.0 2 0.5",
    )
    structure = SpinCifReader(text).get_structure()
    assert len(structure) == 4
    assert {tuple(site.species.items()) for site in structure} == {
        ((Species("Mn", 0.3333333333333333), 0.5),),
        ((Species("Mn", 0.3333333333333334), 0.5),),
    }


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
