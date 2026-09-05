"""Reader-side behaviour: external files, grammar limits, and legacy tags."""

from __future__ import annotations

import re

import numpy as np
import pytest
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
