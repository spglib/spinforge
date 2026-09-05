"""Writer-side behaviour: interface, input rejection, and value formatting."""

from __future__ import annotations

import numpy as np
import pytest
from moyopy import SpaceGroupType
from spinspg.spin import SpinOnlyGroupType

from spinforge.scif import SpinCifReader, SpinCifWriter
from spinforge.scif._operations import _snap_translation, _supercell_translation_denominator


def test_writer_interface(tmp_path, enumerated):
    sog, nssg, sas = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1)[0]
    moments = sas._sample_magnetic_moments(rng=np.random.default_rng(5))

    writer = SpinCifWriter(
        sas, nssg, moments, spin_only_group=sog, parent_space_group_type=SpaceGroupType(194)
    )
    out = tmp_path / "written.scif"
    writer.write_file(str(out))
    assert out.read_text() == str(writer)
    assert '_parent_space_group.name_H-M_alt  "P6_3/mmc"' in str(writer)
    assert "_parent_space_group.IT_number  194" in str(writer)
    assert len(SpinCifReader.from_file(str(out)).sites) == 2  # Mn and Te orbit reps

    with pytest.raises(ValueError, match="does not match"):
        SpinCifWriter(sas, nssg, np.zeros((1, 3)), spin_only_group=sog)


def test_oriented_moments_in_plain_constructor_rejected(enumerated):
    # Oriented moments are rotated by msg.Q; passing them to the plain
    # constructor instead of from_oriented must fail fast (the moments no
    # longer lie in the unrotated SSA basis span) instead of writing
    # symmetry loops that do not reconstruct them.
    sog, nssg, sas = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1)[0]
    oriented = sas.generate_oriented(sog, nssg, rng=np.random.default_rng(4))
    structure, msg = oriented[0]
    moments = np.array([np.asarray(m) for m in structure.site_properties["magmom"]])
    if np.allclose(msg.Q, np.eye(3), atol=1e-8):
        pytest.skip("Oriented rotation happens to be the identity")
    with pytest.raises(ValueError, match="combination of the SSA basis"):
        SpinCifWriter(sas, nssg, moments, spin_only_group=sog)


def test_site_occupancies_are_written_and_checked(enumerated):
    sog, nssg, sas = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1)[0]
    moments = sas._sample_magnetic_moments(rng=np.random.default_rng(5))
    occupancies = [0.5 if z == 25 else 1.0 for z in sas.supercell.numbers]

    text = str(
        SpinCifWriter(sas, nssg, moments, spin_only_group=sog, site_occupancies=occupancies)
    )
    mn_rows = [line for line in text.splitlines() if line.startswith("Mn1 Mn ")]
    assert len(mn_rows) == 1 and "0.500000" in mn_rows[0]
    te_rows = [line for line in text.splitlines() if line.startswith("Te1 Te ")]
    assert len(te_rows) == 1 and "1.000000" in te_rows[0]

    with pytest.raises(ValueError, match="does not match"):
        SpinCifWriter(sas, nssg, moments, spin_only_group=sog, site_occupancies=[0.5])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        SpinCifWriter(
            sas, nssg, moments, spin_only_group=sog, site_occupancies=[1.5] * len(occupancies)
        )


def test_orbit_inconsistent_occupancies_are_rejected(enumerated):
    # In the k=2 antitranslation supercell the two Mn sites form one orbit;
    # giving them different occupancies cannot be represented by a single
    # asymmetric-unit row and must fail instead of writing a wrong file.
    sog, nssg, sas = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 2)[0]
    moments = sas._sample_magnetic_moments(rng=np.random.default_rng(5))
    occupancies = []
    seen_mn = False
    for z in sas.supercell.numbers:
        if z == 25 and not seen_mn:
            occupancies.append(0.5)
            seen_mn = True
        else:
            occupancies.append(1.0)
    with pytest.raises(ValueError, match="not invariant"):
        SpinCifWriter(sas, nssg, moments, spin_only_group=sog, site_occupancies=occupancies)


def test_snap_translation_respects_supercell_grid():
    # A primitive translation (0, 0, 1) in a diag(1, 1, 5) supercell becomes
    # 1/5, which is off the primitive 1/24 grid; the denominator must scale
    # with |det T| so the value is snapped exactly, not corrupted or missed.
    t_mat = np.diag([1.0, 1.0, 5.0])
    denom = _supercell_translation_denominator(t_mat)
    assert denom == 120

    noisy = np.array([0.0, 0.0, 0.2 + 1e-9])
    assert np.array_equal(_snap_translation(noisy, denom=denom, atol=1e-5), [0.0, 0.0, 0.2])


def test_weak_moments_are_not_dropped(enumerated):
    # Moments below the old norm-atol cutoff but above the print resolution
    # must still be written; otherwise weak magnetic structures silently
    # round-trip as nonmagnetic (the reader treats omitted rows as zero).
    sog, nssg, sas = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1)[0]
    moments = sas._sample_magnetic_moments(rng=np.random.default_rng(0))
    weak = moments * (4e-6 / np.max(np.abs(moments)))
    text = str(SpinCifWriter(sas, nssg, weak, spin_only_group=sog))
    assert "_atom_site_spin_moment.label" in text
    structure = SpinCifReader(text).get_structure()
    assert np.max(np.abs(structure.site_properties["magmom"])) > 0


def test_writer_declares_p_abc_frame(enumerated):
    # The frame must go out via transform_spinframe_P_abc (the tag the
    # COMCIFS reference files and VESTA understand), also on a
    # non-orthogonal (hexagonal) cell.
    sog, nssg, sas = enumerated("Mn3Sn", SpinOnlyGroupType.COPLANAR, 1)[0]
    moments = sas._sample_magnetic_moments(rng=np.random.default_rng(6))
    text = str(SpinCifWriter(sas, nssg, moments, spin_only_group=sog))
    assert "_space_group_spin.transform_spinframe_P_abc" in text
    assert "spinframe_orientation_cartn" not in text
