"""Round trips through the spinCIF writer and reader.

Each case writes a spin-symmetry-adapted structure with SpinCifWriter, reads
it back with SpinCifReader, and compares the reconstructed full magnetic
structure against the one the writer was given.
"""

from __future__ import annotations

import numpy as np
import pytest
from spinspg.spin import SpinOnlyGroupType

from spinforge.scif import SPINCIF_REVISION, SpinCifWriter


def test_roundtrip_and_spin_only_tags(roundtrip_case, assert_roundtrip):
    kind, sog, nssg, sas, moments = roundtrip_case
    text, reader = assert_roundtrip(sas, nssg, sog, moments)

    assert (reader.collinear_direction_xyz != ".") == (kind == "collinear")
    assert (reader.coplanar_perp_uvw != ".") == (kind == "coplanar")
    assert "_atom_site_spin_moment.symmform_uvw" in text
    assert SPINCIF_REVISION in text


def test_coplanar_ansatz_with_collinear_sample_keeps_model_tags(
    enumerated, moment_rank, assert_roundtrip
):
    # A coplanar-ansatz SSG may sample genuinely collinear moments; the
    # spin-only tags follow the SSG model (SpinOnlyGroup), not the sample,
    # so the file still declares coplanar_perp_uvw. This also round-trips a
    # coplanar operation set distinct from the rank-2 fixture case.
    for sog, nssg, sas in enumerated("Mn3Sn", SpinOnlyGroupType.COPLANAR, 1):
        moments = sas._sample_magnetic_moments(rng=np.random.default_rng(1))
        if moment_rank(moments) == 1:
            break
    else:
        pytest.skip("No coplanar-ansatz candidate sampled collinear moments")
    _, reader = assert_roundtrip(sas, nssg, sog, moments)

    assert reader.coplanar_perp_uvw != "."
    assert reader.collinear_direction_xyz == "."


def test_spin_lattice_loop_holds_antitranslation_MnTe_k2(enumerated, assert_roundtrip):
    results = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 2)
    for sog, nssg, sas in results:
        if nssg.spin_translation_coset.size > 1:
            break
    else:
        pytest.skip("No k=2 SSG with a nontrivial spin translation coset")
    moments = sas._sample_magnetic_moments(rng=np.random.default_rng(3))
    _, reader = assert_roundtrip(sas, nssg, sog, moments)

    assert len(reader.spin_lattice) == nssg.spin_translation_coset.size
    assert any(time_reversal == -1 for _, _, time_reversal, _ in reader.spin_lattice)


def test_oriented_roundtrip_MnTe(enumerated, assert_roundtrip):
    sog, nssg, sas = enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1)[0]
    oriented = sas.generate_oriented(sog, nssg, rng=np.random.default_rng(4))
    selected = next(
        (
            (structure, msg)
            for structure, msg in oriented
            if not np.allclose(msg.Q, np.eye(3), atol=1e-8)
        ),
        None,
    )
    assert selected is not None
    structure, msg = selected
    assert not np.allclose(msg.Q, np.eye(3), atol=1e-8)
    moments = np.array([np.asarray(m) for m in structure.site_properties["magmom"]])
    writer = SpinCifWriter.from_oriented(sas, nssg, structure, msg, spin_only_group=sog)
    assert_roundtrip(sas, nssg, sog, moments, writer=writer)
