"""Shared fixtures for the spinCIF writer/reader tests."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from spgrep.utils import is_integer_array
from spinspg.spin import SpinOnlyGroupType

from spinforge.configuration import SSAGenerator
from spinforge.scif import SpinCifReader, SpinCifWriter
from spinforge.scif._operations import _crystal_cartesian_frame
from spinforge.testing import (
    load_prim_Mn3Sn,
    load_prim_MnTe,
    load_prim_pyrochlore,
    site_indices,
)

_LOADERS = {
    "MnTe": (load_prim_MnTe, 25),
    "Mn3Sn": (load_prim_Mn3Sn, 25),
    "pyrochlore": (load_prim_pyrochlore, 76),
}


@cache
def _enumerated(material: str, sog_type: SpinOnlyGroupType, k_index: int, max_depth=None):
    loader, z = _LOADERS[material]
    prim = loader()
    gen = SSAGenerator(prim_cell=prim, magnetic_site_indices=site_indices(prim, z))
    results = gen.enumerate(spin_only_group_type=sog_type, k_index=k_index, max_depth=max_depth)
    if len(results) == 0:
        raise ValueError(f"No SSG enumerated for {material}")
    return results


def _moment_rank(moments: np.ndarray, atol: float = 1e-6) -> int:
    nonzero = moments[np.linalg.norm(moments, axis=1) > atol]
    singular_values = np.linalg.svd(nonzero, compute_uv=False)
    return int(np.sum(singular_values > atol * singular_values[0]))


def _assert_roundtrip(sas, nssg, sog, moments, writer=None):
    if writer is None:
        writer = SpinCifWriter(sas, nssg, moments, spin_only_group=sog)
    text = str(writer)
    reader = SpinCifReader(text)

    # Time-reversal flag must equal the determinant of the spin operation.
    for _, _, time_reversal, u in reader.expanded_operations():
        assert np.isclose(np.linalg.det(u), time_reversal, atol=1e-5)

    structure = reader.get_structure()
    expected_positions = np.asarray(sas.supercell.positions)
    assert len(structure) == len(expected_positions)

    # Compare moments in the spin basis, each side via its own lattice frame.
    expected_moments = np.asarray(moments) @ _crystal_cartesian_frame(
        np.asarray(sas.supercell.basis)
    )
    read_moments = np.array(structure.site_properties["magmom"]) @ _crystal_cartesian_frame(
        np.array(structure.lattice.matrix)
    )

    matched = set()
    for position, moment in zip(structure.frac_coords, read_moments):
        for i in range(len(expected_positions)):
            if i in matched:
                continue
            if is_integer_array(position - expected_positions[i], atol=1e-4):
                assert np.allclose(moment, expected_moments[i], atol=1e-5)
                matched.add(i)
                break
        else:
            raise AssertionError(f"Read site {position} not in the original structure")
    return text, reader


@pytest.fixture
def enumerated():
    """The (cached) SSG enumeration helper: ``(material, sog_type, k_index)``."""
    return _enumerated


@pytest.fixture
def moment_rank():
    """Rank of a sampled moment set (1 collinear, 2 coplanar, 3 noncoplanar)."""
    return _moment_rank


@pytest.fixture
def assert_roundtrip():
    """Write, read back, and compare against the structure the writer was given."""
    return _assert_roundtrip


@pytest.fixture(params=["collinear", "coplanar", "noncoplanar"])
def roundtrip_case(request):
    """(kind, sog, nssg, sas, moments) with moments genuinely of the given kind."""
    kind = request.param
    if kind == "collinear":
        sog, nssg, sas = _enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1)[0]
        moments = sas._sample_magnetic_moments(rng=np.random.default_rng(0))
    elif kind == "coplanar":
        # A coplanar-ansatz SSG may still sample a collinear structure; pick
        # the first candidate whose sampled moments are genuinely rank 2.
        for sog, nssg, sas in _enumerated("Mn3Sn", SpinOnlyGroupType.COPLANAR, 1):
            moments = sas._sample_magnetic_moments(rng=np.random.default_rng(1))
            if _moment_rank(moments) == 2:
                break
        else:
            raise ValueError("No coplanar sample found")
    else:
        # The all-in-all-out configuration: dim=1, genuinely 3D moments.
        results = _enumerated("pyrochlore", SpinOnlyGroupType.NONCOPLANAR, 1, 0)
        sog, nssg, sas = next(entry for entry in results if entry[2].dim == 1)
        moments = sas._sample_magnetic_moments(rng=np.random.default_rng(2))
    return kind, sog, nssg, sas, moments


_EXTERNAL_SCIF = """data_external
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_spin.transform_spinframe_P_abc 'a,b,c'
_space_group_spin.collinear_direction_xyz '0,0,1'
_space_group_spin.coplanar_perp_uvw .
loop_
_space_group_symop_spin_operation.id
_space_group_symop_spin_operation.xyzt
_space_group_symop_spin_operation.uvw
1 x,y,z,+1 u,v,w
2 -x,-y,z,+1 u,v,w
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_symmetry_multiplicity
Mn1 Mn 0.25 0.0 0.0 2
loop_
_atom_site_spin_moment.label
_atom_site_spin_moment.axis_u
_atom_site_spin_moment.axis_v
_atom_site_spin_moment.axis_w
_atom_site_spin_moment.symmform_uvw
_atom_site_spin_moment.symmform_rel_uvw
_atom_site_spin_moment.magnitude
Mn1 0.0 0.0 3.0 0,0,w 0,0,w 3.0
"""


@pytest.fixture
def external_scif():
    """External-style file: an id column in the operation loop and extra
    symmform_rel_uvw/magnitude columns in the moment loop, as in the
    FINDSPINGROUP reference files; column access must be name-based."""
    return _EXTERNAL_SCIF
