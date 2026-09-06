"""Tests for spinforge.mcif.MCifWriter and pymatgen's mcif reading.

The fixture is a synthetic BNS 46.247 (I_a m a 2) magnetic cell with
arbitrary lattice parameters and site coordinates. Its hexagonal-axis
operations mix unequal a/b axes, preserving the regression contract from
pymatgen-core #76 without redistributing an experimental structure record.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Structure

from spinforge.mcif import MCifWriter
from spinforge.testing import get_magnetic_space_group_type

SYNTHETIC_AXIS_MIXING_MCIF = """
data_synthetic_axis_mixing
_space_group_magn.number_BNS  46.247
_cell_length_a                 10.00000
_cell_length_b                  5.00000
_cell_length_c                  7.00000
_cell_angle_alpha              90.00
_cell_angle_beta               90.00
_cell_angle_gamma              120.00

loop_
_space_group_symop_magn_operation.id
_space_group_symop_magn_operation.xyz
1 x,y,z,+1
2 -x,-2x+y,-z,+1
3 x,y,-z+1/2,+1
4 -x,-2x+y,z+1/2,+1

loop_
_space_group_symop_magn_centering.id
_space_group_symop_magn_centering.xyz
1 x,y,z,+1
2 x+1/2,y,z+1/2,+1
3 x,y,z+1/2,-1
4 x+1/2,y,z,-1

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Mn1 Mn 0.13700 0.00000 0.00000 1

loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
_atom_site_moment.symmform
_atom_site_moment.magnitude
Mn1 2.0 0.0 0.0 mx,my,0 2.0
"""


@pytest.fixture
def mcif_path(tmp_path):
    path = tmp_path / "synthetic_axis_mixing.mcif"
    path.write_text(SYNTHETIC_AXIS_MIXING_MCIF)
    return str(path)


def test_orbit_expansion_counts(mcif_path: str) -> None:
    structure = Structure.from_file(mcif_path)

    symbols = [site.specie.symbol for site in structure]
    assert len(structure) == 8
    assert symbols == ["Mn"] * 8


def test_moment_magnitudes_are_preserved(mcif_path: str) -> None:
    structure = Structure.from_file(mcif_path)

    moments = np.array([site.properties["magmom"].moment for site in structure])
    assert np.allclose(np.linalg.norm(moments, axis=1), 2.0)


def test_recovers_expected_magnetic_space_group(mcif_path: str) -> None:
    structure = Structure.from_file(mcif_path)

    assert get_magnetic_space_group_type(structure).bns_number == "46.247"


def test_save_writes_symmetrized_mcif_and_roundtrips(mcif_path: str, tmp_path) -> None:
    # Hexagonal axis-mixing case: the writer must emit magnetic operations + an
    # asymmetric unit (not a P1 dump) and round-trip back through the loader.
    original = Structure.from_file(mcif_path)
    out = tmp_path / "written.mcif"
    MCifWriter(original).write_file(str(out))
    text = out.read_text()

    assert "_space_group_symop_magn_operation.xyz" in text
    assert "_space_group_symop_magn_centering.xyz" in text
    # Symmetrized, not P1: fewer asymmetric-unit atoms than the 8-atom full cell.
    assert 0 < text.count("\nMn") < 8

    reloaded = Structure.from_file(str(out))
    assert len(reloaded) == len(original)
    assert get_magnetic_space_group_type(reloaded).bns_number == "46.247"

    moments = np.array([site.properties["magmom"].moment for site in reloaded])
    assert np.allclose(np.linalg.norm(moments, axis=1), 2.0)


def test_mcif_writer_str_matches_write_file(mcif_path: str, tmp_path) -> None:
    # pymatgen-style interface: str(writer) and writer.write_file() agree.
    structure = Structure.from_file(mcif_path)
    writer = MCifWriter(structure)
    out = tmp_path / "writer.mcif"
    writer.write_file(str(out))
    assert out.read_text() == str(writer)


def test_save_requires_magmom_property() -> None:
    from pymatgen.core import Lattice, Structure

    bare = Structure(Lattice.cubic(4.0), ["Fe"], [[0, 0, 0]])
    with pytest.raises(ValueError, match="magmom"):
        MCifWriter(bare)


def test_partial_occupancy_roundtrips(tmp_path) -> None:
    # A partially occupied magnetic site must keep its occupancy through
    # save -> load, and must not be folded into the fully occupied Mn orbit.
    from pymatgen.core import Lattice, Structure
    from pymatgen.electronic_structure.core import Magmom

    lattice = Lattice.from_parameters(4.0, 4.0, 6.0, 90, 90, 120)
    structure = Structure(
        lattice,
        [{"Mn": 1.0}, {"Mn": 1.0}, {"Mn": 0.8}, "Ge"],
        [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.6]],
        site_properties={
            "magmom": [Magmom([0, 0, 3.0]), Magmom([0, 0, 3.0]), Magmom([0, 0, 2.0]), Magmom(0)],
        },
    )
    out = tmp_path / "partial.mcif"
    MCifWriter(structure, symprec=1e-3, mag_symprec=1e-2).write_file(str(out))

    text = out.read_text()
    assert "_atom_site_occupancy" in text
    assert "0.800000" in text  # the partial Mn occupancy survived into the asym unit

    reloaded = Structure.from_file(str(out))
    mn_occupancies = sorted(
        site.species.get_el_amt_dict()["Mn"]
        for site in reloaded
        if "Mn" in site.species.get_el_amt_dict()
    )
    # one partial (0.8) Mn site and two fully occupied Mn sites survive distinctly
    assert mn_occupancies.count(1.0) == 2
    assert sum(abs(occ - 0.8) < 1e-6 for occ in mn_occupancies) == 1


def test_save_rejects_mixed_solid_solution() -> None:
    from pymatgen.core import Lattice, Structure
    from pymatgen.electronic_structure.core import Magmom

    mixed = Structure(
        Lattice.cubic(4.0),
        [{"Mn": 0.5, "Ge": 0.5}],
        [[0, 0, 0]],
        site_properties={"magmom": [Magmom([0, 0, 2.0])]},
    )
    with pytest.raises(NotImplementedError, match="single-species"):
        MCifWriter(mixed)
