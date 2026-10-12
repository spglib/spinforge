"""Tests for spinforge.mcif.MCifWriter and pymatgen's mcif reading.

The fixture is a synthetic BNS 46.247 (I_a m a 2) magnetic cell with
arbitrary lattice parameters and site coordinates. Its hexagonal-axis
operations mix unequal a/b axes, preserving the regression contract from
pymatgen-core #76 without redistributing an experimental structure record.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Composition, DummySpecies, Lattice, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.cif import CifFile, CifParser

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
    data = next(iter(CifFile.from_str(text).data.values())).data
    assert 0.8 in map(float, data["_atom_site_occupancy"])

    reloaded = Structure.from_file(str(out))
    mn_occupancies = sorted(
        site.species.get_el_amt_dict()["Mn"]
        for site in reloaded
        if "Mn" in site.species.get_el_amt_dict()
    )
    # one partial (0.8) Mn site and two fully occupied Mn sites survive distinctly
    assert mn_occupancies.count(1.0) == 2
    assert sum(abs(occ - 0.8) < 1e-6 for occ in mn_occupancies) == 1


def test_mixed_site_retains_minority_species_and_occupied_atom_moment() -> None:
    mixed = Structure(
        Lattice.hexagonal(4.0, 6.0),
        [{"Fe2+": 0.2, "Zn2+": 0.7}, {"Zn2+": 0.7, "Fe2+": 0.2}],
        [[0, 0, 0], [0, 0, 0.5]],
        site_properties={"magmom": [[1, 2, 3], [-1, -2, -3]]},
    )
    text = str(MCifWriter(mixed, magnetic_species=[("Fe2+",), ("Fe2+",)]))
    images = _expand_constituent_rows(text)
    assert len(images) == 4
    for symbol, occupancy, position, moment, selected in images:
        is_iron = symbol == "Fe2+"
        assert symbol in {"Fe2+", "Zn2+"}
        assert occupancy == (0.2 if is_iron else 0.7)
        assert selected == is_iron
        expected = np.array([1, 2, 3]) * (1 if abs(position[2]) < 1e-5 else -1)
        assert np.allclose(moment, expected if is_iron else 0, atol=2e-6)
    # Reordering the composition mapping cannot alter the exported constituent model.
    block = next(iter(CifFile.from_str(text).data.values())).data
    assert Composition(block["_chemical_formula_sum"]).as_dict() == {"Fe": 0.4, "Zn": 1.4}
    assert len(block["_atom_site_label"]) == 2
    assert len(block["_atom_site_moment.label"]) == 1


@pytest.mark.parametrize("moment", [[0, 0, 2], [0, 0, 0]])
def test_symmetry_preserves_constituent_selection_even_for_zero_moments(moment) -> None:
    structure = Structure(
        Lattice.cubic(4),
        [{"Fe": 0.5, "Mn": 0.5}] * 2,
        [[0, 0, 0], [0, 0, 0.5]],
        site_properties={"magmom": [moment, moment]},
    )
    text = str(MCifWriter(structure, magnetic_species=[("Fe",), ("Mn",)]))
    block = next(iter(CifFile.from_str(text).data.values())).data
    assert len(block["_atom_site_label"]) == 4
    assert len(block["_atom_site_moment.label"]) == 2
    images = _expand_constituent_rows(text)
    assert len(images) == 4
    for symbol, _, position, vector, selected in images:
        expected_selected = symbol == ("Fe" if abs(position[2]) < 1e-5 else "Mn")
        assert selected == expected_selected
        assert np.allclose(vector, moment if selected else 0)


def test_oxidation_states_and_neutral_elements_remain_distinct() -> None:
    structure = Structure(
        Lattice.cubic(4),
        [{"Fe": 0.25, "Fe0+": 0.25, "Fe2+": 0.5}],
        [[0, 0, 0]],
        site_properties={"magmom": [[0, 0, 2]]},
    )
    text = str(MCifWriter(structure, magnetic_species=[("Fe2+",)]))
    data = next(iter(CifFile.from_str(text).data.values())).data
    assert set(data["_atom_site_type_symbol"]) == {"Fe", "Fe0+", "Fe2+"}
    assert dict(zip(data["_atom_type_symbol"], data["_atom_type_oxidation_number"])) == {
        "Fe0+": "0",
        "Fe2+": "2",
    }
    assert len(data["_atom_site_moment.label"]) == 1


def test_ionic_single_species_roundtrips_with_pymatgen() -> None:
    structure = Structure(
        Lattice.cubic(4),
        [{"Fe2+": 0.7}],
        [[0, 0, 0]],
        site_properties={"magmom": [[0, 0, 3]]},
    )
    parser = CifParser.from_str(str(MCifWriter(structure)))
    (data,) = parser.as_dict().values()
    assert Composition(data["_chemical_formula_sum"]) == Composition({"Fe": 0.7})
    (reloaded,) = parser.parse_structures(primitive=False)
    (site,) = reloaded
    assert site.species == Composition({"Fe2+": 0.7})
    assert np.allclose(Magmom(site.properties["magmom"]).global_moment, [0, 0, 3])


def test_oxidation_state_decorations_constrain_symmetry() -> None:
    structure = Structure(
        Lattice.cubic(4),
        ["Fe2+", "Fe3+"],
        [[0, 0, 0], [0, 0, 0.5]],
        site_properties={"magmom": [[0, 0, 2], [0, 0, 2]]},
    )
    images = _expand_constituent_rows(str(MCifWriter(structure)))
    assert len(images) == 2
    assert {symbol for symbol, *_ in images} == {"Fe2+", "Fe3+"}


def test_small_minority_occupancy_is_not_rounded_to_zero() -> None:
    structure = Structure(
        Lattice.cubic(4),
        [{"Fe": 1e-7, "Zn": 0.9}],
        [[0, 0, 0]],
        site_properties={"magmom": [[0, 0, 2]]},
    )
    images = _expand_constituent_rows(str(MCifWriter(structure, magnetic_species=[("Fe",)])))
    (iron,) = (row for row in images if row[0] == "Fe")
    assert iron[1] == 1e-7
    assert np.allclose(iron[3], [0, 0, 2])


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ([], "one selection per site"),
        ([("Mn",)], "absent species"),
        ([("Fe", "Fe")], "duplicate species"),
        ([()], "nonzero moment"),
        (["Fe"], "sequence of species"),
    ],
)
def test_rejects_invalid_magnetic_selection(selection, message) -> None:
    structure = Structure(
        Lattice.cubic(4), ["Fe"], [[0, 0, 0]], site_properties={"magmom": [[0, 0, 2]]}
    )
    with pytest.raises(ValueError, match=message):
        MCifWriter(structure, magnetic_species=selection)


@pytest.mark.parametrize(
    ("composition", "exception", "message"),
    [
        (Composition({"Fe": -0.2}, allow_negative=True), ValueError, "positive"),
        (Composition({"Fe": np.nan}), ValueError, "at least one species"),
        (Composition({DummySpecies("X"): 1}), TypeError, "Unsupported species"),
        (Composition({Species("Fe", None): 1}), ValueError, "Unsupported species"),
        (Composition({Species("Fe", 2, spin=1): 1}), ValueError, "Unsupported species"),
    ],
)
def test_rejects_invalid_constituents(composition, exception, message) -> None:
    structure = Structure(
        Lattice.cubic(4), [composition], [[0, 0, 0]], site_properties={"magmom": [[0, 0, 2]]}
    )
    with pytest.raises(exception, match=message):
        MCifWriter(structure)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_rejects_nonfinite_moments(value) -> None:
    structure = Structure(
        Lattice.cubic(4), ["Fe"], [[0, 0, 0]], site_properties={"magmom": [[0, 0, value]]}
    )
    with pytest.raises(ValueError, match="finite"):
        MCifWriter(structure)


def _expand_constituent_rows(text: str) -> list[tuple[str, float, np.ndarray, np.ndarray, bool]]:
    """Read CIF tokens and expand each constituent independently of the writer."""
    parser = CifParser.from_str(text)
    block = next(iter(CifFile.from_str(text).data.values()))
    data = block.data
    lattice = Lattice.from_parameters(
        *[float(data[f"_cell_length_{axis}"]) for axis in ("a", "b", "c")],
        *[float(data[f"_cell_angle_{axis}"]) for axis in ("alpha", "beta", "gamma")],
    )
    moments = dict(
        zip(
            data.get("_atom_site_moment.label", []),
            zip(*[data.get(f"_atom_site_moment.crystalaxis_{axis}", []) for axis in "xyz"]),
        )
    )
    images = []
    for label, symbol, x, y, z, occupancy in zip(
        *[
            data[name]
            for name in (
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_occupancy",
            )
        ]
    ):
        position = np.array([x, y, z], dtype=float)
        mx, my, mz = moments.get(label, (0, 0, 0))
        moment = Magmom.from_moment_relative_to_crystal_axes(
            (float(mx), float(my), float(mz)), lattice
        ).global_moment
        for operation in parser.get_magsymops(block):
            new_position = operation.operate(position) % 1
            cartesian_rotation = (
                lattice.matrix.T @ operation.rotation_matrix @ np.linalg.inv(lattice.matrix.T)
            )
            new_moment = (
                np.linalg.det(cartesian_rotation)
                * operation.time_reversal
                * cartesian_rotation
                @ moment
            )
            matches = [
                row
                for row in images
                if row[0] == symbol
                and np.allclose(row[2] - new_position, np.rint(row[2] - new_position), atol=1e-5)
            ]
            if matches:
                (match,) = matches
                assert match[1] == float(occupancy)
                assert np.allclose(match[3], new_moment, atol=2e-6)
                assert match[4] == (label in moments)
            else:
                images.append(
                    (symbol, float(occupancy), new_position, new_moment, label in moments)
                )
    return images
