"""Constituent chemistry and explicit spin frames survive spinCIF expansion."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Composition, Structure, get_el_sp
from pymatgen.io.cif import CifFile
from spinspg.spin import SpinOnlyGroupType

from spinforge.mcif import MCifWriter
from spinforge.scif import SpinCifReader, SpinCifWriter


@pytest.fixture
def antitranslation_model(enumerated):
    return next(
        (sog, group, ssa)
        for sog, group, ssa in enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 2)
        if group.spin_translation_coset.size > 1 and ssa.dim == 1
    )


def _structure(ssa, magnetic_composition, selected, moments):
    selected = {get_el_sp(species) for species in selected}
    compositions, positions, row_moments, magnetic = [], [], [], []
    for number, position, moment in zip(
        ssa.supercell.numbers, ssa.supercell.positions, moments, strict=True
    ):
        composition = Composition(magnetic_composition if number == 25 else {"Te2-": 0.9})
        for species, occupancy in composition.items():
            is_magnetic = species in selected
            compositions.append({species: occupancy})
            positions.append(position)
            row_moments.append(moment if is_magnetic else np.zeros(3))
            magnetic.append(is_magnetic)
    return Structure(
        ssa.supercell.basis,
        compositions,
        positions,
        site_properties={"magmom": row_moments, "magnetic": magnetic},
    )


@pytest.mark.parametrize(
    ("magnetic_composition", "selected"),
    [
        ({"Mn2+": 0.6, "Zn2+": 0.2}, ("Mn2+",)),
        ({"Mn2+": 0.6, "Fe2+": 0.2}, ("Mn2+", "Fe2+")),
        ({"Mn2+": 0.6, "Mn3+": 0.2}, ("Mn2+",)),
        ({"Mn": 0.6, "Mn0+": 0.2}, ("Mn0+",)),
        ({"Mn2+": 0.6, "Zn2+": 1e-8}, ("Mn2+",)),
        (
            {"Mn0.3333333333333333+": 0.6, "Mn0.3333333333333334+": 0.2},
            ("Mn0.3333333333333333+",),
        ),
    ],
)
@pytest.mark.parametrize("improper", [False, True])
def test_mixed_antitranslation_with_explicit_frame(
    antitranslation_model, magnetic_composition, selected, improper
):
    sog, group, ssa = antitranslation_model
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    if improper:
        rotation = rotation @ np.diag([1.0, 1.0, -1.0])
    (intrinsic,) = ssa.magnetic_moments_basis
    moments = intrinsic @ rotation.T
    original = _structure(ssa, magnetic_composition, selected, moments)
    text = str(
        SpinCifWriter.from_structure(
            ssa,
            group,
            Structure.from_sites(list(reversed(original))),
            spin_only_group=sog,
            spin_rotation=rotation,
            symprec=1e-5,
        )
    )
    reader = SpinCifReader(text)
    structure = reader.get_structure()
    assert len(structure) == len(original)
    assert any(reversal == -1 for _, _, reversal, _ in reader.spin_lattice)
    embedding = np.linalg.inv(structure.lattice.matrix) @ ssa.supercell.basis
    for site in structure:
        matches = [
            expected
            for expected in original
            if np.allclose(
                site.frac_coords
                - expected.frac_coords
                - np.rint(site.frac_coords - expected.frac_coords),
                0,
                atol=1e-5,
                rtol=0,
            )
            and set(site.species) == set(expected.species)
        ]
        (expected,) = matches
        assert dict(site.species.items()) == dict(expected.species.items())
        assert site.properties["magnetic"] == expected.properties["magnetic"]
        np.testing.assert_allclose(
            site.properties["magmom"] @ embedding,
            expected.properties["magmom"],
            atol=1e-5,
            rtol=0,
        )

    direction = np.array([float(value) for value in reader.collinear_direction_xyz.split(",")])
    actual_axis = direction @ ssa.supercell.basis
    expected_axis = rotation @ sog.axis
    np.testing.assert_allclose(np.cross(actual_axis, expected_axis), 0, atol=1e-5)

    original_spin_basis = reader.spin_basis @ embedding
    operations, _ = group.get_full_operations_and_table()
    transformation = np.asarray(ssa.supercell.sublattice.transformation)
    inverse = np.linalg.inv(transformation)
    expanded = reader.expanded_operations()
    assert len(expanded) == len(operations.rotations)
    for w, t, u in zip(
        operations.rotations, operations.translations, operations.spin_rotations, strict=True
    ):
        expected_w = inverse @ w @ transformation
        expected_t = inverse @ t
        expected_u = original_spin_basis @ rotation @ u @ rotation.T @ original_spin_basis.T
        assert any(
            np.allclose(read_w, expected_w, atol=1e-6)
            and np.allclose(read_t - expected_t - np.rint(read_t - expected_t), 0, atol=1e-6)
            and np.allclose(read_u, expected_u, atol=1e-5)
            for read_w, read_t, _, read_u in expanded
        )


def test_selected_zero_moments_remain_selected(antitranslation_model):
    sog, group, ssa = antitranslation_model
    original = _structure(
        ssa,
        {"Mn2+": 0.6, "Zn2+": 0.2},
        ("Mn2+",),
        np.zeros((ssa.supercell.num_supercell_sites, 3)),
    )
    text = str(
        SpinCifWriter.from_structure(
            ssa,
            group,
            original,
            spin_only_group=sog,
            symprec=1e-5,
        )
    )
    (block,) = CifFile.from_str(text).data.values()
    moment_labels = set(block.data["_atom_site_spin_moment.label"])
    assert moment_labels == {
        label
        for label, symbol in zip(
            block.data["_atom_site_label"], block.data["_atom_site_type_symbol"], strict=True
        )
        if symbol == "Mn2+"
    }
    structure = SpinCifReader(text).get_structure()
    assert all(
        site.properties["magnetic"] == (get_el_sp("Mn2+") in site.species) for site in structure
    )
    np.testing.assert_array_equal(structure.site_properties["magmom"], 0)


@pytest.mark.parametrize("change_selection", [False, True])
def test_orbit_inconsistent_constituents_are_rejected(antitranslation_model, change_selection):
    sog, group, ssa = antitranslation_model
    (moments,) = ssa.magnetic_moments_basis
    structure = _structure(ssa, {"Mn2+": 0.6, "Fe2+": 0.2}, ("Mn2+",), moments)
    index = next(
        index for index, site in enumerate(structure) if get_el_sp("Mn2+") in site.species
    )
    site = structure[index]
    if change_selection:
        other = next(
            other
            for other in structure
            if get_el_sp("Fe2+") in other.species
            and np.array_equal(other.frac_coords, site.frac_coords)
        )
        other.properties["magnetic"] = True
        other.properties["magmom"] = site.properties["magmom"]
        site.properties["magnetic"] = False
        site.properties["magmom"] = np.zeros(3)
    else:
        structure.replace(index, {"Mn2+": 0.5}, properties=site.properties)
    with pytest.raises(ValueError, match="invariant under the SSG operations"):
        SpinCifWriter.from_structure(
            ssa,
            group,
            structure,
            spin_only_group=sog,
            symprec=1e-5,
        )


@pytest.mark.parametrize("rotation", [np.eye(2), np.ones((3, 3)), np.eye(3) * np.nan])
def test_invalid_explicit_frames_are_rejected(antitranslation_model, rotation):
    sog, group, ssa = antitranslation_model
    (moments,) = ssa.magnetic_moments_basis
    with pytest.raises(ValueError, match="finite orthogonal 3 by 3"):
        SpinCifWriter(ssa, group, moments, spin_only_group=sog, spin_rotation=rotation)


def test_reader_preserves_different_coincident_constituent_moments(external_scif):
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\nMn1 Mn 0.25 0.0 0.0 2",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n"
        "Mn1 Mn 0.25 0.0 0.0 2 0.5\nFe1 Fe 0.25 0.0 0.0 2 0.5",
    ).replace(
        "Mn1 0.0 0.0 3.0 0,0,w 0,0,w 3.0",
        "Mn1 0.0 0.0 3.0 0,0,w 0,0,w 3.0\nFe1 0.0 0.0 2.0 0,0,w 0,0,w 2.0",
    )
    structure = SpinCifReader(text).get_structure()
    assert len(structure) == 4
    assert all(site.properties["magnetic"] for site in structure)
    for site in structure:
        (species,) = site.species
        np.testing.assert_allclose(
            site.properties["magmom"], [0, 0, 3 if species == get_el_sp("Mn") else 2]
        )
        assert dict(site.species.items()) == {species: 0.5}


def test_reader_constituent_rows_can_be_exported_to_mcif(external_scif):
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\nMn1 Mn 0.25 0.0 0.0 2",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n"
        "Mn1 Mn0.3333333333333333+ 0.25 0.0 0.0 2 0.5\n"
        "Mn2 Mn0.3333333333333334+ 0.25 0.0 0.0 2 0.25",
    ).replace("Mn1 0.0 0.0 3.0 0,0,w 0,0,w 3.0", "Mn1 0.0 0.0 0.0 0,0,w 0,0,w 0.0")
    output = str(MCifWriter(SpinCifReader(text).get_structure(), symprec=1e-5))
    (block,) = CifFile.from_str(output).data.values()
    assert set(
        zip(block.data["_atom_site_type_symbol"], block.data["_atom_site_occupancy"], strict=True)
    ) == {
        ("Mn0.3333333333333333+", "0.5"),
        ("Mn0.3333333333333334+", "0.25"),
    }
    magnetic_labels = set(block.data["_atom_site_moment.label"])
    assert magnetic_labels == {
        label
        for label, symbol in zip(
            block.data["_atom_site_label"], block.data["_atom_site_type_symbol"], strict=True
        )
        if symbol == "Mn0.3333333333333333+"
    }
    for axis in "xyz":
        assert all(
            float(value) == 0 for value in block.data[f"_atom_site_moment.crystalaxis_{axis}"]
        )
