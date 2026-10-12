"""Constituent chemistry and explicit spin frames survive spinCIF expansion."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Composition
from pymatgen.io.cif import CifFile
from spinspg.spin import SpinOnlyGroupType

from spinforge.scif import SpinCifReader, SpinCifWriter


@pytest.fixture
def antitranslation_model(enumerated):
    return next(
        (sog, group, ssa)
        for sog, group, ssa in enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 2)
        if group.spin_translation_coset.size > 1 and ssa.dim == 1
    )


def _chemistry(ssa, magnetic_composition, selected):
    compositions = [
        Composition(magnetic_composition if number == 25 else {"Te2-": 0.9})
        for number in ssa.supercell.numbers
    ]
    selections = [selected if number == 25 else () for number in ssa.supercell.numbers]
    return compositions, selections


@pytest.mark.parametrize(
    ("magnetic_composition", "selected"),
    [
        ({"Mn2+": 0.6, "Zn2+": 0.2}, ("Mn2+",)),
        ({"Mn2+": 0.6, "Fe2+": 0.2}, ("Mn2+", "Fe2+")),
        ({"Mn2+": 0.6, "Mn3+": 0.2}, ("Mn2+",)),
        ({"Mn": 0.6, "Mn0+": 0.2}, ("Mn0+",)),
        ({"Mn2+": 0.6, "Zn2+": 1e-8}, ("Mn2+",)),
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
    compositions, selections = _chemistry(ssa, magnetic_composition, selected)
    text = str(
        SpinCifWriter(
            ssa,
            group,
            moments,
            spin_only_group=sog,
            spin_rotation=rotation,
            site_compositions=compositions,
            magnetic_species=selections,
        )
    )
    reader = SpinCifReader(text)
    structure = reader.get_structure()
    assert len(structure) == ssa.supercell.num_supercell_sites
    assert any(reversal == -1 for _, _, reversal, _ in reader.spin_lattice)
    embedding = np.linalg.inv(structure.lattice.matrix) @ ssa.supercell.basis
    for site in structure:
        indices = [
            index
            for index, position in enumerate(ssa.supercell.positions)
            if np.allclose(
                site.frac_coords - position - np.rint(site.frac_coords - position),
                0,
                atol=1e-5,
            )
        ]
        (index,) = indices
        assert site.species.as_dict() == compositions[index].as_dict()
        assert set(site.properties["magnetic_species"]) == set(selections[index])
        np.testing.assert_allclose(
            site.properties["magmom"] @ embedding, moments[index], atol=1e-5
        )
        assert set(site.properties["species_magmoms"]) == {
            str(species) for species in compositions[index]
        }
        mean = np.zeros(3)
        for species, occupancy in compositions[index].items():
            actual = site.properties["species_magmoms"][str(species)] @ embedding
            expected = moments[index] if str(species) in selections[index] else np.zeros(3)
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            mean += occupancy * actual
        expected_mean = moments[index] * sum(
            occupancy
            for species, occupancy in compositions[index].items()
            if str(species) in selections[index]
        )
        np.testing.assert_allclose(mean, expected_mean, atol=1e-5)

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
    compositions, selections = _chemistry(ssa, {"Mn2+": 0.6, "Zn2+": 0.2}, ("Mn2+",))
    text = str(
        SpinCifWriter(
            ssa,
            group,
            np.zeros((ssa.supercell.num_supercell_sites, 3)),
            spin_only_group=sog,
            site_compositions=compositions,
            magnetic_species=selections,
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
        site.properties["magnetic_species"] == (("Mn2+",) if "Mn2+" in site.species else ())
        for site in structure
    )
    np.testing.assert_array_equal(structure.site_properties["magmom"], 0)


@pytest.mark.parametrize("change_selection", [False, True])
def test_orbit_inconsistent_constituents_are_rejected(antitranslation_model, change_selection):
    sog, group, ssa = antitranslation_model
    (moments,) = ssa.magnetic_moments_basis
    compositions, selections = _chemistry(ssa, {"Mn2+": 0.6, "Fe2+": 0.2}, ("Mn2+",))
    index = next(index for index, number in enumerate(ssa.supercell.numbers) if number == 25)
    if change_selection:
        selections[index] = ("Fe2+",)
    else:
        compositions[index] = Composition({"Mn2+": 0.5, "Fe2+": 0.2})
    with pytest.raises(ValueError, match="invariant under the SSG operations"):
        SpinCifWriter(
            ssa,
            group,
            moments,
            spin_only_group=sog,
            site_compositions=compositions,
            magnetic_species=selections,
        )


@pytest.mark.parametrize("rotation", [np.eye(2), np.ones((3, 3)), np.eye(3) * np.nan])
def test_invalid_explicit_frames_are_rejected(antitranslation_model, rotation):
    sog, group, ssa = antitranslation_model
    (moments,) = ssa.magnetic_moments_basis
    with pytest.raises(ValueError, match="finite orthogonal 3 by 3"):
        SpinCifWriter(ssa, group, moments, spin_only_group=sog, spin_rotation=rotation)


def test_compositions_and_legacy_occupancies_are_exclusive(antitranslation_model):
    sog, group, ssa = antitranslation_model
    (moments,) = ssa.magnetic_moments_basis
    compositions, _ = _chemistry(ssa, {"Mn": 0.5}, ("Mn",))
    with pytest.raises(ValueError, match="either site_compositions or site_occupancies"):
        SpinCifWriter(
            ssa,
            group,
            moments,
            spin_only_group=sog,
            site_compositions=compositions,
            site_occupancies=[0.5] * len(compositions),
        )


def test_reader_rejects_different_selected_constituent_moments(external_scif):
    text = external_scif.replace(
        "_atom_site_symmetry_multiplicity\nMn1 Mn 0.25 0.0 0.0 2",
        "_atom_site_symmetry_multiplicity\n_atom_site_occupancy\n"
        "Mn1 Mn 0.25 0.0 0.0 2 0.5\nFe1 Fe 0.25 0.0 0.0 2 0.5",
    ).replace(
        "Mn1 0.0 0.0 3.0 0,0,w 0,0,w 3.0",
        "Mn1 0.0 0.0 3.0 0,0,w 0,0,w 3.0\nFe1 0.0 0.0 2.0 0,0,w 0,0,w 2.0",
    )
    with pytest.raises(ValueError, match="Unequal selected constituent moments"):
        SpinCifReader(text).get_structure()
