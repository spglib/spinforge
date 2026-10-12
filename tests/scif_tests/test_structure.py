"""Structure-based adaptation to the SSA writer."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Species, Structure
from pymatgen.io.cif import CifFile
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from spinforge.scif import SpinCifReader, SpinCifWriter


@pytest.fixture
def collinear_model(enumerated):
    return next(
        entry for entry in enumerated("MnTe", SpinOnlyGroupType.COLLINEAR, 1) if entry[2].dim == 1
    )


def test_reordered_structure_matches_ssa_sites(collinear_model):
    sog, group, ssa = collinear_model
    (moments,) = ssa.magnetic_moments_basis
    structure = ssa.generate_with_magnetic_moments(moments)
    reversed_structure = Structure.from_sites(list(reversed(structure)))
    reference = SpinCifWriter.from_structure(
        ssa, group, structure, spin_only_group=sog, symprec=1e-8
    )
    reordered = SpinCifWriter.from_structure(
        ssa, group, reversed_structure, spin_only_group=sog, symprec=1e-8
    )
    assert str(reordered) == str(reference)
    reader = SpinCifReader(str(reordered))
    for rotation, translation, _, _ in reader.expanded_operations():
        transformed = structure.frac_coords @ rotation.T + translation
        distances = structure.lattice.get_all_distances(transformed, structure.frac_coords)
        assert np.all(np.min(distances, axis=1) <= 1e-8)


def test_constituent_partition_does_not_break_ssg_orbits(collinear_model):
    sog, group, ssa = collinear_model
    (moments,) = ssa.magnetic_moments_basis
    structure = ssa.generate_with_magnetic_moments(moments)
    species, positions, vectors = [], [], []
    split_done = False
    for site in structure:
        amounts = [0.5, 0.5] if site.specie.symbol == "Mn" and not split_done else [1.0]
        split_done = split_done or site.specie.symbol == "Mn"
        for amount in amounts:
            species.append({site.specie: amount})
            positions.append(site.frac_coords)
            vectors.append(site.properties["magmom"])
    split = Structure(structure.lattice, species, positions, site_properties={"magmom": vectors})
    original = SpinCifWriter.from_structure(
        ssa, group, structure, spin_only_group=sog, symprec=1e-8
    )
    partitioned = SpinCifWriter.from_structure(
        ssa, group, split, spin_only_group=sog, symprec=1e-8
    )
    assert str(partitioned) == str(original)


@pytest.mark.parametrize("mismatch", ["position", "lattice", "missing"])
def test_structure_must_match_ssa_supercell(collinear_model, mismatch):
    sog, group, ssa = collinear_model
    (moments,) = ssa.magnetic_moments_basis
    structure = ssa.generate_with_magnetic_moments(moments)
    index = next(index for index, site in enumerate(structure) if site.specie.symbol == "Mn")
    if mismatch == "position":
        structure.translate_sites(index, [0.01, 0, 0])
    elif mismatch == "lattice":
        structure.scale_lattice(structure.volume * 1.01)
    else:
        structure.remove_sites([index])
    with pytest.raises(ValueError, match="match.*SSA supercell"):
        SpinCifWriter.from_structure(ssa, group, structure, spin_only_group=sog, symprec=1e-8)


def test_explicit_moment_tolerance_controls_ssa_residual(collinear_model):
    sog, group, ssa = collinear_model
    (basis,) = ssa.magnetic_moments_basis
    moments = basis * 1000
    index = next(index for index, number in enumerate(ssa.supercell.numbers) if number == 25)
    moments[index, 0] += 1e-4
    structure = ssa.generate_with_magnetic_moments(moments)
    with pytest.raises(ValueError, match="SSA basis"):
        SpinCifWriter.from_structure(
            ssa, group, structure, spin_only_group=sog, symprec=1e-8, mag_symprec=1e-6
        )
    SpinCifWriter.from_structure(
        ssa, group, structure, spin_only_group=sog, symprec=1e-8, mag_symprec=1e-3
    )


def test_explicit_moment_tolerance_controls_spin_only_axis(collinear_model):
    _, group, ssa = collinear_model
    (moments,) = ssa.magnetic_moments_basis
    structure = ssa.generate_with_magnetic_moments(moments * 1000)
    axis = np.array([1e-4, 0, 1])
    sog = SpinOnlyGroup(SpinOnlyGroupType.COLLINEAR, axis / np.linalg.norm(axis))
    with pytest.raises(ValueError, match="spin-only axis"):
        SpinCifWriter.from_structure(
            ssa, group, structure, spin_only_group=sog, symprec=1e-8, mag_symprec=1e-6
        )
    SpinCifWriter.from_structure(
        ssa, group, structure, spin_only_group=sog, symprec=1e-8, mag_symprec=1
    )


def test_from_oriented_preserves_structure_chemistry(collinear_model):
    sog, group, ssa = collinear_model
    for structure, msg in ssa.generate_oriented(sog, group, rng=np.random.default_rng(4)):
        species = [
            {Species("Mn", 0.331): 0.6, Species("Zn", 2): 0.2}
            if site.specie.symbol == "Mn"
            else {Species("Te", -2): 0.9}
            for site in structure
        ]
        mixed = Structure(
            structure.lattice,
            species,
            structure.frac_coords,
            site_properties=structure.site_properties,
        )
        text = str(SpinCifWriter.from_oriented(ssa, group, mixed, msg, spin_only_group=sog))
        read = SpinCifReader(text).get_structure()
        actual = {(next(iter(site.species)), next(iter(site.species.values()))) for site in read}
        assert actual == {
            (Species("Mn", 0.331), 0.6),
            (Species("Zn", 2), 0.2),
            (Species("Te", -2), 0.9),
        }
        with pytest.raises(ValueError, match="only override ordered input"):
            SpinCifWriter.from_oriented(
                ssa, group, mixed, msg, spin_only_group=sog, site_occupancies=[1] * len(mixed)
            )


def test_from_oriented_legacy_occupancy_override_respects_atol(collinear_model):
    sog, group, ssa = collinear_model
    for structure, msg in ssa.generate_oriented(sog, group, rng=np.random.default_rng(4)):
        occupancies = [0.8] * len(structure)
        index = next(index for index, site in enumerate(structure) if site.specie.symbol == "Mn")
        occupancies[index] += 5e-7
        SpinCifWriter.from_oriented(
            ssa,
            group,
            structure,
            msg,
            spin_only_group=sog,
            site_occupancies=occupancies,
            atol=1e-6,
        )
        with pytest.raises(ValueError, match="not invariant"):
            SpinCifWriter.from_oriented(
                ssa,
                group,
                structure,
                msg,
                spin_only_group=sog,
                site_occupancies=occupancies,
                atol=1e-8,
            )


def test_structure_occupancy_is_independent_of_geometric_tolerance(collinear_model):
    sog, group, ssa = collinear_model
    (moments,) = ssa.magnetic_moments_basis
    source = ssa.generate_with_magnetic_moments(moments)
    occupancy = 1 + 5e-9
    structure = Structure(
        source.lattice,
        [{site.specie: occupancy} for site in source],
        source.frac_coords,
        site_properties=source.site_properties,
    )
    text = str(
        SpinCifWriter.from_structure(
            ssa, group, structure, spin_only_group=sog, symprec=1e-10, atol=1e-12
        )
    )
    (block,) = CifFile.from_str(text).data.values()
    assert all(float(amount) == occupancy for amount in block.data["_atom_site_occupancy"])
