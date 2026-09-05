from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from moyopy import (
    Cell,
    MagneticSpaceGroup,
    MagneticSpaceGroupType,
    MoyoDataset,
    SpaceGroup,
)
from spinspg.spin import SpinOnlyGroupType

from spinforge.configuration import SSAGenerator
from spinforge.msg import (
    ConstructType,
    OrientedSpinSpaceGroupEnumerator,
)
from spinforge.scif import SpinCifReader, SpinCifWriter
from spinforge.testing import (
    get_magnetic_space_group_type,
    load_alpha_Mn_structure,
    load_prim_CoTa3S6,
    load_prim_Cu3Au,
    load_prim_Mn3SiIr,
    load_prim_MnCuSb,
    load_prim_pyrochlore,
    load_prim_UO2,
    load_prim_Yb2O3,
    site_indices,
)

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from spinspg.spin import SpinOnlyGroup

    from spinforge.configuration import SpinSymmetryAdaptedStructure
    from spinforge.msg import MagneticSpaceSubgroup
    from spinforge.ssg import NontrivialSpinSpaceGroup


@pytest.fixture
def output_dir() -> Path:
    path = Path("./tmp")
    path.mkdir(exist_ok=True, parents=True)
    return path


def _check_with_magnetic_structure(
    ms: Structure, msg: MagneticSpaceSubgroup, nssg: NontrivialSpinSpaceGroup
) -> MagneticSpaceGroupType:
    # MSG from generated magnetic structure
    msg_type_from_ms = get_magnetic_space_group_type(ms, magmom_scale=100)

    operations, _ = nssg.get_full_operations_and_table()
    time_reversals = np.zeros(len(operations.rotations), dtype=bool)
    time_reversals[msg.fsg] = True
    time_reversals[msg.xsg] = False

    fsg_rotations, fsg_translations = nssg.sublattice.transform_operations(
        operations.rotations[msg.fsg], operations.translations[msg.fsg]
    )
    msg_from_operations = MagneticSpaceGroup(
        prim_rotations=fsg_rotations.tolist(),
        prim_translations=fsg_translations.tolist(),
        prim_time_reversals=time_reversals[msg.fsg].tolist(),
    )

    # MSG from SSG operations
    assert msg_from_operations.uni_number == msg_type_from_ms.uni_number

    return msg_type_from_ms


def _get_nssg_label(nssg: NontrivialSpinSpaceGroup) -> str:
    operations, _ = nssg.get_full_operations_and_table()
    k_index = nssg.k_index

    nonmag_space_group = SpaceGroup(
        prim_rotations=operations.rotations[::k_index].tolist(),
        prim_translations=operations.translations[::k_index].tolist(),
    )

    invariant_space_subgroup = SpaceGroup(
        prim_rotations=nssg.invariant_rotations.tolist(),
        prim_translations=nssg.invariant_translations.tolist(),
    )

    sublattice_snf: tuple[int, ...] = nssg.sublattice._invariant_factors

    label = f"family-{nonmag_space_group.number}_k-index-{k_index}_snf-{'-'.join(map(str, sublattice_snf))}_t-index-{nssg.t_index}_invariant-{invariant_space_subgroup.number}"
    return label


def _check_and_dump_oriented_structures(
    oriented_structures: list[tuple[Structure, MagneticSpaceSubgroup]],
    sas: SpinSymmetryAdaptedStructure,
    sog: SpinOnlyGroup,
    nssg: NontrivialSpinSpaceGroup,
    output_dir: Path,
    prefix: str,
    *,
    ssg_idx: int | None = None,
    with_sign: bool = False,
) -> None:
    """Verify each oriented structure's MSG against its operations and dump a scif.

    ``prefix`` should carry the material and spin-only group type
    (e.g. ``f"Mn3Sn_{sog.spin_only_group_type}"``); ``ssg_idx`` and
    ``with_sign`` (sign of det Q) add the corresponding filename tokens.
    """
    nssg_label = _get_nssg_label(nssg)
    ssg_token = f"_ssg-{ssg_idx}" if ssg_idx is not None else ""
    for ms_idx, (ms, msg) in enumerate(oriented_structures):
        magnetic_space_group_type = _check_with_magnetic_structure(ms, msg, nssg)
        bns_number = magnetic_space_group_type.bns_number.replace(".", "-")
        sign_token = ""
        if with_sign:
            sign = "p" if np.linalg.det(msg.Q) > 0 else "n"
            sign_token = f"_sign-{sign}"
        writer = SpinCifWriter.from_oriented(sas, nssg, ms, msg, spin_only_group=sog)
        path = (
            output_dir
            / f"{prefix}_{nssg_label}{ssg_token}_bns-{bns_number}{sign_token}_ms-{ms_idx}.scif"
        )
        writer.write_file(str(path))
        _assert_scif_reconstructs(path, ms)


def _assert_scif_reconstructs(path: Path, ms: Structure, atol: float = 1e-4) -> None:
    """Parse the written scif back and compare it against the dumped structure.

    Moments are compared as lattice-fractional components (``m @ L^-1``),
    which are invariant under the Cartesian orientation choice each side's
    lattice makes.
    """
    structure = SpinCifReader.from_file(path).get_structure()
    assert len(structure) == len(ms)
    # The cell tags carry six decimals; fractional coordinates alone would
    # not notice a wrong lattice metric.
    assert np.allclose(structure.lattice.parameters, ms.lattice.parameters, atol=1e-4)

    read_frac = structure.frac_coords
    read_moments = np.array(structure.site_properties["magmom"]) @ np.linalg.inv(
        structure.lattice.matrix
    )
    inv_lattice = np.linalg.inv(ms.lattice.matrix)
    matched: set[int] = set()
    for site in ms:
        delta = read_frac - np.asarray(site.frac_coords)
        delta -= np.rint(delta)
        (hits,) = np.nonzero(np.all(np.abs(delta) <= atol, axis=1))
        assert len(hits) == 1, f"Site {site.frac_coords} not uniquely recovered from {path}"
        index = int(hits[0])
        assert index not in matched
        matched.add(index)
        assert str(structure[index].specie) == str(site.specie)
        expected_moment = np.asarray(site.properties["magmom"]) @ inv_lattice
        assert np.allclose(read_moments[index], expected_moment, atol=atol)


###############################################################################
# Collinear
###############################################################################


@pytest.mark.parametrize(
    "k_index",
    [1, 2],
)
def test_oriented_ssg_enumerator_Mn3Sn_collinear(
    output_dir: Path,
    Mn3Sn_ssa_generator: SSAGenerator,
    k_index: int,
):
    all_magnetic_structures = Mn3Sn_ssa_generator.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COLLINEAR, k_index=k_index, max_depth=0
    )

    for ssg_idx, (sog, nssg, sas) in enumerate(all_magnetic_structures):
        assert sas.dim == 1
        oriented_structures = Mn3Sn_ssa_generator.generate_oriented(
            sas,
            spin_only_group=sog,
            nontrivial_spin_space_group=nssg,
        )
        _check_and_dump_oriented_structures(
            oriented_structures,
            sas,
            sog,
            nssg,
            output_dir,
            f"Mn3Sn_{sog.spin_only_group_type}",
            ssg_idx=ssg_idx,
        )


def test_MnCuSb_collinear(output_dir: Path, MnCuSb_magnetic_site_indices: list[int]):
    prim_cell = load_prim_MnCuSb()
    magnetic_site_indices = MnCuSb_magnetic_site_indices

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COLLINEAR, k_index=2, max_depth=0
    )

    for ssg_idx, (sog, nssg, sas) in enumerate(all_magnetic_structures):
        oriented_structures = scg.generate_oriented(
            sas,
            spin_only_group=sog,
            nontrivial_spin_space_group=nssg,
        )
        _check_and_dump_oriented_structures(
            oriented_structures,
            sas,
            sog,
            nssg,
            output_dir,
            f"MnCuSb_{sog.spin_only_group_type}",
            ssg_idx=ssg_idx,
        )


def test_NiO_collinear_parent_axes_are_explicit_opt_in():
    """The canonical generator uses family R-3m axes, not parent Fm-3m axes."""
    a = 4.175
    prim_cell = Cell(
        basis=[[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]],
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        numbers=[28, 8],  # Ni, O
    )

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=site_indices(prim_cell, 28),
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COLLINEAR, k_index=2, max_depth=1
    )

    # Find SSG with invariant subgroup R-3m (No. 166)
    target_nssg = None
    for sog, nssg, sas in all_magnetic_structures:
        inv_sg = SpaceGroup(
            prim_rotations=nssg.invariant_rotations.tolist(),
            prim_translations=nssg.invariant_translations.tolist(),
            basis=prim_cell.basis,
        )
        if inv_sg.number == 166:
            target_nssg = (sog, nssg, sas)
            break
    assert target_nssg is not None, "Expected SSG with Inv SG = R-3m"
    sog, nssg, sas = target_nssg

    def _bns_set(oriented):
        return {
            get_magnetic_space_group_type(ms, magmom_scale=100).bns_number for ms, _ in oriented
        }

    # BNS 15.90 needs [1,1,-2], which is not an axis of the family group D3d.
    bns_family = _bns_set(
        scg.generate_oriented(sas, spin_only_group=sog, nontrivial_spin_space_group=nssg)
    )
    assert "15.90" not in bns_family

    # The compatibility API can still opt in to axes outside the family group.
    bns_parent = _bns_set(
        sas.generate_oriented(
            spin_only_group=sog,
            nontrivial_spin_space_group=nssg,
            parent_prim_rotations=scg.prim_rotations,
        )
    )
    assert "15.90" in bns_parent
    assert bns_family.issubset(bns_parent)


###############################################################################
# Coplanar
###############################################################################


def test_oriented_ssg_enumerator_coplanar_Mn3Sn(
    output_dir: Path,
    Mn3Sn_ssa_generator: SSAGenerator,
):
    all_magnetic_structures = Mn3Sn_ssa_generator.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COPLANAR, k_index=1, max_depth=0
    )
    assert sum(sas.dim == 1 for _, _, sas in all_magnetic_structures) == 2

    for ssg_idx, (sog, nssg, sas) in enumerate(all_magnetic_structures):
        if sas.dim != 1:
            continue
        oriented_structures = Mn3Sn_ssa_generator.generate_oriented(
            sas,
            spin_only_group=sog,
            nontrivial_spin_space_group=nssg,
        )
        _check_and_dump_oriented_structures(
            oriented_structures,
            sas,
            sog,
            nssg,
            output_dir,
            f"Mn3Sn_{sog.spin_only_group_type}",
            ssg_idx=ssg_idx,
            with_sign=True,
        )


def test_coplanar_Mn3Sn_preserves_spin_planochiral_enantiomorphs(
    Mn3Sn_ssa_generator: SSAGenerator,
):
    results = Mn3Sn_ssa_generator.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COPLANAR,
        k_index=1,
        max_depth=0,
    )

    spin_planochiral = [
        (spin_only_group, nssg, sas)
        for spin_only_group, nssg, sas in results
        if sas.dim == 1 and nssg.spin_planochiral
    ]
    assert len(spin_planochiral) == 2

    for spin_only_group, nssg, sas in spin_planochiral:
        chirality_blind = Mn3Sn_ssa_generator.generate_oriented(
            sas,
            spin_only_group=spin_only_group,
            nontrivial_spin_space_group=nssg,
            preserve_spin_planochirality=False,
        )
        chirality_preserving = Mn3Sn_ssa_generator.generate_oriented(
            sas,
            spin_only_group=spin_only_group,
            nontrivial_spin_space_group=nssg,
        )

        assert len(chirality_blind) == 6
        assert len(chirality_preserving) == 8


def test_Cu3Au_coplanar(output_dir: Path, Cu3Au_magnetic_site_indices: list[int]):
    prim_cell = load_prim_Cu3Au()
    magnetic_site_indices = Cu3Au_magnetic_site_indices

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.COPLANAR, k_index=1, max_depth=0
    )

    for ssg_idx, (sog, nssg, sas) in enumerate(all_magnetic_structures):
        oriented_structures = scg.generate_oriented(
            sas,
            spin_only_group=sog,
            nontrivial_spin_space_group=nssg,
        )
        _check_and_dump_oriented_structures(
            oriented_structures,
            sas,
            sog,
            nssg,
            output_dir,
            f"Cu3Au_{sog.spin_only_group_type}",
            ssg_idx=ssg_idx,
            with_sign=True,
        )


###############################################################################
# Noncoplanar
###############################################################################


@pytest.mark.parametrize(
    "k_index",
    [1, 2],
)
def test_noncoplanar_alpha_Mn(output_dir: Path, k_index: int):
    structure = load_alpha_Mn_structure()
    cell = Cell(
        basis=structure.lattice.matrix.tolist(),
        positions=structure.frac_coords.tolist(),
        numbers=list(structure.atomic_numbers),
    )
    dataset = MoyoDataset(cell)

    sites_8c_in_prim_cell = list(
        set(
            [
                dataset.mapping_std_prim[i]
                for i, letter in enumerate(dataset.wyckoffs)
                if letter == "c"
            ]
        )
    )
    prim_cell = dataset.prim_std_cell

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=sites_8c_in_prim_cell,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR, k_index=k_index, max_depth=0
    )
    assert sum(sas.dim == 1 for _, _, sas in all_magnetic_structures) == 1
    sog, nssg, sas = all_magnetic_structures[0]
    assert sas.dim == 1
    oriented_structures = scg.generate_oriented(
        sas,
        spin_only_group=sog,
        nontrivial_spin_space_group=nssg,
    )
    _check_and_dump_oriented_structures(
        oriented_structures, sas, sog, nssg, output_dir, f"Mn_{sog.spin_only_group_type}"
    )


def test_noncoplanar_UO2_index4(output_dir: Path):
    prim_cell = load_prim_UO2()
    magnetic_site_indices = site_indices(prim_cell, 92)  # U sites

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR,
        k_index=4,
        max_depth=0,
    )
    assert len(all_magnetic_structures) == 1
    sog, nssg, sas = all_magnetic_structures[0]

    oriented_structures = scg.generate_oriented(
        sas,
        spin_only_group=sog,
        nontrivial_spin_space_group=nssg,
        preserve_spin_planochirality=True,
    )
    _check_and_dump_oriented_structures(
        oriented_structures, sas, sog, nssg, output_dir, f"UO2_{sog.spin_only_group_type}"
    )


def test_noncoplanar_CoTa3S6_index4(output_dir: Path, CoTa3S6_magnetic_site_indices: list[int]):
    prim_cell = load_prim_CoTa3S6()
    magnetic_site_indices = CoTa3S6_magnetic_site_indices

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR, k_index=4, max_depth=0
    )
    assert sum(sas.dim == 1 for _, _, sas in all_magnetic_structures) == 2

    for ssg_idx, (sog, nssg, sas) in enumerate(all_magnetic_structures):
        if sas.dim != 1:
            continue
        oriented_structures = scg.generate_oriented(
            sas,
            spin_only_group=sog,
            nontrivial_spin_space_group=nssg,
        )
        _check_and_dump_oriented_structures(
            oriented_structures,
            sas,
            sog,
            nssg,
            output_dir,
            f"CoTa3S6_{sog.spin_only_group_type}",
            ssg_idx=ssg_idx,
        )


def test_noncoplanar_CoTa3S6_has_three_maximal_orientations(
    CoTa3S6_magnetic_site_indices: list[int],
):
    prim_cell = load_prim_CoTa3S6()
    generator = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=CoTa3S6_magnetic_site_indices,
    )
    results = generator.enumerate(
        spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR,
        k_index=4,
        max_depth=0,
    )

    one_dimensional = [
        (spin_only_group, nssg, sas) for spin_only_group, nssg, sas in results if sas.dim == 1
    ]
    assert len(one_dimensional) == 2

    for spin_only_group, nssg, sas in one_dimensional:
        enumerator = OrientedSpinSpaceGroupEnumerator(
            lattice=np.asarray(sas.supercell.prim_cell.basis),
            spin_only_group=spin_only_group,
            spin_space_group=nssg,
        )
        oriented = enumerator.enumerate()
        assert len(oriented) == 3


def test_noncoplanar_pyrochlore_Cd2Os2O7(
    output_dir: Path, pyrochlore_magnetic_site_indices: list[int]
):
    prim_cell = load_prim_pyrochlore()
    magnetic_site_indices = pyrochlore_magnetic_site_indices

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR, k_index=1, max_depth=0
    )
    assert sum(sas.dim == 1 for _, _, sas in all_magnetic_structures) == 1  # All-in-all-out
    sog, nssg, sas = all_magnetic_structures[0]
    assert sas.dim == 1

    # Generate 1 oriented magnetic structure
    oriented_structures = scg.generate_oriented(
        sas,
        spin_only_group=sog,
        nontrivial_spin_space_group=nssg,
    )
    assert len(oriented_structures) == 1
    _, msg = oriented_structures[0]
    assert len(msg.xsg) == 24
    assert len(msg.fsg) == 48
    assert msg.msg_type == ConstructType.TYPE3

    _check_and_dump_oriented_structures(
        oriented_structures, sas, sog, nssg, output_dir, f"Cd2Os2O7_{sog.spin_only_group_type}"
    )


def test_Mn3IrSi_noncoplanar(output_dir: Path, Mn3SiIr_magnetic_site_indices: list[int]):
    prim_cell = load_prim_Mn3SiIr()
    magnetic_site_indices = Mn3SiIr_magnetic_site_indices

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR, k_index=1
    )
    assert len(all_magnetic_structures) >= 1

    for ssg_idx, (sog, nssg, sas) in enumerate(all_magnetic_structures):
        # multidimensional case
        oriented_structures = scg.generate_oriented(
            sas,
            spin_only_group=sog,
            nontrivial_spin_space_group=nssg,
        )
        _check_and_dump_oriented_structures(
            oriented_structures,
            sas,
            sog,
            nssg,
            output_dir,
            f"Mn3IrSi_{sog.spin_only_group_type}",
            ssg_idx=ssg_idx,
        )


def test_Yb2O3_noncoplanar(output_dir: Path, Yb2O3_magnetic_site_indices: list[int]):
    # https://www.cryst.ehu.es/magndata/index.php?this_label=1.720
    # Multiple Wyckoff positions for Yb: 8a, 24d
    prim_cell = load_prim_Yb2O3()
    magnetic_site_indices = Yb2O3_magnetic_site_indices

    scg = SSAGenerator(
        prim_cell=prim_cell,
        magnetic_site_indices=magnetic_site_indices,
    )
    all_magnetic_structures = scg.enumerate(
        spin_only_group_type=SpinOnlyGroupType.NONCOPLANAR, k_index=2
    )
    assert len(all_magnetic_structures) >= 1

    # TODO:
    # for ssg_idx, (sog, nssg, sas) in enumerate(all_magnetic_structures):
    #     oriented_structures = sas.generate_oriented(
    #         spin_only_group=sog,
    #         nontrivial_spin_space_group=nssg,
    #     )
    #     nssg_label = _get_nssg_label(nssg)
    #     for ms_idx, (ms, msg) in enumerate(oriented_structures):
    #         magnetic_space_group_type = _check_with_magnetic_structure(ms, msg, nssg)
    #         bns_number = magnetic_space_group_type.bns_number.replace(".", "-")
    #         ms.to_file(
    #             output_dir
    #             / f"Yb2O3_{sog.spin_only_group_type}_{nssg_label}_ssg-{ssg_idx}_bns-{bns_number}_ms-{ms_idx}.mcif"
    #         )
