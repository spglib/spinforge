from __future__ import annotations

from pathlib import Path

import numpy as np
from moyopy import (
    Cell,
    MagneticSpaceGroupType,
    MoyoDataset,
    MoyoNonCollinearMagneticDataset,
    NonCollinearMagneticCell,
)
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import Magmom

DATA_ROOT = Path(__file__).parent / "assets"


def _load_prim_asset(filename: str) -> Cell:
    return get_prim_cell(Structure.from_file(DATA_ROOT / filename))


def site_indices(cell: Cell, atomic_number: int) -> list[int]:
    """Indices of sites in ``cell`` with the given atomic number."""
    return [i for i, number in enumerate(cell.numbers) if number == atomic_number]


def get_magnetic_space_group_type(
    structure: Structure, *, magmom_scale: float = 1.0
) -> MagneticSpaceGroupType:
    """Identify the magnetic space-group type of a pymatgen magnetic structure.

    ``magmom_scale`` multiplies the moments before identification (useful when
    sampled moments are small relative to moyopy's ``mag_symprec``).
    """
    magmoms = np.array([Magmom(m).global_moment for m in structure.site_properties["magmom"]])
    magnetic_cell = NonCollinearMagneticCell(
        basis=structure.lattice.matrix.tolist(),
        positions=structure.frac_coords.tolist(),
        numbers=[int(z) for z in structure.atomic_numbers],
        magnetic_moments=(magmoms * magmom_scale).tolist(),
    )
    dataset = MoyoNonCollinearMagneticDataset(magnetic_cell)
    return MagneticSpaceGroupType(dataset.uni_number)


def load_prim_pyrochlore() -> Cell:
    # https://next-gen.materialsproject.org/materials/mp-5950?formula=Cd2Os2O7
    # Fd-3m, origin choice 1 (No. 227)
    return _load_prim_asset("mp-5950_Cd2Os2O7.json")


def load_prim_Mn3Sn() -> Cell:
    # P6_3/mmc
    return _load_prim_asset("Mn3Sn.cif")


def load_prim_CoTa3S6() -> Cell:
    # P 6_3 2 2
    return _load_prim_asset("mp-1208409_Ta3CoS6.json")


def load_prim_Ta2FeO6() -> Cell:
    # -P 4n 2n
    return _load_prim_asset("mp-31755_Ta2FeO6.json")


def load_alpha_Mn_structure() -> Structure:
    # I-43m (No. 217)
    structure = Structure.from_file(DATA_ROOT / "mp-35_Mn.json")
    return structure


def load_prim_UO2() -> Cell:
    # Fm-3m (No. 225)
    return _load_prim_asset("mp-1597_UO2.json")


def load_prim_Mn3SiIr() -> Cell:
    # P 2_1 3 (No. 198)
    return _load_prim_asset("mp-20330_Mn3SiIr.json")


def load_prim_Cu3Au() -> Cell:
    # Pm-3m (No. 221)
    return _load_prim_asset("mp-2258_Cu3Au.json")


def load_prim_Yb2O3() -> Cell:
    # Ia-3 (No. 206)
    return _load_prim_asset("mp-2814_Yb2O3.json")


def load_prim_MnS2() -> Cell:
    # Pa-3 (No. 205)
    return _load_prim_asset("mp-1455_MnS2.cif")


def load_prim_MnTe() -> Cell:
    # P6_3/mmc (No. 194)
    return _load_prim_asset("MnTe.cif")


def load_prim_MnCuSb() -> Cell:
    # F-43m (No. 216)
    return _load_prim_asset("mp-5866_MnCuSb.json")


def get_prim_cell(structure: Structure) -> Cell:
    cell = Cell(
        basis=structure.lattice.matrix.tolist(),
        positions=structure.frac_coords.tolist(),
        numbers=list(structure.atomic_numbers),
    )
    dataset = MoyoDataset(cell)
    prim_cell = dataset.prim_std_cell
    return prim_cell
