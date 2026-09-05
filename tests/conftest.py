from __future__ import annotations

import pytest

from spinforge.configuration import SSAGenerator
from spinforge.testing import (
    load_prim_CoTa3S6,
    load_prim_Cu3Au,
    load_prim_Mn3SiIr,
    load_prim_Mn3Sn,
    load_prim_MnCuSb,
    load_prim_MnTe,
    load_prim_pyrochlore,
    load_prim_Ta2FeO6,
    load_prim_Yb2O3,
    site_indices,
)


@pytest.fixture
def Mn3Sn_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_Mn3Sn(), 25)  # Mn sites


@pytest.fixture
def Mn3Sn_ssa_generator(Mn3Sn_magnetic_site_indices: list[int]) -> SSAGenerator:
    return SSAGenerator(
        prim_cell=load_prim_Mn3Sn(),
        magnetic_site_indices=Mn3Sn_magnetic_site_indices,
    )


@pytest.fixture
def pyrochlore_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_pyrochlore(), 76)  # Os(16c) sites


@pytest.fixture
def CoTa3S6_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_CoTa3S6(), 27)  # Co sites


@pytest.fixture
def MnTe_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_MnTe(), 25)  # Mn sites


@pytest.fixture
def Ta2FeO6_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_Ta2FeO6(), 26)  # Fe sites


@pytest.fixture
def Mn3SiIr_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_Mn3SiIr(), 25)  # Mn sites


@pytest.fixture
def Cu3Au_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_Cu3Au(), 29)  # Cu sites


@pytest.fixture
def Yb2O3_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_Yb2O3(), 70)  # Yb sites


@pytest.fixture
def MnCuSb_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_MnCuSb(), 25)  # Mn sites
