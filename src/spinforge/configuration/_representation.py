from collections.abc import Sequence

import numpy as np
from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from ._supercell import Supercell, SupercellSiteIndex


def get_site_representation(
    supercell: Supercell,
    sub_sites: Sequence[SupercellSiteIndex],
    prim_rotations: NDArrayInt,
    prim_translations: NDArrayFloat,
) -> NDArrayFloat | None:
    order = len(prim_rotations)

    rep = np.zeros((order, supercell.num_supercell_sites, supercell.num_supercell_sites))
    for k, (prim_rotation, prim_translation) in enumerate(zip(prim_rotations, prim_translations)):
        permutation = supercell.act_operation(sub_sites, prim_rotation, prim_translation)
        if permutation is None:
            # Spatial part of the spin symmetry operation may not preserve the given `sub_sites`!
            return None

        for i, pi in permutation.items():
            rep[k, pi, i] = 1

    sub_rep = rep[:, sub_sites, :][:, :, sub_sites]
    return sub_rep


def get_spin_only_group_reynolds_operator(
    spin_only_group: SpinOnlyGroup,
) -> NDArrayFloat:
    if spin_only_group.spin_only_group_type == SpinOnlyGroupType.NONMAGNETIC:
        return np.zeros((3, 3))
    elif spin_only_group.spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
        axis = spin_only_group.axis
        if axis is None:
            raise ValueError("Collinear spin-only group must have a defined axis.")
        return axis[None, :] * axis[:, None]
    elif spin_only_group.spin_only_group_type == SpinOnlyGroupType.COPLANAR:
        axis = spin_only_group.axis
        if axis is None:
            raise ValueError("Coplanar spin-only group must have a defined axis.")
        return np.eye(3) - axis[None, :] * axis[:, None]
    elif spin_only_group.spin_only_group_type == SpinOnlyGroupType.NONCOPLANAR:
        return np.eye(3)
    else:
        raise ValueError(f"Unknown spin group type: {spin_only_group.spin_only_group_type}")
