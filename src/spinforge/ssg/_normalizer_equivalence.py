"""Step 2: Deduplicate spin-rotation assignments under S(H') x N_O(3)(B_so).

Uses the active-block projection (modulo B_so) and get_real_intertwiner to
test equivalence via the Schur-averaging formula.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from spgrep.rep.group import get_inverse_index
from spgrep.utils import NDArrayFloat, NDArrayInt
from spinspg.spin import SpinOnlyGroupType

from spinforge.irreps import get_real_intertwiner
from spinforge.space_group import NormalSpaceSubgroup
from spinforge.ssg._spin_only_group import extract_active_block

if TYPE_CHECKING:
    from spinforge.ssg import NontrivialSpinSpaceGroup


def deduplicate_spin_rotation_assignments(
    list_nssg: list[NontrivialSpinSpaceGroup],
    nss: NormalSpaceSubgroup,
    stabilizer: Sequence[int],
    spin_only_group_type: SpinOnlyGroupType,
    family_table: NDArrayInt,
    normalizer_permutations: Sequence[Sequence[int]],
    *,
    atol: float = 1e-5,
) -> list[NontrivialSpinSpaceGroup]:
    """Return one ``NontrivialSpinSpaceGroup`` per ``S(H') x N_O(3)(B_so)``
    orbit from ``list_nssg``.

    Parameters
    ----------
    list_nssg:
        Candidate spin space groups sharing the same ``NormalSpaceSubgroup``.
    nss:
        The common ``NormalSpaceSubgroup`` (the ``H'`` representative).
    stabilizer:
        Indices of normalizer representatives that stabilize ``H′``.
    spin_only_group_type:
        Used to extract the active ``d x d`` block from the embedded (3x3)
        spin rotations.
    family_table:
        Cayley table of the family group ``G′``.
    normalizer_permutations:
        Inverse-conjugation permutations on the family operation order.
    atol:
        Numerical tolerance.

    Returns
    -------
    list[NontrivialSpinSpaceGroup]
        One representative per equivalence class.
    """
    if len(list_nssg) <= 1:
        return list(list_nssg)

    coset_representatives = list(nss.coset_representatives)
    family_to_coset = {
        family_index: coset_index for coset_index, family_index in enumerate(coset_representatives)
    }

    pullback_perms: list[list[int]] = []
    for normalizer_index in stabilizer:
        family_permutation = normalizer_permutations[normalizer_index]
        permutation = [
            _find_coset_index(
                int(family_permutation[family_index]),
                nss,
                family_table,
                family_to_coset,
            )
            for family_index in coset_representatives
        ]
        pullback_perms.append(permutation)

    # Extract active blocks for each candidate NSSG.
    active_blocks: list[NDArrayFloat] = [
        extract_active_block(spin_only_group_type, nssg.nontrivial_coset.spin_rotations)
        for nssg in list_nssg
    ]

    # Greedy orbit grouping.
    representative_indices: list[int] = []
    consumed: list[bool] = [False] * len(list_nssg)

    for i in range(len(list_nssg)):
        if consumed[i]:
            continue
        representative_indices.append(i)
        consumed[i] = True

        # Try to match every later candidate j against representative i.
        for j in range(i + 1, len(list_nssg)):
            if consumed[j]:
                continue
            if _are_equivalent(
                active_blocks[i],
                active_blocks[j],
                pullback_perms,
                atol=atol,
            ):
                consumed[j] = True

    return [list_nssg[i] for i in representative_indices]


def _are_equivalent(
    block_i: NDArrayFloat,
    block_j: NDArrayFloat,
    pullback_perms: list[list[int]],
    *,
    atol: float,
) -> bool:
    """Test whether blocks ``i`` and ``j`` are intertwined by some pullback
    ``h`` in the stabilizer."""
    for perm in pullback_perms:
        pulled = block_i[perm]  # U_i^h
        intertwiner = get_real_intertwiner(pulled, block_j, atol=atol)
        if intertwiner is not None:
            return True
    return False


def _find_coset_index(
    family_index: int,
    nss: NormalSpaceSubgroup,
    family_table: NDArrayInt,
    family_to_coset: dict[int, int],
) -> int:
    """Find the coset of ``G′ / H′`` containing ``family_index``."""
    if family_index in family_to_coset:
        return family_to_coset[family_index]
    for point_subgroup_index in nss.point_subgroup:
        inverse = get_inverse_index(family_table, point_subgroup_index)
        representative = int(family_table[family_index, inverse])
        if representative in family_to_coset:
            return family_to_coset[representative]
    raise ValueError(f"Family element {family_index} does not belong to any coset of G′ / H′.")
