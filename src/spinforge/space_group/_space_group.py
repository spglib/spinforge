from __future__ import annotations

from itertools import product

import numpy as np
from spgrep.utils import NDArrayFloat, NDArrayInt, ndarray2d_to_integer_tuple

from spinforge.utils.group_theory import (
    FactorGroupElementIndex,
    GroupElementIndex,
    all_inverses,
    get_coset_representatives,
)

from ._family_subgroup import (
    FamilySpaceSubgroup,
    FamilySpaceSubgroupEnumerator,
)
from ._normal_space_subgroup import NormalSpaceSubgroup
from ._sublattice import Sublattice


class NormalSpaceSubgroupEnumerator:
    def __init__(
        self,
        prim_rotations: NDArrayInt,
        prim_translations: NDArrayFloat,
        table: NDArrayInt,
        *,
        sublattice: Sublattice | None = None,
        atol: float = 1e-6,
    ):
        self._prim_rotations = prim_rotations
        self._prim_translations = prim_translations
        self._sublattice = sublattice
        self._atol = atol

        self._table = table

        order = len(prim_rotations)
        self._factor_system = np.zeros((order, order, 3), dtype=int)
        for i, (ri, ti) in enumerate(zip(self._prim_rotations, self._prim_translations)):
            for j, tj in enumerate(self._prim_translations):
                k = table[i, j]
                fs = ti + ri @ tj - self._prim_translations[k]
                self._factor_system[i, j] = np.around(fs).astype(int)

        self._tuple_prim_rotations = [
            ndarray2d_to_integer_tuple(rotation) for rotation in self._prim_rotations
        ]
        self._family_subgroup_enumerator = FamilySpaceSubgroupEnumerator(
            self._prim_rotations,
            self._prim_translations,
            epsilon=self._atol,
            target_sublattice=self._sublattice,
            atol=self._atol,
        )
        self._normal_family_subgroups_cache: dict[int, tuple[FamilySpaceSubgroup, ...]] = {}

    def enumerate(
        self,
        normal_point_subgroup: list[GroupElementIndex],
        *,
        k_index: int | None = None,
    ) -> list[NormalSpaceSubgroup]:
        """Enumerate normal space subgroups with the requested point subgroup.

        Moyopy supplies the embedded translationengleiche and klassengleiche
        subgroups. SpinForge filters those candidates to subgroups normal in the
        original parent and converts them to its legacy quotient metadata.

        When ``sublattice`` is not specified, every invariant translation
        sublattice with determinant ``k_index`` is considered.
        """
        if k_index is None:
            if self._sublattice is None:
                raise ValueError("k_index must be specified when sublattice is None.")
            k_index = self._sublattice.order
        if self._sublattice is not None and k_index != self._sublattice.order:
            return []

        normal_space_subgroups = []
        point_subgroup_key = frozenset(normal_point_subgroup)
        for subgroup in self._normal_family_subgroups(k_index):
            if frozenset(subgroup.operation_indices) != point_subgroup_key:
                continue

            translations_by_parent_index = dict(
                zip(subgroup.operation_indices, subgroup.parent_translations, strict=True)
            )
            translations = np.asarray(
                [translations_by_parent_index[index] for index in normal_point_subgroup]
            )
            sublattice = self._sublattice or subgroup.translation_sublattice
            coset_representatives, point_quotient_table = get_coset_representatives(
                normal_point_subgroup, self._table
            )

            adjusted_shifts = translations - self._prim_translations[normal_point_subgroup]
            quotient_table = self._get_quotient_table(
                coset_representatives=coset_representatives,
                point_quotient_table=point_quotient_table,
                sublattice=sublattice,
                normal_point_subgroup=normal_point_subgroup,
                adjusted_shifts=adjusted_shifts,
            )
            if not _is_valid_cayley_table(quotient_table):
                continue

            normal_space_subgroups.append(
                NormalSpaceSubgroup(
                    point_subgroup=normal_point_subgroup,
                    translations=translations,
                    sublattice=sublattice,
                    coset_representatives=coset_representatives,
                    quotient_table=quotient_table,
                )
            )

        return normal_space_subgroups

    def _normal_family_subgroups(
        self,
        k_index: int,
    ) -> tuple[FamilySpaceSubgroup, ...]:
        cached = self._normal_family_subgroups_cache.get(k_index)
        if cached is not None:
            return cached

        normal_subgroups = tuple(
            subgroup
            for subgroup in self._family_subgroup_enumerator.enumerate_normal(
                k_index=k_index,
                up_to_parent_conjugacy=True,
                max_depth=None,
            )
            if subgroup.translation_sublattice.order == k_index
        )
        self._normal_family_subgroups_cache[k_index] = normal_subgroups
        return normal_subgroups

    def _get_quotient_table(
        self,
        coset_representatives: list[GroupElementIndex],
        point_quotient_table: NDArrayInt,
        sublattice: Sublattice,
        normal_point_subgroup: list[GroupElementIndex],
        adjusted_shifts: NDArrayFloat,
    ) -> NDArrayInt:
        factors = [factor for _, _, factor in sublattice.lattice_points]
        num_cosets = len(coset_representatives)
        k_index = sublattice.order
        quotient_order = num_cosets * k_index
        quotient_table = np.zeros((quotient_order, quotient_order), dtype=int)
        inverses: list[GroupElementIndex] = all_inverses(self._table)
        coset_rotation_inverses = [
            np.linalg.inv(self._prim_rotations[g]) for g in coset_representatives
        ]

        tuple_normal_prim_rotations = [
            self._tuple_prim_rotations[g] for g in normal_point_subgroup
        ]

        for (i, gi), (j, gj) in product(enumerate(coset_representatives), repeat=2):
            k: FactorGroupElementIndex = int(point_quotient_table[i, j])
            gk: GroupElementIndex = coset_representatives[k]

            # gi * gj * gk^-1
            # (Ri, ti)(Rj, tj)(Rk, tk)^-1
            #   = (Ri @ Rj, ti + Ri @ tj)(Rk^-1, -Rk^-1 @ tk)
            #   = (Ri @ Rj @ Rk^-1, ti + Ri @ tj - Ri @ Rj @ Rk^-1 @ tk)
            #   =: (r_ijk, t_ijk)
            r_ijk = (
                self._prim_rotations[gi]
                @ self._prim_rotations[gj]
                @ self._prim_rotations[inverses[gk]]
            )
            # (r_ijk, t_ijk) =: (E, t)(Rl, tl) with Rl in normal_point_subgroup
            hl: GroupElementIndex = self._tuple_prim_rotations.index(
                ndarray2d_to_integer_tuple(r_ijk)
            )
            ll = tuple_normal_prim_rotations.index(ndarray2d_to_integer_tuple(r_ijk))

            image = (
                self._factor_system[self._table[gi, gj], hl]
                + self._factor_system[gi, gj]
                + self._prim_rotations[coset_representatives[k]] @ adjusted_shifts[ll]
            )

            factor = sublattice.try_convert_to_factor(image)
            factor_image, _, _ = sublattice.lattice_points[factors.index(factor)]
            for (m, (tm, _, _)), (n, (tn, _, _)) in product(
                enumerate(sublattice.lattice_points),
                repeat=2,
            ):
                quotient_image = (
                    coset_rotation_inverses[k] @ np.asarray(factor_image)
                    + coset_rotation_inverses[j] @ np.asarray(tm)
                    + np.asarray(tn)
                )
                quotient_factor = sublattice.try_convert_to_factor(quotient_image)
                quotient_index = factors.index(quotient_factor)
                quotient_table[i * k_index + m, j * k_index + n] = k * k_index + quotient_index

        return quotient_table


def _is_valid_cayley_table(table: NDArrayInt) -> bool:
    order = len(table)
    elements = np.arange(order)
    if table.shape != (order, order):
        return False
    if not np.all(np.sort(table, axis=0) == elements[:, None]):
        return False
    if not np.all(np.sort(table, axis=1) == elements[None, :]):
        return False

    left = table[table[:, :, None], elements[None, None, :]]
    right = table[elements[:, None, None], table[None, :, :]]
    return bool(np.array_equal(left, right))
