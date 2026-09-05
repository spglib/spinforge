from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from moyopy import SpaceGroup, SpaceGroupType
from spgrep.rep.enumerate import enumerate_unitary_irreps_from_regular_representation
from spgrep.symmetry.group import get_cayley_table
from spgrep.utils import NDArrayFloat, NDArrayInt, ndarray2d_to_integer_tuple
from spinspg.spin import SpinOnlyGroup, SpinOnlyGroupType

from spinforge.irreps import get_real_irreps, get_regular_representation, multi_direct_sum
from spinforge.space_group import (
    NormalSpaceSubgroup,
    NormalSpaceSubgroupEnumerator,
    Sublattice,
    enumerate_normal_groups,
)
from spinforge.utils.combinatorics import combine_dimensions

from ._operation import SpinSymmetryOperations
from ._planochiral import (
    SpinPlanochiral,
    is_spin_planochiral,
)
from ._spin_only_group import embed_real_representation, get_spin_only_group

if TYPE_CHECKING:
    from spinforge.irreps import RealIrrepType


@dataclass(frozen=True)
class NontrivialSpinSpaceGroup:
    # Invariant space subgroup (H)
    invariant_rotations: NDArrayInt
    """Invariant spatial rotation matrices w.r.t. the input cell."""
    invariant_translations: NDArrayFloat
    sublattice: Sublattice
    # Coset of spin translation group over invariant space subgroup (M/H)
    spin_translation_coset: SpinSymmetryOperations
    """Nontrivial spin translation group"""
    # Coset of nontrivial spin space group over spin translation group (M/H)
    nontrivial_coset: SpinSymmetryOperations
    """Nontrivial spin space group"""
    # Whether improper spin rotations exchange distinct coplanar enantiomorphs.
    spin_planochiral: SpinPlanochiral
    # Family translation lattice in the input-cell basis. The identity lattice
    # represents a translationengleiche family.
    family_sublattice: Sublattice = field(
        default_factory=lambda: Sublattice(np.eye(3, dtype=np.int64))
    )

    @property
    def k_index(self) -> int:
        """Translation-lattice index of the invariant group in the input group."""
        return self.sublattice.order

    @property
    def family_k_index(self) -> int:
        """Translation-lattice index of the family group in the input group."""
        return self.family_sublattice.order

    @property
    def relative_k_index(self) -> int:
        """Invariant-lattice index relative to the family translation lattice."""
        if self.k_index % self.family_k_index != 0:
            raise ValueError("Invariant lattice index is not divisible by family lattice index.")
        return self.k_index // self.family_k_index

    @property
    def t_index(self) -> int:
        return self.nontrivial_coset.size

    def transform_to_parent(
        self,
        family_to_parent: Sublattice,
        *,
        atol: float = 1e-5,
    ) -> NontrivialSpinSpaceGroup:
        """Return this family-basis spin space group in parent coordinates.

        ``family_to_parent`` maps coordinates from the current family basis to
        the parent basis. The returned ``family_sublattice`` records that map
        as provenance in the parent basis.
        """
        invariant_rotations, invariant_translations = (
            family_to_parent.transform_operations_to_parent(
                self.invariant_rotations,
                self.invariant_translations,
                atol=atol,
            )
        )
        return NontrivialSpinSpaceGroup(
            invariant_rotations=invariant_rotations,
            invariant_translations=invariant_translations,
            sublattice=Sublattice(
                family_to_parent.transformation @ self.sublattice.transformation
            ),
            spin_translation_coset=self.spin_translation_coset.transform_to_parent(
                family_to_parent, atol=atol
            ),
            nontrivial_coset=self.nontrivial_coset.transform_to_parent(
                family_to_parent, atol=atol
            ),
            spin_planochiral=self.spin_planochiral,
            family_sublattice=family_to_parent,
        )

    def get_full_operations_and_table(self) -> tuple[SpinSymmetryOperations, NDArrayInt]:
        operations, table = self._full_operations_and_table
        return (
            SpinSymmetryOperations(
                rotations=operations.rotations.copy(),
                translations=operations.translations.copy(),
                spin_rotations=operations.spin_rotations.copy(),
            ),
            table.copy(),
        )

    @cached_property
    def _full_operations_and_table(self) -> tuple[SpinSymmetryOperations, NDArrayInt]:
        rotations = []
        translations = []
        spin_rotations = []
        keys = []
        key_indices = {}
        mapping_13 = {}
        for r1, t1, u1 in zip(
            self.nontrivial_coset.rotations,
            self.nontrivial_coset.translations,
            self.nontrivial_coset.spin_rotations,
        ):
            for r3, t3 in zip(
                self.invariant_rotations,
                self.invariant_translations,
            ):
                # (r1, t1)(r3, t3) = (r1 @ r3, r1 @ t3 + t1)
                r13 = r1 @ r3
                t13 = r1 @ t3 + t1
                r13_key = ndarray2d_to_integer_tuple(r13)
                mapping_13[r13_key] = t13

                for t2, u2 in zip(
                    self.spin_translation_coset.translations,
                    self.spin_translation_coset.spin_rotations,
                ):
                    # (r1, t1)(E, t2)(r3, t3) = (r1 @ r3, t1 + r1 @ (t2 + t3))
                    #                         = (E, r1 @ t2) (r13, t13)
                    r123 = r13
                    t123 = r1 @ t2 + t13
                    rotations.append(r123)
                    translations.append(t123)
                    spin_rotations.append(u1 @ u2)

                    key = (
                        r13_key,
                        self.sublattice.try_convert_to_factor(t123 - t13),
                    )
                    key_indices.setdefault(key, len(keys))
                    keys.append(key)

        full_operations = SpinSymmetryOperations(
            rotations=np.array(rotations),
            translations=np.array(translations),
            spin_rotations=np.array(spin_rotations),
        )

        size = full_operations.size
        full_table = np.zeros((size, size), dtype=int)
        for i, j in product(range(size), repeat=2):
            rij = full_operations.rotations[i] @ full_operations.rotations[j]
            tij = (
                full_operations.rotations[i] @ full_operations.translations[j]
                + full_operations.translations[i]
            )

            rij_key = ndarray2d_to_integer_tuple(rij)
            key_ij = (
                rij_key,
                self.sublattice.try_convert_to_factor(tij - mapping_13[rij_key]),
            )
            try:
                full_table[i, j] = key_indices[key_ij]
            except KeyError:
                # Preserve the previous ValueError for an invalid product key.
                full_table[i, j] = keys.index(key_ij)

        return full_operations, full_table

    def get_invariant_space_subgroup_type(self, atol: float = 1e-5) -> SpaceGroupType:
        # The invariant space subgroup has the sublattice as its translation lattice, so
        # transform the operations into the sublattice basis before identification.
        rotations, translations = self.sublattice.transform_operations(
            self.invariant_rotations, self.invariant_translations, atol=atol
        )

        sg = SpaceGroup(
            prim_rotations=rotations.tolist(),
            prim_translations=translations.tolist(),
        )
        return SpaceGroupType(sg.number)

    def get_family_space_group_type(self, atol: float = 1e-5) -> SpaceGroupType:
        """Space-group type of the family space group G (the spatial parts).

        G's coset representatives over its translation lattice are the spatial
        parts of the full SSG operations, deduplicated by rotation (two
        operations sharing a rotation differ by a family-lattice translation).
        The family translation lattice is ``family_sublattice``, so the
        operations are transformed into that basis before identification.
        """
        operations, _ = self._full_operations_and_table
        rotations = []
        translations = []
        seen = set()
        for rotation, translation in zip(operations.rotations, operations.translations):
            key = ndarray2d_to_integer_tuple(rotation)
            if key in seen:
                continue
            seen.add(key)
            rotations.append(rotation)
            translations.append(translation)
        rotations, translations = self.family_sublattice.transform_operations(
            np.array(rotations), np.array(translations), atol=atol
        )

        sg = SpaceGroup(
            prim_rotations=rotations.tolist(),
            prim_translations=translations.tolist(),
        )
        return SpaceGroupType(sg.number)


class SpinSpaceGroupEnumerator:
    def __init__(
        self,
        prim_rotations: Sequence[Sequence[int]],
        prim_translations: Sequence[Sequence[float]],
        *,
        sublattice: Sublattice | None = None,
        atol: float = 1e-5,
    ):
        self._prim_rotations = np.array(prim_rotations)
        self._prim_translations = np.array(prim_translations)
        self._sublattice = sublattice
        self._atol = atol

        self._table = get_cayley_table(self._prim_rotations)
        self._normal_space_subgroup_enumerator = NormalSpaceSubgroupEnumerator(
            prim_rotations=self._prim_rotations,
            prim_translations=self._prim_translations,
            table=self._table,
            sublattice=sublattice,
            atol=self._atol,
        )

        # Enumerate point subgroup of the point group up to conjugacy classes.
        self._list_normal_point_subgroups = enumerate_normal_groups(self._table)

    @property
    def prim_rotations(self) -> NDArrayInt:
        return self._prim_rotations

    @property
    def prim_translations(self) -> NDArrayFloat:
        return self._prim_translations

    @property
    def table(self) -> NDArrayInt:
        """Cayley table of prim_rotations."""
        return self._table

    def enumerate_spin_space_groups(
        self,
        spin_only_group_type: SpinOnlyGroupType,
        k_index: int,
    ) -> tuple[
        SpinOnlyGroup,
        list[tuple[NormalSpaceSubgroup, list[NontrivialSpinSpaceGroup]]],
    ]:
        """Enumerate spin-rotation parts up to O(3), grouped by invariant subgroup.

        The operations supplied to this enumerator define one family space group
        ``G``; this method does not enumerate spatial embeddings of ``G``. For
        this fixed spatial part, it enumerates compatible spin-rotation parts up
        to a global change of spin frame in ``O(3)``. This method does not apply
        a spatial-equivalence relation; parent-normalizer reduction is handled
        downstream.

        Parameters
        ----------
        spin_only_group_type : SpinOnlyGroupType
            Type of spin-only group to pair with the spatial quotient.
        k_index : int
            Klassengleiche index of the invariant space subgroups
            ``H`` normal in ``G`` to enumerate.

        Returns
        -------
        spin_only_group : SpinOnlyGroup
            Spin-only group shared by every enumerated spin space group.
        grouped_spin_space_groups : list of tuple
            Pairs of an invariant space subgroup ``H`` and the spin space group
            representatives constructed with ``H`` as their invariant subgroup.
            The outer order follows :meth:`enumerate_normal_space_subgroups`;
            each inner list follows the enumeration order for its ``H``.

        Notes
        -----
        The grouping lets downstream enumerators act on invariant subgroups
        before classifying their spin assignments. It does not add a new
        equivalence relation or change the representative order. For collinear
        spin-only groups with ``k_index >= 3``, the grouped result is empty.
        """
        spin_only_group = get_spin_only_group(spin_only_group_type)
        if (spin_only_group_type == SpinOnlyGroupType.COLLINEAR) and (k_index >= 3):
            return spin_only_group, []

        list_normal_space_subgroups = self.enumerate_normal_space_subgroups(k_index)
        grouped_spin_space_groups = []
        for normal_space_subgroup in list_normal_space_subgroups:
            _, spin_space_groups = self.enumerate_spin_space_groups_with_normal_space_subgroup(
                spin_only_group_type=spin_only_group_type,
                normal_space_subgroup=normal_space_subgroup,
            )
            grouped_spin_space_groups.append((normal_space_subgroup, spin_space_groups))

        return spin_only_group, grouped_spin_space_groups

    def enumerate_normal_space_subgroups(
        self,
        k_index: int,
    ) -> list[NormalSpaceSubgroup]:
        """Enumerate normal subgroups of given space group with klassengleiche index `k_index`.

        Parameters
        ----------
        k_index: int
            Index of klassengleiche subgroups
        """
        list_normal_space_subgroups = []
        for normal_point_subgroup_fs in self._list_normal_point_subgroups:
            list_normal_space_subgroups_with_point_subgroup = (
                self._normal_space_subgroup_enumerator.enumerate(
                    normal_point_subgroup=list(normal_point_subgroup_fs),
                    k_index=k_index,
                )
            )
            list_normal_space_subgroups.extend(list_normal_space_subgroups_with_point_subgroup)

        return list_normal_space_subgroups

    def enumerate_spin_space_groups_with_normal_space_subgroup(
        self,
        spin_only_group_type: SpinOnlyGroupType,
        normal_space_subgroup: NormalSpaceSubgroup,
    ) -> tuple[SpinOnlyGroup, list[NontrivialSpinSpaceGroup]]:
        spin_only_group = get_spin_only_group(spin_only_group_type)
        if (spin_only_group_type == SpinOnlyGroupType.COLLINEAR) and (
            normal_space_subgroup.k_index >= 3
        ):
            return spin_only_group, []

        # Filter by sublattice if given
        if (self._sublattice is not None) and (
            normal_space_subgroup.sublattice != self._sublattice
        ):
            return spin_only_group, []

        coset_rotations = self._prim_rotations[normal_space_subgroup.coset_representatives]
        k_index = normal_space_subgroup.k_index
        quotient_table = normal_space_subgroup.quotient_table

        reg = get_regular_representation(quotient_table)
        irreps = enumerate_unitary_irreps_from_regular_representation(reg.astype(np.complex128))
        sum_squared_dims = sum(irrep.shape[1] ** 2 for irrep in irreps)
        if sum_squared_dims != len(quotient_table):
            raise RuntimeError(
                f"Irrep enumeration is incomplete: sum of squared dimensions "
                f"{sum_squared_dims} does not match the group order {len(quotient_table)}. "
                "Enumerated spin space groups would be incomplete."
            )

        real_irreps = get_real_irreps(irreps)

        list_spin_space_groups = []
        list_spin_point_groups = self._enumerate_spin_point_groups(
            spin_only_group_type=spin_only_group_type,
            real_irreps=real_irreps,
            atol=self._atol,
        )
        coset_translations = self._prim_translations[normal_space_subgroup.coset_representatives]
        spin_translations = np.array(
            [image for image, _, _ in normal_space_subgroup.sublattice.lattice_points]
        )
        for spin_point_group, real_irrep_types in list_spin_point_groups:
            nssg = NontrivialSpinSpaceGroup(
                invariant_rotations=self._prim_rotations[normal_space_subgroup.point_subgroup],
                invariant_translations=normal_space_subgroup.translations,
                sublattice=normal_space_subgroup.sublattice,
                spin_translation_coset=SpinSymmetryOperations(
                    rotations=np.tile(
                        np.eye(3), reps=(normal_space_subgroup.sublattice.order, 1, 1)
                    ).astype(np.int64),
                    translations=spin_translations,
                    spin_rotations=spin_point_group[:k_index],
                ),
                nontrivial_coset=SpinSymmetryOperations(
                    rotations=coset_rotations,
                    translations=coset_translations,
                    spin_rotations=spin_point_group[::k_index],
                ),
                spin_planochiral=is_spin_planochiral(real_irrep_types),
            )
            _validate_spin_space_group(nssg, quotient_table)
            list_spin_space_groups.append(nssg)

        return spin_only_group, list_spin_space_groups

    @staticmethod
    def _enumerate_spin_point_groups(
        spin_only_group_type: SpinOnlyGroupType,
        real_irreps: list[tuple[NDArrayFloat, RealIrrepType]],
        atol: float,
    ) -> list[tuple[NDArrayFloat, list[RealIrrepType]]]:
        if spin_only_group_type == SpinOnlyGroupType.COLLINEAR:
            rep_dim = 1
        elif spin_only_group_type == SpinOnlyGroupType.COPLANAR:
            rep_dim = 2
        elif spin_only_group_type == SpinOnlyGroupType.NONCOPLANAR:
            rep_dim = 3
        else:
            raise ValueError(f"Unsupported spin only group type: {spin_only_group_type}")

        list_spin_point_groups = []
        combinations = combine_dimensions(
            dimensions=[real_irrep.shape[1] for real_irrep, _ in real_irreps], max_dim=rep_dim
        )[rep_dim]
        for comb in combinations:
            rep, real_irrep_types = multi_direct_sum(real_irreps, comb)
            if rep.shape[1] != rep_dim:
                raise ValueError(
                    f"Combined representation dimension {rep.shape[1]} does not match "
                    f"expected dimension {rep_dim}"
                )
            sum_dims = sum(rit.dim for rit in real_irrep_types)
            if sum_dims != rep_dim:
                raise ValueError(
                    f"Sum of real irrep dimensions {sum_dims} does not match "
                    f"expected dimension {rep_dim}"
                )

            # Filter faithful representation
            if sum([np.allclose(mat, np.eye(rep_dim), atol=atol) for mat in rep]) >= 2:
                continue

            spin_point_group = embed_real_representation(
                spin_only_group_type=spin_only_group_type,
                rep=rep,
            )
            list_spin_point_groups.append((spin_point_group, real_irrep_types))

        return list_spin_point_groups


def _validate_spin_space_group(
    nssg: NontrivialSpinSpaceGroup,
    quotient_table: NDArrayInt,
) -> None:
    spin_rotations = np.asarray(
        [
            coset_rotation @ translation_rotation
            for coset_rotation in nssg.nontrivial_coset.spin_rotations
            for translation_rotation in nssg.spin_translation_coset.spin_rotations
        ]
    )
    for i, j in product(range(len(quotient_table)), repeat=2):
        k = int(quotient_table[i, j])
        if not np.allclose(spin_rotations[i] @ spin_rotations[j], spin_rotations[k]):
            raise ValueError(
                f"Spin rotations are not consistent with the quotient table at pair ({i}, {j})"
            )
