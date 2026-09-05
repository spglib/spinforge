"""Hermann decomposition for family space subgroups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from moyopy import (
    enumerate_klassengleiche_subgroups,
    enumerate_translationengleiche_subgroups,
)
from spgrep.symmetry.group import get_cayley_table
from spgrep.utils import NDArrayFloat, NDArrayInt, is_integer_array

from spinforge.utils.group_theory import (
    GroupElementIndex,
    SubgroupIndices,
    all_inverses,
)

from ._normalizer import (
    ParentNormalizerAction,
    compute_parent_normalizer_action,
    space_group_normalizer,
)
from ._sublattice import Sublattice, enumerate_normal_sublattices


def _enumerate_translationengleiche_classes(
    rotations: NDArrayInt,
    translations: NDArrayFloat,
    *,
    epsilon: float,
) -> tuple[
    list[tuple[SubgroupIndices, tuple[SubgroupIndices, ...], int]],
    frozenset[SubgroupIndices],
]:
    subgroup_classes = enumerate_translationengleiche_subgroups(
        rotations.tolist(),
        translations.tolist(),
        epsilon=epsilon,
    )
    class_data = [
        (
            SubgroupIndices(item.representative.operation_indices),
            tuple(
                SubgroupIndices(conjugate.subgroup.operation_indices)
                for conjugate in item.conjugates
            ),
            item.representative.depth,
        )
        for item in subgroup_classes
    ]
    representatives = frozenset(representative for representative, _, _ in class_data)
    return class_data, representatives


@dataclass(frozen=True, eq=False)
class FamilySpaceSubgroup:
    """A family space subgroup and its Hermann translationengleiche supergroup.

    The affine operations of the family subgroup are given both in the parent
    basis (`parent_rotations` and `parent_translations`) and in the family
    translation-lattice basis (`rotations` and `translations`).
    `operation_indices` map those operations to the input parent-operation
    order.

    `hermann_subgroup` is the intermediate translationengleiche group
    `M` in a Hermann chain `G' <= M <= G`. Its conjugacy class is under the
    parent space group `G`. `hermann_normalizer` contains representatives
    of `N_G(M) / M` in the parent-operation order. Exact-lattice
    klassengleiche subgroups of a fixed `M` are classified by moyopy under
    conjugation by `M`. ``normalizer_action`` contains representatives of the
    full parent-space-group quotient ``N_G(G′) / G′``, including distinct parent-lattice
    translations modulo the translation lattice of ``G′``.

    `depth` is the shortest covering-chain length from `G` to `M` in the
    translationengleiche subgroup lattice. The parent has depth zero and its
    maximal proper translationengleiche subgroups have depth one. For a
    bounded family subgroup, `depth` describes `M`, not the full subgroup.
    """

    hermann_subgroup: SubgroupIndices
    hermann_conjugates: tuple[SubgroupIndices, ...]
    hermann_normalizer: tuple[GroupElementIndex, ...]
    depth: int
    operation_indices: tuple[int, ...]
    translation_sublattice: Sublattice
    parent_rotations: NDArrayInt
    parent_translations: NDArrayFloat
    rotations: NDArrayInt
    translations: NDArrayFloat
    relative_sublattice: Sublattice | None
    normalizer_action: ParentNormalizerAction
    relative_k_index: int | None = None

    @property
    def is_translationengleiche(self) -> bool:
        return self.translation_sublattice.order == 1

    @property
    def translation_key(self) -> tuple[int, ...]:
        return tuple(self.translation_sublattice.transformation.ravel().tolist())

    def is_subgroup_of(self, other: FamilySpaceSubgroup, *, atol: float) -> bool:
        """Return whether this affine family group is contained in `other`."""
        if not other.translation_sublattice.contains(
            self.translation_sublattice,
            atol=atol,
        ):
            return False

        for rotation, translation in zip(self.parent_rotations, self.parent_translations):
            matches = np.all(other.parent_rotations == rotation, axis=(1, 2))
            if not np.any(matches):
                return False
            if not any(
                is_integer_array(
                    other.translation_sublattice.inverse_transformation
                    @ (translation - other_translation),
                    atol=atol,
                )
                for other_translation in other.parent_translations[matches]
            ):
                return False
        return True


class FamilySpaceSubgroupEnumerator:
    """Enumerate family space subgroups through Hermann decomposition.

    For a parent space group `G`, each result is obtained from a chain
    `G' <= M <= G` in which `M` is translationengleiche in `G` and
    `G'` is klassengleiche in `M`.

    With ``up_to_parent_conjugacy=True``, translationengleiche groups `M` are
    identified under conjugation by `G`, restricted to the stabilizer
    `{g in G | gL = L}` when a target translation lattice `L` is fixed. For a
    fixed `M`, candidate family lattices are identified under the rotation
    action of `N_G(M) / M`; with a target `L`, this action is likewise restricted
    to `{g in N_G(M) | gL = L}`. For a fixed `M` and family lattice, moyopy
    identifies exact-lattice klassengleiche subgroups under conjugation by `M`.

    Every result carries the action of its normalizer in the parent space group,
    ``N_G(G′) / G′``. With ``up_to_parent_conjugacy=False``, every translationengleiche
    and klassengleiche conjugate is emitted and the family-lattice normalizer
    reduction is skipped.

    This class handles only space-group operations and lattices. Magnetic-site
    multiplicity and spin-space-group construction belong to the configuration
    layer.

    Parameters
    ----------
    parent_rotations, parent_translations:
        Operations of the parent space group `G`.
    epsilon:
        Tolerance passed to moyopy's translationengleiche enumeration.
    target_sublattice:
        Optional propagation-vector-derived lattice that every enumerated
        family translation lattice must contain.
    atol:
        Tolerance for integral lattice and affine-operation comparisons.
    """

    def __init__(
        self,
        parent_rotations: NDArrayInt,
        parent_translations: NDArrayFloat,
        *,
        epsilon: float,
        target_sublattice: Sublattice | None,
        atol: float,
    ) -> None:
        self._parent_rotations = np.asarray(parent_rotations, dtype=np.int64)
        self._parent_translations = np.asarray(parent_translations, dtype=float)
        self._target_sublattice = target_sublattice
        self._atol = atol

        class_data, self._representative_subgroups = _enumerate_translationengleiche_classes(
            self._parent_rotations,
            self._parent_translations,
            epsilon=epsilon,
        )
        parent_table = get_cayley_table(self._parent_rotations)
        self._parent_table = parent_table
        self._parent_inverses = all_inverses(parent_table)
        primitive_sublattice = Sublattice(np.eye(3, dtype=np.int64))
        self._translationengleiche_subgroups: list[FamilySpaceSubgroup] = []
        self._selected_hermann_subgroups_cache: dict[bool, tuple[FamilySpaceSubgroup, ...]] = {}
        self._bounded_subgroups_cache: dict[tuple[int, bool], tuple[FamilySpaceSubgroup, ...]] = {}

        for representative, conjugates, depth in sorted(
            class_data,
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            for subgroup in conjugates:
                operation_indices = tuple(sorted(subgroup))
                rotations = self._parent_rotations[list(operation_indices)]
                translations = self._parent_translations[list(operation_indices)]
                hermann_normalizer = tuple(space_group_normalizer(parent_table, subgroup))
                normalizer_action = compute_parent_normalizer_action(
                    self._parent_rotations,
                    self._parent_translations,
                    hermann_operation_indices=operation_indices,
                    hermann_normalizer=hermann_normalizer,
                    subgroup_rotations=rotations,
                    subgroup_translations=translations,
                    translation_sublattice=primitive_sublattice,
                    atol=self._atol,
                )
                self._translationengleiche_subgroups.append(
                    FamilySpaceSubgroup(
                        hermann_subgroup=subgroup,
                        hermann_conjugates=conjugates if subgroup == representative else (),
                        hermann_normalizer=hermann_normalizer,
                        depth=depth,
                        operation_indices=operation_indices,
                        translation_sublattice=primitive_sublattice,
                        parent_rotations=rotations,
                        parent_translations=translations,
                        rotations=rotations,
                        translations=translations,
                        relative_sublattice=target_sublattice,
                        normalizer_action=normalizer_action,
                    )
                )

    @property
    def parent_rotations(self) -> NDArrayInt:
        return self._parent_rotations

    @property
    def parent_translations(self) -> NDArrayFloat:
        return self._parent_translations

    @property
    def representative_subgroups(self) -> frozenset[SubgroupIndices]:
        """Representatives of t-subgroups under parent-space-group conjugacy."""
        return self._representative_subgroups

    @property
    def translationengleiche_subgroups(self) -> tuple[FamilySpaceSubgroup, ...]:
        return tuple(self._translationengleiche_subgroups)

    def enumerate(
        self,
        *,
        k_index: int = 1,
        up_to_parent_conjugacy: bool = True,
        max_depth: int | None = None,
    ) -> list[FamilySpaceSubgroup]:
        """Enumerate subgroups satisfying the requested bounds.

        ``max_depth`` bounds the Hermann translationengleiche depth, not the
        depth of the full family subgroup. In particular, ``max_depth=1``
        retains the parent and maximal proper Hermann subgroups, together with
        bounded descendants of those Hermann groups. This does not test affine
        maximality among the emitted family subgroups.

        Translationengleiche subgroups are returned first, ordered by decreasing
        Hermann-group order. Bounded subgroups follow, ordered by increasing
        family-lattice index and then decreasing Hermann-group order.
        """
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be >= 0 or None, got {max_depth}")
        selected_hermann_subgroups = self._select_hermann_subgroups(
            up_to_parent_conjugacy=up_to_parent_conjugacy
        )
        translationengleiche_subgroups = [
            subgroup
            for subgroup in selected_hermann_subgroups
            if max_depth is None or subgroup.depth <= max_depth
        ]
        if k_index == 1:
            return translationengleiche_subgroups

        cache_key = (k_index, up_to_parent_conjugacy)
        bounded_subgroups = self._bounded_subgroups_cache.get(cache_key)
        if bounded_subgroups is None:
            bounded_subgroups = tuple(
                self._enumerate_bounded_subgroups(
                    k_index=k_index,
                    hermann_subgroups=selected_hermann_subgroups,
                    up_to_parent_conjugacy=up_to_parent_conjugacy,
                )
            )
            self._bounded_subgroups_cache[cache_key] = bounded_subgroups

        selected_bounded_subgroups = [
            subgroup
            for subgroup in bounded_subgroups
            if max_depth is None or subgroup.depth <= max_depth
        ]
        selected_bounded_subgroups.sort(
            key=lambda subgroup: (
                subgroup.translation_sublattice.order,
                -len(subgroup.hermann_subgroup),
            )
        )
        return [*translationengleiche_subgroups, *selected_bounded_subgroups]

    def _select_hermann_subgroups(
        self,
        *,
        up_to_parent_conjugacy: bool,
    ) -> list[FamilySpaceSubgroup]:
        cached = self._selected_hermann_subgroups_cache.get(up_to_parent_conjugacy)
        if cached is not None:
            return list(cached)
        if not up_to_parent_conjugacy:
            selected = list(self._translationengleiche_subgroups)
        elif self._target_sublattice is None:
            selected = [
                subgroup
                for subgroup in self._translationengleiche_subgroups
                if subgroup.hermann_subgroup in self._representative_subgroups
            ]
        else:
            stabilizer = [
                index
                for index, rotation in enumerate(self._parent_rotations)
                if self._target_sublattice.is_normal(rotation)
            ]
            selected = []
            for subgroup in self._translationengleiche_subgroups:
                if any(
                    _are_conjugate_subgroups(
                        subgroup.hermann_subgroup,
                        representative.hermann_subgroup,
                        stabilizer,
                        self._parent_table,
                        inverses=self._parent_inverses,
                    )
                    for representative in selected
                ):
                    continue
                selected.append(subgroup)
        self._selected_hermann_subgroups_cache[up_to_parent_conjugacy] = tuple(selected)
        return list(selected)

    def enumerate_normal(
        self,
        *,
        k_index: int = 1,
        up_to_parent_conjugacy: bool = True,
        max_depth: int | None = None,
    ) -> list[FamilySpaceSubgroup]:
        """Enumerate bounded family subgroups normal in the full parent group.

        A candidate ``G′`` is normal precisely when its stored parent
        normalizer quotient exhausts the parent quotient, that is,
        ``|N_G(G′) / G′| = [G : G′]``. This tests normality in the original
        parent ``G``, including for general subgroups whose Hermann group is a
        proper subgroup of ``G``.

        ``k_index`` and ``max_depth`` have the same finite-bound semantics as
        :meth:`enumerate`.
        """
        if k_index < 1:
            raise ValueError(f"k_index must be >= 1, got {k_index}")
        return [
            subgroup
            for subgroup in self.enumerate(
                k_index=k_index,
                up_to_parent_conjugacy=up_to_parent_conjugacy,
                max_depth=max_depth,
            )
            if len(subgroup.normalizer_action)
            == (len(self._parent_rotations) // len(subgroup.parent_rotations))
            * subgroup.translation_sublattice.order
        ]

    def _enumerate_bounded_subgroups(
        self,
        *,
        k_index: int,
        hermann_subgroups: list[FamilySpaceSubgroup],
        up_to_parent_conjugacy: bool,
    ) -> list[FamilySpaceSubgroup]:
        subgroups = []
        for hermann in hermann_subgroups:
            family_sublattices = self._family_sublattices(
                hermann.parent_rotations,
                k_index,
            )
            if up_to_parent_conjugacy:
                family_sublattices = self._reduce_family_sublattices(
                    family_sublattices,
                    hermann.hermann_normalizer,
                )
            for family_sublattice, relative_sublattice in family_sublattices:
                subgroup_classes = enumerate_klassengleiche_subgroups(
                    hermann.parent_rotations.tolist(),
                    hermann.parent_translations.tolist(),
                    family_sublattice.transformation.tolist(),
                    epsilon=self._atol,
                )
                for subgroup_class in subgroup_classes:
                    if up_to_parent_conjugacy:
                        klassengleiche_subgroups = [subgroup_class.representative]
                    else:
                        klassengleiche_subgroups = [
                            conjugate.subgroup for conjugate in subgroup_class.conjugates
                        ]
                    for subgroup in klassengleiche_subgroups:
                        operations = subgroup.operations
                        rotations = np.rint(operations.rotations).astype(np.int64)
                        translations = np.asarray(operations.translations, dtype=float)
                        parent_operations = subgroup.parent_operations
                        parent_rotations = np.rint(parent_operations.rotations).astype(np.int64)
                        parent_translations = np.asarray(
                            parent_operations.translations,
                            dtype=float,
                        )
                        translation_sublattice = Sublattice(
                            np.asarray(subgroup.transformation, dtype=np.int64)
                        )
                        operation_indices = tuple(
                            hermann.operation_indices[index]
                            for index in subgroup.operation_indices
                        )
                        normalizer_action = compute_parent_normalizer_action(
                            self._parent_rotations,
                            self._parent_translations,
                            hermann_operation_indices=hermann.operation_indices,
                            hermann_normalizer=hermann.hermann_normalizer,
                            subgroup_rotations=parent_rotations,
                            subgroup_translations=parent_translations,
                            translation_sublattice=translation_sublattice,
                            atol=self._atol,
                        )
                        subgroups.append(
                            FamilySpaceSubgroup(
                                hermann_subgroup=hermann.hermann_subgroup,
                                hermann_conjugates=hermann.hermann_conjugates,
                                hermann_normalizer=hermann.hermann_normalizer,
                                depth=hermann.depth,
                                operation_indices=operation_indices,
                                translation_sublattice=translation_sublattice,
                                parent_rotations=parent_rotations,
                                parent_translations=parent_translations,
                                rotations=rotations,
                                translations=translations,
                                relative_sublattice=relative_sublattice,
                                normalizer_action=normalizer_action,
                                relative_k_index=(k_index // subgroup.klassengleiche_index),
                            )
                        )
        return subgroups

    def _family_sublattices(
        self,
        rotations: NDArrayInt,
        k_index: int,
    ) -> list[tuple[Sublattice, Sublattice | None]]:
        family_sublattices = []
        for family_index in range(2, k_index + 1):
            if k_index % family_index != 0:
                continue
            for candidate in enumerate_normal_sublattices(rotations, family_index):
                relative_sublattice = self._relative_sublattice(candidate)
                if self._target_sublattice is not None and relative_sublattice is None:
                    continue
                family_sublattices.append((candidate, relative_sublattice))
        return family_sublattices

    def _reduce_family_sublattices(
        self,
        family_sublattices: list[tuple[Sublattice, Sublattice | None]],
        normalizer: tuple[GroupElementIndex, ...],
    ) -> list[tuple[Sublattice, Sublattice | None]]:
        """Choose lattice-orbit representatives under `N_G(M) / M`."""
        normalizer_rotations = [self._parent_rotations[index] for index in normalizer]
        if self._target_sublattice is not None:
            normalizer_rotations = [
                rotation
                for rotation in normalizer_rotations
                if self._target_sublattice.is_normal(rotation)
            ]

        representatives: list[tuple[Sublattice, Sublattice | None]] = []
        for candidate in family_sublattices:
            candidate_sublattice, _ = candidate
            if any(
                _are_conjugate_sublattices(
                    candidate_sublattice,
                    representative_sublattice,
                    normalizer_rotations,
                    atol=self._atol,
                )
                for representative_sublattice, _ in representatives
            ):
                continue
            representatives.append(candidate)
        return representatives

    def _relative_sublattice(self, family_sublattice: Sublattice) -> Sublattice | None:
        if self._target_sublattice is None:
            return None
        return family_sublattice.relative_sublattice(
            self._target_sublattice,
            atol=self._atol,
        )


def _are_conjugate_sublattices(
    left: Sublattice,
    right: Sublattice,
    normalizer_rotations: Sequence[NDArrayInt],
    *,
    atol: float,
) -> bool:
    return any(
        is_integer_array(
            left.inverse_transformation @ rotation @ right.transformation,
            atol=atol,
        )
        for rotation in normalizer_rotations
    )


def _are_conjugate_subgroups(
    left: SubgroupIndices,
    right: SubgroupIndices,
    conjugators: Sequence[GroupElementIndex],
    table: NDArrayInt,
    *,
    inverses: Sequence[GroupElementIndex] | None = None,
) -> bool:
    """Return whether two point subgroups are conjugate by an allowed parent operation."""
    if inverses is None:
        inverses = all_inverses(table)
    for conjugator in conjugators:
        inverse = inverses[conjugator]
        conjugated = frozenset(
            int(table[table[conjugator, element], inverse]) for element in right
        )
        if conjugated == left:
            return True
    return False
