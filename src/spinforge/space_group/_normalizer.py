"""Normalizer of a subgroup of a space group and related helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from spgrep.rep.group import get_inverse_index
from spgrep.utils import NDArrayFloat, NDArrayInt, is_integer_array

from spinforge.utils._equivalence import _classify_equivalence_classes, _EquivalenceClass
from spinforge.utils.group_theory import GroupElementIndex, SubgroupIndices

from ._normal_space_subgroup import NormalSpaceSubgroup
from ._sublattice import Sublattice


@dataclass(frozen=True, eq=False)
class ParentNormalizerAction:
    """Representatives and induced action of ``N_G(G′) / G′``.

    The conjugating operations belong to the parent space group ``G``.
    ``rotations`` and ``translations`` are expressed in the parent primitive
    basis. ``permutations[a][i]`` is the index of the family operation
    obtained from operation ``i`` by inverse conjugation with representative
    ``a``.
    """

    rotations: NDArrayInt
    translations: NDArrayFloat
    permutations: tuple[tuple[int, ...], ...]

    def __len__(self) -> int:
        return len(self.rotations)


def space_group_normalizer(
    table: NDArrayInt,
    subgroup: Iterable[GroupElementIndex],
) -> list[GroupElementIndex]:
    """Compute coset representatives of ``N_G(G') / G'``.

    Returns one element per left coset of ``G'`` in its normalizer
    ``N_G(G') = { h in G : h G' h^{-1} = G' }``.

    Parameters
    ----------
    table:
        Cayley table of the parent group ``G``.
    subgroup:
        Indices of the elements of ``G'`` in the parent group.

    Returns
    -------
    list[GroupElementIndex]
        One representative per coset of ``N_G(G') / G'``.
        Always contains the identity element.
    """
    subgroup_set: SubgroupIndices = frozenset(int(g) for g in subgroup)
    order = len(table)

    normalizer_elements: list[GroupElementIndex] = []
    for h in range(order):
        h_inv = get_inverse_index(table, h)
        conjugated = frozenset(
            int(table[table[h, g], h_inv])
            for g in subgroup_set  # h * g * h^{-1}
        )
        if conjugated == subgroup_set:
            normalizer_elements.append(h)

    # Extract one representative per left coset of G' in N_G(G').
    coset_reps: list[GroupElementIndex] = []
    covered: set[GroupElementIndex] = set()
    for h in normalizer_elements:
        if h in covered:
            continue
        coset_reps.append(h)
        for g in subgroup_set:
            covered.add(int(table[h, g]))  # h * g
    return coset_reps


def compute_parent_normalizer_action(
    parent_rotations: NDArrayInt,
    parent_translations: NDArrayFloat,
    hermann_operation_indices: Sequence[GroupElementIndex],
    hermann_normalizer: Sequence[GroupElementIndex],
    subgroup_rotations: NDArrayInt,
    subgroup_translations: NDArrayFloat,
    translation_sublattice: Sublattice,
    *,
    atol: float,
) -> ParentNormalizerAction:
    """Return the action of ``N_G(G′) / G′`` for ``G′ <= M <= G``.

    Since ``M = G′T`` and the parent translation group ``T`` is normal in
    ``G``, every element of ``N_G(G′)`` also normalizes ``M``. Representatives of
    ``N_G(M) / T′`` are formed from ``N_G(M) / M``, ``M / T``, and
    ``T / T′`` before retaining the stabilizer of ``G′`` and reducing it
    modulo ``G′``. When ``T′ = T``, the supplied Hermann-normalizer coset
    representatives already are the required representatives, so only their
    induced permutations are computed.
    """
    candidate_rotations: list[NDArrayInt] = []
    candidate_translations: list[NDArrayFloat] = []
    is_translationengleiche = translation_sublattice.order == 1
    if is_translationengleiche:
        representative_indices = list(hermann_normalizer)
        candidate_rotations.extend(parent_rotations[representative_indices])
        candidate_translations.extend(parent_translations[representative_indices])
    else:
        lattice_images = [
            np.asarray(image, dtype=float) for image, _, _ in translation_sublattice.lattice_points
        ]
        for normalizer_index in hermann_normalizer:
            normalizer_rotation = parent_rotations[normalizer_index]
            normalizer_translation = parent_translations[normalizer_index]
            for operation_index in hermann_operation_indices:
                operation_rotation = parent_rotations[operation_index]
                operation_translation = parent_translations[operation_index]
                for lattice_image in lattice_images:
                    candidate_rotations.append(normalizer_rotation @ operation_rotation)
                    candidate_translations.append(
                        normalizer_translation
                        + normalizer_rotation
                        @ (operation_translation + operation_rotation @ lattice_image)
                    )

    normalizer_rotations: list[NDArrayInt] = []
    normalizer_translations: list[NDArrayFloat] = []
    normalizer_permutations: list[tuple[int, ...]] = []
    for rotation, translation in zip(candidate_rotations, candidate_translations):
        if not translation_sublattice.is_normal(rotation, atol=atol):
            continue
        permutation = _normalizer_permutation(
            rotation,
            translation,
            subgroup_rotations,
            subgroup_translations,
            translation_sublattice,
            atol=atol,
        )
        if permutation is None:
            if is_translationengleiche:
                raise RuntimeError("Hermann normalizer representative failed to normalize G′.")
            continue
        normalizer_rotations.append(rotation)
        normalizer_translations.append(translation)
        normalizer_permutations.append(permutation)

    if is_translationengleiche:
        representative_indices = list(range(len(normalizer_rotations)))
    else:
        representative_indices = []
        covered: set[int] = set()
        for index, (rotation, translation) in enumerate(
            zip(normalizer_rotations, normalizer_translations)
        ):
            if index in covered:
                continue
            representative_indices.append(index)
            for subgroup_rotation, subgroup_translation in zip(
                subgroup_rotations,
                subgroup_translations,
            ):
                product_index = _find_operation_modulo_sublattice(
                    rotation @ subgroup_rotation,
                    translation + rotation @ subgroup_translation,
                    normalizer_rotations,
                    normalizer_translations,
                    translation_sublattice,
                    atol=atol,
                )
                if product_index is None:
                    raise RuntimeError("Normalizer coset product escaped N_G(G′).")
                covered.add(product_index)

    return ParentNormalizerAction(
        rotations=np.asarray([normalizer_rotations[index] for index in representative_indices]),
        translations=np.asarray(
            [normalizer_translations[index] for index in representative_indices]
        ),
        permutations=tuple(normalizer_permutations[index] for index in representative_indices),
    )


def _normalizer_permutation(
    rotation: NDArrayInt,
    translation: NDArrayFloat,
    subgroup_rotations: NDArrayInt,
    subgroup_translations: NDArrayFloat,
    translation_sublattice: Sublattice,
    *,
    atol: float,
) -> tuple[int, ...] | None:
    """Return the permutation induced by inverse conjugation, if normalizing."""
    inverse_rotation = np.rint(np.linalg.inv(rotation)).astype(np.int64)
    permutation = []
    for subgroup_rotation, subgroup_translation in zip(subgroup_rotations, subgroup_translations):
        conjugated_rotation = inverse_rotation @ subgroup_rotation @ rotation
        conjugated_translation = _inverse_conjugate_translation(
            subgroup_rotation,
            subgroup_translation,
            inverse_rotation,
            translation,
        )
        image = _find_operation_modulo_sublattice(
            conjugated_rotation,
            conjugated_translation,
            subgroup_rotations,
            subgroup_translations,
            translation_sublattice,
            atol=atol,
        )
        if image is None:
            return None
        permutation.append(image)
    return tuple(permutation)


def _inverse_conjugate_translation(
    operation_rotation: NDArrayInt,
    operation_translation: NDArrayFloat,
    inverse_conjugator_rotation: NDArrayInt,
    conjugator_translation: NDArrayFloat,
) -> NDArrayFloat:
    """Return the translation part of ``h^-1 g h``."""
    return inverse_conjugator_rotation @ (
        operation_translation
        - conjugator_translation
        + operation_rotation @ conjugator_translation
    )


def _find_operation_modulo_sublattice(
    rotation: NDArrayInt,
    translation: NDArrayFloat,
    rotations: NDArrayInt | list[NDArrayInt],
    translations: NDArrayFloat | list[NDArrayFloat],
    translation_sublattice: Sublattice,
    *,
    atol: float,
) -> int | None:
    """Find an affine operation modulo ``translation_sublattice``."""
    for index, (candidate_rotation, candidate_translation) in enumerate(
        zip(rotations, translations)
    ):
        if not np.array_equal(rotation, candidate_rotation):
            continue
        if is_integer_array(
            translation_sublattice.inverse_transformation @ (translation - candidate_translation),
            atol=atol,
        ):
            return index
    return None


def _are_conjugated_nss_equal(
    conj_point_subgroup: list[int],
    conj_translations: NDArrayFloat,
    conj_sublattice: NDArrayInt,
    ref: NormalSpaceSubgroup,
    *,
    atol: float,
) -> bool:
    """Test whether a conjugated NSS triple equals ``ref``.

    Comparison is tolerance-based for translations and exact for the
    point-subgroup index set. The sublattice is compared by checking that the
    two integer matrices span the same Z^3-sublattice.
    """
    if frozenset(conj_point_subgroup) != frozenset(ref.point_subgroup):
        return False

    # Sublattice equality: L1 and L2 span the same sublattice iff
    # L2^{-1} @ L1 is unimodular (integer with det +/-1).
    relative = ref.sublattice.inverse_transformation @ conj_sublattice
    if not is_integer_array(relative, atol=atol):
        return False
    if not np.isclose(abs(np.linalg.det(relative)), 1.0, atol=atol):
        return False

    # Compare translations: for each point-subgroup element, the translations
    # must agree modulo the sublattice L (not Z^3).
    idx_ref = {int(g): i for i, g in enumerate(ref.point_subgroup)}
    for g, t_conj in zip(conj_point_subgroup, conj_translations):
        t_ref = ref.translations[idx_ref[int(g)]]
        diff = np.asarray(t_conj) - np.asarray(t_ref)
        if not is_integer_array(diff, atol=atol):
            return False
        zero_factor = (0, 0, 0)
        if ref.sublattice.try_convert_to_factor(diff) != zero_factor:
            return False
    return True


def _conjugate_nss(
    nss: NormalSpaceSubgroup,
    family_rotations: NDArrayInt,
    normalizer_rotation: NDArrayInt,
    normalizer_translation: NDArrayFloat,
    normalizer_permutation: Sequence[int],
) -> tuple[list[int], NDArrayFloat, NDArrayInt]:
    """Inverse-conjugate ``nss`` by one normalizer representative.

    All operations are expressed in the family primitive basis, and the
    permutation acts on the family operation order.
    """
    inverse_rotation = np.rint(np.linalg.inv(normalizer_rotation)).astype(np.int64)

    new_point_subgroup: list[int] = []
    new_translations: list[NDArrayFloat] = []
    for family_index, translation in zip(nss.point_subgroup, nss.translations):
        rotation = family_rotations[family_index]
        new_point_subgroup.append(int(normalizer_permutation[family_index]))
        new_translations.append(
            _inverse_conjugate_translation(
                rotation,
                translation,
                inverse_rotation,
                normalizer_translation,
            )
        )

    new_sublattice = np.rint(inverse_rotation @ nss.sublattice.transformation).astype(np.int64)
    return new_point_subgroup, np.asarray(new_translations), new_sublattice


def classify_normal_space_subgroup_conjugacy_classes(
    list_nss: list[NormalSpaceSubgroup],
    family_rotations: NDArrayInt,
    normalizer_rotations: NDArrayInt,
    normalizer_translations: NDArrayFloat,
    normalizer_permutations: Sequence[Sequence[int]],
    *,
    atol: float = 1e-5,
) -> list[_EquivalenceClass[NormalSpaceSubgroup, int]]:
    """Classify normal space subgroups under a parent-normalizer action.

    Parameters
    ----------
    list_nss:
        Normal space subgroups enumerated for a fixed family ``G′``.
    family_rotations:
        Rotation matrices of ``G′`` in the family primitive basis.
    normalizer_rotations, normalizer_translations:
        Representatives of ``N_G(G′) / G′`` in the family primitive basis.
    normalizer_permutations:
        Inverse-conjugation permutations on the family operation order, aligned
        with the normalizer operations.
    atol:
        Tolerance for comparing fractional translations.

    Returns
    -------
    equivalence_classes:
        Equivalent input objects with aligned normalizer indices for inverse-conjugating
        each object to its representative, plus the representative's stabilizer.
    """
    if not (
        len(normalizer_rotations) == len(normalizer_translations) == len(normalizer_permutations)
    ):
        raise ValueError("Normalizer operations and permutations must be aligned.")
    normalizer_indices = range(len(normalizer_rotations))

    def _conjugate_and_compare(
        nss: NormalSpaceSubgroup,
        ref: NormalSpaceSubgroup,
        normalizer_index: int,
    ) -> bool:
        conj_ps, conj_t, conj_sl = _conjugate_nss(
            nss=nss,
            family_rotations=family_rotations,
            normalizer_rotation=normalizer_rotations[normalizer_index],
            normalizer_translation=normalizer_translations[normalizer_index],
            normalizer_permutation=normalizer_permutations[normalizer_index],
        )
        return _are_conjugated_nss_equal(conj_ps, conj_t, conj_sl, ref, atol=atol)

    def _find_conjugating_normalizer_index(
        nss: NormalSpaceSubgroup,
        representative: NormalSpaceSubgroup,
    ) -> int | None:
        return next(
            (
                normalizer_index
                for normalizer_index in normalizer_indices
                if _conjugate_and_compare(nss, representative, normalizer_index)
            ),
            None,
        )

    def _find_stabilizer(nss: NormalSpaceSubgroup) -> list[int]:
        return [
            normalizer_index
            for normalizer_index in normalizer_indices
            if _conjugate_and_compare(nss, nss, normalizer_index)
        ]

    return _classify_equivalence_classes(
        list_nss, _find_conjugating_normalizer_index, _find_stabilizer
    )
