from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from spgrep.utils import NDArrayFloat, NDArrayInt

from spinforge.utils._equivalence import _classify_equivalence_classes, _EquivalenceClass
from spinforge.utils.group_theory import all_inverses


class ConstructType(StrEnum):
    TYPE1 = "Type-1"
    TYPE3 = "Type-3"
    TYPE4 = "Type-4"


@dataclass
class MagneticSpaceSubgroup:
    xsg: list[int]
    fsg: list[int]
    msg_type: ConstructType
    Q: NDArrayFloat

    def orient_moments_basis(
        self, magnetic_moments_basis: Sequence[NDArrayFloat]
    ) -> list[NDArrayFloat]:
        """Rotate each Cartesian moment-basis field into this subgroup's frame.

        Each field of shape ``(n_sites, 3)`` transforms row-wise by the
        orientation rotation ``Q`` (``field @ Q^T``), giving the
        symmetry-adapted basis expressed in the magnetic space subgroup's
        oriented setting.
        """
        return [np.asarray(field) @ self.Q.T for field in magnetic_moments_basis]


def classify_construct_type(
    xsg: Sequence[int],
    fsg: Sequence[int],
    is_identity_rotation: Sequence[bool],
) -> ConstructType:
    """Classify an MSG as Type-1/3/4 from its XSG and FSG index sets.

    ``is_identity_rotation[g]`` must be True iff operation ``g`` has an identity
    spatial rotation (i.e. is a pure translation).
    """
    if len(fsg) == len(xsg):
        return ConstructType.TYPE1
    if len(fsg) != 2 * len(xsg):
        raise ValueError(
            f"Invalid subgroup structure: len(fsg)={len(fsg)}, len(xsg)={len(xsg)}. "
            "Expected len(fsg) == len(xsg) or len(fsg) == 2 * len(xsg)."
        )
    if any(is_identity_rotation[g] for g in set(fsg) - set(xsg)):
        return ConstructType.TYPE4
    return ConstructType.TYPE3


def assemble_magnetic_space_subgroups(
    table: NDArrayInt,
    list_xsg: list[tuple[list[int], NDArrayFloat]],
    *,
    is_antiunitary_generator: Callable[[int, list[int]], bool],
    find_fsg_intertwiner: Callable[[list[int], list[int]], NDArrayFloat | None],
    classify: Callable[[list[int], list[int], list[int]], ConstructType],
) -> list[MagneticSpaceSubgroup]:
    """Assemble MSG candidates from XSG candidates ``(xsg, Q)``.

    Each XSG yields a Type-1 subgroup as is. Each candidate anti-unitary coset
    generator ``g`` (accepted by ``is_antiunitary_generator(g, xsg)`` and
    normalizing ``xsg``) extends it to ``fsg = xsg + g * xsg`` when
    ``find_fsg_intertwiner(fsg, conj_xsg)`` finds an orientation, classified
    as Type-3/4 by ``classify(xsg, fsg, conj_xsg)``.
    """
    order = len(table)

    list_msg: list[MagneticSpaceSubgroup] = []
    for xsg, Q in list_xsg:
        list_msg.append(
            MagneticSpaceSubgroup(
                xsg=xsg,
                fsg=xsg,
                msg_type=ConstructType.TYPE1,
                Q=Q,
            )
        )
    for xsg, _ in list_xsg:
        visited = [False for _ in range(order)]
        for g in range(order):
            if (g in xsg) or visited[g]:
                continue
            if is_antiunitary_generator(g, xsg) and (
                set(table[g, h] for h in xsg) == set(table[h, g] for h in xsg)
            ):
                conj_xsg = [int(table[g, h]) for h in xsg]
                fsg = sorted(list(xsg + conj_xsg))

                Q = find_fsg_intertwiner(fsg, conj_xsg)
                if Q is None:
                    continue

                list_msg.append(
                    MagneticSpaceSubgroup(
                        xsg=xsg,
                        fsg=fsg,
                        msg_type=classify(xsg, fsg, conj_xsg),
                        Q=Q,  # Q: spin_rotations -> prim_cart_rotations
                    )
                )
                for h in fsg:
                    visited[h] = True

    return list_msg


def _assemble_maximal_magnetic_space_subgroups(
    table: NDArrayInt,
    list_xsg: list[tuple[list[int], NDArrayFloat]],
    *,
    is_antiunitary_generator: Callable[[int, list[int]], bool],
    find_fsg_intertwiner: Callable[[list[int], list[int]], NDArrayFloat | None],
    classify: Callable[[list[int], list[int], list[int]], ConstructType],
) -> list[MagneticSpaceSubgroup]:
    """Assemble MSG candidates and retain only maximal candidates."""
    list_msg = assemble_magnetic_space_subgroups(
        table,
        list_xsg,
        is_antiunitary_generator=is_antiunitary_generator,
        find_fsg_intertwiner=find_fsg_intertwiner,
        classify=classify,
    )
    return filter_maximal_msg(list_msg)


def filter_maximal_msg(list_msg: list[MagneticSpaceSubgroup]) -> list[MagneticSpaceSubgroup]:
    list_maximal_msg: list[MagneticSpaceSubgroup] = []
    for i, msg in enumerate(list_msg):
        xsg = frozenset(msg.xsg)
        fsg = frozenset(msg.fsg)
        is_maximal = True
        for j, other in enumerate(list_msg):
            if i == j:
                continue
            if (msg.msg_type != ConstructType.TYPE1) and (other.msg_type == ConstructType.TYPE1):
                # Type-1 MSG cannot be superior to non-Type-1 MSG
                continue
            if (xsg <= frozenset(other.xsg)) and (fsg <= frozenset(other.fsg)):
                is_maximal = False
                break

        if is_maximal:
            list_maximal_msg.append(msg)

    return list_maximal_msg


def classify_magnetic_space_subgroup_conjugacy_classes(
    list_msg: list[MagneticSpaceSubgroup], table: NDArrayInt
) -> list[_EquivalenceClass[MagneticSpaceSubgroup, int]]:
    """Classify MSG candidates by spatial conjugacy inside the family group."""
    ginvs = all_inverses(table)

    def _key(msg: MagneticSpaceSubgroup) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(sorted(msg.xsg)), tuple(sorted(msg.fsg))

    def _find_conjugating_family_operation(
        msg: MagneticSpaceSubgroup,
        representative: MagneticSpaceSubgroup,
    ) -> int | None:
        representative_key = _key(representative)
        for g, ginv in enumerate(ginvs):
            conj_xsg = tuple(sorted(int(table[table[g, h], ginv]) for h in msg.xsg))
            conj_fsg = tuple(sorted(int(table[table[g, h], ginv]) for h in msg.fsg))
            if (conj_xsg, conj_fsg) == representative_key:
                return g
        return None

    def _find_stabilizer(representative: MagneticSpaceSubgroup) -> list[int]:
        return [
            g
            for g, ginv in enumerate(ginvs)
            if (
                tuple(sorted(int(table[table[g, h], ginv]) for h in representative.xsg)),
                tuple(sorted(int(table[table[g, h], ginv]) for h in representative.fsg)),
            )
            == _key(representative)
        ]

    return _classify_equivalence_classes(
        list_msg, _find_conjugating_family_operation, _find_stabilizer
    )


def filter_conjugacy_magnetic_space_subgroups(
    list_msg: list[MagneticSpaceSubgroup], table: NDArrayInt
) -> list[MagneticSpaceSubgroup]:
    """Return one representative per spatial conjugacy class."""
    return [
        equivalence_class.representative
        for equivalence_class in classify_magnetic_space_subgroup_conjugacy_classes(
            list_msg, table
        )
    ]
