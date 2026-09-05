"""Private representation of equivalence classes and their transformations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")
_A = TypeVar("_A")


@dataclass(frozen=True, eq=False)
class _EquivalenceClass(Generic[_T, _A]):
    """Equivalent input objects with transformations to their representative."""

    representative: _T
    equivalent_objects: tuple[_T, ...]
    transformations_to_representative: tuple[_A, ...]
    stabilizer: tuple[_A, ...]

    def __post_init__(self) -> None:
        if len(self.equivalent_objects) != len(self.transformations_to_representative):
            raise ValueError(
                "Equivalent objects and transformations to the representative must be aligned."
            )


def _classify_equivalence_classes(
    candidates: Sequence[_T],
    find_transformation_to_representative: Callable[[_T, _T], _A | None],
    find_stabilizer: Callable[[_T], Iterable[_A]],
) -> list[_EquivalenceClass[_T, _A]]:
    """Partition candidates and retain transformations to each representative."""
    consumed = [False] * len(candidates)
    equivalence_classes: list[_EquivalenceClass[_T, _A]] = []

    for representative_index, representative in enumerate(candidates):
        if consumed[representative_index]:
            continue

        representative_transformation = find_transformation_to_representative(
            representative, representative
        )
        if representative_transformation is None:
            raise ValueError("Equivalence action does not relate a representative to itself.")

        consumed[representative_index] = True
        equivalent_objects = [representative]
        transformations_to_representative = [representative_transformation]
        for candidate_index in range(representative_index + 1, len(candidates)):
            if consumed[candidate_index]:
                continue
            candidate = candidates[candidate_index]
            transformation = find_transformation_to_representative(candidate, representative)
            if transformation is None:
                continue
            consumed[candidate_index] = True
            equivalent_objects.append(candidate)
            transformations_to_representative.append(transformation)

        equivalence_classes.append(
            _EquivalenceClass(
                representative=representative,
                equivalent_objects=tuple(equivalent_objects),
                transformations_to_representative=tuple(transformations_to_representative),
                stabilizer=tuple(find_stabilizer(representative)),
            )
        )

    return equivalence_classes
