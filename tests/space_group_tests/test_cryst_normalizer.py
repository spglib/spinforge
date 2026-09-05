"""Optional cross-check of parent normalizer actions against GAP/Cryst.

Cryst is licensed under GPL-2.0-or-later, while SpinForge is BSD-licensed.
Keep the programs separate: these tests neither link to Cryst nor distribute
Cryst code or data. They invoke a user-installed ``gap`` executable as an
independent process and compare only computed group invariants.

The tests are deselected by default, so GAP/Cryst is not a build, test, or
distribution dependency of SpinForge. Run them explicitly with::

    pixi run test-cryst
"""

from __future__ import annotations

import subprocess
from fractions import Fraction

import numpy as np
import pytest
from _family_subgroup_helpers import (
    _rotated_translation_coset_subgroup,
    _translation_coset_subgroup,
)
from spgrep.utils import NDArrayFloat, NDArrayInt

from spinforge.space_group import FamilySpaceSubgroup


@pytest.mark.cryst
@pytest.mark.parametrize(
    "case",
    [_translation_coset_subgroup, _rotated_translation_coset_subgroup],
    ids=["translation-cosets", "rotated-translation-cosets"],
)
def test_parent_normalizer_action_against_cryst(case):
    enumerator, subgroup = case()
    program = _cryst_program(
        parent_rotations=enumerator.parent_rotations,
        parent_translations=enumerator.parent_translations,
        subgroup=subgroup,
    )

    result = subprocess.run(
        ["gap", "-q"],
        input=program,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"GAP/Cryst failed:\n{result.stderr}"
    output_lines = [line.lstrip("\r") for line in result.stdout.splitlines()]
    assert f"SPINFORGE_CRYST|ok|{len(subgroup.normalizer_action)}" in output_lines


def _cryst_program(
    *,
    parent_rotations: NDArrayInt,
    parent_translations: NDArrayFloat,
    subgroup: FamilySpaceSubgroup,
) -> str:
    parent_generators = _affine_matrices(parent_rotations, parent_translations)
    parent_generators.extend(_translation_matrices(np.eye(3, dtype=int)))

    subgroup_generators = _affine_matrices(
        subgroup.parent_rotations,
        subgroup.parent_translations,
    )
    subgroup_generators.extend(
        _translation_matrices(subgroup.translation_sublattice.transformation)
    )
    translation_generators = _translation_matrices(subgroup.translation_sublattice.transformation)
    normalizer_representatives = _affine_matrices(
        subgroup.normalizer_action.rotations,
        subgroup.normalizer_action.translations,
    )
    subgroup_representatives = _affine_matrices(
        subgroup.parent_rotations,
        subgroup.parent_translations,
    )
    permutations = [
        f"[{','.join(str(image + 1) for image in permutation)}]"
        for permutation in subgroup.normalizer_action.permutations
    ]

    return f"""
if LoadPackage(\"cryst\") = fail then
    PrintTo(\"*errout*\", \"the GAP Cryst package is unavailable\\n\");
    QUIT_GAP(1);
fi;

parent := AffineCrystGroupOnLeft({_gap_list(parent_generators)});
subgroup := AffineCrystGroupOnLeft({_gap_list(subgroup_generators)});
translations := AffineCrystGroupOnLeft({_gap_list(translation_generators)});
normalizer := Normalizer(parent, subgroup);
spinforgeReps := {_gap_list(normalizer_representatives)};
subgroupReps := {_gap_list(subgroup_representatives)};
spinforgePermutations := [{",".join(permutations)}];

if Index(normalizer, subgroup) <> Length(spinforgeReps) then
    PrintTo(\"*errout*\", \"normalizer quotient order differs\\n\");
    QUIT_GAP(1);
fi;

if not ForAll(spinforgeReps, representative -> representative in normalizer) then
    PrintTo(\"*errout*\", \"a SpinForge representative is outside the normalizer\\n\");
    QUIT_GAP(1);
fi;

for i in [1..Length(spinforgeReps)] do
    if i < Length(spinforgeReps) then
        for j in [i + 1..Length(spinforgeReps)] do
            if spinforgeReps[i] / spinforgeReps[j] in subgroup then
                PrintTo(\"*errout*\", \"SpinForge representatives repeat a normalizer coset\\n\");
                QUIT_GAP(1);
            fi;
        od;
    fi;
od;

for representativeIndex in [1..Length(spinforgeReps)] do
    representative := spinforgeReps[representativeIndex];
    permutation := [];
    for operation in subgroupReps do
        image := operation ^ representative;
        imageIndex := PositionProperty(
            subgroupReps,
            candidate -> image / candidate in translations
        );
        if imageIndex = fail then
            PrintTo(\"*errout*\", \"normalizer action escaped the subgroup quotient\\n\");
            QUIT_GAP(1);
        fi;
        Add(permutation, imageIndex);
    od;
    if permutation <> spinforgePermutations[representativeIndex] then
        PrintTo(\"*errout*\", \"induced permutation differs\\n\");
        QUIT_GAP(1);
    fi;
od;

Print(\"SPINFORGE_CRYST|ok|\", Index(normalizer, subgroup), \"\\n\");
QUIT_GAP(0);
"""


def _affine_matrices(
    rotations: NDArrayInt,
    translations: NDArrayFloat,
) -> list[list[list[str]]]:
    matrices = []
    for rotation, translation in zip(rotations, translations):
        matrix = [[*map(str, rotation[row]), _gap_rational(translation[row])] for row in range(3)]
        matrix.append(["0", "0", "0", "1"])
        matrices.append(matrix)
    return matrices


def _translation_matrices(transformation: NDArrayInt) -> list[list[list[str]]]:
    identity = np.eye(3, dtype=np.int64)
    return _affine_matrices(
        np.repeat(identity[np.newaxis, :, :], 3, axis=0),
        transformation.T.astype(float),
    )


def _gap_rational(value: float) -> str:
    fraction = Fraction(float(value)).limit_denominator(48)
    if not np.isclose(float(fraction), value, atol=1e-8):
        raise ValueError(f"Cannot represent translation {value} as a small GAP rational")
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _gap_list(value: list) -> str:
    if isinstance(value, list):
        return f"[{','.join(_gap_list(item) for item in value)}]"
    return str(value)
