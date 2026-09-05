"""Regression tests for independent family and parent-normalizer SSG equivalence.

Test case: MnS2 (Pa-3, No. 205) with k_index=2. Walking the subgroup chain
yields several family space subgroups ``G' <= G``. Family conjugacy is controlled
independently from the per-family ``N_G(G') x O(3)`` SSG equivalence used for
magnetic-structure generation.
"""

from __future__ import annotations

import pytest
from spinspg.spin import SpinOnlyGroupType

from spinforge.configuration._configuration import SSAGenerator
from spinforge.testing import load_prim_MnS2, site_indices


@pytest.fixture
def MnS2_magnetic_site_indices() -> list[int]:
    return site_indices(load_prim_MnS2(), 25)  # Mn sites


class TestEquivalenceMnS2:
    """MnS2 (Pa-3, No. 205) equivalence regression tests at k_index=2."""

    @pytest.fixture(autouse=True)
    def _setup(self, MnS2_magnetic_site_indices: list[int]):
        prim_cell = load_prim_MnS2()
        self.gen = SSAGenerator(prim_cell, MnS2_magnetic_site_indices)

    def test_collinear_k2(self):
        results = self.gen.enumerate(
            SpinOnlyGroupType.COLLINEAR,
            k_index=2,
            max_depth=None,
        )
        # The Pca2_1 family is reduced from 4 to 2 under N_G(G') x O(3).
        t_family_results = [result for result in results if result[1].family_k_index == 1]
        assert len(t_family_results) == 2
        assert len(results) > len(t_family_results)

    def test_collinear_k2_without_parent_conjugacy(self):
        """Without parent conjugacy, all conjugate family subgroups are retained."""
        results = self.gen.enumerate(
            SpinOnlyGroupType.COLLINEAR,
            k_index=2,
            max_depth=None,
            up_to_parent_conjugacy=False,
        )
        # All t-family conjugate copies are emitted, while each copy is still
        # reduced under its own N_G(G') x O(3) action.
        t_family_results = [result for result in results if result[1].family_k_index == 1]
        assert len(t_family_results) == 6
        assert len(results) > len(t_family_results)

    def test_collinear_k2_without_parent_conjugacy_is_candidate_order_invariant(self, monkeypatch):
        original_family_subgroups = self.gen._family_subgroups
        expected = self.gen.enumerate(
            SpinOnlyGroupType.COLLINEAR,
            k_index=2,
            max_depth=None,
            up_to_parent_conjugacy=False,
        )
        monkeypatch.setattr(
            self.gen,
            "_family_subgroups",
            lambda **kwargs: list(reversed(original_family_subgroups(**kwargs))),
        )

        actual = self.gen.enumerate(
            SpinOnlyGroupType.COLLINEAR,
            k_index=2,
            max_depth=None,
            up_to_parent_conjugacy=False,
        )

        assert len(actual) == len(expected)

    def test_coplanar_k2(self):
        results = self.gen.enumerate(
            SpinOnlyGroupType.COPLANAR,
            k_index=2,
            max_depth=None,
        )
        assert len(results) > 0

    def test_parent_conjugacy_does_not_change_parent_normalizer_equivalence(self, monkeypatch):
        family = next(
            candidate
            for candidate in self.gen._family_subgroups(
                k_index=2,
                up_to_parent_conjugacy=True,
                max_depth=None,
            )
            if candidate.is_translationengleiche
        )
        monkeypatch.setattr(self.gen, "_family_subgroups", lambda **kwargs: [family])

        reduced_families = self.gen.enumerate(
            SpinOnlyGroupType.NONCOPLANAR,
            k_index=2,
            max_depth=None,
        )
        retained_families = self.gen.enumerate(
            SpinOnlyGroupType.NONCOPLANAR,
            k_index=2,
            max_depth=None,
            up_to_parent_conjugacy=False,
        )

        assert len(reduced_families) == len(retained_families)

    def test_noncoplanar_k2(self):
        results = self.gen.enumerate(
            SpinOnlyGroupType.NONCOPLANAR,
            k_index=2,
            max_depth=None,
        )
        assert len(results) > 0
