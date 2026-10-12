from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Composition, DummySpecies, Element, Species
from pymatgen.core.periodic_table import get_el_sp
from spgrep.utils import NDArrayFloat

_ATOM_SITE_LOOP_HEADER = (
    "",
    "loop_",
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z",
    "_atom_site_occupancy",
)


@dataclass(frozen=True)
class _SiteConstituent:
    species: Element | Species
    occupancy: float
    magnetic: bool


def _normalize_site_species(
    site_compositions: Sequence[Composition],
    magnetic_species: Sequence[Sequence[str]] | None,
    magnetic_moments: NDArrayFloat,
) -> tuple[tuple[_SiteConstituent, ...], ...]:
    """Validate chemistry and shared occupied-atom moments for CIF writers."""
    num_sites = len(site_compositions)
    moments = np.asarray(magnetic_moments, dtype=float)
    if moments.shape != (num_sites, 3) or not np.all(np.isfinite(moments)):
        raise ValueError(f"magnetic_moments must be a finite ({num_sites}, 3) array")
    if magnetic_species is not None and len(magnetic_species) != num_sites:
        raise ValueError("magnetic_species must have one selection per site")

    normalized = []
    for index, composition in enumerate(site_compositions):
        items = sorted(composition.items(), key=lambda item: str(item[0]))
        if not items:
            raise ValueError(f"Site {index} must contain at least one species")
        for species, occupancy in items:
            if not isinstance(species, (Element, Species)) or isinstance(species, DummySpecies):
                raise TypeError(f"Unsupported species {species!r} at site {index}")
            if isinstance(species, Species) and (
                species.spin is not None
                or species.oxi_state is None
                or not np.isfinite(species.oxi_state)
            ):
                raise ValueError(f"Unsupported species {species!r} at site {index}")
            if not np.isfinite(occupancy) or occupancy <= 0:
                raise ValueError(f"Site {index} occupancies must be finite and positive")
        if sum(occupancy for _, occupancy in items) > 1 + 1e-8:
            raise ValueError(f"Site {index} total occupancy exceeds 1")

        identities = {str(species) for species, _ in items}
        if magnetic_species is None:
            selected = identities
        else:
            selection = magnetic_species[index]
            if isinstance(selection, str) or any(not isinstance(name, str) for name in selection):
                raise ValueError(f"Site {index} magnetic_species must be a sequence of species")
            selected = {str(get_el_sp(name)) for name in selection}
            if len(selected) != len(selection):
                raise ValueError(f"Site {index} magnetic_species contains duplicate species")
            if not selected <= identities:
                raise ValueError(f"Site {index} magnetic_species contains an absent species")
        if not selected and np.any(moments[index] != 0):
            raise ValueError(f"Site {index} has a nonzero moment but no selected magnetic species")
        normalized.append(
            tuple(
                _SiteConstituent(species, float(occupancy), str(species) in selected)
                for species, occupancy in items
            )
        )
    return tuple(normalized)


def _atom_type_lines(site_species: Sequence[Sequence[_SiteConstituent]]) -> list[str]:
    """Declare ionic atom types while retaining uncharged Element identities."""
    ionic_species = {
        str(row.species): row.species
        for site in site_species
        for row in site
        if isinstance(row.species, Species)
    }
    if not ionic_species:
        return []
    return [
        "",
        "loop_",
        "_atom_type_symbol",
        "_atom_type_oxidation_number",
        *[
            f"{symbol} {species.oxi_state:.12g}"
            for symbol, species in sorted(ionic_species.items())
        ],
    ]
