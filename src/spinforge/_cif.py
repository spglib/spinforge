from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum

import numpy as np
from pymatgen.core import Composition, DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom

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


@dataclass(frozen=True)
class _PhysicalSites:
    structure: Structure
    constituents: tuple[tuple[_SiteConstituent, ...], ...]


def _number(value: float) -> str:
    """A positional decimal that round-trips to the same float."""
    return np.format_float_positional(float(value), unique=True, trim="-")


def _xyzt_string(rotation: np.ndarray, translation: np.ndarray, time_reversal: int) -> str:
    """Serialize integral rotations and float translations without rational approximation."""
    if not np.array_equal(rotation, np.rint(rotation)):
        raise ValueError("Spatial rotation must have integer coefficients")
    expressions = []
    for row, offset in zip(rotation, translation, strict=True):
        expression = ""
        for coefficient, variable in zip(row, "xyz", strict=True):
            if coefficient == 0:
                continue
            sign = "-" if coefficient < 0 else "+" if expression else ""
            magnitude = abs(int(coefficient))
            expression += sign + (str(magnitude) if magnitude != 1 else "") + variable
        if offset != 0:
            expression += ("+" if expression and offset > 0 else "") + _number(offset)
        expressions.append(expression or "0")
    return ",".join(expressions) + f",{time_reversal:+d}"


def _species_token(species: Element | Species) -> str:
    if isinstance(species, Species):
        charge = _oxidation_state(species)
        return f"{species.symbol}{_number(abs(charge))}{'+' if charge >= 0 else '-'}"
    return species.symbol


def _species_key(species: Element | Species) -> tuple[str, bool, float]:
    return (
        species.symbol,
        isinstance(species, Species),
        _oxidation_state(species) if isinstance(species, Species) else 0.0,
    )


def _oxidation_state(species: Species) -> float:
    charge = species.oxi_state
    if charge is None or not np.isfinite(charge):
        raise ValueError(f"Unsupported species {species!r}")
    return float(charge)


def _cartesian_moment(moment) -> np.ndarray:
    """Convert Magmom's spin axis without its small-component zero flushing."""
    if not isinstance(moment, Magmom):
        value = np.asarray(moment, dtype=float)
        return np.array([0.0, 0.0, float(value)]) if value.ndim == 0 else value
    x, y, z = moment.saxis
    alpha = np.arctan2(y, x)
    beta = np.arctan2(np.hypot(x, y), z)
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    return np.asarray(moment.moment) @ np.array(
        [[cb * ca, cb * sa, -sb], [-sa, ca, 0.0], [sb * ca, sb * sa, cb]]
    )


def _physical_sites(structure: Structure, *, symprec: float, mag_symprec: float) -> _PhysicalSites:
    """Adapt constituent rows to the one-vector-per-site writer backends."""
    if "magmom" not in structure.site_properties:
        raise ValueError("structure must have a 'magmom' site property")
    moments = np.asarray(
        [_cartesian_moment(moment) for moment in structure.site_properties["magmom"]]
    )
    if moments.shape != (len(structure), 3) or not np.all(np.isfinite(moments)):
        raise ValueError("magmom must contain finite Cartesian 3-vectors")
    magnetic = structure.site_properties.get("magnetic", [True] * len(structure))
    if any(not isinstance(flag, (bool, np.bool_)) for flag in magnetic):
        raise ValueError("magnetic must contain boolean flags")
    if any(not flag and np.any(moment != 0) for flag, moment in zip(magnetic, moments)):
        raise ValueError("A nonmagnetic row must have a zero moment")

    distances = structure.lattice.get_all_distances(structure.frac_coords, structure.frac_coords)
    groups: list[list[int]] = []
    for index in range(len(structure)):
        matches = [group for group in groups if np.any(distances[index, group] <= symprec)]
        if len(matches) > 1 or (matches and np.any(distances[index, matches[0]] > symprec)):
            raise ValueError("Ambiguous coincident-site grouping within symprec")
        if matches:
            matches[0].append(index)
        else:
            groups.append([index])

    compositions = []
    positions = []
    shared_moments = []
    constituents = []
    for group in groups:
        selected = [moments[index] for index in group if magnetic[index]]
        moment = selected[0] if selected else np.zeros(3)
        if any(
            np.linalg.norm(first - second) > mag_symprec
            for first in selected
            for second in selected
        ):
            raise ValueError("Unequal magnetic moments at one physical site are unsupported")
        row_amounts: dict[tuple[Element | Species, bool], list[float]] = {}
        amounts: dict[Element | Species, list[float]] = {}
        for index in group:
            composition = structure[index].species
            if not composition:
                raise ValueError("Every row must contain at least one species")
            for species, occupancy in composition.items():
                if not isinstance(species, (Element, Species)) or isinstance(
                    species, DummySpecies
                ):
                    raise TypeError(f"Unsupported species {species!r}")
                if isinstance(species, Species):
                    if species.spin is not None:
                        raise ValueError(f"Unsupported species {species!r}")
                    _oxidation_state(species)
                if not np.isfinite(occupancy) or occupancy <= 0:
                    raise ValueError("Occupancies must be finite and positive")
                row_amounts.setdefault((species, bool(magnetic[index])), []).append(occupancy)
                amounts.setdefault(species, []).append(occupancy)
        expected = {species: fsum(values) for species, values in amounts.items()}
        composition = Composition(expected)
        if dict(composition.items()) != expected:
            raise ValueError("The site composition is not representable by pymatgen")
        compositions.append(composition)
        positions.append(structure[group[0]].frac_coords)
        shared_moments.append(moment)
        rows = [
            _SiteConstituent(species, fsum(values), flag)
            for (species, flag), values in row_amounts.items()
        ]
        constituents.append(
            tuple(
                sorted(
                    rows, key=lambda row: (_species_key(row.species), row.occupancy, row.magnetic)
                )
            )
        )
    physical = Structure(
        structure.lattice,
        compositions,
        positions,
        site_properties={"magmom": shared_moments},
    )
    if any(
        dict(site.species.items()) != dict(composition.items())
        for site, composition in zip(physical, compositions, strict=True)
    ):
        raise ValueError("The site composition is not representable by pymatgen")
    return _PhysicalSites(physical, tuple(constituents))


def _atom_type_lines(site_species: Sequence[Sequence[_SiteConstituent]]) -> list[str]:
    """Declare ionic atom types while retaining uncharged Element identities."""
    ionic_species = {
        row.species for site in site_species for row in site if isinstance(row.species, Species)
    }
    if not ionic_species:
        return []
    return [
        "",
        "loop_",
        "_atom_type_symbol",
        "_atom_type_oxidation_number",
        *[
            f"{_species_token(species)} {_number(_oxidation_state(species))}"
            for species in sorted(ionic_species, key=_species_key)
        ],
    ]
