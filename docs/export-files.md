# Export result files

!!! abstract "Page contract"

    - **Starting point:** You have the five objects returned by enumeration and
      orientation: `spin_only_group`, `spin_space_group`, `adapted`,
      `magnetic_structure`, and `magnetic_space_subgroup`.
    - **Destination:** You have a spinCIF, an MCIF, or both, plus a basic parse
      check.
    - **Next:** Apply your downstream calculation's checks to the parsed
      structure.
    - **Skip:** If the workflow remains entirely in Python. If the format is
      undecided, read [Choose an output format](file-formats.md) first.

## Write spinCIF

Use the orientation-aware constructor for a structure returned by
`generate_oriented()`. It applies the same orientation to the stored spin
operations:

```python
from spinforge.scif import SpinCifWriter

spin_cif = SpinCifWriter.from_oriented(
    adapted,
    spin_space_group,
    magnetic_structure,
    magnetic_space_subgroup,
    spin_only_group=spin_only_group,
)
spin_cif.write_file("candidate.scif")
```

For an orientation obtained without an MSG object, use
`SpinCifWriter.from_structure(..., spin_rotation=rotation, symprec=symprec)`.
The structure must already contain the rotated moments. The plain array
constructor assumes the original enumeration frame unless `spin_rotation` is
supplied.

## Write MCIF

The oriented `magnetic_structure` already contains Cartesian moments in its
`magmom` site property:

```python
from spinforge.mcif import MCifWriter

MCifWriter(magnetic_structure).write_file("candidate.mcif")
```

Pass `symprec` or `mag_symprec` to `MCifWriter` when the default symmetry-search
tolerances are unsuitable for the structure.

## Represent mixed magnetic sites

Use separate constituent rows when only some species at a position carry a
magnetic declaration:

```python
from pymatgen.core import Lattice, Species, Structure

mixed_structure = Structure(
    Lattice.cubic(4),
    [{Species("Fe", 2): 0.6}, {Species("Zn", 2): 0.2}],
    [[0, 0, 0], [0, 0, 0]],
    site_properties={
        "magmom": [[0, 0, 2], [0, 0, 0]],
        "magnetic": [True, False],
    },
)
MCifWriter(mixed_structure).write_file("mixed.mcif")
```

The vacancy fraction remains 0.2. Fe's written moment is 2, and its contribution
to the mean-site moment is 1.2. Set `magnetic=True` with a zero vector to preserve
an explicitly magnetic zero. For spinCIF, pass the same structure to
`SpinCifWriter.from_structure` together with its SSA model and orientation.

## Check the files

Parse the spinCIF with SpinForge and expand its asymmetric unit:

```python
from spinforge.scif import SpinCifReader

spin_structure = SpinCifReader.from_file("candidate.scif").get_structure()
```

Each returned row retains its constituent occupancy, Cartesian `magmom`, and
`magnetic` declaration. Distinct constituents can occupy the same position and
have different moments. The current writers require a shared magnetic vector
when grouping such rows for their symmetry backends.

Parse single-species-site MCIFs with pymatgen:

```python
from pymatgen.core import Structure

magnetic_structure_from_file = Structure.from_file("candidate.mcif")
```

Pymatgen's magnetic structure parser does not support mixed-site MCIFs. Those
files need a consumer that preserves their constituent rows and occupancies.

A successful parse catches syntax and expansion errors. Add application-level
checks for composition, lattice, positions, and moment vectors before using a
round-tripped structure in a calculation.
