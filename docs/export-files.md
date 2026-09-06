# Export result files

!!! abstract "Page contract"

    **Starting point:** You have the five objects returned by enumeration and
    orientation: `spin_only_group`, `spin_space_group`, `adapted`,
    `magnetic_structure`, and `magnetic_space_subgroup`. **Destination:** You
    have a spinCIF, an MCIF, or both, plus a basic parse check. **Next:** Apply
    your downstream calculation's checks to the parsed structure. **Skip:** If
    the workflow remains entirely in Python. If the format is undecided, read
    [Choose an output format](file-formats.md) first.

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

Do not replace this with the plain `SpinCifWriter(...)` constructor for an
oriented result; that constructor expects magnetic moments in the original
enumeration frame.

## Write MCIF

The oriented `magnetic_structure` already contains Cartesian moments in its
`magmom` site property:

```python
from spinforge.mcif import MCifWriter

MCifWriter(magnetic_structure).write_file("candidate.mcif")
```

Pass `symprec` or `mag_symprec` to `MCifWriter` when the default symmetry-search
tolerances are unsuitable for the structure.

## Check the files

Parse the spinCIF with SpinForge and expand its asymmetric unit:

```python
from spinforge.scif import SpinCifReader

spin_structure = SpinCifReader.from_file("candidate.scif").get_structure()
```

Parse the MCIF with pymatgen:

```python
from pymatgen.core import Structure

magnetic_structure_from_file = Structure.from_file("candidate.mcif")
```

A successful parse catches syntax and expansion errors. Add application-level
checks for composition, lattice, positions, and moment vectors before using a
round-tripped structure in a calculation.
