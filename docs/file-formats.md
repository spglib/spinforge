# File formats

SpinForge can write symmetry-rich magnetic structures in spinCIF and MCIF.
The formats encode different symmetry descriptions, so choose the one that
matches the object you need to preserve.

| Format | Symmetry encoded | SpinForge interface |
|---|---|---|
| spinCIF (`.scif`) | Spin space group with independent spatial and spin operations | [`SpinCifWriter`][spinforge.scif.SpinCifWriter], [`SpinCifReader`][spinforge.scif.SpinCifReader] |
| MCIF (`.mcif`) | Magnetic space group in BNS/MAGNDATA convention | [`MCifWriter`][spinforge.mcif.MCifWriter] |

## Write an oriented structure as spinCIF

For a `(structure, magnetic_space_subgroup)` pair returned by
`generate_oriented()`, use the orientation-aware constructor:

```python
from spinforge.scif import SpinCifWriter

writer = SpinCifWriter.from_oriented(
    adapted,
    spin_space_group,
    magnetic_structure,
    magnetic_space_subgroup,
    spin_only_group=spin_only_group,
)
writer.write_file("candidate.scif")
```

The writer records the draft spinCIF dictionary revision in its output. The
upstream dictionary remains preliminary; see
[`SPINCIF_REVISION`][spinforge.scif.SPINCIF_REVISION] for the revision currently
targeted by SpinForge.

Read a file produced by SpinForge and expand its asymmetric unit:

```python
from spinforge.scif import SpinCifReader

reader = SpinCifReader.from_file("candidate.scif")
structure = reader.get_structure()
```

The reader intentionally supports the subset emitted by SpinForge plus the
FINDSPINGROUP reference files; it is not a general-purpose spinCIF parser.

## Write a magnetic structure as MCIF

`MCifWriter` identifies the magnetic space group of a pymatgen structure that
has a Cartesian `magmom` site property:

```python
from spinforge.mcif import MCifWriter

MCifWriter(magnetic_structure).write_file("candidate.mcif")
```

The resulting file stores the asymmetric unit and magnetic symmetry operations.
Read it with pymatgen:

```python
from pymatgen.core import Structure

round_tripped = Structure.from_file("candidate.mcif")
```

!!! warning "Mixed occupancies"

    MCIF writing supports partial occupancy of a single species at a site.
    Mixed solid-solution sites are not supported because they cannot be passed
    through the symmetry identification used by the writer.
