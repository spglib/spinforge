# Constrain an enumeration with propagation vectors

!!! abstract "Page contract"

    **Starting point:** One or more commensurate propagation vectors and the
    magnetic sites are known in the input structure's setting. **Destination:**
    Your `SSAGenerator` uses their common translation lattice as its bound.
    **Next:** Enumerate and orient candidates as in
    [Your first structure](quickstart.md). **Skip:** When no propagation vector
    is known, choose `k_index` in
    [Control an enumeration](control-enumeration.md) instead.

## Construct the generator in the input frame

For vectors reported with an experimental CIF, keep the structure and vectors
in that CIF's setting and use `k_frame="input"`:

```python
import numpy as np
from moyopy import Cell
from pymatgen.core import Structure
from spinforge.configuration import SSAGenerator
from spinspg.spin import SpinOnlyGroupType

structure = Structure.from_file("input.cif")
cell = Cell(
    basis=structure.lattice.matrix.tolist(),
    positions=structure.frac_coords.tolist(),
    numbers=list(structure.atomic_numbers),
)
magnetic_site_indices = [
    index
    for index, atomic_number in enumerate(cell.numbers)
    if atomic_number == 25
]

generator = SSAGenerator.with_propagation_vectors(
    cell=cell,
    propagation_vectors=[np.array([0.0, 0.0, 0.5])],
    magnetic_site_indices=magnetic_site_indices,
    k_frame="input",
)

candidates = generator.enumerate(
    spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
)
```

Replace the atomic-number selection with the magnetic species or sites in your
system. Site indices must refer to the exact `cell` passed to
`with_propagation_vectors()`.

SpinForge standardizes the cell, transforms the vectors to its primitive
standardized setting, and logs each vector in both frames. It derives the
translation index automatically. Do not pass a different `k_index` to
`enumerate()`; an explicit value must equal the derived index.

## Use the standardized frame only when it is already known

Set `k_frame="prim_std"` only when the supplied vectors are already expressed
in moyopy's primitive standardized setting. They are then used as-is.

## Diagnose an empty result

If enumeration returns no candidates:

1. Compare the input and standardized vectors in the construction log. A
   surprising permutation or value usually indicates that the vector and cell
   came from different settings.
2. Confirm that the chosen spin-only-group type can realize the translation
   quotient. For example, a collinear spin-only group can realize only the
   identity and spin-flip quotient.
3. Try `COPLANAR` or `NONCOPLANAR` only if that moment geometry is physically
   part of the intended search.

The translation lattice is the common commensurate lattice of the supplied
vectors. For its derivation and the surrounding classification, see the
[SpinForge article](https://doi.org/10.1103/8n3w-h2t1); the operational API
details are in [`SSAGenerator.with_propagation_vectors()`][spinforge.configuration.SSAGenerator.with_propagation_vectors].
