# Propagation vectors

Use [`SSAGenerator.with_propagation_vectors()`][spinforge.configuration.SSAGenerator.with_propagation_vectors]
when one or more commensurate propagation vectors are known. SpinForge derives
their common invariant translation lattice and uses it as the enumeration
bound.

```python
import numpy as np
from pymatgen.core import Structure
from moyopy import Cell
from spinforge.configuration import SSAGenerator
from spinspg.spin import SpinOnlyGroupType

structure = Structure.from_file("input.cif")
cell = Cell(
    basis=structure.lattice.matrix.tolist(),
    positions=structure.frac_coords.tolist(),
    numbers=list(structure.atomic_numbers),
)

generator = SSAGenerator.with_propagation_vectors(
    cell=cell,
    propagation_vectors=[np.array([0.0, 0.0, 0.5])],
    magnetic_site_indices=[0],
    k_frame="input",
)

candidates = generator.enumerate(
    spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
)
```

## Choose the coordinate frame explicitly

`k_frame="input"` means the fractional reciprocal coordinates use the setting
of the supplied cell. This is the safest choice for vectors reported alongside
an experimental CIF. SpinForge transforms both the cell and the vectors into
moyopy's primitive standardized setting.

Use `k_frame="prim_std"` only when the vectors already use that standardized
setting. The constructor logs the accepted vectors in both frames so that axis
permutations and setting mismatches are visible.

## Translation index

The propagation vectors determine a commensurate lattice

$$
L = \{t \in T \mid k_i \cdot t \in \mathbb{Z}\text{ for every }k_i\}.
$$

Its index is supplied automatically during enumeration. If you also pass
`k_index`, it must equal that derived index.

!!! tip "When enumeration returns no candidates"

    First check the transformed vectors in the log. If the frames agree, the
    empty result may be physically meaningful: the translation quotient may
    not be realizable by the selected spin-only-group type. Try a compatible
    coplanar or noncoplanar type only when it matches the problem you intend to
    model.
