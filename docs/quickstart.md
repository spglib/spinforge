# Your first structure

This walkthrough enumerates collinear spin-symmetry-adapted structures for
MnTe and then generates their oriented descendants.

## 1. Prepare a primitive cell

SpinForge accepts a
[`moyopy.Cell`](https://spglib.github.io/moyo/python/api/#moyopy.Cell). The direct constructor of
[`SSAGenerator`][spinforge.configuration.SSAGenerator] requires the cell to be
primitive.

```python
from moyopy import Cell
from pymatgen.core import Structure

structure = Structure.from_file("MnTe.cif")
primitive_cell = Cell(
    basis=structure.lattice.matrix.tolist(),
    positions=structure.frac_coords.tolist(),
    numbers=list(structure.atomic_numbers),
)
```

Select the magnetic sites by their indices in that cell. Here manganese has
atomic number 25:

```python
magnetic_site_indices = [
    index
    for index, atomic_number in enumerate(primitive_cell.numbers)
    if atomic_number == 25
]
```

!!! warning "Site indices follow the cell"

    If you transform, standardize, or reorder the cell, recompute the site
    indices. They refer to the exact `Cell` passed to the generator.

## 2. Enumerate SSA structures

```python
from spinforge.configuration import SSAGenerator
from spinspg.spin import SpinOnlyGroupType

generator = SSAGenerator(
    prim_cell=primitive_cell,
    magnetic_site_indices=magnetic_site_indices,
)

candidates = generator.enumerate(
    spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
    k_index=1,
    max_depth=0,
)
```

`k_index=1` limits the invariant translation lattice to the primitive lattice.
`max_depth=0` retains the parent as the only Hermann group while still allowing
the bounded descendants described in the
[enumeration guide](equivalence.md#enumerate-parameters).

Each result contains:

1. the trivial spin-only group;
2. the nontrivial spin space group;
3. a [`SpinSymmetryAdaptedStructure`][spinforge.configuration.SpinSymmetryAdaptedStructure]
   containing a supercell and a basis for allowed magnetic moments.

## 3. Generate oriented structures

```python
for spin_only_group, spin_space_group, adapted in candidates:
    descendants = generator.generate_oriented(
        adapted,
        spin_only_group=spin_only_group,
        nontrivial_spin_space_group=spin_space_group,
    )

    for magnetic_structure, magnetic_space_subgroup in descendants:
        print(
            magnetic_structure.formula,
            magnetic_space_subgroup.msg_type,
        )
```

`magnetic_structure` is a pymatgen `Structure` whose `magmom` site property
contains Cartesian moment vectors. By default, coplanar enantiomorphs are kept
distinct. Pass `preserve_spin_planochirality=False` to identify structures
related by improper spin-frame transformations.

## 4. Make sampling reproducible

When the adapted moment space has dimension greater than one, `generate()` and
`generate_oriented()` sample coefficients. Pass a NumPy generator when you
need repeatable output:

```python
import numpy as np

rng = np.random.default_rng(42)
structure = adapted.generate(rng=rng)
```

The symmetry-adapted subspace is deterministic; the sampled point within that
subspace is what the random generator controls.

## Next steps

- Use measured or calculated propagation vectors to constrain the translation
  lattice in the [propagation-vector guide](propagation-vectors.md).
- Read [enumeration and equivalence](equivalence.md) before expanding a search.
- Export a result using the [file-format guide](file-formats.md).
