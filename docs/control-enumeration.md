# Control an enumeration

!!! abstract "Page contract"

    - **Starting point:** You can construct an `SSAGenerator` and run the
      quickstart search.
    - **Destination:** You can choose the five search controls deliberately.
    - **Next:** Check the implemented relations in
      [Enumeration and equivalence](equivalence.md) when candidate counts need
      explanation.
    - **Skip:** The quickstart settings already cover your calculation.

## Start with the smallest relevant search

Select the moment geometry required by the problem, then keep the default
equivalence reductions:

```python
from spinspg.spin import SpinOnlyGroupType

candidates = generator.enumerate(
    spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
    k_index=1,
    max_depth=0,
    up_to_parent_conjugacy=True,
)
```

Use `COLLINEAR`, `COPLANAR`, or `NONCOPLANAR` according to the moment geometry
you intend to model. Moving to a less restrictive type changes the physical
search, not merely its cost.

## Set the translation bound

Without measured propagation vectors, `k_index` is required. It is the index
of the final invariant translation lattice that bounds enumeration:

- `k_index=1` restricts the final lattice to the primitive translation lattice.
- A larger value admits compatible family translation lattices whose indices
  divide that value.

If commensurate propagation vectors are known, construct the generator with
`SSAGenerator.with_propagation_vectors()` instead. SpinForge then derives the
lattice and supplies its index automatically; follow [Constrain an enumeration
with propagation vectors](propagation-vectors.md).

## Set the Hermann depth

`max_depth` controls which translationengleiche Hermann groups are searched.
Their bounded klassengleiche descendants remain included.

| Value | Hermann groups retained |
|---|---|
| `0` | Parent only |
| `1` | Parent and maximal proper subgroups (default) |
| `None` | All subgroup depths |

Begin at `0` when validating a workflow. Increase the depth only when the
target family symmetry requires it.

## Choose the spatial reduction

Keep `up_to_parent_conjugacy=True` to return one representative for family
space subgroups related by the crystallographic parent. Set it to `False` only
when you need every conjugate family subgroup explicitly:

```python
all_conjugates = generator.enumerate(
    spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
    k_index=1,
    max_depth=0,
    up_to_parent_conjugacy=False,
)
```

This option changes family-subgroup reduction. It does not disable the
spin-frame equivalence applied within each retained family subgroup.

## Choose the magnetic-site filter

By default, the generator omits family subgroups that change the multiplicity
of the supplied magnetic sites. Disable this constructor-time filter only when
those splittings are part of the intended search:

```python
generator = SSAGenerator(
    prim_cell=primitive_cell,
    magnetic_site_indices=magnetic_site_indices,
    multiplicity_preserving=False,
)
```

Change one control at a time and record the number of returned candidates. For
the exact equivalence relations and subgroup construction, see
[Enumeration and equivalence](equivalence.md). For the underlying scientific
derivation, see the [SpinForge article](https://doi.org/10.1103/8n3w-h2t1).
