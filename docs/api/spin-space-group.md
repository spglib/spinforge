# Spin space groups

Nontrivial spin space groups, symmetry operations, and subgroup enumeration.

## Subgroups of one fixed spin space group

Given an existing `NontrivialSpinSpaceGroup` and its `SpinOnlyGroup`, select
subgroups without generating new spin assignments:

```python
from spinforge.ssg import SpinSpaceSubgroupEnumerator

operations, table = nssg.get_full_operations_and_table()
enumerator = SpinSpaceSubgroupEnumerator.from_spin_space_group(nssg, spin_only_group)
subgroups = enumerator.enumerate_subgroups()

# Both bounds contain original operation indices and must be subgroups.
bounded = enumerator.enumerate_subgroups(containing=lower, contained_in=upper)
```

Each result is a `frozenset` of indices into `operations`, with the assigned spin
cosets unchanged. The finite quotient is taken modulo the invariant translation
lattice of `nssg`. Its translation cosets remain distinct even when their spatial
rotation matrices agree. Every selection retains that invariant lattice and the
entire spin-only group; this API does not enumerate subgroups that reduce either.

Bounds are inclusive and need not be normal. Omitted bounds mean the identity
subgroup and the full quotient. The search returns every embedded subgroup,
including conjugate copies, in order of size and then sorted indices. Construction
checks that spin lifts normalize the supplied spin-only group and close modulo
it, including nonhomomorphic choices of lifts for collinear and coplanar groups.

The ordinary `SpinSpaceSubgroupEnumerator(family_space_subgroup)` constructor
continues to generate spin assignments over a family subgroup. Its properties
and enumeration methods are separate from the fixed-SSG mode; using a method
from the other mode raises `ValueError`.

## API reference

::: spinforge.ssg
