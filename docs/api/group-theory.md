# Finite groups

Enumerate subgroups of a finite group from its Cayley table. Each table entry
`table[i, j]` gives the index of the product of elements `i` and `j`.

```python
import numpy as np

from spinforge.utils.group_theory import enumerate_subgroups

# C4, with multiplication written as addition modulo four.
indices = np.arange(4)
table = (indices[:, None] + indices[None, :]) % 4
subgroups = enumerate_subgroups(table)
# [frozenset({0}), frozenset({0, 2}), frozenset({0, 1, 2, 3})]

bounded = enumerate_subgroups(table, containing={0, 2})
# [frozenset({0, 2}), frozenset({0, 1, 2, 3})]
```

`containing` and `contained_in` are inclusive subgroup bounds. They need not
be normal. Omitted bounds mean the identity subgroup and the full group.
The function retains every embedded subgroup, including conjugate copies,
and preserves the input table's indices. The identity can have any index.
A valid finite-group table is required; the function validates the bounds,
not the group axioms.

For a `NontrivialSpinSpaceGroup`, pass the table from
`nssg.get_full_operations_and_table()`. The returned indices select operations
from that same quotient, retaining its invariant translation lattice. The
caller retains the original spin-only group when interpreting these selections
as spin space subgroups.

## API reference

::: spinforge.utils.group_theory.enumerate_subgroups
