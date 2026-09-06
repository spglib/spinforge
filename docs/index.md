---
hide:
  - navigation
  - toc
---

# Forge magnetic structures from symmetry

SpinForge generates spin-symmetry-adapted and oriented magnetic crystal
structures from crystallographic symmetry. It gives you a group-theoretic
route from a primitive nonmagnetic structure to candidate magnetic structures
that can be inspected, exported, or used in downstream calculations.

[Get started](installation.md){ .md-button .md-button--primary }
[Explore the API](api/configuration.md){ .md-button }

<div class="grid cards" markdown>

-   :material-axis-arrow:{ .lg .middle } **Symmetry first**

    ---

    Enumerate collinear, coplanar, and noncoplanar structures from spin space
    groups instead of guessing moment patterns.

-   :material-tune-variant:{ .lg .middle } **Controlled searches**

    ---

    Bound supercells by translation index or propagation vectors and choose
    exactly which spatial equivalences to retain.

-   :material-rotate-3d-variant:{ .lg .middle } **Oriented descendants**

    ---

    Turn an SSA structure into maximal magnetic-space-subgroup descendants
    with spin directions oriented relative to the crystal lattice.

-   :material-file-export-outline:{ .lg .middle } **Interoperable output**

    ---

    Work with pymatgen structures and export symmetrized spinCIF or MCIF files.

</div>

## The workflow

``` mermaid
flowchart LR
    A[Primitive crystal] --> B[Family space subgroups]
    B --> C[Spin space groups]
    C --> D[SSA moment basis]
    D --> E[Magnetic structure]
    D --> F[Oriented descendants]
    E --> G[spinCIF / MCIF]
    F --> G
```

SpinForge keeps the stages explicit. This matters because family-subgroup
conjugacy, spin-frame equivalence, and the orientation of spin relative to the
lattice are different classification problems. The
[enumeration and equivalence guide](equivalence.md) defines each relation.

## At a glance

```python
from spinforge.configuration import SSAGenerator
from spinspg.spin import SpinOnlyGroupType

generator = SSAGenerator(
    prim_cell=primitive_cell,
    magnetic_site_indices=[0, 1],
)

candidates = generator.enumerate(
    spin_only_group_type=SpinOnlyGroupType.COLLINEAR,
    k_index=1,
    max_depth=0,
)
```

The result is a list of `(spin_only_group, spin_space_group, adapted_structure)`
tuples. Continue with [your first structure](quickstart.md) for a complete
example and the assumptions behind it.

!!! note "Project scope"

    SpinForge generates symmetry-compatible candidates. It does not determine
    or refine a magnetic structure from experimental or first-principles data.
