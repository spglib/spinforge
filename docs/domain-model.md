# Domain model

!!! abstract "Page contract"

    - **Starting point:** You can recognize a magnetic structure but are new to
      SpinForge's vocabulary.
    - **Destination:** You can follow an enumeration result from its
      crystallographic input to an oriented magnetic structure.
    - **Next:** Learn which distinctions affect a candidate set in the
      [classification model](classification-model.md).
    - **Skip:** This page when SpinForge's objects and workflow stages are
      already familiar.

For derivations and the full scientific treatment, see the
[SpinForge article](https://doi.org/10.1103/8n3w-h2t1).

## One workflow, several distinct objects

``` mermaid
flowchart TB
    A[Primitive cell<br/>and magnetic sites]
    B[Family space subgroup]
    C[Spin space group<br/>and spin-only group]
    D[SSA structure<br/>supercell + moment basis]
    E[Oriented magnetic structure<br/>and magnetic space subgroup]

    A -->|enumerate spatial families| B
    B -->|assign spin rotations| C
    C -->|construct allowed moments| D
    D -->|orient relative to lattice| E
```

SpinForge keeps these stages separate because they answer different
questions. A family space subgroup (shortened below to *family subgroup*) is a
spatial symmetry choice relative to the crystallographic parent. A spin space
group adds how spatial operations act on spins. A spin-symmetry-adapted (SSA)
structure represents the magnetic-moment subspace allowed by that symmetry.
An oriented result finally relates the spin frame to the crystal lattice.

## Input state

[`SSAGenerator`][spinforge.configuration.SSAGenerator] starts from:

- a primitive [`moyopy.Cell`](https://spglib.github.io/moyo/python/api/#moyopy.Cell);
- indices identifying the magnetic sites in that exact cell;
- optionally, propagation vectors that fix a commensurate translation lattice.

The cell defines the crystallographic parent space group $G$. The magnetic
site indices determine which parent-site orbits must support moments. Search
bounds and propagation vectors limit the translation lattices considered; they
do not become additional atoms or moments in the input object.

## Enumeration state

Calling `SSAGenerator.enumerate()` returns tuples of:

1. a `SpinOnlyGroup`, which records the requested collinear, coplanar, or
   noncoplanar spin-only symmetry;
2. a [`NontrivialSpinSpaceGroup`][spinforge.ssg.NontrivialSpinSpaceGroup],
   which records the nontrivial coupling between spatial and spin operations;
3. a
   [`SpinSymmetryAdaptedStructure`][spinforge.configuration.SpinSymmetryAdaptedStructure],
   which contains a supercell and a basis for symmetry-allowed magnetic
   moments.

These are candidate symmetry classes, not predicted ground states. SpinForge
does not rank them by energy or fit them to measurements.

## Generated and oriented state

`SpinSymmetryAdaptedStructure.generate()` selects a moment configuration from
the adapted basis. If the basis has more than one dimension, its coefficients
are sampled; pass a NumPy random generator when the selected point must be
reproducible.

`SSAGenerator.generate_oriented()` performs a different step: it enumerates
the allowed orientations of the spin frame relative to the family subgroup's
crystallographic axes. Each result pairs a pymatgen `Structure`, with Cartesian
moments in its `magmom` site property, with a
[`MagneticSpaceSubgroup`][spinforge.msg.MagneticSpaceSubgroup].

!!! tip "What to read next"

    - Read the [classification model](classification-model.md) to choose the
      boundary and equivalence controls that change a candidate set.
    - Follow [your first structure](quickstart.md) to run this workflow.
    - Use the [configuration API](api/configuration.md) when you need exact
      signatures and return types.
