# Classification model

!!! abstract "Page contract"

    - **Starting point:** You understand the objects in the
      [domain model](domain-model.md) but not which distinctions SpinForge
      preserves.
    - **Destination:** You can identify which setting changes the search space,
      an equivalence relation, or only the orientation of a result.
    - **Next:** Apply those distinctions in
      [Control an enumeration](control-enumeration.md).
    - **Skip:** Use the [exact reference](equivalence.md) directly when the
      conceptual distinctions are already familiar.

The [SpinForge article](https://doi.org/10.1103/8n3w-h2t1) is the authoritative
source for the underlying theory and derivations. This page only maps those
concepts to package decisions.

## The decisions are independent

| Question | SpinForge decision | What changes |
|---|---|---|
| What kind of spin configuration is sought? | `SpinOnlyGroupType.COLLINEAR`, `.COPLANAR`, or `.NONCOPLANAR` | The compatible spin-only groups and spin-space-group assignments |
| Which translation lattices are admissible? | An index bound with `k_index`, or a lattice fixed by `with_propagation_vectors()` | The spatial families and supercells searched |
| How far below the parent spatial symmetry should the search go? | `max_depth` and `multiplicity_preserving` | The family space subgroups retained |
| Should parent-conjugate spatial families be listed separately? | `up_to_parent_conjugacy` | The number of family-subgroup representatives |
| Should a coplanar mirror partner remain distinct after orientation? | `preserve_spin_planochirality` | The oriented descendants returned |

Changing one row does not silently change the others. In particular, retaining
more family-space-group conjugates is distinct from changing spin-frame
equivalence, and orientation is a later classification step rather than a new
family-subgroup search.

## Spin class

The requested spin-only-group type classifies the allowed geometry of the
candidate moments:

- `COLLINEAR`: all moments lie on one spin axis;
- `COPLANAR`: moments lie in one spin plane but need not share an axis;
- `NONCOPLANAR`: moments are not restricted to one plane.

This is an input classification, not a label inferred from experimental data.
If a propagation-vector constraint is incompatible with the selected class,
an empty enumeration can be the correct result.

## Translation constraint

`k_index` gives a finite index bound when the translation lattice is unknown.
`SSAGenerator.with_propagation_vectors()` instead fixes the commensurate
translation lattice from measured or calculated propagation vectors.

These are alternative ways to supply the translation constraint. When
propagation vectors are present, their lattice supplies the index, and an
explicit `k_index` must equal that computed index. See
[use propagation vectors](propagation-vectors.md) for coordinate-frame handling
and [control enumeration](control-enumeration.md) for practical search choices.

## Spatial-family breadth

`max_depth` bounds the translationengleiche Hermann groups visited beneath the
parent. It is not a generic "subgroup depth" for every emitted family.
`multiplicity_preserving=True` additionally removes family subgroups that split
the selected magnetic-site orbits incompatibly with the package's multiplicity
criterion.

By default, `up_to_parent_conjugacy=True` retains one representative of each
family-subgroup class under the crystallographic parent $G$. The `False`
setting retains the separate parent-conjugate embeddings.

## Keep spin-frame and spatial equivalence separate

For a fixed family group $G'$, compatible spin-space groups are identified up
to a global orthogonal change of spin frame. The high-level generator also
reduces them by the applicable parent normalizer $N_G(G')$. These relations do
not become weaker when parent-conjugate family groups are retained separately.

Orientation then uses the axes of the fixed family subgroup. For coplanar results,
`preserve_spin_planochirality=True` keeps enantiomorphs related by an improper
spin transformation distinct; `False` identifies them.

For the formal actions, defaults, and lower-level API behavior, consult the
[enumeration and equivalence reference](equivalence.md).
