# Enumeration and equivalence

## Equivalence criteria at a glance

The four enumeration stages classify different objects, so their equivalence controls are intentionally separate:

| Enumerator | Spatial equivalence | Spin-frame equivalence | Control |
|---|---|---|---|
| `FamilySpaceSubgroupEnumerator` | Conjugacy by the crystallographic parent $G$, or no conjugacy reduction | Not applicable | `up_to_parent_conjugacy=True` or `False` |
| `SpinSpaceGroupEnumerator` | Not applicable; $G'$ is fixed input | All orthogonal spin-frame transformations $O(3)$ | Fixed spin-frame equivalence $O(3)$; no equivalence option |
| `SpinSpaceSubgroupEnumerator` | $N_G(G')$ when enabled; no spatial conjugacy reduction otherwise | All orthogonal spin-frame transformations $O(3)$ | `up_to_parent_normalizer_conjugacy=True` or `False` |
| `OrientedSpinSpaceGroupEnumerator` | Operations of the fixed family space group $G'$ | Proper or all orthogonal spin-frame transformations | `preserve_spin_planochirality=True` or `False` |

The first row chooses family space subgroups $G'\leq G$. The second enumerates compatible spin-rotation parts over a fixed family. The third optionally reduces the resulting groups under the parent normalizer $N_G(G')$. The fourth controls oriented spin frames and spin-planochirality relative to the lattice.

## Mathematical convention for equivalence classes

Let a group $A$ act on a set of input objects $X$. SpinForge treats $x,y\in X$ as
equivalent when some transformation $a\in A$ maps one to the other. Each equivalence
class is the set of input objects related through this action. New classification code should
name these results equivalence classes; `orbit` remains reserved for established
crystallographic objects such as site, lattice, and reciprocal-space orbits.

When a later enumeration stage needs to replay the classification, SpinForge retains the
following data for each chosen representative $r$:

- the input objects $x$ equivalent to $r$;
- one transformation $a_x$ for each such object, chosen so that $a_x\cdot x=r$;
- the stabilizer of $r$, consisting of every retained transformation $a$ satisfying
  $a\cdot r=r$.

The transformation $a_x$ need not be unique. It records one explicit route from an input
object to the chosen representative; it is not an additional equivalence criterion.

For the parent-normalizer action that classifies normal space subgroups, the action is
inverse conjugation. Thus a stored normalizer transformation $h_x$ maps a subgroup $H_x$
to the representative $H_r$ when

$$h_x^{-1}H_xh_x=H_r.$$

The corresponding stabilizer consists of the normalizer transformations $h$ satisfying
$h^{-1}H_rh=H_r$. Code and API descriptions use the concrete phrases "equivalent objects,"
"transformations to the representative," and "stabilizer" for these three pieces of data.

## Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `multiplicity_preserving` | `True` | Only consider family space subgroups that preserve the Wyckoff-position multiplicity of magnetic sites. Set to `False` to include every enumerated family space subgroup. |

## `enumerate()` parameters

| Parameter | Default | Description |
|---|---|---|
| `k_index` | required without propagation vectors | Index of the final invariant translation lattice used to bound the search. Family translation indices must divide `k_index`; `with_propagation_vectors()` instead fixes one commensurate invariant lattice and supplies its index automatically. |
| `max_depth` | `1` | Maximum Hermann translationengleiche depth. `0` = the parent as the only Hermann group, `1` = parent + maximal proper translationengleiche subgroups, `None` = all Hermann groups. Bounded klassengleiche descendants of each retained Hermann group are still included. |
| `up_to_parent_conjugacy` | `True` | Classify family space subgroups up to conjugation by the crystallographic parent. Set to `False` to retain every enumerated t-group, family-lattice, and finite-quotient complement conjugate. This controls only whether $G$-conjugate family space subgroups are retained separately; it does not change SSG equivalence within each retained subgroup. |

With defaults (`multiplicity_preserving=True`, `max_depth=1`, `up_to_parent_conjugacy=True`), `enumerate()` returns SSA structures for one representative per conjugacy class of multiplicity-preserving family space subgroups, restricted to Hermann depth 1. For every retained family subgroup $G'\leq G$, spin space groups are classified under $N_G(G')\times O(3)$.

## How family space groups are enumerated

Let $G$ be the parent space group and $T$ its primitive translation subgroup. A family space group $G' \leq G$ has translation subgroup $T' = G' \cap T$. Translationengleiche (t) families have $T'=T$, klassengleiche (k) families have a proper $T'$ but the full parent point group, and general families reduce both the translation and point groups.

SpinForge follows the Hermann theorem. For every $G'\leq G$, there is a unique intermediate group

$$G'\leq M\leq G$$

such that $M$ is a t-subgroup of $G$ and $G'$ is a k-subgroup of $M$. Thus $M$ and $G'$ have the same point group, while $G$ and $M$ have the same translation group $T$. This separates the finite point-group choice from the bounded translation-lattice choice:

1. Enumerate the t-subgroups $M$ of $G$ from the parent point-subgroup lattice.
2. Choose an $M$-invariant family translation lattice $T'$. With `k_index=n`, its index $d=[T:T']$ must divide $n$. The remaining invariant-lattice index passed to the spin-space-group enumeration is $n/d$.
3. Call `moyopy.enumerate_klassengleiche_subgroups` for $M$ and $T'$. Moyopy returns each k-subgroup $G'\leq M\leq G$, its parent-operation mapping, and its conjugacy data.
4. Rewrite the lifted operations in the $T'$ basis and enumerate their spin-space groups. The separate t-family pass covers $T'=T$; the bounded pass starts at $[T:T']=2$.

`with_propagation_vectors()` fixes the final commensurate invariant lattice

$$L=\{t\in T\mid k_i\cdot t\in\mathbb{Z}\text{ for every supplied }k_i\}.$$

In that case, only family lattices satisfying $L\leq T'\leq T$ are considered, and the relative lattice $L\leq T'$ replaces the unrestricted index-$n/d$ search. An explicitly supplied `k_index` must equal $[T:L]$.

The optional filters act as follows:

- `multiplicity_preserving=True` tests the lifted affine group $G'$ and retains it only when every magnetic parent Wyckoff orbit splits into $G'$ orbits with the original multiplicity.
- `max_depth` uses the depth of the Hermann group M in the translationengleiche-subgroup Hasse diagram of the parent space group. Since each bounded descendant is a k-subgroup of M, it inherits the depth of its Hermann group. This is not an affine-maximality filter among the emitted family subgroups.
- `up_to_parent_conjugacy=True` chooses one $G$-conjugacy representative for each Hermann group $M$, reduces its family lattices under $N_G(M)$, then chooses one conjugacy representative of each complement in $M/T'$ for each retained $T'$. With a propagation-vector lattice $L$, only normalizer operations that preserve $L$ may identify family lattices. `False` emits the t-group, lattice, and finite-quotient conjugates explicitly.

The lower-level `FamilySpaceSubgroupEnumerator` controls this relation with `up_to_parent_conjugacy`:

| Value | Spatial equivalence criterion |
|---|---|
| `True` (default) | Classify family space subgroups up to conjugation by the parent space group $G$. In the Hermann construction, this selects $G$-conjugacy representatives of $M$, reduces family lattices by $N_G(M)$, and selects $M$-conjugacy representatives of the lifted complements. When a target lattice $L$ is fixed, only operations preserving $L$ participate. |
| `False` | Apply no spatial conjugacy reduction: emit every enumerated t-group conjugate, family-lattice image, and finite-quotient complement conjugate. |

`SSAGenerator` and `FamilySpaceSubgroupEnumerator` both default
`up_to_parent_conjugacy` to `True`. In both APIs, `True` applies the
family-subgroup conjugacy reduction described above, while `False` retains the
conjugate family subgroups separately. This relation is separate from
equivalence among spin space groups constructed over each retained family
group.

This construction follows the [Hermann theorem described by the Bilbao Crystallographic Server](https://cryst.ehu.es/cryst/help/index.html).

## Spin-space-group equivalence

Conjugacy-class reduction chooses family space subgroups $G'\leq G$, either one representative per $G$-conjugacy class or every conjugate. It is distinct from equivalence among spin space groups constructed over one retained family group.

The lower-level `SpinSpaceGroupEnumerator` receives one fixed family space group $G'$; it does not enumerate conjugate copies of $G'$ within a parent group. For this fixed spatial part, `enumerate_spin_space_groups()` enumerates compatible spin-rotation parts up to a global change of spin frame in $O(3)$ and groups them by invariant space subgroup. It does not apply a spatial-equivalence relation; parent-normalizer reduction is handled by `SpinSpaceSubgroupEnumerator`.

`SpinSpaceSubgroupEnumerator` instead consumes an enumerated `FamilySpaceSubgroup`. The family object records the inclusion $G'\leq G$ and the induced action of $N_G(G')/G'$. By default the class returns representatives under

$$N_G(G') \times O(3).$$

It first classifies invariant space subgroups $H'\trianglelefteq G'$ under $N_G(G')$. For each representative $H'$, it then classifies its spin-rotation assignments under the stabilizer of $H'$ in $N_G(G')$ together with the global $O(3)$ spin-frame action. Results are expressed in the translation-lattice basis of $G'$; `SSAGenerator` transforms them to the translation-lattice basis of the parent group $G$ for magnetic-structure generation.

Set `up_to_parent_normalizer_conjugacy=False` to skip both spatial reductions: every invariant subgroup and spin-rotation assignment returned by the fixed-family enumerator is retained. Global $O(3)$ spin-frame equivalence is still applied.

`SSAGenerator` uses the default parent-normalizer reduction for every retained $G'\leq G$. This is independent of `SSAGenerator.up_to_parent_conjugacy`: setting that argument to `False` retains every $G$-conjugate family space subgroup separately, while SSGs over each retained subgroup are still reduced under its own $N_G(G')\times O(3)$ action.

The criteria exposed by the enumeration APIs are therefore:

| API | Objects being classified | Spatial equivalence | Spin-frame equivalence |
|---|---|---|---|
| `FamilySpaceSubgroupEnumerator` | Family space subgroups $G'\leq G$ | Conjugacy by $G$ when `up_to_parent_conjugacy=True`; no conjugacy reduction otherwise | Not applicable |
| `SpinSpaceGroupEnumerator` | Spin-rotation parts over one fixed $G'$ | Not applicable; $G'$ is fixed input | $O(3)$ |
| `SpinSpaceSubgroupEnumerator` | SSGs over a family subgroup $G'\leq G$ | $N_G(G')$ when `up_to_parent_normalizer_conjugacy=True`; none otherwise | $O(3)$ |
| `SSAGenerator` | SSGs over a retained family subgroup $G'\leq G$ | The parent-space-group normalizer $N_G(G')$ | $O(3)$ |

There is no separate `up_to_family_space_group_conjugacy` option. For an invariant subgroup $H'\trianglelefteq G'$ and spin representation $U$, preconjugation by $g\in G'$ gives

$$U^g(x)=U(g^{-1}xg)=U(g)^{-1}U(x)U(g),$$

which is already identified by the global $O(3)$ spin-frame equivalence. Such a boolean would therefore not change the abstract SSG classes. `SSAGenerator.up_to_parent_conjugacy` remains independent and controls only whether $G$-conjugate family space subgroups are retained separately.

## Oriented spin-space-group equivalence

`OrientedSpinSpaceGroupEnumerator` fixes spatial equivalence to the family space group $G'$. It uses only rotation axes supplied by $G'$ and does not identify orientations through the crystallographic parent $G$ or its normalizer $N_G(G')$. `SSAGenerator.generate_oriented()` follows this family-group criterion.

The spin-side criterion is explicit:

| `preserve_spin_planochirality` | Spin-frame transformations | Spin-planochirality |
|---|---|---|
| `True` (default) | Proper rotations only | Retains distinct coplanar enantiomorphs |
| `False` | Proper and improper orthogonal transformations | Identifies coplanar enantiomorphs related by a spin reflection |

`SSAGenerator.generate_oriented()` and
`SpinSymmetryAdaptedStructure.generate_oriented()` use the same
`preserve_spin_planochirality` option and default.

`SpinSymmetryAdaptedStructure.generate_oriented()` also accepts
`parent_prim_rotations`. Supplying parent rotations there is an explicit opt-in
outside the canonical `OrientedSpinSpaceGroupEnumerator` relation.

## Bounded normal family listing

`FamilySpaceSubgroupEnumerator.enumerate_normal(k_index=..., max_depth=..., up_to_parent_conjugacy=...)` exposes the same bounded t, k, and general family subgroups used by spin-space-group enumeration, restricted to candidates normal in the full parent group $G$. Its `up_to_parent_conjugacy` argument has the same `True` default as `enumerate()`. A candidate $G'$ is retained when its stored parent-normalizer quotient exhausts the parent quotient: $|N_G(G')/G'|=[G:G']$.

`NormalSpaceSubgroupEnumerator`, which supplies normal subgroups to `SpinSpaceGroupEnumerator`, consumes this listing directly. It selects the requested exact translation index and point subgroup, then constructs the quotient table required for irrep enumeration.

The `k_index` argument has the same finite divisibility-bound semantics as `enumerate()`. When a target propagation lattice is supplied, only intermediate family lattices containing that target participate. This normality classification is distinct from the Hermann `max_depth` limit and from magnetic-site multiplicity filtering.
