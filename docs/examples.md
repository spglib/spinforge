# Examples

!!! abstract "Page contract"

    **Starting point:** You have completed [Your first structure](quickstart.md)
    and want to map that workflow to a physical example. **Destination:** You
    can choose the notebook whose spin-only-group class and translation bound
    match your task. **Next:** Open that notebook and use the article for its
    scientific interpretation. **Skip:** This page if you only need an exact
    API signature.

The repository contains the notebooks used for three representative examples
in the [SpinForge article](https://doi.org/10.1103/8n3w-h2t1). Each follows the
same software workflow: choose the magnetic sites, select a spin-only-group
class and `k_index`, enumerate SSA structures, generate oriented descendants,
and write spinCIF output. The article is the source for the formalism,
derivations, and physical interpretation.

<div class="grid cards" markdown>

-   **Collinear MnTe**

    ---

    Uses `SpinOnlyGroupType.COLLINEAR` with `k_index=1`. Start here for the
    closest continuation of the quickstart.

    [:octicons-arrow-right-24: Open notebook](https://github.com/spglib/spinforge/tree/main/examples/paper/collinear_MnTe)

-   **Coplanar Mn₃Sn**

    ---

    Uses `SpinOnlyGroupType.COPLANAR` with `k_index=1` and compares oriented
    enumeration with spin planochirality kept distinct or treated as
    equivalent.

    [:octicons-arrow-right-24: Open notebook](https://github.com/spglib/spinforge/tree/main/examples/paper/coplanar_Mn3Sn)

-   **Noncoplanar CoTa₃S₆**

    ---

    Uses `SpinOnlyGroupType.NONCOPLANAR` with `k_index=4`, demonstrating the
    same workflow with a larger invariant translation cell.

    [:octicons-arrow-right-24: Open notebook](https://github.com/spglib/spinforge/tree/main/examples/paper/noncoplanar_CoTa3S6)

</div>

The notebooks are published as source examples. Consult the article for the
complete discussion and results; MAGNDATA-derived datasets and the raw
283-material SDFT workflow are outside this repository.
