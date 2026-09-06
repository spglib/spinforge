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

## Schematic overview and notebooks

The figures summarize the enumerated spin space groups and representative
oriented structures. Use the notebooks for the corresponding software
workflow, and the article for interpretation.

=== "Collinear MnTe"

    <object class="paper-figure" data="../assets/paper/fig_example_collinear.pdf" type="application/pdf">
      <p>PDF preview unavailable. <a href="assets/paper/fig_example_collinear.pdf">Open the collinear MnTe schematic</a>.</p>
    </object>

    [Open schematic PDF](assets/paper/fig_example_collinear.pdf){ .md-button }
    [Download notebook](paper/collinear_MnTe.ipynb){ .md-button }
    [Download input CIF](paper/MnTe.cif){ .md-button }

=== "Coplanar Mn₃Sn"

    <object class="paper-figure" data="../assets/paper/fig_example_coplanar.pdf" type="application/pdf">
      <p>PDF preview unavailable. <a href="assets/paper/fig_example_coplanar.pdf">Open the coplanar Mn₃Sn schematic</a>.</p>
    </object>

    [Open schematic PDF](assets/paper/fig_example_coplanar.pdf){ .md-button }
    [Download notebook](paper/coplanar_Mn3Sn.ipynb){ .md-button }

=== "Noncoplanar CoTa₃S₆"

    <object class="paper-figure" data="../assets/paper/fig_example_noncoplanar.pdf" type="application/pdf">
      <p>PDF preview unavailable. <a href="assets/paper/fig_example_noncoplanar.pdf">Open the noncoplanar CoTa₃S₆ schematic</a>.</p>
    </object>

    [Open schematic PDF](assets/paper/fig_example_noncoplanar.pdf){ .md-button }
    [Download notebook](paper/noncoplanar_CoTa3S6.ipynb){ .md-button }

!!! info "Figure permission and attribution"

    Figures by Takuya Nomoto, Kohei Shinohara, Hikaru Watanabe, and Ryotaro
    Arita for *Systematic Magnetic Structure Generation Based on Oriented Spin
    Space Groups: Formulation, Applications, and High-Throughput
    First-Principles Calculations* ([doi:10.1103/8n3w-h2t1](https://doi.org/10.1103/8n3w-h2t1)).
    The original PDFs are reproduced without modification, with permission
    from the owner of the
    [`spglib/spinforge-paper`](https://github.com/spglib/spinforge-paper)
    repository granted on September 6, 2026.

Consult the article for the complete discussion and results. MAGNDATA-derived
datasets and the raw 283-material SDFT workflow are outside this repository.
