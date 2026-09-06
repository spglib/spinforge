# Third-party data and source notices

This notice covers data files distributed in the SpinForge repository. The BSD
3-Clause License applies to SpinForge source code; third-party data retain the
terms stated below.

## Materials Project fixtures

The following test fixtures under `src/spinforge/testing/assets/` were
obtained from the [Materials Project](https://materialsproject.org/) and are
distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/):

| File | Materials Project record |
| --- | --- |
| `mp-1208409_Ta3CoS6.json` | [mp-1208409](https://materialsproject.org/materials/mp-1208409) |
| `mp-1455_MnS2.cif` | [mp-1455](https://materialsproject.org/materials/mp-1455) |
| `mp-1597_UO2.json` | [mp-1597](https://materialsproject.org/materials/mp-1597) |
| `mp-17554_LaMnO3.cif` | [mp-17554](https://materialsproject.org/materials/mp-17554) |
| `mp-20330_Mn3SiIr.json` | [mp-20330](https://materialsproject.org/materials/mp-20330) |
| `mp-2258_Cu3Au.json` | [mp-2258](https://materialsproject.org/materials/mp-2258) |
| `mp-2814_Yb2O3.json` | [mp-2814](https://materialsproject.org/materials/mp-2814) |
| `mp-31755_Ta2FeO6.json` | [mp-31755](https://materialsproject.org/materials/mp-31755) |
| `mp-35_Mn.json` | [mp-35](https://materialsproject.org/materials/mp-35) |
| `mp-5866_MnCuSb.json` | [mp-5866](https://materialsproject.org/materials/mp-5866) |
| `mp-5950_Cd2Os2O7.json` | [mp-5950](https://materialsproject.org/materials/mp-5950) |

Please cite:

> A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek,
> S. Cholia, D. Gunter, D. Skinner, G. Ceder, and K. A. Persson,
> “The Materials Project: A materials genome approach to accelerating
> materials innovation,” *APL Materials* **1**, 011002 (2013).
> [doi:10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

## MnTe structural model

`src/spinforge/testing/assets/MnTe.cif` and
`examples/paper/collinear_MnTe/MnTe.cif` are identical to the documentation
copy at `docs/paper/MnTe.cif`. They are locally serialized pymatgen CIFs for
the NiAs-type MnTe structure. The lattice constants
`a = b = 4.17349018 Å` and `c = 6.75345133 Å` follow the supplemental
material to:

> N. Heinsdorf, “Altermagnetic Instabilities from Quantum Geometry,”
> *Physical Review B* **111**, 174407 (2025).
> [doi:10.1103/PhysRevB.111.174407](https://doi.org/10.1103/PhysRevB.111.174407)

The files were generated within the SpinForge project; they are not copied
MAGNDATA records.

## Mn3Sn structural model

`src/spinforge/testing/assets/Mn3Sn.cif` is a locally serialized pymatgen
P1 expansion of the hexagonal D0₁₉ Mn3Sn model. Its structural parameters
(`a = b = 5.677 Å`, `c = 4.534 Å`, and Mn coordinate `x ≈ 0.833`) follow
the crystallographic compilation:

> P. Villars and L. D. Calvert (eds.), *Pearson's Handbook of
> Crystallographic Data for Intermetallic Phases*, Vol. 4, ASM International
> (1991).

For an open record of the closely corresponding structure and its primary
literature source, see
[Crystallography Open Database entry 1522909](https://www.crystallography.net/cod/1522909.html),
which cites U. P. Singh, A. K. Pal, L. Chandrasekaran, and K. P. Gupta,
“Study of the manganese-rich end of the Mn-Sn system,” *Transactions of the
Metallurgical Society of AIME* **242**, 1661–1663 (1968).

The SpinForge fixture is not a byte-for-byte COD or MAGNDATA record.

## Other locally generated fixture

`src/spinforge/testing/assets/NiS2.cif` is a pymatgen serialization created
within the SpinForge project. Repository history records no external database
file as its source, so no third-party dataset license is asserted for this
file.

## Article example figures

The following PNG renderings under `docs/assets/paper/` were converted from
source PDFs in the
[`spglib/spinforge-paper`](https://github.com/spglib/spinforge-paper)
repository:

- `fig_example_collinear.png`
- `fig_example_coplanar.png`
- `fig_example_noncoplanar.png`

The figures were created for:

> T. Nomoto, K. Shinohara, H. Watanabe, and R. Arita,
> “Systematic magnetic structure generation based on oriented spin space
> groups: Formulation, applications, and high-throughput first-principles
> calculations,” *Physical Review X* (accepted 2026).
> [doi:10.1103/8n3w-h2t1](https://doi.org/10.1103/8n3w-h2t1)

They are included with permission from the owner of the
`spglib/spinforge-paper` repository, granted on September 6, 2026. This notice
records that permission without asserting a Creative Commons license before
the article's Version of Record is published.
