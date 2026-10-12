# Choose an output format

!!! abstract "Page contract"

    - **Starting point:** You have an oriented magnetic structure but have not
      chosen a file representation.
    - **Destination:** You know which format preserves the object needed by the
      next tool.
    - **Next:** Write and check it with
      [Export result files](export-files.md).
    - **Skip:** If the workflow remains entirely in Python.

| Choose | When you need to preserve | SpinForge interface |
|---|---|---|
| spinCIF (`.scif`) | The spin space group, including independent spatial and spin operations | [`SpinCifWriter`][spinforge.scif.SpinCifWriter] and [`SpinCifReader`][spinforge.scif.SpinCifReader] |
| MCIF (`.mcif`) | The magnetic space group of one oriented magnetic structure in BNS/MAGNDATA convention | [`MCifWriter`][spinforge.mcif.MCifWriter] |

The distinction is about symmetry, not only filename syntax. A spinCIF records
the spin-space description carried by a SpinForge result. An MCIF
records the magnetic space group that `MCifWriter` identifies from the
oriented structure and its Cartesian `magmom` site property.

Choose spinCIF when another calculation or archive needs the spin-space-group
description. Choose MCIF when the consumer understands conventional magnetic
crystallographic symmetry but not spin-space symmetry. You can write both when
you need both representations.

## Format limits

- The spinCIF dictionary is preliminary. SpinForge stamps the supported draft
  revision, [`SPINCIF_REVISION`][spinforge.scif.SPINCIF_REVISION], into each
  file. The reader supports files written by SpinForge and the FINDSPINGROUP
  reference files; it is not a general-purpose spinCIF parser.
- Both writers preserve mixed species, oxidation states, and partial occupancy.
  Their `Structure` input carries Cartesian occupied-atom `magmom` values and
  an optional boolean `magnetic` property, which defaults to true. False omits a
  magnetic declaration; true preserves it even at zero moment. To assign a
  moment to only part of a mixed site, supply separate constituent rows at the
  same position. Occupancy is written separately from the moment.
- Use `SpinCifWriter.from_structure` for complete chemistry and an explicit
  `spin_rotation`. The matrix maps intrinsic spins to Cartesian spins, including
  improper frames. Supply moments already rotated by that matrix.
- The writers group coincident rows for their symmetry backends. Magnetic
  constituents at one physical site must share a vector. The spinCIF reader
  preserves separate constituent rows and permits unequal coincident vectors.
  Its output therefore needs application-specific checks before use by routines
  that require one site per position.
- Occupancy must survive pymatgen's representation. The reader reports an error
  if conversion would discard a constituent; it does not normalize occupancies.
- Pymatgen's structure parser does not support disordered magnetic CIFs.
  Mixed MCIF output therefore requires a consumer that preserves constituent
  rows and their individual occupancies and moments.

## Numerical tolerances

`MCifWriter` and `SpinCifWriter.from_structure` use `symprec` in Cartesian length
units. MCIF also uses it to bound displacement from rational translation snapping.
An explicit `mag_symprec` controls moment comparisons in moment units, while
spinCIF's `atol` controls dimensionless frame and coefficient checks.

The reader's `symprec` retains its fractional-position units. Its optional
`mag_symprec` and `atol` override moment and normalized-frame tolerances;
omitting them retains the existing `symprec` fallback. Absolute comparisons use
`rtol=0`. For the reader and Structure-based exports, chemical identities and
occupancies are independent of these geometric and magnetic tolerances. The
legacy `site_occupancies` argument retains its `atol`-based occupancy bounds and
orbit comparisons; it writes the representative occupancy for each orbit.

To write and check either format, continue to [Export result files](export-files.md).
For the scientific relation between spin and magnetic symmetry, see the
[SpinForge article](https://doi.org/10.1103/8n3w-h2t1).
