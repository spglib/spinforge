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
- MCIF output supports partial occupancy of a single species at a site. It does
  not support mixed solid-solution sites because those sites cannot pass
  through the symmetry identification used by the writer.

To write and check either format, continue to [Export result files](export-files.md).
For the scientific relation between spin and magnetic symmetry, see the
[SpinForge article](https://doi.org/10.1103/8n3w-h2t1).
