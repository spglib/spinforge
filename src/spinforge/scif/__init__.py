"""Write spin-symmetry-adapted magnetic structures as spinCIF (.scif) files.

Follows the draft spinCIF dictionary (COMCIFS/spinCIF, ``core_spin.dic``).
A spinCIF file describes a magnetic structure under its spin space group
(SSG) instead of a magnetic space group: each symmetry operation carries an
independent spin-space operation (``_space_group_symop_spin_operation.uvw``)
next to its spatial part (``.xyzt``), and moments are given in an explicitly
defined spin basis (``_atom_site_spin_moment.axis_u/v/w``).

Conventions used by this writer:

- The spin basis is the orthonormal crystal-Cartesian frame x || a, z || c*
  (y completing a right-handed set), declared via
  ``_space_group_spin.transform_spinframe_P_abc`` (the unit vectors written
  as linear combinations of the lattice vectors; the tag every COMCIFS
  reference file uses, for interoperability with readers tuned to them).
- The nontrivial SSG operations are written in the *supercell* setting of the
  :class:`~spinforge.configuration.SpinSymmetryAdaptedStructure`; the spin
  translation coset goes into the optional
  ``_space_group_symop_spin_lattice`` loop.
- The trivial spin-only group is not listed as operations (per the spec);
  it enters through ``_space_group_spin.collinear_direction_xyz`` /
  ``_space_group_spin.coplanar_perp_uvw``, filled from the
  ``spinspg.spin.SpinOnlyGroup`` the structure was enumerated with.

The dictionary is explicitly preliminary upstream ("all dictionary aspects
and file formats are subject to change"): the targeted revision is pinned as
:data:`SPINCIF_REVISION` and stamped into every written file, so keep this
writer aligned with https://github.com/COMCIFS/spinCIF when it evolves.
"""

from ._reader import SpinCifReader
from ._writer import SPINCIF_REVISION, SpinCifWriter

__all__ = [
    "SPINCIF_REVISION",
    "SpinCifReader",
    "SpinCifWriter",
]
