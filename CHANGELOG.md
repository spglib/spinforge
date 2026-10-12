# Changelog

All notable changes to SpinForge will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The initial public release includes:

- Spin-space-group enumeration and symmetry-adapted magnetic structure
  generation.
- Oriented magnetic structure generation under maximal magnetic space
  subgroups.
- spinCIF and MCIF utilities.
- Paper examples for collinear MnTe, coplanar Mn3Sn, and noncoplanar CoTa3S6.
- Public project, contribution, security, citation, and data-attribution
  documentation.

## Unreleased

- Preserve mixed-site compositions, partial occupancies, oxidation states, and
  selected magnetic constituents in MCIF and spinCIF exports.
- Add `SpinCifWriter.from_structure` for complete constituent chemistry and
  explicit orthogonal spin frames; preserve chemistry in `from_oriented`.
- Preserve separate spinCIF constituent rows with occupancies, individual
  moments, and explicit magnetic declarations.
- Preserve fractional oxidation identities and weak moments without display
  rounding. Reject loss during pymatgen conversion in the spinCIF reader and
  Structure-based exports.
- Use caller tolerances for position, moment, and frame checks and MCIF
  translation snapping.

## v1.0.0 - 2026/09/06

- Initial public release
