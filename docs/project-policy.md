# Project scope, compatibility, and support

## Scope

SpinForge's public core covers:

- enumeration of family space subgroups and spin space groups;
- construction and sampling of spin-symmetry-adapted magnetic structures;
- orientation to maximal magnetic space subgroups; and
- serialization and parsing utilities for the supported spinCIF and MCIF
  workflows.

Magnetic structure determination or refinement from experiment, scattering
analysis, electronic-structure calculations, and high-throughput DFT workflows
are outside the project scope. The raw 283-material SDFT workflow, paper
figures, and MAGNDATA-derived research datasets are therefore not distributed
with SpinForge.

## Python versions

SpinForge supports Python 3.11 through 3.14, as declared in
`pyproject.toml`. Support for a Python version may be removed only in a
feature release and will be recorded in `CHANGELOG.md`.

## API stability

Beginning with version 1.0, documented interfaces re-exported by public
SpinForge modules are covered by the compatibility policy for the 1.x series.
Incompatible changes to these interfaces require a major release. When
practical, a deprecated public interface remains available for at least one
feature release before removal.

Names or modules with a leading underscore are private implementation details
and may change without deprecation. Example notebooks and test helpers
demonstrate workflows but are not themselves stable APIs.

Corrections required for mathematical validity, data integrity, or security
may require an exceptional incompatible change. Such changes will be explained
in the changelog and release notes.

The spinCIF dictionary is preliminary upstream. SpinForge records the supported
dictionary revision in `spinforge.scif.SPINCIF_REVISION`; spinCIF syntax may
evolve independently of the Python API compatibility policy.

## Support

Use [GitHub Issues](https://github.com/spglib/spinforge/issues) for
reproducible bugs, focused usage questions, and feature requests within the
scope above. Include a minimal input and the SpinForge and Python versions when
possible.

Support is provided on a best-effort basis without a guaranteed response time.
Security-sensitive reports must follow [`SECURITY.md`](../SECURITY.md).
