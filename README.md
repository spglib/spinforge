<div align="center">

# <img src="./docs/assets/logo.svg" alt="SpinForge" width="450">

</div>

SpinForge is a group-theoretic generator for spin-symmetry-adapted (SSA) and
oriented magnetic crystal structures based on spin space groups.

[Documentation](https://spglib.github.io/spinforge/) ·
[PyPI](https://pypi.org/project/spinforge/) ·
[Issue tracker](https://github.com/spglib/spinforge/issues)

Installation, tutorials, examples, citation guidance, and API references are
maintained in the [documentation](https://spglib.github.io/spinforge/). For a
source checkout and contributor setup, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Release flow

Releases are prepared by [tagpr](https://github.com/Songmu/tagpr) and published
by the `release` GitHub Actions workflow.

1. Merge all changes for the release into `main`. Tagpr creates or updates the
   `Release for vX.Y.Z` pull request from its `tagpr-from-v...` branch.
2. Check out that pull request with `gh pr checkout <number>`, update its branch
   directly with the final release changes (including `CHANGELOG.md`), and push
   the commits to the same branch. Do not open another pull request for these
   changes.
3. If the proposed version needs to change, add the `tagpr:minor` or
   `tagpr:major` label to the release pull request; without either label, tagpr
   increments the patch version.
4. Review the generated release notes and wait for the release pull request's
   checks to pass, then merge it.
5. The resulting push to `main` makes tagpr create the version tag. The release
   workflow builds and verifies the wheel and source distribution, publishes
   them to PyPI, and creates the GitHub Release with both artifacts attached.

To rebuild an existing tag, run the `release` workflow manually with that tag.
Set `publish` to false when the artifacts and GitHub Release should be rebuilt
without publishing to PyPI.

## Project scope, compatibility, and support

SpinForge provides symmetry enumeration, SSA and oriented-SSA structure
generation, and spinCIF/MCIF-related utilities. It does not determine or refine
magnetic structures from experimental or first-principles data, and it does
not include scattering, electronic-structure, or high-throughput DFT
workflows.

### Compatibility

SpinForge supports Python 3.11 through 3.14. Beginning with version 1.0,
documented interfaces re-exported by public SpinForge modules are kept
compatible within the 1.x series. Names or modules with a leading underscore
are private and may change without deprecation. Mathematical-validity,
data-integrity, or security corrections are documented in the changelog when
they require an exceptional incompatible change.

The spinCIF dictionary is preliminary upstream. SpinForge records the supported
revision in `spinforge.scif.SPINCIF_REVISION`, and spinCIF syntax may evolve
independently of the Python API policy.

### Support

Please use
[GitHub Issues](https://github.com/spglib/spinforge/issues) for reproducible
bugs and in-scope feature requests. Security reports follow
[SECURITY.md](./SECURITY.md).

## Data attribution and license

Third-party and literature-derived fixture notices are collected in
[`docs/THIRD_PARTY_DATA.md`](./docs/THIRD_PARTY_DATA.md), including all Materials Project
fixtures and the MnTe and Mn3Sn source notices.

SpinForge source code is distributed under the
[BSD 3-Clause License](./LICENSE). Third-party data retain the terms identified
in their notices.
