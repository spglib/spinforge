<div align="center">

# <img src="./logo.svg" alt="spinforge" width=450>

</div>

Magnetic crystal structure generator on the basis of spin space groups

## Installation

```shell
uv sync --all-extras
source .venv/bin/activate
```

## Usage

### SSAGenerator

`SSAGenerator` is the main entry point for enumerating spin-symmetry-adapted (SSA) magnetic structures.

Its family-subgroup, spin-space-group, and oriented spin-frame equivalence
controls are intentionally separate. See the
[enumeration and equivalence guide](./docs/equivalence.md) for the API options
and the precise criteria used at each stage.

## Examples

See [examples](./examples).

## Attribution

### Test Data from Materials Project

Test fixture files in `src/spinforge/testing/assets/` with the `mp-` prefix are sourced from the [Materials Project](https://next-gen.materialsproject.org/) and are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

**Files from Materials Project:**
- `mp-1208409_Ta3CoS6.json`
- `mp-1455_MnS2.cif`
- `mp-1597_UO2.json`
- `mp-17554_LaMnO3.cif`
- `mp-20330_Mn3SiIr.json`
- `mp-2258_Cu3Au.json`
- `mp-2814_Yb2O3.json`
- `mp-31755_Ta2FeO6.json`
- `mp-35_Mn.json`
- `mp-5866_MnCuSb.json`
- `mp-5950_Cd2Os2O7.json`

These crystal structure data files are used exclusively for testing purposes. The Materials Project provides open access to computed materials properties under the CC-BY-4.0 license, which requires attribution.

**Citation:**
> A. Jain, S.P. Ong, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson, *The Materials Project: A materials genome approach to accelerating materials innovation*, APL Materials 1(1), 011002 (2013). [doi:10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

The remainder of the SpinForge codebase is licensed under the BSD 3-Clause License (see [LICENSE](./LICENSE)).
