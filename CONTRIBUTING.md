# Contributing to SpinForge

Contributions that improve SpinForge's public core are welcome. Before starting
a substantial change, open an issue so its scope and scientific validation can
be agreed on.

## Development setup

SpinForge requires Python 3.11 or later, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/).

```shell
git clone https://github.com/spglib/spinforge.git
cd spinforge
uv sync --dev
uv run pre-commit install
```

## Checks

Run the test suite and repository hooks before submitting a pull request:

```shell
just test
just prek
```

Tests marked `slow` are excluded from the default suite. Run them explicitly
when the change affects their scientific contract:

```shell
just test -m slow
```

Add focused tests for behavior changes. Scientific transformations should be
checked against an independent invariant or oracle where practical.

## Pull requests

- Keep each pull request focused on one reviewable change.
- Explain the scientific or user-facing behavior and the validation performed.
- Update documentation and `CHANGELOG.md` when public behavior changes.
- Do not commit generated environments, credentials, private datasets, or
  research workflows outside the public project scope.

By submitting a contribution, you agree that it may be distributed under the
project's [BSD 3-Clause License](./LICENSE).

The public scope and compatibility policy are defined in
[`docs/project-policy.md`](./docs/project-policy.md).
