# Installation

SpinForge supports Python 3.11 and later.

## From PyPI

Create an isolated environment, then install the package:

=== "uv"

    ```shell
    uv venv
    uv pip install spinforge
    ```

=== "pip"

    ```shell
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install spinforge
    ```

Confirm that the installation is importable:

```shell
python -c "import spinforge; print('SpinForge is ready')"
```

## From a source checkout

Contributors need [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems/):

```shell
git clone https://github.com/spglib/spinforge.git
cd spinforge
uv sync --dev
uv run prek install
```

Run the tests and repository checks before opening a pull request:

```shell
just test
just prek
```

## Build the documentation locally

The development environment includes Zensical and mkdocstrings:

```shell
just docs
```

This starts a local preview server with live reload. Use `just docs-build` to
run the same strict production build used by GitHub Pages.
