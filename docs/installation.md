# Installation

SpinForge supports Python 3.11 and later.

!!! abstract "Page contract"

    **Starting point:** You have Python 3.11 or later and can run commands in a
    terminal. **Destination:** SpinForge imports successfully and you are ready
    to run an example. **Next:** Follow
    [Your first structure](quickstart.md). **Skip:** The source and documentation
    sections if you only need the released Python package.

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

Continue with [Your first structure](quickstart.md).

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
