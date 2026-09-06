set shell := ["zsh", "-uc"]
set positional-arguments

default:
    @just --list

prek:
    uv run prek run --all-files

install:
    uv sync --dev
    uv run prek install

docs:
    uv run zensical serve

docs-build:
    uv run zensical build --clean --strict

test *args:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -v -n 4 --dist load --maxschedchunk=1 "$@"

test-serial *args:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -v "$@"
