# Contributing

The Silmaril Firewall Python SDK is public and source-available for Silmaril
customers and integrators. It is not permissive open source. Review
[LICENSE](LICENSE) before copying, redistributing, or modifying the SDK outside
of an integration with Silmaril services.

## Development

Use Python 3.10 or later. Pull-request CI in `.github/workflows/ci.yml` tests
3.10, 3.11, 3.12, and 3.13.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

Install the extras that CI installs for your interpreter. The `dev` extra
provides `build`, `pytest`, `pytest-asyncio`, `ruff`, and `twine`.

```sh
# Python 3.10
python -m pip install -e ".[dev,langchain]"

# Python 3.11 and later, including the release workflow's Python 3.12 job
python -m pip install -e ".[dev,langchain,deepagents]"
```

Run the pull-request checks:

```sh
ruff check src tests
pytest -q
python -m build
python -m twine check dist/*
```

Remove a previous `dist/`, `build/`, or `src/*.egg-info` before `python -m build`
when the checkout is being reused. CI starts from a clean runner.

`.github/workflows/release.yml` uses Python 3.12 and runs `ruff check src tests`,
then `python -m pytest -q -m "not integration"`, then `python -m build` and
`python -m twine check dist/*`. This repository has no
`tests/integ` tree, `pyproject.toml` does not register an `integration`
marker, and no current test uses that mark. Pull-request CI runs `pytest -q`
with no marker filter. A test that calls a deployed Firewall must skip unless
the caller opts in, and must be marked `@pytest.mark.integration` so the
release job excludes it.

## Pull Requests

- Keep changes focused on one behavior, release, or documentation concern.
- Update `README.md` and `CHANGELOG.md` for public behavior or packaging
  changes.
- Keep `pyproject.toml` and `src/silmaril_security/sdk/_version.py` aligned.
- Do not commit generated distributions, virtual environments, caches, or local
  `.env` files.

## Release Process

Releases are published from `main` by `.github/workflows/release.yml` when
`pyproject.toml` contains a version that is not present on PyPI and does not
already have a Git tag.

Before merging a release PR, maintainers must confirm PyPI trusted publishing
is configured for:

- PyPI project: `silmaril-security-sdk`
- Owner: `Silmaril-Security`
- Repository: `sdk-python`
- Workflow: `.github/workflows/release.yml`
- Environment: `pypi`

The GitHub workflow needs `id-token: write` for the publish job and a GitHub
environment named `pypi`. The PyPI project must have a matching trusted
publisher entry. If PyPI returns an `invalid-publisher` error, fix the PyPI
project or organization trusted-publisher configuration before retrying.

Do not move, delete, or reuse release tags. If a tag exists but the matching
PyPI package was never published, recover by bumping to the next patch version
and documenting the skipped package version in `CHANGELOG.md`.
