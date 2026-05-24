# Contributing

The Silmaril Firewall Python SDK is public and source-available for Silmaril
customers and integrators. It is not permissive open source. Review
[LICENSE](LICENSE) before copying, redistributing, or modifying the SDK outside
of an integration with Silmaril services.

## Development

Use Python 3.10 or later.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,langchain]"
```

Run the local release checks before opening a PR:

```sh
python -m pytest -q -m "not integration"
python -m ruff check src tests
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
```

Integration tests under `tests/integ` call a deployed Silmaril Firewall
endpoint and require tenant credentials. Keep those tests marked with
`@pytest.mark.integration` so default CI and local release checks do not call
live infrastructure.

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
