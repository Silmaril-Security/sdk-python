"""Require a release version increase for changes to package inputs."""

import subprocess
import sys
import tomllib

from packaging.version import Version


MANIFEST = "pyproject.toml"
PACKAGE_INPUTS = (
    "src",
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "hatch_build.py",
)


def version_at(revision):
    manifest = subprocess.check_output(
        ["git", "show", f"{revision}:{MANIFEST}"], text=True
    )
    value = tomllib.loads(manifest)["project"]["version"]
    return Version(value)


def main(base, candidate):
    diff = subprocess.run(
        ["git", "diff", "--quiet", "--no-renames", base, candidate, "--", *PACKAGE_INPUTS]
    )
    if diff.returncode == 0:
        print("Package inputs are unchanged.")
        return
    if diff.returncode != 1:
        diff.check_returncode()
    if version_at(candidate) <= version_at(base):
        raise SystemExit(f"Package inputs changed. Increase the version in {MANIFEST} in this PR.")
    print("Package version increased.")


if __name__ == "__main__":
    main(*sys.argv[1:])
