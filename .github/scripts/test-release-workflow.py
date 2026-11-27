#!/usr/bin/env python3
"""Exercise release decisions with local stubs for Git and the registries."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]

STUBS = """
git() {
  [[ "$*" == 'ls-remote --exit-code --tags origin refs/tags/v1.2.3' ]] || return 99
  return "$TAG_STATUS"
}
node() {
  echo 1.2.3
}
python() {
  cat >/dev/null
  if [[ "$#" == 1 ]]; then
    echo 1.2.3
  else
    echo "$REGISTRY_STATUS"
  fi
}
npm() {
  [[ "$*" == 'view @silmaril-security/sdk@1.2.3 version --json' ]] || return 99
  case "$REGISTRY_STATUS" in
    200) echo '"1.2.3"' ;;
    404) echo E404 >&2; return 1 ;;
    *) echo E503 >&2; return 1 ;;
  esac
}
gh() {
  if [[ "$*" == "api --include repos/$GITHUB_REPOSITORY/releases/tags/$TAG" ]]; then
    echo "HTTP/2.0 $RELEASE_STATUS stub"
    [[ "$RELEASE_STATUS" == 200 ]]
  elif [[ "$*" == "release create $TAG --verify-tag --title $TAG --notes Release $TAG" ]]; then
    touch "$RUNNER_TEMP/create-attempted"
    [[ "$TAG_STATUS" == 0 ]] || return 1
    touch "$RUNNER_TEMP/release-created"
  else
    return 99
  fi
}
"""


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
        self.plan = next(step["run"] for step in jobs["plan"]["steps"] if step.get("id") == "plan")
        self.release = next(step["run"] for step in jobs["github_release"]["steps"] if "run" in step)

    def run_step(self, step, workdir, *, tag, registry=200, release=200):
        env = {
            **os.environ,
            "GITHUB_OUTPUT": str(workdir / "output"),
            "GITHUB_REPOSITORY": "example/sdk",
            "RUNNER_TEMP": str(workdir),
            "TAG": "v1.2.3",
            "TAG_STATUS": str(tag),
            "REGISTRY_STATUS": str(registry),
            "RELEASE_STATUS": str(release),
        }
        return subprocess.run(
            ["bash", "-c", STUBS + step], cwd=ROOT, env=env,
            text=True, capture_output=True, check=False,
        )

    def test_plan_distinguishes_missing_versions_from_lookup_failures(self):
        cases = [
            (0, 200, 0, "false", "true"),
            (2, 404, 0, "true", "false"),
            (0, 404, 1, None, None),
            (2, 200, 1, None, None),
            (128, 404, 128, None, None),
            (2, 503, 1, None, None),
        ]
        for tag, registry, exit_code, should_release, reconcile in cases:
            with self.subTest(tag=tag, registry=registry), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "output"
                result = self.run_step(self.plan, Path(temp), tag=tag, registry=registry)
                self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
                if exit_code:
                    self.assertFalse(output.exists())
                    continue
                self.assertEqual(dict(line.split("=", 1) for line in output.read_text().splitlines()), {
                    "should_release": should_release,
                    "reconcile_release": reconcile,
                    "version": "1.2.3",
                    "tag": "v1.2.3",
                })

    def test_release_is_created_only_for_missing_metadata_and_an_existing_tag(self):
        for status, tag, exit_code, attempted, created in [
            (200, 0, 0, False, False),
            (404, 0, 0, True, True),
            (404, 2, 1, True, False),
            (403, 0, 1, False, False),
            (500, 0, 1, False, False),
        ]:
            with self.subTest(status=status, tag=tag), tempfile.TemporaryDirectory() as temp:
                workdir = Path(temp)
                result = self.run_step(self.release, workdir, tag=tag, release=status)
                self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
                self.assertEqual((workdir / "create-attempted").exists(), attempted)
                self.assertEqual((workdir / "release-created").exists(), created)
                if status in (403, 500):
                    self.assertIn(f"HTTP/2.0 {status}", result.stderr)


if __name__ == "__main__":
    unittest.main()
