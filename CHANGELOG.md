# Changelog

All notable changes to the Silmaril Firewall Python SDK are documented here.

## 0.4.1 - 2026-05-24

- Recover the `0.4.x` release line after the `v0.4.0` Git tag was created but
  PyPI publishing failed before `silmaril-security-sdk==0.4.0` was published.
  Consumers should install `0.4.1`; maintainers should not reuse or move the
  stale `v0.4.0` tag.
- Harden the release workflow by publishing to PyPI before creating the Git tag,
  preventing a trusted-publisher configuration failure from leaving another
  tag-without-package split.
- Lower the `requests` dependency floor from `>=2.33.0` to the more broadly
  compatible `>=2.31.0`.
- Make concurrent chunk fanout tests deterministic by validating chunk metadata
  by `chunk_index` rather than request arrival order.
- Add public support, contribution, security, and source-available package
  metadata for external integrators.

## 0.4.0 - 2026-05-24

- Move score threshold decisions to tenant-owned Firewall backend
  configuration.
- Add SDK reconstruction metadata: `sdk_language`, `sdk_version`,
  `request_id`, `input_index`, `chunk_index`, and `chunk_count`.
- Rename blocking exceptions to `FirewallBlockedException` and
  `BatchFirewallBlockedException`, with deprecated prompt-named aliases kept
  for one release.
- Add chunk metadata support for long-input fanout and batch classification.

## 0.3.2

- Add request metadata support.
