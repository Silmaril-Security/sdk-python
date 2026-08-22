# Changelog

All notable changes to the Silmaril Firewall Python SDK are documented here.

## 0.6.0 - 2026-08-22

- Add the existing `shadow | warn | block` request mode contract to single and
  batch classification, with backend control when mode is omitted.
- Return and enforce the backend-resolved effective mode on every result.
- Treat a successful response without `mode` as legacy Block behavior during a
  rolling backend upgrade; current backends always return the effective mode.
- Retain `shadow_mode` compatibility: `True` requests Shadow, `False` requests
  Block, and explicit `mode` takes precedence.
- Make sync and async LangChain handlers enforce only effective Block results;
  Shadow and Warn preserve the host flow.

## 0.5.1 - 2026-07-29

- Add typed support for `code_generation`, `story_script_generation`,
  `game_generation`, `website_generation`, `clickup_terms_violation`, and
  `traditional_ai_abuse`.
- Extend ordered outcome collections, descriptions, validators, normalization
  fixtures, exports, and documentation without changing the `/classify` wire
  shape.

## 0.5.0 - 2026-07-15

- Send every `classify()` input as one complete event without client chunking.
- Preserve exact `metadata.conversationId` and emit one
  `metadata.silmaril.request_id` per event.
- Require backend `prediction` for enforcement while preserving optional
  outcome scores.

## 0.4.2 - 2026-06-02

- Add typed firewall outcome constants, ordered outcome tuples, descriptions,
  validation helpers, and response normalizers.
- Type `BlockResult.primary_outcome`, `outcome_scores`, `detector_scores`,
  and `detector_counts` around the canonical outcome taxonomy.
- Document simple outcome routing examples for shadow-mode `classify()`
  results.

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
