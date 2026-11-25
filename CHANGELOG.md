# Changelog

## 1.0.0

The SDK now calls `POST /v2/classify` and returns the Firewall decision instead
of scores. No configuration changes are needed: keep the same `api_key` and
`api_url`.

- Derive the endpoint from the configured `api_url`: remove a trailing slash
  and a final `/v2/classify` or `/classify` path, keep the query string, and
  call `/v2/classify` on that base.
- Work against runtimes without `/v2/classify`. On HTTP 403, 404, or 405 the
  SDK sends the same request to `/classify`, uses that response as the result,
  and adapts a successful one to the same `BlockResult`. Evidence that version
  1 does not carry is null: `confidence` and `severity` are `None`. A client
  remembers a successful fallback and calls `/classify` directly afterwards.
  A successful response without a `results` key is adapted as a version 1
  response whichever path returned it, so a deployment that routes
  `/v2/classify` to the version 1 handler keeps blocking.
- `BlockResult` carries `action`, `source`, `confidence`, `severity`,
  `outcome`, `governance`, `model_id`, `review`, and `mode`, and adds a
  `blocked` property. `model_id` is provenance: the identifier of the model
  release that classified the item. `action` is the only field the SDK enforces on and is always
  `"allow"` or `"block"`: any response value other than `"allow"` is reported
  as `"block"`. `source`, `confidence`, `severity`, and `outcome` are returned
  as sent, including values this version does not know.
- Remove `prediction`, `score`, `threshold`, `primary_outcome`,
  `outcome_scores`, `detector_scores`, and `detector_counts` from
  `BlockResult`, and `score` and `threshold` from `FirewallBlockedException`,
  whose `result` is now required. Blocking exception messages report the
  decision evidence.
- Remove the `Prediction` type and the `normalize_primary_outcome`,
  `normalize_harmful_outcome`, `normalize_harmful_outcome_float_map`, and
  `normalize_harmful_outcome_int_map` helpers, which only served the removed
  fields. Add the `DecisionAction`, `DecisionSource`, `DecisionConfidence`,
  and `DecisionSeverity` types.
- Remove `OUTCOME_BENIGN`, `PRIMARY_OUTCOMES`, `PrimaryOutcome`,
  `PrimaryOutcomeValue`, and `is_primary_outcome`. `outcome` is `None` on an
  allow, so a benign outcome never appears on a result. The harmful-outcome
  constants, `HARMFUL_OUTCOMES`, `OUTCOME_DESCRIPTIONS`, and
  `is_harmful_outcome` stay.
- Reject a response without a `results` list that matches the inputs, or with
  a result that has no string `action`. Governance validation, the `review`
  acknowledgement requirement, the request body, and the retry policy are
  unchanged.
- SDK 0.x keeps working against `/classify`, which is frozen.

Migration:

| 0.x | 1.0 |
| --- | --- |
| `prediction` | `action` (`"MALICIOUS"` becomes `"block"`, `"BENIGN"` becomes `"allow"`) |
| `primary_outcome` | `outcome` |
| `score` / `threshold` | `confidence` / `severity` |
| `predictions` | `results` |

`confidence` and `severity` are ordered labels rather than numbers. They
describe the Firewall's assessment of the item and are set only when that
assessment is the reason for the block: they are `None` on an allow, on a
block that came from a governance policy, and on a block the Firewall reached
without that assessment.

## 0.11.1

- Honor `Retry-After` on retryable HTTP error responses in the synchronous
  client.
- Retry transient failures while reading successful response bodies, close
  responses reliably, and do not retry malformed successful JSON responses.

## 0.11.0

- Breaking: rename the tenant acceptable-use policy outcome to `abuse`.
  `OUTCOME_ABUSE` replaces the tenant-specific constant, and the
  `PrimaryOutcome` and `HarmfulOutcome` literals, `PRIMARY_OUTCOMES`,
  `HARMFUL_OUTCOMES`, and `OUTCOME_DESCRIPTIONS` list `abuse` in the same
  position. Responses that still carry the previous label continue to decode
  as an open-ended outcome string.

## 0.10.1

- Reject missing, non-numeric, non-finite, or out-of-range verdict `score` and
  `threshold` values, and reject `outcome_scores` and `detector_scores` values
  outside the inclusive `0` to `1` range.

## 0.10.0

- Classify model output by default in the LangChain handlers: `on_llm_end` is
  now part of `DEFAULT_HOOKS`. Pass `hooks=` to keep the previous set.
- Add `Firewall.protect()` and `AsyncFirewall.protect()`, which run a call
  while its input is classified and withhold the result of a blocked input.
  `AsyncFirewall` cancels the call when the block arrives first.
- Add a `parallel` option to the LangChain handlers,
  `protect_langchain_runnable()`, and the Deep Agents middleware. Input and
  tool-result checks run beside the model call; tool-call and model-output
  checks still wait for their verdict.
- Classify in the background in Shadow mode for the LangChain handlers and
  Deep Agents middleware unless `fail_open=False`.
- Send classifications that share `metadata.conversationId` in the order they
  were started, including the Block-mode `protect()` output check. Add
  `max_concurrent` and `max_queued` client options and `flush()`;
  `AsyncFirewall.aclose()` drains background classifications. Background
  classifications are also bounded in total at `max_concurrent * max_queued`.
- Copy `protect()` metadata at call time, and withdraw an unsent Block-mode
  check when the operation fails or the awaiting task is cancelled. A withdrawn
  check keeps its place in the conversation order, and a cancelled operation
  is awaited before `protect()` returns.
- Keep a deferred LangChain block visible to every callback of its run, close
  the async pool even when the task awaiting `aclose()` is cancelled, and add
  `flush()` to the async LangChain handler.

## 0.9.2

- Reject zero, negative, and non-finite asynchronous request timeouts at
  construction, matching the synchronous Firewall validation contract.

## 0.9.1

- Retain the safe optional LangChain ranges `langchain-core>=0.3.85,<1` and
  `langchain-core>=1.3.3,<2`, excluding affected 1.0–1.2 and early 1.3
  serialization intervals while preserving qualified 0.3 applications.
- Honor the released Deep Agents 0.7.x general-purpose profile contract in
  protected factories: disabled defaults stay disabled, profile description and
  prompt customizations are preserved, parent tools and skills reach the
  protected default, string models are constructed once, and explicit named
  children are not duplicated. Released profiles expose `enabled`,
  `description`, and `system_prompt`; they do not expose a rename field.
- Reject duplicate, foreign, or option-conflicting Silmaril middleware instead
  of silently weakening or duplicating a protected boundary. Fail actionably
  when a runtime-only string profile or released prompt/middleware combination
  cannot be represented exactly for the protected child.
- Add optional `isolated`/`fork` mode to protected compiled subagents while
  preserving the exact protected graph and Firewall provenance. Fork mode
  remains beta upstream.

## 0.9.0

- Support native remote Deep Agents subagents with parent delegation/result checks
  and independently protected child graphs. Preserve child thread identity, parent
  lineage and async task lifecycle, and inspect background result notifications.
- Qualify isolated/fork launches, updates, checks, cancellation and failures using
  real framework graphs and SDK transport against a loopback Agent Protocol peer.

- Require Deep Agents 0.7.30 or newer within the 0.7.x line for the qualified
  remote lifecycle contract. Core and other optional integration floors remain unchanged.
- Include Python 3.14 and 3.15 package classifiers alongside the existing tested runtimes.

## 0.8.2

- Reject zero and non-finite synchronous request timeouts when constructing the
  client, before they can fail inside the HTTP transport.

## 0.8.1

- Preserve original native binding factories through post-protection
  `with_config`, `bind`, retry, listener and type composition, including
  minimum supported LangChain versions whose native binding methods drop
  factories while rebuilding.
- Keep native static defaults and dynamic routing active exactly once and
  clear protected stream bookkeeping even through retained binding layers.
- Forward sync/async transforms incrementally and clean up closed or failed
  iterators without buffering upstream chunks.
- Verify post-protection composition against source and installed wheels.

## 0.8.0

- Add `protect_langchain_runnable()` to capture exact per-invocation native
  identity at standalone model/tool boundaries, preserving normal runnable
  callbacks, batching and streaming.
- Automatically use the active native `configurable.thread_id` as Firewall
  conversation identity in sync/async LangChain callbacks and Deep Agents
  model/tool middleware. Reusable agents and concurrent threads retain separate
  identities; threadless checks omit conversation identity.
- **Breaking:** remove handler and middleware `conversation_id` options. Pass
  the framework thread ID at invocation instead. Direct `classify()` metadata
  APIs are unchanged.
- Retain callback identities through metadata-free completion events and clear
  state after completion, errors, blocking exceptions, and root termination.
  Bound outstanding callback bookkeeping and clear protected streaming
  invocations when consumers close early. Callback-only integrations omit
  ambiguous nested standalone identity rather than borrowing a parent ID.
- Cover native identity with real framework execution, subagent, concurrency,
  checkpoint-continuation, and minimum-dependency tests.

## 0.7.6

- Reject negative `detector_counts` before returning results or invoking
  callbacks across sync and async single and batch classification, while
  preserving zero and positive counts and future detector labels.

## 0.7.5

- Add optional `review` (`"standard"` or `"deep"`) on sync and async Firewall
  clients and on single/batch `classify` calls. `None` means that layer makes
  no selection. A per-call `None` inherits an explicit client setting. The wire
  field is omitted only when neither layer has an explicit selection, so tenant
  AppConfig remains authoritative. An explicit per-call value overrides the
  client default without mutating it. Existing adapters inherit the client
  setting.
- Validate top-level review acknowledgement for every explicit selection and
  raise non-retryable `ReviewCompatibilityError` on a missing or mismatched
  value before callbacks or enforcement.

## 0.7.4

- Preserve optional schema 6 governance response evidence (`resource`,
  `identity_revision`, and open-ended `reason`) on public decisions while
  keeping legacy responses compatible.

## 0.7.3

- Send typed governance resources in the runtime's top-level `resource` and
  aligned `resources` fields, preserving compatibility attribution metadata.
- Reject invalid canonical resource references before submitting a request.
  Caller-supplied agent metadata remains attribution, not authenticated identity.

All notable changes to the Silmaril Firewall Python SDK are documented here.

## 0.7.2 - 2026-10-12

- Preserve non-empty outcome labels introduced by future Firewall versions in
  primary outcomes and score/count maps while keeping known-taxonomy guards exact.
  Continue rejecting malformed labels, invalid numeric map values, and `benign`
  harmful-map keys.
- Correct the current mode documentation: omitted response mode remains unset while
  direct SDK calls retain their compatibility Block fallback for enforcement.

## 0.7.1 - 2026-10-04

- Keep API keys on individual Firewall requests instead of mutating a caller-owned
  requests session's default headers. Preserve shared-session isolation through
  retries and batch classification.

## 0.7.0 - 2026-09-30

- Add optional Deep Agents middleware and protected constructors for root, general-purpose,
  declarative, and explicitly protected compiled subagents. The compiled graph factory
  records middleware installation and constructors reject unverified graphs.
- Enforce input, tool-call, tool-result, and non-streamed model-output decisions in
  Block mode with safe continuation and a repeated-denial cap. Shadow and Warn report
  decisions without changing content.
- Enable LangChain tool start/end callbacks by default, remove repeated history scans,
  and emit unique classification request IDs with LangChain run IDs as correlation metadata.
- Add typed governance request context and response decisions for single and batch calls,
  including compatibility with older responses that omit governance.

## 0.6.1 - 2026-09-24

- Add public `AsyncFirewall` single and batch classification with a persistent
  `httpx.AsyncClient` pool, async context management, and task-safe concurrent
  use on one event loop.
- Preserve synchronous mode, callback, blocking, metadata, retry, sanitization,
  and one-request batch behavior in the native async API.
- Support caller-owned async clients, idempotent close, cancellation, and clear
  closed-client and cross-event-loop errors.
- Drain in-flight async requests and retries on `aclose()` while rejecting new
  work, so shutdown cannot fail a concurrent classification. Closing from inside
  one's own in-flight classification raises instead of closing the pool early,
  and concurrent callers return only once the pool is closed. Cancelling an
  `aclose()` waiter does not abort shutdown or leak an SDK-owned client.
- Retry HTTPX transport failures raised while streaming a response body, for
  both success and error bodies, instead of surfacing a partial error.
- Snapshot async batch texts, hooks, and tool names before sending so results
  and blocked items always describe the request that was sent.
- Stream async error responses and stop reading at the 64 KiB cap instead of
  buffering an oversized body before truncating it.
- Allow async LangChain handlers to share an `AsyncFirewall` pool while keeping
  the existing `Firewall.as_async_langchain_handler()` temporary-client path.

## 0.6.0 - 2026-08-22

- Add the existing `shadow | warn | block` request mode contract to single and
  batch classification, with backend control when mode is omitted.
- Return and enforce the backend-resolved effective mode on current-backend
  results.
- Keep an explicit request mode authoritative when a legacy or mixed-version
  backend omits or disagrees about `mode`. When both request and response omit
  mode, leave the result mode unset so integrations can preserve their pre-0.6
  behavior without increasing enforcement.
- Retain `shadow_mode` compatibility: `True` requests Shadow, `False` requests
  Block, and explicit `mode` takes precedence.
- Make sync and async LangChain handlers enforce only effective Block results;
  Shadow and Warn preserve the host flow.

## 0.5.1 - 2026-07-29

- Add typed support for `code_generation`, `story_script_generation`,
  `game_generation`, `website_generation`, `traditional_ai_abuse`, and the
  tenant acceptable-use policy outcome now named `abuse`.
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
