# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import json
from typing import Any

import pytest
import requests
from wire import (
    TEST_API_URL,
    V2_URL,
    FakeResponse,
    RecordingAdapter,
    decision,
    governance_block,
    model_block,
    results,
    v1,
)

from silmaril_security.sdk import (
    OUTCOME_SECRET_EXPOSURE,
    BatchFirewallBlockedException,
    BatchPromptBlockedException,
    BlockResult,
    ClassifyEvent,
    Firewall,
    FirewallBlockedException,
    GovernanceResource,
    HookLabel,
    PromptBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk._version import VERSION
from silmaril_security.sdk.firewall import _MAX_ERROR_BODY_BYTES, _block_results


def test_constructor_requires_key_and_url():
    with pytest.raises(ValueError, match="api_key is required"):
        Firewall(api_key="", api_url=TEST_API_URL)
    with pytest.raises(ValueError, match="api_url is required"):
        Firewall(api_key="sk", api_url="")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_constructor_rejects_invalid_sync_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        Firewall(api_key="sk-test", api_url=TEST_API_URL, timeout=timeout)


def test_constructor_preserves_session_defaults_and_unrelated_request_headers():
    session = requests.Session()
    session.headers["X-Custom-Default"] = "preserved"
    original_headers = dict(session.headers)

    fw = Firewall(api_key="sk-test", api_url=TEST_API_URL, session=session)
    unrelated = session.prepare_request(
        requests.Request("GET", "https://unrelated.test.invalid/resource")
    )

    assert fw._session is session
    assert dict(session.headers) == original_headers
    assert unrelated.headers["X-Custom-Default"] == "preserved"
    assert "x-api-key" not in unrelated.headers
    assert "content-type" not in unrelated.headers


def test_deprecated_exception_names_alias_new_names():
    assert PromptBlockedException is FirewallBlockedException
    assert BatchPromptBlockedException is BatchFirewallBlockedException


@pytest.mark.parametrize("timeout", [0.001, 10.0])
def test_classify_posts_wire_shape_and_returns_result(monkeypatch, timeout):
    fw = Firewall(api_key="sk-test", api_url=TEST_API_URL, timeout=timeout)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, results(decision()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify(
        "hello",
        hook=HookLabel.USER_INPUT,
        tool_name="chat",
        request_id="req-test",
    )

    assert result == BlockResult(action="allow")
    assert calls[0]["url"] == V2_URL
    assert calls[0]["headers"] == {
        "x-api-key": "sk-test",
        "content-type": "application/json",
    }
    assert calls[0]["timeout"] == timeout
    assert calls[0]["allow_redirects"] is False
    assert calls[0]["stream"] is True
    assert json.loads(calls[0]["data"]) == {
        "text": "hello",
        "hook": "user_input",
        "tool_name": "chat",
        "metadata": {
            "silmaril": {
                "sdk_language": "python",
                "sdk_version": VERSION,
                "request_id": "req-test",
            }
        },
    }


def test_classify_preserves_governance_response_evidence_and_absence():
    resource = {
        "kind": "mcp_tool",
        "id": "create_issue",
        "parent_id": "github",
    }
    adapter = RecordingAdapter(
        [
            (
                200,
                results(
                    governance_block(
                        {
                            "action": "block",
                            "policy_version": "schema6",
                            "rule_id": "deny-github",
                            "resource": resource,
                            "identity_revision": "catalog-9",
                            "reason": "identity_unresolved",
                        }
                    )
                ),
            ),
            (
                200,
                results(
                    decision(
                        governance={
                            "action": "allow",
                            "policy_version": "schema6",
                            "resource": resource,
                            "identity_revision": "\u00a0\ufeff\u0085",
                            "reason": "\u001cfuture_identity_reason\u3000",
                        }
                    )
                ),
            ),
            (200, results(decision())),
        ]
    )
    session = requests.Session()
    session.mount("https://api.test.invalid/", adapter)
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, session=session)

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("governed")

    assert exc_info.value.result.source == "governance"
    governance = exc_info.value.result.governance
    assert governance is not None
    assert governance.action == "block"
    assert governance.policy_version == "schema6"
    assert governance.rule_id == "deny-github"
    assert governance.resource == GovernanceResource(
        kind="mcp_tool",
        id="create_issue",
        parent_id="github",
    )
    assert governance.identity_revision == "catalog-9"
    assert governance.reason == "identity_unresolved"

    future = fw.classify("future reason")
    assert future.governance is not None
    assert future.governance.identity_revision == "\u00a0\ufeff\u0085"
    assert future.governance.reason == "\u001cfuture_identity_reason\u3000"
    assert fw.classify("ungoverned").governance is None


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"resource": None, "identity_revision": None, "reason": None},
    ],
    ids=["omitted", "explicit-null"],
)
def test_governance_response_accepts_absent_optional_evidence(evidence):
    adapter = RecordingAdapter(
        [
            (
                200,
                results(
                    decision(
                        governance={
                            "action": "allow",
                            "policy_version": "legacy-policy",
                            "rule_id": "legacy-rule",
                            **evidence,
                        }
                    )
                ),
            )
        ]
    )
    session = requests.Session()
    session.mount("https://api.test.invalid/", adapter)
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, session=session)

    result = fw.classify("optional evidence")

    assert result.governance is not None
    assert result.governance.action == "allow"
    assert result.governance.policy_version == "legacy-policy"
    assert result.governance.rule_id == "legacy-rule"
    assert result.governance.resource is None
    assert result.governance.identity_revision is None
    assert result.governance.reason is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("identity_revision", 1),
        ("identity_revision", {}),
        ("identity_revision", ""),
        ("identity_revision", " \t"),
        (
            "identity_revision",
            "\u001c\u001d\u001e\u001f\u0085\u00a0\u1680\u2003\u2028\u2029\u202f\u205f\u3000",
        ),
        ("reason", 1),
        ("reason", {}),
        ("reason", ""),
        ("reason", " \t"),
        ("reason", "\u001c\u0085\u00a0\u2007\u2029\u3000"),
    ],
)
def test_governance_response_rejects_malformed_string_evidence(field, value):
    adapter = RecordingAdapter(
        [
            (
                200,
                results(
                    decision(
                        governance={
                            "action": "allow",
                            "policy_version": "schema6",
                            field: value,
                        }
                    )
                ),
            )
        ]
    )
    session = requests.Session()
    session.mount("https://api.test.invalid/", adapter)
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, session=session)

    with pytest.raises(
        ValueError,
        match=rf"response governance {field} must be a non-empty string",
    ):
        fw.classify("malformed evidence")


@pytest.mark.parametrize("kind", [None, 1, [], {}])
def test_governance_response_rejects_non_string_resource_kind(kind):
    adapter = RecordingAdapter([(200, results(decision(governance={
        "action": "allow", "policy_version": "schema6",
        "resource": {"kind": kind, "id": "search"},
    })))])
    session = requests.Session()
    session.mount("https://api.test.invalid/", adapter)
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, session=session)
    with pytest.raises(ValueError, match="resource kind must be a string"):
        fw.classify("malformed resource")


def test_shared_session_uses_client_headers_for_batch_and_retries(monkeypatch):
    benign = results(decision())
    adapter = RecordingAdapter(
        [
            (200, benign),
            (200, results(decision(), decision())),
            (429, "rate limited"),
            (200, benign),
            (200, benign),
        ]
    )
    session = requests.Session()
    session.headers["X-Custom-Default"] = "preserved"
    original_headers = dict(session.headers)
    session.mount("https://api.test.invalid/", adapter)
    first = Firewall(api_key="sk-first", api_url=TEST_API_URL, session=session)
    second = Firewall(api_key="sk-second", api_url=TEST_API_URL, session=session)
    monkeypatch.setattr("silmaril_security.sdk.firewall.time.sleep", lambda _delay: None)

    first.classify("first")
    second.classify_batch(["batch-a", "batch-b"])
    first.classify("retry")
    second.classify("second")

    assert [request.headers["x-api-key"] for request in adapter.requests] == [
        "sk-first",
        "sk-second",
        "sk-first",
        "sk-first",
        "sk-second",
    ]
    assert all(
        request.headers["content-type"] == "application/json" for request in adapter.requests
    )
    assert all(
        request.headers["X-Custom-Default"] == "preserved" for request in adapter.requests
    )
    assert json.loads(adapter.requests[1].body)["texts"] == ["batch-a", "batch-b"]
    assert json.loads(adapter.requests[2].body)["text"] == "retry"
    assert json.loads(adapter.requests[3].body)["text"] == "retry"
    assert dict(session.headers) == original_headers


def test_classify_posts_metadata_when_provided(monkeypatch):
    fw = Firewall(api_key="sk-test", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, results(decision()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    fw.classify(
        "hello",
        hook=HookLabel.USER_INPUT,
        metadata={
            "langgraph": {
                "thread_id": "thread-123",
                "run_id": "run-123",
                "message_id": "msg-123",
            }
        },
        request_id="req-meta",
    )

    assert json.loads(calls[0]["data"]) == {
        "text": "hello",
        "hook": "user_input",
        "metadata": {
            "langgraph": {
                "thread_id": "thread-123",
                "run_id": "run-123",
                "message_id": "msg-123",
            },
            "silmaril": {
                "sdk_language": "python",
                "sdk_version": VERSION,
                "request_id": "req-meta",
            },
        },
    }


def test_classify_raises_on_block_action(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            results(
                model_block(
                    confidence="high",
                    severity="moderate",
                    outcome=OUTCOME_SECRET_EXPOSURE,
                )
            ),
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("ignore previous", hook=HookLabel.USER_INPUT, tool_name="chat")

    assert exc_info.value.result == BlockResult(
        action="block",
        source="model",
        confidence="high",
        severity="moderate",
        outcome=OUTCOME_SECRET_EXPOSURE,
    )
    assert exc_info.value.hook == HookLabel.USER_INPUT
    assert exc_info.value.tool_name == "chat"
    assert str(exc_info.value) == (
        "Request blocked by Silmaril Firewall (action=block, source=model, confidence=high, "
        "severity=moderate, outcome=secret_exposure): 'ignore previous'"
    )


@pytest.mark.parametrize("action", ["block", "quarantine", "ALLOW", ""])
def test_classify_blocks_on_any_action_other_than_allow(monkeypatch, action):
    events: list[ClassifyEvent] = []
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, on_classify=events.append)
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(200, results(decision(action))),
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("payload")

    assert exc_info.value.result == BlockResult(action="block")
    assert exc_info.value.result.blocked is True
    assert events[0].blocked is True
    assert fw.classify("payload", mode="warn").action == "block"


def test_evidence_never_affects_enforcement(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200,
            results(
                decision(
                    source="model",
                    confidence="very_high",
                    severity="critical",
                    outcome=OUTCOME_SECRET_EXPOSURE,
                    governance={"action": "block", "policy_version": "schema6"},
                )
            ),
        ),
    )

    result = fw.classify("payload")

    assert result.action == "allow"
    assert result.blocked is False


def test_classify_shadow_mode_suppresses_block_and_emits_event(monkeypatch):
    events = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        shadow_mode=True,
        on_classify=events.append,
    )

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, results(model_block()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("ignore previous", hook=HookLabel.TOOL_RESPONSE)

    assert result.action == "block"
    assert len(events) == 1
    assert events[0].blocked is True
    assert events[0].shadow_mode is True
    assert events[0].result == result


def test_classify_per_call_shadow_mode_override(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, results(model_block()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(FirewallBlockedException):
        fw.classify("attack", shadow_mode=False)


def test_classify_omits_mode_for_backend_control_and_consumes_adapted_warn(monkeypatch):
    events = []
    calls: list[dict[str, Any]] = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        on_classify=events.append,
    )
    fw._route.legacy = True

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, v1("MALICIOUS", mode="warn"))

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("attack", request_id="req-warn")

    assert "mode" not in json.loads(calls[0]["data"])
    assert result.mode == "warn"
    assert events[0].mode == "warn"
    assert events[0].shadow_mode is False


def test_explicit_mode_precedes_legacy_shadow_mode(monkeypatch):
    calls: list[dict[str, Any]] = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        mode="warn",
        shadow_mode=False,
    )
    fw._route.legacy = True

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, v1("MALICIOUS", mode="shadow"))

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("attack", mode="block", shadow_mode=True, request_id="req-mode")

    assert json.loads(calls[0]["data"])["mode"] == "block"
    assert exc_info.value.result.mode == "block"


def test_explicit_shadow_cannot_be_escalated_by_adapted_response(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, mode="shadow")
    fw._route.legacy = True

    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(200, v1("MALICIOUS", mode="block")),
    )

    result = fw.classify("attack")

    assert result.mode == "shadow"


def test_classify_rejects_invalid_adapted_mode(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    fw._route.legacy = True

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, v1(mode="enforce"))

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(ValueError, match="backend mode must be shadow, warn, or block"):
        fw.classify("payload")


def test_version_2_response_ignores_unknown_fields(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200,
            results(
                model_block(mode="enforce", score=0.99, prediction="BENIGN", trace={"id": 7}),
                mode="shadow",
                model_id="silmaril-cascade-1-6",
            ),
        ),
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("payload")

    assert exc_info.value.result == BlockResult(action="block", source="model")


def test_mode_less_response_preserves_absence():
    result = _block_results(results(model_block()), expected_length=1, mode=None, review=None)[0]

    assert result.mode is None


def test_requested_warn_is_preserved_when_legacy_backend_omits_mode(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, mode="warn")

    monkeypatch.setattr(fw, "_post_json", lambda url, payload: v1("MALICIOUS"))

    result = fw.classify("attack")

    assert result.mode == "warn"


def test_legacy_shadow_mode_is_preserved_when_backend_omits_mode(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(200, results(model_block())),
    )

    result = fw.classify("attack")

    assert result.mode == "shadow"


def test_classify_event_positional_constructor_remains_compatible():
    result = BlockResult("allow")
    event = ClassifyEvent(
        HookLabel.USER_INPUT,
        None,
        "hello",
        result,
        False,
        True,
    )

    assert result.blocked is False
    assert result.mode is None
    assert event.mode == "shadow"
    assert event.shadow_mode is True


def test_classify_batch_wire_shape_and_block_error(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, results(model_block(), decision(), model_block()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(BatchPromptBlockedException) as exc_info:
        fw.classify_batch(
            ["first", "second", "third"],
            hooks=[HookLabel.USER_INPUT, HookLabel.TOOL_RESPONSE, HookLabel.TOOL_RESPONSE],
            tool_names=["chat", "read_file", None],
        )

    assert len(exc_info.value.results) == 3
    assert [item.index for item in exc_info.value.blocked] == [0, 2]
    assert exc_info.value.blocked[0].tool_name == "chat"
    assert calls[0]["url"] == V2_URL
    body = json.loads(calls[0]["data"])
    assert body["texts"] == ["first", "second", "third"]
    assert "mode" not in body
    assert all(result.mode is None for result in exc_info.value.results)
    assert "threshold" not in body
    assert body["hooks"] == ["user_input", "tool_response", "tool_response"]
    assert body["tool_names"] == ["chat", "read_file", None]
    assert [item["silmaril"]["input_index"] for item in body["metadata"]] == [0, 1, 2]
    assert all("chunk_index" not in item["silmaril"] for item in body["metadata"])
    assert all("chunk_count" not in item["silmaril"] for item in body["metadata"])


def test_single_blocked_batch_item_message_carries_evidence(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200, results(decision(), model_block(outcome=OUTCOME_SECRET_EXPOSURE))
        ),
    )

    with pytest.raises(BatchFirewallBlockedException) as exc_info:
        fw.classify_batch(["safe", "attack"])

    assert str(exc_info.value) == (
        "Batch request blocked by Silmaril Firewall at index 1 "
        "(action=block, source=model, outcome=secret_exposure)"
    )


def test_classify_batch_serializes_metadata(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, results(decision(), decision()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    fw.classify_batch(
        ["first", "second"],
        metadata=[
            {"langgraph": {"run_id": "run-a"}},
            None,
        ],
        request_id="batch-req",
    )

    assert json.loads(calls[0]["data"]) == {
        "texts": ["first", "second"],
        "metadata": [
            {
                "langgraph": {"run_id": "run-a"},
                "silmaril": {
                    "sdk_language": "python",
                    "sdk_version": VERSION,
                    "request_id": "batch-req",
                    "input_index": 0,
                },
            },
            {
                "silmaril": {
                    "sdk_language": "python",
                    "sdk_version": VERSION,
                    "request_id": "batch-req",
                    "input_index": 1,
                }
            },
        ],
    }


def test_classify_batch_shadow_mode_returns_results(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, results(model_block()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    decisions = fw.classify_batch(["first"], shadow_mode=True)

    assert decisions == [BlockResult(action="block", source="model", mode="shadow")]


def test_classify_batch_rejects_bad_lengths():
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    with pytest.raises(ValueError, match="hooks length 1"):
        fw.classify_batch(["a", "b"], hooks=[HookLabel.USER_INPUT])
    with pytest.raises(ValueError, match="tool_names length 1"):
        fw.classify_batch(["a", "b"], tool_names=["tool"])
    with pytest.raises(ValueError, match="metadata length 1"):
        fw.classify_batch(["a", "b"], metadata=[{"run_id": "run-a"}])
    with pytest.raises(ValueError, match="texts must not be empty"):
        fw.classify_batch([])


def test_classify_sends_long_event_once_with_canonical_metadata(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, results(decision()))

    monkeypatch.setattr(fw._session, "post", fake_post)

    text = "b" * 4001
    result = fw.classify(
        text,
        hook=HookLabel.TOOL_RESPONSE,
        tool_name="fetch_webpage",
        metadata={"conversationId": "conversation-123", "conversation_id": "inert"},
        request_id="event-uuid",
    )

    assert result.action == "allow"
    assert len(calls) == 1
    body = json.loads(calls[0]["data"])
    assert body == {
        "text": text,
        "mode": "shadow",
        "hook": "tool_response",
        "tool_name": "fetch_webpage",
        "metadata": {
            "conversationId": "conversation-123",
            "conversation_id": "inert",
            "silmaril": {
                "sdk_language": "python",
                "sdk_version": VERSION,
                "request_id": "event-uuid",
            },
        },
    }


MALFORMED_BODIES = [
    ({}, "missing required 'prediction' field"),
    ({"action": "allow"}, "missing required 'prediction' field"),
    ({"detail": "accepted"}, "missing required 'prediction' field"),
    ({"results": None}, "'results' must be a list"),
    ({"results": {"action": "allow"}}, "'results' must be a list"),
    ({"results": []}, "results length 0 does not match texts length 1"),
    (results(decision(), decision()), "results length 2 does not match texts length 1"),
    ({"results": ["allow"]}, "missing required 'action' string"),
    ({"results": [{}]}, "missing required 'action' string"),
    ({"results": [{"action": None}]}, "missing required 'action' string"),
    ({"results": [{"action": 1}]}, "missing required 'action' string"),
    ({"results": [{"action": ["allow"]}]}, "missing required 'action' string"),
]


@pytest.mark.parametrize(("body", "message"), MALFORMED_BODIES)
def test_classify_rejects_malformed_response_before_callbacks(monkeypatch, body, message):
    events: list[ClassifyEvent] = []
    fw = Firewall(
        api_key="sk", api_url=TEST_API_URL, shadow_mode=True, on_classify=events.append
    )
    monkeypatch.setattr(
        fw._session, "post", lambda *_args, **_kwargs: FakeResponse(200, body)
    )

    with pytest.raises(ValueError, match=message):
        fw.classify("malformed")

    assert events == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (results(decision()), "results length 1 does not match texts length 2"),
        (results(decision(), {"source": "model"}), "missing required 'action' string"),
        ({"results": None, "predictions": [v1(), v1()]}, "'results' must be a list"),
        ({"predictions": [v1(), {"score": 0.5}]}, "missing required 'prediction' field"),
        ({"predictions": [v1()]}, "results length 1 does not match texts length 2"),
        (v1(), "results length 1 does not match texts length 2"),
    ],
)
def test_classify_batch_rejects_malformed_response_before_callbacks(monkeypatch, body, message):
    events: list[ClassifyEvent] = []
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, mode="shadow", on_classify=events.append)
    monkeypatch.setattr(
        fw._session, "post", lambda *_args, **_kwargs: FakeResponse(200, body)
    )

    with pytest.raises(ValueError, match=message):
        fw.classify_batch(["first", "second"])

    assert events == []


def test_surfaces_decision_evidence(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            results(
                model_block(
                    confidence="high",
                    severity="moderate",
                    outcome=OUTCOME_SECRET_EXPOSURE,
                )
            ),
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("leak token")

    assert result == BlockResult(
        action="block",
        source="model",
        confidence="high",
        severity="moderate",
        outcome=OUTCOME_SECRET_EXPOSURE,
        mode="shadow",
    )


UNFAMILIAR_EVIDENCE = {
    "source": "policy_pack",
    "confidence": "certain",
    "severity": "catastrophic",
    "outcome": "future_policy_violation",
}


@pytest.mark.parametrize("mode", ["warn", "shadow"])
def test_surfaces_unfamiliar_evidence_for_nonblocking_modes(monkeypatch, mode):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200, results(decision("block", **UNFAMILIAR_EVIDENCE))
        ),
    )

    result = fw.classify("future attack", mode=mode)

    assert result == BlockResult(action="block", mode=mode, **UNFAMILIAR_EVIDENCE)


def test_unfamiliar_evidence_explicit_block_raises_with_result(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200, results(decision("block", **UNFAMILIAR_EVIDENCE))
        ),
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("future attack", mode="block")

    assert exc_info.value.result == BlockResult(
        action="block", mode="block", **UNFAMILIAR_EVIDENCE
    )


def test_batch_surfaces_and_enforces_unfamiliar_evidence(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200, results(decision("block", **UNFAMILIAR_EVIDENCE))
        ),
    )

    decisions = fw.classify_batch(["future attack"], mode="warn")
    assert decisions == [BlockResult(action="block", mode="warn", **UNFAMILIAR_EVIDENCE)]

    with pytest.raises(BatchFirewallBlockedException) as exc_info:
        fw.classify_batch(["future attack"], mode="block")
    assert exc_info.value.blocked[0].result.outcome == "future_policy_violation"


def test_retries_retryable_status(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    responses = [
        FakeResponse(429, "rate limited"),
        FakeResponse(503, "unavailable"),
        FakeResponse(200, results(decision())),
    ]
    sleeps: list[float] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(fw._session, "post", fake_post)
    monkeypatch.setattr("silmaril_security.sdk.firewall.time.sleep", sleeps.append)

    result = fw.classify("hello")

    assert result.action == "allow"
    assert sleeps == [1, 2]


def test_api_error_on_non_retryable_status(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(401, "bad key", reason="Unauthorized")

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(SilmarilApiError) as exc_info:
        fw.classify("hello")

    assert exc_info.value.status == 401
    assert exc_info.value.status_text == "Unauthorized"
    assert exc_info.value.body == "bad key"
    assert "bad key" not in str(exc_info.value)


def test_api_error_on_redirect_status(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(302, "redirect", reason="Found", headers={"Location": "https://evil.test/"})

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(SilmarilApiError) as exc_info:
        fw.classify("hello")

    assert calls[0]["allow_redirects"] is False
    assert exc_info.value.status == 302
    assert exc_info.value.status_text == "Found"
    assert exc_info.value.body == "redirect"


def test_api_error_body_is_capped_and_redacted(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, max_retries=0)
    body = "x" * (_MAX_ERROR_BODY_BYTES + 1024)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(500, body, reason="Internal Server Error")

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(SilmarilApiError) as exc_info:
        fw.classify("hello")

    assert exc_info.value.body == body[:_MAX_ERROR_BODY_BYTES]
    assert len(exc_info.value.body) == _MAX_ERROR_BODY_BYTES
    assert body[:128] not in str(exc_info.value)


def test_network_error_retries_then_raises(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, max_retries=1)
    sleeps: list[float] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        raise requests.Timeout("timed out")

    monkeypatch.setattr(fw._session, "post", fake_post)
    monkeypatch.setattr("silmaril_security.sdk.firewall.time.sleep", sleeps.append)

    with pytest.raises(requests.Timeout):
        fw.classify("hello")
    assert sleeps == [1]
