# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from silmaril_security.sdk import (
    OUTCOME_SECRET_EXPOSURE,
    BatchFirewallBlockedException,
    BatchPromptBlockedException,
    BlockResult,
    ClassifyEvent,
    Firewall,
    FirewallBlockedException,
    HookLabel,
    PromptBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk.firewall import _MAX_ERROR_BODY_BYTES, _block_result_from_json

TEST_API_URL = "https://api.test.invalid/classify"


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: dict[str, Any] | str,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.reason = reason or f"status-{status_code}"
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self._body if isinstance(self._body, str) else str(self._body)

    def json(self) -> dict[str, Any]:
        assert isinstance(self._body, dict)
        if isinstance(self._body.get("prediction"), str):
            return {"mode": "block", **self._body}
        predictions = self._body.get("predictions")
        if isinstance(predictions, list):
            return {
                **self._body,
                "predictions": [
                    {"mode": "block", **item} if isinstance(item, dict) else item
                    for item in predictions
                ],
            }
        return self._body


def test_constructor_requires_key_and_url():
    with pytest.raises(ValueError, match="api_key is required"):
        Firewall(api_key="", api_url=TEST_API_URL)
    with pytest.raises(ValueError, match="api_url is required"):
        Firewall(api_key="sk", api_url="")


def test_deprecated_exception_names_alias_new_names():
    assert PromptBlockedException is FirewallBlockedException
    assert BatchPromptBlockedException is BatchFirewallBlockedException


def test_classify_posts_wire_shape_and_returns_result(monkeypatch):
    fw = Firewall(api_key="sk-test", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.12, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify(
        "hello",
        hook=HookLabel.USER_INPUT,
        tool_name="chat",
        request_id="req-test",
    )

    assert result == BlockResult(
        prediction="BENIGN",
        score=0.12,
        threshold=0.5,
        mode="block",
    )
    assert fw._session.headers["x-api-key"] == "sk-test"
    assert fw._session.headers["content-type"] == "application/json"
    assert calls[0]["url"] == TEST_API_URL
    assert calls[0]["timeout"] == 10.0
    assert calls[0]["allow_redirects"] is False
    assert calls[0]["stream"] is True
    assert json.loads(calls[0]["data"]) == {
        "text": "hello",
        "hook": "user_input",
        "tool_name": "chat",
        "metadata": {
            "silmaril": {
                "sdk_language": "python",
                "sdk_version": "0.6.0",
                "request_id": "req-test",
            }
        },
    }


def test_classify_posts_metadata_when_provided(monkeypatch):
    fw = Firewall(api_key="sk-test", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.12, "threshold": 0.5})

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
                "sdk_version": "0.6.0",
                "request_id": "req-meta",
            },
        },
    }


def test_classify_enforces_backend_block_mode(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.1,
                "threshold": 0.9,
                "mode": "block",
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("ignore previous", hook=HookLabel.USER_INPUT, tool_name="chat")

    assert exc_info.value.score == 0.1
    assert exc_info.value.threshold == 0.9
    assert exc_info.value.hook == HookLabel.USER_INPUT
    assert exc_info.value.tool_name == "chat"
    assert exc_info.value.result is not None


def test_classify_shadow_mode_suppresses_block_and_emits_event(monkeypatch):
    events = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        shadow_mode=True,
        on_classify=events.append,
    )

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "mode": "shadow",
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("ignore previous", hook=HookLabel.TOOL_RESPONSE)

    assert result.score == 0.91
    assert len(events) == 1
    assert events[0].blocked is True
    assert events[0].shadow_mode is True
    assert events[0].result == result


def test_classify_per_call_shadow_mode_override(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, {"prediction": "MALICIOUS", "score": 0.91, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(FirewallBlockedException):
        fw.classify("attack", shadow_mode=False)


def test_classify_omits_mode_for_backend_control_and_consumes_effective_warn(monkeypatch):
    events = []
    calls: list[dict[str, Any]] = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        on_classify=events.append,
    )

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "mode": "warn",
            },
        )

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

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "mode": "shadow",
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("attack", mode="block", shadow_mode=True, request_id="req-mode")

    assert json.loads(calls[0]["data"])["mode"] == "block"
    assert result.mode == "shadow"


def test_classify_rejects_invalid_backend_mode(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "prediction": "BENIGN",
                "score": 0.1,
                "threshold": 0.5,
                "mode": "enforce",
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(ValueError, match="backend mode must be shadow, warn, or block"):
        fw.classify("payload")


def test_legacy_mode_less_response_defaults_to_block():
    result = _block_result_from_json(
        {"prediction": "MALICIOUS", "score": 0.9, "threshold": 0.5}
    )

    assert result.mode == "block"


def test_requested_warn_does_not_override_legacy_mode_less_response(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, mode="warn")

    monkeypatch.setattr(
        fw,
        "_post_json",
        lambda payload: {
            "prediction": "MALICIOUS",
            "score": 0.9,
            "threshold": 0.5,
        },
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("attack")

    assert exc_info.value.result.mode == "block"


def test_public_positional_constructors_remain_compatible():
    result = BlockResult("BENIGN", 0.1, 0.5, OUTCOME_SECRET_EXPOSURE)
    event = ClassifyEvent(
        HookLabel.USER_INPUT,
        None,
        "hello",
        result,
        False,
        True,
    )

    assert result.primary_outcome == OUTCOME_SECRET_EXPOSURE
    assert result.mode == "block"
    assert event.mode == "shadow"
    assert event.shadow_mode is True


def test_classify_batch_wire_shape_and_block_error(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, mode="warn")
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(
            200,
            {
                "predictions": [
                    {"prediction": "MALICIOUS", "score": 0.8, "threshold": 0.5},
                    {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5},
                    {"prediction": "MALICIOUS", "score": 0.8, "threshold": 0.5},
                ]
            },
        )

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
    body = json.loads(calls[0]["data"])
    assert body["texts"] == ["first", "second", "third"]
    assert body["mode"] == "warn"
    assert all(result.mode == "block" for result in exc_info.value.results)
    assert "threshold" not in body
    assert body["hooks"] == ["user_input", "tool_response", "tool_response"]
    assert body["tool_names"] == ["chat", "read_file", None]
    assert [item["silmaril"]["input_index"] for item in body["metadata"]] == [0, 1, 2]
    assert all("chunk_index" not in item["silmaril"] for item in body["metadata"])
    assert all("chunk_count" not in item["silmaril"] for item in body["metadata"])


def test_classify_batch_serializes_metadata(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(
            200,
            {
                "predictions": [
                    {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5},
                    {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5},
                ]
            },
        )

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
                    "sdk_version": "0.6.0",
                    "request_id": "batch-req",
                    "input_index": 0,
                },
            },
            {
                "silmaril": {
                    "sdk_language": "python",
                    "sdk_version": "0.6.0",
                    "request_id": "batch-req",
                    "input_index": 1,
                }
            },
        ],
    }


def test_classify_batch_shadow_mode_returns_results(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "predictions": [
                    {
                        "prediction": "MALICIOUS",
                        "score": 0.8,
                        "threshold": 0.5,
                        "mode": "shadow",
                    }
                ]
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    results = fw.classify_batch(["first"], shadow_mode=True)

    assert results[0].prediction == "MALICIOUS"
    assert results[0].threshold == 0.5


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
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    text = "b" * 4001
    result = fw.classify(
        text,
        hook=HookLabel.TOOL_RESPONSE,
        tool_name="fetch_webpage",
        metadata={"conversationId": "conversation-123", "conversation_id": "inert"},
        request_id="event-uuid",
    )

    assert result.prediction == "BENIGN"
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
                "sdk_version": "0.6.0",
                "request_id": "event-uuid",
            },
        },
    }


def test_classify_requires_backend_prediction(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, {"score": 0.99, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(ValueError, match="missing required 'prediction' field"):
        fw.classify("missing prediction")


def test_optional_outcome_fields(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "mode": "shadow",
                "primary_outcome": OUTCOME_SECRET_EXPOSURE,
                "outcome_scores": {OUTCOME_SECRET_EXPOSURE: 0.8},
                "detector_scores": {OUTCOME_SECRET_EXPOSURE: 1.0},
                "detector_counts": {OUTCOME_SECRET_EXPOSURE: 2},
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("leak token")

    assert result.primary_outcome == OUTCOME_SECRET_EXPOSURE
    assert result.outcome_scores == {OUTCOME_SECRET_EXPOSURE: 0.8}
    assert result.detector_scores == {OUTCOME_SECRET_EXPOSURE: 1.0}
    assert result.detector_counts == {OUTCOME_SECRET_EXPOSURE: 2}


def test_rejects_unknown_outcome_fields(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    responses = iter(
        [
            {"prediction": "MALICIOUS", "score": 0.91, "threshold": 0.5, "primary_outcome": "unknown"},
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "outcome_scores": {"unknown": 0.8},
            },
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "detector_scores": {"unknown": 0.8},
            },
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "detector_counts": {"unknown": 1},
            },
        ]
    )

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, next(responses))

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(ValueError, match="invalid primary_outcome"):
        fw.classify("x")
    with pytest.raises(ValueError, match="invalid outcome_scores key"):
        fw.classify("x")
    with pytest.raises(ValueError, match="invalid detector_scores key"):
        fw.classify("x")
    with pytest.raises(ValueError, match="invalid detector_counts key"):
        fw.classify("x")


def test_retries_retryable_status(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    responses = [
        FakeResponse(429, "rate limited"),
        FakeResponse(503, "unavailable"),
        FakeResponse(200, {"prediction": "BENIGN", "score": 0.01, "threshold": 0.5}),
    ]
    sleeps: list[float] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(fw._session, "post", fake_post)
    monkeypatch.setattr("silmaril_security.sdk.firewall.time.sleep", sleeps.append)

    result = fw.classify("hello")

    assert result.prediction == "BENIGN"
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
