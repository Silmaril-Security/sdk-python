# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest
import requests

from silmaril_security.sdk import (
    CHUNK_WINDOW_CHARS,
    DEFAULT_CHUNK_CONCURRENCY,
    SERVER_SINGLE_TEXT_MAX_CHARS,
    BatchFirewallBlockedException,
    BatchPromptBlockedException,
    BlockResult,
    Firewall,
    FirewallBlockedException,
    HookLabel,
    PromptBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk.firewall import _MAX_ERROR_BODY_BYTES

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
        return self._body


def test_constructor_requires_key_and_url():
    with pytest.raises(ValueError, match="api_key is required"):
        Firewall(api_key="", api_url=TEST_API_URL)
    with pytest.raises(ValueError, match="api_url is required"):
        Firewall(api_key="sk", api_url="")


def test_deprecated_exception_names_alias_new_names():
    assert PromptBlockedException is FirewallBlockedException
    assert BatchPromptBlockedException is BatchFirewallBlockedException


def test_constructor_validates_chunk_concurrency():
    with pytest.raises(ValueError, match="chunk_concurrency must be >= 1"):
        Firewall(api_key="sk", api_url=TEST_API_URL, chunk_concurrency=0)


def test_constructor_applies_default_chunk_concurrency():
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    assert fw.chunk_concurrency == DEFAULT_CHUNK_CONCURRENCY


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

    assert result == BlockResult(prediction="BENIGN", score=0.12, threshold=0.5)
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
                "sdk_version": "0.4.1",
                "request_id": "req-test",
                "input_index": 0,
                "chunk_index": 0,
                "chunk_count": 1,
            }
        },
    }


def test_classify_accepts_null_score_when_backend_prediction_is_present(monkeypatch):
    fw = Firewall(api_key="sk-test", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "prediction": "BENIGN",
                "score": None,
                "threshold": 0.9,
                "primary_outcome": "benign",
                "outcome_scores": {"secret_exposure": None},
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("hello")

    assert result.prediction == "BENIGN"
    assert result.score == 0.0
    assert result.threshold == 0.9
    assert result.primary_outcome == "benign"
    assert result.outcome_scores == {}


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
                "sdk_version": "0.4.1",
                "request_id": "req-meta",
                "input_index": 0,
                "chunk_index": 0,
                "chunk_count": 1,
            },
        },
    }


def test_classify_enforces_by_default(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, {"prediction": "MALICIOUS", "score": 0.91, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("ignore previous", hook=HookLabel.USER_INPUT, tool_name="chat")

    assert exc_info.value.score == 0.91
    assert exc_info.value.threshold == 0.5
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
        return FakeResponse(200, {"prediction": "MALICIOUS", "score": 0.91, "threshold": 0.5})

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


def test_classify_batch_wire_shape_and_block_error(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
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
    assert "threshold" not in body
    assert body["hooks"] == ["user_input", "tool_response", "tool_response"]
    assert body["tool_names"] == ["chat", "read_file", None]
    assert [item["silmaril"]["input_index"] for item in body["metadata"]] == [0, 1, 2]
    assert [item["silmaril"]["chunk_index"] for item in body["metadata"]] == [0, 0, 0]
    assert [item["silmaril"]["chunk_count"] for item in body["metadata"]] == [1, 1, 1]


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
                    "sdk_version": "0.4.1",
                    "request_id": "batch-req",
                    "input_index": 0,
                    "chunk_index": 0,
                    "chunk_count": 1,
                },
            },
            {
                "silmaril": {
                    "sdk_language": "python",
                    "sdk_version": "0.4.1",
                    "request_id": "batch-req",
                    "input_index": 1,
                    "chunk_index": 0,
                    "chunk_count": 1,
                }
            },
        ],
    }


def test_classify_batch_shadow_mode_returns_results(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {"predictions": [{"prediction": "MALICIOUS", "score": 0.8, "threshold": 0.5}]},
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


def test_classify_fans_out_long_input_chunks_and_picks_max_score(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)
    calls: list[dict[str, Any]] = []
    lock = threading.Lock()

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        body = json.loads(kwargs["data"])
        chunk_index = body["metadata"]["silmaril"]["chunk_index"]
        with lock:
            calls.append({"url": url, **kwargs})
        score = 0.95 if chunk_index == 1 else 0.1
        prediction = "MALICIOUS" if score >= 0.5 else "BENIGN"
        return FakeResponse(200, {"prediction": prediction, "score": score, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify(
        "a" * (SERVER_SINGLE_TEXT_MAX_CHARS + CHUNK_WINDOW_CHARS),
        hook=HookLabel.USER_INPUT,
        request_id="chunk-req",
    )

    assert result.prediction == "MALICIOUS"
    assert result.score == 0.95
    assert len(calls) > 1
    bodies = [json.loads(call["data"]) for call in calls]
    assert sorted(body["metadata"]["silmaril"]["chunk_index"] for body in bodies) == list(
        range(len(calls))
    )
    for body in bodies:
        index = body["metadata"]["silmaril"]["chunk_index"]
        assert "text" in body
        assert "texts" not in body
        assert body["hook"] == "user_input"
        assert "threshold" not in body
        assert body["metadata"]["silmaril"] == {
            "sdk_language": "python",
            "sdk_version": "0.4.1",
            "request_id": "chunk-req",
            "input_index": 0,
            "chunk_index": index,
            "chunk_count": len(calls),
        }


def test_classify_sends_normal_size_long_input_as_single_request(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    fw.classify(
        "a" * (CHUNK_WINDOW_CHARS * 3),
        hook=HookLabel.TOOL_RESPONSE,
        request_id="server-window-req",
    )

    assert len(calls) == 1
    body = json.loads(calls[0]["data"])
    assert body["text"] == "a" * (CHUNK_WINDOW_CHARS * 3)
    assert body["metadata"]["silmaril"] == {
        "sdk_language": "python",
        "sdk_version": "0.4.1",
        "request_id": "server-window-req",
        "input_index": 0,
        "chunk_index": 0,
        "chunk_count": 1,
    }


def test_classify_fanout_propagates_tool_name_to_every_chunk(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)
    calls: list[dict[str, Any]] = []
    lock = threading.Lock()

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        with lock:
            calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    fw.classify(
        "b" * (SERVER_SINGLE_TEXT_MAX_CHARS + CHUNK_WINDOW_CHARS),
        hook=HookLabel.TOOL_RESPONSE,
        tool_name="fetch_webpage",
        metadata={"langgraph": {"run_id": "run-chunked"}},
        request_id="chunk-meta",
    )

    assert len(calls) > 1
    for call in calls:
        body = json.loads(call["data"])
        assert body["hook"] == "tool_response"
        assert body["tool_name"] == "fetch_webpage"
        assert body["metadata"]["langgraph"] == {"run_id": "run-chunked"}
        assert body["metadata"]["silmaril"]["request_id"] == "chunk-meta"
        assert body["metadata"]["silmaril"]["chunk_count"] == len(calls)
        assert "texts" not in body


def test_classify_chunk_concurrency_limit(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, chunk_concurrency=2, shadow_mode=True)
    active = 0
    max_active = 0
    calls = 0
    lock = threading.Lock()

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        nonlocal active, max_active, calls
        with lock:
            active += 1
            calls += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    fw.classify("c" * (SERVER_SINGLE_TEXT_MAX_CHARS + CHUNK_WINDOW_CHARS * 5))

    assert calls > 2
    assert max_active <= 2


def test_classify_long_input_propagates_chunk_error(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)
    calls = 0
    lock = threading.Lock()

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        with lock:
            calls += 1
            current = calls
        if current == 1:
            return FakeResponse(400, "boom", reason="Bad Request")
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5})

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(SilmarilApiError) as exc_info:
        fw.classify("d" * (SERVER_SINGLE_TEXT_MAX_CHARS + CHUNK_WINDOW_CHARS))

    assert exc_info.value.status == 400
    assert calls > 1


def test_optional_outcome_fields(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
                "threshold": 0.5,
                "primary_outcome": "secret_exposure",
                "outcome_scores": {"secret_exposure": 0.8},
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("leak token")

    assert result.primary_outcome == "secret_exposure"
    assert result.outcome_scores == {"secret_exposure": 0.8}


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
