# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

from typing import Any

import pytest
import requests

from silmaril_security.sdk import (
    BatchPromptBlockedException,
    BlockResult,
    Firewall,
    HookLabel,
    PromptBlockedException,
    SilmarilApiError,
)

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


def test_constructor_validates_thresholds():
    with pytest.raises(ValueError, match="threshold"):
        Firewall(api_key="sk", api_url=TEST_API_URL, threshold=1.5)
    with pytest.raises(ValueError, match="hook_thresholds"):
        Firewall(
            api_key="sk",
            api_url=TEST_API_URL,
            hook_thresholds={HookLabel.USER_INPUT: -0.1},
        )


def test_classify_posts_wire_shape_and_returns_result(monkeypatch):
    fw = Firewall(api_key="sk-test", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.12})

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("hello", hook=HookLabel.USER_INPUT, tool_name="chat")

    assert result == BlockResult(prediction="BENIGN", score=0.12, threshold=0.5)
    assert fw._session.headers["x-api-key"] == "sk-test"
    assert fw._session.headers["content-type"] == "application/json"
    assert calls[0]["url"] == TEST_API_URL
    assert calls[0]["timeout"] == 10.0
    assert calls[0]["data"] == (
        '{"text": "hello", "threshold": 0.5, '
        '"hook": "user_input", "tool_name": "chat"}'
    )


def test_classify_enforces_by_default(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(200, {"prediction": "MALICIOUS", "score": 0.91})

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(PromptBlockedException) as exc_info:
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
        return FakeResponse(200, {"prediction": "MALICIOUS", "score": 0.91})

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
        return FakeResponse(200, {"prediction": "MALICIOUS", "score": 0.91})

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(PromptBlockedException):
        fw.classify("attack", shadow_mode=False)


def test_per_hook_threshold_is_sent_and_enforced(monkeypatch):
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        hook_thresholds={HookLabel.TOOL_RESPONSE: 0.95},
    )
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, {"prediction": "BENIGN", "score": 0.91})

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("tool output", hook=HookLabel.TOOL_RESPONSE)

    assert result.threshold == 0.95
    assert '"threshold": 0.95' in calls[0]["data"]


def test_classify_batch_wire_shape_and_block_error(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(
            200,
            {
                "predictions": [
                    {"prediction": "MALICIOUS", "score": 0.8},
                    {"prediction": "BENIGN", "score": 0.1},
                    {"prediction": "MALICIOUS", "score": 0.7},
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
    assert calls[0]["data"] == (
        '{"texts": ["first", "second", "third"], "threshold": 0.5, '
        '"hooks": ["user_input", "tool_response", "tool_response"], '
        '"tool_names": ["chat", "read_file", null]}'
    )


def test_classify_batch_shadow_mode_returns_results(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {"predictions": [{"prediction": "MALICIOUS", "score": 0.8}]},
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
    with pytest.raises(ValueError, match="texts must not be empty"):
        fw.classify_batch([])


def test_optional_outcome_fields(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, shadow_mode=True)

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.91,
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
        FakeResponse(200, {"prediction": "BENIGN", "score": 0.01}),
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


def test_explain_wire_shape(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(
            200,
            {
                "tokens": ["hello"],
                "attributions": [0.1],
                "score": 0.2,
                "prediction": "BENIGN",
                "prepared_text": "hello",
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.explain("hello", hook=HookLabel.USER_INPUT, steps=100, temperature=1.0)

    assert result.prediction == "BENIGN"
    assert calls[0]["data"] == (
        '{"explain": true, "text": "hello", "hook": "user_input", '
        '"steps": 100, "temperature": 1.0}'
    )
