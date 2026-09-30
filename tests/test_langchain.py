# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

from uuid import uuid4

import pytest

from silmaril_security.sdk import (
    BlockResult,
    ClassifyEvent,
    Firewall,
    FirewallBlockedException,
    GovernanceDecision,
    HookLabel,
    SilmarilApiError,
)

pytest.importorskip("langchain_core.callbacks")


def test_langchain_handlers_reject_invalid_mode_before_classification():
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")

    with pytest.raises(ValueError, match="mode must be shadow, warn, or block"):
        fw.as_langchain_handler(mode="audit")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode must be shadow, warn, or block"):
        fw.as_async_langchain_handler(mode="audit")  # type: ignore[arg-type]


def test_langchain_handler_blocks_last_user_message(monkeypatch):
    events: list[ClassifyEvent] = []
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(on_classify=events.append)
    calls = []

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None, metadata=None):
        calls.append((text, hook, tool_name, request_id, metadata))
        return BlockResult(
            prediction="MALICIOUS",
            score=0.9,
            threshold=0.5,
            mode="block",
        )

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)

    run_id = uuid4()
    with pytest.raises(FirewallBlockedException):
        handler.on_chat_model_start(
            serialized={},
            messages=[
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "second"},
                ]
            ],
            run_id=run_id,
        )

    assert len(calls) == 1
    assert calls[0][:3] == ("second", HookLabel.USER_INPUT, None)
    assert calls[0][3] != str(run_id)
    assert calls[0][4] == {"langgraph": {"run_id": str(run_id)}}
    assert len(events) == 1
    assert events[0].blocked is True


def test_langchain_tool_hooks_have_distinct_request_ids(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    calls = []

    def raw(text, **kwargs):
        calls.append((text, kwargs))
        return BlockResult(prediction="BENIGN", score=0.1, threshold=0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    handler = fw.as_langchain_handler(conversation_id="conversation-1")
    run_id = uuid4()
    handler.on_tool_start({"name": "search"}, '{"query":"safe"}', run_id=run_id)
    handler.on_tool_end("result", run_id=run_id, name="search")
    assert [call[1]["hook"] for call in calls] == [HookLabel.TOOL_CALL, HookLabel.TOOL_RESPONSE]
    assert calls[0][1]["request_id"] != calls[1][1]["request_id"]
    assert all(call[1]["metadata"] == {
        "langgraph": {"run_id": str(run_id)}, "conversationId": "conversation-1",
    } for call in calls)


def test_langchain_explicit_governance_block(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    monkeypatch.setattr(
        fw, "_classify_raw",
        lambda text, **kwargs: BlockResult(
            prediction="BENIGN", score=0.1, threshold=0.5, mode="block",
            governance=GovernanceDecision(action="block", policy_version="v2"),
        ),
    )
    with pytest.raises(FirewallBlockedException):
        fw.as_langchain_handler().on_tool_start(
            {"name": "search"}, "safe", run_id=uuid4(),
        )


def test_langchain_handler_fail_open(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler()

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None, metadata=None):
        raise SilmarilApiError(status=500, status_text="Internal Server Error", body="boom")

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)

    handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "hello"}]],
        run_id=uuid4(),
    )


def test_langchain_requested_warn_survives_legacy_mode_less_response(monkeypatch):
    fw = Firewall(
        api_key="sk",
        api_url="https://api.test.invalid/classify",
        mode="warn",
    )
    handler = fw.as_langchain_handler()

    monkeypatch.setattr(
        fw,
        "_post_json",
        lambda payload: {
            "prediction": "MALICIOUS",
            "score": 0.9,
            "threshold": 0.5,
        },
    )

    handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "attack"}]],
        run_id=uuid4(),
    )


def test_langchain_effective_warn_preserves_flow(monkeypatch):
    events: list[ClassifyEvent] = []
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(on_classify=events.append)

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None, metadata=None):
        return BlockResult(
            prediction="MALICIOUS",
            score=0.9,
            threshold=0.5,
            mode="warn",
        )

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)

    handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "hello"}]],
        run_id=uuid4(),
    )

    assert events[0].mode == "warn"
    assert events[0].blocked is True


def test_langchain_handler_fail_closed(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(fail_open=False)

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None, metadata=None):
        raise SilmarilApiError(status=500, status_text="Internal Server Error", body="boom")

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)

    with pytest.raises(SilmarilApiError):
        handler.on_chat_model_start(
            serialized={},
            messages=[[{"role": "user", "content": "hello"}]],
            run_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_async_langchain_handler_supports_async_callback(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    events: list[ClassifyEvent] = []

    async def on_classify(event: ClassifyEvent) -> None:
        events.append(event)

    handler = fw.as_async_langchain_handler(on_classify=on_classify, shadow_mode=True)

    async def fake_async_raw(
        firewall,
        text,
        *,
        hook=None,
        tool_name=None,
        request_id=None,
        mode=None,
        metadata=None,
    ):
        return BlockResult(
            prediction="MALICIOUS",
            score=0.9,
            threshold=0.5,
            mode=mode or "block",
        )

    monkeypatch.setattr("silmaril_security.sdk.langchain._async_classify_raw", fake_async_raw)

    await handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "hello"}]],
        run_id=uuid4(),
    )

    assert len(events) == 1
    assert events[0].blocked is True
    assert events[0].shadow_mode is True


@pytest.mark.asyncio
async def test_async_classify_raw_sends_long_event_once(monkeypatch):
    from silmaril_security.sdk.langchain import _async_classify_raw

    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    payloads = []

    async def fake_post_json(self, payload):
        payloads.append(payload)
        return {
            "prediction": "BENIGN",
            "score": 0.1,
            "threshold": 0.5,
            "mode": "block",
        }

    monkeypatch.setattr("silmaril_security.sdk.async_firewall.AsyncFirewall._post_json", fake_post_json)

    result = await _async_classify_raw(
        fw,
        "a" * 4001,
        hook=HookLabel.USER_INPUT,
        tool_name="chat",
        metadata={"langgraph": {"run_id": "async-run"}},
        request_id="async-req",
    )

    assert result.score == 0.1
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["text"] == "a" * 4001
    assert payload["hook"] == "user_input"
    assert payload["tool_name"] == "chat"
    assert payload["metadata"]["langgraph"] == {"run_id": "async-run"}
    assert payload["metadata"]["silmaril"] == {
        "sdk_language": "python",
        "sdk_version": "0.7.0",
        "request_id": "async-req",
    }
    assert "threshold" not in payload


@pytest.mark.asyncio
async def test_async_langchain_requested_warn_survives_legacy_mode_less_response(monkeypatch):
    fw = Firewall(
        api_key="sk",
        api_url="https://api.test.invalid/classify",
        mode="warn",
    )
    handler = fw.as_async_langchain_handler()

    async def fake_post_json(self, payload):
        return {
            "prediction": "MALICIOUS",
            "score": 0.9,
            "threshold": 0.5,
        }

    monkeypatch.setattr("silmaril_security.sdk.async_firewall.AsyncFirewall._post_json", fake_post_json)

    await handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "attack"}]],
        run_id=uuid4(),
    )
