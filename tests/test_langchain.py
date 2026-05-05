# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

from uuid import uuid4

import pytest

from silmaril_security.sdk import (
    CHUNK_WINDOW_CHARS,
    BlockResult,
    ClassifyEvent,
    Firewall,
    HookLabel,
    PromptBlockedException,
    SilmarilApiError,
)

pytest.importorskip("langchain_core.callbacks")


def test_langchain_handler_blocks_last_user_message(monkeypatch):
    events: list[ClassifyEvent] = []
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(on_classify=events.append)
    calls = []

    def fake_raw(text, *, hook=None, tool_name=None, threshold=None):
        calls.append((text, hook, tool_name, threshold))
        return BlockResult(prediction="MALICIOUS", score=0.9, threshold=threshold)

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)

    with pytest.raises(PromptBlockedException):
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
            run_id=uuid4(),
        )

    assert calls == [("second", HookLabel.USER_INPUT, None, 0.5)]
    assert len(events) == 1
    assert events[0].blocked is True


def test_langchain_handler_fail_open(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler()

    def fake_raw(text, *, hook=None, tool_name=None, threshold=None):
        raise SilmarilApiError(status=500, status_text="Internal Server Error", body="boom")

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)

    handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "hello"}]],
        run_id=uuid4(),
    )


def test_langchain_handler_fail_closed(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(fail_open=False)

    def fake_raw(text, *, hook=None, tool_name=None, threshold=None):
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

    async def fake_async_raw(firewall, text, *, hook=None, tool_name=None, threshold=None):
        return BlockResult(prediction="MALICIOUS", score=0.9, threshold=threshold)

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
async def test_async_classify_raw_fans_out_long_input_chunks(monkeypatch):
    from silmaril_security.sdk.langchain import _async_classify_raw

    fw = Firewall(
        api_key="sk",
        api_url="https://api.test.invalid/classify",
        chunk_concurrency=2,
    )
    payloads = []
    active = 0
    max_active = 0

    async def fake_post_json(client, firewall, payload):
        nonlocal active, max_active
        payloads.append(payload)
        call_index = len(payloads)
        active += 1
        max_active = max(max_active, active)
        try:
            import asyncio

            await asyncio.sleep(0)
            score = 0.95 if call_index == 2 else 0.1
            return {
                "prediction": "MALICIOUS" if score >= 0.5 else "BENIGN",
                "score": score,
            }
        finally:
            active -= 1

    monkeypatch.setattr("silmaril_security.sdk.langchain._async_post_json", fake_post_json)

    result = await _async_classify_raw(
        fw,
        "a" * (CHUNK_WINDOW_CHARS * 3),
        hook=HookLabel.USER_INPUT,
        tool_name="chat",
        threshold=0.5,
    )

    assert result.score == 0.95
    assert len(payloads) > 1
    assert max_active <= 2
    for payload in payloads:
        assert "text" in payload
        assert "texts" not in payload
        assert payload["hook"] == "user_input"
        assert payload["tool_name"] == "chat"
