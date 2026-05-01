# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

from uuid import uuid4

import pytest

from silmaril_security.sdk import (
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
