# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio

import pytest
from held import WAIT_SECONDS, AsyncHeldClassification, HeldClassification

from silmaril_security.sdk import (
    AsyncFirewall,
    ClassifyEvent,
    Firewall,
    FirewallBlockedException,
    HookLabel,
)
from silmaril_security.sdk._pending import RETRY_LIMIT
from silmaril_security.sdk.firewall import _retry_budget

URL = "https://api.test.invalid/classify"


def _firewall(monkeypatch, held: HeldClassification, **options) -> Firewall:
    fw = Firewall(api_key="sk", api_url=URL, **options)
    monkeypatch.setattr(fw, "_classify_raw", held)
    return fw


def _async_firewall(monkeypatch, held: AsyncHeldClassification, **options) -> AsyncFirewall:
    fw = AsyncFirewall(api_key="sk", api_url=URL, **options)
    monkeypatch.setattr(fw, "_classify_raw", held)
    return fw


def test_protect_runs_the_operation_while_its_input_is_classified(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held)

    def operation() -> str:
        assert held.started.wait(WAIT_SECONDS)
        held.release.set()
        return "answer"

    assert fw.protect(operation, text="prompt", hook=HookLabel.USER_INPUT) == "answer"
    assert held.hooks == [HookLabel.USER_INPUT]


def test_protect_withholds_the_result_of_a_blocked_input(monkeypatch):
    held = HeldClassification("MALICIOUS")
    held.release.set()
    fw = _firewall(monkeypatch, held)
    finished = []

    with pytest.raises(FirewallBlockedException) as blocked:
        fw.protect(lambda: finished.append("ran"), text="attack", hook=HookLabel.USER_INPUT)

    assert finished == ["ran"]
    assert blocked.value.hook == HookLabel.USER_INPUT


def test_protect_checks_extracted_output_before_returning_it(monkeypatch):
    held = HeldClassification("BENIGN", "MALICIOUS")
    held.release.set()
    fw = _firewall(monkeypatch, held)

    with pytest.raises(FirewallBlockedException) as blocked:
        fw.protect(lambda: "leaked secret", text="prompt", hook=HookLabel.USER_INPUT, output=str)

    assert blocked.value.hook == HookLabel.LLM_OUTPUT
    assert [text for text, _ in held.calls] == ["prompt", "leaked secret"]


def test_protect_in_shadow_returns_before_the_verdict(monkeypatch):
    held = HeldClassification("MALICIOUS")
    events: list[ClassifyEvent] = []
    fw = _firewall(monkeypatch, held, mode="shadow", on_classify=events.append)

    assert fw.protect(lambda: "answer", text="attack", hook=HookLabel.USER_INPUT) == "answer"
    assert events == []

    held.release.set()
    fw.flush()
    assert [event.blocked for event in events] == [True]


def test_background_classifications_for_one_conversation_are_sent_in_order(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held, mode="shadow")

    fw.protect(
        lambda: "answer",
        text="prompt",
        hook=HookLabel.USER_INPUT,
        metadata={"conversationId": "conversation-1"},
        output=str,
    )
    assert held.started.wait(WAIT_SECONDS)
    assert [text for text, _ in held.calls] == ["prompt"]

    held.release.set()
    fw.flush()
    assert [text for text, _ in held.calls] == ["prompt", "answer"]
    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]


def test_full_conversation_queue_drops_background_classifications(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held, mode="shadow", max_queued=1)
    metadata = {"conversationId": "conversation-1"}

    fw.protect(lambda: None, text="first", metadata=metadata)
    fw.protect(lambda: None, text="dropped", metadata=metadata)
    held.release.set()
    fw.flush()

    assert [text for text, _ in held.calls] == ["first"]


def test_background_classifications_retry_once():
    fw = Firewall(api_key="sk", api_url=URL)

    background = fw._pending.submit(lambda: _retry_budget(5), background=True)
    joined = fw._pending.submit(lambda: _retry_budget(5))

    assert (background.result(), joined.result()) == (1, 5)
    assert RETRY_LIMIT.get() is None


@pytest.mark.parametrize("option", ["max_concurrent", "max_queued"])
def test_limits_must_be_positive(option):
    with pytest.raises(ValueError, match=f"{option} must be positive"):
        Firewall(api_key="sk", api_url=URL, **{option: 0})
    with pytest.raises(ValueError, match=f"{option} must be positive"):
        AsyncFirewall(api_key="sk", api_url=URL, **{option: 0})


@pytest.mark.asyncio
async def test_async_protect_runs_the_operation_while_its_input_is_classified(monkeypatch):
    held = AsyncHeldClassification()
    fw = _async_firewall(monkeypatch, held)

    async def operation() -> str:
        await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
        held.release.set()
        return "answer"

    assert await fw.protect(operation, text="prompt", hook=HookLabel.USER_INPUT) == "answer"


@pytest.mark.asyncio
async def test_async_protect_cancels_the_operation_when_the_block_lands_first(monkeypatch):
    held = AsyncHeldClassification("MALICIOUS")
    held.release.set()
    fw = _async_firewall(monkeypatch, held)
    cancelled = asyncio.Event()

    async def operation() -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"

    with pytest.raises(FirewallBlockedException):
        await fw.protect(operation, text="attack", hook=HookLabel.USER_INPUT)

    await asyncio.wait_for(cancelled.wait(), WAIT_SECONDS)


@pytest.mark.asyncio
async def test_async_protect_in_shadow_returns_before_the_verdict_and_close_drains(monkeypatch):
    held = AsyncHeldClassification("MALICIOUS")
    events: list[ClassifyEvent] = []
    fw = _async_firewall(monkeypatch, held, mode="shadow", on_classify=events.append)

    async def operation() -> str:
        return "answer"

    assert await fw.protect(operation, text="attack", hook=HookLabel.USER_INPUT, output=str) == "answer"
    assert events == []

    held.release.set()
    await fw.aclose()
    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]
    assert len(events) == 2


@pytest.mark.asyncio
async def test_async_protect_propagates_an_operation_failure(monkeypatch):
    held = AsyncHeldClassification()
    held.release.set()
    fw = _async_firewall(monkeypatch, held)

    async def operation() -> str:
        raise LookupError("model unavailable")

    with pytest.raises(LookupError, match="model unavailable"):
        await fw.protect(operation, text="prompt", hook=HookLabel.USER_INPUT)
    await fw.flush()
