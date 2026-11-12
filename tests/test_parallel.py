# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
import threading

import pytest
from held import WAIT_SECONDS, AsyncHeldClassification, HeldClassification, verdict

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


def test_block_mode_output_check_waits_its_turn_in_the_conversation_queue(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held, mode="shadow")
    metadata = {"conversationId": "conversation-1"}
    fw.protect(lambda: None, text="earlier event", metadata=metadata)
    assert held.started.wait(WAIT_SECONDS)
    finished = threading.Event()

    def protected() -> None:
        fw.protect(
            lambda: "answer", text="prompt", metadata=metadata, mode="block", output=str
        )
        finished.set()

    caller = threading.Thread(target=protected)
    caller.start()
    assert [text for text, _ in held.calls] == ["earlier event"]
    assert not finished.is_set()

    held.release.set()
    caller.join(WAIT_SECONDS)
    assert finished.is_set()
    assert [text for text, _ in held.calls] == ["earlier event", "prompt", "answer"]


def test_queued_classification_keeps_the_metadata_it_was_started_with(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held, mode="shadow")
    metadata = {"conversationId": "conversation-1", "tenant": {"plan": "trial"}}

    fw.protect(lambda: None, text="first", metadata=metadata)
    assert held.started.wait(WAIT_SECONDS)
    fw.protect(lambda: None, text="second", metadata=metadata)
    metadata["conversationId"] = "conversation-2"
    metadata["tenant"]["plan"] = "enterprise"
    held.release.set()
    fw.flush()

    assert [options["metadata"] for _, options in held.calls] == [
        {"conversationId": "conversation-1", "tenant": {"plan": "trial"}}
    ] * 2


def test_background_classifications_without_a_conversation_share_a_total_limit(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held, mode="shadow", max_concurrent=1, max_queued=2)

    for text in ("first", "second", "dropped"):
        fw.protect(lambda: None, text=text)
    held.release.set()
    fw.flush()

    assert [text for text, _ in held.calls] == ["first", "second"]


def test_failed_operation_withdraws_a_verdict_that_was_not_sent(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held, mode="shadow")
    metadata = {"conversationId": "conversation-1"}
    fw.protect(lambda: None, text="earlier event", metadata=metadata)
    assert held.started.wait(WAIT_SECONDS)

    def operation() -> None:
        raise LookupError("model unavailable")

    with pytest.raises(LookupError, match="model unavailable"):
        fw.protect(operation, text="never sent", metadata=metadata, mode="block")
    fw.protect(lambda: None, text="later event", metadata=metadata)
    assert [text for text, _ in held.calls] == ["earlier event"]

    held.release.set()
    fw.flush()
    assert [text for text, _ in held.calls] == ["earlier event", "later event"]


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
        finally:
            await asyncio.sleep(0)
            cancelled.set()
        return "unreachable"

    with pytest.raises(FirewallBlockedException):
        await fw.protect(operation, text="attack", hook=HookLabel.USER_INPUT)

    assert cancelled.is_set()


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


@pytest.mark.asyncio
async def test_async_block_mode_output_check_goes_through_the_queue(monkeypatch):
    held = AsyncHeldClassification()
    fw = _async_firewall(monkeypatch, held, max_concurrent=1)
    metadata = {"conversationId": "conversation-1"}
    earlier = fw._pending.submit(lambda: fw.classify("earlier event", metadata=metadata), "conversation-1")

    async def operation() -> str:
        return "answer"

    protected = asyncio.ensure_future(
        fw.protect(operation, text="prompt", metadata=metadata, output=str)
    )
    await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
    assert [text for text, _ in held.calls] == ["earlier event"]

    held.release.set()
    assert await protected == "answer"
    await earlier
    assert [text for text, _ in held.calls] == ["earlier event", "prompt", "answer"]


@pytest.mark.asyncio
async def test_cancelled_async_protect_leaves_nothing_queued(monkeypatch):
    held = AsyncHeldClassification()
    fw = _async_firewall(monkeypatch, held)
    metadata = {"conversationId": "conversation-1"}
    earlier = fw._pending.submit(lambda: fw.classify("earlier event", metadata=metadata), "conversation-1")
    await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
    operating = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation() -> None:
        operating.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    protected = asyncio.ensure_future(fw.protect(operation, text="never sent", metadata=metadata))
    await asyncio.wait_for(operating.wait(), WAIT_SECONDS)
    protected.cancel()
    with pytest.raises(asyncio.CancelledError):
        await protected
    assert cancelled.is_set()
    later = fw._pending.submit(lambda: fw.classify("later event", metadata=metadata), "conversation-1")
    for _ in range(5):
        await asyncio.sleep(0)
    assert [text for text, _ in held.calls] == ["earlier event"]

    held.release.set()
    await earlier
    await later
    await fw.flush()
    assert [text for text, _ in held.calls] == ["earlier event", "later event"]
    assert fw._pending._outstanding == set()


@pytest.mark.asyncio
async def test_async_queued_classification_keeps_the_metadata_it_was_started_with(monkeypatch):
    held = AsyncHeldClassification()
    fw = _async_firewall(monkeypatch, held, mode="shadow")
    metadata = {"conversationId": "conversation-1", "tenant": {"plan": "trial"}}

    async def operation() -> None:
        return None

    await fw.protect(operation, text="first", metadata=metadata)
    await fw.protect(operation, text="second", metadata=metadata)
    metadata["tenant"]["plan"] = "enterprise"
    held.release.set()
    await fw.flush()

    assert [options["metadata"]["tenant"] for _, options in held.calls] == [{"plan": "trial"}] * 2


@pytest.mark.asyncio
async def test_cancelled_close_still_closes_the_pool_after_background_work(monkeypatch):
    held = AsyncHeldClassification()
    fw = _async_firewall(monkeypatch, held, mode="shadow")
    client = fw._ensure_client()

    async def operation() -> None:
        return None

    await fw.protect(operation, text="background")
    closing = asyncio.ensure_future(fw.aclose())
    await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    held.release.set()
    await fw.aclose()
    assert [text for text, _ in held.calls] == ["background"]
    assert client.is_closed


@pytest.mark.asyncio
async def test_closing_from_a_callback_does_not_wait_for_work_queued_behind_it(monkeypatch):
    held = AsyncHeldClassification()
    held.release.set()
    closed = asyncio.Event()

    async def close_on_first_event(event: ClassifyEvent) -> None:
        if event.hook == HookLabel.USER_INPUT:
            await fw.aclose()
            closed.set()

    fw = _async_firewall(monkeypatch, held, mode="shadow", on_classify=close_on_first_event)

    async def operation() -> str:
        return "answer"

    await fw.protect(
        operation,
        text="prompt",
        hook=HookLabel.USER_INPUT,
        metadata={"conversationId": "conversation-1"},
        output=str,
    )

    await asyncio.wait_for(closed.wait(), WAIT_SECONDS)
    with pytest.raises(RuntimeError, match="closed"):
        fw._ensure_client()


def test_concurrent_conversations_keep_their_order_within_the_in_flight_limit(monkeypatch):
    fw = Firewall(api_key="sk", api_url=URL, mode="shadow", max_concurrent=2, max_queued=64)
    lock = threading.Lock()
    sent: dict[str, list[str]] = {}
    in_flight = 0
    peak = 0
    overlap = threading.Barrier(2, timeout=WAIT_SECONDS)

    def classify(text, **options):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
            sent.setdefault(options["metadata"]["conversationId"], []).append(text)
        if text == "0:prompt":
            overlap.wait()
        with lock:
            in_flight -= 1
        return verdict(mode=options.get("mode"))

    monkeypatch.setattr(fw, "_classify_raw", classify)

    def conversation(name: str) -> None:
        for turn in range(5):
            fw.protect(
                lambda: f"{turn}:answer",
                text=f"{turn}:prompt",
                metadata={"conversationId": name},
                output=str,
            )

    callers = [threading.Thread(target=conversation, args=(f"conversation-{i}",)) for i in range(8)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(WAIT_SECONDS)
    fw.flush()

    expected = [f"{turn}:{part}" for turn in range(5) for part in ("prompt", "answer")]
    assert sent == {f"conversation-{i}": expected for i in range(8)}
    assert peak == 2


@pytest.mark.asyncio
async def test_concurrent_async_conversations_keep_their_order_within_the_in_flight_limit(monkeypatch):
    fw = AsyncFirewall(api_key="sk", api_url=URL, mode="shadow", max_concurrent=2)
    sent: dict[str, list[str]] = {}
    in_flight = 0
    peak = 0

    async def classify(text, **options):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        sent.setdefault(options["metadata"]["conversationId"], []).append(text)
        await asyncio.sleep(0)
        in_flight -= 1
        return verdict(mode=options.get("mode"))

    monkeypatch.setattr(fw, "_classify_raw", classify)

    async def conversation(name: str) -> None:
        for turn in range(5):
            async def operation(turn: int = turn) -> str:
                return f"{turn}:answer"

            await fw.protect(
                operation, text=f"{turn}:prompt", metadata={"conversationId": name}, output=str
            )

    await asyncio.gather(*(conversation(f"conversation-{i}") for i in range(8)))
    await fw.flush()

    expected = [f"{turn}:{part}" for turn in range(5) for part in ("prompt", "answer")]
    assert sent == {f"conversation-{i}": expected for i in range(8)}
    assert peak == 2


@pytest.mark.asyncio
async def test_close_from_a_callback_waits_for_independent_work_only(monkeypatch):
    releases = {"prompt": asyncio.Event(), "independent": asyncio.Event(), "answer": asyncio.Event()}
    started = {text: asyncio.Event() for text in releases}
    releases["prompt"].set()
    closed = asyncio.Event()

    async def classify(text, **options):
        started[text].set()
        await asyncio.wait_for(releases[text].wait(), WAIT_SECONDS)
        return verdict(mode=options.get("mode"))

    async def close_on_prompt(event: ClassifyEvent) -> None:
        if event.text == "prompt":
            await fw.aclose()
            closed.set()

    fw = AsyncFirewall(api_key="sk", api_url=URL, mode="shadow", on_classify=close_on_prompt)
    monkeypatch.setattr(fw, "_classify_raw", classify)

    async def operation() -> str:
        return "answer"

    await fw.protect(operation, text="independent", metadata={"conversationId": "conversation-2"})
    await asyncio.wait_for(started["independent"].wait(), WAIT_SECONDS)
    await fw.protect(
        operation, text="prompt", metadata={"conversationId": "conversation-1"}, output=str
    )
    await asyncio.wait_for(started["prompt"].wait(), WAIT_SECONDS)
    for _ in range(10):
        await asyncio.sleep(0)
    assert not closed.is_set()
    assert not started["answer"].is_set()

    releases["independent"].set()
    await asyncio.wait_for(closed.wait(), WAIT_SECONDS)
    releases["answer"].set()
    with pytest.raises(RuntimeError, match="closed"):
        fw._ensure_client()


@pytest.mark.asyncio
async def test_cancelled_flush_leaves_background_work_running_for_the_next_flush(monkeypatch):
    held = AsyncHeldClassification()
    fw = _async_firewall(monkeypatch, held, mode="shadow")

    async def operation() -> str:
        return "answer"

    await asyncio.gather(*(
        fw.protect(operation, text=f"prompt-{i}", metadata={"conversationId": f"conversation-{i}"}, output=str)
        for i in range(4)
    ))
    flushing = asyncio.ensure_future(fw.flush())
    await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
    flushing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flushing

    held.release.set()
    await fw.flush()
    assert sorted(text for text, _ in held.calls) == sorted(
        [f"prompt-{i}" for i in range(4)] + ["answer"] * 4
    )
    assert fw._pending._outstanding == set()


@pytest.mark.asyncio
async def test_callback_close_during_a_close_started_elsewhere_does_not_deadlock(monkeypatch):
    held = AsyncHeldClassification()
    closed_from_callback = asyncio.Event()

    async def close_from_callback(event: ClassifyEvent) -> None:
        await fw.aclose()
        closed_from_callback.set()

    fw = _async_firewall(monkeypatch, held, mode="shadow", on_classify=close_from_callback)
    client = fw._ensure_client()

    async def operation() -> None:
        return None

    await fw.protect(operation, text="background", metadata={"conversationId": "conversation-1"})
    await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
    outside = asyncio.ensure_future(fw.aclose())
    for _ in range(5):
        await asyncio.sleep(0)
    assert not outside.done()

    held.release.set()
    await asyncio.wait_for(outside, WAIT_SECONDS)
    await asyncio.wait_for(closed_from_callback.wait(), WAIT_SECONDS)
    assert client.is_closed


@pytest.mark.asyncio
async def test_flush_from_a_callback_does_not_hide_its_work_from_other_flushes(monkeypatch):
    held = AsyncHeldClassification()
    held.release.set()
    flushed_inside = asyncio.Event()
    finish_callback = asyncio.Event()

    async def flush_then_keep_working(event: ClassifyEvent) -> None:
        if event.hook == HookLabel.USER_INPUT:
            await fw.flush()
            flushed_inside.set()
            await asyncio.wait_for(finish_callback.wait(), WAIT_SECONDS)

    fw = _async_firewall(monkeypatch, held, mode="shadow", on_classify=flush_then_keep_working)

    async def operation() -> str:
        return "answer"

    await fw.protect(
        operation,
        text="prompt",
        hook=HookLabel.USER_INPUT,
        metadata={"conversationId": "conversation-1"},
        output=str,
    )
    await asyncio.wait_for(flushed_inside.wait(), WAIT_SECONDS)
    outside = asyncio.ensure_future(fw.flush())
    for _ in range(5):
        await asyncio.sleep(0)
    assert not outside.done()
    assert [text for text, _ in held.calls] == ["prompt"]

    finish_callback.set()
    await asyncio.wait_for(outside, WAIT_SECONDS)
    assert [text for text, _ in held.calls] == ["prompt", "answer"]
