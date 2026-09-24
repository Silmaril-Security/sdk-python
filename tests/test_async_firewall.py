# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
import builtins
import json
from typing import Any
from uuid import uuid4

import httpx
import pytest

from silmaril_security.sdk import (
    AsyncFirewall,
    BatchFirewallBlockedException,
    ClassifyEvent,
    FirewallBlockedException,
    HookLabel,
    SilmarilApiError,
)
from silmaril_security.sdk.firewall import _MAX_ERROR_BODY_BYTES

TEST_API_URL = "https://api.test.invalid/classify"


def response(
    request: httpx.Request,
    *,
    prediction: str = "BENIGN",
    score: float = 0.1,
    mode: str = "block",
    status: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status,
        request=request,
        json={
            "prediction": prediction,
            "score": score,
            "threshold": 0.5,
            "mode": mode,
        },
    )


def test_constructor_reports_async_extra_when_httpx_is_unavailable(monkeypatch):
    real_import = builtins.__import__

    def import_without_httpx(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_httpx)
    with pytest.raises(ImportError, match=r"silmaril-security-sdk\[async\]"):
        AsyncFirewall(api_key="sk", api_url=TEST_API_URL)


@pytest.mark.asyncio
async def test_concurrent_classify_calls_overlap_and_keep_state_independent():
    active = 0
    max_active = 0
    both_started = asyncio.Event()
    release = asyncio.Event()
    requests: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        payload = json.loads(request.content)
        requests.append(payload)
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            both_started.set()
        await release.wait()
        active -= 1
        score = 0.1 if payload["text"] == "first" else 0.2
        return response(request, score=score, mode=payload["mode"])

    metadata_a = {"application": {"request": "a"}}
    metadata_b = {"application": {"request": "b"}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, http_client=client)
        first = asyncio.create_task(
            fw.classify(
                "first",
                hook=HookLabel.USER_INPUT,
                metadata=metadata_a,
                mode="warn",
                request_id="req-a",
            )
        )
        second = asyncio.create_task(
            fw.classify(
                "second",
                hook=HookLabel.TOOL_RESPONSE,
                metadata=metadata_b,
                mode="shadow",
                request_id="req-b",
            )
        )
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        await fw.aclose()

        assert client.is_closed is False

    assert max_active == 2
    assert (first_result.score, first_result.mode) == (0.1, "warn")
    assert (second_result.score, second_result.mode) == (0.2, "shadow")
    by_text = {item["text"]: item for item in requests}
    assert by_text["first"]["metadata"]["silmaril"]["request_id"] == "req-a"
    assert by_text["second"]["metadata"]["silmaril"]["request_id"] == "req-b"
    assert by_text["first"]["hook"] == "user_input"
    assert by_text["second"]["hook"] == "tool_response"
    assert metadata_a == {"application": {"request": "a"}}
    assert metadata_b == {"application": {"request": "b"}}


@pytest.mark.asyncio
async def test_batch_is_one_ordered_request_and_preserves_block_details():
    payloads: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "predictions": [
                    {
                        "prediction": "MALICIOUS",
                        "score": 0.9,
                        "threshold": 0.5,
                        "mode": "block",
                    },
                    {
                        "prediction": "BENIGN",
                        "score": 0.1,
                        "threshold": 0.5,
                        "mode": "block",
                    },
                ]
            },
        )

    caller_metadata = [{"trace": "one"}, {"trace": "two"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, http_client=client)
        with pytest.raises(BatchFirewallBlockedException) as exc_info:
            await fw.classify_batch(
                ["attack", "safe"],
                hooks=[HookLabel.USER_INPUT, HookLabel.TOOL_RESPONSE],
                tool_names=["chat", "tool"],
                metadata=caller_metadata,
                request_id="batch-id",
            )

    assert len(payloads) == 1
    assert payloads[0]["texts"] == ["attack", "safe"]
    assert [item["silmaril"]["input_index"] for item in payloads[0]["metadata"]] == [0, 1]
    assert [item["silmaril"]["request_id"] for item in payloads[0]["metadata"]] == [
        "batch-id",
        "batch-id",
    ]
    assert caller_metadata == [{"trace": "one"}, {"trace": "two"}]
    assert [result.prediction for result in exc_info.value.results] == ["MALICIOUS", "BENIGN"]
    assert len(exc_info.value.blocked) == 1
    blocked = exc_info.value.blocked[0]
    assert (blocked.index, blocked.text, blocked.hook, blocked.tool_name) == (
        0,
        "attack",
        HookLabel.USER_INPUT,
        "chat",
    )


@pytest.mark.asyncio
async def test_cancellation_stops_request_and_retry_sleep_while_siblings_succeed(monkeypatch):
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["text"] == "in-flight":
            request_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                request_cancelled.set()
        if payload["text"] == "retry":
            return response(request, status=503)
        return response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=1,
        )
        in_flight = asyncio.create_task(fw.classify("in-flight"))
        await request_started.wait()
        assert (await fw.classify("sibling")).prediction == "BENIGN"
        in_flight.cancel()
        with pytest.raises(asyncio.CancelledError):
            await in_flight
        await asyncio.wait_for(request_cancelled.wait(), timeout=1)

        sleep_started = asyncio.Event()

        async def wait_forever(attempt: int, retry_after: str | None) -> None:
            sleep_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(fw, "_sleep_before_retry", wait_forever)
        retrying = asyncio.create_task(fw.classify("retry"))
        await sleep_started.wait()
        assert (await fw.classify("another sibling")).prediction == "BENIGN"
        retrying.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retrying


@pytest.mark.asyncio
async def test_owned_client_is_reused_closed_once_and_rejects_closed_use(monkeypatch):
    instances: list[Any] = []

    class TrackingClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.sends = 0
            self.close_calls = 0
            self.inner = real_async_client(
                transport=httpx.MockTransport(lambda request: response(request)),
                **kwargs,
            )
            instances.append(self)

        @property
        def is_closed(self) -> bool:
            return bool(self.inner.is_closed)

        def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
            return self.inner.build_request(*args, **kwargs)

        async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            self.sends += 1
            return await self.inner.send(request, **kwargs)

        async def aclose(self) -> None:
            self.close_calls += 1
            await self.inner.aclose()

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", TrackingClient)
    fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL)
    async with fw:
        await fw.classify("one")
        await fw.classify("two")

    await fw.aclose()
    assert len(instances) == 1
    assert instances[0].sends == 2
    assert instances[0].close_calls == 1
    assert instances[0].is_closed is True
    with pytest.raises(RuntimeError, match="closed"):
        await fw.classify("three")


@pytest.mark.asyncio
async def test_wrong_event_loop_is_rejected():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response(request))) as client:
        fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, http_client=client)
        await fw.classify("bind")

        def classify_on_another_loop() -> None:
            asyncio.run(fw.classify("wrong loop"))

        with pytest.raises(RuntimeError, match="different event loop"):
            await asyncio.to_thread(classify_on_another_loop)


@pytest.mark.asyncio
async def test_direct_async_callback_runs_before_block_and_failures_do_not_replace_verdict():
    events: list[ClassifyEvent] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        return response(request, prediction="MALICIOUS", score=0.9)

    async def callback(event: ClassifyEvent) -> None:
        events.append(event)
        raise RuntimeError("callback failure")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            on_classify=callback,
        )
        with pytest.raises(FirewallBlockedException):
            await fw.classify("attack")

    assert len(events) == 1
    assert events[0].blocked is True


@pytest.mark.asyncio
async def test_async_handler_uses_supplied_async_firewall_pool_and_semantics(monkeypatch):
    pytest.importorskip("langchain_core.callbacks")
    events: list[ClassifyEvent] = []
    calls = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response(request, prediction="MALICIOUS", score=0.9)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, http_client=client)

        async def public_classify_must_not_be_called(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("handler must preserve its own enforcement semantics")

        monkeypatch.setattr(fw, "classify", public_classify_must_not_be_called)
        handler = fw.as_async_langchain_handler(on_classify=events.append)
        run_id = uuid4()
        with pytest.raises(FirewallBlockedException) as exc_info:
            await handler.on_chat_model_start(
                serialized={},
                messages=[[{"role": "user", "content": "attack"}]],
                run_id=run_id,
            )

    assert calls == 1
    assert events[0].blocked is True
    assert exc_info.value.run_id == run_id


@pytest.mark.asyncio
async def test_async_handler_with_supplied_firewall_preserves_fail_open():
    pytest.importorskip("langchain_core.callbacks")

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, text="unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=0,
        )
        fail_open = fw.as_async_langchain_handler()
        await fail_open.on_chat_model_start(
            serialized={},
            messages=[[{"role": "user", "content": "allowed on outage"}]],
            run_id=uuid4(),
        )

        fail_closed = fw.as_async_langchain_handler(fail_open=False)
        with pytest.raises(SilmarilApiError):
            await fail_closed.on_chat_model_start(
                serialized={},
                messages=[[{"role": "user", "content": "fail closed"}]],
                run_id=uuid4(),
            )


@pytest.mark.asyncio
async def test_post_json_rejects_redirects():
    calls: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, request=request, text="redirect")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        original_send = client.send

        async def tracking_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            calls.append({"url": str(request.url), **kwargs})
            return await original_send(request, **kwargs)

        client.send = tracking_send  # type: ignore[method-assign]
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=0,
        )
        with pytest.raises(SilmarilApiError) as exc_info:
            await fw._post_json({"text": "hello", "threshold": 0.5})

    assert calls[0]["follow_redirects"] is False
    assert exc_info.value.status == 302
    assert exc_info.value.status_text == "Found"
    assert exc_info.value.body == "redirect"


@pytest.mark.asyncio
async def test_post_json_caps_error_body_and_redacts_message():
    body = "x" * (_MAX_ERROR_BODY_BYTES + 1024)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=0,
        )
        with pytest.raises(SilmarilApiError) as exc_info:
            await fw._post_json({"text": "hello", "threshold": 0.5})

    assert exc_info.value.body == body[:_MAX_ERROR_BODY_BYTES]
    assert body[:128] not in str(exc_info.value)


@pytest.mark.asyncio
async def test_error_body_read_stops_at_cap_instead_of_downloading_whole_response():
    chunk = b"x" * 8192
    total_chunks = (_MAX_ERROR_BODY_BYTES // len(chunk)) * 4
    produced = 0

    async def oversized_body() -> Any:
        nonlocal produced
        for _ in range(total_chunks):
            produced += len(chunk)
            yield chunk

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, content=oversized_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=0,
        )
        with pytest.raises(SilmarilApiError) as exc_info:
            await fw._post_json({"text": "hello"})

    assert exc_info.value.body == "x" * _MAX_ERROR_BODY_BYTES
    assert produced <= _MAX_ERROR_BODY_BYTES + len(chunk)
    assert produced < total_chunks * len(chunk)


@pytest.mark.asyncio
async def test_aclose_drains_active_retry_and_rejects_new_work(monkeypatch):
    attempts: list[str] = []
    sibling_started = asyncio.Event()
    sibling_release = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["text"]
        attempts.append(text)
        if text == "sibling":
            sibling_started.set()
            await sibling_release.wait()
            return response(request)
        if text == "retrying" and attempts.count("retrying") == 1:
            return response(request, status=503)
        return response(request)

    real_async_client = httpx.AsyncClient

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, max_retries=1)

    sleeping = asyncio.Event()
    sleep_release = asyncio.Event()

    async def held_sleep(attempt: int, retry_after: str | None) -> None:
        sleeping.set()
        await sleep_release.wait()

    monkeypatch.setattr(fw, "_sleep_before_retry", held_sleep)

    retrying = asyncio.create_task(fw.classify("retrying"))
    await asyncio.wait_for(sleeping.wait(), timeout=1)
    sibling = asyncio.create_task(fw.classify("sibling"))
    await asyncio.wait_for(sibling_started.wait(), timeout=1)

    closing = asyncio.create_task(fw.aclose())
    for _ in range(3):
        await asyncio.sleep(0)
    assert closing.done() is False
    assert fw._client.is_closed is False

    with pytest.raises(RuntimeError, match="closing"):
        await fw.classify("rejected")

    sleep_release.set()
    sibling_release.set()
    assert (await retrying).prediction == "BENIGN"
    assert (await sibling).prediction == "BENIGN"
    await asyncio.wait_for(closing, timeout=1)

    assert attempts == ["retrying", "sibling", "retrying"]
    assert fw._client.is_closed is True
    await fw.aclose()


@pytest.mark.asyncio
async def test_aclose_from_classify_callback_does_not_deadlock(monkeypatch):
    real_async_client = httpx.AsyncClient

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(lambda request: response(request)),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    closed_from_callback: list[bool] = []

    async def on_classify(event: ClassifyEvent) -> None:
        await fw.aclose()
        closed_from_callback.append(True)

    fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, on_classify=on_classify)

    result = await asyncio.wait_for(fw.classify("hello"), timeout=1)

    assert result.prediction == "BENIGN"
    assert closed_from_callback == [True]
    assert fw._client.is_closed is True


@pytest.mark.asyncio
async def test_aclose_inside_active_request_raises_and_leaves_owned_pool_open(monkeypatch):
    attempts: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(json.loads(request.content)["text"])
        if len(attempts) == 1:
            return response(request, status=503)
        return response(request)

    real_async_client = httpx.AsyncClient

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, max_retries=1)
    close_errors: list[str] = []

    async def close_during_retry(attempt: int, retry_after: str | None) -> None:
        with pytest.raises(RuntimeError, match="active classification") as exc_info:
            await fw.aclose()
        close_errors.append(str(exc_info.value))

    monkeypatch.setattr(fw, "_sleep_before_retry", close_during_retry)

    result = await asyncio.wait_for(fw.classify("retry then succeed"), timeout=1)

    assert result.prediction == "BENIGN"
    assert attempts == ["retry then succeed", "retry then succeed"]
    assert len(close_errors) == 1
    assert fw._closing is False
    assert fw._client.is_closed is False

    await fw.aclose()

    assert fw._client.is_closed is True
    with pytest.raises(RuntimeError, match="closed"):
        await fw.classify("after close")


@pytest.mark.asyncio
async def test_concurrent_aclose_callers_wait_for_owned_client_close(monkeypatch):
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_calls = 0
    real_async_client = httpx.AsyncClient

    class SlowClosingClient:
        def __init__(self, **kwargs: Any) -> None:
            self.inner = real_async_client(
                transport=httpx.MockTransport(lambda request: response(request)),
                **kwargs,
            )

        @property
        def is_closed(self) -> bool:
            return bool(self.inner.is_closed)

        def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
            return self.inner.build_request(*args, **kwargs)

        async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return await self.inner.send(request, **kwargs)

        async def aclose(self) -> None:
            nonlocal close_calls
            close_calls += 1
            close_started.set()
            await close_release.wait()
            await self.inner.aclose()

    monkeypatch.setattr(httpx, "AsyncClient", SlowClosingClient)
    fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL)
    await fw.classify("warm the pool")

    first = asyncio.create_task(fw.aclose())
    second = asyncio.create_task(fw.aclose())
    await asyncio.wait_for(close_started.wait(), timeout=1)
    for _ in range(3):
        await asyncio.sleep(0)

    assert first.done() is False
    assert second.done() is False

    close_release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert close_calls == 1
    assert fw._client.is_closed is True


@pytest.mark.asyncio
async def test_body_read_error_is_retried_and_then_succeeds(monkeypatch):
    attempts = 0
    sleeps: list[tuple[int, str | None]] = []

    async def failing_body() -> Any:
        raise httpx.ReadError("stream broke")
        yield b""  # pragma: no cover - unreachable, keeps this an async generator

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, request=request, content=failing_body())
        return response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=1,
        )

        async def record_sleep(attempt: int, retry_after: str | None) -> None:
            sleeps.append((attempt, retry_after))

        monkeypatch.setattr(fw, "_sleep_before_retry", record_sleep)

        result = await asyncio.wait_for(fw.classify("body read fails once"), timeout=1)

    assert attempts == 2
    assert sleeps == [(0, None)]
    assert result.prediction == "BENIGN"
    assert result.score == 0.1


@pytest.mark.asyncio
async def test_error_body_read_failure_follows_transport_retry_policy(monkeypatch):
    attempts = 0

    async def failing_error_body() -> Any:
        yield b"partial"
        raise httpx.ReadError("stream broke")

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(403, request=request, content=failing_error_body())
        return response(request)

    async def no_sleep(attempt: int, retry_after: str | None) -> None:
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        retrying = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=1,
        )
        monkeypatch.setattr(retrying, "_sleep_before_retry", no_sleep)

        result = await asyncio.wait_for(retrying.classify("error body fails once"), timeout=1)

        assert attempts == 2
        assert result.prediction == "BENIGN"

        attempts = 0
        exhausted = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=0,
        )
        with pytest.raises(httpx.ReadError):
            await exhausted.classify("error body always fails")

    assert attempts == 1


@pytest.mark.asyncio
async def test_classify_batch_snapshots_inputs_before_awaiting():
    started = asyncio.Event()
    release = asyncio.Event()
    events: list[ClassifyEvent] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            request=request,
            json={
                "predictions": [
                    {
                        "prediction": "MALICIOUS",
                        "score": 0.9,
                        "threshold": 0.5,
                        "mode": "block",
                    },
                    {
                        "prediction": "BENIGN",
                        "score": 0.1,
                        "threshold": 0.5,
                        "mode": "block",
                    },
                ]
            },
        )

    texts = ["attack", "safe"]
    hooks = [HookLabel.USER_INPUT, HookLabel.TOOL_RESPONSE]
    tool_names: list[str | None] = ["chat", "tool"]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            on_classify=events.append,
        )
        batch = asyncio.create_task(
            fw.classify_batch(texts, hooks=hooks, tool_names=tool_names)
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        texts.clear()
        hooks.clear()
        tool_names.clear()
        release.set()

        with pytest.raises(BatchFirewallBlockedException) as exc_info:
            await asyncio.wait_for(batch, timeout=1)

    assert [event.text for event in events] == ["attack", "safe"]
    assert [event.hook for event in events] == [HookLabel.USER_INPUT, HookLabel.TOOL_RESPONSE]
    assert [result.prediction for result in exc_info.value.results] == ["MALICIOUS", "BENIGN"]
    blocked = exc_info.value.blocked[0]
    assert (blocked.index, blocked.text, blocked.hook, blocked.tool_name) == (
        0,
        "attack",
        HookLabel.USER_INPUT,
        "chat",
    )
