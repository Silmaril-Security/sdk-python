# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
import builtins
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
        payload = __import__("json").loads(request.content)
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
        payloads.append(__import__("json").loads(request.content))
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
        payload = __import__("json").loads(request.content)
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
            self.posts = 0
            self.close_calls = 0
            self.is_closed = False
            instances.append(self)

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            self.posts += 1
            request = httpx.Request("POST", url)
            return response(request)

        async def aclose(self) -> None:
            self.close_calls += 1
            self.is_closed = True

    monkeypatch.setattr(httpx, "AsyncClient", TrackingClient)
    fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL)
    async with fw:
        await fw.classify("one")
        await fw.classify("two")

    await fw.aclose()
    assert len(instances) == 1
    assert instances[0].posts == 2
    assert instances[0].close_calls == 1
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
        original_post = client.post

        async def tracking_post(url: str, **kwargs: Any) -> httpx.Response:
            calls.append({"url": url, **kwargs})
            return await original_post(url, **kwargs)

        client.post = tracking_post  # type: ignore[method-assign]
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
