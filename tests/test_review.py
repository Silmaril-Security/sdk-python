# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest
import requests

from silmaril_security.sdk import (
    AsyncFirewall,
    BatchFirewallBlockedException,
    Firewall,
    FirewallBlockedException,
    ReviewCompatibilityError,
    ReviewLevel,
)
from silmaril_security.sdk.exceptions import (
    ReviewCompatibilityError as DirectReviewCompatibilityError,
)
from silmaril_security.sdk.types import ReviewLevel as DirectReviewLevel

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


class RecordingAdapter(requests.adapters.BaseAdapter):
    def __init__(self, responses: list[tuple[int, dict[str, Any] | str]]) -> None:
        self.responses = responses
        self.requests: list[requests.PreparedRequest] = []

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        self.requests.append(request)
        status_code, body = self.responses.pop(0)
        response = requests.Response()
        response.status_code = status_code
        response.request = request
        response.url = request.url
        response._content = (
            json.dumps(body).encode() if isinstance(body, dict) else body.encode()
        )
        return response

    def close(self) -> None:
        pass


INVALID_REVIEWS = (
    "",
    " ",
    "standard ",
    "STANDARD",
    "Deep",
    "lite",
    True,
    1,
    ["standard"],
)


def _benign(*, review: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"prediction": "BENIGN", "score": 0.12, "threshold": 0.5}
    if review is not None:
        body["review"] = review
    return body


def test_review_types_are_exported():
    assert ReviewLevel is DirectReviewLevel
    assert ReviewCompatibilityError is DirectReviewCompatibilityError


@pytest.mark.parametrize("review", ["standard", "deep"])
def test_sync_explicit_review_is_serialized_and_acknowledged(monkeypatch, review: ReviewLevel):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, _benign(review=review))

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("hello", review=review, request_id="req-review")

    payload = json.loads(calls[0]["data"])
    assert payload["review"] == review
    assert result.review == review
    assert fw.review is None


def test_sync_omitted_review_keeps_legacy_wire_and_result(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, _benign())

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("hello", request_id="req-omit")

    payload = json.loads(calls[0]["data"])
    assert "review" not in payload
    assert result.review is None


def test_sync_none_review_means_omitted_not_null(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, review=None)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **kwargs: calls.append(kwargs) or FakeResponse(200, _benign()),
    )

    result = fw.classify("hello", review=None)

    assert "review" not in json.loads(calls[0]["data"])
    assert result.review is None
    assert fw.review is None


def test_per_call_review_overrides_client_default_without_mutating_it(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, review="standard")
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, _benign(review="deep"))

    monkeypatch.setattr(fw._session, "post", fake_post)

    result = fw.classify("hello", review="deep")

    assert json.loads(calls[0]["data"])["review"] == "deep"
    assert result.review == "deep"
    assert fw.review == "standard"


def test_client_default_review_is_sent_when_per_call_omitted(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, review="standard")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **kwargs: calls.append(kwargs)
        or FakeResponse(200, _benign(review="standard")),
    )

    result = fw.classify("hello")

    assert json.loads(calls[0]["data"])["review"] == "standard"
    assert result.review == "standard"


def test_sync_batch_uses_top_level_review_not_predictions(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(
            200,
            {
                "review": "deep",
                "predictions": [
                    {
                        "prediction": "BENIGN",
                        "score": 0.1,
                        "threshold": 0.5,
                        "review": "standard",
                    },
                    {"prediction": "BENIGN", "score": 0.2, "threshold": 0.5},
                ],
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    results = fw.classify_batch(["a", "b"], review="deep", request_id="req-batch")

    payload = json.loads(calls[0]["data"])
    assert payload["review"] == "deep"
    assert payload["texts"] == ["a", "b"]
    assert [result.review for result in results] == ["deep", "deep"]


def test_missing_and_mismatched_ack_do_not_reach_callback_or_block(monkeypatch):
    events: list[Any] = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        on_classify=events.append,
        max_retries=3,
    )
    calls = 0

    def missing_ack(url: str, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.99,
                "threshold": 0.5,
                "metadata": {"silmaril": {"review": "deep"}},
            },
        )

    monkeypatch.setattr(fw._session, "post", missing_ack)
    monkeypatch.setattr("silmaril_security.sdk.firewall.time.sleep", lambda _delay: None)

    with pytest.raises(ReviewCompatibilityError) as missing:
        fw.classify("attack", review="deep")

    assert missing.value.requested == "deep"
    assert missing.value.acknowledged is None
    assert calls == 1
    assert events == []

    def mismatched(url: str, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(
            200,
            {
                "prediction": "MALICIOUS",
                "score": 0.99,
                "threshold": 0.5,
                "review": "standard",
            },
        )

    calls = 0
    monkeypatch.setattr(fw._session, "post", mismatched)

    with pytest.raises(ReviewCompatibilityError) as mismatch:
        fw.classify("attack", review="deep")

    assert mismatch.value.acknowledged == "standard"
    assert calls == 1
    assert events == []


def test_batch_missing_envelope_ack_ignores_prediction_review(monkeypatch):
    events: list[Any] = []
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, on_classify=events.append, max_retries=2)
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(
            200,
            {
                "predictions": [
                    {
                        "prediction": "MALICIOUS",
                        "score": 0.9,
                        "threshold": 0.5,
                        "review": "standard",
                    }
                ]
            },
        )

    monkeypatch.setattr(fw._session, "post", fake_post)

    with pytest.raises(ReviewCompatibilityError):
        fw.classify_batch(["attack"], review="standard")

    assert calls == 1
    assert events == []


def test_retries_preserve_review_option(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, max_retries=2)
    calls: list[dict[str, Any]] = []
    responses = [
        FakeResponse(429, "rate limited"),
        FakeResponse(200, _benign(review="standard")),
    ]

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(fw._session, "post", fake_post)
    monkeypatch.setattr("silmaril_security.sdk.firewall.time.sleep", lambda _delay: None)

    result = fw.classify("hello", review="standard")

    assert result.review == "standard"
    assert [json.loads(call["data"])["review"] for call in calls] == ["standard", "standard"]


@pytest.mark.parametrize("value", INVALID_REVIEWS)
def test_invalid_review_values_error_locally(value: Any):
    with pytest.raises(ValueError, match="review must be standard or deep"):
        Firewall(api_key="sk", api_url=TEST_API_URL, review=value)  # type: ignore[arg-type]

    fw = Firewall(api_key="sk", api_url=TEST_API_URL)

    def fail_post(*_args: Any, **_kwargs: Any) -> FakeResponse:
        raise AssertionError("invalid review must not send HTTP")

    fw._session.post = fail_post  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="review must be standard or deep"):
        fw.classify("hello", review=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="review must be standard or deep"):
        fw.classify_batch(["hello"], review=value)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", INVALID_REVIEWS)
async def test_async_invalid_review_values_error_locally(value: Any):
    import httpx

    with pytest.raises(ValueError, match="review must be standard or deep"):
        AsyncFirewall(api_key="sk", api_url=TEST_API_URL, review=value)  # type: ignore[arg-type]

    calls = 0

    async def handle(request: Any) -> Any:
        nonlocal calls
        calls += 1
        return httpx.Response(200, request=request, json=_benign())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(api_key="sk", api_url=TEST_API_URL, http_client=client)
        with pytest.raises(ValueError, match="review must be standard or deep"):
            await fw.classify("hello", review=value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="review must be standard or deep"):
            await fw.classify_batch(["hello"], review=value)  # type: ignore[arg-type]
        await fw.aclose()

    assert calls == 0


def test_ack_failure_precedes_block_exception(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200,
            {"prediction": "MALICIOUS", "score": 0.99, "threshold": 0.5, "review": "standard"},
        ),
    )
    with pytest.raises(ReviewCompatibilityError):
        fw.classify("attack", review="deep")


def test_successful_ack_still_enforces_block(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200,
            {"prediction": "MALICIOUS", "score": 0.99, "threshold": 0.5, "review": "standard"},
        ),
    )
    with pytest.raises(FirewallBlockedException) as exc_info:
        fw.classify("attack", review="standard")
    assert exc_info.value.result is not None
    assert exc_info.value.result.review == "standard"


@pytest.mark.asyncio
async def test_async_client_and_per_call_precedence():
    payloads: list[dict[str, Any]] = []

    async def handle(request: Any) -> Any:
        import httpx

        payload = json.loads(request.content)
        payloads.append(payload)
        body: dict[str, Any] = {
            "prediction": "BENIGN",
            "score": 0.1,
            "threshold": 0.5,
        }
        if "review" in payload:
            body["review"] = payload["review"]
        return httpx.Response(200, request=request, json=body)

    import httpx

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            review="standard",
        )
        inherited = await fw.classify("inherited")
        overridden = await fw.classify("overridden", review="deep")
        omitted_client = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
        )
        omitted = await omitted_client.classify("omitted")
        await fw.aclose()

    assert inherited.review == "standard"
    assert overridden.review == "deep"
    assert omitted.review is None
    assert payloads[0]["review"] == "standard"
    assert payloads[1]["review"] == "deep"
    assert "review" not in payloads[2]
    assert fw.review == "standard"


@pytest.mark.asyncio
async def test_async_batch_and_ack_mismatch_is_not_retried():
    import httpx

    calls = 0

    async def handle(request: Any) -> Any:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=request,
            json={
                "predictions": [
                    {
                        "prediction": "MALICIOUS",
                        "score": 0.9,
                        "threshold": 0.5,
                        "review": "deep",
                    }
                ]
            },
        )

    events: list[Any] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=4,
            on_classify=events.append,
        )
        with pytest.raises(ReviewCompatibilityError):
            await fw.classify_batch(["attack"], review="deep")
        await fw.aclose()

    assert calls == 1
    assert events == []


@pytest.mark.asyncio
async def test_async_retries_preserve_review_and_successful_batch_ack():
    import httpx

    payloads: list[dict[str, Any]] = []
    statuses = [429, 200]

    async def handle(request: Any) -> Any:
        payloads.append(json.loads(request.content))
        status = statuses.pop(0)
        if status == 429:
            return httpx.Response(429, request=request, text="rate limited")
        return httpx.Response(
            200,
            request=request,
            json={
                "review": "standard",
                "predictions": [
                    {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5},
                    {"prediction": "MALICIOUS", "score": 0.9, "threshold": 0.5, "mode": "warn"},
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            max_retries=1,
        )
        results = await fw.classify_batch(["a", "b"], review="standard", mode="warn")
        await fw.aclose()

    assert [payload["review"] for payload in payloads] == ["standard", "standard"]
    assert [result.review for result in results] == ["standard", "standard"]
    assert results[1].prediction == "MALICIOUS"


@pytest.mark.asyncio
async def test_concurrent_reviews_do_not_bleed_client_state():
    import httpx

    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    payloads: list[dict[str, Any]] = []

    async def handle(request: Any) -> Any:
        nonlocal active
        payload = json.loads(request.content)
        payloads.append(payload)
        active += 1
        if active == 2:
            both_started.set()
        await release.wait()
        active -= 1
        return httpx.Response(
            200,
            request=request,
            json={
                "prediction": "BENIGN",
                "score": 0.1 if payload["text"] == "first" else 0.2,
                "threshold": 0.5,
                "review": payload["review"],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            review="standard",
        )
        first = asyncio.create_task(fw.classify("first", review="deep"))
        second = asyncio.create_task(fw.classify("second"))
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        await fw.aclose()

    assert fw.review == "standard"
    assert first_result.review == "deep"
    assert second_result.review == "standard"
    by_text = {item["text"]: item["review"] for item in payloads}
    assert by_text == {"first": "deep", "second": "standard"}


def test_langchain_and_deepagents_inherit_client_review_without_new_options():
    pytest.importorskip("langchain_core.callbacks")
    adapter = RecordingAdapter([(200, _benign(review="deep")), (200, _benign(review="deep"))])
    session = requests.Session()
    session.mount("https://api.test.invalid/", adapter)
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, session=session, review="deep")
    handler = fw.as_langchain_handler()
    handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "hello"}]],
        run_id=uuid4(),
    )
    assert json.loads(adapter.requests[0].body)["review"] == "deep"
    with pytest.raises(TypeError):
        fw.as_langchain_handler(review="standard")  # type: ignore[call-arg]

    pytest.importorskip("deepagents")
    from langchain_core.messages import HumanMessage

    from silmaril_security.sdk.deepagents import create_deepagents_middleware

    middleware = create_deepagents_middleware(fw)
    request = type(
        "Request",
        (),
        {
            "messages": [HumanMessage("hello")],
            "state": {"messages": []},
            "runtime": type("Runtime", (), {"config": {"run_id": "run-1"}})(),
            "tool_call": {"id": "call-1", "name": "search", "args": {"query": "safe"}},
        },
    )()
    middleware.wrap_model_call(request, lambda _: type("AI", (), {"text": "ok", "content": "ok"})())
    assert json.loads(adapter.requests[1].body)["review"] == "deep"


def test_langchain_fail_open_does_not_swallow_review_compatibility():
    pytest.importorskip("langchain_core.callbacks")
    adapter = RecordingAdapter(
        [(200, {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5})]
    )
    session = requests.Session()
    session.mount("https://api.test.invalid/", adapter)
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, session=session, review="standard")
    handler = fw.as_langchain_handler(fail_open=True)
    with pytest.raises(ReviewCompatibilityError):
        handler.on_chat_model_start(
            serialized={},
            messages=[[{"role": "user", "content": "hello"}]],
            run_id=uuid4(),
        )
    assert len(adapter.requests) == 1


def test_sync_batch_block_after_ack(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL)
    monkeypatch.setattr(
        fw._session,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            200,
            {
                "review": "standard",
                "predictions": [
                    {"prediction": "MALICIOUS", "score": 0.9, "threshold": 0.5, "mode": "block"}
                ],
            },
        ),
    )
    with pytest.raises(BatchFirewallBlockedException) as exc_info:
        fw.classify_batch(["attack"], review="standard")
    assert exc_info.value.results[0].review == "standard"


def test_per_call_none_inherits_explicit_client_review(monkeypatch):
    fw = Firewall(api_key="sk", api_url=TEST_API_URL, review="standard")
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse(200, _benign(review="standard"))

    monkeypatch.setattr(fw._session, "post", fake_post)
    result = fw.classify("hello", review=None)

    assert json.loads(calls[0]["data"])["review"] == "standard"
    assert result.review == "standard"
    assert fw.review == "standard"


@pytest.mark.asyncio
async def test_async_per_call_none_inherits_explicit_client_review():
    import httpx

    payloads: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(200, request=request, json=_benign(review=payload["review"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            review="standard",
        )
        result = await fw.classify("hello", review=None)
        await fw.aclose()

    assert payloads[0]["review"] == "standard"
    assert result.review == "standard"
    assert fw.review == "standard"


def _boundary_request() -> Any:
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content="hello")]
    return SimpleNamespace(
        messages=messages,
        state={"messages": []},
        runtime=SimpleNamespace(config={"run_id": "run-1"}),
        tool_call={"id": "call-1", "name": "search", "args": {"query": "safe"}},
    )


def _boundary_handler(surface: str, called: list[str]) -> Any:
    from langchain_core.messages import AIMessage, ToolMessage

    if surface == "model":

        def handle(_request: Any) -> Any:
            called.append(surface)
            return AIMessage(content="allowed")

        return handle

    def handle(_request: Any) -> Any:
        called.append(surface)
        return ToolMessage(content="allowed", tool_call_id="call-1")

    return handle


def _invoke_boundary(middleware: Any, surface: str, called: list[str]) -> Any:
    request = _boundary_request()
    handler = _boundary_handler(surface, called)
    if surface == "model":
        return middleware.wrap_model_call(request, handler)
    return middleware.wrap_tool_call(request, handler)


async def _ainvoke_boundary(middleware: Any, surface: str, called: list[str]) -> Any:
    from langchain_core.messages import AIMessage, ToolMessage

    request = _boundary_request()

    if surface == "model":

        async def handle(_request: Any) -> Any:
            called.append(surface)
            return AIMessage(content="allowed")

        return await middleware.awrap_model_call(request, handle)

    async def handle(_request: Any) -> Any:
        called.append(surface)
        return ToolMessage(content="allowed", tool_call_id="call-1")

    return await middleware.awrap_tool_call(request, handle)


def _assert_boundary_payloads(payloads: list[dict[str, Any]], surface: str, *, acknowledge: bool) -> None:
    expected_hooks = (
        ["user_input", "llm_output"] if surface == "model" else ["tool_call", "tool_response"]
    )
    assert [payload["review"] for payload in payloads] == ["standard"] * (2 if acknowledge else 1)
    assert [payload["hook"] for payload in payloads] == expected_hooks[: len(payloads)]


@pytest.mark.parametrize("surface", ["model", "tool"])
@pytest.mark.parametrize("acknowledge", [True, False])
def test_sync_deepagents_boundaries_inherit_review(surface: str, acknowledge: bool):
    pytest.importorskip("deepagents")
    from silmaril_security.sdk.deepagents import create_deepagents_middleware

    responses = [(200, _benign(review="standard") if acknowledge else _benign())] * (
        2 if acknowledge else 1
    )
    adapter = RecordingAdapter(responses)
    session = requests.Session()
    session.mount("https://api.test.invalid/", adapter)
    firewall_events: list[Any] = []
    adapter_events: list[Any] = []
    called: list[str] = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        session=session,
        review="standard",
        on_classify=firewall_events.append,
    )
    middleware = create_deepagents_middleware(
        fw,
        fail_open=True,
        on_classify=adapter_events.append,
    )

    if acknowledge:
        _invoke_boundary(middleware, surface, called)
        assert called == [surface]
        assert firewall_events
        assert adapter_events
    else:
        with pytest.raises(ReviewCompatibilityError):
            _invoke_boundary(middleware, surface, called)
        assert called == []
        assert firewall_events == []
        assert adapter_events == []

    payloads = [json.loads(request.body) for request in adapter.requests]
    _assert_boundary_payloads(payloads, surface, acknowledge=acknowledge)
    assert fw.review == "standard"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["model", "tool"])
@pytest.mark.parametrize("acknowledge", [True, False])
async def test_async_deepagents_boundaries_inherit_review(surface: str, acknowledge: bool):
    pytest.importorskip("deepagents")
    import httpx

    from silmaril_security.sdk.deepagents import create_deepagents_middleware

    payloads: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        body = _benign(review=payload["review"]) if acknowledge else _benign()
        return httpx.Response(200, request=request, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        firewall_events: list[Any] = []
        adapter_events: list[Any] = []
        called: list[str] = []
        fw = AsyncFirewall(
            api_key="sk",
            api_url=TEST_API_URL,
            http_client=client,
            review="standard",
            on_classify=firewall_events.append,
        )
        middleware = create_deepagents_middleware(
            fw,
            fail_open=True,
            on_classify=adapter_events.append,
        )
        if acknowledge:
            await _ainvoke_boundary(middleware, surface, called)
            assert called == [surface]
            assert firewall_events
            assert adapter_events
        else:
            with pytest.raises(ReviewCompatibilityError):
                await _ainvoke_boundary(middleware, surface, called)
            assert called == []
            assert firewall_events == []
            assert adapter_events == []
        await fw.aclose()

    _assert_boundary_payloads(payloads, surface, acknowledge=acknowledge)
    assert fw.review == "standard"


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledge", [True, False])
async def test_async_langchain_bridge_inherits_client_review(monkeypatch, acknowledge: bool):
    pytest.importorskip("langchain_core.callbacks")
    from silmaril_security.sdk.async_firewall import AsyncFirewall as AsyncClient

    recorded: list[dict[str, Any]] = []

    async def post_json(self: AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        recorded.append(payload)
        body = _benign()
        if acknowledge:
            body["review"] = payload["review"]
        return body

    monkeypatch.setattr(AsyncClient, "_post_json", post_json)
    firewall_events: list[Any] = []
    callback_events: list[Any] = []
    fw = Firewall(
        api_key="sk",
        api_url=TEST_API_URL,
        review="standard",
        on_classify=firewall_events.append,
    )
    handler = fw.as_async_langchain_handler(fail_open=True, on_classify=callback_events.append)
    call = handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "hello"}]],
        run_id=uuid4(),
    )
    if acknowledge:
        await call
        assert callback_events
    else:
        with pytest.raises(ReviewCompatibilityError):
            await call
        assert callback_events == []
    assert firewall_events == []
    assert recorded[0]["review"] == "standard"
    assert recorded[0]["hook"] == "user_input"
    assert len(recorded) == 1
    assert fw.review == "standard"
