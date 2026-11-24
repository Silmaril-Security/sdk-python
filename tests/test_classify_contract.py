# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import dataclass, field, replace
from operator import itemgetter
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import requests
from wire import (
    V1_URL,
    V2_URL,
    RecordingAdapter,
    decision,
    governance_block,
    model_block,
    results,
    v1,
)

from silmaril_security.sdk import (
    AsyncFirewall,
    BatchFirewallBlockedException,
    BlockResult,
    Firewall,
    FirewallBlockedException,
    GovernanceDecision,
    GovernanceResource,
    ReviewCompatibilityError,
    SilmarilApiError,
)
from silmaril_security.sdk._classify_contract import (
    FALLBACK_STATUS_CODES,
    ClassifyRoute,
    adapt_v1_response,
)
from silmaril_security.sdk.firewall import _block_results

CONTRACT = Path(__file__).parent / "contracts" / "classify" / "v2"
CASES = json.loads((CONTRACT / "cases.json").read_bytes())
CASE_NAME = itemgetter("name")
ORIGIN = "https://api.test.invalid"
MODEL_BLOCK_V1 = v1(
    "MALICIOUS",
    primary_outcome="secret_exposure",
    outcome_scores={"secret_exposure": 0.75},
)
MODEL_BLOCK = BlockResult(action="block", source="model", outcome="secret_exposure")
KINDS = ["sync", "async"]


@dataclass
class Runtime:
    client: Firewall | AsyncFirewall = field(init=False)
    urls: list[str] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list)

    def record(self, url: str, body: bytes, api_key: str) -> None:
        self.urls.append(url)
        self.payloads.append(json.loads(body))
        self.api_keys.append(api_key)

    async def call(self, method: str, *args: Any, **options: Any) -> Any:
        outcome = getattr(self.client, method)(*args, **options)
        return await outcome if inspect.isawaitable(outcome) else outcome


class RuntimeAdapter(RecordingAdapter):
    def __init__(self, responses: list[tuple[int, Any]], runtime: Runtime) -> None:
        super().__init__(responses)
        self.runtime = runtime

    def send(self, request: requests.PreparedRequest, **options: Any) -> requests.Response:
        self.runtime.record(request.url, request.body, request.headers["x-api-key"])
        return super().send(request, **options)


def connect(
    kind: str,
    responses: list[tuple[int, Any]],
    *,
    api_url: str = V1_URL,
    **options: Any,
) -> Runtime:
    runtime = Runtime()
    options = {"api_key": "sk-contract", "api_url": api_url, "max_retries": 0, **options}
    if kind == "sync":
        session = requests.Session()
        session.mount(ORIGIN, RuntimeAdapter(responses, runtime))
        runtime.client = Firewall(session=session, **options)
        return runtime

    def handle(request: httpx.Request) -> httpx.Response:
        runtime.record(str(request.url), request.content, request.headers["x-api-key"])
        status, body = responses.pop(0)
        content = {"text": body} if isinstance(body, str) else {"json": body}
        return httpx.Response(status, request=request, **content)

    runtime.client = AsyncFirewall(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), **options
    )
    return runtime


def expected_governance(governance: dict[str, Any] | None) -> GovernanceDecision | None:
    if governance is None:
        return None
    return GovernanceDecision(
        action=governance["action"],
        policy_version=governance["policy_version"],
        rule_id=governance["rule_id"],
        resource=GovernanceResource(**governance["resource"]),
        identity_revision=governance["identity_revision"],
    )


def expected_results(body: dict[str, Any], **carried: Any) -> list[BlockResult]:
    return [
        BlockResult(
            action=item["action"],
            source=item["source"],
            confidence=item["confidence"],
            severity=item["severity"],
            outcome=item["outcome"],
            governance=expected_governance(item["governance"]),
            model_id=item["model_id"],
            review=body.get("review"),
            **carried,
        )
        for item in body["results"]
    ]


def decode(case: dict[str, Any], version: str) -> list[BlockResult]:
    return _block_results(
        case[version],
        expected_length=len(case["v2"]["results"]),
        mode=None,
        review=case[version].get("review"),
    )


def test_vendored_cases_match_pinned_digest():
    pinned = dict(
        reversed(line.split()) for line in (CONTRACT / "SHA256SUMS").read_text().splitlines()
    )

    assert (
        hashlib.sha256((CONTRACT / "cases.json").read_bytes()).hexdigest() == pinned["cases.json"]
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_NAME)
def test_version_2_body_decodes_to_its_decisions(case):
    decisions = decode(case, "v2")

    assert decisions == expected_results(case["v2"])
    assert [result.blocked for result in decisions] == [
        item["action"] != "allow" for item in case["v2"]["results"]
    ]


@pytest.mark.parametrize("case", CASES, ids=CASE_NAME)
def test_version_1_body_adapts_to_the_pinned_body(case):
    assert adapt_v1_response(case["v1"]) == case["adapted"]


@pytest.mark.parametrize("case", CASES, ids=CASE_NAME)
def test_adapted_decisions_equal_version_2_without_judge_evidence(case):
    adapted = decode(case, "v1")

    assert adapted == expected_results(case["adapted"])
    assert adapted == [
        replace(result, confidence=None, severity=None) for result in decode(case, "v2")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", [case for case in CASES if case["v2"]["results"]], ids=CASE_NAME)
@pytest.mark.parametrize(
    ("responses", "body", "urls"),
    [
        (lambda case: [(200, case["v2"])], "v2", [V2_URL]),
        (lambda case: [(404, "{}"), (200, case["v1"])], "adapted", [V2_URL, V1_URL]),
        (lambda case: [(200, case["v1"])], "adapted", [V2_URL]),
    ],
    ids=["version-2", "version-1-by-fallback", "version-1-from-the-version-2-path"],
)
async def test_clients_return_every_case_whichever_path_answers(kind, case, responses, body, urls):
    batch = "predictions" in case["v1"]
    texts = [f"input-{index}" for index in range(len(case["v2"]["results"]))]
    runtime = connect(kind, responses(case))
    options = {"mode": "warn", "review": case["v2"].get("review")}

    decisions = (
        await runtime.call("classify_batch", texts, **options)
        if batch
        else [await runtime.call("classify", texts[0], **options)]
    )

    assert decisions == expected_results(case[body], mode="warn")
    assert runtime.urls == urls


@pytest.mark.parametrize(
    ("api_url", "base", "query"),
    [
        (ORIGIN, ORIGIN, ""),
        (f"{ORIGIN}/", ORIGIN, ""),
        (f"{ORIGIN}/classify", ORIGIN, ""),
        (f"{ORIGIN}/classify/", ORIGIN, ""),
        (f"{ORIGIN}/v2/classify", ORIGIN, ""),
        (f"{ORIGIN}/v2/classify/", ORIGIN, ""),
        (f"{ORIGIN}/alpha", f"{ORIGIN}/alpha", ""),
        (f"{ORIGIN}/alpha/", f"{ORIGIN}/alpha", ""),
        (f"{ORIGIN}/alpha/classify", f"{ORIGIN}/alpha", ""),
        (f"{ORIGIN}/alpha/v2/classify", f"{ORIGIN}/alpha", ""),
        (f"{ORIGIN}/prod/alpha/classify/", f"{ORIGIN}/prod/alpha", ""),
        (f"{ORIGIN}/reclassify", f"{ORIGIN}/reclassify", ""),
        (f"{ORIGIN}/classify/v2", f"{ORIGIN}/classify/v2", ""),
        (f"{ORIGIN}?tenant=alpha", ORIGIN, "?tenant=alpha"),
        (f"{ORIGIN}/?tenant=alpha", ORIGIN, "?tenant=alpha"),
        (f"{ORIGIN}/classify?tenant=alpha&stage=prod", ORIGIN, "?tenant=alpha&stage=prod"),
        (f"{ORIGIN}/v2/classify?tenant=alpha", ORIGIN, "?tenant=alpha"),
        (f"{ORIGIN}/alpha/classify/?path=/classify", f"{ORIGIN}/alpha", "?path=/classify"),
        ("https://api.test.invalid:8443/prod/classify", "https://api.test.invalid:8443/prod", ""),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080", ""),
    ],
)
def test_endpoints_derive_from_the_configured_url(api_url, base, query):
    route = ClassifyRoute.from_api_url(api_url)

    assert (route.v2_url, route.v1_url) == (f"{base}/v2/classify{query}", f"{base}/classify{query}")
    assert route.legacy is False
    assert Firewall(api_key="sk", api_url=api_url).api_url == api_url
    assert AsyncFirewall(api_key="sk", api_url=api_url).api_url == api_url


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "api_url",
    [f"{ORIGIN}/alpha", f"{ORIGIN}/alpha/classify", f"{ORIGIN}/alpha/v2/classify/?tenant=alpha"],
)
async def test_clients_call_the_derived_endpoints(kind, api_url):
    query = "?tenant=alpha" if "?" in api_url else ""
    runtime = connect(kind, [(404, "{}"), (200, v1())], api_url=api_url)

    await runtime.call("classify", "hello")

    assert runtime.urls == [
        f"{ORIGIN}/alpha/v2/classify{query}",
        f"{ORIGIN}/alpha/classify{query}",
    ]


def test_fallback_statuses_are_the_contract_statuses():
    assert FALLBACK_STATUS_CODES == {403, 404, 405}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status", [403, 404, 405])
async def test_fallback_adapts_version_1_and_is_remembered(kind, status):
    runtime = connect(
        kind,
        [(status, "no such route"), (200, MODEL_BLOCK_V1), (200, v1())],
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        await runtime.call("classify", "attack", request_id="req-fallback")
    remembered = await runtime.call("classify", "safe")

    assert exc_info.value.result == MODEL_BLOCK
    assert remembered == BlockResult(action="allow")
    assert runtime.urls == [V2_URL, V1_URL, V1_URL]
    assert runtime.payloads[0] == runtime.payloads[1]
    assert runtime.api_keys == ["sk-contract"] * 3
    assert runtime.client._route.legacy is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status", [403, 404, 405])
async def test_batch_fallback_adapts_version_1_and_is_remembered(kind, status):
    predictions = {"predictions": [v1(), MODEL_BLOCK_V1]}
    runtime = connect(
        kind,
        [(status, "no such route"), (200, predictions), (200, predictions)],
    )

    with pytest.raises(BatchFirewallBlockedException) as exc_info:
        await runtime.call("classify_batch", ["safe", "attack"], request_id="req-fallback")
    remembered = await runtime.call("classify_batch", ["safe", "attack"], mode="warn")

    assert exc_info.value.results == [BlockResult(action="allow"), MODEL_BLOCK]
    assert [item.index for item in exc_info.value.blocked] == [1]
    assert [result.action for result in remembered] == ["allow", "block"]
    assert runtime.urls == [V2_URL, V1_URL, V1_URL]
    assert runtime.payloads[0] == runtime.payloads[1]
    assert runtime.payloads[0]["texts"] == ["safe", "attack"]
    assert runtime.client._route.legacy is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ["classify", "classify_batch"])
@pytest.mark.parametrize(
    "status", [301, 302, 400, 401, 402, 406, 409, 410, 413, 422, 429, 500, 503]
)
async def test_other_statuses_do_not_fall_back(kind, method, status):
    runtime = connect(kind, [(status, "refused"), (200, results(decision()))])
    argument = "hello" if method == "classify" else ["hello"]

    with pytest.raises(SilmarilApiError) as exc_info:
        await runtime.call(method, argument)
    after = await runtime.call(method, argument)

    assert exc_info.value.status == status
    assert after in (BlockResult(action="allow"), [BlockResult(action="allow")])
    assert runtime.urls == [V2_URL, V2_URL]
    assert runtime.client._route.legacy is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_version_2_success_never_calls_version_1(kind):
    runtime = connect(kind, [(200, results(model_block())), (200, results(decision()))])

    with pytest.raises(FirewallBlockedException):
        await runtime.call("classify", "attack")
    await runtime.call("classify", "safe")

    assert runtime.urls == [V2_URL, V2_URL]
    assert runtime.client._route.legacy is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fallback_status", [401, 403, 404, 500])
async def test_failed_fallback_is_the_result_and_is_not_remembered(kind, fallback_status):
    runtime = connect(
        kind,
        [(404, "no such route"), (fallback_status, "refused"), (200, results(decision()))],
    )

    with pytest.raises(SilmarilApiError) as exc_info:
        await runtime.call("classify", "hello")
    after = await runtime.call("classify", "hello")

    assert (exc_info.value.status, exc_info.value.body) == (fallback_status, "refused")
    assert after == BlockResult(action="allow")
    assert runtime.urls == [V2_URL, V1_URL, V2_URL]
    assert runtime.client._route.legacy is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_each_endpoint_keeps_the_retry_policy(kind, monkeypatch):
    runtime = connect(
        kind,
        [(503, "busy"), (404, "no such route"), (429, "slow down"), (200, v1())],
        max_retries=2,
    )
    waits: list[int] = []

    def wait(attempt: int, _hint: Any) -> None:
        waits.append(attempt)

    async def async_wait(attempt: int, _hint: Any) -> None:
        waits.append(attempt)

    monkeypatch.setattr(
        runtime.client, "_sleep_before_retry", wait if kind == "sync" else async_wait
    )

    result = await runtime.call("classify", "hello")

    assert result == BlockResult(action="allow")
    assert runtime.urls == [V2_URL, V2_URL, V1_URL, V1_URL]
    assert waits == [0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_mode_precedence_is_requested_then_adapted_then_block(kind):
    runtime = connect(
        kind,
        [(404, "no such route")]
        + [(200, v1("MALICIOUS", mode="warn"))] * 2
        + [(200, v1("MALICIOUS"))],
    )

    adapted_mode = await runtime.call("classify", "attack")
    with pytest.raises(FirewallBlockedException) as requested:
        await runtime.call("classify", "attack", mode="block")
    with pytest.raises(FirewallBlockedException) as defaulted:
        await runtime.call("classify", "attack")

    assert adapted_mode.mode == "warn"
    assert requested.value.result.mode == "block"
    assert defaulted.value.result.mode is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_adapted_governance_block_names_its_source(kind):
    governance = {"action": "block", "policy_version": "policy-9", "rule_id": "block-shell"}
    runtime = connect(
        kind,
        [
            (404, "no such route"),
            (200, v1("MALICIOUS", model_prediction="BENIGN", governance=governance)),
        ],
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        await runtime.call("classify", "governed")

    assert exc_info.value.result == BlockResult(
        action="block",
        source="governance",
        governance=GovernanceDecision(
            action="block", policy_version="policy-9", rule_id="block-shell"
        ),
    )


@pytest.mark.parametrize("governance", [None, {"action": "allow", "policy_version": "policy-9"}])
def test_version_1_block_the_model_did_not_make_adapts_to_a_governance_block(governance):
    body = v1("MALICIOUS", model_prediction="BENIGN", governance=governance)

    assert adapt_v1_response(body) == results(governance_block(governance))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_adapted_result_never_allows_what_version_1_blocked(kind):
    runtime = connect(
        kind,
        [(404, "no such route"), (200, v1("MALICIOUS", model_prediction="BENIGN"))],
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        await runtime.call("classify", "attack")

    assert exc_info.value.result == BlockResult(action="block", source="governance")


@pytest.mark.parametrize("model_id", ["silmaril-cascade-1-6", None])
def test_version_1_model_id_carries_over(model_id):
    assert adapt_v1_response(v1(model_id=model_id)) == results(decision(model_id=model_id))
    assert adapt_v1_response(v1()) == results(decision())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("action", ["allow", "block"])
@pytest.mark.parametrize(
    ("result", "model_id"),
    [
        ({"model_id": "silmaril-cascade-1-6"}, "silmaril-cascade-1-6"),
        ({"model_id": None}, None),
        ({}, None),
    ],
    ids=["set", "null", "absent"],
)
async def test_model_id_is_provenance_and_never_affects_enforcement(kind, action, result, model_id):
    runtime = connect(kind, [(200, {"results": [{"action": action, **result}]})])

    decided = await runtime.call("classify", "hello", mode="warn")

    assert decided == BlockResult(action=action, model_id=model_id, mode="warn")
    assert decided.blocked is (action == "block")


def test_version_1_allow_carries_no_source_or_outcome():
    body = v1(
        model_prediction="MALICIOUS",
        primary_outcome="secret_exposure",
        outcome_scores={"secret_exposure": 0.75},
    )

    assert adapt_v1_response(body) == results(decision())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_fallback_keeps_the_review_acknowledgement_requirement(kind):
    runtime = connect(
        kind,
        [(404, "no such route"), (200, v1(review="deep")), (200, v1())],
    )

    acknowledged = await runtime.call("classify", "hello", review="deep")
    with pytest.raises(ReviewCompatibilityError):
        await runtime.call("classify", "hello", review="deep")

    assert acknowledged == BlockResult(action="allow", review="deep")
    assert [payload["review"] for payload in runtime.payloads] == ["deep"] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"score": 0.99, "threshold": 0.5}, "missing required 'prediction' field"),
        ({"model_prediction": "BENIGN"}, "missing required 'prediction' field"),
        (v1("SUSPICIOUS"), "invalid prediction 'SUSPICIOUS'"),
        (v1(model_prediction="SUSPICIOUS"), "invalid prediction 'SUSPICIOUS'"),
        ({"predictions": v1()}, "predictions must be a list of objects"),
        ({"predictions": ["BENIGN"]}, "predictions must be a list of objects"),
        ({"predictions": []}, "results length 0 does not match texts length 1"),
        (v1(governance="block"), "response governance must be an object"),
        (v1(governance={"action": "block"}), "governance policy_version"),
    ],
)
async def test_malformed_version_1_fallback_is_rejected_and_not_remembered(kind, body, message):
    runtime = connect(
        kind,
        [(404, "no such route"), (200, body), (200, results(decision()))],
        mode="shadow",
    )

    with pytest.raises(ValueError, match=message):
        await runtime.call("classify", "hello")
    after = await runtime.call("classify", "hello")

    assert after == BlockResult(action="allow", mode="shadow")
    assert runtime.urls == [V2_URL, V1_URL, V2_URL]
    assert runtime.client._route.legacy is False


@pytest.mark.parametrize("body", [None, [], "BENIGN", 7])
def test_non_object_version_1_body_is_rejected(body):
    with pytest.raises(ValueError, match="backend response must be an object"):
        _block_results(body, expected_length=1, mode=None, review=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_version_1_body_from_the_version_2_path_blocks_and_keeps_the_route(kind):
    runtime = connect(
        kind,
        [
            (200, MODEL_BLOCK_V1),
            (200, {"predictions": [v1(), MODEL_BLOCK_V1]}),
            (200, v1("MALICIOUS", mode="warn")),
            (200, results(decision())),
        ],
    )

    with pytest.raises(FirewallBlockedException) as single:
        await runtime.call("classify", "attack")
    with pytest.raises(BatchFirewallBlockedException) as batch:
        await runtime.call("classify_batch", ["safe", "attack"])
    warned = await runtime.call("classify", "attack")
    allowed = await runtime.call("classify", "safe")

    assert single.value.result == MODEL_BLOCK
    assert batch.value.results == [BlockResult(action="allow"), MODEL_BLOCK]
    assert [item.index for item in batch.value.blocked] == [1]
    assert (warned.action, warned.mode) == ("block", "warn")
    assert allowed == BlockResult(action="allow")
    assert runtime.urls == [V2_URL] * 4
    assert runtime.client._route.legacy is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ["classify", "classify_batch"])
@pytest.mark.parametrize(
    "body",
    [{}, {"detail": "accepted"}, {"action": "block"}, {"score": 0.99, "threshold": 0.5}],
    ids=["empty", "detail", "bare-action", "scores-only"],
)
async def test_body_that_is_neither_version_is_rejected(kind, method, body):
    events: list[Any] = []
    runtime = connect(kind, [(200, body)], mode="shadow", on_classify=events.append)

    with pytest.raises(ValueError, match="missing required 'prediction' field"):
        await runtime.call(method, "hello" if method == "classify" else ["hello"])

    assert events == []
    assert runtime.urls == [V2_URL]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fail_open", [True, False])
async def test_langchain_handler_blocks_on_a_version_1_body_from_the_version_2_path(
    kind, fail_open
):
    pytest.importorskip("langchain_core.callbacks")
    runtime = connect(kind, [(200, MODEL_BLOCK_V1)])
    handler = (
        runtime.client.as_langchain_handler(fail_open=fail_open)
        if kind == "sync"
        else runtime.client.as_async_langchain_handler(fail_open=fail_open)
    )

    with pytest.raises(FirewallBlockedException) as exc_info:
        started = handler.on_chat_model_start(
            serialized={}, messages=[[{"role": "user", "content": "attack"}]], run_id=uuid4()
        )
        if inspect.isawaitable(started):
            await started

    assert exc_info.value.result == MODEL_BLOCK
    assert runtime.urls == [V2_URL]


def scripted_async_client(monkeypatch, handle: Any) -> None:
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **options: real_async_client(transport=httpx.MockTransport(handle), **options),
    )


@pytest.mark.asyncio
async def test_async_close_drains_a_classification_that_is_falling_back(monkeypatch):
    falling_back = asyncio.Event()
    release = asyncio.Event()
    urls: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if str(request.url) == V2_URL:
            falling_back.set()
            await release.wait()
            return httpx.Response(404, request=request, text="no such route")
        return httpx.Response(200, request=request, json=v1())

    scripted_async_client(monkeypatch, handle)
    fw = AsyncFirewall(api_key="sk", api_url=V1_URL)
    classifying = asyncio.create_task(fw.classify("hello"))
    await asyncio.wait_for(falling_back.wait(), timeout=1)
    closing = asyncio.create_task(fw.aclose())
    for _ in range(3):
        await asyncio.sleep(0)
    assert closing.done() is False

    release.set()
    result = await asyncio.wait_for(classifying, timeout=1)
    await asyncio.wait_for(closing, timeout=1)

    assert result == BlockResult(action="allow")
    assert urls == [V2_URL, V1_URL]
    assert fw._client.is_closed is True


@pytest.mark.asyncio
async def test_async_langchain_bridge_shares_the_remembered_fallback(monkeypatch):
    pytest.importorskip("langchain_core.callbacks")
    urls: list[str] = []
    responses = [(404, "no such route"), (200, v1()), (200, v1())]

    def handle(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        status, body = responses.pop(0)
        content = {"text": body} if isinstance(body, str) else {"json": body}
        return httpx.Response(status, request=request, **content)

    scripted_async_client(monkeypatch, handle)
    fw = Firewall(api_key="sk", api_url=V1_URL, max_retries=0)
    handler = fw.as_async_langchain_handler(fail_open=False)

    for text in ("first", "second"):
        await handler.on_chat_model_start(
            serialized={}, messages=[[{"role": "user", "content": text}]], run_id=uuid4()
        )

    assert urls == [V2_URL, V1_URL, V1_URL]
    assert fw._route.legacy is True
