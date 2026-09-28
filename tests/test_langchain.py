# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from silmaril_security.sdk import (
    BlockResult,
    ClassifyEvent,
    Firewall,
    FirewallBlockedException,
    FirewallHook,
    HookLabel,
    SilmarilApiError,
)
from silmaril_security.sdk.langchain import _ABANDONED_MODEL_RUN_TTL_SECONDS

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

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None):
        calls.append((text, hook, tool_name, request_id))
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

    assert calls == [("second", HookLabel.USER_INPUT, None, str(run_id))]
    assert len(events) == 1
    assert events[0].blocked is True


def test_langchain_handler_fail_open(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler()

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None):
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

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None):
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

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None):
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
        "sdk_version": "0.6.1",
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


class _Generation:
    def __init__(self, text: str) -> None:
        self.text = text


class _LLMResult:
    def __init__(self, text: str) -> None:
        self.generations = [[_Generation(text)]]


class _ToolMessage:
    def __init__(self, content: str, name: str) -> None:
        self.role = "tool"
        self.content = content
        self.name = name


_MODEL_HOOKS = [
    FirewallHook.CHAT_MODEL_START,
    FirewallHook.LLM_START,
    FirewallHook.LLM_END,
    FirewallHook.TOOL_START,
    FirewallHook.TOOL_END,
    FirewallHook.RETRIEVER_START,
    FirewallHook.RETRIEVER_END,
]


def _agent_model_id(metadata):
    if metadata is None:
        return None
    return metadata["silmaril"]["agent_model_id"]


def _record_calls(monkeypatch, firewall: Firewall):
    calls = []

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None, metadata=None):
        calls.append(
            {
                "text": text,
                "hook": hook,
                "tool_name": tool_name,
                "request_id": request_id,
                "metadata": metadata,
            }
        )
        return BlockResult(prediction="BENIGN", score=0.1, threshold=0.5, mode=mode or "block")

    monkeypatch.setattr(firewall, "_classify_raw", fake_raw)
    return calls


def test_langchain_handler_sends_selected_agent_model_id(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=_MODEL_HOOKS)
    calls = _record_calls(monkeypatch, fw)

    serialized_only = uuid4()
    handler.on_chat_model_start(
        serialized={
            "id": ["langchain_openai", "chat_models", "base", "ChatOpenAI"],
            "name": "ChatOpenAI",
            "kwargs": {"model_name": "gpt-4o", "temperature": 0},
        },
        messages=[
            [
                {"role": "user", "content": "hello"},
                _ToolMessage("file contents", "read_file"),
            ]
        ],
        run_id=serialized_only,
    )

    selected = uuid4()
    handler.on_llm_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        prompts=["complete this"],
        run_id=selected,
        invocation_params={"model": "gpt-4.1-mini", "model_name": "gpt-4o"},
        metadata={"ls_model_name": "claude-3-5-sonnet"},
    )

    from_metadata = uuid4()
    handler.on_chat_model_start(
        serialized={"kwargs": {"temperature": 0}},
        messages=[[{"role": "user", "content": "meta"}]],
        run_id=from_metadata,
        metadata={"model": "application-model", "ls_model_name": "  claude-3-5-sonnet  "},
    )

    class_path = ["langchain_openai", "chat_models", "base", "ChatOpenAI"]
    handler.on_chat_model_start(
        serialized={"id": class_path, "kwargs": {"model_name": "gpt-4o"}},
        messages=[[{"role": "user", "content": "fallback"}]],
        run_id=uuid4(),
        invocation_params={"model": ".".join(class_path)},
    )
    handler.on_llm_start(
        serialized={"kwargs": {"model": "m" * 256}},
        prompts=["max length"],
        run_id=uuid4(),
    )

    assert [call["hook"] for call in calls[:3]] == [
        HookLabel.USER_INPUT,
        HookLabel.TOOL_RESPONSE,
        HookLabel.USER_INPUT,
    ]
    assert calls[1]["tool_name"] == "read_file"
    assert _agent_model_id(calls[0]["metadata"]) == "gpt-4o"
    assert _agent_model_id(calls[1]["metadata"]) == "gpt-4o"
    assert calls[0]["request_id"] == str(serialized_only)
    assert _agent_model_id(calls[2]["metadata"]) == "gpt-4.1-mini"
    assert _agent_model_id(calls[3]["metadata"]) == "claude-3-5-sonnet"
    assert _agent_model_id(calls[4]["metadata"]) == "gpt-4o"
    assert _agent_model_id(calls[5]["metadata"]) == "m" * 256


def test_langchain_handler_omits_agent_model_id_when_untrustworthy(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler()
    calls = _record_calls(monkeypatch, fw)
    class_path = ["langchain_openai", "chat_models", "base", "ChatOpenAI"]

    handler.on_chat_model_start(
        serialized={"id": class_path, "name": "ChatOpenAI", "kwargs": {}},
        messages=[[{"role": "user", "content": "class path"}]],
        run_id=uuid4(),
    )
    handler.on_llm_start(
        serialized={"id": class_path, "kwargs": {"model": ".".join(class_path)}},
        prompts=["path in model field"],
        run_id=uuid4(),
        invocation_params={
            "model": "/".join(class_path),
            "model_name": "",
            "model_id": {"id": "gpt-4o"},
        },
        metadata={"ls_model_name": " \n ", "model": "x" * 257},
    )
    handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "blank"}]],
        run_id=uuid4(),
        invocation_params={"model": "   ", "model_name": None},
    )
    handler.on_chat_model_start(
        serialized={},
        messages=[[{"role": "user", "content": "app label"}]],
        run_id=uuid4(),
        metadata={"model": "my-pipeline", "model_name": "pipeline", "model_id": "pipeline-id"},
    )
    handler.on_llm_start(
        serialized={},
        prompts=["overlong"],
        run_id=uuid4(),
        invocation_params={"model": "m" * 257},
        metadata={"ls_model_name": "n" * 257},
    )

    assert [call["metadata"] for call in calls] == [None, None, None, None, None]


def test_langchain_handler_keeps_agent_model_id_on_same_run_output(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=_MODEL_HOOKS)
    calls = _record_calls(monkeypatch, fw)
    run_a = uuid4()
    run_b = uuid4()
    run_c = uuid4()

    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "provider/model-a"}},
        messages=[[{"role": "user", "content": "a"}]],
        run_id=run_a,
    )
    handler.on_llm_start(
        serialized={},
        prompts=["b"],
        run_id=run_b,
        invocation_params={"model_name": "provider/model-b"},
    )
    handler.on_tool_start(
        serialized={"name": "search", "kwargs": {"model": "provider/model-a"}},
        input_str="tool query",
        run_id=uuid4(),
        invocation_params={"model": "provider/model-a"},
        metadata={"ls_model_name": "provider/model-a"},
    )
    handler.on_retriever_start(
        serialized={"id": ["retriever"], "kwargs": {"model": "provider/model-b"}},
        query="retriever query",
        run_id=uuid4(),
    )
    handler.on_llm_end(_LLMResult("answer a"), run_id=run_a)
    handler.on_llm_end(_LLMResult("answer b"), run_id=run_b)
    handler.on_llm_end(_LLMResult("answer a again"), run_id=run_a)
    handler.on_tool_end("tool output", run_id=uuid4(), name="search")
    handler.on_retriever_end([type("Doc", (), {"page_content": "doc"})()], run_id=uuid4())
    handler.on_llm_end(_LLMResult("never started"), run_id=uuid4())

    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "provider/model-c"}},
        messages=[[{"role": "user", "content": "c"}]],
        run_id=run_c,
    )
    handler.on_llm_error(RuntimeError("provider failed"), run_id=run_c)
    handler.on_llm_end(_LLMResult("should not reuse c"), run_id=run_c)

    assert [(_agent_model_id(call["metadata"]), call["hook"], call["text"]) for call in calls] == [
        ("provider/model-a", HookLabel.USER_INPUT, "a"),
        ("provider/model-b", HookLabel.USER_INPUT, "b"),
        (None, HookLabel.TOOL_CALL, "tool query"),
        (None, HookLabel.TOOL_CALL, "retriever query"),
        ("provider/model-a", HookLabel.LLM_OUTPUT, "answer a"),
        ("provider/model-b", HookLabel.LLM_OUTPUT, "answer b"),
        (None, HookLabel.LLM_OUTPUT, "answer a again"),
        (None, HookLabel.TOOL_RESPONSE, "tool output"),
        (None, HookLabel.TOOL_RESPONSE, "doc"),
        (None, HookLabel.LLM_OUTPUT, "never started"),
        ("provider/model-c", HookLabel.USER_INPUT, "c"),
        (None, HookLabel.LLM_OUTPUT, "should not reuse c"),
    ]


def test_langchain_output_uses_model_recorded_when_start_hook_is_disabled(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=[FirewallHook.LLM_END])
    calls = _record_calls(monkeypatch, fw)
    run_id = uuid4()

    handler.on_llm_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        prompts=["not classified"],
        run_id=run_id,
    )
    handler.on_llm_end(_LLMResult("visible output"), run_id=run_id)

    assert [(call["hook"], call["text"], _agent_model_id(call["metadata"])) for call in calls] == [
        (HookLabel.LLM_OUTPUT, "visible output", "gpt-4o"),
    ]


def test_langchain_handler_retains_models_for_many_concurrent_runs(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=_MODEL_HOOKS)
    calls = _record_calls(monkeypatch, fw)
    run_ids = [uuid4() for _ in range(257)]

    for index, run_id in enumerate(run_ids):
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": f"model-{index}"}},
            messages=[[{"role": "user", "content": "hi"}]],
            run_id=run_id,
        )
    for run_id in run_ids:
        handler.on_llm_end(_LLMResult("done"), run_id=run_id)

    by_request = {}
    for call in calls:
        by_request.setdefault(call["request_id"], []).append(call)
    for index, run_id in enumerate(run_ids):
        start, end = by_request[str(run_id)]
        assert _agent_model_id(start["metadata"]) == f"model-{index}"
        assert _agent_model_id(end["metadata"]) == f"model-{index}"
        assert end["hook"] == HookLabel.LLM_OUTPUT


def test_langchain_handler_expires_abandoned_starts_after_24_hours(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("silmaril_security.sdk.langchain.time.monotonic", lambda: clock["now"])
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=_MODEL_HOOKS)
    calls = _record_calls(monkeypatch, fw)
    abandoned_ids = [uuid4() for _ in range(257)]
    fresh = uuid4()

    for index, run_id in enumerate(abandoned_ids):
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": f"old-{index}"}},
            messages=[[{"role": "user", "content": "old"}]],
            run_id=run_id,
        )
    clock["now"] = 60.0
    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "fresh-model"}},
        messages=[[{"role": "user", "content": "fresh"}]],
        run_id=fresh,
    )
    clock["now"] = _ABANDONED_MODEL_RUN_TTL_SECONDS + 1
    handler.on_llm_end(_LLMResult("oldest expired"), run_id=abandoned_ids[0])

    assert str(fresh) in handler._run_models._ids
    assert str(abandoned_ids[0]) not in handler._run_models._ids
    assert str(abandoned_ids[-1]) not in handler._run_models._ids

    handler.on_llm_end(_LLMResult("fresh output"), run_id=fresh)
    handler.on_llm_end(_LLMResult("newest expired"), run_id=abandoned_ids[-1])

    fresh_calls = [call for call in calls if call["request_id"] == str(fresh)]
    assert [_agent_model_id(call["metadata"]) for call in fresh_calls] == ["fresh-model", "fresh-model"]
    expired_ends = [
        call
        for call in calls
        if call["request_id"] in {str(abandoned_ids[0]), str(abandoned_ids[-1])}
        and call["hook"] == HookLabel.LLM_OUTPUT
    ]
    assert [call["text"] for call in expired_ends] == ["oldest expired", "newest expired"]
    assert [_agent_model_id(call["metadata"]) for call in expired_ends] == [None, None]


def test_langchain_rejected_start_drops_remembered_model(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=_MODEL_HOOKS, fail_open=False)
    kept = uuid4()
    blocked = uuid4()
    rejected = uuid4()
    cancelled = uuid4()
    calls = []

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None, metadata=None):
        calls.append({"text": text, "hook": hook, "metadata": metadata, "request_id": request_id})
        if text == "blocked":
            return BlockResult(prediction="MALICIOUS", score=0.9, threshold=0.5, mode="block")
        if text == "rejected":
            raise SilmarilApiError(status=500, status_text="Internal Server Error", body="boom")
        if text == "cancel-me":
            raise asyncio.CancelledError()
        return BlockResult(prediction="BENIGN", score=0.1, threshold=0.5, mode=mode or "block")

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)
    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "kept-model"}},
        messages=[[{"role": "user", "content": "kept"}]],
        run_id=kept,
    )
    with pytest.raises(FirewallBlockedException):
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "blocked-model"}},
            messages=[[{"role": "user", "content": "blocked"}]],
            run_id=blocked,
        )
    with pytest.raises(SilmarilApiError):
        handler.on_llm_start(
            serialized={"kwargs": {"model": "rejected-model"}},
            prompts=["rejected"],
            run_id=rejected,
        )
    with pytest.raises(asyncio.CancelledError):
        handler.on_llm_start(
            serialized={"kwargs": {"model": "cancelled-model"}},
            prompts=["cancel-me"],
            run_id=cancelled,
        )

    assert str(blocked) not in handler._run_models._ids
    assert str(rejected) not in handler._run_models._ids
    assert str(cancelled) not in handler._run_models._ids
    assert str(kept) in handler._run_models._ids

    handler.on_llm_end(_LLMResult("kept output"), run_id=kept)
    handler.on_llm_end(_LLMResult("blocked output"), run_id=blocked)
    handler.on_llm_end(_LLMResult("rejected output"), run_id=rejected)
    handler.on_llm_end(_LLMResult("cancelled output"), run_id=cancelled)

    outputs = [call for call in calls if call["hook"] == HookLabel.LLM_OUTPUT]
    assert [(call["text"], _agent_model_id(call["metadata"])) for call in outputs] == [
        ("kept output", "kept-model"),
        ("blocked output", None),
        ("rejected output", None),
        ("cancelled output", None),
    ]


def test_langchain_fail_open_start_keeps_model_for_output(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=_MODEL_HOOKS)
    calls = []

    def fake_raw(text, *, hook=None, tool_name=None, request_id=None, mode=None, metadata=None):
        calls.append({"text": text, "hook": hook, "metadata": metadata})
        if hook == HookLabel.USER_INPUT:
            raise SilmarilApiError(status=500, status_text="Internal Server Error", body="boom")
        return BlockResult(prediction="BENIGN", score=0.1, threshold=0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", fake_raw)
    run_id = uuid4()
    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[[{"role": "user", "content": "outage"}]],
        run_id=run_id,
    )
    handler.on_llm_end(_LLMResult("after outage"), run_id=run_id)

    assert [(call["hook"], _agent_model_id(call["metadata"])) for call in calls] == [
        (HookLabel.USER_INPUT, "gpt-4o"),
        (HookLabel.LLM_OUTPUT, "gpt-4o"),
    ]


def test_langchain_selected_model_merges_into_request_metadata(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_langchain_handler(hooks=[FirewallHook.CHAT_MODEL_START, FirewallHook.LLM_END])
    payloads = []

    def fake_post(payload):
        payloads.append(payload)
        return {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5, "mode": "block"}

    monkeypatch.setattr(fw, "_post_json", fake_post)
    run_id = uuid4()
    handler.on_chat_model_start(
        serialized={
            "id": ["langchain_openai", "chat_models", "base", "ChatOpenAI"],
            "kwargs": {"model": "gpt-4o"},
        },
        messages=[[{"role": "user", "content": "hello"}]],
        run_id=run_id,
        invocation_params={"model_id": "openai/gpt-4.1-mini"},
    )
    handler.on_llm_end(_LLMResult("answer"), run_id=run_id)
    handler.on_chat_model_start(
        serialized={"id": ["langchain_openai", "chat_models", "base", "ChatOpenAI"], "name": "ChatOpenAI"},
        messages=[[{"role": "user", "content": "missing"}]],
        run_id=uuid4(),
    )

    assert payloads[0]["hook"] == "user_input"
    assert payloads[1]["hook"] == "llm_output"
    for payload in payloads[:2]:
        assert payload["metadata"]["silmaril"]["agent_model_id"] == "openai/gpt-4.1-mini"
        assert payload["metadata"]["silmaril"]["sdk_language"] == "python"
        assert payload["metadata"]["silmaril"]["request_id"] == str(run_id)
    assert "agent_model_id" not in payloads[2]["metadata"]["silmaril"]


@pytest.mark.asyncio
async def test_async_langchain_cancelled_or_blocked_start_drops_run_model(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_async_langchain_handler(hooks=_MODEL_HOOKS)
    calls = []

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
        calls.append({"text": text, "hook": hook, "metadata": metadata, "request_id": request_id})
        if text == "cancel-me":
            raise asyncio.CancelledError()
        if text == "blocked":
            return BlockResult(prediction="MALICIOUS", score=0.9, threshold=0.5, mode="block")
        return BlockResult(prediction="BENIGN", score=0.1, threshold=0.5, mode=mode or "block")

    monkeypatch.setattr("silmaril_security.sdk.langchain._async_classify_raw", fake_async_raw)
    kept = uuid4()
    cancelled = uuid4()
    blocked = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "kept-model"}},
        messages=[[{"role": "user", "content": "kept"}]],
        run_id=kept,
    )
    with pytest.raises(asyncio.CancelledError):
        await handler.on_chat_model_start(
            serialized={"kwargs": {"model": "cancelled-model"}},
            messages=[[{"role": "user", "content": "cancel-me"}]],
            run_id=cancelled,
        )
    with pytest.raises(FirewallBlockedException):
        await handler.on_llm_start(
            serialized={"kwargs": {"model": "blocked-model"}},
            prompts=["blocked"],
            run_id=blocked,
        )

    tracked = handler._sync_handler._run_models._ids
    assert str(cancelled) not in tracked
    assert str(blocked) not in tracked
    assert str(kept) in tracked

    await handler.on_llm_end(_LLMResult("kept output"), run_id=kept)
    await handler.on_llm_end(_LLMResult("cancelled output"), run_id=cancelled)
    await handler.on_llm_end(_LLMResult("blocked output"), run_id=blocked)

    outputs = [call for call in calls if call["hook"] == HookLabel.LLM_OUTPUT]
    assert [(call["text"], _agent_model_id(call["metadata"])) for call in outputs] == [
        ("kept output", "kept-model"),
        ("cancelled output", None),
        ("blocked output", None),
    ]


@pytest.mark.asyncio
async def test_async_langchain_task_cancellation_while_classifying_drops_run_model(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_async_langchain_handler(hooks=_MODEL_HOOKS)
    entered = asyncio.Event()

    async def pending_classification(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("silmaril_security.sdk.langchain._async_classify_raw", pending_classification)
    run_id = uuid4()
    task = asyncio.create_task(handler.on_llm_start(
        serialized={"kwargs": {"model": "provider/model-a"}},
        prompts=["pending"],
        run_id=run_id,
    ))
    await entered.wait()
    assert str(run_id) in handler._sync_handler._run_models._ids
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert str(run_id) not in handler._sync_handler._run_models._ids


@pytest.mark.asyncio
async def test_async_langchain_handler_attributes_agent_model_id_per_run(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_async_langchain_handler(hooks=_MODEL_HOOKS)
    calls = []

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
        calls.append({"text": text, "hook": hook, "request_id": request_id, "metadata": metadata})
        return BlockResult(prediction="BENIGN", score=0.1, threshold=0.5, mode=mode or "block")

    monkeypatch.setattr("silmaril_security.sdk.langchain._async_classify_raw", fake_async_raw)
    run_a = uuid4()
    run_b = uuid4()
    class_path = ["langchain", "llms", "openai", "OpenAI"]

    await handler.on_chat_model_start(
        serialized={"id": class_path, "name": "OpenAI", "kwargs": {"model_name": "provider/model-a"}},
        messages=[[{"role": "user", "content": "a"}]],
        run_id=run_a,
    )
    await handler.on_llm_start(
        serialized={"kwargs": {"model": "provider/model-a"}},
        prompts=["b"],
        run_id=run_b,
        invocation_params={"model": "provider/model-b"},
    )
    await handler.on_tool_start(
        serialized={"name": "search", "kwargs": {"model": "provider/model-a"}},
        input_str="tool query",
        run_id=uuid4(),
        metadata={"ls_model_name": "provider/model-a"},
    )
    await handler.on_llm_end(_LLMResult("answer a"), run_id=run_a)
    await handler.on_llm_end(_LLMResult("answer b"), run_id=run_b)
    await handler.on_retriever_start(serialized={"kwargs": {"model": "provider/model-b"}}, query="lookup", run_id=uuid4())
    await handler.on_chat_model_start(
        serialized={"id": class_path, "kwargs": {"model": ".".join(class_path)}},
        messages=[[{"role": "user", "content": "missing"}]],
        run_id=uuid4(),
        metadata={"model": "my-pipeline", "model_name": "pipeline", "model_id": "pipeline-id"},
    )
    run_c = uuid4()
    await handler.on_llm_start(
        serialized={"kwargs": {"model_id": "provider/model-c"}},
        prompts=["c"],
        run_id=run_c,
    )
    await handler.on_llm_error(RuntimeError("boom"), run_id=run_c)
    await handler.on_llm_end(_LLMResult("cleared by error"), run_id=run_c)

    assert [(_agent_model_id(call["metadata"]), call["hook"], call["text"]) for call in calls] == [
        ("provider/model-a", HookLabel.USER_INPUT, "a"),
        ("provider/model-b", HookLabel.USER_INPUT, "b"),
        (None, HookLabel.TOOL_CALL, "tool query"),
        ("provider/model-a", HookLabel.LLM_OUTPUT, "answer a"),
        ("provider/model-b", HookLabel.LLM_OUTPUT, "answer b"),
        (None, HookLabel.TOOL_CALL, "lookup"),
        (None, HookLabel.USER_INPUT, "missing"),
        ("provider/model-c", HookLabel.USER_INPUT, "c"),
        (None, HookLabel.LLM_OUTPUT, "cleared by error"),
    ]


@pytest.mark.asyncio
async def test_async_langchain_selected_model_merges_into_request_metadata(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://api.test.invalid/classify")
    handler = fw.as_async_langchain_handler(hooks=[FirewallHook.LLM_START, FirewallHook.LLM_END])
    payloads = []

    async def fake_post_json(self, payload):
        payloads.append(payload)
        return {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5, "mode": "block"}

    monkeypatch.setattr("silmaril_security.sdk.async_firewall.AsyncFirewall._post_json", fake_post_json)
    run_id = uuid4()
    await handler.on_llm_start(
        serialized={"id": ["langchain", "llms", "openai", "OpenAI"], "kwargs": {}},
        prompts=["hello"],
        run_id=run_id,
        metadata={"ls_model_name": "openai/text-davinci-003"},
    )
    await handler.on_llm_end(_LLMResult("answer"), run_id=run_id)

    assert payloads[0]["metadata"]["silmaril"]["agent_model_id"] == "openai/text-davinci-003"
    assert payloads[1]["metadata"]["silmaril"]["agent_model_id"] == "openai/text-davinci-003"
    assert payloads[0]["metadata"]["silmaril"]["sdk_language"] == "python"
    assert "threshold" not in payloads[0]
