# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from held import WAIT_SECONDS, AsyncHeldClassification, HeldClassification

pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from silmaril_security.sdk import AsyncFirewall, Firewall, HookLabel  # noqa: E402
from silmaril_security.sdk.deepagents import (  # noqa: E402
    SAFE_FINAL_MESSAGE,
    SAFE_OUTPUT_MESSAGE,
    _with_protection,
    create_deepagents_middleware,
    create_protected_deep_agent,
)

URL = "https://example.com/classify"


def _request(messages, *, args=None):
    state = {"messages": messages}
    return SimpleNamespace(
        messages=messages,
        state=state,
        runtime=SimpleNamespace(config={"configurable": {"thread_id": "conversation-1"}}),
        tool_call={"id": "call-1", "name": "search", "args": args or {"query": "safe"}},
    )


def _firewall(monkeypatch, held: HeldClassification, **options) -> Firewall:
    fw = Firewall(api_key="sk", api_url=URL, **options)
    monkeypatch.setattr(fw, "_classify_raw", held)
    return fw


def test_model_runs_while_its_input_is_classified(monkeypatch):
    held = HeldClassification()
    middleware = create_deepagents_middleware(_firewall(monkeypatch, held), parallel=True)

    def model(_):
        assert held.started.wait(WAIT_SECONDS)
        held.release.set()
        return AIMessage("answer")

    response = middleware.wrap_model_call(_request([HumanMessage("question")]), model)

    assert response.content == "answer"
    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]
    assert all(options["metadata"]["conversationId"] == "conversation-1" for _, options in held.calls)


def test_blocked_input_withholds_the_model_response(monkeypatch):
    held = HeldClassification("MALICIOUS")
    held.release.set()
    middleware = create_deepagents_middleware(_firewall(monkeypatch, held), parallel=True)

    response = middleware.wrap_model_call(_request([HumanMessage("attack")]), lambda _: AIMessage("leak"))

    assert response.content == SAFE_OUTPUT_MESSAGE
    assert held.hooks == [HookLabel.USER_INPUT]


def test_blocked_attempt_cap_is_decided_before_the_model_runs(monkeypatch):
    held = HeldClassification("MALICIOUS", "BENIGN")
    held.release.set()
    middleware = create_deepagents_middleware(
        _firewall(monkeypatch, held), parallel=True, max_blocked_attempts=1
    )
    denied = middleware.wrap_tool_call(
        _request([], args={"query": "attack"}),
        lambda _: ToolMessage("unreachable", tool_call_id="call-1"),
    )
    called = []

    response = middleware.wrap_model_call(
        _request([HumanMessage("question"), denied]), lambda _: called.append("model")
    )

    assert response.content == SAFE_FINAL_MESSAGE
    assert called == []


def test_shadow_classifies_in_the_background(monkeypatch):
    held = HeldClassification("MALICIOUS")
    fw = _firewall(monkeypatch, held, mode="shadow")
    middleware = create_deepagents_middleware(fw)

    response = middleware.wrap_model_call(_request([HumanMessage("attack")]), lambda _: AIMessage("answer"))

    assert response.content == "answer"
    held.release.set()
    fw.flush()
    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]


def test_existing_middleware_must_match_the_requested_parallel_setting(monkeypatch):
    fw = _firewall(monkeypatch, HeldClassification())

    with pytest.raises(ValueError, match="parallel"):
        _with_protection([create_deepagents_middleware(fw)], fw, {"parallel": True})


def test_protected_agent_generates_while_its_input_is_classified(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held)

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

        def _generate(self, *args, **kwargs):
            assert held.started.wait(WAIT_SECONDS)
            held.release.set()
            return super()._generate(*args, **kwargs)

    agent = create_protected_deep_agent(
        fw, model=Model(responses=[AIMessage(content="done")]), middleware_options={"parallel": True}
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "question"}]},
        config={"configurable": {"thread_id": "conversation-1"}},
    )

    assert result["messages"][-1].content == "done"
    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]


@pytest.mark.asyncio
async def test_async_block_cancels_the_model_call(monkeypatch):
    held = AsyncHeldClassification("MALICIOUS")
    held.release.set()
    fw = AsyncFirewall(api_key="sk", api_url=URL)
    monkeypatch.setattr(fw, "_classify_raw", held)
    middleware = create_deepagents_middleware(fw, parallel=True)
    cancelled = asyncio.Event()

    async def model(_):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    response = await middleware.awrap_model_call(_request([HumanMessage("attack")]), model)

    assert response.content == SAFE_OUTPUT_MESSAGE
    await asyncio.wait_for(cancelled.wait(), WAIT_SECONDS)


@pytest.mark.asyncio
async def test_async_execution_with_a_sync_client_still_runs_in_parallel(monkeypatch):
    held = HeldClassification()
    middleware = create_deepagents_middleware(_firewall(monkeypatch, held), parallel=True)

    async def model(_):
        assert await asyncio.to_thread(held.started.wait, WAIT_SECONDS)
        held.release.set()
        return AIMessage("answer")

    response = await middleware.awrap_model_call(_request([HumanMessage("question")]), model)

    assert response.content == "answer"
