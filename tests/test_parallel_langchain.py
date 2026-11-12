# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import pytest
from held import WAIT_SECONDS, AsyncHeldClassification, HeldClassification

from silmaril_security.sdk import AsyncFirewall, Firewall, FirewallBlockedException, HookLabel

pytest.importorskip("langchain_core.callbacks")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402

from silmaril_security.sdk.langchain import protect_langchain_runnable  # noqa: E402

URL = "https://api.test.invalid/classify"
USER = [[{"role": "user", "content": "question"}]]
ANSWER = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="answer"))]])


def _firewall(monkeypatch, held: HeldClassification, **options) -> Firewall:
    fw = Firewall(api_key="sk", api_url=URL, **options)
    monkeypatch.setattr(fw, "_classify_raw", held)
    return fw


def test_model_output_is_classified_by_default(monkeypatch):
    held = HeldClassification()
    held.release.set()
    handler = _firewall(monkeypatch, held).as_langchain_handler()
    run_id = uuid4()

    handler.on_chat_model_start(serialized={}, messages=USER, run_id=run_id)
    handler.on_llm_end(ANSWER, run_id=run_id)

    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]


def test_parallel_input_check_is_joined_when_the_model_ends(monkeypatch):
    held = HeldClassification("MALICIOUS")
    handler = _firewall(monkeypatch, held).as_langchain_handler(parallel=True)
    run_id = uuid4()

    handler.on_chat_model_start(serialized={}, messages=USER, run_id=run_id)
    assert held.started.wait(WAIT_SECONDS)
    handler.on_llm_new_token("partial", run_id=run_id)

    held.release.set()
    with pytest.raises(FirewallBlockedException) as blocked:
        handler.on_llm_end(ANSWER, run_id=run_id)

    assert blocked.value.hook == HookLabel.USER_INPUT
    assert held.hooks == [HookLabel.USER_INPUT]


def test_parallel_block_surfaces_at_the_next_hook_once_it_has_landed(monkeypatch):
    held = HeldClassification("MALICIOUS")
    held.release.set()
    fw = _firewall(monkeypatch, held)
    handler = fw.as_langchain_handler(parallel=True)
    run_id = uuid4()

    handler.on_chat_model_start(serialized={}, messages=USER, run_id=run_id)
    fw.flush()

    with pytest.raises(FirewallBlockedException):
        handler.on_llm_new_token("partial", run_id=run_id)


def test_parallel_tool_result_is_joined_before_the_next_tool_runs(monkeypatch):
    held = HeldClassification("BENIGN", "MALICIOUS")
    held.release.set()
    handler = _firewall(monkeypatch, held).as_langchain_handler(parallel=True)
    graph, first_tool, second_tool = uuid4(), uuid4(), uuid4()

    handler.on_chain_start({}, {}, run_id=graph)
    handler.on_tool_start({"name": "search"}, "query", run_id=first_tool, parent_run_id=graph)
    handler.on_tool_end("poisoned result", run_id=first_tool, parent_run_id=graph, name="search")

    with pytest.raises(FirewallBlockedException) as blocked:
        handler.on_tool_start({"name": "send_email"}, "body", run_id=second_tool, parent_run_id=graph)

    assert blocked.value.hook == HookLabel.TOOL_RESPONSE
    assert held.hooks == [HookLabel.TOOL_CALL, HookLabel.TOOL_RESPONSE]


def test_parallel_tool_call_still_waits_for_its_verdict(monkeypatch):
    held = HeldClassification("MALICIOUS")
    held.release.set()
    handler = _firewall(monkeypatch, held).as_langchain_handler(parallel=True)

    with pytest.raises(FirewallBlockedException) as blocked:
        handler.on_tool_start({"name": "delete_records"}, "all", run_id=uuid4())

    assert blocked.value.hook == HookLabel.TOOL_CALL


def test_parallel_tool_result_is_joined_at_the_next_model_end(monkeypatch):
    held = HeldClassification("BENIGN", "MALICIOUS")
    held.release.set()
    fw = _firewall(monkeypatch, held)
    handler = fw.as_langchain_handler(parallel=True)
    graph, tool, model = uuid4(), uuid4(), uuid4()
    thread = {"thread_id": "conversation-1", "langgraph_step": 1}

    handler.on_chain_start({}, {}, run_id=graph, metadata=thread)
    handler.on_tool_start({"name": "search"}, "query", run_id=tool, parent_run_id=graph)
    handler.on_tool_end("poisoned result", run_id=tool, parent_run_id=graph, name="search")
    handler.on_chat_model_start(serialized={}, messages=USER, run_id=model, parent_run_id=graph)

    with pytest.raises(FirewallBlockedException) as blocked:
        handler.on_llm_end(ANSWER, run_id=model, parent_run_id=graph)

    fw.flush()
    assert blocked.value.hook == HookLabel.TOOL_RESPONSE
    assert held.hooks == [HookLabel.TOOL_CALL, HookLabel.TOOL_RESPONSE, HookLabel.USER_INPUT]
    assert all(options["metadata"]["conversationId"] == "conversation-1" for _, options in held.calls)


def test_parallel_block_stays_visible_to_sibling_callbacks_until_the_run_ends(monkeypatch):
    held = HeldClassification("MALICIOUS")
    fw = _firewall(monkeypatch, held)
    handler = fw.as_langchain_handler(parallel=True)
    graph, model, tool = uuid4(), uuid4(), uuid4()
    handler.on_chain_start({}, {}, run_id=graph)
    handler.on_chat_model_start(serialized={}, messages=USER, run_id=model, parent_run_id=graph)
    assert held.started.wait(WAIT_SECONDS)
    joined: list[BaseException] = []

    def model_end() -> None:
        try:
            handler.on_llm_end(ANSWER, run_id=model, parent_run_id=graph)
        except FirewallBlockedException as blocked:
            joined.append(blocked)

    joining = threading.Thread(target=model_end)
    joining.start()
    held.release.set()
    with pytest.raises(FirewallBlockedException) as sibling:
        handler.on_tool_start({"name": "send_email"}, "body", run_id=tool, parent_run_id=graph)
    joining.join(WAIT_SECONDS)

    assert sibling.value.hook == HookLabel.USER_INPUT
    assert [blocked.hook for blocked in joined] == [HookLabel.USER_INPUT]
    assert held.hooks == [HookLabel.USER_INPUT]
    handler.on_chain_error(RuntimeError("run failed"), run_id=graph)
    assert handler._deferred.pending(str(graph)) == []


def test_failed_run_leaves_no_deferred_verdicts(monkeypatch):
    held = HeldClassification("MALICIOUS")
    held.release.set()
    fw = _firewall(monkeypatch, held)
    handler = fw.as_langchain_handler(parallel=True)
    run_id = uuid4()

    handler.on_chat_model_start(serialized={}, messages=USER, run_id=run_id)
    handler.on_llm_error(RuntimeError("provider failed"), run_id=run_id)
    fw.flush()

    assert handler._deferred.take(str(run_id)) == []


def test_shadow_classifies_in_the_background_unless_failures_must_surface(monkeypatch):
    held = HeldClassification("MALICIOUS")
    fw = _firewall(monkeypatch, held, mode="shadow")
    run_id = uuid4()

    fw.as_langchain_handler().on_chat_model_start(serialized={}, messages=USER, run_id=run_id)
    assert held.started.wait(WAIT_SECONDS)
    held.release.set()
    fw.flush()

    fw.as_langchain_handler(fail_open=False).on_chat_model_start(
        serialized={}, messages=USER, run_id=run_id
    )
    assert len(held.calls) == 2


def test_protected_model_generates_while_its_input_is_classified(monkeypatch):
    held = HeldClassification()
    fw = _firewall(monkeypatch, held)

    class Model(FakeMessagesListChatModel):
        def _generate(self, *args, **kwargs):
            assert held.started.wait(WAIT_SECONDS)
            held.release.set()
            return super()._generate(*args, **kwargs)

    protected = protect_langchain_runnable(fw, Model(responses=[AIMessage(content="done")]), parallel=True)

    result = protected.invoke("question", config={"configurable": {"thread_id": "conversation-1"}})

    assert result.content == "done"
    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]
    assert all(options["metadata"]["conversationId"] == "conversation-1" for _, options in held.calls)


def test_protected_model_withholds_output_for_a_blocked_input(monkeypatch):
    held = HeldClassification("MALICIOUS")
    held.release.set()
    fw = _firewall(monkeypatch, held)
    protected = protect_langchain_runnable(
        fw, FakeMessagesListChatModel(responses=[AIMessage(content="done")]), parallel=True
    )

    with pytest.raises(FirewallBlockedException):
        protected.invoke("attack")


@pytest.mark.asyncio
async def test_async_parallel_input_check_is_joined_when_the_model_ends(monkeypatch):
    held = AsyncHeldClassification("MALICIOUS")
    fw = AsyncFirewall(api_key="sk", api_url=URL)
    monkeypatch.setattr(fw, "_classify_raw", held)
    handler = fw.as_async_langchain_handler(parallel=True)
    run_id = uuid4()

    await handler.on_chat_model_start(serialized={}, messages=USER, run_id=run_id)
    await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
    await handler.on_llm_new_token("partial", run_id=run_id)

    held.release.set()
    with pytest.raises(FirewallBlockedException) as blocked:
        await handler.on_llm_end(ANSWER, run_id=run_id)

    assert blocked.value.hook == HookLabel.USER_INPUT
    await fw.flush()


@pytest.mark.asyncio
async def test_async_protected_model_generates_while_its_input_is_classified(monkeypatch):
    held = AsyncHeldClassification()
    fw = AsyncFirewall(api_key="sk", api_url=URL)
    monkeypatch.setattr(fw, "_classify_raw", held)

    class Model(FakeMessagesListChatModel):
        async def _agenerate(self, *args, **kwargs):
            await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)
            held.release.set()
            return await super()._agenerate(*args, **kwargs)

    protected = protect_langchain_runnable(fw, Model(responses=[AIMessage(content="done")]), parallel=True)

    result = await protected.ainvoke("question")

    assert result.content == "done"
    assert held.hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]
    await fw.flush()


@pytest.mark.asyncio
async def test_async_handler_flush_waits_for_background_checks_of_a_sync_client(monkeypatch):
    held = AsyncHeldClassification("MALICIOUS")
    events = []
    fw = Firewall(api_key="sk", api_url=URL, mode="shadow")

    async def classify(firewall, text, **options):
        return await held(text, **options)

    monkeypatch.setattr("silmaril_security.sdk.langchain._async_classify_raw", classify)
    handler = fw.as_async_langchain_handler(on_classify=events.append)

    await handler.on_chat_model_start(serialized={}, messages=USER, run_id=uuid4())
    assert events == []

    held.release.set()
    await handler.flush()
    assert [event.blocked for event in events] == [True]


@pytest.mark.asyncio
async def test_async_sibling_callbacks_join_the_same_deferred_block(monkeypatch):
    held = AsyncHeldClassification("MALICIOUS")
    fw = AsyncFirewall(api_key="sk", api_url=URL)
    monkeypatch.setattr(fw, "_classify_raw", held)
    handler = fw.as_async_langchain_handler(parallel=True)
    graph, model, first_tool, second_tool = uuid4(), uuid4(), uuid4(), uuid4()
    await handler.on_chain_start({}, {}, run_id=graph)
    await handler.on_chat_model_start(serialized={}, messages=USER, run_id=model, parent_run_id=graph)
    await asyncio.wait_for(held.started.wait(), WAIT_SECONDS)

    joining = asyncio.gather(
        handler.on_llm_end(ANSWER, run_id=model, parent_run_id=graph),
        handler.on_tool_start({"name": "send_email"}, "body", run_id=first_tool, parent_run_id=graph),
        handler.on_tool_start({"name": "delete_records"}, "all", run_id=second_tool, parent_run_id=graph),
        return_exceptions=True,
    )
    for _ in range(5):
        await asyncio.sleep(0)
    held.release.set()
    outcomes = await joining

    assert [type(outcome) for outcome in outcomes] == [FirewallBlockedException] * 3
    assert {outcome.hook for outcome in outcomes} == {HookLabel.USER_INPUT}
    assert held.hooks == [HookLabel.USER_INPUT]
    await handler.on_chain_error(RuntimeError("run failed"), run_id=graph)
    assert handler._deferred.pending(str(graph)) == []
    await fw.flush()


def test_many_sibling_threads_join_the_same_deferred_block(monkeypatch):
    held = HeldClassification("MALICIOUS")
    fw = _firewall(monkeypatch, held)
    handler = fw.as_langchain_handler(parallel=True)
    graph, model = uuid4(), uuid4()
    handler.on_chain_start({}, {}, run_id=graph)
    handler.on_chat_model_start(serialized={}, messages=USER, run_id=model, parent_run_id=graph)
    assert held.started.wait(WAIT_SECONDS)
    ready = threading.Barrier(9, timeout=WAIT_SECONDS)
    outcomes: list[BaseException | None] = []

    def tool_start() -> None:
        ready.wait()
        try:
            handler.on_tool_start({"name": "send_email"}, "body", run_id=uuid4(), parent_run_id=graph)
        except FirewallBlockedException as blocked:
            outcomes.append(blocked)
        else:
            outcomes.append(None)

    siblings = [threading.Thread(target=tool_start) for _ in range(8)]
    for sibling in siblings:
        sibling.start()
    ready.wait()
    held.release.set()
    for sibling in siblings:
        sibling.join(WAIT_SECONDS)

    assert len(outcomes) == 8
    assert all(isinstance(outcome, FirewallBlockedException) for outcome in outcomes)
    assert held.hooks == [HookLabel.USER_INPUT]
