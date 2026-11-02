"""Thread identity contracts exercised through real LangChain/LangGraph execution."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import tool

from silmaril_security.sdk import ALL_HOOKS, AsyncFirewall, BlockResult, Firewall, FirewallHook
from silmaril_security.sdk._thread_identity import thread_identity


class EchoModel(FakeMessagesListChatModel):
    responses: list = [AIMessage(content="unused")]

    def bind_tools(self, *args, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, HumanMessage) and last.content.startswith("tool:"):
            message = AIMessage(content="", tool_calls=[{
                "name": "echo", "args": {"value": last.content}, "id": str(uuid4()),
            }])
        else:
            message = AIMessage(content=str(last.content))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(0)
        return self._generate(messages, stop, run_manager, **kwargs)


class Retriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content=query)]


@tool
def echo(value: str) -> str:
    """Return the supplied value."""
    return value


def firewall(monkeypatch, *, asynchronous=False):
    fw = (AsyncFirewall if asynchronous else Firewall)(api_key="test", api_url="https://example.invalid/classify")
    captured = []

    def record(text, **kwargs):
        captured.append((text, kwargs))
        return BlockResult("BENIGN", 0.1, 0.5, mode="block")

    async def arecord(text, **kwargs):
        await asyncio.sleep(0)
        return record(text, **kwargs)

    monkeypatch.setattr(fw, "_classify_raw", arecord if asynchronous else record)
    return fw, captured


def test_native_config_wins_and_preserves_exact_id():
    token = var_child_runnable_config.set({"configurable": {"thread_id": " exact "}})
    try:
        assert thread_identity(metadata={"thread_id": "other"}) == (True, " exact ")
        assert thread_identity({"configurable": {"thread_id": "request"}}) == (True, "request")
        assert thread_identity({"configurable": {"thread_id": " "}}, {"thread_id": "other"}) == (True, None)
    finally:
        var_child_runnable_config.reset(token)


@pytest.mark.parametrize("identity", [None, "", "   ", 123, [], {}])
def test_invalid_native_identity_is_unavailable(identity):
    assert thread_identity({"configurable": {"thread_id": identity}}) == (True, None)


def test_protected_runnables_reused_with_nested_tools_retrievers_and_threadless(monkeypatch):
    fw, seen = firewall(monkeypatch)
    model = EchoModel()
    retriever = Retriever()

    def work(value):
        result = protected_echo.invoke({"value": value})
        retriever.invoke(result)
        return model.invoke(value).content

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    protected_echo = protect_langchain_runnable(fw, echo, hooks=ALL_HOOKS)
    runnable = RunnableLambda(work)
    model = protect_langchain_runnable(fw, model, hooks=ALL_HOOKS)
    retriever = protect_langchain_runnable(fw, retriever, hooks=ALL_HOOKS)
    for thread in ["thread-a", "thread-a", "thread-b", None]:
        before = len(seen)
        config = {}
        if thread is not None:
            config["configurable"] = {"thread_id": thread}
        runnable.invoke("safe", config=config)
        events = seen[before:]
        assert len(events) >= 6
        assert all(event[1]["metadata"].get("conversationId") == thread for event in events)
        assert len({event[1]["request_id"] for event in events}) == len(events)


def test_disabled_start_classification_still_supplies_end_identity(monkeypatch):
    fw, seen = firewall(monkeypatch)
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    hooks = {FirewallHook.TOOL_END, FirewallHook.LLM_END, FirewallHook.RETRIEVER_END}
    for runnable, value in [(echo, {"value": "safe"}), (Retriever(), "safe"), (EchoModel(), "safe")]:
        protected = protect_langchain_runnable(fw, runnable, hooks=hooks)
        protected.invoke(value, config={"configurable": {"thread_id": "ends"}})
    assert len(seen) == 3
    assert all(event[1]["metadata"]["conversationId"] == "ends" for event in seen)


def test_concurrent_sync_protected_runnable_threads_do_not_mix(monkeypatch):
    fw, seen = firewall(monkeypatch)
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    model = protect_langchain_runnable(fw, EchoModel(), hooks=ALL_HOOKS)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name: model.invoke(name, config={
            "configurable": {"thread_id": name},
        }), ["thread-a", "thread-b", "thread-c", "thread-d"]))
    assert len(seen) == 8
    assert all(text == kwargs["metadata"]["conversationId"] for text, kwargs in seen)


@pytest.mark.asyncio
async def test_concurrent_async_protected_runnable_and_threadless_reuse(monkeypatch):
    fw, seen = firewall(monkeypatch, asynchronous=True)
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    model = protect_langchain_runnable(fw, EchoModel(), hooks=ALL_HOOKS)
    await asyncio.gather(*(model.ainvoke(name, config={
        "configurable": {"thread_id": name},
    }) for name in ["thread-a", "thread-b", "thread-c"]))
    assert len(seen) == 6
    assert all(text == kwargs["metadata"]["conversationId"] for text, kwargs in seen)
    seen.clear()
    await model.ainvoke("threadless")
    assert all("conversationId" not in kwargs["metadata"] for _, kwargs in seen)
    await fw.aclose()


def test_parent_inheritance_child_override_and_root_error_cleanup(monkeypatch):
    fw, seen = firewall(monkeypatch)
    handler = fw.as_langchain_handler(hooks=ALL_HOOKS)
    root, child, grandchild = uuid4(), uuid4(), uuid4()
    handler.on_chain_start({}, {}, run_id=root, metadata={"thread_id": "parent", "langgraph_step": 0})
    handler.on_chain_start({}, {}, run_id=child, parent_run_id=root)
    handler.on_tool_start({"name": "echo"}, "safe", run_id=grandchild, parent_run_id=child, metadata={"thread_id": "child", "langgraph_node": "child"})
    handler.on_tool_end("safe", run_id=grandchild)
    assert all(kwargs["metadata"]["conversationId"] == "child" for _, kwargs in seen)
    handler.on_chain_error(error=RuntimeError("failed"), run_id=root)
    assert handler._thread_runs._runs == {}
    assert handler._thread_runs._children == {}
    handler.on_tool_end("threadless", run_id=grandchild)
    assert "conversationId" not in seen[-1][1]["metadata"]


def test_real_callback_failure_and_block_cleanup(monkeypatch):
    fw, _ = firewall(monkeypatch)
    handler = fw.as_langchain_handler(fail_open=False)

    def broken(text, **kwargs):
        raise RuntimeError("classifier failed")

    monkeypatch.setattr(fw, "_classify_raw", broken)
    with pytest.raises(RuntimeError, match="classifier failed"):
        EchoModel().invoke("safe", config={"callbacks": [handler], "configurable": {"thread_id": "broken"}})
    assert handler._thread_runs._runs == {}
    monkeypatch.setattr(fw, "_classify_raw", lambda *a, **k: BlockResult("MALICIOUS", 0.9, 0.5, mode="block"))
    with pytest.raises(Exception, match="blocked"):
        EchoModel().invoke("safe", config={"callbacks": [handler], "configurable": {"thread_id": "blocked"}})
    assert handler._thread_runs._runs == {}


def test_removed_constructor_options_rejected(monkeypatch):
    fw, _ = firewall(monkeypatch)
    for make in [fw.as_langchain_handler, fw.as_async_langchain_handler]:
        for key in ["conversation_id", "conversationId"]:
            with pytest.raises(TypeError, match="configurable.thread_id"):
                make(**{key: "legacy"})


def test_real_graph_callbacks_checkpoint_continuity(monkeypatch):
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    fw, seen = firewall(monkeypatch)
    handler = fw.as_langchain_handler(hooks=ALL_HOOKS)
    graph = StateGraph(MessagesState)
    graph.add_node("model", lambda state: {"messages": [EchoModel().invoke(state["messages"])]})
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    agent = graph.compile(checkpointer=InMemorySaver())
    config = {"callbacks": [handler], "configurable": {"thread_id": "persistent"}, "metadata": {"thread_id": "conflicting-metadata"}}
    agent.invoke({"messages": [HumanMessage("first")]}, config=config)
    result = agent.invoke({"messages": [HumanMessage("second")]}, config=config)
    assert len(result["messages"]) == 4
    assert all(kwargs["metadata"]["conversationId"] == "persistent" for _, kwargs in seen)
    assert handler._thread_runs._runs == {}


@pytest.mark.parametrize("subagent_name", [None, "general-purpose", "research", "compiled", "nested"])
def test_deepagents_native_thread_at_all_boundaries(monkeypatch, subagent_name):
    pytest.importorskip("deepagents")
    from silmaril_security.sdk.deepagents import (
        create_protected_compiled_subagent,
        create_protected_deep_agent,
    )

    fw, seen = firewall(monkeypatch)
    specs = []
    compiled = []
    if subagent_name:
        submodel = FakeMessagesListChatModel

        class Model(submodel):
            def bind_tools(self, *args, **kwargs):
                return self

        inner = Model(responses=[AIMessage("subagent-result")])
        if subagent_name == "nested":
            nested = create_protected_deep_agent(fw, model=inner)
            specs = [{"name": "nested", "description": "Nested", "runnable": nested}]
            # The public constructor intentionally rejects arbitrary compiled graphs.
            # A protected compiled subagent may contain a nested protected agent as a tool.
            @tool
            def nested_work() -> str:
                """Delegate work to a nested protected graph."""
                return nested.invoke({"messages": [HumanMessage("nested-request")]})["messages"][-1].content

            outer = Model(responses=[AIMessage("", tool_calls=[{"name": "nested_work", "args": {}, "id": "inner-call"}]), AIMessage("nested-result")])
            compiled = [create_protected_compiled_subagent(fw, name="nested", description="Nested", model=outer, tools=[nested_work])]
            specs = []
        elif subagent_name == "compiled":
            compiled = [create_protected_compiled_subagent(fw, name="compiled", description="Compiled", model=inner)]
        else:
            specs = [{"name": subagent_name, "description": "Research", "model": inner}]
        model = Model(responses=[AIMessage("", tool_calls=[{"name": "task", "args": {"description": "safe-task", "subagent_type": subagent_name}, "id": "task-call"}]), AIMessage("root-result")])
    else:
        model = EchoModel()
    agent = create_protected_deep_agent(fw, model=model, tools=[echo], subagents=specs, protected_compiled_subagents=compiled)
    for thread in ["deep-a", "deep-b", None]:
        seen.clear()
        config = {"configurable": {"thread_id": thread}} if thread else {}
        agent.invoke({"messages": [HumanMessage("tool:request")]}, config=config)
        assert len(seen) >= 4
        assert all(kwargs["metadata"].get("conversationId") == thread for _, kwargs in seen)
        if subagent_name:
            assert any(text == "subagent-result" or text == "nested-request" for text, _ in seen)


@pytest.mark.asyncio
async def test_deepagents_concurrent_async_and_checkpoint_resume(monkeypatch):
    pytest.importorskip("deepagents")
    from langgraph.checkpoint.memory import InMemorySaver

    from silmaril_security.sdk.deepagents import create_protected_deep_agent

    fw, seen = firewall(monkeypatch, asynchronous=True)
    agent = create_protected_deep_agent(fw, model=EchoModel(), tools=[echo], checkpointer=InMemorySaver())
    await asyncio.gather(*(agent.ainvoke({"messages": [HumanMessage("tool:" + name)]}, config={"configurable": {"thread_id": name}}) for name in ["deep-a", "deep-b"]))
    assert len(seen) == 10
    assert all(kwargs["metadata"]["conversationId"] in text for text, kwargs in seen)
    seen.clear()
    result = await agent.ainvoke({"messages": [HumanMessage("followup")]}, config={"configurable": {"thread_id": "deep-a"}})
    assert len(result["messages"]) == 6
    assert all(kwargs["metadata"]["conversationId"] == "deep-a" for _, kwargs in seen)
    await fw.aclose()


def test_deepagents_removed_options_and_explicit_runtime_precedence(monkeypatch):
    pytest.importorskip("deepagents")
    from silmaril_security.sdk.deepagents import (
        create_deepagents_middleware,
        create_protected_compiled_subagent,
        create_protected_deep_agent,
    )

    fw, seen = firewall(monkeypatch)
    for make in [create_deepagents_middleware, create_protected_deep_agent, create_protected_compiled_subagent]:
        options = {"conversation_id": "old"}
        kwargs = options if make is create_deepagents_middleware else {"middleware_options": options, "model": EchoModel()}
        if make is create_protected_compiled_subagent:
            kwargs.update(name="compiled", description="Compiled")
        with pytest.raises(TypeError, match="configurable.thread_id"):
            make(fw, **kwargs)
    middleware = create_deepagents_middleware(fw)
    request = SimpleNamespace(runtime=SimpleNamespace(config={"configurable": {"thread_id": "request"}}))
    token = var_child_runnable_config.set({"configurable": {"thread_id": "ambient"}})
    try:
        assert middleware._metadata(request)["conversationId"] == "request"
    finally:
        var_child_runnable_config.reset(token)


def test_bare_root_callback_uses_only_framework_exposed_identity(monkeypatch):
    from silmaril_security.sdk.langchain import SilmarilFirewallHandler

    exposed = []

    class Observer(SilmarilFirewallHandler):
        def on_chat_model_start(self, *args, **kwargs):
            exposed.append(thread_identity(metadata=kwargs.get("metadata"))[1])
            return super().on_chat_model_start(*args, **kwargs)

    fw, seen = firewall(monkeypatch)
    handler = Observer(fw, hooks=ALL_HOOKS)
    EchoModel().invoke("bare", config={"callbacks": [handler], "configurable": {"thread_id": "bare-root"}})
    # Recent core versions reserve configurable metadata for LangSmith tracers
    # on a bare root call. Ordinary callbacks must omit unavailable identity.
    assert all(kwargs["metadata"].get("conversationId") == exposed[0] for _, kwargs in seen)
    assert exposed[0] in {None, "bare-root"}
    assert handler._thread_runs._runs == {}


@pytest.mark.asyncio
async def test_async_typed_error_callbacks_release_all_state(monkeypatch):
    fw, _ = firewall(monkeypatch, asynchronous=True)
    handler = fw.as_async_langchain_handler()
    for name in ["chain", "llm", "tool", "retriever"]:
        run_id = uuid4()
        await handler.on_chain_start({}, {}, run_id=run_id, metadata={"thread_id": "errors"})
        await getattr(handler, "on_" + name + "_error")(error=RuntimeError("failed"), run_id=run_id)
        assert handler._thread_runs._runs == {}
    await fw.aclose()


def test_protected_standalone_model_exact_child_override_and_caller_config(monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    called = []

    class CallerHandler(BaseCallbackHandler):
        def on_chat_model_start(self, *args, **kwargs):
            called.append(kwargs.get("metadata"))

    protected = protect_langchain_runnable(fw, EchoModel(), hooks=ALL_HOOKS)
    runnable = RunnableLambda(lambda value: protected.invoke(value, config={
        "configurable": {"thread_id": "child-b"},
        "metadata": {"thread_id": "wrong"},
    }))
    runnable.invoke("child", config={
        "configurable": {"thread_id": "parent-a"}, "callbacks": [CallerHandler()],
        "metadata": {"application": "preserved"},
    })
    assert seen and all(kwargs["metadata"]["conversationId"] == "child-b" for _, kwargs in seen)
    assert len(called) == 1 and called[0]["application"] == "preserved"
    seen.clear()
    protected.invoke("threadless")
    assert seen and all("conversationId" not in kwargs["metadata"] for _, kwargs in seen)


def test_protected_standalone_batch_and_stream_preserve_native_threads(monkeypatch):
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    protected = protect_langchain_runnable(fw, EchoModel(), hooks=ALL_HOOKS)
    protected.batch(["batch-a", "batch-b"], config=[
        {"configurable": {"thread_id": "batch-a"}},
        {"configurable": {"thread_id": "batch-b"}},
    ])
    assert len(seen) == 4
    assert all(text == kwargs["metadata"]["conversationId"] for text, kwargs in seen)
    seen.clear()
    assert list(protected.stream("stream", config={"configurable": {"thread_id": "stream"}}))
    assert seen and all(kwargs["metadata"]["conversationId"] == "stream" for _, kwargs in seen)


@pytest.mark.asyncio
async def test_protected_standalone_async_batch_stream_and_tool(monkeypatch):
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch, asynchronous=True)
    protected = protect_langchain_runnable(fw, EchoModel(), hooks=ALL_HOOKS)
    await protected.abatch(["async-a", "async-b"], config=[
        {"configurable": {"thread_id": "async-a"}},
        {"configurable": {"thread_id": "async-b"}},
    ])
    assert len(seen) == 4
    assert all(text == kwargs["metadata"]["conversationId"] for text, kwargs in seen)
    seen.clear()
    assert [chunk async for chunk in protected.astream("stream", config={"configurable": {"thread_id": "stream"}})]
    assert seen and all(kwargs["metadata"]["conversationId"] == "stream" for _, kwargs in seen)
    seen.clear()
    protected_tool = protect_langchain_runnable(fw, echo)
    assert await protected_tool.ainvoke({"value": "tool"}, config={"configurable": {"thread_id": "tool-thread"}}) == "tool"
    assert len(seen) == 2
    assert all(kwargs["metadata"]["conversationId"] == "tool-thread" for _, kwargs in seen)
    await fw.aclose()


def test_bare_nested_model_does_not_infer_hidden_child_id_from_parent(monkeypatch):
    fw, seen = firewall(monkeypatch)
    handler = fw.as_langchain_handler(hooks=ALL_HOOKS)
    runnable = RunnableLambda(lambda value: EchoModel().invoke(value, config={"configurable": {"thread_id": "child-b"}}))
    runnable.invoke("child", config={"callbacks": [handler], "configurable": {"thread_id": "parent-a"}})
    # Older core versions expose the actual child ID in callback metadata;
    # newer versions hide it. Neither may incorrectly assign the ambient parent.
    assert seen and all(kwargs["metadata"].get("conversationId") in {None, "child-b"} for _, kwargs in seen)


def test_protected_parent_runnable_preserves_separate_child_graph_thread(monkeypatch):
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, MessagesState, StateGraph

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    graph = StateGraph(MessagesState)
    graph.add_node("model", lambda state: {"messages": [EchoModel().invoke(state["messages"])]})
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    child = graph.compile()
    outer = protect_langchain_runnable(fw, RunnableLambda(lambda text: child.invoke(
        {"messages": [HumanMessage(text)]}, config={"configurable": {"thread_id": "child-graph"}},
    )), hooks=ALL_HOOKS)
    outer.invoke("safe", config={"configurable": {"thread_id": "outer-parent"}})
    assert seen and all(kwargs["metadata"]["conversationId"] == "child-graph" for _, kwargs in seen)


@pytest.mark.parametrize("identity", [None, "", "   ", 123, [], {}])
def test_protected_standalone_invalid_ids_do_not_use_metadata_fallback(monkeypatch, identity):
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    protected = protect_langchain_runnable(fw, EchoModel(), hooks=ALL_HOOKS)
    protected.invoke("invalid", config={
        "configurable": {"thread_id": identity}, "metadata": {"thread_id": "not-native"},
    })
    assert seen and all("conversationId" not in kwargs["metadata"] for _, kwargs in seen)
    with pytest.raises(TypeError, match="configurable.thread_id"):
        protect_langchain_runnable(fw, EchoModel(), conversation_id="old")


def test_protected_model_child_override_inside_native_parent_graph(monkeypatch):
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, MessagesState, StateGraph

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    protected_model = protect_langchain_runnable(fw, EchoModel(), hooks=ALL_HOOKS)
    graph = StateGraph(MessagesState)
    graph.add_node("model", lambda state: {"messages": [protected_model.invoke(
        state["messages"], config={"configurable": {"thread_id": "child-model"}},
    )]})
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    graph.compile().invoke({"messages": [HumanMessage("safe")]}, config={"configurable": {"thread_id": "parent-graph"}})
    assert seen and all(kwargs["metadata"]["conversationId"] == "child-model" for _, kwargs in seen)


def test_callback_bookkeeping_is_bounded_when_framework_omits_terminal_events(monkeypatch):
    from silmaril_security.sdk._thread_identity import MAX_TRACKED_RUNS

    fw, seen = firewall(monkeypatch)
    handler = fw.as_langchain_handler(hooks={FirewallHook.TOOL_END})
    old_root = uuid4()
    handler.on_chain_start({}, {}, run_id=old_root, metadata={"thread_id": "old", "langgraph_step": 0})
    child = uuid4()
    handler.on_tool_start({"name": "echo"}, "safe", run_id=child, parent_run_id=old_root)
    for _ in range(MAX_TRACKED_RUNS + 10):
        handler.on_chain_start({}, {}, run_id=uuid4(), metadata={"thread_id": "new", "langgraph_step": 0})
    assert len(handler._thread_runs._runs) <= MAX_TRACKED_RUNS
    assert len(handler._thread_runs._children) <= MAX_TRACKED_RUNS
    handler.on_tool_end("late", run_id=child)
    assert "conversationId" not in seen[-1][1]["metadata"]


def test_protected_stream_close_clears_own_handler_without_framework_terminal_event(monkeypatch):
    from langchain_core.callbacks import CallbackManager
    from langchain_core.runnables import Runnable

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    retained = []

    class Unfinished(Runnable):
        def invoke(self, input, config=None, **kwargs):
            return input

        def stream(self, input, config=None, **kwargs):
            manager = CallbackManager.configure(inheritable_callbacks=config["callbacks"])
            retained.extend(manager.handlers)
            manager.on_chain_start({}, input)
            yield input
            yield input

    fw, _ = firewall(monkeypatch)
    stream = protect_langchain_runnable(fw, Unfinished()).stream("safe", config={"configurable": {"thread_id": "stream"}})
    assert next(stream) == "safe"
    assert retained[0]._thread_runs._runs
    stream.close()
    assert retained[0]._thread_runs._runs == {}


@pytest.mark.asyncio
async def test_protected_async_stream_close_clears_own_handler(monkeypatch):
    from langchain_core.callbacks import AsyncCallbackManager
    from langchain_core.runnables import Runnable

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    retained = []

    class Unfinished(Runnable):
        def invoke(self, input, config=None, **kwargs):
            return input

        async def astream(self, input, config=None, **kwargs):
            manager = AsyncCallbackManager.configure(inheritable_callbacks=config["callbacks"])
            retained.extend(manager.handlers)
            await manager.on_chain_start({}, input)
            yield input
            yield input

    fw, _ = firewall(monkeypatch, asynchronous=True)
    stream = protect_langchain_runnable(fw, Unfinished()).astream("safe", config={"configurable": {"thread_id": "stream"}})
    assert await anext(stream) == "safe"
    assert retained[0]._thread_runs._runs
    await stream.aclose()
    assert retained[0]._thread_runs._runs == {}
    await fw.aclose()


def test_protected_binding_composition_preserves_listener_and_native_override(monkeypatch):
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    starts = []
    protected = protect_langchain_runnable(fw, EchoModel()).with_config(
        configurable={"thread_id": "default"}, metadata={"application": "preserved"},
    ).with_listeners(on_start=lambda run: starts.append(run.name))
    assert list(protected.stream("safe", config={"configurable": {"thread_id": "override"}}))
    assert starts
    assert seen and all(kwargs["metadata"]["conversationId"] == "override" for _, kwargs in seen)


def test_explicit_child_callback_native_metadata_overrides_parent(monkeypatch):
    fw, seen = firewall(monkeypatch)
    handler = fw.as_langchain_handler()
    root, child = uuid4(), uuid4()
    handler.on_chain_start({}, {}, run_id=root, metadata={"thread_id": "parent"})
    handler.on_tool_start({"name": "echo"}, "safe", run_id=child, parent_run_id=root, metadata={"thread_id": "child"})
    handler.on_tool_end("safe", run_id=child)
    assert all(kwargs["metadata"]["conversationId"] == "child" for _, kwargs in seen)
    handler.on_chain_end({}, run_id=root)
    assert handler._thread_runs._runs == {}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_shared_callback_handler_isolates_concurrent_graph_threads(monkeypatch, asynchronous):
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, MessagesState, StateGraph

    fw, seen = firewall(monkeypatch, asynchronous=asynchronous)
    handler = (fw.as_async_langchain_handler if asynchronous else fw.as_langchain_handler)(hooks=ALL_HOOKS)
    graph = StateGraph(MessagesState)

    def node(state):
        return {"messages": [EchoModel().invoke(state["messages"])]}

    async def anode(state):
        return {"messages": [await EchoModel().ainvoke(state["messages"])]}

    graph.add_node("model", anode if asynchronous else node)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    agent = graph.compile()
    names = ["shared-a", "shared-b", "shared-c"]
    if asynchronous:
        await asyncio.gather(*(agent.ainvoke({"messages": [HumanMessage(name)]}, config={
            "callbacks": [handler], "configurable": {"thread_id": name},
        }) for name in names))
        await fw.aclose()
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda name: agent.invoke({"messages": [HumanMessage(name)]}, config={
                "callbacks": [handler], "configurable": {"thread_id": name},
            }), names))
    assert len(seen) == 6
    assert all(text == kwargs["metadata"]["conversationId"] for text, kwargs in seen)
    assert handler._thread_runs._runs == {}
    assert handler._thread_runs._children == {}


def test_protected_binding_retry_keeps_native_boundary_on_each_attempt(monkeypatch):
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)

    class OnceFailing(EchoModel):
        attempts: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("try again")
            return super()._generate(messages, stop, run_manager, **kwargs)

    model = OnceFailing()
    protected = protect_langchain_runnable(fw, model, hooks=ALL_HOOKS).with_retry(
        wait_exponential_jitter=False, stop_after_attempt=2,
    )
    protected.invoke("safe", config={"configurable": {"thread_id": "retry"}})
    assert model.attempts == 2
    assert seen and all(kwargs["metadata"]["conversationId"] == "retry" for _, kwargs in seen)


@pytest.mark.parametrize("method", ["invoke", "batch", "stream"])
def test_protection_uses_saved_runnable_native_config_before_invocation_overrides(monkeypatch, method):
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    saved = EchoModel().with_config(configurable={"thread_id": "saved-thread"})
    protected = protect_langchain_runnable(fw, saved, hooks=ALL_HOOKS)
    for config, expected in [(None, "saved-thread"), ({"configurable": {"thread_id": "override"}}, "override")]:
        seen.clear()
        if method == "batch":
            protected.batch(["safe"], config=config)
        elif method == "stream":
            assert list(protected.stream("safe", config=config))
        else:
            protected.invoke("safe", config=config)
        assert seen and all(kwargs["metadata"]["conversationId"] == expected for _, kwargs in seen)


@pytest.mark.asyncio
async def test_protection_uses_saved_native_config_for_async_model(monkeypatch):
    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch, asynchronous=True)
    protected = protect_langchain_runnable(fw, EchoModel().with_config(configurable={"thread_id": "saved-thread"}))
    await protected.ainvoke("safe")
    assert seen and all(kwargs["metadata"]["conversationId"] == "saved-thread" for _, kwargs in seen)
    seen.clear()
    await protected.ainvoke("safe", config={"configurable": {"thread_id": "override"}})
    assert seen and all(kwargs["metadata"]["conversationId"] == "override" for _, kwargs in seen)
    await fw.aclose()


def test_saved_native_model_defaults_override_ambient_parent_without_duplicating_callbacks(monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    starts = []

    class Observer(BaseCallbackHandler):
        def on_chat_model_start(self, *args, **kwargs):
            starts.append(kwargs["run_id"])

    protected = protect_langchain_runnable(fw, EchoModel().with_config(
        configurable={"thread_id": "saved-thread"}, callbacks=[Observer()],
    ))
    RunnableLambda(lambda value: protected.invoke(value)).invoke("safe", config={"configurable": {"thread_id": "ambient-parent"}})
    assert seen and all(kwargs["metadata"]["conversationId"] == "saved-thread" for _, kwargs in seen)
    assert len(starts) == 1


@pytest.mark.parametrize("method", ["invoke", "batch", "stream"])
def test_protection_runs_existing_dynamic_config_factories_once_before_native_snapshot(monkeypatch, method):
    from langchain_core.runnables import RunnableBinding

    from silmaril_security.sdk.langchain import protect_langchain_runnable

    fw, seen = firewall(monkeypatch)
    calls = []

    def configure(config):
        calls.append(config)
        return {"configurable": {"thread_id": "routed-" + config["configurable"]["thread_id"]}}

    configured = RunnableBinding(
        bound=EchoModel(), config={"configurable": {"thread_id": "saved"}},
        config_factories=[configure],
    )
    protected = protect_langchain_runnable(fw, configured, hooks=ALL_HOOKS)
    config = {"configurable": {"thread_id": "requested"}}
    if method == "invoke":
        protected.invoke("safe", config=config)
    elif method == "batch":
        protected.batch(["safe"], config=config)
    else:
        assert list(protected.stream("safe", config=config))
    assert len(calls) == 1
    assert seen and all(kwargs["metadata"]["conversationId"] == "routed-requested" for _, kwargs in seen)
