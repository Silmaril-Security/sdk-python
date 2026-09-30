from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from silmaril_security.sdk import AsyncFirewall, BlockResult, Firewall, HookLabel
from silmaril_security.sdk.deepagents import (
    SAFE_FINAL_MESSAGE,
    SAFE_OUTPUT_MESSAGE,
    SAFE_TOOL_MESSAGE,
    create_deepagents_middleware,
    create_protected_compiled_subagent,
    create_protected_deep_agent,
)


def _request(messages=None, *, args=None):
    state = {"messages": messages or []}
    return SimpleNamespace(
        messages=state["messages"], state=state,
        runtime=SimpleNamespace(config={"run_id": "run-1"}),
        tool_call={"id": "call-1", "name": "search", "args": args or {"query": "safe"}},
    )


def _blocked_tool_message(middleware, call_id: str):
    request = _request(args={"query": "deny"})
    request.tool_call["id"] = call_id
    return middleware.wrap_tool_call(request, lambda _: ToolMessage("unreachable", tool_call_id=call_id))


async def _ablocked_tool_message(middleware, call_id: str):
    request = _request(args={"query": "deny"})
    request.tool_call["id"] = call_id

    async def handler(_):
        return ToolMessage("unreachable", tool_call_id=call_id)

    return await middleware.awrap_tool_call(request, handler)


def test_sync_boundaries_continue_without_leaking(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    events = []
    seen = []

    def raw(text, **kwargs):
        seen.append((text, kwargs))
        denied = "deny" in text
        return BlockResult("MALICIOUS" if denied else "BENIGN", 0.9 if denied else 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    middleware = create_deepagents_middleware(fw, on_classify=events.append, max_blocked_attempts=2, conversation_id="conversation-1")
    calls = []
    request = _request([HumanMessage("hello")], args={"query": "deny"})
    result = middleware.wrap_tool_call(request, lambda _: calls.append("ran"))
    assert calls == []
    assert result.content == SAFE_TOOL_MESSAGE and result.tool_call_id == "call-1"
    request.tool_call["args"] = {"query": "safe"}
    result = middleware.wrap_tool_call(request, lambda _: ToolMessage("deny result", tool_call_id="call-1"))
    assert result.content == SAFE_TOOL_MESSAGE
    request.state["messages"] = [result, _blocked_tool_message(middleware, "call-2")]
    assert middleware.wrap_model_call(request, lambda _: AIMessage("unreachable")).content == SAFE_FINAL_MESSAGE
    request.state["messages"] = []
    request.messages = [ToolMessage("safe", tool_call_id="call-1")]
    assert middleware.wrap_model_call(request, lambda _: ModelResponse([AIMessage("deny output")])).result[0].content == SAFE_OUTPUT_MESSAGE
    assert middleware.wrap_model_call(request, lambda _: AIMessage("allowed alternative")).content == "allowed alternative"
    assert all(kwargs["metadata"]["conversationId"] == "conversation-1" for _, kwargs in seen)
    assert len({kwargs["request_id"] for _, kwargs in seen}) == len(seen)
    assert any(event.hook == HookLabel.TOOL_CALL and event.blocked for event in events)


def test_latest_user_is_checked_after_assistant_and_tool_messages(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    seen = []

    def raw(text, **kwargs):
        seen.append((text, kwargs["hook"]))
        return BlockResult("MALICIOUS" if "deny" in text else "BENIGN", 0.9, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    middleware = create_deepagents_middleware(fw)
    request = _request([HumanMessage("deny input"), AIMessage("intermediate"), ToolMessage("safe", tool_call_id="call-1")])
    called = []
    result = middleware.wrap_model_call(request, lambda _: called.append(True))
    assert result.content == SAFE_OUTPUT_MESSAGE
    assert called == []
    assert seen[0] == ("deny input", HookLabel.USER_INPUT)


def test_denial_cap_does_not_reclassify_safe_replacements(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    seen = []

    def raw(text, **kwargs):
        seen.append((text, kwargs["hook"]))
        if kwargs["hook"] == HookLabel.TOOL_CALL:
            return BlockResult("MALICIOUS", 0.9, 0.5, mode="block")
        return BlockResult("BENIGN", 0.1, 0.5, mode="warn")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    middleware = create_deepagents_middleware(fw, max_blocked_attempts=2)
    request = _request([
        _blocked_tool_message(middleware, "call-1"),
        _blocked_tool_message(middleware, "call-2"),
    ])
    assert middleware.wrap_model_call(request, lambda _: AIMessage("allowed")).content == SAFE_FINAL_MESSAGE
    assert len(seen) == 2


def test_denial_cap_resets_on_new_user_turn(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    monkeypatch.setattr(fw, "_classify_raw", lambda text, **kwargs: BlockResult("MALICIOUS" if "deny" in text else "BENIGN", 0.9, 0.5, mode="block"))
    middleware = create_deepagents_middleware(fw, max_blocked_attempts=2)
    history = [
        HumanMessage("first"),
        _blocked_tool_message(middleware, "call-1"),
        _blocked_tool_message(middleware, "call-2"),
        HumanMessage("new safe request"),
        ToolMessage("safe result", tool_call_id="call-3"),
    ]
    request = _request(history)
    assert middleware.wrap_model_call(request, lambda _: AIMessage("allowed")).content == "allowed"


def test_allowed_tool_text_matching_safe_message_does_not_count(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    monkeypatch.setattr(fw, "_classify_raw", lambda text, **kwargs: BlockResult("BENIGN", 0.1, 0.5, mode="block"))
    middleware = create_deepagents_middleware(fw, max_blocked_attempts=2)
    request = _request()
    allowed = middleware.wrap_tool_call(
        request, lambda _: ToolMessage(SAFE_TOOL_MESSAGE, tool_call_id="call-1"),
    )
    history = [HumanMessage("safe input"), allowed, allowed]
    assert middleware.wrap_model_call(_request(history), lambda _: AIMessage("allowed")).content == "allowed"


def test_warn_reports_without_replacing_content(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    monkeypatch.setattr(
        fw, "_classify_raw",
        lambda text, **kwargs: BlockResult("MALICIOUS", 0.9, 0.5, mode="warn"),
    )
    events = []
    middleware = create_deepagents_middleware(fw, mode="warn", on_classify=events.append)
    request = _request([HumanMessage("unsafe")])
    assert middleware.wrap_model_call(request, lambda _: AIMessage("original output")).content == "original output"
    assert events and all(event.blocked and event.mode == "warn" for event in events)


@pytest.mark.parametrize("mode", ["warn", "shadow"])
def test_observation_mode_does_not_apply_denial_cap(monkeypatch, mode):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    monkeypatch.setattr(
        fw, "_classify_raw",
        lambda text, **kwargs: BlockResult("MALICIOUS", 0.9, 0.5, mode=mode),
    )
    middleware = create_deepagents_middleware(fw, mode=mode, max_blocked_attempts=2)
    messages = [
        ToolMessage(SAFE_TOOL_MESSAGE, tool_call_id="call-1"),
        ToolMessage(SAFE_TOOL_MESSAGE, tool_call_id="call-2"),
    ]
    request = _request(messages)
    assert middleware.wrap_model_call(request, lambda _: AIMessage("original output")).content == "original output"


@pytest.mark.asyncio
async def test_async_boundaries(monkeypatch):
    fw = AsyncFirewall(api_key="sk", api_url="https://example.com/classify")

    async def raw(text, **kwargs):
        denied = "deny" in text
        return BlockResult("MALICIOUS" if denied else "BENIGN", 0.9 if denied else 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    middleware = create_deepagents_middleware(fw)
    request = _request([HumanMessage("deny input")])
    called = []

    async def model_handler(_):
        called.append("model")
        return AIMessage("safe")

    assert (await middleware.awrap_model_call(request, model_handler)).content == SAFE_OUTPUT_MESSAGE
    assert called == []
    request.messages = [ToolMessage("safe", tool_call_id="call-1")]

    async def tool_handler(_):
        called.append("tool")
        return ToolMessage("deny result", tool_call_id="call-1")

    result = await middleware.awrap_tool_call(request, tool_handler)
    assert result.content == SAFE_TOOL_MESSAGE and called == ["tool"]
    await fw.aclose()


@pytest.mark.asyncio
async def test_async_latest_user_and_new_turn_cap(monkeypatch):
    fw = AsyncFirewall(api_key="sk", api_url="https://example.com/classify")

    async def raw(text, **kwargs):
        return BlockResult("MALICIOUS" if "deny" in text else "BENIGN", 0.9, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    middleware = create_deepagents_middleware(fw, max_blocked_attempts=2)
    request = _request([HumanMessage("deny input"), AIMessage("intermediate"), ToolMessage("safe", tool_call_id="call-1")])
    called = []

    async def handler(_):
        called.append(True)
        return AIMessage("allowed")

    assert (await middleware.awrap_model_call(request, handler)).content == SAFE_OUTPUT_MESSAGE
    assert called == []
    request.messages = [
        HumanMessage("first"),
        await _ablocked_tool_message(middleware, "call-1"),
        await _ablocked_tool_message(middleware, "call-2"),
        HumanMessage("new safe request"),
        ToolMessage("safe result", tool_call_id="call-3"),
    ]
    request.state["messages"] = request.messages
    assert (await middleware.awrap_model_call(request, handler)).content == "allowed"
    assert called == [True]
    await fw.aclose()


def test_compiled_subagent_requires_protected_runnable():
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.raises(ValueError, match="Compiled subagents"):
        create_protected_deep_agent(fw, subagents=[{"name": "compiled", "runnable": object()}])
    forged = {"name": "compiled", "description": "Forged", "runnable": object()}
    with pytest.raises(ValueError, match="create_protected_compiled_subagent"):
        create_protected_deep_agent(fw, protected_compiled_subagents=[forged])


def test_constructor_covers_root_general_and_declarative(monkeypatch):
    import deepagents

    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    captured = {}

    def make(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(deepagents, "create_deep_agent", make)
    create_protected_deep_agent(
        fw, subagents=[{"name": "research", "description": "research"}],
        model="test-model",
    )
    assert isinstance(captured["middleware"][-1], type(create_deepagents_middleware(fw)))
    specs = {spec["name"]: spec for spec in captured["subagents"]}
    assert specs["general-purpose"]["middleware"]
    assert specs["research"]["middleware"]


def test_compiled_subagent_is_bound_to_factory_and_firewall():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    other = Firewall(api_key="sk", api_url="https://example.com/classify")
    spec = create_protected_compiled_subagent(
        fw, name="compiled", description="Protected", model=Model(responses=[AIMessage("ok")]),
    )
    with pytest.raises(ValueError, match="this Firewall client"):
        create_protected_deep_agent(other, protected_compiled_subagents=[spec])
    with pytest.raises(ValueError, match="this Firewall client"):
        create_protected_deep_agent(fw, protected_compiled_subagents=[{**spec, "runnable": object()}])


def test_graph_continues_to_allowed_tool_after_denial(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import tool

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    fw = Firewall(api_key="sk", api_url="https://example.com/classify")

    def raw(text, **kwargs):
        denied = "deny" in text
        return BlockResult("MALICIOUS" if denied else "BENIGN", 0.9 if denied else 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    called = []

    @tool
    def search(query: str) -> str:
        """Search for a query."""
        called.append(query)
        return "safe result"

    model = Model(responses=[
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "deny"}, "id": "call-1"}]),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "safe"}, "id": "call-2"}]),
        AIMessage(content="done"),
    ])
    agent = create_protected_deep_agent(fw, model=model, tools=[search])
    output = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
    assert called == ["safe"]
    assert [(m.tool_call_id, m.content) for m in output["messages"] if isinstance(m, ToolMessage)] == [
        ("call-1", SAFE_TOOL_MESSAGE), ("call-2", "safe result")]


def test_graph_stops_after_repeated_denied_tool_calls(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import tool

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    monkeypatch.setattr(
        fw, "_classify_raw",
        lambda text, **kwargs: BlockResult("MALICIOUS" if "deny" in text else "BENIGN", 0.9, 0.5, mode="block"),
    )
    called = []

    @tool
    def search(query: str) -> str:
        """Search for a query."""
        called.append(query)
        return "safe result"

    model = Model(responses=[
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "deny first"}, "id": "call-1"}]),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "deny second"}, "id": "call-2"}]),
        AIMessage(content="unreachable"),
    ])
    agent = create_protected_deep_agent(
        fw, model=model, tools=[search], middleware_options={"max_blocked_attempts": 2},
    )
    output = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
    assert called == []
    assert output["messages"][-1].content == SAFE_FINAL_MESSAGE


@pytest.mark.parametrize("subagent_name", ["general-purpose", "research", "compiled"])
def test_subagent_output_is_protected_in_graph(monkeypatch, subagent_name):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    fw = Firewall(api_key="sk", api_url="https://example.com/classify")

    def raw(text, **kwargs):
        denied = "deny" in text
        return BlockResult("MALICIOUS" if denied else "BENIGN", 0.9 if denied else 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    root_responses = [
        AIMessage(content="", tool_calls=[{
            "name": "task", "args": {"description": "Summarize safe text", "subagent_type": subagent_name}, "id": "call-1",
        }]),
        AIMessage(content="done"),
    ]
    compiled_specs = []
    if subagent_name == "compiled":
        compiled_specs = [create_protected_compiled_subagent(
            fw, name="compiled", description="Protected compiled",
            model=Model(responses=[AIMessage(content="deny secret")]),
        )]
    else:
        root_responses.insert(1, AIMessage(content="deny secret"))
    agent = create_protected_deep_agent(
        fw, model=Model(responses=root_responses),
        subagents=[{"name": "research", "description": "Research"}],
        protected_compiled_subagents=compiled_specs,
    )
    output = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
    assert [(m.tool_call_id, m.content) for m in output["messages"] if isinstance(m, ToolMessage)] == [
        ("call-1", SAFE_OUTPUT_MESSAGE)]


@pytest.mark.asyncio
async def test_async_graph_replaces_denied_output(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    fw = AsyncFirewall(api_key="sk", api_url="https://example.com/classify")

    async def raw(text, **kwargs):
        denied = "deny" in text
        return BlockResult("MALICIOUS" if denied else "BENIGN", 0.9 if denied else 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    agent = create_protected_deep_agent(fw, model=Model(responses=[AIMessage(content="deny output")]))
    output = await agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]})
    assert output["messages"][-1].content == SAFE_OUTPUT_MESSAGE
    await fw.aclose()
