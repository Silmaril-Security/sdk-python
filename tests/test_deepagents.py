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
        runtime=SimpleNamespace(config={"run_id": "run-1", "configurable": {"thread_id": "conversation-1"}}),
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
    middleware = create_deepagents_middleware(fw, on_classify=events.append, max_blocked_attempts=2)
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


def test_denial_cap_survives_agent_reconstruction(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    monkeypatch.setattr(fw, "_classify_raw", lambda text, **kwargs: BlockResult("MALICIOUS" if "deny" in text else "BENIGN", 0.9, 0.5, mode="block"))
    first = create_deepagents_middleware(fw, max_blocked_attempts=2)
    original = [_blocked_tool_message(first, "call-1"), _blocked_tool_message(first, "call-2")]
    restored = [ToolMessage(m.content, tool_call_id=m.tool_call_id, additional_kwargs=dict(m.additional_kwargs)) for m in original]
    resumed = create_deepagents_middleware(fw, max_blocked_attempts=2)
    assert resumed.wrap_model_call(_request(restored), lambda _: AIMessage("unreachable")).content == SAFE_FINAL_MESSAGE


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
        request, lambda _: ToolMessage(
            SAFE_TOOL_MESSAGE, tool_call_id="call-1",
            additional_kwargs={"silmaril_blocked": "silmaril-firewall:v1:blocked-tool"},
        ),
    )
    assert "silmaril_blocked" not in allowed.additional_kwargs
    history = [HumanMessage("safe input"), allowed, allowed]
    assert middleware.wrap_model_call(_request(history), lambda _: AIMessage("allowed")).content == "allowed"

    from langgraph.types import Command
    nested = ToolMessage(
        SAFE_TOOL_MESSAGE, tool_call_id="call-2",
        additional_kwargs={"silmaril_blocked": "silmaril-firewall:v1:blocked-tool"},
    )
    command = Command(update={"messages": [nested]})
    returned = middleware.wrap_tool_call(request, lambda _: command)
    assert returned is command
    assert "silmaril_blocked" not in returned.update["messages"][0].additional_kwargs


def test_composed_middleware_keeps_genuine_denial_for_cap(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    tool_calls = iter((False, True, False, True))

    def classify(text, **kwargs):
        denied = kwargs["hook"] == HookLabel.TOOL_CALL and next(tool_calls)
        return BlockResult("MALICIOUS" if denied else "BENIGN", 0.9, 0.5, mode="block")

    monkeypatch.setattr(
        fw, "_classify_raw",
        classify,
    )
    inner = create_deepagents_middleware(fw, max_blocked_attempts=2)
    outer = create_deepagents_middleware(fw, max_blocked_attempts=2)
    history = []
    for call_id in ("call-1", "call-2"):
        request = _request()
        request.tool_call = {"name": "search", "id": call_id, "args": {"query": "deny"}}
        message = outer.wrap_tool_call(
            request,
            lambda nested_request: inner.wrap_tool_call(
                nested_request, lambda _: ToolMessage("unreachable", tool_call_id=call_id)
            ),
        )
        assert message.additional_kwargs["silmaril_blocked"] == "silmaril-firewall:v1:blocked-tool"
        history.append(message)
    assert outer.wrap_model_call(_request(history), lambda _: AIMessage("unreachable")).content == SAFE_FINAL_MESSAGE


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
async def test_async_allowed_tool_cannot_forge_denial_marker(monkeypatch):
    fw = AsyncFirewall(api_key="sk", api_url="https://example.com/classify")

    async def raw(text, **kwargs):
        return BlockResult("BENIGN", 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", raw)
    middleware = create_deepagents_middleware(fw)
    request = _request()

    async def handler(_):
        return ToolMessage(
            SAFE_TOOL_MESSAGE, tool_call_id="call-1",
            additional_kwargs={"silmaril_blocked": "silmaril-firewall:v1:blocked-tool"},
        )

    allowed = await middleware.awrap_tool_call(request, handler)
    assert "silmaril_blocked" not in allowed.additional_kwargs
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
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    captured = {}

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    def make(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(deepagents, "create_deep_agent", make)
    create_protected_deep_agent(
        fw, subagents=[{"name": "research", "description": "research"}],
        model=Model(responses=[AIMessage("ok")]),
    )
    assert isinstance(captured["middleware"][-1], type(create_deepagents_middleware(fw)))
    specs = {spec["name"]: spec for spec in captured["subagents"]}
    assert specs["general-purpose"]["middleware"]
    assert specs["research"]["middleware"]
    assert all(
        len([entry for entry in spec.get("middleware", []) if hasattr(entry, "firewall")]) == 1
        for spec in specs.values()
    )


def test_profile_disabled_default_is_not_replaced(monkeypatch):
    import deepagents
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )
    monkeypatch.setattr(
        harness_profiles, "_harness_profile_for_model", lambda model, spec: profile
    )
    captured = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    create_protected_deep_agent(fw, model=Model(responses=[AIMessage("ok")]))
    assert captured["subagents"] == []


def test_disabled_profile_omits_upstream_task_middleware(monkeypatch):
    import deepagents.graph
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
    from deepagents.middleware.subagents import SubAgentMiddleware
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )

    def resolver(model, spec):
        return profile

    monkeypatch.setattr(harness_profiles, "_harness_profile_for_model", resolver)
    monkeypatch.setattr(deepagents.graph, "_harness_profile_for_model", resolver)
    captured = {}
    monkeypatch.setattr(
        deepagents.graph,
        "create_agent",
        lambda *args, **kwargs: captured.update(kwargs)
        or SimpleNamespace(with_config=lambda config: None),
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    create_protected_deep_agent(fw, model=Model(responses=[AIMessage("ok")]))
    assert not any(
        isinstance(entry, SubAgentMiddleware) for entry in captured["middleware"]
    )


def test_profile_customized_default_keeps_one_protected_boundary(monkeypatch):
    import deepagents
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    profile = HarnessProfile(
        system_prompt_suffix="Profile suffix",
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            description="Profile description",
            system_prompt="Profile prompt",
        ),
    )
    monkeypatch.setattr(
        harness_profiles, "_harness_profile_for_model", lambda model, spec: profile
    )
    captured = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    model = Model(responses=[AIMessage("ok")])
    create_protected_deep_agent(fw, model=model)
    assert captured["model"] is model
    assert len(captured["subagents"]) == 1
    spec = captured["subagents"][0]
    assert spec["name"] == "general-purpose"
    assert spec["description"] == "Profile description"
    assert spec["system_prompt"] == "Profile prompt"
    assert len(spec["middleware"]) == 1
    assert spec["middleware"][0].firewall is fw


def test_upstream_applies_custom_profile_prompt_and_middleware_once(monkeypatch):
    import deepagents.graph
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
    from deepagents.backends import StateBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.skills import SkillsMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware
    from deepagents.profiles.harness import harness_profiles
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.tools import tool

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    class ProfileMiddleware(AgentMiddleware):
        pass

    @tool
    def parent_tool(query: str) -> str:
        """Return a deterministic parent tool result."""
        return query

    @tool
    def excluded_tool(query: str) -> str:
        """A tool removed by the active profile."""
        return query

    marker = ProfileMiddleware()
    root_filesystem = FilesystemMiddleware(backend=StateBackend())
    skill_sources = ["/skills/project/"]
    profile = HarnessProfile(
        system_prompt_suffix="Profile suffix",
        tool_description_overrides={"parent_tool": "Profile tool description"},
        excluded_tools=frozenset({"excluded_tool"}),
        extra_middleware=(marker,),
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            description="Profile description",
            system_prompt="Profile prompt",
        ),
    )

    def resolver(model, spec):
        return profile

    monkeypatch.setattr(harness_profiles, "_harness_profile_for_model", resolver)
    monkeypatch.setattr(deepagents.graph, "_harness_profile_for_model", resolver)
    captured = {}
    monkeypatch.setattr(
        deepagents.graph,
        "create_agent",
        lambda *args, **kwargs: captured.update(kwargs)
        or SimpleNamespace(with_config=lambda config: None),
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    create_protected_deep_agent(
        fw,
        model=Model(responses=[AIMessage("ok")]),
        tools=[parent_tool, excluded_tool],
        skills=skill_sources,
        middleware=[root_filesystem],
    )
    task = next(
        entry
        for entry in captured["middleware"]
        if isinstance(entry, SubAgentMiddleware)
    )
    default = task._subagents[0]
    assert default["description"] == "Profile description"
    assert default["system_prompt"] == "Profile prompt\n\nProfile suffix"
    assert [entry.name for entry in default["tools"]] == [
        "parent_tool",
        "excluded_tool",
    ]
    assert default["tools"][0].description == "Profile tool description"
    assert root_filesystem in default["middleware"]
    assert default["middleware"].count(marker) == 1
    skill_middleware = [
        entry
        for entry in default["middleware"]
        if isinstance(entry, SkillsMiddleware)
    ]
    assert len(skill_middleware) == 1
    assert skill_middleware[0].sources == skill_sources
    assert (
        len([
            entry
            for entry in default["middleware"]
            if entry.__class__.__name__ == "_ToolExclusionMiddleware"
        ])
        == 1
    )
    assert (
        len([
            entry
            for entry in default["middleware"]
            if getattr(entry, "firewall", None) is fw
        ])
        == 1
    )


def test_auto_default_retains_root_override_middleware_intent(monkeypatch):
    import deepagents
    from deepagents import HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    class FilesystemMiddleware(AgentMiddleware):
        pass

    override = FilesystemMiddleware()
    monkeypatch.setattr(
        harness_profiles, "_harness_profile_for_model", lambda model, spec: HarnessProfile()
    )
    captured = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    create_protected_deep_agent(
        fw,
        model=Model(responses=[AIMessage("ok")]),
        middleware=[override],
    )
    default_middleware = captured["subagents"][0]["middleware"]
    assert default_middleware[0] is override
    assert len([entry for entry in default_middleware if entry is override]) == 1
    assert default_middleware[-1].firewall is fw


def test_existing_protection_is_not_duplicated_and_conflicts_are_rejected(monkeypatch):
    import deepagents
    from deepagents import HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    monkeypatch.setattr(
        harness_profiles, "_harness_profile_for_model", lambda model, spec: HarnessProfile()
    )
    captured = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    existing = create_deepagents_middleware(fw)
    create_protected_deep_agent(
        fw,
        model=Model(responses=[AIMessage("ok")]),
        middleware=[existing],
    )
    assert captured["middleware"] == [existing]

    other = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.raises(ValueError, match="multiple Firewall clients"):
        create_protected_deep_agent(
            fw,
            model=Model(responses=[AIMessage("ok")]),
            middleware=[create_deepagents_middleware(other)],
        )


def test_existing_protection_requires_exact_options_and_preserves_order(monkeypatch):
    import deepagents
    from deepagents import HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    class BeforeMiddleware(AgentMiddleware):
        pass

    class AfterMiddleware(AgentMiddleware):
        pass

    monkeypatch.setattr(
        harness_profiles, "_harness_profile_for_model", lambda model, spec: HarnessProfile()
    )
    captured = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")

    def callback(event):
        return None

    options = {
        "mode": "warn",
        "fail_open": False,
        "max_blocked_attempts": 5,
        "on_classify": callback,
    }
    before = BeforeMiddleware()
    existing = create_deepagents_middleware(fw, **options)
    after = AfterMiddleware()
    create_protected_deep_agent(
        fw,
        model=Model(responses=[AIMessage("ok")]),
        middleware=[before, existing, after],
        middleware_options=options,
    )
    assert captured["middleware"] == [before, existing, after]

    with pytest.raises(ValueError, match="mode, fail_open, max_blocked_attempts, on_classify"):
        create_protected_deep_agent(
            fw,
            model=Model(responses=[AIMessage("ok")]),
            middleware=[existing],
        )


def test_existing_protection_rejects_duplicate_boundaries_before_construction(monkeypatch):
    import deepagents
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    monkeypatch.setattr(
        deepagents,
        "create_deep_agent",
        lambda **kwargs: pytest.fail("duplicate boundaries must fail before construction"),
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.raises(ValueError, match="duplicate SilmarilDeepAgentsMiddleware"):
        create_protected_deep_agent(
            fw,
            model=Model(responses=[AIMessage("ok")]),
            middleware=[
                create_deepagents_middleware(fw),
                create_deepagents_middleware(fw),
            ],
        )
    other = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.raises(ValueError, match="multiple Firewall clients"):
        create_protected_deep_agent(
            fw,
            model=Model(responses=[AIMessage("ok")]),
            middleware=[
                create_deepagents_middleware(fw),
                create_deepagents_middleware(other),
            ],
        )


def test_reused_root_boundary_classifies_each_model_hook_once(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    hooks = []

    def classify(text, **kwargs):
        hooks.append(kwargs["hook"])
        return BlockResult("BENIGN", 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", classify)
    existing = create_deepagents_middleware(fw)
    agent = create_protected_deep_agent(
        fw,
        model=Model(responses=[AIMessage("allowed")]),
        middleware=[existing],
    )
    agent.invoke({"messages": [{"role": "user", "content": "safe"}]})
    assert hooks == [HookLabel.USER_INPUT, HookLabel.LLM_OUTPUT]


def test_explicit_default_remains_explicit_when_profile_disables_auto_default(monkeypatch):
    import deepagents
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )
    monkeypatch.setattr(
        harness_profiles, "_harness_profile_for_model", lambda model, spec: profile
    )
    captured = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    create_protected_deep_agent(
        fw,
        model=Model(responses=[AIMessage("ok")]),
        subagents=[{
            "name": "general-purpose",
            "description": "Caller-owned default",
            "system_prompt": "Caller prompt",
        }],
    )
    assert [spec["name"] for spec in captured["subagents"]] == ["general-purpose"]
    assert captured["subagents"][0]["description"] == "Caller-owned default"
    assert len(captured["subagents"][0]["middleware"]) == 1


def test_profile_precedence_that_cannot_be_preserved_fails_closed(monkeypatch):
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    from silmaril_security.sdk._deepagents_compat import DeepAgentsCompatibilityError

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    profile = HarnessProfile(
        base_system_prompt="Root base",
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            system_prompt="More-specific child prompt"
        ),
    )
    monkeypatch.setattr(
        harness_profiles, "_harness_profile_for_model", lambda model, spec: profile
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.raises(DeepAgentsCompatibilityError, match="precedence cannot be preserved"):
        create_protected_deep_agent(
            fw, model=Model(responses=[AIMessage("unreachable")])
        )


def test_private_compatibility_version_guard_is_actionable(monkeypatch):
    import silmaril_security.sdk._deepagents_compat as compat

    monkeypatch.setattr(compat, "version", lambda distribution: "0.8.0")
    with pytest.raises(
        compat.DeepAgentsCompatibilityError,
        match=r"deepagents>=0\.7\.30,<0\.8.*found deepagents==0\.8\.0",
    ):
        compat.resolve_model_and_profile(object())


def test_registered_string_profile_constructs_model_once_and_preserves_default(monkeypatch):
    import deepagents.graph
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
    from deepagents.middleware.subagents import SubAgentMiddleware
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    model = Model(responses=[AIMessage("ok")])
    model_spec = "unrecoverable-provider:registered-alias"
    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            description="Registered description",
            system_prompt="Registered prompt",
        )
    )
    harness_profiles._ensure_harness_profiles_loaded()
    monkeypatch.setitem(harness_profiles._HARNESS_PROFILES, model_spec, profile)
    assert harness_profiles._harness_profile_for_model(model, None) != profile

    resolved_strings = []
    resolved_objects = []

    def resolve(value):
        if isinstance(value, str):
            resolved_strings.append(value)
            return model
        resolved_objects.append(value)
        return value

    monkeypatch.setattr(deepagents.graph, "resolve_model", resolve)
    captured = {}

    class Agent:
        def with_config(self, config):
            return self

    def make(model_arg, **kwargs):
        captured["model"] = model_arg
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr(deepagents.graph, "create_agent", make)
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    create_protected_deep_agent(fw, model=model_spec)

    task = next(
        entry
        for entry in captured["middleware"]
        if isinstance(entry, SubAgentMiddleware)
    )
    default = task._subagents[0]
    assert resolved_strings == [model_spec]
    assert resolved_objects == [model]
    assert captured["model"] is model
    assert default["model"] is model
    assert default["description"] == "Registered description"
    assert default["system_prompt"] == "Registered prompt"


def test_registered_string_profile_fails_closed_for_unrecoverable_root_overlays(monkeypatch):
    import deepagents.graph
    from deepagents import HarnessProfile
    from deepagents.profiles.harness import harness_profiles

    from silmaril_security.sdk._deepagents_compat import DeepAgentsCompatibilityError

    model_spec = "unrecoverable-provider:root-overlay"
    harness_profiles._ensure_harness_profiles_loaded()
    monkeypatch.setitem(
        harness_profiles._HARNESS_PROFILES,
        model_spec,
        HarnessProfile(system_prompt_suffix="Cannot safely infer child recovery"),
    )
    monkeypatch.setattr(
        deepagents.graph,
        "resolve_model",
        lambda value: pytest.fail("compatibility failure must precede model construction"),
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.raises(
        DeepAgentsCompatibilityError,
        match="runtime-registered string profile.*root-level",
    ):
        create_protected_deep_agent(fw, model=model_spec)


def test_default_model_is_built_once_and_reused_without_sdk_credential_access(monkeypatch):
    import deepagents
    import deepagents.graph
    from deepagents import HarnessProfile
    from deepagents.profiles.harness import harness_profiles
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    model = Model(responses=[AIMessage("ok")])
    calls = []
    monkeypatch.setattr(
        deepagents.graph,
        "_build_default_model",
        lambda: calls.append("built") or model,
    )
    monkeypatch.setattr(
        harness_profiles,
        "_harness_profile_for_model",
        lambda actual, spec: HarnessProfile(),
    )
    captured = {}
    monkeypatch.setattr(
        deepagents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.warns(DeprecationWarning, match="model=None"):
        create_protected_deep_agent(fw)
    assert calls == ["built"]
    assert captured["model"] is model


@pytest.mark.parametrize("mode", ["isolated", "fork"])
def test_compiled_subagent_propagates_mode_and_graph_identity(monkeypatch, mode):
    import langchain.agents

    class Graph:
        pass

    graph = Graph()
    captured = {}

    def make(**kwargs):
        captured.update(kwargs)
        return graph

    monkeypatch.setattr(langchain.agents, "create_agent", make)
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    spec = create_protected_compiled_subagent(
        fw,
        name="worker",
        description="does work",
        mode=mode,
        model="fake",
    )
    assert spec["mode"] == mode
    assert spec["runnable"] is graph
    assert "mode" not in captured


def test_compiled_subagent_omits_unset_mode(monkeypatch):
    import langchain.agents

    class Graph:
        pass

    monkeypatch.setattr(langchain.agents, "create_agent", lambda **kwargs: Graph())
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    spec = create_protected_compiled_subagent(
        fw,
        name="worker",
        description="does work",
        model="fake",
    )
    assert "mode" not in spec


def test_compiled_subagent_rejects_invalid_mode_before_graph_build(monkeypatch):
    import langchain.agents

    called = False

    def make(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(langchain.agents, "create_agent", make)
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    with pytest.raises(ValueError, match="isolated.*fork"):
        create_protected_compiled_subagent(
            fw,
            name="worker",
            description="does work",
            mode="shared",
            model="fake",
        )
    assert called is False


@pytest.mark.parametrize("mutated_mode", ["shared", None])
def test_parent_revalidates_mutated_compiled_mode(monkeypatch, mutated_mode):
    import deepagents
    import langchain.agents

    class Graph:
        pass

    graph = Graph()
    monkeypatch.setattr(langchain.agents, "create_agent", lambda **kwargs: graph)
    monkeypatch.setattr(
        deepagents,
        "create_deep_agent",
        lambda **kwargs: pytest.fail("invalid mode must fail before parent construction"),
    )
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    spec = create_protected_compiled_subagent(
        fw,
        name="worker",
        description="does work",
        mode="isolated",
        model="fake",
    )
    spec["mode"] = mutated_mode
    with pytest.raises(ValueError, match="isolated.*fork"):
        create_protected_deep_agent(fw, protected_compiled_subagents=[spec])


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
