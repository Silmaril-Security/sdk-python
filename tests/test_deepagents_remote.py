"""Real Deep Agents graphs and SDK transport against a loopback Agent Protocol peer."""
from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from threading import Lock, Thread
from uuid import UUID

import pytest

pytest.importorskip("deepagents")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from silmaril_security.sdk import BlockResult, Firewall, HookLabel
from silmaril_security.sdk.deepagents import (
    SAFE_OUTPUT_MESSAGE,
    SAFE_TOOL_MESSAGE,
    create_protected_deep_agent,
)


class Model(FakeMessagesListChatModel):
    def bind_tools(self, *args, **kwargs):
        return self


def firewall(monkeypatch, *, block=True):
    fw = Firewall(api_key="test", api_url="https://unused.invalid/classify")
    seen = []

    def classify(text, **options):
        seen.append((text, options))
        denied = block and "unsafe" in text
        return BlockResult("MALICIOUS" if denied else "BENIGN", 0.9 if denied else 0.1, 0.5, mode="block")

    monkeypatch.setattr(fw, "_classify_raw", classify)
    return fw, seen


def call(name, args, ident="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident}])


@pytest.fixture
def peer():
    """Only the Agent Protocol endpoints consumed by the real upstream SDK."""
    state = {"threads": {}, "runs": {}, "requests": [], "child": None, "tamper": False, "fail": False}
    mutex, thread_ids, run_ids = Lock(), count(1), count(100)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, data):
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            state["requests"].append(("GET", self.path, None))
            if len(parts) == 2:
                return self.respond(200, state["threads"][parts[1]])
            if len(parts) == 4:
                return self.respond(200, state["runs"][parts[3]])
            if len(parts) == 3:
                return self.respond(200, [x for x in state["runs"].values() if x["thread_id"] == parts[1]][::-1])
            self.respond(404, {"error": "unexpected path"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            parts = self.path.split("?")[0].strip("/").split("/")
            state["requests"].append(("POST", self.path, body))
            if state["fail"]:
                return self.respond(400, {"detail": "remote launch unavailable"})
            if parts == ["threads"]:
                with mutex:
                    ident = str(UUID(int=next(thread_ids)))
                value = {"thread_id": ident, "metadata": body.get("metadata", {}), "status": "idle", "values": {}}
                state["threads"][ident] = value
                return self.respond(200, value)
            if len(parts) == 3 and parts[2] == "runs":
                thread = state["threads"][parts[1]]
                with mutex:
                    ident = str(UUID(int=next(run_ids)))
                config = body.get("config") or {}
                config["configurable"] = {**config.get("configurable", {}), "thread_id": parts[1]}
                config["metadata"] = thread["metadata"]
                result = asyncio.run(state["child"].ainvoke(body["input"], config))
                last = result["messages"][-1].content
                if state["tamper"]:
                    last = "unsafe remote result"
                thread["values"] = {"messages": [{"type": "ai", "content": last}]}
                run = {"thread_id": parts[1], "run_id": ident, "status": "success"}
                state["runs"][ident] = run
                return self.respond(200, run)
            if len(parts) == 5 and parts[4] == "cancel":
                state["runs"][parts[3]]["status"] = "interrupted"
                return self.respond(200, {})
            self.respond(404, {"error": "unexpected path"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def attach_child(monkeypatch, peer, responses=None):
    fw, seen = firewall(monkeypatch)
    executed = []

    @tool
    def search(query: str) -> str:
        """Search a test corpus."""
        executed.append(query)
        return "unsafe tool result"

    model = Model(responses=responses or [
        call("search", {"query": "unsafe action"}, "child-1"),
        call("search", {"query": "safe action"}, "child-2"),
        AIMessage(content="unsafe final answer"),
        call("search", {"query": "unsafe action"}, "child-3"),
        call("search", {"query": "safe action"}, "child-4"),
        AIMessage(content="unsafe final answer"),
    ])
    peer["child"] = create_protected_deep_agent(fw, model=model, tools=[search], checkpointer=MemorySaver())
    return seen, executed


def parent(monkeypatch, peer, responses, *, mode="isolated", block=True):
    fw, seen = firewall(monkeypatch, block=block)
    spec = {"name": "research", "description": "Remote research", "graph_id": "research", "url": peer["url"],
            "mode": mode, "notify_parent": False, "update_strategy": "enqueue", "headers": {"x-test-auth": "local-only"}}
    agent = create_protected_deep_agent(fw, model=Model(responses=responses), subagents=[spec])
    return agent, seen


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["isolated", "fork"])
@pytest.mark.parametrize("execution", ["sync", "async"])
async def test_remote_lifecycle_protects_parent_and_child(monkeypatch, peer, mode, execution):
    child_seen, executed = attach_child(monkeypatch, peer)
    task = str(UUID(int=1))
    agent, parent_seen = parent(monkeypatch, peer, [
        call("start_async_task", {"description": "safe research", "subagent_type": "research"}),
        call("check_async_task", {"task_id": task}, "check"),
        call("update_async_task", {"task_id": task, "message": "safe update"}, "update"),
        call("list_async_tasks", {}, "list"),
        call("cancel_async_task", {"task_id": task}, "cancel"),
        AIMessage(content="done"),
    ], mode=mode)
    inputs = {"messages": [HumanMessage("hello")]}
    config = {"configurable": {"thread_id": "parent-thread"}}
    out = await agent.ainvoke(inputs, config) if execution == "async" else await asyncio.to_thread(agent.invoke, inputs, config)
    assert out["async_tasks"][task]["status"] == "cancelled"
    assert executed == ["safe action", "safe action"]
    assert all(x[1]["metadata"]["conversationId"] == task for x in child_seen)
    assert all(x[1]["metadata"]["langgraph"]["parent_thread_id"] == "parent-thread" for x in child_seen)
    assert all(x[1]["metadata"]["conversationId"] == "parent-thread" for x in parent_seen)
    assert any(x[1]["hook"] == HookLabel.TOOL_CALL and "unsafe action" in x[0] for x in child_seen)
    assert any(x[1]["hook"] == HookLabel.TOOL_RESPONSE and "unsafe tool result" in x[0] for x in child_seen)
    assert any(x[1]["hook"] == HookLabel.LLM_OUTPUT and "unsafe final" in x[0] for x in child_seen)
    assert any(x[0] == "POST" and x[1].split("?")[0].endswith("/cancel") for x in peer["requests"])
    launches = [x[2] for x in peer["requests"] if x[0] == "POST" and x[1].endswith("/runs")]
    assert launches[1]["multitask_strategy"] == "enqueue"
    assert len(launches[0]["input"]["messages"]) > 1 if mode == "fork" else len(launches[0]["input"]["messages"]) == 1


@pytest.mark.asyncio
async def test_parent_blocks_launch_before_remote_execution(monkeypatch, peer):
    agent, _ = parent(monkeypatch, peer, [call("start_async_task", {"description": "unsafe delegation", "subagent_type": "research"}), AIMessage(content="done")])
    out = await agent.ainvoke({"messages": [HumanMessage("hello")]})
    assert peer["requests"] == []
    assert any(isinstance(x, ToolMessage) and x.content == SAFE_TOOL_MESSAGE for x in out["messages"])


@pytest.mark.asyncio
async def test_remote_child_blocks_input_independently(monkeypatch, peer):
    seen, executed = attach_child(monkeypatch, peer)
    agent, _ = parent(monkeypatch, peer, [call("start_async_task", {"description": "unsafe child input", "subagent_type": "research"}), AIMessage(content="done")], block=False)
    await agent.ainvoke({"messages": [HumanMessage("hello")]})
    assert executed == []
    assert any(x[1]["hook"] == HookLabel.USER_INPUT and "unsafe child input" in x[0] for x in seen)
    assert next(iter(peer["threads"].values()))["values"]["messages"][-1]["content"] == SAFE_OUTPUT_MESSAGE


@pytest.mark.asyncio
async def test_parent_withholds_remote_result_and_keeps_task_state(monkeypatch, peer):
    attach_child(monkeypatch, peer, [AIMessage(content="safe child answer")])
    peer["tamper"] = True
    task = str(UUID(int=1))
    agent, seen = parent(monkeypatch, peer, [call("start_async_task", {"description": "safe", "subagent_type": "research"}), call("check_async_task", {"task_id": task}, "check"), AIMessage(content="done")])
    out = await agent.ainvoke({"messages": [HumanMessage("hello")]})
    assert out["async_tasks"][task]["status"] == "success"
    assert any(x[1]["hook"] == HookLabel.TOOL_RESPONSE and "unsafe remote result" in x[0] for x in seen)
    assert any(isinstance(x, ToolMessage) and x.tool_call_id == "check" and x.content == SAFE_TOOL_MESSAGE for x in out["messages"])


@pytest.mark.asyncio
async def test_launch_failure_is_visible_without_fabricated_task(monkeypatch, peer):
    peer["fail"] = True
    agent, _ = parent(monkeypatch, peer, [call("start_async_task", {"description": "safe", "subagent_type": "research"}), AIMessage(content="done")])
    out = await agent.ainvoke({"messages": [HumanMessage("hello")]})
    assert not out.get("async_tasks")
    assert any(isinstance(x, ToolMessage) and "Failed to launch" in x.content for x in out["messages"])


@pytest.mark.parametrize("key", ["lc_source", "deepagents_notification"])
@pytest.mark.parametrize("mode", ["block", "warn", "shadow"])
def test_background_notification_is_filtered_before_newer_user_message(monkeypatch, key, mode):
    from langchain.agents.middleware import ModelRequest

    from silmaril_security.sdk.deepagents import create_deepagents_middleware

    fw, seen = firewall(monkeypatch)
    middleware = create_deepagents_middleware(fw, mode=mode)
    note = HumanMessage(content="unsafe remote notification", additional_kwargs={
        key: "async_subagent" if key == "lc_source" else {"status": "success"},
    })
    messages = [note, HumanMessage("safe new user question")]
    request = ModelRequest(model=Model(responses=[AIMessage("done")]), messages=messages, state={"messages": messages})
    received = []

    def handler(req):
        received.extend(req.messages)
        return AIMessage("done")

    middleware.wrap_model_call(request, handler)
    assert received[0].content == (SAFE_TOOL_MESSAGE if mode == "block" else note.content)
    assert note.content == "unsafe remote notification"
    assert any(options["hook"] == HookLabel.TOOL_RESPONSE for _, options in seen)


@pytest.mark.asyncio
async def test_concurrent_remote_tasks_keep_distinct_child_identities(monkeypatch, peer):
    child_seen, _ = attach_child(monkeypatch, peer, [AIMessage(content="safe child answer")])
    fw, _ = firewall(monkeypatch)
    model = Model(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "start_async_task", "args": {"description": "safe one", "subagent_type": "research"}, "id": "one"},
            {"name": "start_async_task", "args": {"description": "safe two", "subagent_type": "research"}, "id": "two"},
        ]), AIMessage(content="done"),
    ])
    agent = create_protected_deep_agent(fw, model=model, subagents=[{
        "name": "research", "description": "Remote", "graph_id": "research", "url": peer["url"], "notify_parent": False,
    }])
    out = await agent.ainvoke({"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": "parent"}})
    assert len(out["async_tasks"]) == 2
    assert {opts["metadata"]["conversationId"] for _, opts in child_seen} == set(out["async_tasks"])
    assert all(opts["metadata"]["langgraph"]["parent_thread_id"] == "parent" for _, opts in child_seen)


@pytest.mark.asyncio
async def test_native_notification_is_checked_once_and_stays_withheld_after_resume(monkeypatch, peer):
    attach_child(monkeypatch, peer, [AIMessage(content="safe child answer")])
    peer["tamper"] = True
    captured = []

    class RecordingModel(Model):
        def _generate(self, messages, *args, **kwargs):
            captured.extend(message.content for message in messages)
            return super()._generate(messages, *args, **kwargs)

    fw, seen = firewall(monkeypatch)
    agent = create_protected_deep_agent(fw, model=RecordingModel(responses=[
        call("start_async_task", {"description": "safe", "subagent_type": "research"}),
        AIMessage(content="done"), AIMessage(content="done again"),
    ]), subagents=[{"name": "research", "description": "Remote", "graph_id": "research",
                   "url": peer["url"], "notify_parent": True}], checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "parent-resume"}}
    first = await agent.ainvoke({"messages": [HumanMessage("hello")]}, config)
    # Assert the real upstream graph produced a marked notification, not just a test message.
    assert any("unsafe remote result" in message.text and (
        message.additional_kwargs.get("lc_source") == "async_subagent"
        or "deepagents_notification" in message.additional_kwargs
    ) for message in first["messages"])
    await agent.ainvoke({"messages": [HumanMessage("safe newer question")]}, config)
    assert "unsafe remote result" not in json.dumps(captured)
    checks = [text for text, opts in seen if opts["hook"] == HookLabel.TOOL_RESPONSE
              and opts.get("tool_name") == "async_subagent" and "unsafe remote result" in text]
    assert len(checks) == 1


def test_notification_cache_is_bound_to_thread_content_and_success(monkeypatch):
    from types import SimpleNamespace

    from langchain.agents.middleware import ModelRequest

    from silmaril_security.sdk.deepagents import create_deepagents_middleware

    fw, seen = firewall(monkeypatch)
    middleware = create_deepagents_middleware(fw)

    def request(thread, content):
        messages = [HumanMessage(id="note", content=content, additional_kwargs={"lc_source": "async_subagent"}), HumanMessage("safe")]
        return ModelRequest(model=Model(responses=[AIMessage("done")]), messages=messages, state={"messages": messages},
                            runtime=SimpleNamespace(config={"configurable": {"thread_id": thread}}))

    for thread, content in [("one", "safe note"), ("one", "safe note"), ("two", "safe note"), ("one", "unsafe changed note")]:
        middleware.wrap_model_call(request(thread, content), lambda _: AIMessage("done"))
    assert len([opts for _, opts in seen if opts.get("tool_name") == "async_subagent"]) == 3

    raw = fw._classify_raw
    failed = False

    def transient(text, **kwargs):
        nonlocal failed
        if kwargs.get("tool_name") == "async_subagent" and not failed:
            failed = True
            raise ConnectionError("temporary")
        return raw(text, **kwargs)

    monkeypatch.setattr(fw, "_classify_raw", transient)
    req = request("fresh", "unsafe notification")
    assert middleware._filter_remote_notifications(req).messages[0].content == "unsafe notification"
    assert middleware._filter_remote_notifications(req).messages[0].content == SAFE_TOOL_MESSAGE
