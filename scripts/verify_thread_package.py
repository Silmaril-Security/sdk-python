"""Exercise the installed distribution; run outside the source checkout."""

import asyncio
import sys
from importlib.metadata import version

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableBinding, RunnableGenerator

from silmaril_security.sdk import Firewall, __version__
from silmaril_security.sdk.langchain import protect_langchain_runnable


class Model(FakeMessagesListChatModel):
    def bind_tools(self, *args, **kwargs):
        return self


assert __version__ == version("silmaril-security-sdk")
fw = Firewall(api_key="package-test", api_url="https://package.invalid/classify")
payloads = []


def classify(payload):
    payloads.append(payload)
    return {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5, "mode": "block"}


fw._post_json = classify
model = Model(responses=[AIMessage(content="done")])
runnable = protect_langchain_runnable(fw, model)
for thread in ["package-a", "package-b", None]:
    before = len(payloads)
    config = {}
    if thread is not None:
        config["configurable"] = {"thread_id": thread}
    runnable.invoke("safe", config=config)
    assert len(payloads) > before
    assert all(p["metadata"].get("conversationId") == thread for p in payloads[before:])

# Existing static and dynamic bindings must resolve before the protected
# boundary captures identity. This guards the release artifact regression.
before = len(payloads)
saved = protect_langchain_runnable(fw, model.with_config(configurable={"thread_id": "saved-before-protection"}))
saved.invoke("safe")
assert len(payloads) > before
assert all(p["metadata"]["conversationId"] == "saved-before-protection" for p in payloads[before:])

factory_calls = []


def configure(config):
    factory_calls.append(config)
    return {"configurable": {"thread_id": "dynamic-before-protection"}}


before = len(payloads)
dynamic = protect_langchain_runnable(fw, RunnableBinding(bound=model, config_factories=[configure]))
dynamic.invoke("safe")
assert len(factory_calls) == 1
assert len(payloads) > before
assert all(p["metadata"]["conversationId"] == "dynamic-before-protection" for p in payloads[before:])

# Post-protection methods must layer around (rather than replace) original
# factories on every supported core version, including core 0.2.0.
for operation in ("config", "bind", "retry", "listeners", "types"):
    before = len(payloads)
    factory_calls.clear()
    composed = protect_langchain_runnable(fw, RunnableBinding(bound=model, config_factories=[configure]))
    if operation == "config":
        composed = composed.with_config(configurable={"thread_id": "requested"})
    elif operation == "bind":
        composed = composed.bind(stop=["unused"])
    elif operation == "retry":
        composed = composed.with_retry(wait_exponential_jitter=False)
    elif operation == "listeners":
        composed = composed.with_listeners(on_start=lambda run: None)
    else:
        composed = composed.with_types(input_type=str)
    composed.invoke("safe")
    assert len(factory_calls) == 1
    assert len(payloads) > before
    assert all(p["metadata"]["conversationId"] == "dynamic-before-protection" for p in payloads[before:])

# A protected downstream stage must consume and emit chunks incrementally.
consumed = []


def upstream(inputs):
    consumed.append("first")
    yield "one"
    consumed.append("second")
    yield "two"


def downstream(inputs):
    for value in inputs:
        yield value.upper()


pipeline = RunnableGenerator(upstream) | protect_langchain_runnable(fw, RunnableGenerator(downstream))
stream = pipeline.stream("start")
assert next(stream) == "ONE" and consumed == ["first"]
assert next(stream) == "TWO" and consumed == ["first", "second"]
stream.close()

async def verify_incremental_async_transform():
    consumed = []

    async def upstream(inputs):
        consumed.append("first")
        yield "one"
        consumed.append("second")
        yield "two"

    async def downstream(inputs):
        async for value in inputs:
            yield value.upper()

    pipeline = RunnableGenerator(upstream) | protect_langchain_runnable(fw, RunnableGenerator(downstream))
    stream = pipeline.astream("start")
    assert await anext(stream) == "ONE" and consumed == ["first"]
    assert await anext(stream) == "TWO" and consumed == ["first", "second"]
    try:
        await anext(stream)
    except StopAsyncIteration:
        pass
    else:
        raise AssertionError("unexpected extra stream chunk")
    await stream.aclose()


asyncio.run(verify_incremental_async_transform())

if sys.version_info >= (3, 11):
    from silmaril_security.sdk.deepagents import create_protected_deep_agent

    agent = create_protected_deep_agent(fw, model=model)
    for thread in ["package-a", "package-b", None]:
        before = len(payloads)
        config = {"configurable": {"thread_id": thread}} if thread else {}
        agent.invoke({"messages": [{"role": "user", "content": "safe"}]}, config=config)
        assert len(payloads) > before
        assert all(p["metadata"].get("conversationId") == thread for p in payloads[before:])

assert all(p["metadata"]["silmaril"]["sdk_version"] == __version__ for p in payloads)
assert len({p["metadata"]["silmaril"]["request_id"] for p in payloads}) == len(payloads)
print("Installed SDK native thread identity verified")
