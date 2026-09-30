"""Optional Deep Agents middleware and protected agent construction."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple
from uuid import uuid4
from weakref import ref

from silmaril_security.sdk.async_firewall import AsyncFirewall
from silmaril_security.sdk.firewall import Firewall
from silmaril_security.sdk.hooks import HookLabel
from silmaril_security.sdk.types import ClassifyEvent, FirewallMode

try:
    from langchain.agents.middleware import (
        AgentMiddleware,
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
    )
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'Deep Agents support requires pip install "silmaril-security-sdk[deepagents]"'
    ) from exc

SAFE_TOOL_MESSAGE = "Silmaril Firewall blocked this tool interaction. Choose a different safe action."
SAFE_FINAL_MESSAGE = "Silmaril Firewall stopped this request after repeated unsafe actions."
SAFE_OUTPUT_MESSAGE = "Silmaril Firewall blocked this response."

# Only graphs compiled by the factory below can be attached as protected.
# Use object identity rather than graph equality; weak references clear stale IDs.
_PROTECTED_COMPILED_GRAPHS: dict[int, tuple[Any, Firewall | AsyncFirewall]] = {}


def _register_protected_graph(runnable: Any, firewall: Firewall | AsyncFirewall) -> None:
    identity = id(runnable)

    def forget(reference: Any) -> None:
        entry = _PROTECTED_COMPILED_GRAPHS.get(identity)
        if entry is not None and entry[0] is reference:
            del _PROTECTED_COMPILED_GRAPHS[identity]

    _PROTECTED_COMPILED_GRAPHS[identity] = (ref(runnable, forget), firewall)


def _is_protected_graph(runnable: Any, firewall: Firewall | AsyncFirewall) -> bool:
    entry = _PROTECTED_COMPILED_GRAPHS.get(id(runnable))
    return entry is not None and entry[0]() is runnable and entry[1] is firewall


class _Decision(NamedTuple):
    enforce: bool
    mode: FirewallMode | None


class SilmarilDeepAgentsMiddleware(AgentMiddleware):
    """Protect model and tool boundaries in one Deep Agents graph."""

    def __init__(
        self,
        firewall: Firewall | AsyncFirewall,
        *,
        mode: FirewallMode | None = None,
        fail_open: bool = True,
        max_blocked_attempts: int = 3,
        conversation_id: str | None = None,
        on_classify: Callable[[ClassifyEvent], None] | None = None,
    ) -> None:
        if max_blocked_attempts < 1:
            raise ValueError("max_blocked_attempts must be positive")
        self.firewall = firewall
        self.mode = firewall._effective_mode(mode, None)
        self.fail_open = fail_open
        self.max_blocked_attempts = max_blocked_attempts
        self.conversation_id = conversation_id
        self.on_classify = on_classify

    def _metadata(self, request: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        config = getattr(getattr(request, "runtime", None), "config", None)
        if isinstance(config, dict) and config.get("run_id") is not None:
            metadata["langgraph"] = {"run_id": str(config["run_id"])}
        if self.conversation_id is not None:
            metadata["conversationId"] = self.conversation_id
        return metadata

    def _observe(self, event: ClassifyEvent) -> None:
        if isinstance(self.firewall, Firewall):
            self.firewall._fire_on_classify(event)
        if self.on_classify is not None:
            try:
                self.on_classify(event)
            except Exception:
                logging.getLogger(__name__).warning("on_classify callback raised", exc_info=True)

    async def _aobserve(self, event: ClassifyEvent) -> None:
        if isinstance(self.firewall, AsyncFirewall):
            await self.firewall._fire_on_classify(event)
        self._observe(event)

    def _classify_decision(self, text: str, hook: HookLabel, request: Any, tool_name: str | None = None) -> _Decision:
        if not text.strip():
            return _Decision(False, self.mode)
        if isinstance(self.firewall, AsyncFirewall):
            raise TypeError("Use async Deep Agents execution with AsyncFirewall")
        try:
            result = self.firewall._classify_raw(
                text, hook=hook, tool_name=tool_name, metadata=self._metadata(request),
                governance=None, request_id=str(uuid4()), mode=self.mode,
            )
        except Exception:
            if not self.fail_open:
                raise
            logging.getLogger(__name__).warning("Firewall classification failed", exc_info=True)
            return _Decision(False, self.mode)
        effective_mode = self.mode or result.mode or "block"
        blocked = result.prediction == "MALICIOUS" or result.governance is not None and result.governance.action == "block"
        self._observe(ClassifyEvent(hook, tool_name, text, result, blocked, effective_mode == "shadow", mode=effective_mode))
        return _Decision(blocked and effective_mode == "block", effective_mode)

    async def _aclassify_decision(self, text: str, hook: HookLabel, request: Any, tool_name: str | None = None) -> _Decision:
        if not text.strip():
            return _Decision(False, self.mode)
        if not isinstance(self.firewall, AsyncFirewall):
            return self._classify_decision(text, hook, request, tool_name)
        try:
            result = await self.firewall._classify_raw(
                text, hook=hook, tool_name=tool_name, metadata=self._metadata(request),
                governance=None, request_id=str(uuid4()), mode=self.mode,
            )
        except Exception:
            if not self.fail_open:
                raise
            logging.getLogger(__name__).warning("Firewall classification failed", exc_info=True)
            return _Decision(False, self.mode)
        effective_mode = self.mode or result.mode or "block"
        blocked = result.prediction == "MALICIOUS" or result.governance is not None and result.governance.action == "block"
        await self._aobserve(ClassifyEvent(hook, tool_name, text, result, blocked, effective_mode == "shadow", mode=effective_mode))
        return _Decision(blocked and effective_mode == "block", effective_mode)

    def _classify(self, text: str, hook: HookLabel, request: Any, tool_name: str | None = None) -> bool:
        return self._classify_decision(text, hook, request, tool_name).enforce

    async def _aclassify(self, text: str, hook: HookLabel, request: Any, tool_name: str | None = None) -> bool:
        return (await self._aclassify_decision(text, hook, request, tool_name)).enforce

    def _blocked_count(self, request: Any) -> int:
        messages = request.state.get("messages", [])
        last_user_index = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
            -1,
        )
        return sum(
            isinstance(message, ToolMessage) and message.content == SAFE_TOOL_MESSAGE
            for message in messages[last_user_index + 1:]
        )

    def _input(self, request: ModelRequest) -> str:
        return next(
            (message.text for message in reversed(request.messages) if isinstance(message, HumanMessage)),
            "",
        )

    def _cap_mode(self, request: ModelRequest, input_mode: FirewallMode | None) -> FirewallMode | None:
        if input_mode is not None:
            return input_mode
        messages = request.state.get("messages", [])
        latest = request.messages[-1] if request.messages else (messages[-1] if messages else None)
        if isinstance(latest, ToolMessage):
            return self._classify_decision(str(latest.content), HookLabel.TOOL_RESPONSE, request).mode
        return None

    async def _acap_mode(self, request: ModelRequest, input_mode: FirewallMode | None) -> FirewallMode | None:
        if input_mode is not None:
            return input_mode
        messages = request.state.get("messages", [])
        latest = request.messages[-1] if request.messages else (messages[-1] if messages else None)
        if isinstance(latest, ToolMessage):
            return (await self._aclassify_decision(str(latest.content), HookLabel.TOOL_RESPONSE, request)).mode
        return None

    def _filter_output(self, response: Any, request: ModelRequest) -> Any:
        if isinstance(response, ExtendedModelResponse):
            if response.command is not None and self._classify(
                str(response.command.update), HookLabel.LLM_OUTPUT, request
            ):
                return AIMessage(content=SAFE_OUTPUT_MESSAGE)
            response.model_response = self._filter_output(response.model_response, request)
            if any(
                isinstance(message, AIMessage) and message.content == SAFE_OUTPUT_MESSAGE
                for message in response.model_response.result
            ):
                return AIMessage(content=SAFE_OUTPUT_MESSAGE)
            return response
        if isinstance(response, AIMessage):
            return AIMessage(content=SAFE_OUTPUT_MESSAGE) if self._classify(response.text, HookLabel.LLM_OUTPUT, request) else response
        if isinstance(response, ModelResponse):
            response.result = [
                AIMessage(content=SAFE_OUTPUT_MESSAGE)
                if isinstance(message, AIMessage) and self._classify(message.text, HookLabel.LLM_OUTPUT, request)
                else message for message in response.result
            ]
        return response

    async def _afilter_output(self, response: Any, request: ModelRequest) -> Any:
        if isinstance(response, ExtendedModelResponse):
            if response.command is not None and await self._aclassify(
                str(response.command.update), HookLabel.LLM_OUTPUT, request
            ):
                return AIMessage(content=SAFE_OUTPUT_MESSAGE)
            response.model_response = await self._afilter_output(response.model_response, request)
            if any(
                isinstance(message, AIMessage) and message.content == SAFE_OUTPUT_MESSAGE
                for message in response.model_response.result
            ):
                return AIMessage(content=SAFE_OUTPUT_MESSAGE)
            return response
        if isinstance(response, AIMessage):
            return AIMessage(content=SAFE_OUTPUT_MESSAGE) if await self._aclassify(response.text, HookLabel.LLM_OUTPUT, request) else response
        if isinstance(response, ModelResponse):
            result = []
            for message in response.result:
                if isinstance(message, AIMessage) and await self._aclassify(message.text, HookLabel.LLM_OUTPUT, request):
                    message = AIMessage(content=SAFE_OUTPUT_MESSAGE)
                result.append(message)
            response.result = result
        return response

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        input_decision = self._classify_decision(self._input(request), HookLabel.USER_INPUT, request)
        if input_decision.enforce:
            return AIMessage(content=SAFE_OUTPUT_MESSAGE)
        if self._blocked_count(request) >= self.max_blocked_attempts and self._cap_mode(request, input_decision.mode) == "block":
            return AIMessage(content=SAFE_FINAL_MESSAGE)
        return self._filter_output(handler(request), request)

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        input_decision = await self._aclassify_decision(self._input(request), HookLabel.USER_INPUT, request)
        if input_decision.enforce:
            return AIMessage(content=SAFE_OUTPUT_MESSAGE)
        if self._blocked_count(request) >= self.max_blocked_attempts and await self._acap_mode(request, input_decision.mode) == "block":
            return AIMessage(content=SAFE_FINAL_MESSAGE)
        return await self._afilter_output(await handler(request), request)

    def _tool_text(self, request: Any) -> str:
        import json
        return json.dumps(request.tool_call.get("args", {}), ensure_ascii=False, default=str)

    def _safe_tool_result(self, request: Any) -> ToolMessage:
        return ToolMessage(content=SAFE_TOOL_MESSAGE, tool_call_id=request.tool_call["id"], name=request.tool_call["name"])

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        name = request.tool_call["name"]
        if self._classify(self._tool_text(request), HookLabel.TOOL_CALL, request, name):
            return self._safe_tool_result(request)
        result = handler(request)
        if self._classify(str(result.content) if isinstance(result, ToolMessage) else str(result), HookLabel.TOOL_RESPONSE, request, name):
            return self._safe_tool_result(request)
        return result

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        name = request.tool_call["name"]
        if await self._aclassify(self._tool_text(request), HookLabel.TOOL_CALL, request, name):
            return self._safe_tool_result(request)
        result = await handler(request)
        if await self._aclassify(str(result.content) if isinstance(result, ToolMessage) else str(result), HookLabel.TOOL_RESPONSE, request, name):
            return self._safe_tool_result(request)
        return result


def create_deepagents_middleware(firewall: Firewall | AsyncFirewall, **kwargs: Any) -> SilmarilDeepAgentsMiddleware:
    """Create optional middleware for a Deep Agents graph."""
    return SilmarilDeepAgentsMiddleware(firewall, **kwargs)


def create_protected_compiled_subagent(
    firewall: Firewall | AsyncFirewall, *, name: str, description: str,
    model: Any, tools: Sequence[Any] = (), middleware: Sequence[Any] = (),
    middleware_options: dict[str, Any] | None = None, **agent_kwargs: Any,
) -> dict[str, Any]:
    """Compile a subagent with Silmaril installed and register its graph identity."""
    from langchain.agents import create_agent

    runnable = create_agent(
        model=model, tools=list(tools),
        middleware=[*middleware, create_deepagents_middleware(firewall, **(middleware_options or {}))],
        **agent_kwargs,
    )
    _register_protected_graph(runnable, firewall)
    return {"name": name, "description": description, "runnable": runnable}


def create_protected_deep_agent(
    firewall: Firewall | AsyncFirewall, *, subagents: Sequence[dict[str, Any]] = (),
    protected_compiled_subagents: Sequence[dict[str, Any]] = (),
    middleware_options: dict[str, Any] | None = None, **kwargs: Any,
) -> Any:
    """Protect root and inline subagents; accept only factory-verified compiled graphs."""
    try:
        from deepagents import create_deep_agent
        from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
    except ImportError as exc:  # pragma: no cover
        raise ImportError('Install "silmaril-security-sdk[deepagents]"') from exc
    options = middleware_options or {}
    root = create_deepagents_middleware(firewall, **options)
    protected: list[dict[str, Any]] = []
    for spec in subagents:
        if "runnable" in spec:
            raise ValueError("Compiled subagents must be protected inside their compiled graph")
        protected.append({**spec, "middleware": [*(spec.get("middleware") or []), create_deepagents_middleware(firewall, **options)]})
    for spec in protected_compiled_subagents:
        runnable = spec.get("runnable")
        if runnable is None or not _is_protected_graph(runnable, firewall):
            raise ValueError(
                "Compiled subagents must be built with create_protected_compiled_subagent "
                "for this Firewall client"
            )
        protected.append(dict(spec))
    if not any(spec.get("name") == "general-purpose" for spec in protected):
        protected.insert(0, {**GENERAL_PURPOSE_SUBAGENT, "middleware": [create_deepagents_middleware(firewall, **options)]})
    return create_deep_agent(
        subagents=protected, middleware=[*(kwargs.pop("middleware", ()) or ()), root], **kwargs
    )
