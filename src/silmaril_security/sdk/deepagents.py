"""Optional Deep Agents middleware and protected agent construction."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from threading import RLock
from typing import Any, Literal, NamedTuple, cast
from uuid import uuid4
from weakref import WeakValueDictionary, ref

from silmaril_security.sdk._thread_identity import (
    conversation_metadata,
    reject_conversation_options,
)
from silmaril_security.sdk.async_firewall import AsyncFirewall
from silmaril_security.sdk.exceptions import ReviewCompatibilityError
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
    from langchain_core.runnables.config import ensure_config
    from langgraph.types import Command
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'Deep Agents support requires pip install "silmaril-security-sdk[deepagents]"'
    ) from exc

SAFE_TOOL_MESSAGE = "Silmaril Firewall blocked this tool interaction. Choose a different safe action."
SAFE_FINAL_MESSAGE = "Silmaril Firewall stopped this request after repeated unsafe actions."
SAFE_OUTPUT_MESSAGE = "Silmaril Firewall blocked this response."
_BLOCKED_TOOL_MARKER = "silmaril-firewall:v1:blocked-tool"
_ISSUED_BLOCKED_MESSAGES: WeakValueDictionary[int, ToolMessage] = WeakValueDictionary()

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
    cacheable: bool = True


class SilmarilDeepAgentsMiddleware(AgentMiddleware):
    """Protect model and tool boundaries in one Deep Agents graph."""

    def __init__(
        self,
        firewall: Firewall | AsyncFirewall,
        *,
        mode: FirewallMode | None = None,
        fail_open: bool = True,
        max_blocked_attempts: int = 3,
        on_classify: Callable[[ClassifyEvent], None] | None = None,
        **removed_options: Any,
    ) -> None:
        reject_conversation_options(removed_options)
        if removed_options:
            raise TypeError(f"Unexpected options: {sorted(removed_options)}")
        if max_blocked_attempts < 1:
            raise ValueError("max_blocked_attempts must be positive")
        self.firewall = firewall
        self.mode = firewall._effective_mode(mode, None)
        self.fail_open = fail_open
        self.max_blocked_attempts = max_blocked_attempts
        self.on_classify = on_classify
        self._notification_checks: OrderedDict[tuple[str, str, str, FirewallMode | None], bool] = OrderedDict()
        self._notification_lock = RLock()

    def _metadata(self, request: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        config = getattr(getattr(request, "runtime", None), "config", None)
        if not isinstance(config, dict):
            config = ensure_config()
        if isinstance(config, dict) and config.get("run_id") is not None:
            metadata["langgraph"] = {"run_id": str(config["run_id"])}
        # Remote task lineage is attribution, never the child's sequence identity.
        if isinstance(config, dict):
            native = config.get("metadata")
            native = native if isinstance(native, dict) else {}
            configurable = config.get("configurable")
            parent = configurable.get("deepagents_parent") if isinstance(configurable, dict) else None
            parent = parent if isinstance(parent, dict) else {}
            for key, value in {
                "parent_thread_id": native.get("parent_thread_id") or parent.get("thread_id"),
                "subagent_name": native.get("subagent_name"),
            }.items():
                if isinstance(value, str) and value.strip():
                    metadata.setdefault("langgraph", {})[key] = value
        metadata.update(conversation_metadata(config))
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
        except ReviewCompatibilityError:
            raise
        except Exception:
            if not self.fail_open:
                raise
            logging.getLogger(__name__).warning("Firewall classification failed", exc_info=True)
            return _Decision(False, self.mode, False)
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
        except ReviewCompatibilityError:
            raise
        except Exception:
            if not self.fail_open:
                raise
            logging.getLogger(__name__).warning("Firewall classification failed", exc_info=True)
            return _Decision(False, self.mode, False)
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
            isinstance(message, ToolMessage)
            and message.content == SAFE_TOOL_MESSAGE
            and message.additional_kwargs.get("silmaril_blocked") == _BLOCKED_TOOL_MARKER
            for message in messages[last_user_index + 1:]
        )

    def _input(self, request: ModelRequest) -> str:
        return next(
            (message.text for message in reversed(request.messages) if isinstance(message, HumanMessage)),
            "",
        )

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

    @staticmethod
    def _remote_notification(message: Any) -> bool:
        extra = getattr(message, "additional_kwargs", {})
        source = extra.get("lc_source")
        # Markers select extra inspection; they confer no trust or authority.
        return (isinstance(source, str) and "async_subagent" in source) or isinstance(
            extra.get("deepagents_notification"), dict
        )

    def _notification_key(self, request: ModelRequest, message: Any) -> tuple[str, str, str, FirewallMode | None] | None:
        thread_id = self._metadata(request).get("conversationId")
        if not thread_id or not isinstance(message.id, str) or not message.id:
            return None
        try:
            payload = json.dumps([message.content, message.additional_kwargs], sort_keys=True)
        except (TypeError, ValueError):
            return None
        return thread_id, message.id, hashlib.sha256(payload.encode("ascii")).hexdigest(), self.mode

    def _cached_notification(self, key: Any) -> bool | None:
        with self._notification_lock:
            if key not in self._notification_checks:
                return None
            self._notification_checks.move_to_end(key)
            return self._notification_checks[key]

    def _remember_notification(self, key: Any, decision: _Decision) -> None:
        if key is None or not decision.cacheable:
            return
        with self._notification_lock:
            self._notification_checks[key] = decision.enforce
            self._notification_checks.move_to_end(key)
            while len(self._notification_checks) > 1024:
                self._notification_checks.popitem(last=False)

    def _filter_remote_notifications(self, request: ModelRequest) -> ModelRequest:
        messages = []
        for message in request.messages:
            if self._remote_notification(message):
                key = self._notification_key(request, message)
                blocked = self._cached_notification(key)
                if blocked is None:
                    decision = self._classify_decision(message.text, HookLabel.TOOL_RESPONSE, request, "async_subagent")
                    self._remember_notification(key, decision)
                    blocked = decision.enforce
                if blocked:
                    message = message.model_copy(update={"content": SAFE_TOOL_MESSAGE})
            messages.append(message)
        return request.override(messages=messages) if any(a is not b for a, b in zip(messages, request.messages)) else request

    async def _afilter_remote_notifications(self, request: ModelRequest) -> ModelRequest:
        messages = []
        for message in request.messages:
            if self._remote_notification(message):
                key = self._notification_key(request, message)
                blocked = self._cached_notification(key)
                if blocked is None:
                    decision = await self._aclassify_decision(message.text, HookLabel.TOOL_RESPONSE, request, "async_subagent")
                    self._remember_notification(key, decision)
                    blocked = decision.enforce
                if blocked:
                    message = message.model_copy(update={"content": SAFE_TOOL_MESSAGE})
            messages.append(message)
        return request.override(messages=messages) if any(a is not b for a, b in zip(messages, request.messages)) else request

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        request = self._filter_remote_notifications(request)
        input_decision = self._classify_decision(self._input(request), HookLabel.USER_INPUT, request)
        if input_decision.enforce:
            return AIMessage(content=SAFE_OUTPUT_MESSAGE)
        # Safe tool messages are emitted only for Block decisions; do not
        # reclassify their fixed text to decide whether to cap those denials.
        cap_mode = input_decision.mode or self.mode or "block"
        if self._blocked_count(request) >= self.max_blocked_attempts and cap_mode == "block":
            return AIMessage(content=SAFE_FINAL_MESSAGE)
        return self._filter_output(handler(request), request)

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        request = await self._afilter_remote_notifications(request)
        input_decision = await self._aclassify_decision(self._input(request), HookLabel.USER_INPUT, request)
        if input_decision.enforce:
            return AIMessage(content=SAFE_OUTPUT_MESSAGE)
        cap_mode = input_decision.mode or self.mode or "block"
        if self._blocked_count(request) >= self.max_blocked_attempts and cap_mode == "block":
            return AIMessage(content=SAFE_FINAL_MESSAGE)
        return await self._afilter_output(await handler(request), request)

    def _tool_text(self, request: Any) -> str:
        import json
        return json.dumps(request.tool_call.get("args", {}), ensure_ascii=False, default=str)

    def _safe_tool_result(self, request: Any) -> ToolMessage:
        message = ToolMessage(
            content=SAFE_TOOL_MESSAGE, tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            additional_kwargs={"silmaril_blocked": _BLOCKED_TOOL_MARKER},
        )
        _ISSUED_BLOCKED_MESSAGES[id(message)] = message
        return message

    def _blocked_tool_result(self, request: Any, result: Any) -> Any:
        safe = self._safe_tool_result(request)
        # Keep native async task bookkeeping even when the returned text is
        # withheld, so the parent can still check, update or cancel the task.
        if request.tool_call["name"] in {
            "start_async_task", "check_async_task", "update_async_task",
            "cancel_async_task", "list_async_tasks",
        } and isinstance(result, Command) and isinstance(result.update, dict):
            tasks = result.update.get("async_tasks")
            if isinstance(tasks, dict):
                return Command(update={"messages": [safe], "async_tasks": tasks})
        return safe

    def _allowed_tool_result(self, result: Any) -> Any:
        if isinstance(result, ToolMessage) and result.additional_kwargs.get("silmaril_blocked") == _BLOCKED_TOOL_MARKER:
            if _ISSUED_BLOCKED_MESSAGES.get(id(result)) is result:
                # A second Silmaril middleware may wrap this same denied call.
                # Preserve its marker so the model boundary counts the denial.
                return result
            # Only this middleware may create denial markers. An allowed tool
            # result that copies one must not count as a denied attempt.
            return result.model_copy(update={
                "additional_kwargs": {
                    key: value for key, value in result.additional_kwargs.items()
                    if key != "silmaril_blocked"
                },
            })
        if isinstance(result, Command) and isinstance(result.update, dict):
            messages = result.update.get("messages")
            if isinstance(messages, list):
                result.update["messages"] = [self._allowed_tool_result(message) for message in messages]
        return result

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        name = request.tool_call["name"]
        if self._classify(self._tool_text(request), HookLabel.TOOL_CALL, request, name):
            return self._safe_tool_result(request)
        result = handler(request)
        if self._classify(str(result.content) if isinstance(result, ToolMessage) else str(result), HookLabel.TOOL_RESPONSE, request, name):
            return self._blocked_tool_result(request, result)
        return self._allowed_tool_result(result)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        name = request.tool_call["name"]
        if await self._aclassify(self._tool_text(request), HookLabel.TOOL_CALL, request, name):
            return self._safe_tool_result(request)
        result = await handler(request)
        if await self._aclassify(str(result.content) if isinstance(result, ToolMessage) else str(result), HookLabel.TOOL_RESPONSE, request, name):
            return self._blocked_tool_result(request, result)
        return self._allowed_tool_result(result)


def create_deepagents_middleware(firewall: Firewall | AsyncFirewall, **kwargs: Any) -> SilmarilDeepAgentsMiddleware:
    """Create optional middleware for a Deep Agents graph."""
    return SilmarilDeepAgentsMiddleware(firewall, **kwargs)


def _with_protection(
    middleware: Sequence[Any],
    firewall: Firewall | AsyncFirewall,
    options: dict[str, Any],
) -> list[Any]:
    """Append one Silmaril boundary or retain an equivalent existing one."""
    result = list(middleware)
    existing = [
        entry for entry in result if isinstance(entry, SilmarilDeepAgentsMiddleware)
    ]
    if any(entry.firewall is not firewall for entry in existing):
        raise ValueError(
            "A Deep Agents middleware stack cannot use multiple Firewall clients"
        )
    if len(existing) > 1:
        raise ValueError(
            "A Deep Agents middleware stack must contain exactly one Silmaril "
            "boundary; remove duplicate SilmarilDeepAgentsMiddleware entries"
        )

    requested = create_deepagents_middleware(firewall, **options)
    if existing:
        current = existing[0]
        conflicts = [
            attribute
            for attribute in ("mode", "fail_open", "max_blocked_attempts")
            if getattr(current, attribute) != getattr(requested, attribute)
        ]
        if current.on_classify is not requested.on_classify:
            conflicts.append("on_classify")
        if conflicts:
            raise ValueError(
                "Existing SilmarilDeepAgentsMiddleware conflicts with requested "
                f"middleware_options for: {', '.join(conflicts)}"
            )
        return result

    result.append(requested)
    return result


def _validate_compiled_mode(spec: dict[str, Any]) -> None:
    mode = spec.get("mode")
    if "mode" in spec and mode not in ("isolated", "fork"):
        raise ValueError(
            f"Compiled subagent {spec.get('name')!r} has invalid mode {mode!r}; "
            "expected 'isolated' or 'fork'"
        )


def create_protected_compiled_subagent(
    firewall: Firewall | AsyncFirewall, *, name: str, description: str,
    model: Any, tools: Sequence[Any] = (), middleware: Sequence[Any] = (),
    middleware_options: dict[str, Any] | None = None,
    mode: Literal["isolated", "fork"] | None = None,
    **agent_kwargs: Any,
) -> dict[str, Any]:
    """Compile a subagent with Silmaril installed and register its graph identity."""
    from langchain.agents import create_agent

    reject_conversation_options(agent_kwargs)
    if mode not in (None, "isolated", "fork"):
        raise ValueError("mode must be 'isolated', 'fork', or None")
    options = middleware_options or {}
    runnable = create_agent(
        model=model, tools=list(tools),
        middleware=_with_protection(middleware, firewall, options),
        **agent_kwargs,
    )
    _register_protected_graph(runnable, firewall)
    spec = {"name": name, "description": description, "runnable": runnable}
    if mode is not None:
        spec["mode"] = mode
    return spec


def create_protected_deep_agent(
    firewall: Firewall | AsyncFirewall, *, subagents: Sequence[dict[str, Any]] = (),
    protected_compiled_subagents: Sequence[dict[str, Any]] = (),
    middleware_options: dict[str, Any] | None = None, **kwargs: Any,
) -> Any:
    """Protect this graph and inline children; support native remote subagent specs.

    Deploy remote children with this same constructor on their own Agent Protocol
    server. Parent middleware protects delegation/results, not remote internals.
    """
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:  # pragma: no cover
        raise ImportError('Install "silmaril-security-sdk[deepagents]"') from exc
    from silmaril_security.sdk._deepagents_compat import (
        protected_default_subagent,
        resolve_model_and_profile,
    )

    reject_conversation_options(kwargs)
    options = middleware_options or {}
    supplied_root_middleware = list(kwargs.pop("middleware", ()) or ())
    root_middleware = _with_protection(
        supplied_root_middleware, firewall, options
    )
    protected: list[dict[str, Any]] = []
    for spec in subagents:
        if "runnable" in spec:
            raise ValueError("Compiled subagents must be protected inside their compiled graph")
        if "graph_id" in spec:
            # AsyncSubAgent runs on its server; an injected local middleware field
            # would be ignored. The deployed graph installs its own protection.
            protected.append(dict(spec))
        else:
            protected.append({
                **spec,
                "middleware": _with_protection(
                    spec.get("middleware") or (), firewall, options
                ),
            })
    for spec in protected_compiled_subagents:
        runnable = spec.get("runnable")
        if runnable is None or not _is_protected_graph(runnable, firewall):
            raise ValueError(
                "Compiled subagents must be built with create_protected_compiled_subagent "
                "for this Firewall client"
            )
        _validate_compiled_mode(spec)
        protected.append(dict(spec))

    parent_model, profile = resolve_model_and_profile(kwargs.pop("model", None))
    default = protected_default_subagent(
        profile,
        create_deepagents_middleware(firewall, **options),
        has_explicit_default=any(
            "graph_id" not in spec and spec.get("name") == "general-purpose"
            for spec in protected
        ),
        root_middleware=[
            entry
            for entry in supplied_root_middleware
            if not isinstance(entry, SilmarilDeepAgentsMiddleware)
        ],
        tools=kwargs.get("tools"),
        skills=kwargs.get("skills"),
    )
    if default is not None:
        protected.insert(0, default)
    return create_deep_agent(
        model=parent_model,
        subagents=cast(Any, protected),
        middleware=root_middleware,
        **kwargs,
    )
