# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Optional LangChain callback handlers for Silmaril Firewall."""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

from silmaril_security.sdk._utils import (
    _TOOL_ROLES,
    extract_content_text,
    extract_last_user_text,
    extract_text_from_documents,
    extract_text_from_llm_result,
    extract_text_from_prompts,
    extract_text_from_tool_input,
    get_content,
    get_role,
)
from silmaril_security.sdk.async_firewall import AsyncFirewall
from silmaril_security.sdk.exceptions import FirewallBlockedException
from silmaril_security.sdk.firewall import Firewall
from silmaril_security.sdk.hooks import (
    FIREWALL_HOOK_TO_LABEL,
    FirewallHook,
    HookLabel,
    resolve_hooks,
)
from silmaril_security.sdk.types import (
    BlockResult,
    ClassificationMetadata,
    ClassifyEvent,
    FirewallMode,
)

try:
    from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised by packaging consumers
    raise ImportError(
        "LangChain support requires the langchain extra: "
        'pip install "silmaril-security-sdk[langchain]"'
    ) from exc

LOG = logging.getLogger("silmaril_security.sdk.langchain")

# In-flight chat/LLM runs remembered so the matching output can reuse the model.
# Completed and abandoned runs are removed; the cap drops the oldest if ends never arrive.
_MAX_TRACKED_MODEL_RUNS = 256
_MAX_AGENT_MODEL_ID_LENGTH = 256
# Call selection wins over the constructor, which wins over tracing metadata.
_INVOCATION_MODEL_KEYS = ("model", "model_name", "model_id")
_SERIALIZED_MODEL_KEYS = ("model", "model_name", "model_id")
_METADATA_MODEL_KEYS = ("ls_model_name", "model_name", "model_id", "model")


def _nonempty_model_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    model_id = value.strip()
    if not model_id or len(model_id) > _MAX_AGENT_MODEL_ID_LENGTH:
        return None
    if any(ord(char) < 32 for char in model_id):
        return None
    return model_id


def _serialized_class_paths(serialized: Any) -> set[str]:
    """Class-path identities that must not be reported as a selected model."""
    if not isinstance(serialized, Mapping):
        return set()
    raw_id = serialized.get("id")
    paths: set[str] = set()
    if isinstance(raw_id, str):
        stripped = raw_id.strip()
        if stripped:
            paths.add(stripped)
        return paths
    if isinstance(raw_id, Sequence) and not isinstance(raw_id, (str, bytes)):
        parts = [part.strip() for part in raw_id if isinstance(part, str) and part.strip()]
        if parts:
            paths.add(".".join(parts))
            paths.add("/".join(parts))
    return paths


def _model_id_from_mapping(
    mapping: Any,
    keys: tuple[str, ...],
    *,
    class_paths: set[str],
) -> str | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        model_id = _nonempty_model_id(mapping.get(key))
        if model_id is None or model_id in class_paths:
            continue
        return model_id
    return None


def _selected_agent_model_id(serialized: Any, callback_kwargs: Mapping[str, Any]) -> str | None:
    """Return the model selected for this callback, or None when it is not trustworthy."""
    class_paths = _serialized_class_paths(serialized)
    invocation_params = callback_kwargs.get("invocation_params")
    model_id = _model_id_from_mapping(
        invocation_params,
        _INVOCATION_MODEL_KEYS,
        class_paths=class_paths,
    )
    if model_id is not None:
        return model_id
    serialized_kwargs = serialized.get("kwargs") if isinstance(serialized, Mapping) else None
    model_id = _model_id_from_mapping(
        serialized_kwargs,
        _SERIALIZED_MODEL_KEYS,
        class_paths=class_paths,
    )
    if model_id is not None:
        return model_id
    return _model_id_from_mapping(
        callback_kwargs.get("metadata"),
        _METADATA_MODEL_KEYS,
        class_paths=class_paths,
    )


def _agent_model_metadata(model_id: str | None) -> ClassificationMetadata | None:
    if model_id is None:
        return None
    return {"silmaril": {"agent_model_id": model_id}}


class _RunModelIds:
    """Bounded run_id -> model id map. Missing and finished runs stay absent."""

    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._lock = threading.Lock()

    def remember(self, run_id: UUID | str, model_id: str | None) -> None:
        key = str(run_id)
        with self._lock:
            self._ids.pop(key, None)
            if model_id is None:
                return
            self._ids[key] = model_id
            while len(self._ids) > _MAX_TRACKED_MODEL_RUNS:
                del self._ids[next(iter(self._ids))]

    def pop(self, run_id: UUID | str) -> str | None:
        with self._lock:
            return self._ids.pop(str(run_id), None)


class SilmarilFirewallHandler(BaseCallbackHandler):
    """Synchronous LangChain callback handler.

    Infrastructure errors are fail-open by default. Set ``fail_open=False`` to
    propagate API and transport failures.
    """

    raise_error: bool = True
    run_inline: bool = True

    def __init__(
        self,
        firewall: Firewall,
        *,
        hooks: Iterable[FirewallHook | str] | None = None,
        include_system: bool = True,
        include_tool: bool = True,
        fail_open: bool = True,
        mode: FirewallMode | None = None,
        shadow_mode: bool | None = None,
        on_classify: Callable[[ClassifyEvent], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.firewall = firewall
        self._enabled_hooks = resolve_hooks(hooks)
        self.include_system = include_system
        self.include_tool = include_tool
        self.fail_open = fail_open
        self.mode = firewall._effective_mode(mode, shadow_mode)
        self.shadow_mode = self.mode == "shadow"
        self.on_classify = on_classify
        self.logger = logger or LOG
        self._run_models = _RunModelIds()

    def _remember_selected_model(
        self,
        serialized: Any,
        run_id: UUID | str,
        callback_kwargs: Mapping[str, Any],
    ) -> str | None:
        model_id = _selected_agent_model_id(serialized, callback_kwargs)
        self._run_models.remember(run_id, model_id)
        return model_id

    def _fire_on_classify(self, event: ClassifyEvent) -> None:
        if self.on_classify is None:
            return
        try:
            self.on_classify(event)
        except Exception:
            self.logger.warning("on_classify callback raised", exc_info=True)

    def _classify(
        self,
        text: str,
        run_id: UUID | str,
        hook_label: HookLabel,
        tool_name: str | None = None,
        metadata: ClassificationMetadata | None = None,
    ) -> None:
        try:
            request: dict[str, Any] = {}
            if metadata is not None:
                request["metadata"] = metadata
            result = self.firewall._classify_raw(
                text,
                hook=hook_label,
                tool_name=tool_name,
                request_id=str(run_id),
                mode=self.mode,
                **request,
            )
        except Exception:
            if not self.fail_open:
                raise
            self.logger.warning(
                "Firewall classification failed, allowing prompt through",
                exc_info=True,
            )
            return

        blocked = result.prediction == "MALICIOUS"
        effective_mode = self.mode or result.mode or "block"
        event = ClassifyEvent(
            hook=hook_label,
            tool_name=tool_name,
            text=text,
            result=result,
            blocked=blocked,
            mode=effective_mode,
            shadow_mode=effective_mode == "shadow",
        )
        self._fire_on_classify(event)
        if blocked and effective_mode == "block":
            raise FirewallBlockedException(
                score=result.score,
                threshold=result.threshold,
                prompt_text=text,
                hook=hook_label,
                tool_name=tool_name,
                result=result,
                run_id=run_id,
            )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model_id = self._remember_selected_model(serialized, run_id, kwargs)
        if FirewallHook.CHAT_MODEL_START not in self._enabled_hooks:
            return

        metadata = _agent_model_metadata(model_id)
        all_messages: list[Any] = []
        for batch in messages:
            all_messages.extend(batch)

        text = extract_last_user_text(all_messages)
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.CHAT_MODEL_START],
                metadata=metadata,
            )

        if self.include_tool:
            for msg in all_messages:
                if get_role(msg) in _TOOL_ROLES:
                    tool_text = extract_content_text(get_content(msg)).strip()
                    if tool_text:
                        self._classify(
                            tool_text,
                            run_id,
                            HookLabel.TOOL_RESPONSE,
                            tool_name=getattr(msg, "name", None),
                            metadata=metadata,
                        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model_id = self._remember_selected_model(serialized, run_id, kwargs)
        if FirewallHook.LLM_START not in self._enabled_hooks:
            return
        text = extract_text_from_prompts(prompts)
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_START],
                metadata=_agent_model_metadata(model_id),
            )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.TOOL_START not in self._enabled_hooks:
            return
        text = extract_text_from_tool_input(input_str)
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_START],
                tool_name=serialized.get("name") or kwargs.get("name"),
            )

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.RETRIEVER_START not in self._enabled_hooks:
            return
        text = query.strip()
        if text:
            self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.RETRIEVER_START])

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        model_id = self._run_models.pop(run_id)
        if FirewallHook.LLM_END not in self._enabled_hooks:
            return
        text = extract_text_from_llm_result(response)
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_END],
                metadata=_agent_model_metadata(model_id),
            )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del error, kwargs
        self._run_models.pop(run_id)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if FirewallHook.TOOL_END not in self._enabled_hooks:
            return
        text = str(output).strip()
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_END],
                tool_name=kwargs.get("name"),
            )

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.RETRIEVER_END not in self._enabled_hooks:
            return
        text = extract_text_from_documents(documents)
        if text:
            self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.RETRIEVER_END])


class AsyncSilmarilFirewallHandler(AsyncCallbackHandler):
    """Asynchronous LangChain callback handler."""

    raise_error: bool = True
    run_inline: bool = True

    def __init__(
        self,
        firewall: Firewall | AsyncFirewall,
        *,
        hooks: Iterable[FirewallHook | str] | None = None,
        include_system: bool = True,
        include_tool: bool = True,
        fail_open: bool = True,
        mode: FirewallMode | None = None,
        shadow_mode: bool | None = None,
        on_classify: Callable[[ClassifyEvent], None | Awaitable[None]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self._sync_handler = SilmarilFirewallHandler(
            firewall,
            hooks=hooks,
            include_system=include_system,
            include_tool=include_tool,
            fail_open=fail_open,
            mode=mode,
            shadow_mode=shadow_mode,
            on_classify=None,
            logger=logger,
        )
        self.on_classify = on_classify
        self.logger = logger or LOG

    async def _fire_on_classify(self, event: ClassifyEvent) -> None:
        if self.on_classify is None:
            return
        try:
            result = self.on_classify(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            self.logger.warning("on_classify callback raised", exc_info=True)

    async def _classify(
        self,
        text: str,
        run_id: UUID | str,
        hook_label: HookLabel,
        tool_name: str | None = None,
        metadata: ClassificationMetadata | None = None,
    ) -> None:
        try:
            request: dict[str, Any] = {}
            if metadata is not None:
                request["metadata"] = metadata
            result = await _async_classify_raw(
                self._sync_handler.firewall,
                text,
                hook=hook_label,
                tool_name=tool_name,
                request_id=str(run_id),
                mode=self._sync_handler.mode,
                **request,
            )
        except Exception:
            if not self._sync_handler.fail_open:
                raise
            self.logger.warning(
                "Firewall classification failed, allowing prompt through",
                exc_info=True,
            )
            return

        blocked = result.prediction == "MALICIOUS"
        effective_mode = self._sync_handler.mode or result.mode or "block"
        event = ClassifyEvent(
            hook=hook_label,
            tool_name=tool_name,
            text=text,
            result=result,
            blocked=blocked,
            mode=effective_mode,
            shadow_mode=effective_mode == "shadow",
        )
        await self._fire_on_classify(event)
        if blocked and effective_mode == "block":
            raise FirewallBlockedException(
                score=result.score,
                threshold=result.threshold,
                prompt_text=text,
                hook=hook_label,
                tool_name=tool_name,
                result=result,
                run_id=run_id,
            )

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model_id = self._sync_handler._remember_selected_model(serialized, run_id, kwargs)
        if FirewallHook.CHAT_MODEL_START not in self._sync_handler._enabled_hooks:
            return
        metadata = _agent_model_metadata(model_id)
        all_messages: list[Any] = []
        for batch in messages:
            all_messages.extend(batch)
        text = extract_last_user_text(all_messages)
        if text:
            await self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.CHAT_MODEL_START],
                metadata=metadata,
            )
        if self._sync_handler.include_tool:
            for msg in all_messages:
                if get_role(msg) in _TOOL_ROLES:
                    tool_text = extract_content_text(get_content(msg)).strip()
                    if tool_text:
                        await self._classify(
                            tool_text,
                            run_id,
                            HookLabel.TOOL_RESPONSE,
                            tool_name=getattr(msg, "name", None),
                            metadata=metadata,
                        )

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model_id = self._sync_handler._remember_selected_model(serialized, run_id, kwargs)
        if FirewallHook.LLM_START not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_prompts(prompts)
        if text:
            await self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_START],
                metadata=_agent_model_metadata(model_id),
            )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.TOOL_START not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_tool_input(input_str)
        if text:
            await self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_START],
                tool_name=serialized.get("name") or kwargs.get("name"),
            )

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.RETRIEVER_START not in self._sync_handler._enabled_hooks:
            return
        text = query.strip()
        if text:
            await self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.RETRIEVER_START])

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        model_id = self._sync_handler._run_models.pop(run_id)
        if FirewallHook.LLM_END not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_llm_result(response)
        if text:
            await self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_END],
                metadata=_agent_model_metadata(model_id),
            )

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del error, kwargs
        self._sync_handler._run_models.pop(run_id)

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if FirewallHook.TOOL_END not in self._sync_handler._enabled_hooks:
            return
        text = str(output).strip()
        if text:
            await self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_END],
                tool_name=kwargs.get("name"),
            )

    async def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.RETRIEVER_END not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_documents(documents)
        if text:
            await self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.RETRIEVER_END])


async def _async_classify_raw(
    firewall: Firewall | AsyncFirewall,
    text: str,
    *,
    hook: HookLabel | str | None,
    tool_name: str | None,
    metadata: ClassificationMetadata | None = None,
    request_id: str | None = None,
    mode: FirewallMode | None = None,
) -> BlockResult:
    request_id_value = request_id or str(uuid4())
    if isinstance(firewall, AsyncFirewall):
        return await firewall._classify_raw(
            text,
            hook=hook,
            tool_name=tool_name,
            metadata=metadata,
            request_id=request_id_value,
            mode=mode,
        )

    async with AsyncFirewall(
        api_key=firewall.api_key,
        api_url=firewall.api_url,
        timeout=firewall.timeout,
        mode=firewall.mode,
        max_retries=firewall.max_retries,
    ) as async_firewall:
        return await async_firewall._classify_raw(
            text,
            hook=hook,
            tool_name=tool_name,
            metadata=metadata,
            request_id=request_id_value,
            mode=mode,
        )
