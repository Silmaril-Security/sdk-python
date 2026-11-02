# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Optional LangChain callback handlers for Silmaril Firewall."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from functools import partial
from typing import Any, TypeVar
from uuid import UUID, uuid4

from silmaril_security.sdk._thread_identity import (
    ThreadRuns,
    reject_conversation_options,
    thread_identity,
    track_callback,
)
from silmaril_security.sdk._utils import (
    extract_last_user_text,
    extract_text_from_documents,
    extract_text_from_llm_result,
    extract_text_from_prompts,
    extract_text_from_tool_input,
)
from silmaril_security.sdk.async_firewall import AsyncFirewall
from silmaril_security.sdk.exceptions import FirewallBlockedException, ReviewCompatibilityError
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
    from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig
    from langchain_core.runnables.base import RunnableBindingBase

    from silmaril_security.sdk._langchain_runnable import PreservingRunnable, register_handler
except ImportError as exc:  # pragma: no cover - exercised by packaging consumers
    raise ImportError(
        "LangChain support requires the langchain extra: "
        'pip install "silmaril-security-sdk[langchain]"'
    ) from exc

LOG = logging.getLogger("silmaril_security.sdk.langchain")
_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


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
        **removed_options: Any,
    ) -> None:
        reject_conversation_options(removed_options)
        if removed_options:
            raise TypeError(f"Unexpected options: {sorted(removed_options)}")
        super().__init__()
        self._thread_runs = ThreadRuns()
        self.firewall = firewall
        self._enabled_hooks = resolve_hooks(hooks)
        self.include_system = include_system
        self.include_tool = include_tool
        self.fail_open = fail_open
        self.mode = firewall._effective_mode(mode, shadow_mode)
        self.shadow_mode = self.mode == "shadow"
        self.on_classify = on_classify
        self.logger = logger or LOG

    @track_callback(start=True)
    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    @track_callback(start=False)
    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    @track_callback(start=False)
    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    on_llm_error = on_chain_error
    on_tool_error = on_chain_error
    on_retriever_error = on_chain_error

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
    ) -> None:
        try:
            result = self.firewall._classify_raw(
                text,
                hook=hook_label,
                tool_name=tool_name,
                metadata=self._thread_runs.metadata(run_id),
                request_id=str(uuid4()),
                mode=self.mode,
            )
        except ReviewCompatibilityError:
            raise
        except Exception:
            if not self.fail_open:
                raise
            self.logger.warning(
                "Firewall classification failed, allowing prompt through",
                exc_info=True,
            )
            return

        blocked = result.prediction == "MALICIOUS" or (
            result.governance is not None and result.governance.action == "block"
        )
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

    @track_callback(start=True)
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.CHAT_MODEL_START not in self._enabled_hooks:
            return

        all_messages: list[Any] = []
        for batch in messages:
            all_messages.extend(batch)

        text = extract_last_user_text(all_messages)
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.CHAT_MODEL_START],
            )


    @track_callback(start=True)
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.LLM_START not in self._enabled_hooks:
            return
        text = extract_text_from_prompts(prompts)
        if text:
            self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_START])

    @track_callback(start=True)
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not self.include_tool or FirewallHook.TOOL_START not in self._enabled_hooks:
            return
        text = extract_text_from_tool_input(input_str)
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_START],
                tool_name=serialized.get("name") or kwargs.get("name"),
            )

    @track_callback(start=True)
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

    @track_callback(start=False)
    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if FirewallHook.LLM_END not in self._enabled_hooks:
            return
        text = extract_text_from_llm_result(response)
        if text:
            self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_END])

    @track_callback(start=False)
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not self.include_tool or FirewallHook.TOOL_END not in self._enabled_hooks:
            return
        text = str(output).strip()
        if text:
            self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_END],
                tool_name=kwargs.get("name"),
            )

    @track_callback(start=False)
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
        **removed_options: Any,
    ) -> None:
        reject_conversation_options(removed_options)
        if removed_options:
            raise TypeError(f"Unexpected options: {sorted(removed_options)}")
        super().__init__()
        self._thread_runs = ThreadRuns()
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

    @track_callback(start=True)
    async def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    @track_callback(start=False)
    async def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    @track_callback(start=False)
    async def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        pass

    on_llm_error = on_chain_error
    on_tool_error = on_chain_error
    on_retriever_error = on_chain_error

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
    ) -> None:
        try:
            result = await _async_classify_raw(
                self._sync_handler.firewall,
                text,
                hook=hook_label,
                tool_name=tool_name,
                metadata=self._thread_runs.metadata(run_id),
                request_id=str(uuid4()),
                mode=self._sync_handler.mode,
            )
        except ReviewCompatibilityError:
            raise
        except Exception:
            if not self._sync_handler.fail_open:
                raise
            self.logger.warning(
                "Firewall classification failed, allowing prompt through",
                exc_info=True,
            )
            return

        blocked = result.prediction == "MALICIOUS" or (
            result.governance is not None and result.governance.action == "block"
        )
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

    @track_callback(start=True)
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.CHAT_MODEL_START not in self._sync_handler._enabled_hooks:
            return
        all_messages: list[Any] = []
        for batch in messages:
            all_messages.extend(batch)
        text = extract_last_user_text(all_messages)
        if text:
            await self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.CHAT_MODEL_START])

    @track_callback(start=True)
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if FirewallHook.LLM_START not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_prompts(prompts)
        if text:
            await self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_START])

    @track_callback(start=True)
    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not self._sync_handler.include_tool or FirewallHook.TOOL_START not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_tool_input(input_str)
        if text:
            await self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_START],
                tool_name=serialized.get("name") or kwargs.get("name"),
            )

    @track_callback(start=True)
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

    @track_callback(start=False)
    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if FirewallHook.LLM_END not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_llm_result(response)
        if text:
            await self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_END])

    @track_callback(start=False)
    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not self._sync_handler.include_tool or FirewallHook.TOOL_END not in self._sync_handler._enabled_hooks:
            return
        text = str(output).strip()
        if text:
            await self._classify(
                text,
                run_id,
                FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_END],
                tool_name=kwargs.get("name"),
            )

    @track_callback(start=False)
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
        review=firewall.review,
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


def protect_langchain_runnable(
    firewall: Firewall | AsyncFirewall, runnable: Runnable[_Input, _Output], **handler_options: Any,
) -> Runnable[_Input, _Output]:
    """Protect a runnable using its native config at every invocation boundary.

    The standard binding retains normal invoke/ainvoke, batch and streaming
    behavior. Existing callbacks and configuration are merged by LangChain.
    Use asynchronous invocation with ``AsyncFirewall``.
    """
    from langchain_core.runnables.config import ensure_config, merge_configs

    reject_conversation_options(handler_options)
    if isinstance(runnable, RunnableBindingBase):
        # Keep existing bindings outside protection. Their saved defaults and
        # dynamic config factories then run exactly once before we snapshot
        # native identity at the actual model/tool invocation boundary.
        protected = protect_langchain_runnable(firewall, runnable.bound, **handler_options)
        copy = getattr(runnable, "model_copy", None) or runnable.copy
        return PreservingRunnable(copy(update={"bound": protected}))

    # Validate options eagerly; each actual invocation gets an independent
    # handler so no thread identity is stored on the reusable binding.
    handler_type = AsyncSilmarilFirewallHandler if isinstance(firewall, AsyncFirewall) else SilmarilFirewallHandler
    handler_type(firewall, **handler_options)

    def make_callbacks(config):
        handler = handler_type(firewall, **handler_options)
        register_handler(handler)
        normalized = ensure_config(config)
        _, identity = thread_identity(normalized, ambient=False)
        handler._thread_snapshot = (True, identity)
        # Binding merges explicit callbacks itself. When callbacks are only
        # inherited from a parent runnable, retain that manager before adding
        # ours; otherwise a fresh callback list would replace it.
        callbacks = {"callbacks": [handler]}
        if config.get("callbacks") is None:
            callbacks = merge_configs({"callbacks": normalized.get("callbacks")}, callbacks)
        return handler, callbacks

    def native_callbacks(config, *, created_handlers=None):
        handler, callbacks = make_callbacks(config)
        if created_handlers is not None:
            created_handlers.append(handler)
        return callbacks

    class _ProtectedBinding(RunnableBinding[_Input, _Output]):
        def __init__(self, **data: Any) -> None:
            # Older core releases rebuild bindings without config_factories
            # in with_config/bind/with_types. Never lose the protection factory.
            factories = list(data.get("config_factories") or [])
            if native_callbacks not in factories:
                factories.append(native_callbacks)
            super().__init__(**{**data, "config_factories": factories})

        def with_retry(self, **kwargs: Any) -> Runnable[_Input, _Output]:
            # Retry the protected boundary, rather than moving the callback
            # factory outside an unprotected model retry runnable.
            return Runnable.with_retry(self, **kwargs)

        def _stream_binding(self, created_handlers):
            return RunnableBinding(
                bound=self.bound, kwargs=self.kwargs, config=self.config,
                config_factories=[
                    partial(native_callbacks, created_handlers=created_handlers)
                    if factory is native_callbacks else factory
                    for factory in self.config_factories
                ],
                custom_input_type=self.custom_input_type,
                custom_output_type=self.custom_output_type,
                name=self.name,
            )

        def stream(self, input: _Input, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[_Output]:
            created = []
            try:
                yield from self._stream_binding(created).stream(input, config, **kwargs)
            finally:
                for handler in created:
                    handler._thread_runs.clear()

        async def astream(self, input: _Input, config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[_Output]:
            created = []
            iterator = self._stream_binding(created).astream(input, config, **kwargs)
            try:
                async for chunk in iterator:
                    yield chunk
            finally:
                try:
                    close = getattr(iterator, "aclose", None)
                    if close is not None:
                        result = close()
                        if inspect.isawaitable(result):
                            await result
                finally:
                    for handler in created:
                        handler._thread_runs.clear()

    return PreservingRunnable(_ProtectedBinding(bound=runnable, config_factories=[native_callbacks]))
