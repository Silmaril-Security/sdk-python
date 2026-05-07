# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Optional LangChain callback handlers for Silmaril Firewall."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any
from uuid import UUID

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
from silmaril_security.sdk.exceptions import PromptBlockedException
from silmaril_security.sdk.firewall import Firewall, adaptive_threshold
from silmaril_security.sdk.hooks import (
    FIREWALL_HOOK_TO_LABEL,
    FirewallHook,
    HookLabel,
    resolve_hooks,
)
from silmaril_security.sdk.types import BlockResult, ClassifyEvent

try:
    from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised by packaging consumers
    raise ImportError(
        "LangChain support requires the langchain extra: "
        'pip install "silmaril-security-sdk[langchain]"'
    ) from exc

LOG = logging.getLogger("silmaril_security.sdk.langchain")


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
        self.shadow_mode = firewall.shadow_mode if shadow_mode is None else shadow_mode
        self.on_classify = on_classify
        self.logger = logger or LOG

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
            )
        except Exception:
            if not self.fail_open:
                raise
            self.logger.warning(
                "Firewall classification failed, allowing prompt through",
                exc_info=True,
            )
            return

        blocked = result.score >= result.threshold
        event = ClassifyEvent(
            hook=hook_label,
            tool_name=tool_name,
            text=text,
            result=result,
            blocked=blocked,
            shadow_mode=self.shadow_mode,
        )
        self._fire_on_classify(event)
        if blocked and not self.shadow_mode:
            raise PromptBlockedException(
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
                        )

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
        if FirewallHook.LLM_END not in self._enabled_hooks:
            return
        text = extract_text_from_llm_result(response)
        if text:
            self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_END])

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
        firewall: Firewall,
        *,
        hooks: Iterable[FirewallHook | str] | None = None,
        include_system: bool = True,
        include_tool: bool = True,
        fail_open: bool = True,
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
    ) -> None:
        try:
            result = await _async_classify_raw(
                self._sync_handler.firewall,
                text,
                hook=hook_label,
                tool_name=tool_name,
            )
        except Exception:
            if not self._sync_handler.fail_open:
                raise
            self.logger.warning(
                "Firewall classification failed, allowing prompt through",
                exc_info=True,
            )
            return

        blocked = result.score >= result.threshold
        event = ClassifyEvent(
            hook=hook_label,
            tool_name=tool_name,
            text=text,
            result=result,
            blocked=blocked,
            shadow_mode=self._sync_handler.shadow_mode,
        )
        await self._fire_on_classify(event)
        if blocked and not self._sync_handler.shadow_mode:
            raise PromptBlockedException(
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
        if FirewallHook.CHAT_MODEL_START not in self._sync_handler._enabled_hooks:
            return
        all_messages: list[Any] = []
        for batch in messages:
            all_messages.extend(batch)
        text = extract_last_user_text(all_messages)
        if text:
            await self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.CHAT_MODEL_START])
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
                        )

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
        if FirewallHook.LLM_END not in self._sync_handler._enabled_hooks:
            return
        text = extract_text_from_llm_result(response)
        if text:
            await self._classify(text, run_id, FIREWALL_HOOK_TO_LABEL[FirewallHook.LLM_END])

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
    firewall: Firewall,
    text: str,
    *,
    hook: HookLabel | str | None,
    tool_name: str | None,
) -> BlockResult:
    import asyncio

    import httpx

    from silmaril_security.sdk.firewall import _block_result_from_json
    from silmaril_security.sdk.hooks import hook_value

    chunks = __import__("silmaril_security.sdk.chunking", fromlist=["chunk_text"]).chunk_text(text)
    threshold = adaptive_threshold(len(chunks))
    headers = {"x-api-key": firewall.api_key, "content-type": "application/json"}
    async with httpx.AsyncClient(
        headers=headers,
        timeout=firewall.timeout,
        follow_redirects=False,
    ) as client:
        if len(chunks) == 1:
            payload: dict[str, Any] = {"text": chunks[0], "threshold": threshold}
            hook_str = hook_value(hook)
            if hook_str:
                payload["hook"] = hook_str
            if tool_name:
                payload["tool_name"] = tool_name
            data = await _async_post_json(client, firewall, payload)
            return _block_result_from_json(data, threshold)

        semaphore = asyncio.Semaphore(firewall.chunk_concurrency)

        async def classify_chunk(chunk: str) -> BlockResult:
            payload: dict[str, Any] = {"text": chunk, "threshold": threshold}
            hook_str = hook_value(hook)
            if hook_str:
                payload["hook"] = hook_str
            if tool_name:
                payload["tool_name"] = tool_name
            async with semaphore:
                data = await _async_post_json(client, firewall, payload)
            return _block_result_from_json(data, threshold)

        chunk_results = await asyncio.gather(
            *(classify_chunk(chunk) for chunk in chunks),
            return_exceptions=True,
        )
        for result in chunk_results:
            if isinstance(result, BaseException):
                raise result
        results = [result for result in chunk_results if isinstance(result, BlockResult)]
        return max(results, key=lambda result: result.score)


async def _async_post_json(client: Any, firewall: Firewall, payload: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    import httpx

    from silmaril_security.sdk.firewall import (
        _MAX_ERROR_BODY_BYTES,
        _RETRYABLE_STATUS_CODES,
        _retry_after_seconds,
    )

    for attempt in range(firewall.max_retries + 1):
        try:
            response = await client.post(
                firewall.api_url,
                json=payload,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            if attempt < firewall.max_retries:
                await asyncio.sleep(min(2**attempt, 30.0))
                continue
            raise
        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < firewall.max_retries:
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            await response.aclose()
            await asyncio.sleep(retry_after if retry_after is not None else min(2**attempt, 30.0))
            continue
        if response.status_code >= 300:
            try:
                body = response.text[:_MAX_ERROR_BODY_BYTES]
            except Exception:
                body = ""
            await response.aclose()
            raise __import__(
                "silmaril_security.sdk.exceptions",
                fromlist=["SilmarilApiError"],
            ).SilmarilApiError(
                status=response.status_code,
                status_text=response.reason_phrase,
                body=body,
            )
        return response.json()
    raise RuntimeError("Firewall: exhausted retries")
