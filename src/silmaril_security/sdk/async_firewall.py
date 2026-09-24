# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Native asynchronous Silmaril Firewall client."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from silmaril_security.sdk.exceptions import (
    BatchFirewallBlockedException,
    FirewallBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk.firewall import (
    _MAX_BACKOFF_SECONDS,
    _MAX_ERROR_BODY_BYTES,
    _RETRYABLE_STATUS_CODES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    _batch_payload,
    _batch_results,
    _block_result_from_json,
    _effective_mode,
    _new_classify_event,
    _retry_after_seconds,
    _single_payload,
    _validate_mode,
)
from silmaril_security.sdk.hooks import HookLabel
from silmaril_security.sdk.types import (
    BlockedBatchItem,
    BlockResult,
    ClassificationMetadata,
    ClassifyEvent,
    FirewallMode,
)

if TYPE_CHECKING:
    import httpx

LOG = logging.getLogger("silmaril_security.sdk")


class AsyncFirewall:
    """Async client for the Silmaril Firewall /classify endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        mode: FirewallMode | None = None,
        shadow_mode: bool | None = None,
        on_classify: Callable[[ClassifyEvent], None | Awaitable[None]] | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised by packaging consumers
            raise ImportError(
                "AsyncFirewall requires the async extra: "
                'pip install "silmaril-security-sdk[async]"'
            ) from exc

        if not api_key:
            raise ValueError("Firewall: api_key is required")
        if not api_url:
            raise ValueError("Firewall: api_url is required")
        if timeout < 0:
            raise ValueError(f"Firewall: timeout must be non-negative, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"Firewall: max_retries must be non-negative, got {max_retries}")

        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout
        self.mode = (
            _validate_mode(mode)
            if mode is not None
            else ("shadow" if shadow_mode else "block" if shadow_mode is False else None)
        )
        self.shadow_mode = self.mode == "shadow"
        self.on_classify = on_classify
        self.max_retries = max_retries
        self._httpx = httpx
        self._client = http_client
        self._owns_client = http_client is None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    async def __aenter__(self) -> AsyncFirewall:
        self._ensure_client()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close an SDK-owned client. Caller-injected clients remain caller-owned."""
        if self._closed:
            return
        if self._loop is not None and asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("AsyncFirewall cannot be used from a different event loop")
        self._closed = True
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def classify(
        self,
        text: str,
        *,
        hook: HookLabel | str | None = None,
        tool_name: str | None = None,
        metadata: ClassificationMetadata | None = None,
        mode: FirewallMode | None = None,
        shadow_mode: bool | None = None,
        request_id: str | None = None,
    ) -> BlockResult:
        """Classify one text without blocking the event loop."""
        request_id_value = request_id or str(uuid4())
        requested_mode = self._effective_mode(mode, shadow_mode)
        result = await self._classify_raw(
            text,
            hook=hook,
            tool_name=tool_name,
            metadata=metadata,
            request_id=request_id_value,
            mode=requested_mode,
        )
        event = _new_classify_event(text=text, hook=hook, tool_name=tool_name, result=result)
        await self._fire_on_classify(event)
        if event.blocked and event.mode == "block":
            raise FirewallBlockedException(
                score=result.score,
                threshold=result.threshold,
                prompt_text=text,
                hook=event.hook,
                tool_name=tool_name,
                result=result,
            )
        return result

    async def classify_batch(
        self,
        texts: Sequence[str],
        *,
        hooks: Sequence[HookLabel | str] | None = None,
        tool_names: Sequence[str | None] | None = None,
        metadata: Sequence[ClassificationMetadata | None] | None = None,
        mode: FirewallMode | None = None,
        shadow_mode: bool | None = None,
        request_id: str | None = None,
    ) -> list[BlockResult]:
        """Classify independent texts in one async request."""
        request_id_value = request_id or str(uuid4())
        requested_mode = self._effective_mode(mode, shadow_mode)
        results = await self._classify_batch_raw(
            texts,
            hooks=hooks,
            tool_names=tool_names,
            metadata=metadata,
            request_id=request_id_value,
            mode=requested_mode,
        )
        blocked: list[BlockedBatchItem] = []
        for index, result in enumerate(results):
            hook = hooks[index] if hooks is not None else None
            tool_name = tool_names[index] if tool_names is not None else None
            event = _new_classify_event(
                text=texts[index],
                hook=hook,
                tool_name=tool_name,
                result=result,
            )
            await self._fire_on_classify(event)
            if event.blocked and event.mode == "block":
                blocked.append(
                    BlockedBatchItem(
                        index=index,
                        text=texts[index],
                        hook=event.hook,
                        tool_name=tool_name,
                        result=result,
                    )
                )
        if blocked:
            raise BatchFirewallBlockedException(blocked=blocked, results=results)
        return results

    def as_async_langchain_handler(self, **options: Any) -> Any:
        """Create an asynchronous LangChain callback handler sharing this client."""
        from silmaril_security.sdk.langchain import AsyncSilmarilFirewallHandler

        return AsyncSilmarilFirewallHandler(self, **options)

    async def _classify_raw(
        self,
        text: str,
        *,
        hook: HookLabel | str | None = None,
        tool_name: str | None = None,
        metadata: ClassificationMetadata | None = None,
        request_id: str,
        mode: FirewallMode | None = None,
    ) -> BlockResult:
        payload = _single_payload(
            text,
            hook=hook,
            tool_name=tool_name,
            metadata=metadata,
            request_id=request_id,
            mode=mode,
        )
        return _block_result_from_json(await self._post_json(payload), mode)

    async def _classify_batch_raw(
        self,
        texts: Sequence[str],
        *,
        hooks: Sequence[HookLabel | str] | None = None,
        tool_names: Sequence[str | None] | None = None,
        metadata: Sequence[ClassificationMetadata | None] | None = None,
        request_id: str,
        mode: FirewallMode | None = None,
    ) -> list[BlockResult]:
        text_list, payload = _batch_payload(
            texts,
            hooks=hooks,
            tool_names=tool_names,
            metadata=metadata,
            request_id=request_id,
            mode=mode,
        )
        data = await self._post_json(payload)
        return _batch_results(data, expected_length=len(text_list), mode=mode)

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        headers = {"x-api-key": self.api_key, "content-type": "application/json"}
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    follow_redirects=False,
                )
            except self._httpx.HTTPError:
                if attempt < self.max_retries:
                    await self._sleep_before_retry(attempt, None)
                    continue
                raise

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                await response.aclose()
                await self._sleep_before_retry(attempt, retry_after)
                continue
            if response.status_code >= 300:
                try:
                    content = getattr(response, "content", None)
                    if content is None:
                        content = str(response.text).encode()
                    body = content[:_MAX_ERROR_BODY_BYTES].decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                await response.aclose()
                raise SilmarilApiError(
                    status=response.status_code,
                    status_text=response.reason_phrase,
                    body=body,
                )
            return response.json()
        raise RuntimeError("Firewall: exhausted retries")

    def _ensure_client(self) -> Any:
        if self._closed:
            raise RuntimeError("AsyncFirewall is closed")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError("AsyncFirewall cannot be used from a different event loop")
        if self._client is None:
            self._client = self._httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,
            )
        elif getattr(self._client, "is_closed", False):
            raise RuntimeError("AsyncFirewall HTTP client is closed")
        return self._client

    async def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        parsed_retry_after = _retry_after_seconds(retry_after)
        wait = (
            parsed_retry_after
            if parsed_retry_after is not None
            else min(2**attempt, _MAX_BACKOFF_SECONDS)
        )
        LOG.debug("retrying firewall request in %.2fs after attempt %d", wait, attempt + 1)
        await asyncio.sleep(wait)

    def _effective_mode(
        self,
        mode: FirewallMode | None,
        shadow_mode: bool | None,
    ) -> FirewallMode | None:
        return _effective_mode(self.mode, mode, shadow_mode)

    async def _fire_on_classify(self, event: ClassifyEvent) -> None:
        if self.on_classify is None:
            return
        try:
            callback_result = self.on_classify(event)
            if inspect.isawaitable(callback_result):
                await callback_result
        except Exception:
            LOG.warning("on_classify callback raised", exc_info=True)
