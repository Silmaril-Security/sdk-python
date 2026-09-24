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


async def _aread_capped_error_body(
    response: Any,
    transport_error: type[BaseException],
    limit: int = _MAX_ERROR_BODY_BYTES,
) -> str:
    """Read at most ``limit`` bytes of a streamed error body."""
    chunks: list[bytes] = []
    remaining = limit
    try:
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            if len(chunk) >= remaining:
                chunks.append(chunk[:remaining])
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except transport_error:
        # A stream that breaks before the cap is a transport failure, so it
        # follows the retry policy instead of reporting a partial error body.
        raise
    except Exception:
        LOG.debug("failed to read firewall error body", exc_info=True)
    return b"".join(chunks).decode("utf-8", errors="replace")


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
        self._closing = False
        self._closed = False
        self._active_requests = 0
        self._request_tasks: set[asyncio.Task[Any]] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> AsyncFirewall:
        self._ensure_client()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop accepting work, drain active requests, then close an SDK-owned client.

        New classifications are rejected as soon as shutdown starts, while requests
        that are already sending or waiting to retry run to completion. Close is
        idempotent, every caller returns only once an SDK-owned client is actually
        closed, and a caller-injected client stays caller-owned.

        Closing from inside one's own in-flight request raises ``RuntimeError``
        without starting shutdown, because draining would deadlock on the calling
        task and closing early would break that task's own request. An
        ``on_classify`` callback runs after its request completes, so closing from
        a callback is supported.

        Cancelling a task that is awaiting ``aclose()`` cancels only that waiter.
        The drain-and-close work continues on a shared close task, so an SDK-owned
        pool is not leaked. A failed close is logged, re-raised to remaining
        waiters, and can be retried; it does not leave a permanent closing state.
        """
        if self._closed:
            return
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError("AsyncFirewall cannot be used from a different event loop")
        if asyncio.current_task() in self._request_tasks:
            raise RuntimeError(
                "AsyncFirewall.aclose() cannot be called from inside an active "
                "classification on the same task; close it after the call returns"
            )
        await asyncio.shield(self._ensure_close_task())

    def _ensure_close_task(self) -> asyncio.Task[None]:
        if self._close_task is None:
            self._closing = True
            task = asyncio.get_running_loop().create_task(self._drain_and_close())
            task.add_done_callback(self._observe_close_task)
            self._close_task = task
        return self._close_task

    async def _drain_and_close(self) -> None:
        try:
            if self._active_requests:
                await self._idle.wait()
            if self._owns_client and self._client is not None:
                await self._client.aclose()
            self._closed = True
        except BaseException:
            self._closing = False
            self._close_task = None
            raise

    def _observe_close_task(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            LOG.warning("AsyncFirewall close task was cancelled")
            return
        exc = task.exception()
        if exc is not None:
            LOG.warning("AsyncFirewall close failed", exc_info=exc)

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
        # Results must describe the request that was sent, so the inputs are
        # snapshotted before the caller can mutate them during the await.
        sent_texts = list(texts)
        sent_hooks = list(hooks) if hooks is not None else None
        sent_tool_names = list(tool_names) if tool_names is not None else None
        results = await self._classify_batch_raw(
            sent_texts,
            hooks=sent_hooks,
            tool_names=sent_tool_names,
            metadata=metadata,
            request_id=request_id_value,
            mode=requested_mode,
        )
        blocked: list[BlockedBatchItem] = []
        for index, result in enumerate(results):
            hook = sent_hooks[index] if sent_hooks is not None else None
            tool_name = sent_tool_names[index] if sent_tool_names is not None else None
            event = _new_classify_event(
                text=sent_texts[index],
                hook=hook,
                tool_name=tool_name,
                result=result,
            )
            await self._fire_on_classify(event)
            if event.blocked and event.mode == "block":
                blocked.append(
                    BlockedBatchItem(
                        index=index,
                        text=sent_texts[index],
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
        self._begin_request()
        try:
            for attempt in range(self.max_retries + 1):
                request = client.build_request(
                    "POST",
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                retry_after: str | None = None
                should_retry = False
                # Sending and streaming the body share one retry scope, so a
                # transport failure mid-download is retried like a failed send.
                try:
                    response = await client.send(
                        request,
                        stream=True,
                        follow_redirects=False,
                    )
                    try:
                        if (
                            response.status_code in _RETRYABLE_STATUS_CODES
                            and attempt < self.max_retries
                        ):
                            retry_after = response.headers.get("Retry-After")
                            should_retry = True
                        elif response.status_code >= 300:
                            # Read the error body as a capped stream so an oversized
                            # response is never buffered in full just to be truncated.
                            body = await _aread_capped_error_body(
                                response, self._httpx.HTTPError
                            )
                            raise SilmarilApiError(
                                status=response.status_code,
                                status_text=response.reason_phrase,
                                body=body,
                            )
                        else:
                            await response.aread()
                    finally:
                        await response.aclose()
                except self._httpx.HTTPError:
                    if attempt < self.max_retries:
                        await self._sleep_before_retry(attempt, None)
                        continue
                    raise

                if should_retry:
                    await self._sleep_before_retry(attempt, retry_after)
                    continue
                return response.json()
            raise RuntimeError("Firewall: exhausted retries")
        finally:
            self._end_request()

    def _begin_request(self) -> None:
        self._active_requests += 1
        self._idle.clear()
        task = asyncio.current_task()
        if task is not None:
            self._request_tasks.add(task)

    def _end_request(self) -> None:
        self._active_requests -= 1
        task = asyncio.current_task()
        if task is not None:
            self._request_tasks.discard(task)
        if self._active_requests == 0:
            self._idle.set()

    def _ensure_client(self) -> Any:
        if self._closed:
            raise RuntimeError("AsyncFirewall is closed")
        if self._closing:
            raise RuntimeError("AsyncFirewall is closing")
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
