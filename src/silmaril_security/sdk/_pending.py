# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Classifications that run beside the host flow and are joined at a later boundary."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextvars import ContextVar, copy_context
from copy import deepcopy
from threading import Lock
from typing import Any, TypeVar

LOG = logging.getLogger("silmaril_security.sdk")

DEFAULT_MAX_CONCURRENT = 32
DEFAULT_MAX_QUEUED = 64
BACKGROUND_RETRIES = 1
RETRY_LIMIT: ContextVar[int | None] = ContextVar("silmaril_retry_limit", default=None)

_Verdict = TypeVar("_Verdict")
_Result = TypeVar("_Result")


def conversation_id(metadata: Any) -> str | None:
    value = metadata.get("conversationId") if isinstance(metadata, Mapping) else None
    return value if isinstance(value, str) and value else None


def snapshot(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Copy caller-owned request data so a later send cannot observe its mutation."""
    return None if metadata is None else deepcopy(dict(metadata))


def validate_limits(max_concurrent: int, max_queued: int) -> None:
    if max_concurrent < 1:
        raise ValueError(f"Firewall: max_concurrent must be positive, got {max_concurrent}")
    if max_queued < 1:
        raise ValueError(f"Firewall: max_queued must be positive, got {max_queued}")


def _report(done: Future[Any] | asyncio.Task[Any]) -> None:
    if not done.cancelled() and done.exception() is not None:
        LOG.warning("background classification failed", exc_info=done.exception())


class _Depths:
    """Bounded counts of unjoined classifications, per conversation and in the background."""

    def __init__(self, max_concurrent: int, max_queued: int) -> None:
        self._max_queued = max_queued
        self._max_background = max_concurrent * max_queued
        self._background = 0
        self._depths: dict[str, int] = {}

    def enter(self, conversation: str | None, background: bool) -> bool:
        depth = self._depths.get(conversation, 0) if conversation is not None else 0
        if background and (depth >= self._max_queued or self._background >= self._max_background):
            LOG.warning("dropped background classification: queue is full")
            return False
        if conversation is not None:
            self._depths[conversation] = depth + 1
        self._background += background
        return True

    def leave(self, conversation: str | None, background: bool) -> None:
        self._background -= background
        if conversation is None:
            return
        depth = self._depths[conversation] - 1
        if depth:
            self._depths[conversation] = depth
        else:
            del self._depths[conversation]


class PendingVerdicts:
    """Thread-backed classifications sent in order within one conversation."""

    def __init__(self, max_concurrent: int, max_queued: int) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="silmaril-classify"
        )
        self._lock = Lock()
        self._depths = _Depths(max_concurrent, max_queued)
        self._tails: dict[str, Future[Any]] = {}
        self._outstanding: set[Future[Any]] = set()

    def submit(
        self,
        call: Callable[[], _Verdict],
        conversation: str | None = None,
        *,
        background: bool = False,
    ) -> Future[_Verdict | None]:
        """Start a classification; a background one is never joined by its caller.

        Cancelling the returned future before it is sent releases its place in the queue.
        """
        pending: Future[_Verdict | None] = Future()
        context = copy_context()
        if background:
            context.run(RETRY_LIMIT.set, BACKGROUND_RETRIES)
            pending.add_done_callback(_report)
        with self._lock:
            if not self._depths.enter(conversation, background):
                pending.set_result(None)
                return pending
            previous = self._tails.get(conversation) if conversation is not None else None
            if conversation is not None:
                self._tails[conversation] = pending
            self._outstanding.add(pending)

        def start(_: Any = None) -> None:
            self._executor.submit(
                self._run, context.run, call, pending, conversation, background
            )

        if previous is None:
            start()
        else:
            previous.add_done_callback(start)
        return pending

    def _run(
        self,
        run: Callable[[Callable[[], Any]], Any],
        call: Callable[[], Any],
        pending: Future[Any],
        conversation: str | None,
        background: bool,
    ) -> None:
        sending = pending.set_running_or_notify_cancel()
        outcome, failure = None, None
        if sending:
            try:
                outcome = run(call)
            except BaseException as exc:
                failure = exc
        with self._lock:
            self._outstanding.discard(pending)
            self._depths.leave(conversation, background)
            if conversation is not None and self._tails.get(conversation) is pending:
                del self._tails[conversation]
        if not sending:
            return
        if failure is None:
            pending.set_result(outcome)
        else:
            pending.set_exception(failure)

    def drain(self) -> None:
        """Wait until every classification started so far has finished."""
        while True:
            with self._lock:
                outstanding = set(self._outstanding)
            if not outstanding:
                return
            wait(outstanding)


class AsyncPendingVerdicts:
    """Task-backed classifications sent in order within one conversation."""

    def __init__(self, max_concurrent: int, max_queued: int) -> None:
        self._max_concurrent = max_concurrent
        self._slots: asyncio.Semaphore | None = None
        self._depths = _Depths(max_concurrent, max_queued)
        self._tails: dict[str, asyncio.Task[Any]] = {}
        self._outstanding: set[asyncio.Task[Any]] = set()

    def submit(
        self,
        call: Callable[[], Awaitable[_Verdict]],
        conversation: str | None = None,
        *,
        background: bool = False,
    ) -> asyncio.Task[_Verdict | None]:
        """Start a classification; a background one is never joined by its caller.

        Cancelling the returned task releases its place in the queue.
        """
        if self._slots is None:
            self._slots = asyncio.Semaphore(self._max_concurrent)
        if not self._depths.enter(conversation, background):
            return asyncio.get_running_loop().create_task(_dropped())
        previous = self._tails.get(conversation) if conversation is not None else None
        task = asyncio.get_running_loop().create_task(self._run(call, previous, background))
        self._outstanding.add(task)
        if conversation is not None:
            self._tails[conversation] = task
        task.add_done_callback(lambda done: self._finish(done, conversation, background))
        if background:
            task.add_done_callback(_report)
        return task

    async def _run(
        self,
        call: Callable[[], Awaitable[_Verdict]],
        previous: asyncio.Task[Any] | None,
        background: bool,
    ) -> _Verdict:
        if background:
            RETRY_LIMIT.set(BACKGROUND_RETRIES)
        if previous is not None:
            await asyncio.wait({previous})
        assert self._slots is not None
        async with self._slots:
            return await call()

    def _finish(
        self, done: asyncio.Task[Any], conversation: str | None, background: bool
    ) -> None:
        self._outstanding.discard(done)
        self._depths.leave(conversation, background)
        if conversation is not None and self._tails.get(conversation) is done:
            del self._tails[conversation]
        if not done.cancelled():
            done.exception()

    async def drain(self) -> None:
        """Wait until every classification started so far has finished."""
        while waiting := self._outstanding - {asyncio.current_task()}:
            await asyncio.wait(waiting)


async def _dropped() -> None:
    return None


async def race(
    pending: Awaitable[_Verdict],
    operation: Awaitable[_Result],
    stops: Callable[[_Verdict], bool] = lambda verdict: False,
) -> tuple[_Verdict, _Result | None]:
    """Run an operation beside its verdict and cancel it once the verdict stops it.

    A caller that is cancelled, or an operation that fails, also cancels a verdict
    that is still waiting, so it leaves the conversation queue.
    """
    verdict_task = asyncio.ensure_future(pending)
    running = asyncio.ensure_future(operation)
    try:
        await asyncio.wait({verdict_task, running}, return_when=asyncio.FIRST_COMPLETED)
        if verdict_task.done() and stops(verdict_task.result()):
            return verdict_task.result(), None
        result = await running
        verdict = await verdict_task
        return verdict, None if stops(verdict) else result
    finally:
        running.cancel()
        verdict_task.cancel()


class Deferred:
    """Pending verdicts grouped by the execution that must join them."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._scopes: dict[str, list[Any]] = {}

    def add(self, scope: str, pending: Any) -> None:
        with self._lock:
            self._scopes.setdefault(scope, []).append(pending)

    def take(self, scope: str) -> list[Any]:
        with self._lock:
            return self._scopes.pop(scope, [])

    def failure(self, scope: str) -> BaseException | None:
        """Return the failure of a finished verdict without waiting for the rest."""
        with self._lock:
            finished = [
                pending
                for pending in self._scopes.get(scope, ())
                if pending.done() and not pending.cancelled()
            ]
        return next(
            (pending.exception() for pending in finished if pending.exception() is not None),
            None,
        )
