"""Native LangChain thread identity; no dependency on LangGraph is required."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from functools import wraps
from threading import RLock
from typing import Any

REMOVED_CONVERSATION_OPTION = (
    "conversation_id/conversationId was removed in 0.8.0; pass the framework's "
    "configurable.thread_id when invoking the agent instead"
)


def reject_conversation_options(options: Mapping[str, Any]) -> None:
    if "conversation_id" in options or "conversationId" in options:
        raise TypeError(REMOVED_CONVERSATION_OPTION)


def _native_identity(config: Any) -> tuple[bool, str | None]:
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    if not isinstance(configurable, Mapping) or "thread_id" not in configurable:
        return False, None
    return True, _valid_id(configurable["thread_id"])


def _valid_id(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def thread_identity(config: Any = None, metadata: Any = None, *, ambient: bool = True) -> tuple[bool, str | None]:
    """Resolve native configuration before callback metadata, retaining explicit absence."""
    from langchain_core.runnables.config import ensure_config

    candidates = (config, ensure_config()) if ambient else (config,)
    for candidate in candidates:
        present, identity = _native_identity(candidate)
        if present:
            return present, identity
    if isinstance(metadata, Mapping) and "thread_id" in metadata:
        return True, _valid_id(metadata["thread_id"])
    return False, None


def conversation_metadata(config: Any = None) -> dict[str, str]:
    _, identity = thread_identity(config)
    return {"conversationId": identity} if identity is not None else {}


MAX_TRACKED_RUNS = 4096


class ThreadRuns:
    """Per-run bookkeeping for callbacks whose terminal events omit metadata."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, tuple[str | None, str | None, bool]] = {}
        self._children: dict[str, set[str]] = {}

    def start(self, run_id: Any, parent_run_id: Any, metadata: Any, snapshot: tuple[bool, str | None] | None = None) -> None:
        run = str(run_id)
        parent = str(parent_run_id) if parent_run_id is not None else None
        with self._lock:
            self.finish(run)
            while len(self._runs) >= MAX_TRACKED_RUNS:
                oldest = next(iter(self._runs))
                visited = set()
                while oldest not in visited:
                    visited.add(oldest)
                    ancestor = self._runs[oldest][1]
                    if ancestor not in self._runs:
                        break
                    oldest = ancestor
                self.finish(oldest)
            parent_entry = self._runs.get(parent)
            graph_metadata = isinstance(metadata, Mapping) and (
                "langgraph_node" in metadata or "langgraph_step" in metadata
            )
            if parent_entry is None and snapshot is not None:
                present, identity = snapshot
                proven_graph = False
            elif graph_metadata:
                present, identity = thread_identity(metadata=metadata)
                proven_graph = True
            elif isinstance(metadata, Mapping) and "thread_id" in metadata:
                present, identity = thread_identity(metadata=metadata, ambient=False)
                if parent_entry is not None and identity == parent_entry[0] and not parent_entry[2]:
                    # An unchanged standalone metadata value may be inherited
                    # despite an invisible child override in the actual config.
                    identity = None
                proven_graph = False
            else:
                # Standalone callback metadata can itself be inherited from an
                # ambient parent even when the child overrides native config.
                present, identity = False, None
                proven_graph = False
            if not present and parent_entry is not None and parent_entry[2]:
                identity = parent_entry[0]
                proven_graph = True
            self._runs[run] = (identity, parent, proven_graph)
            if parent is not None:
                self._children.setdefault(parent, set()).add(run)

    def metadata(self, run_id: Any) -> dict[str, Any]:
        with self._lock:
            identity = self._runs.get(str(run_id), (None, None, False))[0]
        result: dict[str, Any] = {"langgraph": {"run_id": str(run_id)}}
        if identity is not None:
            result["conversationId"] = identity
        return result

    def root(self, run_id: Any) -> str:
        """Return the outermost tracked ancestor, which scopes deferred verdicts."""
        with self._lock:
            run = str(run_id)
            visited = {run}
            while (parent := self._runs.get(run, (None, None, False))[1]) in self._runs:
                if parent in visited:
                    break
                visited.add(parent)
                run = parent
            return run

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            self._children.clear()

    def finish(self, run_id: Any) -> None:
        with self._lock:
            pending = [str(run_id)]
            while pending:
                run = pending.pop()
                pending.extend(self._children.pop(run, ()))
                entry = self._runs.pop(run, None)
                if entry is not None and entry[1] in self._children:
                    siblings = self._children[entry[1]]
                    siblings.discard(run)
                    if not siblings:
                        del self._children[entry[1]]


def track_callback(*, start: bool):
    """Capture before all start hooks; release state even when enforcement raises."""
    def decorate(method):
        def begin(self, kwargs):
            if start:
                self._thread_runs.start(
                    kwargs["run_id"], kwargs.get("parent_run_id"), kwargs.get("metadata"),
                    getattr(self, "_thread_snapshot", None),
                )

        if inspect.iscoroutinefunction(method):
            @wraps(method)
            async def asynchronous(self, *args, **kwargs):
                begin(self, kwargs)
                try:
                    return await method(self, *args, **kwargs)
                except BaseException:
                    self._thread_runs.finish(kwargs["run_id"])
                    raise
                finally:
                    if not start:
                        self._thread_runs.finish(kwargs["run_id"])
            return asynchronous

        @wraps(method)
        def synchronous(self, *args, **kwargs):
            begin(self, kwargs)
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                self._thread_runs.finish(kwargs["run_id"])
                raise
            finally:
                if not start:
                    self._thread_runs.finish(kwargs["run_id"])
        return synchronous
    return decorate
