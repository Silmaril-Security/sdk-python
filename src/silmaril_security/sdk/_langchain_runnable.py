"""Preserve native runnable layers while composing protected boundaries."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from contextvars import ContextVar
from typing import Any, TypeVar

from langchain_core.runnables import Runnable, RunnableConfig

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")
_HANDLERS: ContextVar[list[Any] | None] = ContextVar("silmaril_runnable_handlers", default=None)


def register_handler(handler: Any) -> None:
    handlers = _HANDLERS.get()
    if handlers is not None:
        handlers.append(handler)


def _clear(handlers: list[Any]) -> None:
    for handler in handlers:
        handler._thread_runs.clear()
    handlers.clear()


class PreservingRunnable(Runnable[_Input, _Output]):
    """Factory-free bridge: composing it never flattens original bindings."""

    def __init__(self, bound: Runnable[_Input, _Output]) -> None:
        self.bound = bound

    @property
    def InputType(self):
        return self.bound.InputType

    @property
    def OutputType(self):
        return self.bound.OutputType

    def get_input_schema(self, config: RunnableConfig | None = None):
        return self.bound.get_input_schema(config)

    def get_output_schema(self, config: RunnableConfig | None = None):
        return self.bound.get_output_schema(config)

    def _scope(self):
        inherited = _HANDLERS.get()
        handlers = inherited if inherited is not None else []
        return handlers, inherited is None

    def invoke(self, input: _Input, config: RunnableConfig | None = None, **kwargs: Any) -> _Output:
        handlers, owner = self._scope()
        token = _HANDLERS.set(handlers)
        try:
            return self.bound.invoke(input, config, **kwargs)
        finally:
            _HANDLERS.reset(token)
            if owner:
                _clear(handlers)

    async def ainvoke(self, input: _Input, config: RunnableConfig | None = None, **kwargs: Any) -> _Output:
        handlers, owner = self._scope()
        token = _HANDLERS.set(handlers)
        try:
            return await self.bound.ainvoke(input, config, **kwargs)
        finally:
            _HANDLERS.reset(token)
            if owner:
                _clear(handlers)

    def batch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        handlers, owner = self._scope()
        token = _HANDLERS.set(handlers)
        try:
            return self.bound.batch(inputs, config, return_exceptions=return_exceptions, **kwargs)
        finally:
            _HANDLERS.reset(token)
            if owner:
                _clear(handlers)

    async def abatch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        handlers, owner = self._scope()
        token = _HANDLERS.set(handlers)
        try:
            return await self.bound.abatch(inputs, config, return_exceptions=return_exceptions, **kwargs)
        finally:
            _HANDLERS.reset(token)
            if owner:
                _clear(handlers)

    def _iterate(self, make_iterator) -> Iterator[_Output]:
        handlers, owner = self._scope()
        token = _HANDLERS.set(handlers)
        try:
            iterator = make_iterator()
        except BaseException:
            if owner:
                _clear(handlers)
            raise
        finally:
            _HANDLERS.reset(token)
        try:
            while True:
                token = _HANDLERS.set(handlers)
                try:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        return
                finally:
                    _HANDLERS.reset(token)
                # Never leave context set across a yield: generators may be
                # interleaved in the same task by their consumer.
                yield chunk
        finally:
            token = _HANDLERS.set(handlers)
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            finally:
                _HANDLERS.reset(token)
                if owner:
                    _clear(handlers)

    async def _aiterate(self, make_iterator) -> AsyncIterator[_Output]:
        handlers, owner = self._scope()
        token = _HANDLERS.set(handlers)
        try:
            iterator = make_iterator()
        except BaseException:
            if owner:
                _clear(handlers)
            raise
        finally:
            _HANDLERS.reset(token)
        try:
            while True:
                token = _HANDLERS.set(handlers)
                try:
                    try:
                        chunk = await anext(iterator)
                    except StopAsyncIteration:
                        return
                finally:
                    _HANDLERS.reset(token)
                yield chunk
        finally:
            token = _HANDLERS.set(handlers)
            try:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
            finally:
                _HANDLERS.reset(token)
                if owner:
                    _clear(handlers)

    def stream(self, input: _Input, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[_Output]:
        yield from self._iterate(lambda: self.bound.stream(input, config, **kwargs))

    def transform(self, input: Iterator[_Input], config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[_Output]:
        yield from self._iterate(lambda: self.bound.transform(input, config, **kwargs))

    async def astream(self, input: _Input, config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[_Output]:
        iterator = self._aiterate(lambda: self.bound.astream(input, config, **kwargs))
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()

    async def atransform(self, input: AsyncIterator[_Input], config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[_Output]:
        iterator = self._aiterate(lambda: self.bound.atransform(input, config, **kwargs))
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()

    def with_config(self, config=None, **kwargs):
        return PreservingRunnable(Runnable.with_config(self, config, **kwargs))

    def bind(self, **kwargs):
        return PreservingRunnable(Runnable.bind(self, **kwargs))

    def with_retry(self, **kwargs):
        return PreservingRunnable(Runnable.with_retry(self, **kwargs))

    def with_listeners(self, **kwargs):
        return PreservingRunnable(Runnable.with_listeners(self, **kwargs))

    if hasattr(Runnable, "with_alisteners"):
        def with_alisteners(self, **kwargs):
            return PreservingRunnable(Runnable.with_alisteners(self, **kwargs))

    def with_types(self, **kwargs):
        return PreservingRunnable(Runnable.with_types(self, **kwargs))
