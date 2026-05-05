# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Silmaril Firewall Python SDK."""

from __future__ import annotations

from silmaril_security.sdk.chunking import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP,
    CHUNK_OVERLAP_CHARS,
    CHUNK_WINDOW,
    CHUNK_WINDOW_CHARS,
    MAX_INPUT_CHARS,
    MAX_INPUT_TOKENS,
    chunk_text,
)
from silmaril_security.sdk.exceptions import (
    APIError,
    BatchPromptBlockedException,
    PromptBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk.firewall import (
    DEFAULT_CHUNK_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_THRESHOLD,
    DEFAULT_TIMEOUT,
    Firewall,
    SilmarilFirewall,
)
from silmaril_security.sdk.hooks import (
    ALL_HOOKS,
    DEFAULT_HOOK_THRESHOLDS,
    DEFAULT_HOOKS,
    FIREWALL_HOOK_TO_LABEL,
    INPUT_HOOKS,
    OUTPUT_HOOKS,
    FirewallHook,
    HookLabel,
    prepend_hook,
    prepend_tool_name,
    resolve_hooks,
)
from silmaril_security.sdk.types import (
    BlockedBatchItem,
    BlockResult,
    ClassifyEvent,
    ExplainResult,
    Prediction,
)

__version__ = "0.1.1"

__all__ = [
    "ALL_HOOKS",
    "APIError",
    "BlockResult",
    "BlockedBatchItem",
    "BatchPromptBlockedException",
    "CHARS_PER_TOKEN",
    "CHUNK_OVERLAP",
    "CHUNK_OVERLAP_CHARS",
    "CHUNK_WINDOW",
    "CHUNK_WINDOW_CHARS",
    "ClassifyEvent",
    "DEFAULT_CHUNK_CONCURRENCY",
    "DEFAULT_HOOKS",
    "DEFAULT_HOOK_THRESHOLDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TIMEOUT",
    "ExplainResult",
    "FIREWALL_HOOK_TO_LABEL",
    "Firewall",
    "FirewallHook",
    "HookLabel",
    "INPUT_HOOKS",
    "MAX_INPUT_CHARS",
    "MAX_INPUT_TOKENS",
    "OUTPUT_HOOKS",
    "Prediction",
    "PromptBlockedException",
    "SilmarilApiError",
    "SilmarilFirewall",
    "chunk_text",
    "prepend_hook",
    "prepend_tool_name",
    "resolve_hooks",
]


def __getattr__(name: str):
    if name == "SilmarilFirewallHandler":
        from silmaril_security.sdk.langchain import SilmarilFirewallHandler

        return SilmarilFirewallHandler
    if name == "AsyncSilmarilFirewallHandler":
        from silmaril_security.sdk.langchain import AsyncSilmarilFirewallHandler

        return AsyncSilmarilFirewallHandler
    raise AttributeError(f"module 'silmaril_security.sdk' has no attribute {name!r}")
