# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Hook labels for pipeline-stage-aware classification."""

from __future__ import annotations

import enum
from collections.abc import Iterable


class HookLabel(str, enum.Enum):
    """LLM pipeline stage labels."""

    USER_INPUT = "user_input"
    SYSTEM_PROMPT = "system_prompt"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    LLM_OUTPUT = "llm_output"
    UNKNOWN = "unknown"


class FirewallHook(str, enum.Enum):
    """LangChain callback hooks the firewall can intercept."""

    LLM_START = "on_llm_start"
    CHAT_MODEL_START = "on_chat_model_start"
    TOOL_START = "on_tool_start"
    RETRIEVER_START = "on_retriever_start"
    LLM_END = "on_llm_end"
    TOOL_END = "on_tool_end"
    RETRIEVER_END = "on_retriever_end"


DEFAULT_HOOKS: frozenset[FirewallHook] = frozenset(
    {FirewallHook.LLM_START, FirewallHook.CHAT_MODEL_START}
)

INPUT_HOOKS: frozenset[FirewallHook] = frozenset(
    {
        FirewallHook.LLM_START,
        FirewallHook.CHAT_MODEL_START,
        FirewallHook.TOOL_START,
        FirewallHook.RETRIEVER_START,
    }
)

OUTPUT_HOOKS: frozenset[FirewallHook] = frozenset(
    {
        FirewallHook.LLM_END,
        FirewallHook.TOOL_END,
        FirewallHook.RETRIEVER_END,
    }
)

ALL_HOOKS: frozenset[FirewallHook] = INPUT_HOOKS | OUTPUT_HOOKS

FIREWALL_HOOK_TO_LABEL: dict[FirewallHook, HookLabel] = {
    FirewallHook.CHAT_MODEL_START: HookLabel.USER_INPUT,
    FirewallHook.LLM_START: HookLabel.USER_INPUT,
    FirewallHook.TOOL_START: HookLabel.TOOL_CALL,
    FirewallHook.TOOL_END: HookLabel.TOOL_RESPONSE,
    FirewallHook.RETRIEVER_START: HookLabel.TOOL_CALL,
    FirewallHook.RETRIEVER_END: HookLabel.TOOL_RESPONSE,
    FirewallHook.LLM_END: HookLabel.LLM_OUTPUT,
}

def resolve_hooks(hooks: Iterable[FirewallHook | str] | None) -> frozenset[FirewallHook]:
    """Normalize callback hook values into a frozenset of FirewallHook members."""
    if hooks is None:
        return DEFAULT_HOOKS
    return frozenset(h if isinstance(h, FirewallHook) else FirewallHook(h) for h in hooks)


def hook_value(hook: HookLabel | str | None) -> str | None:
    """Serialize a HookLabel or string to the API wire value."""
    if hook is None:
        return None
    return hook.value if isinstance(hook, HookLabel) else str(hook)


def normalize_hook_label(hook: HookLabel | str | None) -> HookLabel:
    """Normalize hook input for events and structured request metadata."""
    value = hook_value(hook)
    if not value:
        return HookLabel.UNKNOWN
    try:
        return HookLabel(value)
    except ValueError:
        return HookLabel.UNKNOWN


def prepend_hook(text: str, hook: HookLabel | str | None) -> str:
    """Prefix text with a [HOOK:<label>] marker for offline parity checks."""
    value = hook_value(hook)
    if value is None or value == HookLabel.UNKNOWN.value:
        return text
    return f"[HOOK:{value}] {text}"


def prepend_tool_name(text: str, tool_name: str | None) -> str:
    """Prefix text with a [TOOL:<name>] marker for offline parity checks."""
    if not tool_name:
        return text
    return f"[TOOL:{tool_name}] {text}"
