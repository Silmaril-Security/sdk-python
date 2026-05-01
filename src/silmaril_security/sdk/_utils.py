# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Utilities for extracting text from adapter message structures."""

from __future__ import annotations

from typing import Any, Sequence

_SKIP_ROLES = {"ai", "assistant"}
_SYSTEM_ROLES = {"system"}
_TOOL_ROLES = {"tool", "function"}
_USER_ROLES = {"human", "user"}


def extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def get_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("role", "")
        return role.lower() if isinstance(role, str) else ""
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role.lower()
    msg_type = getattr(message, "type", "")
    return msg_type.lower() if isinstance(msg_type, str) else ""


def get_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def find_last_user_message(messages: Sequence[Any]) -> Any | None:
    for message in reversed(messages):
        if get_role(message) in _USER_ROLES:
            return message
    return None


def extract_last_user_text(messages: Sequence[Any]) -> str:
    last = find_last_user_message(messages)
    if last is None:
        return ""
    return extract_content_text(get_content(last)).strip()


def extract_text_from_messages(
    messages: Sequence[Any],
    *,
    include_system: bool = True,
    include_tool: bool = True,
) -> str:
    parts: list[str] = []
    for message in messages:
        role = get_role(message)
        if role in _SKIP_ROLES:
            continue
        if not include_system and role in _SYSTEM_ROLES:
            continue
        if not include_tool and role in _TOOL_ROLES:
            continue
        text = extract_content_text(get_content(message)).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_text_from_prompts(prompts: Sequence[str]) -> str:
    return "\n".join(prompt.strip() for prompt in prompts if prompt.strip())


def extract_text_from_tool_input(input_str: str) -> str:
    return input_str.strip()


def extract_text_from_llm_result(response: Any) -> str:
    parts: list[str] = []
    for gen_list in getattr(response, "generations", []):
        for gen in gen_list:
            text = (getattr(gen, "text", "") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def extract_text_from_documents(documents: Sequence[Any]) -> str:
    parts: list[str] = []
    for doc in documents:
        text = (getattr(doc, "page_content", "") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)
