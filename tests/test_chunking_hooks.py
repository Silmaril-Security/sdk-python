# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

import pytest

from silmaril_security.sdk import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_WINDOW_CHARS,
    FIREWALL_HOOK_TO_LABEL,
    MAX_INPUT_CHARS,
    FirewallHook,
    HookLabel,
    chunk_text,
    prepend_hook,
    prepend_tool_name,
    resolve_hooks,
)


def test_namespace_import_exports_expected_symbols():
    import silmaril_security.sdk as sdk

    assert sdk.Firewall is not None
    assert sdk.SilmarilFirewall is sdk.Firewall
    assert sdk.HookLabel.USER_INPUT == "user_input"


def test_chunk_text_short_input():
    assert chunk_text("hello") == ["hello"]


def test_chunk_text_long_input_overlaps():
    text = "a" * (CHUNK_WINDOW_CHARS + 100)
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert chunks[0][-CHUNK_OVERLAP_CHARS:] == chunks[1][:CHUNK_OVERLAP_CHARS]


def test_chunk_text_rejects_too_long():
    with pytest.raises(ValueError, match="tokens.*chars"):
        chunk_text("a" * (MAX_INPUT_CHARS + 1))


def test_hooks_and_helpers():
    assert prepend_hook("text", HookLabel.USER_INPUT) == "[HOOK:user_input] text"
    assert prepend_hook("text", HookLabel.UNKNOWN) == "text"
    assert prepend_tool_name("text", "read_file") == "[TOOL:read_file] text"
    assert prepend_tool_name("text", None) == "text"
    assert resolve_hooks(None) == {FirewallHook.LLM_START, FirewallHook.CHAT_MODEL_START}
    assert FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_END] == HookLabel.TOOL_RESPONSE
