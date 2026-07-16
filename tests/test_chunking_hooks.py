# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from silmaril_security.sdk import (
    FIREWALL_HOOK_TO_LABEL,
    FirewallHook,
    HookLabel,
    prepend_hook,
    prepend_tool_name,
    resolve_hooks,
)


def test_namespace_import_exports_expected_symbols():
    import silmaril_security.sdk as sdk

    assert sdk.Firewall is not None
    assert sdk.SilmarilFirewall is sdk.Firewall
    assert sdk.HookLabel.USER_INPUT == "user_input"


def test_hooks_and_helpers():
    assert prepend_hook("text", HookLabel.USER_INPUT) == "[HOOK:user_input] text"
    assert prepend_hook("text", HookLabel.UNKNOWN) == "text"
    assert prepend_tool_name("text", "read_file") == "[TOOL:read_file] text"
    assert prepend_tool_name("text", None) == "text"
    assert resolve_hooks(None) == {FirewallHook.LLM_START, FirewallHook.CHAT_MODEL_START}
    assert FIREWALL_HOOK_TO_LABEL[FirewallHook.TOOL_END] == HookLabel.TOOL_RESPONSE
