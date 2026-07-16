# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Silmaril Firewall Python SDK."""

from __future__ import annotations

from silmaril_security.sdk._version import VERSION
from silmaril_security.sdk.exceptions import (
    APIError,
    BatchFirewallBlockedException,
    BatchPromptBlockedException,
    FirewallBlockedException,
    PromptBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk.firewall import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    Firewall,
    SilmarilFirewall,
)
from silmaril_security.sdk.hooks import (
    ALL_HOOKS,
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
from silmaril_security.sdk.outcomes import (
    HARMFUL_OUTCOMES,
    OUTCOME_BENIGN,
    OUTCOME_CONTROL_ABUSE,
    OUTCOME_DESCRIPTIONS,
    OUTCOME_INFORMATION_DISCLOSURE,
    OUTCOME_SECRET_EXPOSURE,
    OUTCOME_SERVICE_DISRUPTION,
    OUTCOME_SYSTEM_COMPROMISE,
    PRIMARY_OUTCOMES,
    HarmfulOutcome,
    PrimaryOutcome,
    is_harmful_outcome,
    is_primary_outcome,
    normalize_harmful_outcome,
    normalize_harmful_outcome_float_map,
    normalize_harmful_outcome_int_map,
    normalize_primary_outcome,
)
from silmaril_security.sdk.types import (
    BlockedBatchItem,
    BlockResult,
    ClassificationMetadata,
    ClassifyEvent,
    Prediction,
)

__version__ = VERSION

__all__ = [
    "ALL_HOOKS",
    "APIError",
    "BlockResult",
    "BlockedBatchItem",
    "BatchFirewallBlockedException",
    "BatchPromptBlockedException",
    "ClassificationMetadata",
    "ClassifyEvent",
    "DEFAULT_HOOKS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT",
    "FIREWALL_HOOK_TO_LABEL",
    "Firewall",
    "FirewallBlockedException",
    "FirewallHook",
    "HARMFUL_OUTCOMES",
    "HarmfulOutcome",
    "HookLabel",
    "INPUT_HOOKS",
    "OUTCOME_BENIGN",
    "OUTCOME_CONTROL_ABUSE",
    "OUTCOME_DESCRIPTIONS",
    "OUTCOME_INFORMATION_DISCLOSURE",
    "OUTCOME_SECRET_EXPOSURE",
    "OUTCOME_SERVICE_DISRUPTION",
    "OUTCOME_SYSTEM_COMPROMISE",
    "OUTPUT_HOOKS",
    "PRIMARY_OUTCOMES",
    "Prediction",
    "PrimaryOutcome",
    "PromptBlockedException",
    "SilmarilApiError",
    "SilmarilFirewall",
    "is_harmful_outcome",
    "is_primary_outcome",
    "normalize_harmful_outcome",
    "normalize_harmful_outcome_float_map",
    "normalize_harmful_outcome_int_map",
    "normalize_primary_outcome",
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
