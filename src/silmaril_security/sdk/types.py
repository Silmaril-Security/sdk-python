# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Public data types for the Silmaril Firewall SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from silmaril_security.sdk.hooks import HookLabel
from silmaril_security.sdk.outcomes import HarmfulOutcomeValue

DecisionAction = Literal["allow", "block"]
DecisionSource = Literal["model", "governance"]
DecisionConfidence = Literal["very_low", "low", "medium", "high", "very_high"]
DecisionSeverity = Literal["minimal", "low", "moderate", "high", "critical"]
FirewallMode = Literal["shadow", "warn", "block"]
ReviewLevel = Literal["standard", "deep"]
ClassificationMetadata = Mapping[str, Any]
GovernanceAction = Literal["allow", "block"]
GovernanceResourceKind = Literal[
    "agent", "tool", "mcp_server", "mcp_tool", "plugin", "skill", "extension"
]


@dataclass(frozen=True)
class GovernanceResource:
    kind: GovernanceResourceKind
    id: str | None = None
    parent_id: str | None = None


@dataclass(frozen=True)
class GovernanceContext:
    agent: str | None = None
    resource: GovernanceResource | None = None


@dataclass(frozen=True)
class GovernanceDecision:
    action: GovernanceAction
    policy_version: str
    rule_id: str | None = None
    resource: GovernanceResource | None = None
    identity_revision: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class BlockResult:
    """Decision of a firewall classification call."""

    action: DecisionAction
    source: DecisionSource | str | None = None
    confidence: DecisionConfidence | str | None = None
    severity: DecisionSeverity | str | None = None
    outcome: HarmfulOutcomeValue | None = None
    governance: GovernanceDecision | None = None
    model_id: str | None = None
    review: ReviewLevel | None = None
    mode: FirewallMode | None = None

    @property
    def blocked(self) -> bool:
        return self.action != "allow"


@dataclass(frozen=True, init=False)
class ClassifyEvent:
    """Classification decision emitted by direct calls and adapters."""

    hook: HookLabel
    tool_name: str | None
    text: str
    result: BlockResult
    blocked: bool
    shadow_mode: bool
    mode: FirewallMode

    def __init__(
        self,
        hook: HookLabel,
        tool_name: str | None,
        text: str,
        result: BlockResult,
        blocked: bool,
        shadow_mode: bool,
        *,
        mode: FirewallMode | None = None,
    ) -> None:
        """Preserve the pre-0.6 positional signature while adding effective mode."""
        effective_mode = mode or ("shadow" if shadow_mode else result.mode or "block")
        if effective_mode not in ("shadow", "warn", "block"):
            raise ValueError("Firewall: mode must be shadow, warn, or block")
        object.__setattr__(self, "hook", hook)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "blocked", blocked)
        object.__setattr__(self, "shadow_mode", effective_mode == "shadow")
        object.__setattr__(self, "mode", effective_mode)


@dataclass(frozen=True)
class BlockedBatchItem:
    """One blocked item from a batch classification call."""

    index: int
    text: str
    hook: HookLabel
    tool_name: str | None
    result: BlockResult
