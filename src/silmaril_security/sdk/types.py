# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Public data types for the Silmaril Firewall SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from silmaril_security.sdk.hooks import HookLabel
from silmaril_security.sdk.outcomes import HarmfulOutcome, PrimaryOutcome

Prediction = Literal["BENIGN", "MALICIOUS"]
FirewallMode = Literal["shadow", "warn", "block"]
ClassificationMetadata = Mapping[str, Any]
GovernanceAction = Literal["allow", "block"]
GovernanceReason = Literal["identity_unresolved"]
GovernanceResourceKind = Literal[
    "agent",
    "tool",
    "mcp_server",
    "mcp_tool",
    "plugin",
    "skill",
    "extension",
]
GOVERNANCE_RESOURCE_KINDS: tuple[GovernanceResourceKind, ...] = (
    "agent",
    "tool",
    "mcp_server",
    "mcp_tool",
    "plugin",
    "skill",
    "extension",
)


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or not any(not char.isspace() for char in value):
        raise ValueError(f"Firewall: {field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class GovernanceResource:
    """Canonical concrete governance resource identity."""

    kind: GovernanceResourceKind
    id: str
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in GOVERNANCE_RESOURCE_KINDS:
            raise ValueError(f"Firewall: invalid governance resource kind {self.kind!r}")
        _nonempty_string(self.id, "governance resource id")
        if self.kind == "mcp_tool":
            _nonempty_string(self.parent_id, "mcp_tool parent_id")
        elif self.parent_id is not None:
            raise ValueError("Firewall: parent_id is only valid for mcp_tool resources")

    def to_wire(self) -> dict[str, str]:
        """Return the contract's snake-case wire representation."""

        result = {"kind": self.kind, "id": self.id}
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        return result

    @classmethod
    def from_wire(cls, value: object, field_name: str = "resource") -> GovernanceResource:
        """Validate and decode a resource returned by the service."""

        if not isinstance(value, Mapping):
            raise ValueError(f"Firewall: {field_name} must be an object")
        unknown = set(value) - {"kind", "id", "parent_id"}
        if unknown:
            raise ValueError(f"Firewall: {field_name} contains unknown fields: {sorted(unknown)!r}")
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in GOVERNANCE_RESOURCE_KINDS:
            raise ValueError(f"Firewall: invalid {field_name} kind {kind!r}")
        return cls(
            kind=cast(GovernanceResourceKind, kind),
            id=_nonempty_string(value.get("id"), f"{field_name}.id"),
            parent_id=value.get("parent_id"),
        )


@dataclass(frozen=True)
class GovernanceDecision:
    """Governance decision returned alongside threat classification."""

    action: GovernanceAction
    policy_version: str
    rule_id: str | None = None
    resource: GovernanceResource | None = None
    identity_revision: str | None = None
    reason: GovernanceReason | None = None

    def __post_init__(self) -> None:
        if self.action not in ("allow", "block"):
            raise ValueError(f"Firewall: invalid governance action {self.action!r}")
        _nonempty_string(self.policy_version, "governance policy_version")
        if self.rule_id is not None and not isinstance(self.rule_id, str):
            raise ValueError("Firewall: governance rule_id must be a string")
        if self.resource is not None and not isinstance(
            self.resource, GovernanceResource
        ):
            raise ValueError("Firewall: governance resource must be a GovernanceResource")
        if self.identity_revision is not None:
            _nonempty_string(
                self.identity_revision, "governance identity_revision"
            )
        if self.reason is not None and self.reason != "identity_unresolved":
            raise ValueError(f"Firewall: invalid governance reason {self.reason!r}")


@dataclass(frozen=True)
class BlockResult:
    """Result of a firewall classification call."""

    prediction: Prediction
    score: float
    threshold: float
    primary_outcome: PrimaryOutcome | None = None
    outcome_scores: dict[HarmfulOutcome, float] | None = None
    detector_scores: dict[HarmfulOutcome, float] | None = None
    detector_counts: dict[HarmfulOutcome, int] | None = None
    # None only when a legacy backend omitted mode and no override was requested.
    mode: FirewallMode | None = None
    governance: GovernanceDecision | None = None


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
