# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Public data types for the Silmaril Firewall SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from silmaril_security.sdk.hooks import HookLabel
from silmaril_security.sdk.outcomes import HarmfulOutcome, PrimaryOutcome

Prediction = Literal["BENIGN", "MALICIOUS"]
ClassificationMetadata = Mapping[str, Any]


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


@dataclass(frozen=True)
class ClassifyEvent:
    """Classification decision emitted by direct calls and adapters."""

    hook: HookLabel
    tool_name: str | None
    text: str
    result: BlockResult
    blocked: bool
    shadow_mode: bool


@dataclass(frozen=True)
class BlockedBatchItem:
    """One blocked item from a batch classification call."""

    index: int
    text: str
    hook: HookLabel
    tool_name: str | None
    result: BlockResult
