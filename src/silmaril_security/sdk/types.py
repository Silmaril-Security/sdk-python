# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Public data types for the Silmaril Firewall SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from silmaril_security.sdk.hooks import HookLabel

Prediction = Literal["BENIGN", "MALICIOUS"]


@dataclass(frozen=True)
class BlockResult:
    """Result of a firewall classification call."""

    prediction: Prediction
    score: float
    threshold: float
    primary_outcome: str | None = None
    outcome_scores: dict[str, float] | None = None


@dataclass(frozen=True)
class ExplainResult:
    """Token-level attribution result from the explain endpoint."""

    tokens: list[str]
    attributions: list[float]
    score: float
    prediction: Prediction
    prepared_text: str
    primary_outcome: str | None = None
    outcome_scores: dict[str, float] | None = None


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
