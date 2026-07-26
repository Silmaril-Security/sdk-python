# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Outcome taxonomy for Silmaril Firewall classification results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, cast

PrimaryOutcome = Literal[
    "benign",
    "information_disclosure",
    "secret_exposure",
    "control_abuse",
    "system_compromise",
    "service_disruption",
    "code_generation",
    "story_script_generation",
    "game_generation",
    "website_generation",
    "clickup_terms_violation",
    "traditional_ai_abuse",
]
HarmfulOutcome = Literal[
    "information_disclosure",
    "secret_exposure",
    "control_abuse",
    "system_compromise",
    "service_disruption",
    "code_generation",
    "story_script_generation",
    "game_generation",
    "website_generation",
    "clickup_terms_violation",
    "traditional_ai_abuse",
]

OUTCOME_BENIGN: PrimaryOutcome = "benign"
OUTCOME_INFORMATION_DISCLOSURE: HarmfulOutcome = "information_disclosure"
OUTCOME_SECRET_EXPOSURE: HarmfulOutcome = "secret_exposure"
OUTCOME_CONTROL_ABUSE: HarmfulOutcome = "control_abuse"
OUTCOME_SYSTEM_COMPROMISE: HarmfulOutcome = "system_compromise"
OUTCOME_SERVICE_DISRUPTION: HarmfulOutcome = "service_disruption"
OUTCOME_CODE_GENERATION: HarmfulOutcome = "code_generation"
OUTCOME_STORY_SCRIPT_GENERATION: HarmfulOutcome = "story_script_generation"
OUTCOME_GAME_GENERATION: HarmfulOutcome = "game_generation"
OUTCOME_WEBSITE_GENERATION: HarmfulOutcome = "website_generation"
OUTCOME_CLICKUP_TERMS_VIOLATION: HarmfulOutcome = "clickup_terms_violation"
OUTCOME_TRADITIONAL_AI_ABUSE: HarmfulOutcome = "traditional_ai_abuse"

PRIMARY_OUTCOMES: tuple[PrimaryOutcome, ...] = (
    OUTCOME_BENIGN,
    OUTCOME_INFORMATION_DISCLOSURE,
    OUTCOME_SECRET_EXPOSURE,
    OUTCOME_CONTROL_ABUSE,
    OUTCOME_SYSTEM_COMPROMISE,
    OUTCOME_SERVICE_DISRUPTION,
    OUTCOME_CODE_GENERATION,
    OUTCOME_STORY_SCRIPT_GENERATION,
    OUTCOME_GAME_GENERATION,
    OUTCOME_WEBSITE_GENERATION,
    OUTCOME_CLICKUP_TERMS_VIOLATION,
    OUTCOME_TRADITIONAL_AI_ABUSE,
)

HARMFUL_OUTCOMES: tuple[HarmfulOutcome, ...] = (
    OUTCOME_INFORMATION_DISCLOSURE,
    OUTCOME_SECRET_EXPOSURE,
    OUTCOME_CONTROL_ABUSE,
    OUTCOME_SYSTEM_COMPROMISE,
    OUTCOME_SERVICE_DISRUPTION,
    OUTCOME_CODE_GENERATION,
    OUTCOME_STORY_SCRIPT_GENERATION,
    OUTCOME_GAME_GENERATION,
    OUTCOME_WEBSITE_GENERATION,
    OUTCOME_CLICKUP_TERMS_VIOLATION,
    OUTCOME_TRADITIONAL_AI_ABUSE,
)

OUTCOME_DESCRIPTIONS: Mapping[PrimaryOutcome, str] = {
    OUTCOME_BENIGN: "No harmful firewall outcome detected.",
    OUTCOME_INFORMATION_DISCLOSURE: (
        "Exposes private data, documents, internal context, logs, traces, customer data, "
        "SQL rows, topology, or similar non-secret sensitive information."
    ),
    OUTCOME_SECRET_EXPOSURE: (
        "Exposes credentials, tokens, API keys, cookies, passwords, signing keys, OAuth "
        "secrets, session material, or webhook secrets."
    ),
    OUTCOME_CONTROL_ABUSE: (
        "Misuses authorized tools or user privileges to send, change, approve, delete, "
        "operate, or bypass policy/RBAC without a stronger outcome."
    ),
    OUTCOME_SYSTEM_COMPROMISE: (
        "Enables privilege escalation, account takeover, hostile integration or plugin "
        "takeover, persistence, lateral movement, attacker webhook registration, or "
        "code/plugin execution."
    ),
    OUTCOME_SERVICE_DISRUPTION: (
        "Causes downtime, lockout, degradation, alert suppression, destructive loops, "
        "resource exhaustion, cost spikes, or hidden outage evidence."
    ),
    OUTCOME_CODE_GENERATION: (
        "Requests generation or material modification of executable code, scripts, "
        "workflows, or configuration."
    ),
    OUTCOME_STORY_SCRIPT_GENERATION: (
        "Requests generation of narrative prose, dialogue, scripts, or story artifacts."
    ),
    OUTCOME_GAME_GENERATION: (
        "Requests generation of a game, quest, level, mechanic, or playable experience."
    ),
    OUTCOME_WEBSITE_GENERATION: (
        "Requests generation of a website, landing page, storefront, or web experience."
    ),
    OUTCOME_CLICKUP_TERMS_VIOLATION: (
        "Requests content or actions that violate the configured ClickUp tenant policy."
    ),
    OUTCOME_TRADITIONAL_AI_ABUSE: (
        "Requests unsafe AI assistance outside the concrete security outcome classes."
    ),
}

_PRIMARY_OUTCOMES_SET = frozenset(PRIMARY_OUTCOMES)
_HARMFUL_OUTCOMES_SET = frozenset(HARMFUL_OUTCOMES)


def is_primary_outcome(value: str) -> bool:
    """Return true when value is a canonical primary outcome."""

    return value in _PRIMARY_OUTCOMES_SET


def is_harmful_outcome(value: str) -> bool:
    """Return true when value is a canonical harmful outcome."""

    return value in _HARMFUL_OUTCOMES_SET


def normalize_primary_outcome(value: object, field_name: str = "primary_outcome") -> PrimaryOutcome:
    """Validate and return a canonical primary outcome."""

    if not isinstance(value, str) or not is_primary_outcome(value):
        raise ValueError(f"Firewall: invalid {field_name} {value!r}")
    return cast(PrimaryOutcome, value)


def normalize_harmful_outcome(value: object, field_name: str = "outcome") -> HarmfulOutcome:
    """Validate and return a canonical harmful outcome."""

    if not isinstance(value, str) or not is_harmful_outcome(value):
        raise ValueError(f"Firewall: invalid {field_name} {value!r}")
    return cast(HarmfulOutcome, value)


def normalize_harmful_outcome_float_map(
    value: object,
    field_name: str,
) -> dict[HarmfulOutcome, float] | None:
    """Validate a backend harmful-outcome score map."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Firewall: invalid {field_name} {value!r}")
    result: dict[HarmfulOutcome, float] = {}
    for key, raw in value.items():
        outcome_key = normalize_harmful_outcome(key, f"{field_name} key")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(
                f"Firewall: invalid {field_name} value for {key!r}: {raw!r} (expected number)"
            )
        score = float(raw)
        if not math.isfinite(score):
            raise ValueError(
                f"Firewall: invalid {field_name} value for {key!r}: {raw!r} (non-finite)"
            )
        result[outcome_key] = score
    return result


def normalize_harmful_outcome_int_map(
    value: object,
    field_name: str,
) -> dict[HarmfulOutcome, int] | None:
    """Validate a backend harmful-outcome count map."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Firewall: invalid {field_name} {value!r}")
    result: dict[HarmfulOutcome, int] = {}
    for key, raw in value.items():
        outcome_key = normalize_harmful_outcome(key, f"{field_name} key")
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ValueError(
                f"Firewall: invalid {field_name} value for {key!r}: {raw!r} (expected int)"
            )
        result[outcome_key] = raw
    return result
