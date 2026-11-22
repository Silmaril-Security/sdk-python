# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import silmaril_security.sdk as sdk
from silmaril_security.sdk import (
    HARMFUL_OUTCOMES,
    OUTCOME_DESCRIPTIONS,
    OUTCOME_SECRET_EXPOSURE,
    is_harmful_outcome,
)


def test_outcome_taxonomy_exports_ordered_values():
    assert HARMFUL_OUTCOMES == (
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
    )
    assert tuple(OUTCOME_DESCRIPTIONS) == HARMFUL_OUTCOMES
    assert all(OUTCOME_DESCRIPTIONS.values())


def test_outcome_validation_helpers():
    assert is_harmful_outcome(OUTCOME_SECRET_EXPOSURE)
    assert not is_harmful_outcome("benign")
    assert not is_harmful_outcome("future_outcome")


def test_benign_outcome_is_not_part_of_the_public_taxonomy():
    assert not {"OUTCOME_BENIGN", "PRIMARY_OUTCOMES", "is_primary_outcome"} & set(dir(sdk))
