# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from __future__ import annotations

import pytest

from silmaril_security.sdk import (
    HARMFUL_OUTCOMES,
    OUTCOME_BENIGN,
    OUTCOME_DESCRIPTIONS,
    OUTCOME_SECRET_EXPOSURE,
    PRIMARY_OUTCOMES,
    is_harmful_outcome,
    is_primary_outcome,
    normalize_harmful_outcome,
    normalize_harmful_outcome_float_map,
    normalize_harmful_outcome_int_map,
    normalize_primary_outcome,
)


def test_outcome_taxonomy_exports_ordered_values():
    assert PRIMARY_OUTCOMES == (
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
    )
    assert HARMFUL_OUTCOMES == PRIMARY_OUTCOMES[1:]
    assert OUTCOME_DESCRIPTIONS[OUTCOME_BENIGN]


def test_outcome_validation_helpers():
    assert is_primary_outcome(OUTCOME_BENIGN)
    assert is_harmful_outcome(OUTCOME_SECRET_EXPOSURE)
    assert not is_harmful_outcome(OUTCOME_BENIGN)
    assert normalize_primary_outcome(OUTCOME_BENIGN) == OUTCOME_BENIGN
    assert normalize_harmful_outcome(OUTCOME_SECRET_EXPOSURE) == OUTCOME_SECRET_EXPOSURE
    assert normalize_harmful_outcome_float_map(
        {OUTCOME_SECRET_EXPOSURE: 0.8}, "outcome_scores"
    ) == {OUTCOME_SECRET_EXPOSURE: 0.8}
    assert normalize_harmful_outcome_int_map(
        {OUTCOME_SECRET_EXPOSURE: 2}, "detector_counts"
    ) == {OUTCOME_SECRET_EXPOSURE: 2}
    with pytest.raises(ValueError, match="invalid primary_outcome"):
        normalize_primary_outcome("unknown")
    with pytest.raises(ValueError, match="invalid outcome_scores key"):
        normalize_harmful_outcome_float_map({"unknown": 0.8}, "outcome_scores")
    with pytest.raises(ValueError, match="invalid detector_counts key"):
        normalize_harmful_outcome_int_map({"unknown": 1}, "detector_counts")
    with pytest.raises(ValueError, match="invalid outcome_scores value"):
        normalize_harmful_outcome_float_map({OUTCOME_SECRET_EXPOSURE: "N/A"}, "outcome_scores")
    with pytest.raises(ValueError, match="expected number"):
        normalize_harmful_outcome_float_map({OUTCOME_SECRET_EXPOSURE: True}, "outcome_scores")
    with pytest.raises(ValueError, match="non-finite"):
        normalize_harmful_outcome_float_map(
            {OUTCOME_SECRET_EXPOSURE: float("nan")}, "outcome_scores"
        )
    with pytest.raises(ValueError, match="non-finite"):
        normalize_harmful_outcome_float_map(
            {OUTCOME_SECRET_EXPOSURE: float("inf")}, "outcome_scores"
        )
    with pytest.raises(ValueError, match="invalid detector_counts value"):
        normalize_harmful_outcome_int_map({OUTCOME_SECRET_EXPOSURE: 2.9}, "detector_counts")
