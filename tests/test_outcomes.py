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
    )
    assert HARMFUL_OUTCOMES == PRIMARY_OUTCOMES[1:]
    assert OUTCOME_DESCRIPTIONS[OUTCOME_BENIGN]


def test_outcome_validation_helpers():
    assert is_primary_outcome(OUTCOME_BENIGN)
    assert is_harmful_outcome(OUTCOME_SECRET_EXPOSURE)
    assert not is_harmful_outcome(OUTCOME_BENIGN)
    assert normalize_primary_outcome(OUTCOME_BENIGN) == OUTCOME_BENIGN
    assert normalize_harmful_outcome(OUTCOME_SECRET_EXPOSURE) == OUTCOME_SECRET_EXPOSURE
    with pytest.raises(ValueError, match="invalid primary_outcome"):
        normalize_primary_outcome("unknown")
    with pytest.raises(ValueError, match="invalid outcome_scores key"):
        normalize_harmful_outcome_float_map({"unknown": 0.8}, "outcome_scores")
