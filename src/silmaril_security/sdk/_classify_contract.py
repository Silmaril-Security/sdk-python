# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Client rule of classify contract 2.0.0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

FALLBACK_STATUS_CODES = frozenset({403, 404, 405})
_V2_PATH = "/v2/classify"
_V1_PATH = "/classify"
_MODEL_LABELS = ("BENIGN", "MALICIOUS")


@dataclass
class ClassifyRoute:
    """Endpoints derived from a configured URL and the version its runtime answers."""

    v2_url: str
    v1_url: str
    legacy: bool = False

    @classmethod
    def from_api_url(cls, api_url: str) -> ClassifyRoute:
        url = urlsplit(api_url)
        path = url.path.removesuffix("/")
        base = next(
            (path.removesuffix(suffix) for suffix in (_V2_PATH, _V1_PATH) if path.endswith(suffix)),
            path,
        )
        return cls(*(url._replace(path=base + suffix).geturl() for suffix in (_V2_PATH, _V1_PATH)))


def is_v2_response(body: Any) -> bool:
    return isinstance(body, Mapping) and "results" in body


def v1_items(body: Any) -> list[Mapping[str, Any]]:
    if not isinstance(body, Mapping):
        raise ValueError("Firewall: backend response must be an object")
    items = body["predictions"] if "predictions" in body else [body]
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise ValueError("Firewall: backend predictions must be a list of objects")
    return items


def adapt_v1_response(body: Any) -> dict[str, Any]:
    """Derive the version 2 body a version 1 response stands for."""
    adapted: dict[str, Any] = {"results": [_adapt_v1_result(item) for item in v1_items(body)]}
    if "review" in body:
        adapted["review"] = body["review"]
    return adapted


def _adapt_v1_result(item: Mapping[str, Any]) -> dict[str, Any]:
    prediction = item.get("prediction")
    if prediction is None:
        raise ValueError("Firewall: backend response missing required 'prediction' field")
    label = item.get("model_prediction")
    if label is None:
        label = prediction
    for value in (prediction, label):
        if value not in _MODEL_LABELS:
            raise ValueError(f"Firewall: invalid prediction {value!r}")
    governance = item.get("governance")
    blocks = prediction == "MALICIOUS" or (
        isinstance(governance, Mapping) and governance.get("action") == "block"
    )
    model_blocks = blocks and label == "MALICIOUS"
    return {
        "action": "block" if blocks else "allow",
        "source": "model" if model_blocks else "governance" if blocks else None,
        "confidence": None,
        "severity": None,
        "outcome": (
            item.get("primary_outcome") if model_blocks and item.get("outcome_scores") else None
        ),
        "governance": governance,
        "model_id": item.get("model_id"),
    }
