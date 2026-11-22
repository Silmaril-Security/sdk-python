# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Classify wire bodies and transports shared by client tests."""

from __future__ import annotations

import json
from typing import Any

import requests

TEST_API_URL = "https://api.test.invalid/classify"
V1_URL = TEST_API_URL
V2_URL = "https://api.test.invalid/v2/classify"


def decision(action: str = "allow", **evidence: Any) -> dict[str, Any]:
    return {
        "action": action,
        "source": None,
        "confidence": None,
        "severity": None,
        "outcome": None,
        "governance": None,
        "model_id": None,
        **evidence,
    }


def model_block(**evidence: Any) -> dict[str, Any]:
    return decision("block", source="model", **evidence)


def governance_block(governance: dict[str, Any]) -> dict[str, Any]:
    return decision("block", source="governance", governance=governance)


def results(*decisions: dict[str, Any], **envelope: Any) -> dict[str, Any]:
    return {"results": list(decisions), **envelope}


def v1(prediction: str = "BENIGN", **fields: Any) -> dict[str, Any]:
    return {"prediction": prediction, "score": 0.5, "threshold": 0.5, **fields}


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: dict[str, Any] | str,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.reason = reason or f"status-{status_code}"
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self._body if isinstance(self._body, str) else str(self._body)

    def json(self) -> dict[str, Any]:
        assert isinstance(self._body, dict)
        return self._body


class RecordingAdapter(requests.adapters.BaseAdapter):
    def __init__(self, responses: list[tuple[int, dict[str, Any] | str]]) -> None:
        self.responses = responses
        self.requests: list[requests.PreparedRequest] = []

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        self.requests.append(request)
        status_code, body = self.responses.pop(0)
        response = requests.Response()
        response.status_code = status_code
        response.request = request
        response.url = request.url
        response._content = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
        response._content_consumed = True
        return response

    def close(self) -> None:
        pass
