from __future__ import annotations

import pytest

from silmaril_security.sdk import (
    AsyncFirewall,
    Firewall,
    FirewallBlockedException,
    GovernanceContext,
    GovernanceResource,
)
from silmaril_security.sdk.firewall import _block_result_from_json


def test_single_governance_wire_decision_and_legacy(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    payloads = []

    def post(payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5,
                    "governance": {"action": "block", "policy_version": "v2", "rule_id": "rule-1"}}
        return {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5}

    monkeypatch.setattr(fw, "_post_json", post)
    context = GovernanceContext(agent="agent-1", resource=GovernanceResource(kind="tool", id="search", parent_id="server-1"))
    with pytest.raises(FirewallBlockedException) as exc:
        fw.classify("search", governance=context)
    assert exc.value.result.governance.rule_id == "rule-1"
    assert payloads[0]["metadata"]["silmaril"]["governance"] == {
        "agent": "agent-1", "resource": {"kind": "tool", "id": "search", "parent_id": "server-1"}}
    assert fw.classify("safe").governance is None
    with pytest.raises(ValueError, match="governance action"):
        _block_result_from_json({"prediction": "BENIGN", "score": 0.1, "threshold": 0.5,
                                 "governance": {"action": "other", "policy_version": "v2"}})


@pytest.mark.asyncio
async def test_async_batch_governance(monkeypatch):
    fw = AsyncFirewall(api_key="sk", api_url="https://example.com/classify")
    payloads = []

    async def post(payload):
        payloads.append(payload)
        return {"predictions": [
            {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5, "mode": "warn",
             "governance": {"action": "block", "policy_version": "v2"}},
            {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5, "mode": "warn"},
        ]}

    monkeypatch.setattr(fw, "_post_json", post)
    results = await fw.classify_batch(["a", "b"], governance=[GovernanceContext(agent="agent-1"), None])
    assert results[0].governance.action == "block"
    assert results[1].governance is None
    assert payloads[0]["metadata"][0]["silmaril"]["governance"] == {"agent": "agent-1"}
    assert "governance" not in payloads[0]["metadata"][1]["silmaril"]
    await fw.aclose()
