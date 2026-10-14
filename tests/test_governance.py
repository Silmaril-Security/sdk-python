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
    context = GovernanceContext(agent="agent-1", resource=GovernanceResource(kind="mcp_tool", id="search", parent_id="server-1"))
    with pytest.raises(FirewallBlockedException) as exc:
        fw.classify("search", governance=context)
    assert exc.value.result.governance.rule_id == "rule-1"
    assert payloads[0]["metadata"]["silmaril"]["governance"] == {
        "agent": "agent-1", "resource": {"kind": "mcp_tool", "id": "search", "parent_id": "server-1"}}
    assert payloads[0]["resource"] == {"kind": "mcp_tool", "id": "search", "parent_id": "server-1"}
    assert "agent" not in payloads[0]
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
    assert payloads[0]["resources"] == [None, None]
    await fw.aclose()


@pytest.mark.parametrize("resource", [
    GovernanceResource(kind="plugin"),
    GovernanceResource(kind="plugin", id=" "),
    GovernanceResource(kind="mcp_tool", id="tool"),
    GovernanceResource(kind="tool", id="tool", parent_id="server"),
    GovernanceResource(kind="unknown", id="tool"),
])
def test_invalid_resource_never_reaches_transport(monkeypatch, resource):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    def unexpected(payload):
        pytest.fail("invalid canonical resource reached transport")
    monkeypatch.setattr(fw, "_post_json", unexpected)
    with pytest.raises(ValueError):
        fw.classify("synthetic", governance=GovernanceContext(resource=resource))


def test_resource_deny_uses_producer_fields_not_caller_metadata(monkeypatch):
    fw = Firewall(api_key="sk", api_url="https://example.com/classify")
    def producer(payload):
        assert "agent" not in payload
        denied = payload.get("resource") == {"kind": "plugin", "id": "blocked-plugin"}
        return {"prediction": "BENIGN", "score": 0.1, "threshold": 0.5,
                "governance": {"action": "block" if denied else "allow", "policy_version": "fixture"}}
    monkeypatch.setattr(fw, "_post_json", producer)
    with pytest.raises(FirewallBlockedException):
        fw.classify("synthetic", governance=GovernanceContext(resource=GovernanceResource(kind="plugin", id="blocked-plugin")))
    result = fw.classify("synthetic", metadata={"silmaril": {"governance": {"resource": {"kind": "plugin", "id": "blocked-plugin"}}}})
    assert result.governance.action == "allow"


@pytest.mark.asyncio
async def test_async_nullable_resources_preserve_batch_order(monkeypatch):
    fw = AsyncFirewall(api_key="sk", api_url="https://example.com/classify", mode="warn")
    async def producer(payload):
        assert payload["resources"] == [None, {"kind": "plugin", "id": "blocked-plugin"}, None]
        return {"predictions": [{"prediction": "BENIGN", "score": 0.1, "threshold": 0.5} for _ in payload["texts"]]}
    monkeypatch.setattr(fw, "_post_json", producer)
    await fw.classify_batch(["a", "b", "c"], governance=[None, GovernanceContext(resource=GovernanceResource(kind="plugin", id="blocked-plugin")), GovernanceContext(agent="attribution-only")])
    await fw.aclose()
