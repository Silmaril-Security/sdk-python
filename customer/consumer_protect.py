"""Installed-wheel customer application using parallel protection. Inputs and assertions live in CustomerTests."""

import json
import os
from pathlib import Path

from silmaril_security.sdk import ClassifyEvent, Firewall, FirewallBlockedException, HookLabel

scenario = json.loads(Path(os.environ["CUSTOMER_SCENARIO"]).read_text())
run_id = os.environ["CUSTOMER_RUN_ID"]
events: list[ClassifyEvent] = []
client = Firewall(
    api_key=os.environ["SILMARIL_API_KEY"],
    api_url=os.environ["SILMARIL_API_URL"],
    mode="block",
    max_retries=0,
    on_classify=events.append,
)
client.classify(scenario["user_input"], hook=HookLabel.USER_INPUT, request_id=run_id + "-user")
rows = []
for case in scenario["cases"]:
    request_id = run_id + "-" + case["id"]
    summary = Path(case["id"] + ".summary.txt")
    blocked = False
    try:
        summary.write_text(
            client.protect(
                lambda: "Search performance improved; settings fixed.\n",
                text=case["text"],
                hook=HookLabel.TOOL_RESPONSE,
                tool_name=scenario["tool_name"],
                request_id=request_id,
            )
        )
    except FirewallBlockedException:
        blocked = True
    rows.append(
        dict(
            id=case["id"],
            request_id=request_id,
            prediction=events[-1].result.prediction,
            blocked=blocked,
            wrote_summary=summary.is_file(),
        )
    )
Path(os.environ["CUSTOMER_RESULT"]).write_text(json.dumps(dict(run_id=run_id, cases=rows)))
