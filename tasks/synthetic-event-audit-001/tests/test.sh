#!/bin/bash
set -euo pipefail
WORKSPACE="${WORKSPACE:-/app}"
LOG_DIR="${LOG_DIR:-/logs/verifier}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$LOG_DIR"
"$PYTHON_BIN" - <<'PY'
import collections
import json
import os
from pathlib import Path

workspace = Path(os.environ.get("WORKSPACE", "/app"))
log_dir = Path(os.environ.get("LOG_DIR", "/logs/verifier"))
details = {}

events = (workspace / "input" / "events.log").read_text().splitlines()
errors = collections.Counter()
warnings = 0
for line in events:
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    if fields["level"] == "ERROR":
        errors[fields["service"]] += 1
    if fields["level"] == "WARN":
        warnings += 1
expected = {
    "total_events": len(events),
    "error_count": sum(errors.values()),
    "warning_count": warnings,
    "error_services": dict(sorted(errors.items())),
}

report_path = workspace / "report.json"
details["report_exists"] = report_path.exists()
report = None
if report_path.exists():
    try:
        report = json.loads(report_path.read_text())
        details["valid_json"] = isinstance(report, dict)
    except Exception:
        details["valid_json"] = False
else:
    details["valid_json"] = False

if isinstance(report, dict):
    details["total_events"] = report.get("total_events") == expected["total_events"]
    details["error_count"] = report.get("error_count") == expected["error_count"]
    details["warning_count"] = report.get("warning_count") == expected["warning_count"]
    details["error_services"] = report.get("error_services") == expected["error_services"]
else:
    details["total_events"] = False
    details["error_count"] = False
    details["warning_count"] = False
    details["error_services"] = False

score = sum(1 for passed in details.values() if passed) / len(details)
payload = {
    "reward": score,
    "correctness": score,
    "details": details,
}
(log_dir / "reward.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
if score < 1.0:
    raise SystemExit(1)
PY
