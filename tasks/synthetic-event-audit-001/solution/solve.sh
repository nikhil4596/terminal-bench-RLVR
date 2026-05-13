#!/bin/bash
set -euo pipefail
WORKSPACE="${WORKSPACE:-/app}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import collections
import json
import os
from pathlib import Path

workspace = Path(os.environ.get("WORKSPACE", "/app"))
events = (workspace / "input" / "events.log").read_text().splitlines()
errors = collections.Counter()
warnings = 0
for line in events:
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    if fields["level"] == "ERROR":
        errors[fields["service"]] += 1
    if fields["level"] == "WARN":
        warnings += 1
report = {
    "total_events": len(events),
    "error_count": sum(errors.values()),
    "warning_count": warnings,
    "error_services": dict(sorted(errors.items())),
}
(workspace / "report.json").write_text(json.dumps(report, sort_keys=True) + "\n")
PY
