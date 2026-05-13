from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


SERVICE_NAMES = ("auth", "billing", "search", "worker", "gateway")
LEVELS = ("INFO", "WARN", "ERROR")


@dataclass(frozen=True)
class SyntheticTaskSpec:
    task_id: str
    instruction: str
    events_log: str


def generate_event_audit_task(seed: int, task_id: str | None = None) -> SyntheticTaskSpec:
    """Generate a small Harbor-format terminal task spec.

    This intentionally creates a fresh synthetic task from a skill template
    instead of adapting Terminal-Bench 2.0 task text, tests, solutions, or file
    names.
    """

    rng = random.Random(seed)
    task_name = task_id or f"synthetic-event-audit-{seed}"
    rows = []
    for idx in range(12):
        service = rng.choice(SERVICE_NAMES)
        level = rng.choices(LEVELS, weights=(6, 2, 2), k=1)[0]
        rows.append(
            f"2026-01-01T00:{idx:02d}:00Z service={service} level={level} event=e{idx}"
        )

    instruction = (
        "Analyze `/app/input/events.log` and create `/app/report.json`. "
        "The JSON object must contain `total_events`, `error_count`, "
        "`warning_count`, and `error_services`. `error_services` must map each "
        "service name with at least one ERROR event to its count. Do not modify "
        "the input log."
    )
    return SyntheticTaskSpec(
        task_id=task_name,
        instruction=instruction,
        events_log="\n".join(rows) + "\n",
    )


def render_harbor_task(spec: SyntheticTaskSpec, output_root: Path) -> Path:
    task_dir = output_root / spec.task_id
    (task_dir / "environment" / "workspace" / "input").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(spec.instruction + "\n", encoding="utf-8")
    (task_dir / "environment" / "workspace" / "input" / "events.log").write_text(
        spec.events_log, encoding="utf-8"
    )
    (task_dir / "environment" / "Dockerfile").write_text(_dockerfile(), encoding="utf-8")
    (task_dir / "task.toml").write_text(_task_toml(spec.task_id), encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text(_test_script(), encoding="utf-8")
    (task_dir / "solution" / "solve.sh").write_text(_solution_script(), encoding="utf-8")
    return task_dir


def _task_toml(task_id: str) -> str:
    return f"""schema_version = "1.1"

[task]
name = "{task_id}"
description = "Synthetic Harbor terminal task for RLVR training."
authors = [{{ name = "terminal-bench-rlvr" }}]
keywords = ["synthetic", "terminal", "rlvr", "harbor"]

[metadata]
difficulty = "easy"
category = "data-processing"
source = "synthetic-skill-template"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 600.0

[environment]
build_timeout_sec = 600.0
os = "linux"
cpus = 1
memory_mb = 1024
storage_mb = 2048
allow_internet = false
"""


def _dockerfile() -> str:
    return """FROM python:3.11-slim
WORKDIR /app
COPY workspace/ /app/
"""


def _solution_script() -> str:
    return """#!/bin/bash
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
(workspace / "report.json").write_text(json.dumps(report, sort_keys=True) + "\\n")
PY
"""


def _test_script() -> str:
    expected_source = """
import collections
from pathlib import Path

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
"""
    return f"""#!/bin/bash
set -euo pipefail
WORKSPACE="${{WORKSPACE:-/app}}"
LOG_DIR="${{LOG_DIR:-/logs/verifier}}"
PYTHON_BIN="${{PYTHON_BIN:-python3}}"
mkdir -p "$LOG_DIR"
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

workspace = Path(os.environ.get("WORKSPACE", "/app"))
log_dir = Path(os.environ.get("LOG_DIR", "/logs/verifier"))
details = {{}}

{expected_source}

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
payload = {{
    "reward": score,
    "correctness": score,
    "details": details,
}}
(log_dir / "reward.json").write_text(json.dumps(payload, sort_keys=True) + "\\n")
if score < 1.0:
    raise SystemExit(1)
PY
"""


def spec_to_json(spec: SyntheticTaskSpec) -> str:
    return json.dumps(
        {
            "task_id": spec.task_id,
            "instruction": spec.instruction,
            "events_log": spec.events_log,
        },
        indent=2,
        sort_keys=True,
    )
