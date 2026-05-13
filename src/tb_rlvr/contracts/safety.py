from __future__ import annotations

import re
from dataclasses import dataclass

from .actions import AgentAction


PROTECTED_PATHS = (
    "/tests",
    "/solution",
    "/logs/verifier",
    "/logs/reward",
    "/oracle",
    "/grader",
)

FORBIDDEN_COMMAND_SNIPPETS = (
    "rm -rf /",
    "chmod -r 777 /",
    "mkfs",
    "dd if=",
    "curl http",
    "curl https",
    "wget http",
    "wget https",
)


@dataclass(frozen=True)
class SafetyResult:
    ok: bool
    reason: str = ""


def check_action_safety(action: AgentAction) -> SafetyResult:
    if action.kind == "finish":
        return SafetyResult(True)

    command = action.command.strip()
    lowered = command.lower()

    for snippet in FORBIDDEN_COMMAND_SNIPPETS:
        if snippet in lowered:
            return SafetyResult(False, f"forbidden command snippet: {snippet}")

    for protected in PROTECTED_PATHS:
        if _mentions_path(lowered, protected):
            return SafetyResult(False, f"protected path access: {protected}")

    return SafetyResult(True)


def _mentions_path(command: str, protected_path: str) -> bool:
    escaped = re.escape(protected_path.lower())
    return re.search(rf"(^|[\s'\"=;:]){escaped}(/|\b)", command) is not None
