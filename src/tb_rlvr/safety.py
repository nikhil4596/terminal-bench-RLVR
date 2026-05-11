from __future__ import annotations

import posixpath
from dataclasses import dataclass

from .actions import AgentAction


PROTECTED_PREFIXES = (
    "/tests",
    "/app/tests",
    "/solution",
    "/app/solution",
    "/oracle",
    "/app/oracle",
    "/grader",
    "/app/grader",
)

FORBIDDEN_COMMAND_SNIPPETS = (
    "rm -rf /",
    "chmod -R 777 /",
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
    if action.kind == "patch":
        if action.path is None:
            return SafetyResult(False, "patch action is missing path")
        normalized = _normalize_path(action.path)
        for prefix in PROTECTED_PREFIXES:
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return SafetyResult(False, f"protected path mutation: {normalized}")

    if action.kind == "bash":
        lowered = action.payload.lower()
        for snippet in FORBIDDEN_COMMAND_SNIPPETS:
            if snippet in lowered:
                return SafetyResult(False, f"forbidden command snippet: {snippet}")

    return SafetyResult(True)


def _normalize_path(path: str) -> str:
    raw = path if path.startswith("/") else f"/app/{path}"
    return "/" + posixpath.normpath(raw).lstrip("/")
