from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ActionKind = Literal["bash", "patch", "finish"]


class ActionParseError(ValueError):
    """Raised when a model response does not contain exactly one valid action."""


@dataclass(frozen=True)
class AgentAction:
    kind: ActionKind
    payload: str
    path: str | None = None

    def to_model_text(self) -> str:
        if self.kind == "bash":
            return f"<bash>{self.payload}</bash>"
        if self.kind == "patch":
            if self.path is None:
                raise ValueError("patch action requires a path")
            return f"<patch path='{self.path}'>{self.payload}</patch>"
        return f"<finish>{self.payload}</finish>"


_BASH_RE = re.compile(r"<bash>\s*(?P<payload>.*?)\s*</bash>", re.DOTALL)
_FINISH_RE = re.compile(r"<finish>\s*(?P<payload>.*?)\s*</finish>", re.DOTALL)
_PATCH_RE = re.compile(
    r"<patch\s+path=[\"'](?P<path>[^\"']+)[\"']>\s*(?P<payload>.*?)\s*</patch>",
    re.DOTALL,
)


def parse_action(text: str) -> AgentAction:
    """Parse one structured action from model text.

    The action protocol intentionally allows only one environment mutation per
    assistant turn. This makes rollouts easier to audit and train on.
    """

    matches: list[AgentAction] = []

    for match in _BASH_RE.finditer(text):
        payload = match.group("payload").strip()
        if payload:
            matches.append(AgentAction(kind="bash", payload=payload))

    for match in _PATCH_RE.finditer(text):
        payload = match.group("payload").strip()
        path = match.group("path").strip()
        if payload and path:
            matches.append(AgentAction(kind="patch", payload=payload, path=path))

    for match in _FINISH_RE.finditer(text):
        payload = match.group("payload").strip()
        matches.append(AgentAction(kind="finish", payload=payload))

    if len(matches) != 1:
        raise ActionParseError(f"expected exactly one action, found {len(matches)}")

    return matches[0]
