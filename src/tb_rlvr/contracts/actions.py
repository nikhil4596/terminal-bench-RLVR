from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


ActionKind = Literal["command", "finish"]


class ActionParseError(ValueError):
    """Raised when a model response is not one valid terminal-agent turn."""


@dataclass(frozen=True)
class AgentAction:
    """Local contract for one Terminus-style terminal-agent turn.

    The training target is a JSON command turn, not a custom XML patch format.
    File edits happen through shell commands inside the container, matching the
    Harbor/Terminus evaluation loop used for Terminal-Bench 2.0.
    """

    kind: ActionKind
    command: str = ""
    rationale: str = ""
    task_complete: bool = False

    @property
    def payload(self) -> str:
        return self.command

    def to_model_text(self) -> str:
        if self.kind == "finish":
            return json.dumps(
                {
                    "rationale": self.rationale,
                    "command": "",
                    "task_complete": True,
                },
                sort_keys=True,
            )

        return json.dumps(
            {
                "rationale": self.rationale,
                "command": self.command,
                "task_complete": False,
            },
            sort_keys=True,
        )


def parse_action(text: str) -> AgentAction:
    """Parse one JSON terminal-agent action.

    Supported shape:

    ```json
    {"rationale": "...", "command": "python solve.py", "task_complete": false}
    ```

    For compatibility with common agent schemas, `"commands": ["..."]` is also
    accepted when it contains exactly one command.
    """

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionParseError("expected a JSON object action") from exc

    if not isinstance(data, dict):
        raise ActionParseError("expected a JSON object action")

    rationale = str(data.get("rationale") or data.get("analysis") or "").strip()
    task_complete = bool(data.get("task_complete", False))

    command = data.get("command", "")
    if command == "" and "commands" in data:
        commands = data["commands"]
        if not isinstance(commands, list) or len(commands) != 1:
            raise ActionParseError("commands must contain exactly one command")
        command = commands[0]

    if command is None:
        command = ""
    if not isinstance(command, str):
        raise ActionParseError("command must be a string")

    command = command.strip()

    if task_complete:
        if command:
            raise ActionParseError("finish action must not include a command")
        return AgentAction(kind="finish", rationale=rationale, task_complete=True)

    if not command:
        raise ActionParseError("command action requires a non-empty command")

    return AgentAction(kind="command", command=command, rationale=rationale)
