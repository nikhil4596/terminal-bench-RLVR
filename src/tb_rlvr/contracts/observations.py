from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Observation:
    """Local contract for visible state in a Harbor terminal-agent rollout."""

    task_id: str
    instruction: str
    cwd: str = "/app"
    terminal_prompt: str = "root@container:/app#"
    recent_history: tuple[str, ...] = field(default_factory=tuple)
    atif_history: tuple[str, ...] = field(default_factory=tuple)
    last_stdout: str = ""
    last_stderr: str = ""
    steps_remaining: int = 20

    def to_prompt(self) -> str:
        return "\n\n".join(
            part
            for part in [
                f"Task: {self.task_id}",
                f"Instruction:\n{self.instruction}",
                f"CWD: {self.cwd}",
                f"Terminal prompt: {self.terminal_prompt}",
                "Recent terminal turns:\n" + "\n".join(self.recent_history),
                "ATIF history summary:\n" + "\n".join(self.atif_history),
                f"Last stdout:\n{self.last_stdout}",
                f"Last stderr:\n{self.last_stderr}",
                f"Turns remaining: {self.steps_remaining}",
                (
                    "Respond with exactly one JSON object containing "
                    "`rationale`, `command`, and `task_complete`."
                ),
            ]
            if part.strip()
        )
