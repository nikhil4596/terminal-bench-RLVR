from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Observation:
    task_id: str
    instruction: str
    cwd: str = "/app"
    directory_summary: str = ""
    recent_history: tuple[str, ...] = field(default_factory=tuple)
    last_stdout: str = ""
    last_stderr: str = ""
    selected_files: dict[str, str] = field(default_factory=dict)
    steps_remaining: int = 30

    def to_prompt(self) -> str:
        file_blocks = []
        for path, content in sorted(self.selected_files.items()):
            file_blocks.append(f"File: {path}\n{content}")

        return "\n\n".join(
            part
            for part in [
                f"Task: {self.task_id}",
                f"Instruction:\n{self.instruction}",
                f"CWD: {self.cwd}",
                f"Directory summary:\n{self.directory_summary}",
                "Recent actions:\n" + "\n".join(self.recent_history),
                f"Last stdout:\n{self.last_stdout}",
                f"Last stderr:\n{self.last_stderr}",
                "\n\n".join(file_blocks),
                f"Steps remaining: {self.steps_remaining}",
            ]
            if part.strip()
        )

