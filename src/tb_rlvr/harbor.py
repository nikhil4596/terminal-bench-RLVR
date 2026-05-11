from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class HarborRunConfig:
    dataset: str = "terminal-bench@2.0"
    task_id: str | None = "openssl-selfsigned-cert"
    agent: str = "oracle"
    runner: tuple[str, ...] = ("uvx", "harbor")
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HarborRunResult:
    command: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False


def build_harbor_run_command(config: HarborRunConfig) -> tuple[str, ...]:
    command: list[str] = [*config.runner, "run", "-d", config.dataset]
    if config.task_id:
        command.extend(["-t", config.task_id])
    if config.agent:
        command.extend(["-a", config.agent])
    command.extend(config.extra_args)
    return tuple(command)


def harbor_runner_available(runner: Sequence[str] = ("uvx", "harbor")) -> bool:
    if not runner:
        return False
    return shutil.which(runner[0]) is not None


def run_harbor_smoke(
    config: HarborRunConfig | None = None, *, dry_run: bool = True
) -> HarborRunResult:
    config = config or HarborRunConfig()
    command = build_harbor_run_command(config)

    if dry_run:
        return HarborRunResult(command=command, returncode=None, skipped=True)

    if not harbor_runner_available(config.runner):
        return HarborRunResult(
            command=command,
            returncode=127,
            stderr=f"runner not found on PATH: {config.runner[0]}",
            skipped=True,
        )

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return HarborRunResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        skipped=False,
    )


def command_to_shell(command: Sequence[str]) -> str:
    return " ".join(_quote(part) for part in command)


def _quote(value: str) -> str:
    safe_value = (
        value.replace("-", "")
        .replace("_", "")
        .replace("@", "")
        .replace(".", "")
        .replace("/", "")
    )
    if safe_value.isalnum():
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
