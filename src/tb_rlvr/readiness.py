from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadinessItem:
    area: str
    status: str
    detail: str
    next_action: str


@dataclass(frozen=True)
class ReadinessReport:
    items: tuple[ReadinessItem, ...]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_markdown(self) -> str:
        lines = [
            "# Training Readiness Report",
            "",
            "| Area | Status | Detail | Next action |",
            "| --- | --- | --- | --- |",
        ]
        for item in self.items:
            lines.append(
                f"| {item.area} | {item.status} | {item.detail} | {item.next_action} |"
            )
        return "\n".join(lines)


def evaluate_training_readiness(root: Path | str | None = None) -> ReadinessReport:
    repo_root = Path(root) if root is not None else Path.cwd()
    items = [
        _files_item(
            repo_root,
            "Environment interface",
            [
                "src/tb_rlvr/actions.py",
                "src/tb_rlvr/observations.py",
                "src/tb_rlvr/env.py",
                "src/tb_rlvr/harbor.py",
                "src/tb_rlvr/safety.py",
            ],
            "Keep Harbor runtime execution behind this interface.",
        ),
        _files_item(
            repo_root,
            "Reward and rollout schema",
            [
                "src/tb_rlvr/rewards.py",
                "src/tb_rlvr/rollout.py",
                "src/tb_rlvr/trainers/export.py",
            ],
            "Export JSONL rollouts, then convert to trainer datasets.",
        ),
        _files_item(
            repo_root,
            "Training configs",
            [
                "configs/training/harbor_rollout.toml",
                "configs/training/trl_grpo_pilot.toml",
                "configs/training/verl_grpo_large_scale.toml",
            ],
            "Pin exact package versions when a compute target is chosen.",
        ),
        _files_item(
            repo_root,
            "Submission docs",
            [
                "README.md",
                "docs/submission_writeup.md",
                "docs/implementation_plan.md",
                "docs/data_lifecycle.md",
                "docs/training_readiness.md",
            ],
            "Keep exploratory research notes outside the submission docs.",
        ),
        _files_item(
            repo_root,
            "Laptop execution scripts",
            [
                "scripts/harbor_oracle_smoke.py",
                "scripts/export_mock_samples.py",
                "examples/run_mock_rollout.py",
            ],
            "Use dry-run scripts locally; execute Harbor only when Docker/uvx are available.",
        ),
        _cli_item(
            "Harbor CLI runner",
            ("uvx", "harbor"),
            "Install uv or Harbor before running real oracle smoke tasks.",
        ),
        _dependency_item(
            "TRL package",
            ("trl",),
            "Install only when moving to GPU-backed online GRPO pilot training.",
        ),
        _dependency_item(
            "Transformers package",
            ("transformers",),
            "Install with TRL when model policy sampling/logprobs are needed.",
        ),
        _dependency_item(
            "verl package",
            ("verl",),
            "Install only in a large-scale CUDA/Ray training image.",
        ),
        ReadinessItem(
            area="Local LLM/GPU training resources",
            status="external",
            detail="No local GPU, policy server, or API credentials are assumed by this repo.",
            next_action="Use Harbor oracle for runtime validation; add model access only for real policy rollouts.",
        ),
    ]
    return ReadinessReport(tuple(items))


def _files_item(
    root: Path, area: str, rel_paths: list[str], next_action: str
) -> ReadinessItem:
    missing = [path for path in rel_paths if not (root / path).exists()]
    if missing:
        return ReadinessItem(
            area=area,
            status="partial",
            detail="Missing: " + ", ".join(missing),
            next_action=next_action,
        )
    return ReadinessItem(
        area=area,
        status="ready",
        detail="Required repo files are present.",
        next_action=next_action,
    )


def _dependency_item(
    area: str, module_names: tuple[str, ...], next_action: str
) -> ReadinessItem:
    installed = any(importlib.util.find_spec(name) is not None for name in module_names)
    if installed:
        return ReadinessItem(
            area=area,
            status="installed",
            detail="Python package import is available in this environment.",
            next_action="Smoke-test against the pinned training image.",
        )
    return ReadinessItem(
        area=area,
        status="external",
        detail="Not installed locally; not required for laptop no-training validation.",
        next_action=next_action,
    )


def _cli_item(area: str, commands: tuple[str, ...], next_action: str) -> ReadinessItem:
    available = [command for command in commands if shutil.which(command)]
    if available:
        return ReadinessItem(
            area=area,
            status="available",
            detail="Available command(s): " + ", ".join(available),
            next_action="Run scripts/harbor_oracle_smoke.py --execute when Docker is running.",
        )
    return ReadinessItem(
        area=area,
        status="external",
        detail="No Harbor runner found on PATH.",
        next_action=next_action,
    )
