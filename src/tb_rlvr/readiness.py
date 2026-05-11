from __future__ import annotations

import importlib.util
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
                "src/tb_rlvr/safety.py",
            ],
            "Keep Harbor adapter behind this interface.",
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
                "docs/training_readiness.md",
            ],
            "Keep exploratory research notes outside the submission docs.",
        ),
        _dependency_item(
            "Harbor package",
            ("harbor",),
            "Install Harbor in the training image before real Terminal-Bench rollouts.",
        ),
        _dependency_item(
            "TRL package",
            ("trl",),
            "Install only for the single-node or small-cluster GRPO pilot.",
        ),
        _dependency_item(
            "verl package",
            ("verl",),
            "Install only in the large-scale CUDA/Ray training image.",
        ),
        ReadinessItem(
            area="Distributed infrastructure",
            status="external",
            detail="No GPU cluster, Ray runtime, object store, or checkpoint volume is configured locally.",
            next_action="Attach Slurm/Kubernetes/Ray launch files once the target cluster is known.",
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
        detail="Not installed locally; this is expected for the no-training repo.",
        next_action=next_action,
    )
