from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path


TERMINAL_BENCH_2_TASK_IDS = (
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "break-filter-js-from-html",
    "build-cython-ext",
    "build-pmars",
    "build-pov-ray",
    "caffe-cifar-10",
    "cancel-async-tasks",
    "chess-best-move",
    "circuit-fibsqrt",
    "cobol-modernization",
    "code-from-image",
    "compile-compcert",
    "configure-git-webserver",
    "constraints-scheduling",
    "count-dataset-tokens",
    "crack-7z-hash",
    "custom-memory-heap-crash",
    "db-wal-recovery",
    "distribution-search",
    "dna-assembly",
    "dna-insert",
    "extract-elf",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "filter-js-from-html",
    "financial-document-processor",
    "fix-code-vulnerability",
    "fix-git",
    "fix-ocaml-gc",
    "gcode-to-text",
    "git-leak-recovery",
    "git-multibranch",
    "gpt2-codegolf",
    "headless-terminal",
    "hf-model-inference",
    "install-windows-3.11",
    "kv-store-grpc",
    "large-scale-text-editing",
    "largest-eigenval",
    "llm-inference-batching-scheduler",
    "log-summary-date-ranges",
    "mailman",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "merge-diff-arc-agi-task",
    "model-extraction-relu-logits",
    "modernize-scientific-stack",
    "mteb-leaderboard",
    "mteb-retrieve",
    "multi-source-data-merger",
    "nginx-request-logging",
    "openssl-selfsigned-cert",
    "overfull-hbox",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-c-py",
    "polyglot-rust-c",
    "portfolio-optimization",
    "protein-assembly",
    "prove-plus-comm",
    "pypi-server",
    "pytorch-model-cli",
    "pytorch-model-recovery",
    "qemu-alpine-ssh",
    "qemu-startup",
    "query-optimize",
    "raman-fitting",
    "regex-chess",
    "regex-log",
    "reshard-c4-data",
    "rstan-to-pystan",
    "sam-cell-seg",
    "sanitize-git-repo",
    "schemelike-metacircular-eval",
    "sparql-university",
    "sqlite-db-truncate",
    "sqlite-with-gcov",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "tune-mjcf",
    "video-processing",
    "vulnerable-secret",
    "winning-avg-corewars",
    "write-compressor",
)


@dataclass(frozen=True)
class ValidationIssue:
    task: str
    severity: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_markdown(self) -> str:
        if not self.issues:
            return "No validation issues found."
        lines = ["| Task | Severity | Message |", "| --- | --- | --- |"]
        for issue in self.issues:
            lines.append(f"| {issue.task} | {issue.severity} | {issue.message} |")
        return "\n".join(lines)


def validate_tasks_root(root: Path | str, *, run_local: bool = False) -> ValidationReport:
    task_root = Path(root)
    issues: list[ValidationIssue] = []
    for task_dir in sorted(task_root.glob("synthetic-*")):
        issues.extend(validate_task_dir(task_dir, run_local=run_local).issues)
    if not list(task_root.glob("synthetic-*")):
        issues.append(ValidationIssue(str(task_root), "error", "no synthetic-* tasks found"))
    return ValidationReport(tuple(issues))


def validate_task_dir(task_dir: Path | str, *, run_local: bool = False) -> ValidationReport:
    path = Path(task_dir)
    issues: list[ValidationIssue] = []
    task_name = path.name

    required = (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/workspace/input/events.log",
        "tests/test.sh",
        "solution/solve.sh",
    )
    for rel_path in required:
        if not (path / rel_path).exists():
            issues.append(ValidationIssue(task_name, "error", f"missing {rel_path}"))

    if (path / "task.toml").exists():
        try:
            task_config = tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            issues.append(ValidationIssue(task_name, "error", f"invalid task.toml: {exc}"))
        else:
            if task_config.get("schema_version") != "1.1":
                issues.append(
                    ValidationIssue(task_name, "warning", "expected Harbor schema_version 1.1")
                )
            authors = task_config.get("task", {}).get("authors", [])
            if authors and not all(isinstance(author, dict) for author in authors):
                issues.append(
                    ValidationIssue(
                        task_name,
                        "error",
                        "task.authors should use Harbor author objects",
                    )
                )

    if (path / "instruction.md").exists():
        instruction = (path / "instruction.md").read_text(encoding="utf-8")
        for tb_task_id in TERMINAL_BENCH_2_TASK_IDS:
            if _contains_tb2_name(task_name, instruction, tb_task_id):
                issues.append(
                    ValidationIssue(
                        task_name,
                        "error",
                        f"possible Terminal-Bench 2.0 contamination: {tb_task_id}",
                    )
                )

    if (path / "tests" / "test.sh").exists():
        test_script = (path / "tests" / "test.sh").read_text(encoding="utf-8")
        if "reward.json" not in test_script:
            issues.append(
                ValidationIssue(task_name, "error", "test.sh does not write reward.json")
            )
        if "correctness" not in test_script:
            issues.append(
                ValidationIssue(task_name, "error", "test.sh lacks correctness reward")
            )

    if run_local:
        issues.extend(_validate_local_oracle_and_dummy(path))

    return ValidationReport(tuple(issues))


def _contains_tb2_name(task_name: str, instruction: str, tb_task_id: str) -> bool:
    normalized_haystack = _normalize_for_contamination(task_name + " " + instruction)
    normalized_needle = _normalize_for_contamination(tb_task_id)
    return normalized_needle in normalized_haystack


def _normalize_for_contamination(text: str) -> str:
    chars = []
    for char in text.lower():
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def _validate_local_oracle_and_dummy(task_dir: Path) -> list[ValidationIssue]:
    task_name = task_dir.name
    if os.name == "nt":
        return [
            ValidationIssue(
                task_name,
                "warning",
                "local oracle/dummy execution skipped on Windows; run this check in Harbor or Linux",
            )
        ]
    if shutil.which("bash") is None:
        return [
            ValidationIssue(
                task_name,
                "warning",
                "bash not available; skipped local oracle/dummy execution",
            )
        ]

    issues: list[ValidationIssue] = []
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        dummy_reward = _run_task_scripts(task_dir, temp_root / "dummy", run_solution=False)
        if dummy_reward >= 1.0:
            issues.append(ValidationIssue(task_name, "error", "dummy workspace passed"))

        oracle_reward = _run_task_scripts(task_dir, temp_root / "oracle", run_solution=True)
        if oracle_reward < 1.0:
            issues.append(
                ValidationIssue(task_name, "error", "oracle solution did not pass")
            )

    return issues


def _run_task_scripts(task_dir: Path, run_root: Path, *, run_solution: bool) -> float:
    workspace = run_root / "workspace"
    log_dir = run_root / "logs" / "verifier"
    workspace.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    shutil.copytree(task_dir / "environment" / "workspace", workspace, dirs_exist_ok=True)

    env = {
        **os.environ,
        "WORKSPACE": str(workspace),
        "LOG_DIR": str(log_dir),
        "PYTHON_BIN": "python3",
    }
    if run_solution:
        solution = subprocess.run(
            ["bash", str(task_dir / "solution" / "solve.sh")],
            check=False,
            env=env,
            cwd=str(task_dir),
            capture_output=True,
            text=True,
        )
        if solution.returncode != 0:
            return 0.0

    subprocess.run(
        ["bash", str(task_dir / "tests" / "test.sh")],
        check=False,
        env=env,
        cwd=str(task_dir),
        capture_output=True,
        text=True,
    )
    reward_file = log_dir / "reward.json"
    if not reward_file.exists():
        return 0.0
    payload = json.loads(reward_file.read_text(encoding="utf-8"))
    return float(payload.get("correctness", payload.get("reward", 0.0)))
