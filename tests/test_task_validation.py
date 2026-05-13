from pathlib import Path

from tb_rlvr.validate_tasks import (
    TERMINAL_BENCH_2_TASK_IDS,
    validate_task_dir,
    validate_tasks_root,
)


def test_decontamination_list_matches_tb2_task_count() -> None:
    assert len(TERMINAL_BENCH_2_TASK_IDS) == 89


def test_synthetic_task_validates_static_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate_task_dir(root / "tasks" / "synthetic-event-audit-001")
    assert report.ok, report.to_markdown()


def test_synthetic_tasks_root_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate_tasks_root(root / "tasks")
    assert report.ok, report.to_markdown()
