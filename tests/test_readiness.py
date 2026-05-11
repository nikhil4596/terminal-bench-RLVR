from pathlib import Path

from tb_rlvr import evaluate_training_readiness


def test_readiness_report_has_core_ready_items() -> None:
    root = Path(__file__).resolve().parents[1]
    report = evaluate_training_readiness(root)
    by_area = {item.area: item for item in report.items}

    assert by_area["Environment interface"].status == "ready"
    assert by_area["Reward and rollout schema"].status == "ready"
    assert "Training configs" in by_area
