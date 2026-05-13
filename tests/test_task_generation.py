from tb_rlvr.task_generation import generate_event_audit_task


def test_generation_is_deterministic() -> None:
    first = generate_event_audit_task(7)
    second = generate_event_audit_task(7)
    assert first == second
    assert first.task_id == "synthetic-event-audit-7"
    assert "Terminal-Bench" not in first.instruction
