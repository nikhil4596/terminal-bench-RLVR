from tb_rlvr.contracts.actions import AgentAction
from tb_rlvr.contracts.rewards import RewardComponents
from tb_rlvr.contracts.rollout import (
    RolloutRecord,
    records_to_training_samples,
    terminal_records,
)


def test_rollout_exports_training_sample() -> None:
    record = RolloutRecord(
        task_id="synthetic-event-audit-001",
        step=3,
        action=AgentAction(kind="command", command="python solve.py"),
        reward=RewardComponents(correctness=1.0, total=0.95),
        done=True,
        observation_hash="abc",
        observation_prompt="Task: synthetic-event-audit-001",
        model_output='{"command":"python solve.py","task_complete":false}',
        terminal_reason="success",
        info={},
    )

    samples = records_to_training_samples([record])

    assert samples[0]["prompt"].startswith("Task: synthetic-event-audit-001")
    assert "python solve.py" in samples[0]["completion"]
    assert samples[0]["reward"] == 0.95
    assert samples[0]["metadata"]["terminal_reason"] == "success"


def test_terminal_records_filters_done_records() -> None:
    done = RolloutRecord(
        task_id="synthetic",
        step=1,
        action=AgentAction(kind="finish", task_complete=True),
        reward=RewardComponents(),
        done=True,
        observation_hash="done",
        info={},
    )
    active = RolloutRecord(
        task_id="synthetic",
        step=1,
        action=AgentAction(kind="command", command="ls"),
        reward=RewardComponents(),
        done=False,
        observation_hash="active",
        info={},
    )

    assert terminal_records([active, done]) == [done]
