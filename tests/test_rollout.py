from tb_rlvr.contracts.actions import AgentAction
from tb_rlvr.contracts.rewards import RewardComponents
from tb_rlvr.contracts.rollout import RolloutRecord


def test_rollout_jsonl_roundtrip() -> None:
    record = RolloutRecord(
        task_id="synthetic",
        step=1,
        action=AgentAction(kind="command", command="ls"),
        reward=RewardComponents(total=0.1),
        done=False,
        observation_hash="abc",
        info={"x": 1},
        atif_turn={"role": "assistant"},
        token_ids=(1, 2),
        mask_ids=(1, 1),
        logprobs=(-0.1, -0.2),
    )
    restored = RolloutRecord.from_jsonl(record.to_jsonl())
    assert restored == record


def test_training_sample_includes_token_fields() -> None:
    record = RolloutRecord(
        task_id="synthetic",
        step=1,
        action=AgentAction(kind="command", command="ls"),
        reward=RewardComponents(total=0.1),
        done=False,
        observation_hash="abc",
        observation_prompt="prompt",
        model_output='{"command":"ls","task_complete":false}',
        info={},
        token_ids=(1, 2),
        mask_ids=(1, 1),
    )
    sample = record.to_training_sample()
    assert sample["token_ids"] == [1, 2]
    assert sample["mask_ids"] == [1, 1]
    assert sample["metadata"]["token_capture_required_for_on_policy_rl"] is False
