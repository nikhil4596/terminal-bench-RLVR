from tb_rlvr.actions import AgentAction
from tb_rlvr.rewards import RewardComponents
from tb_rlvr.rollout import RolloutRecord


def test_rollout_jsonl_roundtrip() -> None:
    record = RolloutRecord(
        task_id="toy",
        step=1,
        action=AgentAction(kind="bash", payload="ls"),
        reward=RewardComponents(total=0.1),
        done=False,
        observation_hash="abc",
        info={"x": 1},
    )
    restored = RolloutRecord.from_jsonl(record.to_jsonl())
    assert restored == record

