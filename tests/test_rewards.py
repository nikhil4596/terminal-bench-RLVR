from tb_rlvr.rewards import combine_rewards


def test_success_reward_dominates_progress() -> None:
    reward = combine_rewards(
        success=True,
        progress_delta=0.0,
        integrity_violation=False,
        step_count=5,
    )
    assert reward.success == 1.0
    assert reward.total > 0.9


def test_progress_reward_is_capped() -> None:
    reward = combine_rewards(
        success=False,
        progress_delta=10.0,
        integrity_violation=False,
        step_count=0,
    )
    assert reward.progress == 0.20


def test_integrity_violation_is_negative() -> None:
    reward = combine_rewards(
        success=False,
        progress_delta=0.20,
        integrity_violation=True,
        step_count=1,
    )
    assert reward.integrity == -1.0
    assert reward.total < 0

