from tb_rlvr.contracts.rewards import combine_rewards, reward_from_verifier


def test_correctness_dominates_reward() -> None:
    reward = combine_rewards(
        correctness=1.0,
        integrity_violation=False,
        turn_count=5,
        max_turns=20,
    )
    assert reward.correctness == 1.0
    assert reward.total > 0.9
    assert reward.success == 1.0


def test_efficiency_is_correctness_gated() -> None:
    reward = combine_rewards(
        correctness=0.0,
        integrity_violation=False,
        turn_count=1,
        max_turns=20,
    )
    assert reward.efficiency == 0.0
    assert reward.total == 0.0


def test_integrity_violation_is_negative() -> None:
    reward = combine_rewards(
        correctness=1.0,
        integrity_violation=True,
        turn_count=1,
        max_turns=20,
    )
    assert reward.integrity == -1.0
    assert reward.total < 0


def test_reward_from_verifier_uses_correctness() -> None:
    reward = reward_from_verifier(
        {"correctness": 0.5, "reward": 0.1},
        integrity_violation=False,
        turn_count=10,
    )
    assert reward.correctness == 0.5
