from tb_rlvr import MockTerminalBenchEnv, parse_action


def test_mock_env_scripted_success() -> None:
    env = MockTerminalBenchEnv()
    env.reset()
    result = env.step(parse_action("<bash>echo solved > /app/answer.txt</bash>"))
    assert result.done
    assert result.reward.success == 1.0


def test_mock_env_integrity_violation_terminates() -> None:
    env = MockTerminalBenchEnv()
    env.reset()
    result = env.step(parse_action("<patch path='/app/tests/test.py'>pass</patch>"))
    assert result.done
    assert result.reward.integrity == -1.0


def test_mock_env_timeout() -> None:
    env = MockTerminalBenchEnv(max_steps=1)
    env.reset()
    result = env.step(parse_action("<bash>ls</bash>"))
    assert result.done
    assert result.reward.success == 0.0

