from tb_rlvr import MockTerminalBenchEnv, parse_action, records_to_grpo_samples


def test_mock_rollout_exports_training_sample() -> None:
    env = MockTerminalBenchEnv()
    env.reset()
    env.step(parse_action("<bash>echo solved > /app/answer.txt</bash>"))

    samples = records_to_grpo_samples(env.records)

    assert samples[0]["prompt"].startswith("Task: toy-create-answer")
    assert samples[0]["completion"] == "<bash>echo solved > /app/answer.txt</bash>"
    assert samples[0]["reward"] == 0.99
    assert samples[0]["metadata"]["terminal_reason"] == "success"
