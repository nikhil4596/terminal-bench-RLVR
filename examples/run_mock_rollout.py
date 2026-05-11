from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_rlvr import MockTerminalBenchEnv, parse_action


def main() -> None:
    env = MockTerminalBenchEnv()
    observation = env.reset()
    print(observation.to_prompt())

    for text_action in [
        "<bash>echo solved > /app/answer.txt</bash>",
        "<finish>ready for grading</finish>",
    ]:
        result = env.step(parse_action(text_action))
        print(result.reward)
        if result.done:
            break

    for record in env.records:
        print(record.to_jsonl())


if __name__ == "__main__":
    main()
