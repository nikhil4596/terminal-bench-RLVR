from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_rlvr import MockTerminalBenchEnv, parse_action, records_to_training_samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one toy rollout and export trainer-neutral samples."
    )
    parser.add_argument(
        "--output",
        default="outputs/mock_training_samples.jsonl",
        help="Path for exported JSONL samples.",
    )
    args = parser.parse_args()

    env = MockTerminalBenchEnv()
    env.reset()
    env.step(parse_action("<bash>echo solved > /app/answer.txt</bash>"))
    samples = records_to_training_samples(env.records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")

    print(f"wrote {len(samples)} sample(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
