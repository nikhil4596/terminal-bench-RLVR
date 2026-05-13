from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_rlvr.task_generation import generate_event_audit_task, render_harbor_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--output-root", default="tasks")
    args = parser.parse_args()

    spec = generate_event_audit_task(args.seed, args.task_id)
    task_dir = render_harbor_task(spec, Path(args.output_root))
    print(task_dir)


if __name__ == "__main__":
    main()
