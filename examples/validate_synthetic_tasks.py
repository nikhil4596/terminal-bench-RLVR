from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_rlvr.validate_tasks import validate_tasks_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", default="tasks")
    parser.add_argument("--run-local", action="store_true")
    args = parser.parse_args()

    report = validate_tasks_root(Path(args.tasks_root), run_local=args.run_local)
    print(report.to_markdown())
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
