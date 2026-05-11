from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_rlvr import HarborRunConfig, command_to_shell, run_harbor_smoke


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or dry-run a Harbor oracle smoke task."
    )
    parser.add_argument("--dataset", default="terminal-bench@2.0")
    parser.add_argument("--task", default="openssl-selfsigned-cert")
    parser.add_argument("--agent", default="oracle")
    parser.add_argument(
        "--runner",
        choices=("uvx", "harbor"),
        default="uvx",
        help="Use 'uvx harbor ...' or an installed 'harbor ...' CLI.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run Harbor. Without this flag the script only prints the command.",
    )
    args = parser.parse_args()

    runner = ("uvx", "harbor") if args.runner == "uvx" else ("harbor",)
    config = HarborRunConfig(
        dataset=args.dataset,
        task_id=args.task,
        agent=args.agent,
        runner=runner,
    )
    result = run_harbor_smoke(config, dry_run=not args.execute)
    print(command_to_shell(result.command))

    if result.skipped and result.returncode is None:
        print("dry run only; pass --execute to run the command")
        return 0

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
