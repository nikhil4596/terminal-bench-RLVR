from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_rlvr import evaluate_training_readiness


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(evaluate_training_readiness(root).to_markdown())


if __name__ == "__main__":
    main()
