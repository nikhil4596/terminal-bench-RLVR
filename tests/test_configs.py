import tomllib
from pathlib import Path


def test_training_configs_are_valid_toml() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs" / "training").glob("*.toml"):
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        assert data


def test_only_core_training_configs_are_present() -> None:
    root = Path(__file__).resolve().parents[1]
    names = {path.name for path in (root / "configs" / "training").glob("*.toml")}
    assert names == {"harbor_rollout.toml", "skyrl_grpo_pilot.toml"}
