"""Terminal-Bench RLVR environment prototype.

This package intentionally stops before model training. It defines the
environment, action, reward, safety, and rollout interfaces that a future
GRPO/PPO trainer would consume.
"""

from .actions import AgentAction, ActionParseError, parse_action
from .env import MockTerminalBenchEnv
from .harbor import (
    HarborRunConfig,
    HarborRunResult,
    build_harbor_run_command,
    command_to_shell,
    harbor_runner_available,
    run_harbor_smoke,
)
from .observations import Observation
from .readiness import ReadinessItem, ReadinessReport, evaluate_training_readiness
from .rewards import RewardComponents, RewardWeights, combine_rewards
from .rollout import RolloutRecord
from .trainers import records_to_grpo_samples, records_to_training_samples, terminal_records

__all__ = [
    "ActionParseError",
    "AgentAction",
    "HarborRunConfig",
    "HarborRunResult",
    "MockTerminalBenchEnv",
    "Observation",
    "ReadinessItem",
    "ReadinessReport",
    "RewardComponents",
    "RewardWeights",
    "RolloutRecord",
    "build_harbor_run_command",
    "combine_rewards",
    "command_to_shell",
    "evaluate_training_readiness",
    "harbor_runner_available",
    "parse_action",
    "records_to_grpo_samples",
    "records_to_training_samples",
    "run_harbor_smoke",
    "terminal_records",
]
