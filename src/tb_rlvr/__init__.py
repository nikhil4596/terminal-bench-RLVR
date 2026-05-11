"""Terminal-Bench RLVR environment prototype.

This package intentionally stops before model training. It defines the
environment, action, reward, safety, and rollout interfaces that a future
GRPO/PPO trainer would consume.
"""

from .actions import AgentAction, ActionParseError, parse_action
from .env import MockTerminalBenchEnv
from .observations import Observation
from .readiness import ReadinessItem, ReadinessReport, evaluate_training_readiness
from .rewards import RewardComponents, RewardWeights, combine_rewards
from .rollout import RolloutRecord
from .trainers import records_to_grpo_samples, terminal_records

__all__ = [
    "ActionParseError",
    "AgentAction",
    "MockTerminalBenchEnv",
    "Observation",
    "ReadinessItem",
    "ReadinessReport",
    "RewardComponents",
    "RewardWeights",
    "RolloutRecord",
    "combine_rewards",
    "evaluate_training_readiness",
    "parse_action",
    "records_to_grpo_samples",
    "terminal_records",
]
