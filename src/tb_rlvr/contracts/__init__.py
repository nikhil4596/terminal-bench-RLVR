"""Local contracts for a custom Harbor/SkyRL agent adapter.

Harbor consumes task directories directly, and Terminus-2/SkyRL have their own
runtime types. These dataclasses document and test the action, observation,
reward, safety, and rollout semantics expected by a future adapter or custom
agent loop.
"""

from .actions import AgentAction, ActionParseError, parse_action
from .observations import Observation
from .rewards import RewardComponents, RewardWeights, combine_rewards, reward_from_verifier
from .rollout import RolloutRecord, records_to_training_samples, terminal_records
from .safety import SafetyResult, check_action_safety

__all__ = [
    "ActionParseError",
    "AgentAction",
    "Observation",
    "RewardComponents",
    "RewardWeights",
    "RolloutRecord",
    "SafetyResult",
    "check_action_safety",
    "combine_rewards",
    "parse_action",
    "records_to_training_samples",
    "reward_from_verifier",
    "terminal_records",
]
