from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from .actions import AgentAction
from .rewards import RewardComponents


@dataclass(frozen=True)
class RolloutRecord:
    """Local audit/export record for Harbor/SkyRL rollouts.

    Production training may convert Harbor/Terminus trial results directly into
    SkyRL GeneratorOutput. This record exists to make the expected fields
    explicit and testable in the submission repo.
    """

    task_id: str
    step: int
    action: AgentAction
    reward: RewardComponents
    done: bool
    observation_hash: str
    info: dict
    observation_prompt: str = ""
    model_output: str = ""
    next_observation_hash: str = ""
    terminal_reason: str = ""
    backend: str = "mock-harbor"
    episode_id: str = ""
    atif_turn: dict = field(default_factory=dict)
    token_ids: tuple[int, ...] = field(default_factory=tuple)
    mask_ids: tuple[int, ...] = field(default_factory=tuple)
    logprobs: tuple[float, ...] = field(default_factory=tuple)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def to_training_sample(self) -> dict[str, object]:
        """Return the logical fields needed by SkyRL/GRPO rollout loaders."""

        if not self.observation_prompt:
            raise ValueError("training sample requires observation_prompt")
        if not self.model_output:
            raise ValueError("training sample requires model_output")

        return {
            "prompt": self.observation_prompt,
            "completion": self.model_output,
            "reward": self.reward.total,
            "task_id": self.task_id,
            "done": self.done,
            "token_ids": list(self.token_ids),
            "mask_ids": list(self.mask_ids),
            "logprobs": list(self.logprobs),
            "atif_turn": self.atif_turn,
            "metadata": {
                "backend": self.backend,
                "episode_id": self.episode_id,
                "step": self.step,
                "terminal_reason": self.terminal_reason,
                "reward_components": asdict(self.reward),
                "observation_hash": self.observation_hash,
                "next_observation_hash": self.next_observation_hash,
                "info": self.info,
                "token_capture_required_for_on_policy_rl": not (
                    self.token_ids and self.mask_ids
                ),
            },
        }

    @classmethod
    def from_jsonl(cls, line: str) -> "RolloutRecord":
        data = json.loads(line)
        return cls(
            task_id=data["task_id"],
            step=data["step"],
            action=AgentAction(**data["action"]),
            reward=RewardComponents(**data["reward"]),
            done=data["done"],
            observation_hash=data["observation_hash"],
            info=data["info"],
            observation_prompt=data.get("observation_prompt", ""),
            model_output=data.get("model_output", ""),
            next_observation_hash=data.get("next_observation_hash", ""),
            terminal_reason=data.get("terminal_reason", ""),
            backend=data.get("backend", "mock-harbor"),
            episode_id=data.get("episode_id", ""),
            atif_turn=data.get("atif_turn", {}),
            token_ids=tuple(data.get("token_ids", ())),
            mask_ids=tuple(data.get("mask_ids", ())),
            logprobs=tuple(data.get("logprobs", ())),
        )


def records_to_training_samples(
    records: Iterable[RolloutRecord],
) -> list[dict[str, object]]:
    """Convert local rollout records to logical training sample rows.

    This is not a trainer integration. A production SkyRL adapter should return
    SkyRL GeneratorOutput directly; this helper is for audit/debug exports.
    """

    return [record.to_training_sample() for record in records]


def terminal_records(records: Iterable[RolloutRecord]) -> list[RolloutRecord]:
    """Keep only terminal records when exporting episode-level rewards."""

    return [record for record in records if record.done]
