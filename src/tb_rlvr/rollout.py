from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .actions import AgentAction
from .rewards import RewardComponents


@dataclass(frozen=True)
class RolloutRecord:
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
    backend: str = "mock"
    episode_id: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def to_training_sample(self) -> dict[str, object]:
        """Return a backend-neutral GRPO/PPO sample row.

        TRL and verl use different concrete dataset loaders, but both need the
        same logical fields: prompt, sampled completion/action, scalar reward,
        and metadata for slicing failures.
        """

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
            "metadata": {
                "backend": self.backend,
                "episode_id": self.episode_id,
                "step": self.step,
                "terminal_reason": self.terminal_reason,
                "reward_components": asdict(self.reward),
                "observation_hash": self.observation_hash,
                "next_observation_hash": self.next_observation_hash,
                "info": self.info,
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
            backend=data.get("backend", "mock"),
            episode_id=data.get("episode_id", ""),
        )
