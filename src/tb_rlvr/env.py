from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .actions import AgentAction
from .observations import Observation
from .rewards import RewardComponents, combine_rewards
from .rollout import RolloutRecord
from .safety import check_action_safety


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: RewardComponents
    done: bool
    info: dict


@dataclass
class MockTerminalBenchEnv:
    """A deterministic mock of the Terminal-Bench RLVR interface.

    The real implementation would delegate execution to Harbor. This mock lets
    us test the action, reward, safety, and rollout contracts without Docker.
    """

    task_id: str = "toy-create-answer"
    instruction: str = "Create /app/answer.txt containing the string solved."
    max_steps: int = 5
    files: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    records: list[RolloutRecord] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    current_observation: Observation | None = None

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        del seed
        if task_id is not None:
            self.task_id = task_id
        self.files = {}
        self.history = []
        self.records = []
        self.step_count = 0
        self.done = False
        observation = self._observation(last_stdout="ready")
        self.current_observation = observation
        return observation

    def step(self, action: AgentAction) -> StepResult:
        if self.done:
            raise RuntimeError("episode is already done")

        prompt_before = (
            self.current_observation.to_prompt()
            if self.current_observation is not None
            else self._observation(last_stdout="ready").to_prompt()
        )
        self.step_count += 1
        safety = check_action_safety(action)
        success = False
        progress = 0.0
        stdout = ""
        stderr = ""
        info: dict = {"safety_ok": safety.ok}
        terminal_reason = ""

        if not safety.ok:
            self.done = True
            terminal_reason = "safety_violation"
            stderr = safety.reason
            reward = combine_rewards(
                success=False,
                progress_delta=0.0,
                integrity_violation=True,
                step_count=self.step_count,
            )
        else:
            if action.kind == "patch":
                assert action.path is not None
                self.files[action.path] = action.payload
                stdout = f"patched {action.path}"
                progress = 0.10 if action.path == "/app/answer.txt" else 0.02
            elif action.kind == "bash":
                stdout = self._run_bash(action.payload)
                progress = 0.05 if "test" in action.payload or "cat" in action.payload else 0.0
            elif action.kind == "finish":
                stdout = "running final verifier"

            success = self.files.get("/app/answer.txt", "").strip() == "solved"
            if success or action.kind == "finish" or self.step_count >= self.max_steps:
                self.done = True
                if success:
                    terminal_reason = "success"
                elif action.kind == "finish":
                    terminal_reason = "finish"
                else:
                    terminal_reason = "timeout"

            reward = combine_rewards(
                success=success,
                progress_delta=progress,
                integrity_violation=False,
                step_count=self.step_count,
            )

        self.history.append(f"{action.kind}: {action.payload[:80]}")
        obs = self._observation(last_stdout=stdout, last_stderr=stderr)
        next_observation_hash = _hash(obs.to_prompt())
        record = RolloutRecord(
            task_id=self.task_id,
            step=self.step_count,
            action=action,
            reward=reward,
            done=self.done,
            observation_hash=_hash(prompt_before),
            observation_prompt=prompt_before,
            model_output=action.to_model_text(),
            next_observation_hash=next_observation_hash,
            terminal_reason=terminal_reason,
            backend="mock",
            episode_id=f"{self.task_id}:mock",
            info=info,
        )
        self.records.append(record)
        self.current_observation = obs
        return StepResult(observation=obs, reward=reward, done=self.done, info=info)

    def _run_bash(self, command: str) -> str:
        if command.strip() == "cat /app/answer.txt":
            return self.files.get("/app/answer.txt", "")
        if "echo solved > /app/answer.txt" in command:
            self.files["/app/answer.txt"] = "solved"
            return "wrote /app/answer.txt"
        return f"mock executed: {command}"

    def _observation(self, *, last_stdout: str = "", last_stderr: str = "") -> Observation:
        return Observation(
            task_id=self.task_id,
            instruction=self.instruction,
            directory_summary="\n".join(sorted(self.files)) or "(empty)",
            recent_history=tuple(self.history[-4:]),
            last_stdout=last_stdout,
            last_stderr=last_stderr,
            selected_files=dict(self.files),
            steps_remaining=max(self.max_steps - self.step_count, 0),
        )


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
