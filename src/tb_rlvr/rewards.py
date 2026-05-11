from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    success: float = 1.0
    progress: float = 1.0
    integrity: float = 1.0
    step: float = 1.0
    token: float = 1.0


@dataclass(frozen=True)
class RewardComponents:
    success: float = 0.0
    progress: float = 0.0
    integrity: float = 0.0
    step: float = 0.0
    token: float = 0.0
    total: float = 0.0


def combine_rewards(
    *,
    success: bool,
    progress_delta: float,
    integrity_violation: bool,
    step_count: int,
    generated_tokens: int = 0,
    progress_cap: float = 0.20,
    weights: RewardWeights | None = None,
) -> RewardComponents:
    """Combine auditable reward components for a Terminal-Bench-style task."""

    weights = weights or RewardWeights()

    success_reward = 1.0 if success else 0.0
    progress_reward = max(0.0, min(progress_delta, progress_cap))
    integrity_reward = -1.0 if integrity_violation else 0.0
    step_reward = -0.01 * max(step_count, 0)
    token_reward = -0.00001 * max(generated_tokens, 0)

    total = (
        weights.success * success_reward
        + weights.progress * progress_reward
        + weights.integrity * integrity_reward
        + weights.step * step_reward
        + weights.token * token_reward
    )

    return RewardComponents(
        success=success_reward,
        progress=progress_reward,
        integrity=integrity_reward,
        step=step_reward,
        token=token_reward,
        total=total,
    )

