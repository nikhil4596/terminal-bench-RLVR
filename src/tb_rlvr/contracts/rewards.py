from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    correctness: float = 0.80
    efficiency: float = 0.20
    integrity: float = 1.0


@dataclass(frozen=True)
class RewardComponents:
    correctness: float = 0.0
    efficiency: float = 0.0
    integrity: float = 0.0
    success: float = 0.0
    total: float = 0.0


def combine_rewards(
    *,
    correctness: float,
    integrity_violation: bool,
    turn_count: int,
    max_turns: int = 20,
    generated_tokens: int = 0,
    max_tokens: int = 32768,
    weights: RewardWeights | None = None,
) -> RewardComponents:
    """Combine verifier-backed reward components.

    Correctness is the primary verifier score, usually the per-test fraction
    from `/logs/verifier/reward.json`. Efficiency is gated by correctness so a
    failed short trajectory cannot earn reward merely for stopping early.
    """

    weights = weights or RewardWeights()
    bounded_correctness = _clamp01(correctness)
    integrity_reward = -1.0 if integrity_violation else 0.0

    turn_efficiency = 1.0 - min(max(turn_count, 0), max_turns) / max(max_turns, 1)
    token_efficiency = 1.0 - min(max(generated_tokens, 0), max_tokens) / max(
        max_tokens, 1
    )
    efficiency_reward = bounded_correctness * (
        0.70 * turn_efficiency + 0.30 * token_efficiency
    )

    total = (
        weights.correctness * bounded_correctness
        + weights.efficiency * efficiency_reward
        + weights.integrity * integrity_reward
    )

    return RewardComponents(
        correctness=bounded_correctness,
        efficiency=efficiency_reward,
        integrity=integrity_reward,
        success=1.0 if bounded_correctness >= 1.0 and not integrity_violation else 0.0,
        total=total,
    )


def reward_from_verifier(
    rewards: Mapping[str, float | int],
    *,
    integrity_violation: bool,
    turn_count: int,
    max_turns: int = 20,
    generated_tokens: int = 0,
    weights: RewardWeights | None = None,
) -> RewardComponents:
    """Build reward components from Harbor/RewardKit verifier output."""

    if "correctness" in rewards:
        correctness = float(rewards["correctness"])
    elif "reward" in rewards:
        correctness = float(rewards["reward"])
    else:
        correctness = 0.0

    return combine_rewards(
        correctness=correctness,
        integrity_violation=integrity_violation,
        turn_count=turn_count,
        max_turns=max_turns,
        generated_tokens=generated_tokens,
        weights=weights,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
