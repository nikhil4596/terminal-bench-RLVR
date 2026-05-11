from __future__ import annotations

from collections.abc import Iterable

from ..rollout import RolloutRecord


def records_to_training_samples(
    records: Iterable[RolloutRecord],
) -> list[dict[str, object]]:
    """Convert rollout records to backend-neutral sample rows."""

    return [record.to_training_sample() for record in records]


def records_to_grpo_samples(records: Iterable[RolloutRecord]) -> list[dict[str, object]]:
    """Backward-compatible alias for older docs/tests."""

    return records_to_training_samples(records)


def terminal_records(records: Iterable[RolloutRecord]) -> list[RolloutRecord]:
    """Keep only terminal records when training on episode-level rewards."""

    return [record for record in records if record.done]
