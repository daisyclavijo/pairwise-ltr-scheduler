"""
Priority prompt handling.

Two ideas combined here:
1. Prompt-aware ranking score from the pairwise predictor (PARS style)
2. User-assigned priority labels (high / normal / low)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class PriorityLevel(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class InferenceRequest:
    """A request waiting in the scheduler queue."""

    request_id: str
    prompt: str
    output_length: int          # ground truth for simulation / eval
    priority: PriorityLevel = PriorityLevel.NORMAL
    arrival_time: float = field(default_factory=time.time)
    rank_score: float = 0.0     # filled by ProD-M or PARS (higher = longer job)
    predicted_length: int = 0   # ProD-M pointwise estimate
    metadata: dict = field(default_factory=dict)

    def effective_score(self, boosts: dict[str, float]) -> float:
        """
        Final scheduling score. Lower value = served sooner.

        rank_score estimates expected length.
        priority boost nudges urgent prompts ahead.
        """
        boost = boosts.get(self.priority.value, 0.0)
        return self.rank_score + boost


def parse_priority(value: str) -> PriorityLevel:
    """Turn a string into a priority level."""
    value = value.lower().strip()
    if value in ("high", "urgent", "p1"):
        return PriorityLevel.HIGH
    if value in ("low", "background", "p3"):
        return PriorityLevel.LOW
    return PriorityLevel.NORMAL
