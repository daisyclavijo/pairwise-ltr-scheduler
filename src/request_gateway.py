"""
Incoming request gateway with priority support.

Every prompt that enters the system passes through here.
Users set priority when submitting; the scheduler uses it later.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from src.priority import InferenceRequest, PriorityLevel, parse_priority


@dataclass
class IncomingPrompt:
  """What a client sends to the serving system."""

  prompt: str
  priority: str = "normal"       # high | normal | low
  request_id: str | None = None
  metadata: dict = field(default_factory=dict)


class RequestGateway:
  """
  Accepts incoming prompts and turns them into scheduler requests.

  Example:
    gateway = RequestGateway()
    req = gateway.submit(IncomingPrompt("Summarize this.", priority="high"))
  """

  def __init__(self):
    self._counter = 0

  def submit(
    self,
    incoming: IncomingPrompt,
    output_length: int = 0,
    rank_score: float = 0.0,
  ) -> InferenceRequest:
    self._counter += 1
    req_id = incoming.request_id or f"req_{self._counter}_{uuid.uuid4().hex[:6]}"

    return InferenceRequest(
      request_id=req_id,
      prompt=incoming.prompt.strip(),
      output_length=output_length,
      priority=parse_priority(incoming.priority),
      arrival_time=time.time(),
      rank_score=rank_score,
      metadata=incoming.metadata,
    )

  def submit_batch(self, items: list[IncomingPrompt], **kwargs) -> list[InferenceRequest]:
    return [self.submit(item, **kwargs) for item in items]
