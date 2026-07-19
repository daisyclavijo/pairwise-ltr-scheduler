"""
Schedulers we compare in this project.

  fcfs   - First-Come-First-Serve (production default; suffers HOL blocking)
  ltr    - Main-paper style pointwise LTR: sort by predicted output length
           (implemented with our ProD-M length predictor)
  pars   - OUR improvement: pairwise ranking scores + priority + starvation
  oracle - Perfect SJF using true median length (upper bound)

`prod_m` is kept as an alias for `ltr` so older scripts still work.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from src.requests import Request

# length-aware policies all use the same min-heap ordering
LENGTH_AWARE = {"ltr", "prod_m", "pars", "oracle", "pairwise_ltr", "prod_m_pars"}


@dataclass(order=True)
class _Item:
    key: float
    req: Request = field(compare=False)


class Scheduler:
    def __init__(self, policy="fcfs", batch_size=8, starvation_sec=120.0, boosts=None):
        # normalize aliases
        if policy == "prod_m":
            policy = "ltr"
        if policy in ("pairwise_ltr", "prod_m_pars"):
            policy = "pars"

        self.policy = policy
        self.batch_size = batch_size
        self.starvation_sec = starvation_sec
        # lower effective score = served sooner; high priority subtracts
        self.boosts = boosts or {"high": -3.0, "normal": 0.0, "low": 3.0}
        self.waiting = []

    def add(self, req: Request):
        self.waiting.append(req)

    def _maybe_promote(self, now):
        # fairness: if a request waits too long, treat it as high priority
        # (same idea as PARS starvation prevention, ~2 minutes)
        for req in self.waiting:
            if now - req.arrival_time >= self.starvation_sec:
                req.priority = "high"

    def next_batch(self, now=0.0, n=None):
        """Pick up to n requests from the waiting queue."""
        self._maybe_promote(now)
        n = self.batch_size if n is None else max(0, n)
        if n == 0 or not self.waiting:
            return []

        if self.policy == "fcfs":
            batch = self.waiting[:n]
            self.waiting = self.waiting[n:]
            return batch

        # LTR / PARS / Oracle: shortest (lowest score) first, after priority boost
        heap = []
        for req in self.waiting:
            heapq.heappush(heap, _Item(key=req.effective_score(self.boosts), req=req))

        batch = []
        for _ in range(min(n, len(heap))):
            batch.append(heapq.heappop(heap).req)

        picked = {r.request_id for r in batch}
        self.waiting = [r for r in self.waiting if r.request_id not in picked]
        return batch
