"""
Discrete-event simulator for LLM request scheduling.

Runs on CPU — good for comparing policies before cloud GPU tests.
Each token takes a fixed decode time so we can see HOL blocking effects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.data import PromptRecord, stream_requests
from src.metrics import RequestMetrics, RunSummary, summarize
from src.pairwise_predictor import PairwiseRanker
from src.prod_m import ProDMPredictor
from src.priority import InferenceRequest, PriorityLevel, parse_priority
from src.scheduler import BaseScheduler, make_scheduler


# Rough seconds per generated token in simulation
DECODE_TIME_PER_TOKEN = 0.01
PREFILL_TIME = 0.05


@dataclass
class SimConfig:
    policy: str = "pairwise_ltr"
    batch_size: int = 8
    arrival_rate: float = 5.0
    seed: int = 42
    priority_boosts: dict[str, float] | None = None


def records_to_requests(
    records: list[PromptRecord],
    ranker: PairwiseRanker | None = None,
    prod_m: ProDMPredictor | None = None,
    hidden_states=None,
    device: str = "cpu",
) -> list[InferenceRequest]:
    """Build scheduler requests and score with PARS or ProD-M."""
    requests = []

    for rec in records:
        req = InferenceRequest(
            request_id=rec.prompt_id,
            prompt=rec.text,
            output_length=rec.output_length,
            priority=parse_priority(rec.priority),
            arrival_time=0.0,
        )
        requests.append(req)

    if ranker is not None:
        ranker.to(device)
        scores = ranker.score_prompts([r.prompt for r in requests])
        for req, score in zip(requests, scores):
            req.rank_score = score

    if prod_m is not None and hidden_states is not None:
        prod_m.to(device)
        lengths = prod_m.predict_lengths(hidden_states.to(device))
        for req, length in zip(requests, lengths):
            req.rank_score = float(length)
            req.predicted_length = int(round(length))

    return requests


def run_simulation(
    records: list[PromptRecord],
    config: SimConfig,
    ranker: PairwiseRanker | None = None,
    prod_m: ProDMPredictor | None = None,
    hidden_states=None,
    device: str = "cpu",
) -> tuple[list[RequestMetrics], RunSummary]:
    """
    Simulate requests arriving over time and being scheduled.

    Returns per-request metrics and an aggregate summary.
    """
    rng = random.Random(config.seed)
    scheduler: BaseScheduler = make_scheduler(
        config.policy,
        batch_size=config.batch_size,
        priority_boosts=config.priority_boosts,
    )

    # Pre-score all prompts once (predictor cost is amortized)
    all_requests = records_to_requests(
        records, ranker=ranker, prod_m=prod_m, hidden_states=hidden_states, device=device
    )
    request_map = {r.request_id: r for r in all_requests}

    clock = 0.0
    pending_arrivals = list(stream_requests(records, config.arrival_rate, config.seed))
    arrival_idx = 0
    completed: list[RequestMetrics] = []

    # Track active jobs: (finish_time, request)
    active: list[tuple[float, InferenceRequest]] = []

    while arrival_idx < len(pending_arrivals) or active or scheduler.waiting:
        # Bring in new arrivals
        while arrival_idx < len(pending_arrivals):
            arrive_t, rec = pending_arrivals[arrival_idx]
            if arrive_t > clock:
                break
            req = request_map[rec.prompt_id]
            req.arrival_time = arrive_t
            scheduler.add_request(req)
            arrival_idx += 1

        # Finish any jobs done at this clock tick
        still_active = []
        for finish_t, req in active:
            if finish_t <= clock:
                wait = max(0.0, clock - req.arrival_time - _service_time(req))
                service = _service_time(req)
                completed.append(
                    RequestMetrics(
                        request_id=req.request_id,
                        wait_time=wait,
                        service_time=service,
                        total_latency=clock - req.arrival_time,
                        output_length=req.output_length,
                        priority=req.priority.value,
                    )
                )
            else:
                still_active.append((finish_t, req))
        active = still_active

        # Start new batch if we have capacity
        slots = config.batch_size - len(active)
        if slots > 0 and scheduler.waiting:
            batch = scheduler.pick_next_batch(now=clock)[:slots]
            for req in batch:
                duration = _service_time(req)
                active.append((clock + duration, req))

        # Advance time to next event
        next_arrival = pending_arrivals[arrival_idx][0] if arrival_idx < len(pending_arrivals) else float("inf")
        next_finish = min((t for t, _ in active), default=float("inf"))
        next_event = min(next_arrival, next_finish)

        if next_event == float("inf"):
            break

        clock = next_event

    summary = summarize(config.policy, completed, total_time=clock)
    return completed, summary


def _service_time(req: InferenceRequest) -> float:
    """Prefill + decode time based on output length."""
    return PREFILL_TIME + req.output_length * DECODE_TIME_PER_TOKEN


def compare_policies(
    records: list[PromptRecord],
    ranker: PairwiseRanker | None,
    policies: list[str],
    config: SimConfig,
    prod_m: ProDMPredictor | None = None,
    hidden_states=None,
    device: str = "cpu",
) -> list[RunSummary]:
    """Run the same workload under multiple schedulers."""
    results = []
    for policy in policies:
        cfg = SimConfig(
            policy=policy,
            batch_size=config.batch_size,
            arrival_rate=config.arrival_rate,
            seed=config.seed,
            priority_boosts=config.priority_boosts,
        )
        use_pars = policy in ("pairwise_ltr", "prod_m_pars", "pars")
        use_prod = policy in ("ltr_pointwise", "prod_m")
        _, summary = run_simulation(
            records,
            cfg,
            ranker=ranker if use_pars else None,
            prod_m=prod_m if use_prod else None,
            hidden_states=hidden_states if use_prod else None,
            device=device,
        )
        results.append(summary)
    return results
