"""
Metrics for comparing FCFS vs LTR vs Pairwise LTR.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    request_id: str
    wait_time: float
    service_time: float
    total_latency: float
    output_length: int
    priority: str


@dataclass
class RunSummary:
    policy: str
    num_requests: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    avg_wait: float
    throughput_rps: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def summarize(policy: str, rows: list[RequestMetrics], total_time: float) -> RunSummary:
    latencies = [r.total_latency for r in rows]
    waits = [r.wait_time for r in rows]

    return RunSummary(
        policy=policy,
        num_requests=len(rows),
        avg_latency=statistics.mean(latencies) if latencies else 0.0,
        p50_latency=percentile(latencies, 50),
        p95_latency=percentile(latencies, 95),
        p99_latency=percentile(latencies, 99),
        avg_wait=statistics.mean(waits) if waits else 0.0,
        throughput_rps=len(rows) / total_time if total_time > 0 else 0.0,
    )


def mean_absolute_error(true_lengths: list[float], pred_lengths: list[float]) -> float:
    if not true_lengths:
        return 0.0
    return sum(abs(t - p) for t, p in zip(true_lengths, pred_lengths)) / len(true_lengths)


def kendall_tau(predicted_order: list[int], true_order: list[int]) -> float:
    """
    Ranking quality metric from the LTR papers.

    predicted_order / true_order = list of output lengths sorted by each method.
    """
    n = len(predicted_order)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            pred_sign = predicted_order[i] - predicted_order[j]
            true_sign = true_order[i] - true_order[j]
            if pred_sign == 0 or true_sign == 0:
                continue
            if pred_sign * true_sign > 0:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total
