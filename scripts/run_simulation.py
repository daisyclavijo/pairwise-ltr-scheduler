#!/usr/bin/env python3
"""
Evaluate schedulers on real ProD-M labels from Llama 3.1 8B.

Run after the pipeline:
  python scripts/run_simulation.py --compare-all --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_prod_labels
from src.metrics import kendall_tau
from src.pairwise_predictor import load_model
from src.prod_m import load_hidden_states, load_prod_m
from src.simulator import SimConfig, compare_policies, run_simulation


def priority_map(cfg: dict) -> dict[str, float]:
    p = cfg["priority"]
    return {"high": p["high_boost"], "normal": p["normal_boost"], "low": p["low_boost"]}


def print_summary(summary):
    print(f"\n=== {summary.policy.upper()} ===")
    print(f"  Requests:     {summary.num_requests}")
    print(f"  Avg latency:  {summary.avg_latency:.3f}s")
    print(f"  P50 latency:  {summary.p50_latency:.3f}s")
    print(f"  P95 latency:  {summary.p95_latency:.3f}s")
    print(f"  P99 latency:  {summary.p99_latency:.3f}s")
    print(f"  Avg wait:     {summary.avg_wait:.3f}s")
    print(f"  Throughput:   {summary.throughput_rps:.2f} req/s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on real ProD-M labels")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--prod-m", default="checkpoints/prod_m.pt")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found. Run generate_prod_labels.py first.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_req = args.num_requests or cfg["datasets"].get("eval_limit", 200)
    records = load_prod_labels(args.labels).records[:num_req]

    for i, rec in enumerate(records):
        if i % 10 == 0:
            rec.priority = "high"
        elif i % 7 == 0:
            rec.priority = "low"

    ranker = None
    if os.path.exists(args.checkpoint):
        print(f"Loading PARS from {args.checkpoint}")
        ranker = load_model(args.checkpoint, device=args.device)

    prod_m = None
    hidden = None
    if os.path.exists(args.prod_m):
        print(f"Loading ProD-M from {args.prod_m}")
        prod_m = load_prod_m(args.prod_m, device=args.device)
        hidden_path = load_prod_labels(args.labels).meta.get("hidden_states_path")
        if hidden_path and os.path.exists(hidden_path):
            hidden = load_hidden_states(hidden_path)[:num_req]
        else:
            print("WARNING: hidden states not cached — ProD-M eval may be inaccurate")

    boosts = priority_map(cfg)
    base_config = SimConfig(
        policy=cfg["scheduler"]["policy"],
        batch_size=cfg["scheduler"]["batch_size"],
        arrival_rate=cfg["simulation"]["arrival_rate"],
        seed=cfg["simulation"]["seed"],
        priority_boosts=boosts,
    )

    if args.compare_all:
        policies = ["fcfs", "prod_m", "prod_m_pars"]
        summaries = compare_policies(
            records, ranker, policies, base_config,
            prod_m=prod_m, hidden_states=hidden, device=args.device,
        )
        for s in summaries:
            print_summary(s)

        if ranker:
            lengths = [r.output_length for r in records]
            scores = ranker.score_prompts([r.text for r in records])
            pred_order = [lengths[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
            print(f"\nPARS Kendall Tau: {kendall_tau(pred_order, sorted(lengths)):.3f}")
        return

    policy = args.policy or cfg["scheduler"]["policy"]
    config = SimConfig(
        policy=policy,
        batch_size=base_config.batch_size,
        arrival_rate=base_config.arrival_rate,
        seed=base_config.seed,
        priority_boosts=boosts,
    )
    use_pars = policy in ("prod_m_pars", "pairwise_ltr", "pars")
    use_prod = policy in ("prod_m", "ltr_pointwise")
    _, summary = run_simulation(
        records, config,
        ranker=ranker if use_pars else None,
        prod_m=prod_m if use_prod else None,
        hidden_states=hidden if use_prod else None,
        device=args.device,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
