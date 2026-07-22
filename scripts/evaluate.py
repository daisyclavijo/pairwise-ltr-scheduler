#!/usr/bin/env python3
"""
Compare exactly three schedulers (proposal evaluation plan):

  1) FCFS                          - baseline
  2) LTR                           - MAIN PAPER (pointwise, single-sample labels)
  3) PARS + ProD-M + Priority      - OURS
       - ProD-M: median-of-r labels (not in main paper)
       - PARS: pairwise ranking
       - Priority + starvation in the scheduler
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels
from src.metrics import kendall_tau, mae, ndcg_at_k, pairwise_accuracy
from src.prod_m import load_hidden, load_prod_m
from src.ranker import load_ranker
from src.simulate import SimConfig, compare
from src.utils import load_config

POLICY_TITLE = {
    "fcfs": "FCFS (baseline)",
    "ltr": "LTR scheduler (MAIN PAPER)",
    "pars": "PARS + ProD-M + Priority (OURS)",
}


def print_summary(s):
    title = POLICY_TITLE.get(s.policy, s.policy.upper())
    print(f"\n=== {title} ===")
    print(f"  policy id:   {s.policy}")
    print(f"  requests:    {s.num_requests}")
    print(f"  avg latency: {s.avg_latency:.3f}s")
    print(f"  p50:         {s.p50_latency:.3f}s")
    print(f"  p95:         {s.p95_latency:.3f}s")
    print(f"  p99:         {s.p99_latency:.3f}s")
    print(f"  avg wait:    {s.avg_wait:.3f}s")
    print(f"  avg TTFT:    {s.avg_ttft:.3f}s")
    print(f"  throughput:  {s.throughput_rps:.2f} req/s")


def _pct_gain(base, other):
    if base <= 0:
        return None
    return 100.0 * (base - other) / base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ltr", default="checkpoints/ltr_pointwise.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found. Run generate_labels.py first.")
        sys.exit(1)

    cfg = load_config(args.config)
    records, meta = load_labels(args.labels)
    limit = args.limit or cfg["datasets"].get("eval_limit", len(records))
    records = records[:limit]

    # Priority is part of OURS (PARS + ProD-M + Priority)
    n_high = n_low = 0
    for i, rec in enumerate(records):
        if i % 8 == 0:
            rec.priority = "high"
            n_high += 1
        elif i % 5 == 0:
            rec.priority = "low"
            n_low += 1
    print(
        f"Priority mix (used by OURS): high={n_high}, low={n_low}, "
        f"normal={len(records) - n_high - n_low}"
    )

    ltr_model = load_prod_m(args.ltr, device=args.device) if os.path.exists(args.ltr) else None
    ranker = load_ranker(args.ranker, device=args.device) if os.path.exists(args.ranker) else None

    hidden = None
    hidden_path = meta.get("hidden_states_path", "data/processed/prod_hidden.pt")
    if os.path.exists(hidden_path):
        hidden = load_hidden(hidden_path)[: len(records)]

    true = [r.output_length for r in records]

    print("\n--- 1) Model quality ---")
    print("OURS uses ProD-M median labels to train PARS (ProD-M is not in the main paper).")
    if ltr_model is not None and hidden is not None:
        print(
            "LTR MAE vs median: "
            f"{mae(true, ltr_model.predict_lengths(hidden.to(args.device))):.2f} tokens"
        )
    else:
        print("LTR checkpoint missing — run train_prod_m.py --target single")

    if ranker is not None:
        scores = ranker.score([r.text for r in records])
        order = [true[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
        print(f"OURS (PARS) Kendall Tau:       {kendall_tau(order, sorted(true)):.3f}")
        print(f"OURS (PARS) Pairwise Accuracy: {pairwise_accuracy(scores, true):.3f}")
        print(f"OURS (PARS) NDCG:              {ndcg_at_k(scores, true):.3f}")
    else:
        print("PARS checkpoint missing")

    boosts = {
        "high": cfg["priority"]["high_boost"],
        "normal": cfg["priority"]["normal_boost"],
        "low": cfg["priority"]["low_boost"],
    }
    # Main-paper LTR: no priority boosts (fair comparison to paper setting)
    ltr_boosts = {"high": 0.0, "normal": 0.0, "low": 0.0}

    print("\n--- 2) Scheduler comparison (3 methods only) ---")
    print("1. FCFS                         = baseline")
    print("2. LTR                          = MAIN PAPER")
    print("3. PARS + ProD-M + Priority     = OURS")

    # Run FCFS and LTR without priority; OURS with priority boosts
    summaries = []
    for policy, b in (("fcfs", ltr_boosts), ("ltr", ltr_boosts), ("pars", boosts)):
        cfg_one = SimConfig(
            policy=policy,
            batch_size=cfg["scheduler"]["batch_size"],
            arrival_rate=cfg["simulation"]["arrival_rate"],
            seed=cfg["simulation"]["seed"],
            boosts=b,
        )
        summaries.extend(
            compare(
                records,
                [policy],
                cfg_one,
                ranker=ranker,
                ltr_model=ltr_model,
                prod_m=None,
                hidden=hidden,
                device=args.device,
            )
        )

    for s in summaries:
        print_summary(s)

    by_name = {s.policy: s for s in summaries}

    def show(label, a, b):
        if a in by_name and b in by_name:
            g = _pct_gain(by_name[a].p95_latency, by_name[b].p95_latency)
            if g is not None:
                print(f"{label}: {g:.1f}%")

    print("\n--- p95 latency improvements ---")
    show("LTR vs FCFS", "fcfs", "ltr")
    show("OURS vs LTR (main paper)", "ltr", "pars")
    show("OURS vs FCFS", "fcfs", "pars")

    # Save machine-readable summary for plot_results / report
    out_json = "data/processed/eval_summary.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "summaries": {
            s.policy: {
                "avg_latency": s.avg_latency,
                "p50_latency": s.p50_latency,
                "p95_latency": s.p95_latency,
                "p99_latency": s.p99_latency,
                "avg_wait": s.avg_wait,
                "avg_ttft": s.avg_ttft,
                "throughput_rps": s.throughput_rps,
                "num_requests": s.num_requests,
            }
            for s in summaries
        },
        "p95_improvements_pct": {},
    }
    for label, a, b in (
        ("LTR vs FCFS", "fcfs", "ltr"),
        ("OURS vs LTR (main paper)", "ltr", "pars"),
        ("OURS vs FCFS", "fcfs", "pars"),
    ):
        if a in by_name and b in by_name:
            g = _pct_gain(by_name[a].p95_latency, by_name[b].p95_latency)
            if g is not None:
                payload["p95_improvements_pct"][label] = g
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_json}")
    print("For report graphs: python scripts/plot_results.py --device cuda")


if __name__ == "__main__":
    main()
