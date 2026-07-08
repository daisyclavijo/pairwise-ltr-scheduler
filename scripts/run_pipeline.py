#!/usr/bin/env python3
"""
Full real pipeline: Llama 3.1 8B + GSM8K/MBPP/LMSYS/LongBench.

  export HF_TOKEN=hf_...
  python scripts/run_pipeline.py --device cuda

Requires a GPU with ~16GB VRAM (T4/A10 works with 4-bit Llama).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_prod_labels
from src.metrics import kendall_tau, mean_absolute_error
from src.pairwise_predictor import load_model as load_pars
from src.prod_m import load_hidden_states, load_prod_m
from src.request_gateway import IncomingPrompt, RequestGateway
from src.simulator import SimConfig, compare_policies


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def priority_map(cfg: dict) -> dict[str, float]:
    p = cfg["priority"]
    return {"high": p["high_boost"], "normal": p["normal_boost"], "low": p["low_boost"]}


def main():
    parser = argparse.ArgumentParser(description="Real ProD-M + PARS pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--prod-m", default="checkpoints/prod_m.pt")
    parser.add_argument("--pars", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--llm-profile", default=None, help="llama33|qwen25|deepseek_r1|llama31")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("ERROR: export HF_TOKEN before running the real pipeline.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.llm_profile:
        cfg.setdefault("llm", {})["profile"] = args.llm_profile

    limit = args.limit or cfg["datasets"]["limit"]
    dataset = args.dataset or cfg["datasets"]["name"]

    if not args.skip_train:
        run([
            sys.executable, "scripts/generate_prod_labels.py",
            "--dataset", dataset,
            "--limit", str(limit),
            "--output", args.labels,
            "--device", args.device,
            *(["--llm-profile", args.llm_profile] if args.llm_profile else []),
        ])
        run([
            sys.executable, "scripts/train_prod_m.py",
            "--labels", args.labels,
            "--output", args.prod_m,
            "--device", args.device,
        ])
        run([
            sys.executable, "scripts/train_predictor.py",
            "--labels", args.labels,
            "--train-samples", str(limit),
            "--output", args.pars,
            "--device", args.device,
        ])

    label_file = load_prod_labels(args.labels)
    records = label_file.records

    for i, rec in enumerate(records):
        if i % 8 == 0:
            rec.priority = "high"
        elif i % 5 == 0:
            rec.priority = "low"

    pars = load_pars(args.pars, device=args.device)
    prod_m = load_prod_m(args.prod_m, device=args.device)

    hidden_path = label_file.meta.get("hidden_states_path")
    if hidden_path and os.path.exists(hidden_path):
        hidden = load_hidden_states(hidden_path)
    else:
        print("WARNING: no cached hidden states found")
        hidden = torch.zeros(len(records), prod_m.mlp[0].in_features)

    pred_lengths = prod_m.predict_lengths(hidden.to(args.device))
    true_lengths = [r.output_length for r in records]
    print(f"\nProD-M MAE vs median: {mean_absolute_error(true_lengths, pred_lengths):.2f} tokens")

    pars_scores = pars.score_prompts([r.text for r in records])
    pred_order = [true_lengths[i] for i in sorted(range(len(pars_scores)), key=lambda k: pars_scores[k])]
    print(f"PARS Kendall Tau: {kendall_tau(pred_order, sorted(true_lengths)):.3f}")

    print("\n--- Priority demo ---")
    gateway = RequestGateway()
    boosts = priority_map(cfg)
    demo = [
        gateway.submit(IncomingPrompt("Write a long essay", priority="low"), rank_score=9.0),
        gateway.submit(IncomingPrompt("Yes or no?", priority="normal"), rank_score=2.0),
        gateway.submit(IncomingPrompt("Urgent summary", priority="high"), rank_score=5.0),
    ]
    order = sorted(demo, key=lambda r: r.effective_score(boosts))
    print("Serve order:", [r.request_id for r in order])

    eval_n = min(cfg["datasets"].get("eval_limit", 200), len(records))
    base = SimConfig(
        batch_size=cfg["scheduler"]["batch_size"],
        arrival_rate=cfg["simulation"]["arrival_rate"],
        seed=cfg["simulation"]["seed"],
        priority_boosts=boosts,
    )
    summaries = compare_policies(
        records[:eval_n], pars, ["fcfs", "prod_m", "prod_m_pars"], base,
        prod_m=prod_m, hidden_states=hidden[:eval_n], device=args.device,
    )
    for s in summaries:
        print(f"\n=== {s.policy.upper()} ===")
        print(f"  P95 latency: {s.p95_latency:.3f}s  |  Throughput: {s.throughput_rps:.2f} req/s")


if __name__ == "__main__":
    main()
