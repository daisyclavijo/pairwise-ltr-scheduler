#!/usr/bin/env python3
"""
Live three-way comparison on a real vLLM engine:

  1) FCFS                         - baseline
  2) LTR                          - MAIN PAPER (pointwise predicted length)
  3) PARS + ProD-M + Priority     - OURS

Unlike scripts/evaluate.py (discrete-event simulator), this measures real
GPU wall time / throughput while vLLM schedules with scheduling_policy=priority.

Typical high-prompt flow:
  # 1) Label more prompts (chunked + Drive backup)
  python scripts/generate_labels.py --config configs/live_run.yaml \\
      --limit 1000 --chunk-size 50 --resume --num-samples 3 --device cuda \\
      --backup-dir /content/drive/MyDrive/capstone_results

  # 2) Train
  python scripts/train_prod_m.py --config configs/live_run.yaml \\
      --target single --output checkpoints/ltr_pointwise.pt --device cuda
  python scripts/train_ranker.py --config configs/live_run.yaml \\
      --train-samples 1000 --device cuda

  # 3) Live serve
  python scripts/evaluate_live.py --config configs/live_run.yaml \\
      --limit 1000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels
from src.live_serve import (
    assign_eval_priorities,
    build_llm,
    effective_scores,
    free_cuda,
    parse_priorities_from_records,
    print_live_result,
    results_to_payload,
    run_vllm_policy,
    save_live_results,
    scores_to_vllm_priorities,
)
from src.prod_m import load_hidden, load_prod_m
from src.ranker import load_ranker
from src.utils import load_config, resolve_llm


POLICY_TITLES = {
    "fcfs": "FCFS (baseline) — live vLLM",
    "ltr": "LTR scheduler (MAIN PAPER) — live vLLM",
    "pars": "PARS + ProD-M + Priority (OURS) — live vLLM",
}


def _score_ltr(ltr_model, hidden, device):
    ltr_model.to(device)
    lengths = ltr_model.predict_lengths(hidden.to(device))
    return [float(x) for x in lengths]


def _score_pars(ranker, texts, device, max_length):
    ranker.to(device)
    # score in chunks to avoid OOM on large prompt sets
    scores = []
    bs = 32
    for i in range(0, len(texts), bs):
        scores.extend(ranker.score(texts[i : i + bs], max_length=max_length))
    return [float(s) for s in scores]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ltr", default="checkpoints/ltr_pointwise.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--output",
        default="data/processed/live_eval_results.json",
        help="where to write JSON results",
    )
    parser.add_argument(
        "--policies",
        default="fcfs,ltr,pars",
        help="comma list: fcfs,ltr,pars",
    )
    args = parser.parse_args()

    try:
        from vllm import SamplingParams
    except ImportError:
        print("ERROR: vLLM is not installed / not importable in this runtime.")
        print("Colab fix (dependency warnings about numba/cuml are usually harmless):")
        print("  1) Runtime → Change runtime type → GPU")
        print("  2) !python scripts/ensure_vllm.py --install")
        print("  3) If still failing: Runtime → Restart session, restore from Drive,")
        print("     re-run ensure_vllm.py --install, then this script again")
        print("Simulator fallback (not live engine):")
        print("  python scripts/evaluate.py --limit 1000 --device cuda")
        print("Recommended on Colab when vLLM import fails (still LIVE GPU):")
        print("  python scripts/evaluate_live_hf.py --config configs/live_run.yaml --limit 1000 --device cuda")
        sys.exit(1)

    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found. Run generate_labels.py first.")
        sys.exit(1)

    cfg = load_config(args.config)
    llm_cfg = resolve_llm(cfg)
    records, meta = load_labels(args.labels)
    limit = args.limit or cfg["datasets"].get("eval_limit", len(records))
    records = records[:limit]
    if not records:
        print("ERROR: no labeled prompts to serve")
        sys.exit(1)

    n_high, n_low = assign_eval_priorities(records)
    print(
        f"Live eval on {len(records)} prompts | "
        f"priority mix: high={n_high}, low={n_low}, "
        f"normal={len(records) - n_high - n_low}"
    )

    texts = [r.text for r in records]
    priority_labels = parse_priorities_from_records(records)
    boosts = {
        "high": cfg["priority"]["high_boost"],
        "normal": cfg["priority"]["normal_boost"],
        "low": cfg["priority"]["low_boost"],
    }

    want = {p.strip() for p in args.policies.split(",") if p.strip()}
    ltr_scores = None
    pars_scores = None

    # --- score with small models first, then free GPU for vLLM ---
    if "ltr" in want:
        if not os.path.exists(args.ltr):
            print(f"ERROR: LTR checkpoint missing: {args.ltr}")
            sys.exit(1)
        hidden_path = meta.get("hidden_states_path", "data/processed/prod_hidden.pt")
        if not os.path.exists(hidden_path):
            print(f"ERROR: hidden states missing: {hidden_path}")
            sys.exit(1)
        hidden = load_hidden(hidden_path)[: len(records)]
        if hidden.shape[0] < len(records):
            print(
                f"ERROR: only {hidden.shape[0]} hidden rows for {len(records)} prompts. "
                "Re-run labeling so labels and hiddens stay in sync."
            )
            sys.exit(1)
        print("Scoring LTR (pointwise predicted lengths)...")
        ltr_model = load_prod_m(args.ltr, device=args.device)
        ltr_scores = _score_ltr(ltr_model, hidden, args.device)
        del ltr_model
        free_cuda()

    if "pars" in want:
        if not os.path.exists(args.ranker):
            print(f"ERROR: PARS checkpoint missing: {args.ranker}")
            sys.exit(1)
        print("Scoring PARS (pairwise ranker)...")
        ranker = load_ranker(args.ranker, device=args.device)
        pars_scores = _score_pars(
            ranker,
            texts,
            args.device,
            max_length=cfg["training"]["max_prompt_length"],
        )
        del ranker
        free_cuda()

    # Build vLLM priority ints per policy
    policy_priorities = {}
    if "fcfs" in want:
        policy_priorities["fcfs"] = None  # no priority arg => arrival order
    if "ltr" in want and ltr_scores is not None:
        # main paper: length only, no business priority boosts
        policy_priorities["ltr"] = scores_to_vllm_priorities(
            effective_scores(ltr_scores, priority_labels, boosts, use_priority=False)
        )
    if "pars" in want and pars_scores is not None:
        policy_priorities["pars"] = scores_to_vllm_priorities(
            effective_scores(pars_scores, priority_labels, boosts, use_priority=True)
        )

    model_name = meta.get("llm") or llm_cfg["model"]
    max_tokens = args.max_tokens or llm_cfg.get("max_new_tokens", 512)
    print(f"\nLoading vLLM: {model_name}")
    print(f"  scheduling_policy=priority | max_tokens={max_tokens} | n={len(texts)}")
    free_cuda()
    llm = build_llm(
        model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        temperature=cfg["prod_m"]["temperature"],
        top_p=cfg["prod_m"]["top_p"],
        max_tokens=max_tokens,
    )

    results = []
    for policy in ("fcfs", "ltr", "pars"):
        if policy not in policy_priorities:
            continue
        print(f"\n----- running {policy.upper()} on live engine -----")
        result, _ = run_vllm_policy(
            llm,
            texts,
            sampling,
            priorities=policy_priorities[policy],
            policy_name=policy,
        )
        print_live_result(result, POLICY_TITLES[policy])
        results.append(result)

    print("\n--- live wall-time improvements ---")
    by = {r.policy: r for r in results}

    def show(label, a, b):
        if a in by and b in by and by[a].wall_time_s > 0:
            g = 100.0 * (by[a].wall_time_s - by[b].wall_time_s) / by[a].wall_time_s
            print(f"{label}: {g:.1f}%")

    show("LTR vs FCFS", "fcfs", "ltr")
    show("OURS vs LTR (main paper)", "ltr", "pars")
    show("OURS vs FCFS", "fcfs", "pars")

    payload = results_to_payload(
        {
            "mode": "live_vllm",
            "model": model_name,
            "num_prompts": len(records),
            "max_tokens": max_tokens,
            "labels": args.labels,
            "ltr": args.ltr,
            "ranker": args.ranker,
            "priority_boosts": boosts,
        },
        results,
    )
    save_live_results(args.output, payload)

    # also print a compact terminal summary block
    print("\n" + "=" * 60)
    print(" LIVE RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"  {r.policy:5s}  wall={r.wall_time_s:8.2f}s  "
            f"tput={r.throughput_rps:6.2f} req/s  "
            f"avg_tok={r.avg_output_tokens:.1f}"
        )
    gains = payload.get("wall_time_improvements_pct") or {}
    for k, v in gains.items():
        if v is not None:
            print(f"  {k}: {v:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
