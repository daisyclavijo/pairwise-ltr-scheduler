#!/usr/bin/env python3
"""
Live three-way comparison on a REAL GPU using HuggingFace Transformers.

Use this when vLLM will not import on Colab (common). Still live generation —
not the discrete-event simulator — just without vLLM's continuous batcher.

  1) FCFS
  2) LTR          (main paper)
  3) PARS+priority (ours)

Prompts are ordered by each policy, then generated in micro-batches on Llama.
We report total wall time plus mean / p95 completion time (order matters).

Example (Colab A100, after labels + checkpoints on Drive):
  python scripts/evaluate_live_hf.py --config configs/live_run.yaml \\
      --limit 1000 --batch-size 4 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels
from src.llama import LlamaServer
from src.live_serve import (
    LivePolicyResult,
    assign_eval_priorities,
    effective_scores,
    free_cuda,
    parse_priorities_from_records,
    print_live_result,
    results_to_payload,
    save_live_results,
)
from src.prod_m import load_hidden, load_prod_m
from src.ranker import load_ranker
from src.utils import load_config, resolve_llm


POLICY_TITLES = {
    "fcfs": "FCFS (baseline) — live HF GPU",
    "ltr": "LTR scheduler (MAIN PAPER) — live HF GPU",
    "pars": "PARS + ProD-M + Priority (OURS) — live HF GPU",
}


def _percentile(vals, q):
    if not vals:
        return 0.0
    xs = sorted(vals)
    idx = min(len(xs) - 1, max(0, int(round((q / 100.0) * (len(xs) - 1)))))
    return float(xs[idx])


def _score_ltr(ltr_model, hidden, device):
    ltr_model.to(device)
    return [float(x) for x in ltr_model.predict_lengths(hidden.to(device))]


def _score_pars(ranker, texts, device, max_length):
    ranker.to(device)
    scores = []
    bs = 32
    for i in range(0, len(texts), bs):
        scores.extend(ranker.score(texts[i : i + bs], max_length=max_length))
    return [float(s) for s in scores]


def _order_from_scores(scores):
    """Lower score served first (same convention as vLLM priorities)."""
    return sorted(range(len(scores)), key=lambda i: scores[i])


@torch.no_grad()
def run_hf_policy(server, texts, order, max_new_tokens, temperature, top_p, batch_size, policy_name):
    """Generate in schedule order; record per-request completion times."""
    n = len(texts)
    finish = [0.0] * n
    out_lens = [0] * n
    start = time.perf_counter()

    for bi in range(0, len(order), batch_size):
        idxs = order[bi : bi + batch_size]
        batch_texts = [server._format(texts[i]) for i in idxs]
        batch = server.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=server.max_prompt_tokens,
            return_tensors="pt",
        )
        batch = server._to_device(batch)
        prompt_lens = batch["attention_mask"].sum(dim=1).tolist()

        out = server.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=server.tokenizer.eos_token_id,
        )
        now = time.perf_counter() - start
        for row, idx, plen in zip(out, idxs, prompt_lens):
            out_lens[idx] = int(row.shape[0] - int(plen))
            finish[idx] = now

        done = min(bi + batch_size, n)
        if done % max(batch_size, 20) == 0 or done == n:
            print(f"  {policy_name}: {done}/{n}  elapsed={now:.1f}s", flush=True)

    wall = time.perf_counter() - start
    total_tok = int(sum(out_lens))
    result = LivePolicyResult(
        policy=policy_name,
        num_requests=n,
        wall_time_s=wall,
        throughput_rps=(n / wall) if wall > 0 else 0.0,
        avg_output_tokens=(total_tok / n) if n else 0.0,
        total_output_tokens=total_tok,
        notes=(
            f"HF live micro-batches={batch_size}; "
            f"mean_completion={sum(finish)/n:.2f}s; "
            f"p95_completion={_percentile(finish, 95):.2f}s"
        ),
    )
    extras = {
        "mean_completion_s": (sum(finish) / n) if n else 0.0,
        "p50_completion_s": _percentile(finish, 50),
        "p95_completion_s": _percentile(finish, 95),
    }
    return result, extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ltr", default="checkpoints/ltr_pointwise.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--output",
        default="data/processed/live_eval_results.json",
    )
    parser.add_argument("--policies", default="fcfs,ltr,pars")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found. Run generate_labels.py first.")
        sys.exit(1)

    cfg = load_config(args.config)
    llm_cfg = resolve_llm(cfg)
    records, meta = load_labels(args.labels)
    limit = args.limit or cfg["datasets"].get("eval_limit", len(records))
    records = records[:limit]

    n_high, n_low = assign_eval_priorities(records)
    print(
        f"HF live eval on {len(records)} prompts | "
        f"high={n_high}, low={n_low}, normal={len(records) - n_high - n_low} | "
        f"batch_size={args.batch_size}"
    )

    texts = [r.text for r in records]
    biz_pri = parse_priorities_from_records(records)
    boosts = {
        "high": cfg["priority"]["high_boost"],
        "normal": cfg["priority"]["normal_boost"],
        "low": cfg["priority"]["low_boost"],
    }
    want = {p.strip() for p in args.policies.split(",") if p.strip()}
    max_tokens = args.max_tokens or llm_cfg.get("max_new_tokens", 512)

    # score small models first
    ltr_scores = pars_scores = None
    if "ltr" in want:
        hidden = load_hidden(meta.get("hidden_states_path", "data/processed/prod_hidden.pt"))[
            : len(records)
        ]
        ltr_model = load_prod_m(args.ltr, device=args.device)
        ltr_scores = _score_ltr(ltr_model, hidden, args.device)
        del ltr_model
        free_cuda()
    if "pars" in want:
        ranker = load_ranker(args.ranker, device=args.device)
        pars_scores = _score_pars(
            ranker, texts, args.device, cfg["training"]["max_prompt_length"]
        )
        del ranker
        free_cuda()

    orders = {}
    if "fcfs" in want:
        orders["fcfs"] = list(range(len(texts)))
    if "ltr" in want and ltr_scores is not None:
        orders["ltr"] = _order_from_scores(
            effective_scores(ltr_scores, biz_pri, boosts, use_priority=False)
        )
    if "pars" in want and pars_scores is not None:
        orders["pars"] = _order_from_scores(
            effective_scores(pars_scores, biz_pri, boosts, use_priority=True)
        )

    model_name = meta.get("llm") or llm_cfg["model"]
    print(f"Loading HF Llama for live generation: {model_name}")
    server = LlamaServer(
        model_name,
        device=args.device,
        load_in_4bit=llm_cfg["load_in_4bit"],
        max_prompt_tokens=llm_cfg["max_prompt_tokens"],
    )

    results = []
    extras_by = {}
    for policy in ("fcfs", "ltr", "pars"):
        if policy not in orders:
            continue
        print(f"\n----- {policy.upper()} (HF live) -----", flush=True)
        result, extras = run_hf_policy(
            server,
            texts,
            orders[policy],
            max_new_tokens=max_tokens,
            temperature=cfg["prod_m"]["temperature"],
            top_p=cfg["prod_m"]["top_p"],
            batch_size=max(1, args.batch_size),
            policy_name=policy,
        )
        print_live_result(result, POLICY_TITLES[policy])
        print(
            f"  mean completion:  {extras['mean_completion_s']:.2f}s\n"
            f"  p50 completion:   {extras['p50_completion_s']:.2f}s\n"
            f"  p95 completion:   {extras['p95_completion_s']:.2f}s"
        )
        results.append(result)
        extras_by[policy] = extras

    print("\n--- live completion-time improvements (mean) ---")
    by = {r.policy: extras_by[r.policy]["mean_completion_s"] for r in results}

    def show(label, a, b):
        if a in by and b in by and by[a] > 0:
            g = 100.0 * (by[a] - by[b]) / by[a]
            print(f"{label}: {g:.1f}%")

    show("LTR vs FCFS", "fcfs", "ltr")
    show("OURS vs LTR (main paper)", "ltr", "pars")
    show("OURS vs FCFS", "fcfs", "pars")

    payload = results_to_payload(
        {
            "mode": "live_hf",
            "model": model_name,
            "num_prompts": len(records),
            "batch_size": args.batch_size,
            "max_tokens": max_tokens,
            "completion_extras": extras_by,
        },
        results,
    )
    save_live_results(args.output, payload)

    print("\n" + "=" * 60)
    print(" LIVE HF RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        e = extras_by[r.policy]
        print(
            f"  {r.policy:5s}  wall={r.wall_time_s:8.2f}s  "
            f"mean_lat={e['mean_completion_s']:7.2f}s  "
            f"p95={e['p95_completion_s']:7.2f}s"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
