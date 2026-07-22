#!/usr/bin/env python3
"""
Build report-ready RESULTS section (tables + graphs like the main paper).

Produces under --out-dir (default data/processed/figures/):
  fig_length_distributions.png   # paper Fig.2 style
  fig_latency_vs_rate.png        # paper Fig.3 style (avg latency)
  fig_p95_vs_rate.png
  fig_latency_bars.png
  fig_throughput_bars.png
  fig_improvements.png
  fig_ranking_quality.png
  results_summary.json

Also prints a markdown-friendly results table to the terminal.

Example (Colab, after labels + checkpoints exist):
  python scripts/plot_results.py --config configs/live_run.yaml \\
      --limit 1000 --device cuda \\
      --out-dir /content/drive/MyDrive/capstone_results/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels
from src.metrics import kendall_tau, mae, ndcg_at_k, pairwise_accuracy
from src.plots import (
    plot_improvement_bars,
    plot_latency_vs_rate,
    plot_length_distributions,
    plot_policy_bars,
    plot_ranking_quality,
    print_results_table,
)
from src.prod_m import load_hidden, load_prod_m
from src.ranker import load_ranker
from src.simulate import SimConfig, compare
from src.utils import load_config, resolve_llm


def _pct_gain(base, other):
    if base is None or other is None or base <= 0:
        return None
    return 100.0 * (base - other) / base


def _summary_to_dict(s):
    return {
        "policy": s.policy,
        "num_requests": s.num_requests,
        "avg_latency": s.avg_latency,
        "p50_latency": s.p50_latency,
        "p95_latency": s.p95_latency,
        "p99_latency": s.p99_latency,
        "avg_wait": s.avg_wait,
        "avg_ttft": s.avg_ttft,
        "throughput_rps": s.throughput_rps,
    }


def _estimate_input_lens(records, model_name, token):
    """Tokenize prompts once for Fig.2-style input length histogram."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, token=token)
        lens = []
        for r in records:
            ids = tok(r.text, add_special_tokens=False)["input_ids"]
            lens.append(len(ids))
        return lens
    except Exception as e:
        print(f"[warn] tokenizer failed ({e}); using whitespace token estimate")
        return [max(1, len(r.text.split())) for r in records]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ltr", default="checkpoints/ltr_pointwise.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rates",
        default="5,10,15,20,30,40,50,60",
        help="comma-separated arrival rates for Fig.3-style sweep",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/figures",
        help="where to write PNGs + results_summary.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found")
        sys.exit(1)

    cfg = load_config(args.config)
    llm = resolve_llm(cfg)
    records, meta = load_labels(args.labels)
    limit = args.limit or cfg["datasets"].get("eval_limit", len(records))
    records = records[:limit]
    print(f"Plotting results for {len(records)} labeled prompts")

    # Priority mix (same as evaluate.py)
    for i, rec in enumerate(records):
        if i % 8 == 0:
            rec.priority = "high"
        elif i % 5 == 0:
            rec.priority = "low"
        else:
            rec.priority = "normal"

    boosts = {
        "high": cfg["priority"]["high_boost"],
        "normal": cfg["priority"]["normal_boost"],
        "low": cfg["priority"]["low_boost"],
    }
    no_boost = {"high": 0.0, "normal": 0.0, "low": 0.0}

    ltr_model = load_prod_m(args.ltr, device=args.device) if os.path.exists(args.ltr) else None
    ranker = load_ranker(args.ranker, device=args.device) if os.path.exists(args.ranker) else None
    hidden = None
    hidden_path = meta.get("hidden_states_path", "data/processed/prod_hidden.pt")
    if os.path.exists(hidden_path):
        hidden = load_hidden(hidden_path)[: len(records)]

    # --- ranking quality (for report text) ---
    ranking = {}
    true = [r.output_length for r in records]
    if ltr_model is not None and hidden is not None:
        ranking["ltr_mae"] = mae(true, ltr_model.predict_lengths(hidden.to(args.device)))
        print(f"LTR MAE vs median: {ranking['ltr_mae']:.2f} tokens")
    if ranker is not None:
        scores = ranker.score([r.text for r in records])
        order = [true[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
        ranking["kendall"] = kendall_tau(order, sorted(true))
        ranking["pairwise_acc"] = pairwise_accuracy(scores, true)
        ranking["ndcg"] = ndcg_at_k(scores, true)
        print(
            f"PARS Kendall={ranking['kendall']:.3f}  "
            f"pairwise={ranking['pairwise_acc']:.3f}  "
            f"NDCG={ranking['ndcg']:.3f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Fig 2 style: length distributions ---
    print("\n[1/4] Length distributions…")
    from src.utils import get_hf_token

    model_name = meta.get("llm") or llm["model"]
    input_lens = _estimate_input_lens(records, model_name, get_hf_token())
    output_lens = [int(r.output_length) for r in records]
    plot_length_distributions(
        input_lens,
        output_lens,
        args.out_dir,
        title_suffix=f" (n={len(records)})",
    )

    # --- default-rate comparison bars ---
    print("\n[2/4] Default-rate scheduler comparison…")
    base_rate = cfg["simulation"]["arrival_rate"]
    summaries = {}
    for policy, b in (("fcfs", no_boost), ("ltr", no_boost), ("pars", boosts)):
        cfg_one = SimConfig(
            policy=policy,
            batch_size=cfg["scheduler"]["batch_size"],
            arrival_rate=base_rate,
            seed=cfg["simulation"]["seed"],
            boosts=b,
        )
        s = compare(
            records,
            [policy],
            cfg_one,
            ranker=ranker,
            ltr_model=ltr_model,
            hidden=hidden,
            device=args.device,
        )[0]
        summaries[policy] = _summary_to_dict(s)
        print(
            f"  {policy}: avg={s.avg_latency:.3f}s  p95={s.p95_latency:.3f}s  "
            f"tput={s.throughput_rps:.2f}"
        )

    gains = {}
    if "fcfs" in summaries and "ltr" in summaries:
        gains["LTR vs FCFS"] = _pct_gain(
            summaries["fcfs"]["p95_latency"], summaries["ltr"]["p95_latency"]
        )
    if "ltr" in summaries and "pars" in summaries:
        gains["OURS vs LTR (main paper)"] = _pct_gain(
            summaries["ltr"]["p95_latency"], summaries["pars"]["p95_latency"]
        )
    if "fcfs" in summaries and "pars" in summaries:
        gains["OURS vs FCFS"] = _pct_gain(
            summaries["fcfs"]["p95_latency"], summaries["pars"]["p95_latency"]
        )
    gains = {k: v for k, v in gains.items() if v is not None}

    plot_policy_bars(summaries, args.out_dir)
    plot_improvement_bars(gains, args.out_dir)
    plot_ranking_quality(ranking, args.out_dir)
    print_results_table(summaries, gains)

    # --- Fig 3 style: latency vs request rate ---
    print("\n[3/4] Latency vs request-rate sweep…")
    rates = [float(x) for x in args.rates.split(",") if x.strip()]
    avg_rows, p95_rows = [], []
    for rate in rates:
        row_avg = {"rate": rate}
        row_p95 = {"rate": rate}
        for policy, b in (("fcfs", no_boost), ("ltr", no_boost), ("pars", boosts)):
            cfg_one = SimConfig(
                policy=policy,
                batch_size=cfg["scheduler"]["batch_size"],
                arrival_rate=rate,
                seed=cfg["simulation"]["seed"],
                boosts=b,
            )
            s = compare(
                records,
                [policy],
                cfg_one,
                ranker=ranker,
                ltr_model=ltr_model,
                hidden=hidden,
                device=args.device,
            )[0]
            row_avg[policy] = s.avg_latency
            row_p95[policy] = s.p95_latency
        avg_rows.append(row_avg)
        p95_rows.append(row_p95)
        print(
            f"  rate={rate:4.0f}  "
            f"FCFS={row_avg['fcfs']:.3f}  LTR={row_avg['ltr']:.3f}  "
            f"OURS={row_avg['pars']:.3f}"
        )

    plot_latency_vs_rate(avg_rows, args.out_dir, metric="avg_latency")
    plot_latency_vs_rate(p95_rows, args.out_dir, metric="p95_latency")

    # --- save JSON for report ---
    print("\n[4/4] Writing results_summary.json…")
    payload = {
        "num_prompts": len(records),
        "model": model_name,
        "default_arrival_rate": base_rate,
        "summaries": summaries,
        "p95_improvements_pct": gains,
        "ranking_quality": ranking,
        "latency_vs_rate_avg": avg_rows,
        "latency_vs_rate_p95": p95_rows,
        "input_length_mean": float(sum(input_lens) / len(input_lens)),
        "output_length_mean": float(sum(output_lens) / len(output_lens)),
        "figures_dir": args.out_dir,
    }
    summary_path = os.path.join(args.out_dir, "results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved {summary_path}")

    # also drop a short markdown snippet for the report
    md_path = os.path.join(args.out_dir, "results_section.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Results\n\n")
        f.write(f"Evaluation on **{len(records)}** prompts (`{model_name}`).\n\n")
        f.write("## Scheduler comparison\n\n")
        f.write("| Policy | Avg latency (s) | p95 (s) | Throughput (req/s) |\n")
        f.write("|--------|----------------:|--------:|-------------------:|\n")
        for key, label in (
            ("fcfs", "FCFS (baseline)"),
            ("ltr", "LTR (main paper)"),
            ("pars", "PARS+ProD-M+Priority (ours)"),
        ):
            s = summaries[key]
            f.write(
                f"| {label} | {s['avg_latency']:.3f} | {s['p95_latency']:.3f} | "
                f"{s['throughput_rps']:.2f} |\n"
            )
        f.write("\n## p95 latency improvements\n\n")
        for k, v in gains.items():
            f.write(f"- **{k}**: {v:.1f}%\n")
        f.write("\n## Figures\n\n")
        f.write("- `fig_length_distributions.png` — input/output length distributions (paper Fig. 2 style)\n")
        f.write("- `fig_latency_vs_rate.png` — latency vs request rate (paper Fig. 3 style)\n")
        f.write("- `fig_latency_bars.png` / `fig_throughput_bars.png` — three-way comparison\n")
        f.write("- `fig_improvements.png` — relative p95 gains\n")
    print(f"  saved {md_path}")
    print("\nDone. Copy the PNGs from the figures folder into your report.")


if __name__ == "__main__":
    main()
