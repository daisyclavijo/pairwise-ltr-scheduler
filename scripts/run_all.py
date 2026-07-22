#!/usr/bin/env python3
"""
Full pipeline — three-way comparison only:

  FCFS  |  LTR (main paper)  |  PARS + ProD-M + Priority (ours)

Steps:
  1) Llama x r -> median labels (ProD-M supervision for ours)
  2) train LTR on single-sample labels (main paper style)
  3) train PARS on median pairs (ours)
  4) evaluate the three schedulers

Example:
  python scripts/run_all.py --config configs/live_run.yaml --limit 1000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_hf_token, load_config, resolve_llm


def run(cmd):
    print("\n>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-profile", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--ltr", default="checkpoints/ltr_pointwise.pt")
    parser.add_argument("--ranker", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="save label progress every N prompts (default 50 for 1000-prompt runs)",
    )
    parser.add_argument(
        "--backup-dir",
        default="",
        help="copy labels/hidden here after each chunk (e.g. Drive path)",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="only run chunked label generation (then train later)",
    )
    args = parser.parse_args()

    if not get_hf_token() and not args.skip_train:
        print("ERROR: set HF_TOKEN before running")
        sys.exit(1)

    cfg = load_config(args.config)
    llm = resolve_llm(cfg, args.llm_profile)
    dataset = args.dataset or cfg["datasets"]["name"]
    limit = args.limit or cfg["datasets"]["limit"]

    print(f"Using {llm['profile']} -> {llm['model']}")
    print(f"Dataset={dataset}, limit={limit}, device={args.device}")
    print("Compare: FCFS | LTR (main paper) | PARS+ProD-M+Priority (ours)")
    if not args.skip_train:
        print(f"Label checkpoints: every {args.chunk_size} prompts (--resume safe)")

    py = sys.executable

    if not args.skip_train:
        label_cmd = [
            py, "scripts/generate_labels.py",
            "--dataset", dataset,
            "--limit", str(limit),
            "--output", args.labels,
            "--device", args.device,
            "--llm-profile", llm["profile"],
            "--chunk-size", str(args.chunk_size),
            "--resume",
        ]
        if args.backup_dir:
            label_cmd += ["--backup-dir", args.backup_dir]
        run(label_cmd)

        if args.labels_only:
            print("\nLabels done. Next:")
            print("  python scripts/run_all.py --config configs/live_run.yaml --skip-train --limit 1000 --device cuda")
            print("or train manually, then evaluate / plot_results.")
            return

        run([
            py, "scripts/train_prod_m.py",
            "--config", args.config,
            "--labels", args.labels,
            "--target", "single",
            "--output", args.ltr,
            "--device", args.device,
        ])
        run([
            py, "scripts/train_ranker.py",
            "--config", args.config,
            "--labels", args.labels,
            "--train-samples", str(limit),
            "--output", args.ranker,
            "--device", args.device,
        ])

    run([
        py, "scripts/evaluate.py",
        "--config", args.config,
        "--labels", args.labels,
        "--ltr", args.ltr,
        "--ranker", args.ranker,
        "--device", args.device,
        "--limit", str(limit),
    ])


if __name__ == "__main__":
    main()
