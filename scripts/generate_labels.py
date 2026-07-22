#!/usr/bin/env python3
"""
Step 1: generate ProD-M labels with the real LLM.

Supports checkpointed runs so Colab disconnects don't wipe progress:

  # 1000 prompts in saves of 50
  python scripts/generate_labels.py --config configs/live_run.yaml \\
      --limit 1000 --chunk-size 50 --resume --num-samples 3 --device cuda

After each chunk we rewrite:
  data/processed/prod_labels.json
  data/processed/prod_hidden.pt

Use --backup-dir /content/drive/MyDrive/capstone_results to copy after every chunk.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from statistics import median

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import PromptRecord, load_labels, save_labels
from src.datasets import load_prompts
from src.llama import LlamaServer
from src.prod_m import load_hidden, save_hidden
from src.utils import get_hf_token, load_config, resolve_llm


def backup(paths, backup_dir):
    if not backup_dir:
        return
    os.makedirs(backup_dir, exist_ok=True)
    for path in paths:
        if os.path.exists(path):
            dest = os.path.join(backup_dir, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"  backup -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-profile", default=None)
    parser.add_argument("--output", default="data/processed/prod_labels.json")
    parser.add_argument("--hidden-output", default="data/processed/prod_hidden.pt")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="save progress every N prompts (default 50 = 20 checkpoints for limit=1000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip prompts already present in --output",
    )
    parser.add_argument(
        "--backup-dir",
        default="",
        help="optional folder (e.g. Drive) to copy labels/hidden after each chunk",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="override prod_m.num_samples (default from config is 3 at 1000-prompt scale)",
    )
    args = parser.parse_args()

    if not get_hf_token():
        print("ERROR: export HF_TOKEN=hf_... first")
        print("Also accept the license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        sys.exit(1)

    cfg = load_config(args.config)
    llm = resolve_llm(cfg, args.llm_profile)
    dataset = args.dataset or cfg["datasets"]["name"]
    limit = args.limit or cfg["datasets"]["limit"]
    r = args.num_samples or cfg["prod_m"]["num_samples"]
    chunk_size = max(1, args.chunk_size)

    print(f"LLM: {llm['profile']} -> {llm['model']}")
    print(f"Loading {limit} prompts from {dataset}...")
    records = load_prompts(
        dataset,
        limit=limit,
        per_dataset=cfg["datasets"].get("per_dataset"),
    )
    print(f"Got {len(records)} prompts")
    print(f"Checkpoint every {chunk_size} prompts "
          f"({(len(records) + chunk_size - 1) // chunk_size} chunks)")

    labeled = []
    hidden_list = []
    done_ids = set()

    if args.resume and os.path.exists(args.output):
        labeled, old_meta = load_labels(args.output)
        done_ids = {rec.prompt_id for rec in labeled}
        if os.path.exists(args.hidden_output):
            prev_h = load_hidden(args.hidden_output)
            # keep only as many rows as labels (safety)
            n = min(len(labeled), prev_h.shape[0])
            labeled = labeled[:n]
            done_ids = {rec.prompt_id for rec in labeled}
            hidden_list = [prev_h[:n]]
            print(f"Resuming: already have {n} labeled prompts")
        else:
            print("WARNING: labels exist but hidden file missing — regenerating hidden later")
            labeled = []
            done_ids = set()

    todo = [rec for rec in records if rec.prompt_id not in done_ids]
    if not todo:
        print("Nothing left to label — already complete.")
        return

    server = LlamaServer(
        llm["model"],
        device=args.device,
        load_in_4bit=llm["load_in_4bit"],
        max_prompt_tokens=llm["max_prompt_tokens"],
    )

    print(f"Sampling {r} times per prompt for {len(todo)} remaining...")
    chunk_new = []
    for i, rec in enumerate(todo):
        samples = server.generate_lengths(
            rec.text,
            num_samples=r,
            max_new_tokens=llm["max_new_tokens"],
            temperature=cfg["prod_m"]["temperature"],
            top_p=cfg["prod_m"]["top_p"],
        )
        med = int(median(samples))
        chunk_new.append(
            PromptRecord(
                prompt_id=rec.prompt_id,
                text=rec.text,
                output_length=med,
                priority=rec.priority,
                sample_lengths=samples,
                single_sample_length=samples[0],
            )
        )
        total_done = len(labeled) + len(chunk_new)
        if total_done % 5 == 0 or i + 1 == len(todo):
            print(f"  {total_done}/{len(records)}  (last median={med})")

        # end of chunk or last item
        if len(chunk_new) >= chunk_size or i + 1 == len(todo):
            print(f"\n--- checkpoint: encoding + saving {len(chunk_new)} new prompts ---")
            h = server.encode(
                [x.text for x in chunk_new],
                batch_size=cfg["prod_m"]["encode_batch_size"],
            )
            labeled.extend(chunk_new)
            hidden_list.append(h)
            hidden = torch.cat(hidden_list, dim=0)

            meta = {
                "dataset": dataset,
                "llm": llm["model"],
                "profile": llm["profile"],
                "num_samples": r,
                "total_prompts": len(labeled),
                "chunk_size": chunk_size,
            }
            save_hidden(args.hidden_output, hidden)
            save_labels(args.output, labeled, meta=meta, hidden_path=args.hidden_output)
            print(f"Saved {len(labeled)}/{len(records)} -> {args.output}")
            backup([args.output, args.hidden_output], args.backup_dir or None)

            # keep a single stacked tensor in memory for next concat
            hidden_list = [hidden]
            chunk_new = []

    print(f"\nDone. Labels: {args.output} ({len(labeled)} prompts)")
    print(f"Hidden: {args.hidden_output}")


if __name__ == "__main__":
    main()
