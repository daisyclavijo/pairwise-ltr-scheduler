#!/usr/bin/env python3
"""Step 3: train the pairwise ranker (PARS-style) on ProD-M median pairs."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_pairs, load_labels
from src.ranker import PairwiseRanker, load_ranker, ranking_loss, save_ranker
from src.utils import load_config


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, y = self.pairs[idx]
        return a, b, y


def collate(batch):
    a, b, y = zip(*batch)
    return list(a), list(b), torch.tensor(y, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--output", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="load --output if it already exists and keep training from there "
        "(use after a disconnect instead of restarting from scratch)",
    )
    parser.add_argument(
        "--ablation-single-sample",
        action="store_true",
        help="train on single-sample lengths instead of ProD-M medians "
        "(ablation from the midterm slides)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    epochs = args.epochs or cfg["training"]["epochs"]

    records, _ = load_labels(args.labels)
    if args.train_samples:
        records = records[: args.train_samples]

    pairs = build_pairs(
        records,
        min_diff=cfg["model"]["min_length_diff"],
        max_pairs=cfg["training"].get("max_pairs", 5000),
        use_single_sample=args.ablation_single_sample,
    )
    label_kind = "single-sample" if args.ablation_single_sample else "median"
    print(f"Built {len(pairs)} training pairs from {len(records)} prompts ({label_kind} labels)")
    if not pairs:
        print("No pairs — try lowering min_length_diff or using more prompts")
        sys.exit(1)

    loader = DataLoader(
        PairDataset(pairs),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )

    model = PairwiseRanker(cfg["model"]["backbone"]).to(args.device)
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        model = load_ranker(args.output, device=args.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for a, b, y in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            y = y.to(args.device)
            loss = ranking_loss(
                model, a, b, y,
                margin=cfg["model"]["margin"],
                max_length=cfg["training"]["max_prompt_length"],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        avg_loss = total / max(n, 1)
        print(f"epoch {epoch + 1} loss={avg_loss:.4f}")

        # Save after every epoch, not just at the end, so a disconnect
        # mid-training only costs the current epoch, not the whole run.
        save_ranker(model, args.output)
        print(f"  checkpoint saved -> {args.output}")

    print(f"Done. Final checkpoint at {args.output}")


if __name__ == "__main__":
    main()