#!/usr/bin/env python3
"""
Ablation: median labels (ProD-M) vs single-sample labels.

Midterm slides call this out explicitly — we want to show that the
gain comes from the supervision target (median), not just the model.

Trains two small rankers and compares Kendall Tau / pairwise accuracy.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_pairs, load_labels
from src.metrics import kendall_tau, pairwise_accuracy
from src.ranker import PairwiseRanker, ranking_loss
from src.utils import load_config


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch):
    a, b, y = zip(*batch)
    return list(a), list(b), torch.tensor(y, dtype=torch.float32)


def train_once(pairs, cfg, device, epochs):
    model = PairwiseRanker(cfg["model"]["backbone"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    loader = DataLoader(
        PairDataset(pairs),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    model.train()
    for epoch in range(epochs):
        for a, b, y in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            y = y.to(device)
            loss = ranking_loss(
                model, a, b, y,
                margin=cfg["model"]["margin"],
                max_length=cfg["training"]["max_prompt_length"],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def eval_ranker(model, records):
    lengths = [r.output_length for r in records]
    scores = model.score([r.text for r in records])
    order = [lengths[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
    return {
        "kendall": kendall_tau(order, sorted(lengths)),
        "pairwise_acc": pairwise_accuracy(scores, lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    records, _ = load_labels(args.labels)
    if args.limit:
        records = records[: args.limit]

    median_pairs = build_pairs(
        records,
        min_diff=cfg["model"]["min_length_diff"],
        max_pairs=cfg["training"].get("max_pairs", 5000),
        use_single_sample=False,
    )
    single_pairs = build_pairs(
        records,
        min_diff=cfg["model"]["min_length_diff"],
        max_pairs=cfg["training"].get("max_pairs", 5000),
        use_single_sample=True,
    )
    print(f"median pairs: {len(median_pairs)} | single-sample pairs: {len(single_pairs)}")

    print("\nTraining on MEDIAN labels...")
    median_model = train_once(median_pairs, cfg, args.device, args.epochs)
    med = eval_ranker(median_model, records)

    print("Training on SINGLE-SAMPLE labels...")
    single_model = train_once(single_pairs, cfg, args.device, args.epochs)
    sin = eval_ranker(single_model, records)

    print("\n=== Ablation: median vs single-sample ===")
    print(f"  median        Kendall={med['kendall']:.3f}  pairwise_acc={med['pairwise_acc']:.3f}")
    print(f"  single-sample Kendall={sin['kendall']:.3f}  pairwise_acc={sin['pairwise_acc']:.3f}")
    print(f"  delta Kendall: {med['kendall'] - sin['kendall']:+.3f}  (positive => median helps)")


if __name__ == "__main__":
    main()
