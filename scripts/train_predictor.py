#!/usr/bin/env python3
"""
Train the pairwise ranker on cloud GPU.

Example (Google Colab / RunPod):
  python scripts/train_predictor.py --epochs 3 --train-samples 2000
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_prod_labels, make_pairwise_samples, make_single_sample_pairs
from src.pairwise_predictor import PairwiseRanker, margin_ranking_step, save_model


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        return a, b, label


def collate_batch(batch):
    prompts_a, prompts_b, labels = zip(*batch)
    return list(prompts_a), list(prompts_b), torch.tensor(labels, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train pairwise LTR predictor")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--output", default="checkpoints/pairwise_ranker.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--labels", default="data/processed/prod_labels.json",
                        help="ProD-M median labels (run generate_prod_labels.py first)")
    parser.add_argument("--ablation-single-sample", action="store_true",
                        help="Train on single-sample labels instead of medians")
    parser.add_argument("--max-pairs", type=int, default=5000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    margin = cfg["model"]["margin"]
    min_diff = cfg["model"]["min_length_diff"]
    max_len = cfg["training"]["max_prompt_length"]
    backbone = cfg["model"]["backbone"]

    print(f"Device: {args.device}")

    # Phase 3: PARS pairs from ProD-M median labels
    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found.")
        print("Run first: python scripts/generate_prod_labels.py --device cuda")
        sys.exit(1)

    print(f"Loading ProD-M labels from {args.labels}")
    records = load_prod_labels(args.labels).records
    if args.train_samples:
        records = records[: args.train_samples]

    if args.ablation_single_sample:
        print("Ablation: using single-sample labels (noisy)")
        pairs = make_single_sample_pairs(records, min_length_diff=min_diff)
    else:
        print("Using ProD-M median labels for pairwise pairs")
        pairs = make_pairwise_samples(records, min_length_diff=min_diff)
    if len(pairs) > args.max_pairs:
        random.shuffle(pairs)
        pairs = pairs[: args.max_pairs]
    print(f"Built {len(pairs)} training pairs")

    if len(pairs) == 0:
        print("No pairs found. Try lowering min_length_diff or adding more data.")
        return

    dataset = PairDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    model = PairwiseRanker(backbone=backbone).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for prompts_a, prompts_b, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            labels = labels.to(args.device)
            loss = margin_ranking_step(
                model, prompts_a, prompts_b, labels, margin=margin, max_length=max_len
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(steps, 1)
        print(f"Epoch {epoch + 1} — avg loss: {avg:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_model(model, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
