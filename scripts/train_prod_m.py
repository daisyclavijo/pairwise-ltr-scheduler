#!/usr/bin/env python3
"""
Train a pointwise length predictor (2-layer MLP on Llama hidden states).

Two training modes (important distinction for the report):

  default / --target median
      ProD-M (OUR addition, from the ProD paper — NOT in the main paper)
      Train on median-of-r labels.

  --target single
      Main-paper style LTR supervision: one noisy sample per prompt.
      Saved separately (e.g. checkpoints/ltr_pointwise.pt).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels
from src.metrics import mae
from src.prod_m import (
    ProDMPredictor,
    length_to_bin,
    load_hidden,
    make_bins,
    save_prod_m,
)
from src.utils import load_config, resolve_llm


def target_length(rec, target):
    if target == "single":
        return rec.single_sample_length or (
            rec.sample_lengths[0] if rec.sample_lengths else rec.output_length
        )
    return rec.output_length  # median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--output", default="checkpoints/prod_m.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--target",
        choices=("median", "single"),
        default="median",
        help="median = ProD-M (ours); single = main-paper style LTR labels",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm = resolve_llm(cfg)
    epochs = args.epochs or cfg["prod_m"]["epochs"]

    records, meta = load_labels(args.labels)
    print(f"Loaded {len(records)} labeled prompts")
    print(f"Training target: {args.target} "
          f"({'ProD-M / ours' if args.target == 'median' else 'main-paper style LTR'})")

    bin_edges = make_bins(cfg["prod_m"]["max_length"], cfg["prod_m"]["num_bins"])
    y = torch.tensor(
        [length_to_bin(target_length(r, args.target), bin_edges) for r in records],
        dtype=torch.long,
    )

    hidden_path = meta.get("hidden_states_path", "data/processed/prod_hidden.pt")
    if not os.path.exists(hidden_path):
        print(f"ERROR: missing hidden states at {hidden_path}")
        sys.exit(1)
    x = load_hidden(hidden_path)
    print(f"Hidden states: {tuple(x.shape)}")

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=cfg["prod_m"]["batch_size"],
        shuffle=True,
    )
    model = ProDMPredictor(x.shape[1], cfg["prod_m"]["num_bins"], bin_edges).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["prod_m"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for feats, labels in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            feats, labels = feats.to(args.device), labels.to(args.device)
            loss = loss_fn(model(feats), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"epoch {epoch + 1} loss={total / max(n, 1):.4f}")

    preds = model.predict_lengths(x.to(args.device))
    # always report MAE vs median target (stable eval target from ProD-M paper)
    true_median = [r.output_length for r in records]
    err = mae(true_median, preds)
    print(f"MAE vs median lengths: {err:.2f} tokens")

    save_prod_m(
        model,
        args.output,
        meta={"llm": llm["model"], "mae_vs_median": err, "train_target": args.target},
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
