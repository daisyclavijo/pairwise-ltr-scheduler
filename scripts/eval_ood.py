#!/usr/bin/env python3
"""
In-distribution vs out-of-distribution evaluation (midterm Phase 2).

Idea from the ProD paper / our slides:
  train the length predictor on math-style prompts, then check whether
  MAE gets worse on chat / long-context / coding prompts.

Usage (after generate_labels.py has finished):
  python scripts/eval_ood.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_labels, split_id_ood
from src.metrics import kendall_tau, mae, pairwise_accuracy
from src.prod_m import ProDMPredictor, length_to_bin, load_hidden, make_bins
from src.ranker import PairwiseRanker, ranking_loss
from src.utils import load_config


def _indices_for(records, all_records):
    id_map = {r.prompt_id: i for i, r in enumerate(all_records)}
    return [id_map[r.prompt_id] for r in records]


def train_prod_m_on(hidden, records, cfg, device):
    bin_edges = make_bins(cfg["prod_m"]["max_length"], cfg["prod_m"]["num_bins"])
    y = torch.tensor(
        [length_to_bin(r.output_length, bin_edges) for r in records],
        dtype=torch.long,
    )
    loader = DataLoader(
        TensorDataset(hidden, y),
        batch_size=cfg["prod_m"]["batch_size"],
        shuffle=True,
    )
    model = ProDMPredictor(hidden.shape[1], cfg["prod_m"]["num_bins"], bin_edges).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["prod_m"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(cfg["prod_m"]["epochs"]):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            loss = loss_fn(model(feats), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_ranker_on(records, cfg, device):
    from src.data import build_pairs

    pairs = build_pairs(
        records,
        min_diff=cfg["model"]["min_length_diff"],
        max_pairs=cfg["training"].get("max_pairs", 5000),
    )
    if len(pairs) < 10:
        return None

    model = PairwiseRanker(cfg["model"]["backbone"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    # keep this short — OOD script is for the gap, not a full retrain
    epochs = min(3, cfg["training"]["epochs"])

    model.train()
    for _ in range(epochs):
        for i in range(0, len(pairs), cfg["training"]["batch_size"]):
            batch = pairs[i : i + cfg["training"]["batch_size"]]
            a, b, y = zip(*batch)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            loss = ranking_loss(
                model, list(a), list(b), y,
                margin=cfg["model"]["margin"],
                max_length=cfg["training"]["max_prompt_length"],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/live_run.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"ERROR: {args.labels} not found")
        sys.exit(1)

    cfg = load_config(args.config)
    records, meta = load_labels(args.labels)
    id_recs, ood_recs = split_id_ood(records)

    print(f"Total prompts: {len(records)}")
    print(f"ID  (gsm8k/math):              {len(id_recs)}")
    print(f"OOD (chat/code/longbench):     {len(ood_recs)}")

    if len(id_recs) < 8 or len(ood_recs) < 8:
        print("Need more mixed-dataset labels for a meaningful ID/OOD split.")
        print("Re-run: python scripts/generate_labels.py --config configs/live_run.yaml --dataset all --limit 1000")
        sys.exit(1)

    hidden_path = meta.get("hidden_states_path", "data/processed/prod_hidden.pt")
    if not os.path.exists(hidden_path):
        print(f"ERROR: missing {hidden_path}")
        sys.exit(1)

    all_hidden = load_hidden(hidden_path)
    id_idx = _indices_for(id_recs, records)
    ood_idx = _indices_for(ood_recs, records)
    id_h = all_hidden[id_idx]
    ood_h = all_hidden[ood_idx]

    print("\nTraining ProD-M on ID only...")
    model = train_prod_m_on(id_h, id_recs, cfg, args.device)

    id_pred = model.predict_lengths(id_h.to(args.device))
    ood_pred = model.predict_lengths(ood_h.to(args.device))
    id_mae = mae([r.output_length for r in id_recs], id_pred)
    ood_mae = mae([r.output_length for r in ood_recs], ood_pred)

    print("\n=== ProD-M ID vs OOD ===")
    print(f"  ID  MAE:  {id_mae:.2f} tokens")
    print(f"  OOD MAE:  {ood_mae:.2f} tokens")
    print(f"  OOD gap:  {ood_mae - id_mae:+.2f} tokens  (positive = worse on OOD)")

    print("\nTraining a small PARS ranker on ID pairs...")
    ranker = train_ranker_on(id_recs, cfg, args.device)
    if ranker is None:
        print("Not enough ID pairs for ranker OOD check.")
        return

    def rank_stats(recs):
        lengths = [r.output_length for r in recs]
        scores = ranker.score([r.text for r in recs])
        order = [lengths[i] for i in sorted(range(len(scores)), key=lambda k: scores[k])]
        return kendall_tau(order, sorted(lengths)), pairwise_accuracy(scores, lengths)

    id_tau, id_acc = rank_stats(id_recs)
    ood_tau, ood_acc = rank_stats(ood_recs)
    print("\n=== PARS ID vs OOD ===")
    print(f"  ID  Kendall Tau: {id_tau:.3f} | pairwise acc: {id_acc:.3f}")
    print(f"  OOD Kendall Tau: {ood_tau:.3f} | pairwise acc: {ood_acc:.3f}")


if __name__ == "__main__":
    main()
