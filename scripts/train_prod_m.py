#!/usr/bin/env python3
"""
Phase 2: train ProD-M on real Llama hidden states + median labels.

Example:
  python scripts/train_prod_m.py --labels data/processed/prod_labels.json --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_prod_labels
from src.llm_config import resolve_llm
from src.metrics import mean_absolute_error
from src.prod_m import (
    LlamaServer,
    ProDMPredictor,
    length_to_bin,
    load_hidden_states,
    make_length_bins,
    save_prod_m,
)


class ProDDataset(Dataset):
    def __init__(self, hidden_states, bin_labels):
        self.hidden_states = hidden_states
        self.bin_labels = bin_labels

    def __len__(self):
        return len(self.bin_labels)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.bin_labels[idx]


def main():
    parser = argparse.ArgumentParser(description="Train ProD-M on real Llama features")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--labels", default="data/processed/prod_labels.json")
    parser.add_argument("--output", default="checkpoints/prod_m.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["prod_m"]["epochs"]
    batch_size = cfg["prod_m"]["batch_size"]
    lr = cfg["prod_m"]["learning_rate"]
    num_bins = cfg["prod_m"]["num_bins"]
    max_length = cfg["prod_m"]["max_length"]
    llm_cfg = resolve_llm(cfg)
    llm_model = llm_cfg["model"]

    label_file = load_prod_labels(args.labels)
    records = label_file.records
    print(f"Loaded {len(records)} ProD-M labeled prompts")

    bin_edges = make_length_bins(max_length, num_bins)
    bin_labels = torch.tensor(
        [length_to_bin(r.output_length, bin_edges) for r in records],
        dtype=torch.long,
    )

    hidden_path = label_file.meta.get("hidden_states_path")
    if hidden_path and os.path.exists(hidden_path):
        print(f"Loading cached hidden states from {hidden_path}")
        hidden_states = load_hidden_states(hidden_path)
    else:
        print(f"No cached hidden states — extracting from {llm_model}...")
        llm_server = LlamaServer(
            llm_model,
            device=args.device,
            load_in_4bit=llm_cfg["load_in_4bit"],
            max_prompt_tokens=llm_cfg["max_prompt_tokens"],
        )
        hidden_states = llm_server.encode(
            [r.text for r in records],
            batch_size=cfg["prod_m"]["encode_batch_size"],
        )

    hidden_dim = hidden_states.shape[1]
    dataset = ProDDataset(hidden_states, bin_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ProDMPredictor(hidden_dim, num_bins, bin_edges).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        steps = 0
        for feats, labels in tqdm(loader, desc=f"ProD-M epoch {epoch + 1}/{epochs}"):
            feats = feats.to(args.device)
            labels = labels.to(args.device)
            loss = loss_fn(model(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        print(f"Epoch {epoch + 1} — loss: {total / max(steps, 1):.4f}")

    model.eval()
    with torch.no_grad():
        pred_lengths = model.predict_lengths(hidden_states.to(args.device))
    true_lengths = [r.output_length for r in records]
    mae = mean_absolute_error(true_lengths, pred_lengths)
    print(f"Training MAE vs median target: {mae:.2f} tokens")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_prod_m(model, args.output, meta={"llm_model": llm_model, "mae": mae, "dataset": label_file.meta.get("dataset")})
    print(f"Saved ProD-M -> {args.output}")


if __name__ == "__main__":
    main()
