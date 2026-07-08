#!/usr/bin/env python3
"""
Phase 1: generate real ProD-M labels using Llama 3.1 8B.

Runs the served LLM r times per prompt, saves median lengths + hidden states.

Example:
  export HF_TOKEN=hf_...
  python scripts/generate_prod_labels.py --dataset all --limit 400 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import median

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import PromptRecord, save_prod_labels
from src.datasets import load_prompts
from src.llm_config import resolve_llm
from src.prod_m import LlamaServer, save_hidden_states


def check_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN before running.")
        print("  1. Accept license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("  2. Create token: https://huggingface.co/settings/tokens")
        print("  3. export HF_TOKEN=hf_...")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate real ProD-M median labels")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/processed/prod_labels.json")
    parser.add_argument("--hidden-output", default="data/processed/prod_hidden.pt")
    parser.add_argument("--dataset", default=None, help="gsm8k|mbpp|lmsys|longbench|all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-profile", default=None, help="llama33|qwen25|deepseek_r1|llama31")
    args = parser.parse_args()

    check_hf_token()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.llm_profile:
        cfg.setdefault("llm", {})["profile"] = args.llm_profile

    llm_cfg = resolve_llm(cfg)
    ds_name = args.dataset or cfg["datasets"]["name"]
    limit = args.limit or cfg["datasets"]["limit"]
    num_samples = args.num_samples or cfg["prod_m"]["num_samples"]
    max_new = llm_cfg["max_new_tokens"]
    temperature = cfg["prod_m"]["temperature"]
    top_p = cfg["prod_m"]["top_p"]
    llm_model = llm_cfg["model"]
    load_4bit = llm_cfg["load_in_4bit"]
    encode_batch = cfg["prod_m"]["encode_batch_size"]

    print(f"LLM profile: {llm_cfg['profile']} -> {llm_model}")
    print(f"Loading {limit} prompts from dataset: {ds_name}")
    records = load_prompts(
        ds_name,
        limit=limit,
        per_dataset_limits=cfg["datasets"].get("per_dataset"),
    )
    print(f"Loaded {len(records)} prompts")

    server = LlamaServer(
        llm_model,
        device=args.device,
        load_in_4bit=load_4bit,
        max_prompt_tokens=llm_cfg["max_prompt_tokens"],
    )

    labeled = []
    print(f"Generating {num_samples} samples per prompt with {llm_model}...")
    for i, rec in enumerate(records):
        samples = server.generate_lengths(
            rec.text,
            num_samples=num_samples,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
        )
        med = int(median(samples))
        labeled.append(
            PromptRecord(
                prompt_id=rec.prompt_id,
                text=rec.text,
                output_length=med,
                priority=rec.priority,
                sample_lengths=samples,
                single_sample_length=samples[0],
            )
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(records):
            print(f"  generated {i + 1}/{len(records)}  (last median={med} tokens)")

    print("Extracting Llama hidden states (prefill only)...")
    hidden = server.encode([r.text for r in labeled], batch_size=encode_batch)
    save_hidden_states(args.hidden_output, hidden)

    meta = {
        "mode": "llm",
        "model": llm_model,
        "llm_profile": llm_cfg["profile"],
        "dataset": ds_name,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "num_prompts": len(labeled),
    }
    save_prod_labels(args.output, labeled, meta, hidden_states_path=args.hidden_output)
    print(f"Saved labels -> {args.output}")
    print(f"Saved hidden states -> {args.hidden_output}")


if __name__ == "__main__":
    main()
