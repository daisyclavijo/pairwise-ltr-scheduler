"""Small helpers used across the project."""

from __future__ import annotations

import os

import yaml


# Default HuggingFace model ids for each profile.
# llama32 is the one we use for Colab (fits on a T4 with 4-bit).
PROFILES = {
    "llama32": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "max_new_tokens": 512,
    },
    "llama31": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "max_new_tokens": 512,
    },
    "qwen25": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "max_new_tokens": 512,
    },
    "deepseek_r1": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "max_new_tokens": 2048,
    },
}


def load_config(path="configs/live_run.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolve_llm(cfg, profile=None):
    """Pick which LLM to use from config (or override)."""
    llm = cfg.get("llm", {})
    name = profile or llm.get("profile", "llama32")

    # merge built-in defaults with anything in the yaml
    profiles = {**PROFILES, **llm.get("profiles", {})}
    if name not in profiles:
        raise ValueError(f"Unknown profile '{name}'. Options: {list(profiles)}")

    chosen = profiles[name]
    return {
        "profile": name,
        "model": chosen["model"],
        "max_new_tokens": chosen.get("max_new_tokens", cfg["prod_m"]["max_new_tokens"]),
        "load_in_4bit": llm.get("load_in_4bit", True),
        "max_prompt_tokens": llm.get("max_prompt_tokens", 4096),
    }
