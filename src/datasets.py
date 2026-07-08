"""
Benchmark datasets for ProD-M + PARS (updated for 2025–2026).

Recommended mix (config datasets.name=all):
  gsm8k          — short math (baseline)
  math           — competition math, longer chains (MATH)
  livecodebench  — contamination-resistant coding (replaces MBPP)
  wildchat       — real user chat (replaces LMSYS when gated)
  longbench_v2   — long-context MCQ (replaces LongBench v1)
"""

from __future__ import annotations

import os
import random

from datasets import load_dataset

from src.data import PromptRecord


def load_gsm8k(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    return [
        PromptRecord(
            prompt_id=f"gsm8k_{i}",
            text=f"Solve step by step.\n\n{row['question'].strip()}",
            output_length=0,
        )
        for i, row in enumerate(rows)
    ]


def load_math(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    """Competition math — longer outputs than GSM8K."""
    parts = []
    for subset in ("algebra", "number_theory", "counting_and_probability"):
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subset, split="train")
            parts.extend(list(ds))
        except Exception:
            continue
    if not parts:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        parts = list(ds)

    random.Random(seed).shuffle(parts)
    if limit:
        parts = parts[:limit]

    return [
        PromptRecord(
            prompt_id=f"math_{i}",
            text=f"Solve this competition math problem. Show your reasoning.\n\n{row['problem'].strip()}",
            output_length=0,
        )
        for i, row in enumerate(parts)
    ]


def load_livecodebench(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    """LiveCodeBench — rolling competitive programming problems."""
    ds = load_dataset("livecodebench/code_generation", split="test", streaming=True)
    records = []
    for row in ds:
        content = row.get("question_content", "").strip()
        if not content:
            continue
        records.append(
            PromptRecord(
                prompt_id=f"lcb_{row.get('question_id', len(records))}",
                text=f"Write a complete Python solution.\n\n{content}",
                output_length=0,
            )
        )
        if limit and len(records) >= limit:
            break
    random.Random(seed).shuffle(records)
    return records


def load_mbpp(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    try:
        return load_livecodebench(limit=limit, seed=seed)
    except Exception:
        ds = load_dataset("google-research-datasets/mbpp", split="train")
        rows = list(ds)
        random.Random(seed).shuffle(rows)
        if limit:
            rows = rows[:limit]
        return [
            PromptRecord(
                prompt_id=f"mbpp_{i}",
                text=f"Write a Python function.\n\n{row['text'].strip()}",
                output_length=0,
            )
            for i, row in enumerate(rows)
        ]


def load_wildchat(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    records = []
    seen = set()
    for row in ds:
        convo = row.get("conversation", [])
        if not convo:
            continue
        user_msg = convo[0].get("content", "").strip()
        if not user_msg or user_msg in seen:
            continue
        seen.add(user_msg)
        records.append(
            PromptRecord(prompt_id=f"wildchat_{len(records)}", text=user_msg, output_length=0)
        )
        if limit and len(records) >= limit:
            break
    random.Random(seed).shuffle(records)
    return records


def load_lmsys(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    records = []
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True, token=token)
        seen = set()
        for row in ds:
            convo = row.get("conversation", [])
            if not convo:
                continue
            user_msg = convo[0].get("content", "").strip()
            if not user_msg or user_msg in seen:
                continue
            seen.add(user_msg)
            records.append(
                PromptRecord(prompt_id=f"lmsys_{len(records)}", text=user_msg, output_length=0)
            )
            if limit and len(records) >= limit:
                break
    except Exception:
        pass
    if records:
        random.Random(seed).shuffle(records)
        return records
    return load_wildchat(limit=limit, seed=seed)


def load_longbench_v2(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]

    records = []
    for i, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        choices = "\n".join(
            f"{k[-1]}. {row[k]}"
            for k in ("choice_A", "choice_B", "choice_C", "choice_D")
            if row.get(k)
        )
        text = f"{question}\n\n{choices}\n\nAnswer with the best choice and explain briefly."
        records.append(PromptRecord(prompt_id=f"lb2_{row.get('_id', i)}", text=text, output_length=0))
    return records


def load_longbench(limit: int | None = None, seed: int = 42) -> list[PromptRecord]:
    try:
        return load_longbench_v2(limit=limit, seed=seed)
    except Exception:
        ds = load_dataset("hotpot_qa", "fullwiki", split="train", streaming=True)
        records = []
        for row in ds:
            titles = " ".join(row.get("context", {}).get("title", [])[:3])
            sents = row.get("context", {}).get("sentences", [[]])[0][:15]
            context = f"{titles}\n" + " ".join(s[:200] for s in sents if s)
            q = row.get("question", "").strip()
            records.append(
                PromptRecord(
                    prompt_id=f"hotpot_{len(records)}",
                    text=f"Context:\n{context[:4000]}\n\nQuestion: {q}\n\nAnswer:",
                    output_length=0,
                )
            )
            if limit and len(records) >= limit:
                break
        random.Random(seed).shuffle(records)
        return records


LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "livecodebench": load_livecodebench,
    "wildchat": load_wildchat,
    "longbench_v2": load_longbench_v2,
    "mbpp": load_mbpp,
    "lmsys": load_lmsys,
    "longbench": load_longbench,
}


def load_prompts(
    name: str,
    limit: int | None = None,
    seed: int = 42,
    per_dataset_limits: dict | None = None,
) -> list[PromptRecord]:
    name = name.lower()

    if name == "all":
        limits = per_dataset_limits or {}
        primary = ["gsm8k", "math", "livecodebench", "wildchat", "longbench_v2"]
        active = primary if any(k in limits for k in primary) else list(LOADERS.keys())
        parts = []
        for ds_name in active:
            if ds_name not in LOADERS:
                continue
            ds_limit = limits.get(ds_name)
            if ds_limit is None and limit:
                ds_limit = max(1, limit // len(active))
            parts.extend(LOADERS[ds_name](limit=ds_limit, seed=seed))
        random.Random(seed).shuffle(parts)
        if limit:
            parts = parts[:limit]
        return parts

    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose: {sorted(set(LOADERS.keys()) | {'all'})}")

    return LOADERS[name](limit=limit, seed=seed)
