"""
Load prompts, ProD-M labels, and build PARS training pairs.

Pipeline data flow (from midterm slides):
  prompts -> repeated sampling -> median labels (ProD-M)
          -> filtered pairs -> PARS ranker training
"""

from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import dataclass, field
from typing import Iterator

from datasets import load_dataset

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "sample_prompts.json")
PROD_LABELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


@dataclass
class PromptRecord:
    """One prompt with a length label (median or proxy)."""

    prompt_id: str
    text: str
    output_length: int
    priority: str = "normal"
    sample_lengths: list[int] = field(default_factory=list)
    single_sample_length: int = 0


@dataclass
class ProDLabelFile:
    """Saved output of generate_prod_labels.py."""

    records: list[PromptRecord]
    meta: dict = field(default_factory=dict)


def load_local_sample_prompts(limit: int | None = None) -> list[PromptRecord]:
    with open(SAMPLE_DATA) as f:
        rows = json.load(f)

    records = []
    for i, row in enumerate(rows):
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")
        text = f"{instruction}\n{inp}".strip() if inp else instruction
        length = max(1, len(output.split()))
        records.append(
            PromptRecord(
                prompt_id=f"sample_{i}",
                text=text,
                output_length=length,
                single_sample_length=length,
            )
        )

    if limit:
        records = records[:limit]
    return records


def load_alpaca_prompts(
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
    use_local: bool = False,
) -> list[PromptRecord]:
    if use_local or os.environ.get("USE_LOCAL_DATA") == "1":
        return load_local_sample_prompts(limit=limit)

    try:
        ds = load_dataset("tatsu-lab/alpaca", split=split)
        rows = list(ds)
    except Exception:
        return load_local_sample_prompts(limit=limit)

    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]

    records = []
    for i, row in enumerate(rows):
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")
        text = f"{instruction}\n{inp}".strip() if inp else instruction
        length = max(1, len(output.split()))
        records.append(
            PromptRecord(
                prompt_id=f"alpaca_{split}_{i}",
                text=text.strip(),
                output_length=length,
                single_sample_length=length,
            )
        )
    return records


def save_prod_labels(
    path: str,
    records: list[PromptRecord],
    meta: dict,
    hidden_states_path: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if hidden_states_path:
        meta = {**meta, "hidden_states_path": hidden_states_path}
    payload = {
        "meta": meta,
        "records": [
            {
                "prompt_id": r.prompt_id,
                "text": r.text,
                "priority": r.priority,
                "sample_lengths": r.sample_lengths,
                "median_length": r.output_length,
                "single_sample_length": r.single_sample_length or (
                    r.sample_lengths[0] if r.sample_lengths else r.output_length
                ),
            }
            for r in records
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_prod_labels(path: str) -> ProDLabelFile:
    with open(path) as f:
        payload = json.load(f)

    records = []
    for row in payload["records"]:
        records.append(
            PromptRecord(
                prompt_id=row["prompt_id"],
                text=row["text"],
                output_length=int(row["median_length"]),
                priority=row.get("priority", "normal"),
                sample_lengths=row.get("sample_lengths", []),
                single_sample_length=int(row.get("single_sample_length", row["median_length"])),
            )
        )
    return ProDLabelFile(records=records, meta=payload.get("meta", {}))


def simulate_repeated_lengths(
    records: list[PromptRecord],
    num_samples: int = 5,
    noise_ratio: float = 0.3,
    seed: int = 42,
) -> list[PromptRecord]:
    """
    Offline fallback when no GPU is available.

    Adds noise around a base length, then takes the median — same idea as ProD-M
  without running a real LLM.
    """
    rng = random.Random(seed)
    out = []

    for rec in records:
        base = rec.single_sample_length or rec.output_length
        samples = []
        for _ in range(num_samples):
            jitter = rng.uniform(-noise_ratio, noise_ratio) * base
            samples.append(max(1, int(base + jitter)))

        median = int(statistics.median(samples))
        out.append(
            PromptRecord(
                prompt_id=rec.prompt_id,
                text=rec.text,
                output_length=median,
                priority=rec.priority,
                sample_lengths=samples,
                single_sample_length=base,
            )
        )
    return out


def make_pairwise_samples(
    records: list[PromptRecord],
    min_length_diff: float = 0.2,
    use_median: bool = True,
) -> list[tuple[str, str, int]]:
    """
    Build (prompt_a, prompt_b, label) pairs for PARS margin ranking loss.

    When use_median=True (default), pairs come from ProD-M median labels.
    """
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            len_a = records[i].output_length
            len_b = records[j].output_length
            max_len = max(len_a, len_b)
            if max_len == 0:
                continue

            rel_diff = abs(len_a - len_b) / max_len
            if rel_diff < min_length_diff:
                continue

            if len_a > len_b:
                pairs.append((records[i].text, records[j].text, 1))
            else:
                pairs.append((records[j].text, records[i].text, 1))

    random.shuffle(pairs)
    return pairs


def make_single_sample_pairs(
    records: list[PromptRecord],
    min_length_diff: float = 0.2,
) -> list[tuple[str, str, int]]:
    """Ablation baseline: pairs from one-sample labels (noisy)."""
    ablation = []
    for rec in records:
        single = rec.single_sample_length or rec.output_length
        ablation.append(
            PromptRecord(
                prompt_id=rec.prompt_id,
                text=rec.text,
                output_length=single,
                priority=rec.priority,
            )
        )
    return make_pairwise_samples(ablation, min_length_diff=min_length_diff)


def stream_requests(
    records: list[PromptRecord],
    arrival_rate: float,
    seed: int = 42,
) -> Iterator[tuple[float, PromptRecord]]:
    rng = random.Random(seed)
    t = 0.0
    for record in records:
        yield t, record
        t += rng.expovariate(arrival_rate)
