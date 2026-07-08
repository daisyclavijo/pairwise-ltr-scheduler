#!/usr/bin/env python3
"""Verify GPU, HF token, and dataset access before running the real pipeline."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ok = True

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("[ok] HF_TOKEN is set")
    else:
        print("[FAIL] HF_TOKEN not set — required for Llama 3.1 8B")
        ok = False

    try:
        import torch
        if torch.cuda.is_available():
            print(f"[ok] CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARN] No CUDA GPU — pipeline needs cloud GPU")
    except ImportError:
        print("[FAIL] torch not installed")
        ok = False

    try:
        import bitsandbytes  # noqa: F401
        print("[ok] bitsandbytes installed (4-bit Llama)")
    except ImportError:
        print("[WARN] bitsandbytes missing — pip install bitsandbytes")

    for pkg in ["transformers", "datasets", "accelerate"]:
        try:
            __import__(pkg)
            print(f"[ok] {pkg} installed")
        except ImportError:
            print(f"[FAIL] {pkg} not installed")
            ok = False

    print("\nDatasets (quick load test):")
    try:
        from src.datasets import load_gsm8k, load_math, load_livecodebench, load_wildchat, load_longbench_v2
        gsm = load_gsm8k(limit=2)
        math = load_math(limit=2)
        lcb = load_livecodebench(limit=2)
        wc = load_wildchat(limit=2)
        lb = load_longbench_v2(limit=2)
        print(f"[ok] GSM8K ({len(gsm)}), MATH ({len(math)}), LiveCodeBench ({len(lcb)})")
        print(f"[ok] WildChat ({len(wc)}), LongBench-v2 ({len(lb)})")
    except Exception as e:
        print(f"[FAIL] dataset load: {e}")
        ok = False

    if ok:
        print("\nReady. Run: python scripts/run_pipeline.py --device cuda")
    else:
        print("\nFix the issues above, then run the pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
