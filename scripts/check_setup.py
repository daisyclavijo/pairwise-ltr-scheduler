#!/usr/bin/env python3
"""Quick checks before running the real pipeline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_hf_token


def main():
    ok = True

    if get_hf_token():
        print("[ok] HF_TOKEN is set")
    else:
        print("[FAIL] set HF_TOKEN (needed for Llama)")
        ok = False

    try:
        import torch
        print(f"[ok] torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[ok] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[FAIL] torch.cuda.is_available() is False")
            print("       If you are on Colab: Runtime → GPU, then restart session.")
            print("       Do NOT `pip install torch` from PyPI (that installs CPU-only).")
            print("       Repair with:")
            print("         !pip install -q --upgrade torch --index-url "
                  "https://download.pytorch.org/whl/cu121")
            ok = False
    except ImportError:
        print("[FAIL] torch not installed")
        ok = False

    for pkg in ("transformers", "datasets", "accelerate"):
        try:
            __import__(pkg)
            print(f"[ok] {pkg}")
        except ImportError:
            print(f"[FAIL] {pkg} missing")
            ok = False

    try:
        import bitsandbytes  # noqa: F401
        print("[ok] bitsandbytes")
    except ImportError:
        print("[WARN] bitsandbytes missing (needed for 4-bit on GPU)")

    print("\nTrying a tiny dataset load...")
    try:
        from src.datasets import load_gsm8k
        rows = load_gsm8k(limit=2)
        print(f"[ok] gsm8k loaded ({len(rows)} rows)")
    except Exception as e:
        print(f"[FAIL] dataset: {e}")
        ok = False

    if ok:
        print("\nReady. Next: label 1000 prompts (chunked) or scripts/run_live.py")
    else:
        print("\nFix the failures above first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
