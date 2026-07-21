#!/usr/bin/env python3
"""
Install + verify vLLM on Google Colab / cloud GPUs.

Colab often shows harmless dependency warnings (numba / cuml / opentelemetry).
Those are NOT why vLLM fails — a failed wheel install or a missing Runtime
restart after install is.

Usage:
  python scripts/ensure_vllm.py
  python scripts/ensure_vllm.py --install
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run(cmd):
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd)


def _cuda_major_minor():
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.STDOUT
        )
    except Exception:
        return None
    # look for "CUDA Version: 12.4"
    for line in out.splitlines():
        if "CUDA Version:" in line:
            ver = line.split("CUDA Version:")[-1].strip().split()[0]
            parts = ver.split(".")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    return None


def _try_import():
    try:
        import vllm  # noqa: F401
        from vllm import LLM, SamplingParams  # noqa: F401

        print(f"[ok] vLLM import works (v{getattr(vllm, '__version__', '?')})")
        return True
    except Exception as e:
        print(f"[FAIL] import vllm: {type(e).__name__}: {e}")
        return False


def install():
    if not _cuda_major_minor() and not os.path.exists("/dev/nvidia0"):
        print("ERROR: no NVIDIA GPU visible. Runtime → Change runtime type → GPU")
        return False

    mm = _cuda_major_minor()
    print(f"Driver CUDA reported: {mm}")

    # Prefer a CUDA-12-compatible install path for Colab T4/L4/A100.
    # Quiet pip hides failures — do NOT use -q here.
    candidates = []
    if mm and mm[0] >= 12:
        # Match common Colab drivers (12.1–12.6+) with cu124/cu121 torch indexes
        candidates.append(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-U",
                "vllm",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu124",
            ]
        )
        candidates.append(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-U",
                "vllm",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
    candidates.append([sys.executable, "-m", "pip", "install", "-U", "vllm"])

    for cmd in candidates:
        rc = _run(cmd).returncode
        if rc == 0 and _try_import():
            return True
        print("Install attempt did not yield a working import; trying next…")

    print(
        "\nStill broken after install attempts.\n"
        "On Colab: Runtime → Restart session, re-run setup (restore from Drive),\n"
        "then: python scripts/ensure_vllm.py --install\n"
        "Then re-run evaluate_live.py.\n"
    )
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="attempt pip install of vLLM matched to Colab CUDA",
    )
    args = parser.parse_args()

    if _try_import():
        sys.exit(0)

    if args.install:
        ok = install()
        sys.exit(0 if ok else 1)

    print(
        "vLLM not importable.\n"
        "Fix:\n"
        "  1) Runtime → GPU\n"
        "  2) python scripts/ensure_vllm.py --install\n"
        "  3) If import still fails: Runtime → Restart session, restore repo, repeat step 2\n"
        "Fallback (simulator, not live):\n"
        "  python scripts/evaluate.py --limit 1000 --device cuda\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
