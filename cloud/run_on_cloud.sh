#!/usr/bin/env bash
# Real ProD-M + PARS pipeline on cloud GPU with Llama 3.1 8B
#
# Prerequisites:
#   export HF_TOKEN=hf_...   (accept Llama license on HuggingFace first)
#
# Datasets: GSM8K + MBPP + LMSYS + LongBench

set -e

if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
  echo "ERROR: Set HF_TOKEN before running."
  echo "  export HF_TOKEN=hf_..."
  exit 1
fi

echo "=== Real ProD-M + PARS (Llama 3.1 8B) ==="

pip install -q -r requirements.txt

python scripts/run_pipeline.py \
  --dataset all \
  --limit 400 \
  --device cuda

echo ""
echo "Done. Real median labels from Llama -> ProD-M -> PARS -> scheduler eval."
