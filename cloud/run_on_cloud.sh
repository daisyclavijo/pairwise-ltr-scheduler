#!/usr/bin/env bash
# Run the full 1000-prompt pipeline on a cloud GPU (Colab / RunPod / etc.)
#
#   export HF_TOKEN=hf_...
#   bash cloud/run_on_cloud.sh

set -e

if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
  echo "ERROR: export HF_TOKEN=hf_... first"
  exit 1
fi

pip install -q -r requirements.txt
python scripts/check_setup.py
python scripts/run_all.py \
  --config configs/live_run.yaml \
  --llm-profile llama32 \
  --limit 1000 \
  --chunk-size 50 \
  --device cuda
