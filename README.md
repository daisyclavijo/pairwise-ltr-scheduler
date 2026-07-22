# Improving LTR with PARS + ProD-M + Priority

**FDU Vancouver Capstone (CS Master's)**

## Three-way comparison only

| # | Method | Whose? |
|---|--------|--------|
| 1 | **FCFS** | Baseline |
| 2 | **LTR scheduler** | Main paper (pointwise, single-sample labels) |
| 3 | **PARS + ProD-M + Priority** | **Ours** |

- **ProD-M** (not in the main paper): sample Llama `r` times, take the **median** length as the label for training our ranker.  
- **PARS**: pairwise BERT ranker.  
- **Priority**: high / normal / low + starvation prevention.

## Primary run scale: **1000 prompts**

Config: `configs/live_run.yaml` (1000 prompts, chunk size 50, 3 samples/prompt).  
Use Colab Pro / A100 when possible. Labels resume from Drive after disconnects.

```python
import os
from google.colab import drive
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"
drive.mount("/content/drive")

!git clone https://github.com/anmolsaluja/pairwise-ltr-scheduler.git
%cd pairwise-ltr-scheduler
!pip install -q -r requirements.txt
!python scripts/check_setup.py

# 1000 prompts in chunks of 50 (--resume safe)
!python scripts/generate_labels.py --config configs/live_run.yaml \
  --limit 1000 --chunk-size 50 --num-samples 3 --resume --device cuda \
  --backup-dir /content/drive/MyDrive/capstone_results

!python scripts/train_prod_m.py --config configs/live_run.yaml \
  --target single --output checkpoints/ltr_pointwise.pt --device cuda
!python scripts/train_ranker.py --config configs/live_run.yaml \
  --train-samples 1000 --device cuda
!python scripts/evaluate.py --config configs/live_run.yaml --limit 1000 --device cuda

# Report graphs (paper Fig. 2 / Fig. 3 style)
!python scripts/plot_results.py --config configs/live_run.yaml --limit 1000 --device cuda \
  --out-dir /content/drive/MyDrive/capstone_results/figures
```

Or use `notebooks/colab_run.ipynb` / `python scripts/run_live.py --limit 1000 ...`.

Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct  

**Time (A100):** labeling is still the long step (multi-session OK with `--resume`); train/eval/plots are much faster.

## Final printed result looks like

```text
=== FCFS (baseline) ===
=== LTR scheduler (MAIN PAPER) ===
=== PARS + ProD-M + Priority (OURS) ===

LTR vs FCFS: ...%
OURS vs LTR (main paper): ...%
OURS vs FCFS: ...%
```

See `docs/PROJECT_OVERVIEW.md` for the report write-up.

## Live GPU serving (optional)

```bash
# HuggingFace live path (recommended on Colab if vLLM import fails)
python scripts/evaluate_live_hf.py --config configs/live_run.yaml --limit 1000 --device cuda

# vLLM path (when install works)
pip install vllm
python scripts/evaluate_live.py --config configs/live_run.yaml --limit 1000 --device cuda
```

Results save to `data/processed/live_eval_results.json` and figures under Drive `capstone_results/figures/`.
