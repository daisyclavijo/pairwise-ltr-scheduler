# Improving LLM Serving Latency with Median Labels + Pairwise LTR

**FDU Vancouver Capstone (CS Master's)**  
Team: Mohammed Sirajuddin, Anmol Saluja, Chandra Sekhar Venigalla,
Daisy Lucia Clavijo Navas, Veera Venkata Sai Sravan Bhamidipati

## Idea in one sentence

The main paper uses **Learning-to-Rank (LTR)** to schedule shorter LLM requests
first (better than FCFS). We **improve** that by using **ProD-M median length
labels**, a **PARS pairwise ranker**, and **request priority / starvation
prevention**.

## What we compare

| Policy | Meaning |
|--------|---------|
| **FCFS** | Baseline (arrival order; HOL blocking) |
| **LTR** | Main-paper style: pointwise predicted length → SJF (our ProD-M MLP) |
| **PARS** | **Ours:** pairwise ranking + priority + starvation |
| **Oracle** | Perfect SJF with true median lengths (upper bound) |

## Pipeline

```
Prompts
  -> Llama sampled r=5 times -> MEDIAN length label (+ hidden states)
  -> ProD-M MLP ..............-> LTR pointwise scores
  -> Pair build from medians -> PARS pairwise ranker
  -> Scheduler with priority
  -> Compare FCFS vs LTR vs PARS vs Oracle
```

---

## Run on Google Colab (GPU)

1. Runtime → GPU (T4)
2. Accept: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Token: https://huggingface.co/settings/tokens

```python
import os
from google.colab import drive
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN"
drive.mount("/content/drive")

!git clone https://github.com/anmolsaluja/pairwise-ltr-scheduler.git
%cd pairwise-ltr-scheduler
!pip install -q -r requirements.txt

!python scripts/check_setup.py
!python scripts/run_all.py --limit 100 --device cuda
!python scripts/eval_ood.py --device cuda
!python scripts/ablation_labels.py --device cuda --epochs 3

!mkdir -p /content/drive/MyDrive/capstone_results
!cp -r checkpoints data/processed /content/drive/MyDrive/capstone_results/
```

Or use `notebooks/colab_run.ipynb`.

**Time (T4, `--limit 100`):** about 2.5–4.5 hours (labels + PARS training dominate).

---

## Step-by-step locally / cloud

```bash
export HF_TOKEN=hf_...
pip install -r requirements.txt
python scripts/check_setup.py

# Phase 1: median labels
python scripts/generate_labels.py --limit 100 --device cuda

# Phase 2: ProD-M (feeds LTR baseline)
python scripts/train_prod_m.py --device cuda

# Phase 3: PARS pairwise ranker (our method)
python scripts/train_ranker.py --device cuda

# Phase 4: FCFS vs LTR vs PARS vs Oracle
python scripts/evaluate.py --device cuda
```

One command: `python scripts/run_all.py --limit 100 --device cuda`

CPU-only scheduler demo (no Llama): `python scripts/demo_cpu.py`

---

## Project layout

```
src/
  llama.py       # sample lengths + hidden states
  prod_m.py      # ProD-M length predictor (used by LTR policy)
  ranker.py      # PARS pairwise BERT ranker
  scheduler.py   # FCFS / LTR / PARS (+ priority, starvation)
  simulate.py    # discrete-event comparison
  ...
scripts/
  generate_labels.py   # median labels
  train_prod_m.py      # train LTR length model
  train_ranker.py      # train PARS
  evaluate.py          # comparison table for the report
  eval_ood.py / ablation_labels.py
  vllm_integration.py  # stretch: real vLLM priorities
```

More detail for the report: `docs/PROJECT_OVERVIEW.md`

## References

1. Main paper — LTR scheduling for LLM latency (FDU / Saravana Kumar et al.)
2. Fu et al. — Efficient LLM Scheduling by Learning to Rank
3. Wang et al. — ProD (median / robust length prediction)
4. Tao et al. — PARS (pairwise LTR serving)
5. Kwon et al. — vLLM
