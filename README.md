# ProD-M + PARS — Llama 3.3 8B + 2025–2026 benchmark datasets

## Recommended upgrades (now in `configs/default.yaml`)

### LLM profiles (`llm.profile`)

| Profile | Model | Best for |
|---------|-------|----------|
| **`llama33`** (default) | `meta-llama/Llama-3.3-8B-Instruct` | Direct upgrade from PPT's Llama 3.1; same 8B VRAM footprint |
| `llama31` | `Meta-Llama-3.1-8B-Instruct` | Original midterm baseline |
| `qwen25` | `Qwen/Qwen2.5-7B-Instruct` | ProD paper served model; strong length variance |
| `deepseek_r1` | `DeepSeek-R1-Distill-Llama-8B` | **PARS paper focus** — reasoning traces, huge length spread |

Switch profile:
```bash
python scripts/generate_prod_labels.py --llm-profile deepseek_r1 --device cuda
```

### Datasets (`datasets.name=all`)

| Dataset | Replaces | Why better |
|---------|----------|------------|
| **GSM8K** | — | Short math baseline (kept) |
| **MATH** | — | Competition math; longer reasoning chains |
| **LiveCodeBench** | MBPP | Rolling LeetCode/AtCoder; contamination-resistant (2025 standard) |
| **WildChat-1M** | LMSYS | Real open chat logs; no gating |
| **LongBench v2** | LongBench v1 | 503 tasks, 8k–2M context; 2025 long-context benchmark |

## Run

```bash
export HF_TOKEN=hf_...
pip install -r requirements.txt
python scripts/check_setup.py

# Default: Llama 3.3 + all updated datasets
python scripts/run_pipeline.py --device cuda

# Reasoning experiment (PARS paper scenario)
python scripts/run_pipeline.py --llm-profile deepseek_r1 --device cuda
```

## Why these choices fit *this* project

ProD-M + PARS cares about **output-length variance**, not benchmark accuracy:
- **LiveCodeBench** + **MATH** → short vs long solutions in same workload
- **WildChat** → realistic mixed chat lengths
- **LongBench v2** → long-context prompts stress the scheduler
- **DeepSeek-R1** → reasoning CoT makes HOL blocking worse (exactly what PARS targets)

## Team

FDU Vancouver Capstone — 2026.
