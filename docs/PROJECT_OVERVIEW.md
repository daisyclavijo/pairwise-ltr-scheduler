# Project Overview (final report / viva)

## Story we are telling

Production LLM servers often use **FCFS**. Long answers block short ones
(**HOL blocking**). The **main paper** shows that a **Learning-to-Rank (LTR)**
scheduler can approximate Shortest-Job-First and cut latency.

**Our contribution:** improve that LTR pipeline with:

1. **Median length labels (ProD-M)** — sample the LLM `r` times per prompt;
   use the median instead of one noisy sample.
2. **Pairwise ranking (PARS)** — learn which prompt is longer than which,
   instead of only predicting an absolute length.
3. **Priority + starvation prevention** — high/normal/low user priority;
   boost requests that wait too long (~2 minutes).

Then we **compare**:

| Policy | Role |
|--------|------|
| FCFS | Baseline |
| LTR (pointwise) | Main-paper style, using our ProD-M length predictor |
| PARS (pairwise) | **Our improved scheduler** |
| Oracle | Upper bound (true median lengths) |

We are **not** claiming to reload the main paper’s exact OPT-125M checkpoint.
We re-implement the **same scheduling idea** (pointwise length → SJF) and show
that our median + pairwise + priority stack improves on it under the same
simulator.

## Pipeline

```
Prompts (GSM8K / MATH / code / chat / longbench)
        |
        v
Llama x r=5  --> median length + hidden state     [ProD-M labels]
        |
        +--> ProD-M MLP --> predicted length      [LTR policy]
        |
        +--> pairs from medians --> BERT ranker   [PARS policy]
        |
        v
Priority-aware SJF scheduler (+ starvation)
        |
        v
Compare FCFS | LTR | PARS | Oracle
```

## Code map

| Piece | File |
|-------|------|
| Median labels | `scripts/generate_labels.py`, `src/llama.py` |
| ProD-M / LTR scores | `scripts/train_prod_m.py`, `src/prod_m.py` |
| PARS ranker | `scripts/train_ranker.py`, `src/ranker.py` |
| Priority + starvation | `src/requests.py`, `src/scheduler.py` |
| Comparison | `scripts/evaluate.py`, `src/simulate.py` |
| ID/OOD + ablation | `scripts/eval_ood.py`, `scripts/ablation_labels.py` |
| vLLM stretch | `scripts/vllm_integration.py` |

## Metrics for the report

- ProD-M MAE vs median target
- PARS Kendall Tau / pairwise accuracy / NDCG
- Latency: avg, p50, p95, p99; queue wait; approx TTFT
- Throughput (req/s)
- Relative gains: LTR vs FCFS, **PARS vs LTR**, PARS vs FCFS

## How to reproduce

```bash
export HF_TOKEN=hf_...
python scripts/run_all.py --limit 100 --device cuda
python scripts/eval_ood.py --device cuda
python scripts/ablation_labels.py --device cuda --epochs 3
```

Colab T4, `--limit 100`: roughly **2.5–4.5 hours**.

## Honest scope

- Simulator uses `prefill + tokens * decode_time` — fair for policy ranking,
  not a full GPU trace.
- Default model: Llama-3.2-3B (Colab). Llama-3.1-8B via `configs/full_run.yaml`.
- Full high-QPS vLLM study is the stretch goal (midterm slide 7).
