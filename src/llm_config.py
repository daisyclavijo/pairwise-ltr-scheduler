"""Pick the served LLM from config profile."""

from __future__ import annotations


PROFILES = {
    # Best default: same 8B footprint as the PPT, newer weights (Dec 2024)
    "llama33": {
        "model": "meta-llama/Llama-3.3-8B-Instruct",
        "max_new_tokens": 512,
        "note": "Updated Llama instruct model; fits T4 with 4-bit.",
    },
    # Original midterm target
    "llama31": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "max_new_tokens": 512,
        "note": "Original PPT served model.",
    },
    # Used in the ProD paper; strong length variance
    "qwen25": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "max_new_tokens": 512,
        "note": "ProD paper baseline; good OOD length spread.",
    },
    # Best for PARS reasoning experiments (long CoT traces)
    "deepseek_r1": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "max_new_tokens": 2048,
        "note": "Reasoning model; huge output-length variance (PARS paper focus).",
    },
}


def resolve_llm(cfg: dict) -> dict:
    """Return model id + generation settings for the active profile."""
    llm_cfg = cfg.get("llm", {})
    profile = llm_cfg.get("profile", "llama33")
    profiles = llm_cfg.get("profiles", PROFILES)

    if profile not in profiles:
        raise ValueError(f"Unknown llm.profile '{profile}'. Choose: {list(profiles.keys())}")

    chosen = profiles[profile]
    return {
        "model": chosen["model"],
        "load_in_4bit": llm_cfg.get("load_in_4bit", True),
        "max_prompt_tokens": llm_cfg.get("max_prompt_tokens", 2048),
        "max_new_tokens": llm_cfg.get(
            "max_new_tokens", chosen.get("max_new_tokens", cfg["prod_m"]["max_new_tokens"])
        ),
        "profile": profile,
        "note": chosen.get("note", ""),
    }
