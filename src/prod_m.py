"""
ProD-M: median-supervised length predictor (Wang et al., arXiv:2604.07931).

Uses the real served LLM (Llama 3.1 8B) for:
  - repeated generation to build median labels
  - last-layer hidden states at prefill time
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def make_length_bins(max_length: int, num_bins: int) -> torch.Tensor:
    return torch.linspace(1, max_length, num_bins + 1)


def length_to_bin(length: float, bin_edges: torch.Tensor) -> int:
    idx = torch.bucketize(torch.tensor([length]), bin_edges[1:]).item()
    return min(idx, len(bin_edges) - 2)


def bin_to_length(bin_idx: int, bin_edges: torch.Tensor) -> float:
    lo = bin_edges[bin_idx].item()
    hi = bin_edges[bin_idx + 1].item()
    return (lo + hi) / 2.0


class ProDMPredictor(nn.Module):
    """2-layer MLP over K length bins."""

    def __init__(self, hidden_dim: int, num_bins: int, bin_edges: torch.Tensor):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bin_edges", bin_edges)
        mid = max(hidden_dim // 2, 256)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, num_bins),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

    @torch.no_grad()
    def predict_lengths(self, hidden_states: torch.Tensor) -> list[float]:
        self.eval()
        logits = self.forward(hidden_states)
        bins = logits.argmax(dim=-1).cpu().tolist()
        return [bin_to_length(b, self.bin_edges) for b in bins]


class LlamaServer:
    """
    Real Llama 3.1 inference for label generation and hidden-state extraction.

    Loads in 4-bit on cloud GPU (fits T4/A10 with 16GB).
    Set HF_TOKEN env var for gated meta-llama models.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_prompt_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.device = device
        self.max_prompt_tokens = max_prompt_tokens

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"token": token, "output_hidden_states": True}

        if load_in_4bit and device != "cpu":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float16 if device != "cpu" else torch.float32
            load_kwargs["device_map"] = device if device != "cpu" else None

        print(f"Loading {model_name} (4bit={load_in_4bit})...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size

    def format_chat(self, prompt: str) -> str:
        """Wrap prompt in Llama 3.1 Instruct chat template."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def encode(self, prompts: list[str], batch_size: int = 4) -> torch.Tensor:
        """Extract last-layer hidden state at the last prompt token (ProD paper)."""
        all_hidden = []

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            formatted = [self.format_chat(p) for p in batch_prompts]

            batch = self.tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
                return_tensors="pt",
            )

            if self.device != "cpu" and not hasattr(self.model, "hf_device_map"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            elif self.device != "cpu":
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

            outputs = self.model(**batch, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            idx = batch["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(last_hidden.size(0), device=last_hidden.device)
            all_hidden.append(last_hidden[rows, idx, :].float().cpu())

        return torch.cat(all_hidden, dim=0)

    @torch.no_grad()
    def generate_lengths(
        self,
        prompt: str,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[int]:
        """Run Llama num_samples times; return output token counts."""
        formatted = self.format_chat(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        lengths = []

        for _ in range(num_samples):
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            lengths.append(int(out.shape[1] - prompt_len))

        return lengths


# Keep old name for backward compatibility
HiddenStateExtractor = LlamaServer


def save_hidden_states(path: str, hidden: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(hidden, path)


def load_hidden_states(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def save_prod_m(model: ProDMPredictor, path: str, meta: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.mlp[0].in_features,
            "num_bins": model.num_bins,
            "bin_edges": model.bin_edges,
            "meta": meta,
        },
        path,
    )


def load_prod_m(path: str, device: str = "cpu") -> ProDMPredictor:
    ckpt = torch.load(path, map_location=device)
    model = ProDMPredictor(
        hidden_dim=ckpt["hidden_dim"],
        num_bins=ckpt["num_bins"],
        bin_edges=ckpt["bin_edges"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model
