"""
Full serving pipeline: ProD-M -> PARS -> priority-aware scheduler.

Matches the midterm presentation architecture:
  1. ProD-M gives robust length estimates (pointwise baseline)
  2. PARS ranker orders the waiting queue (pairwise, trained on ProD-M medians)
  3. User priority boosts urgent prompts ahead of normal traffic
"""

from __future__ import annotations

from src.pairwise_predictor import PairwiseRanker, load_model as load_pars
from src.priority import InferenceRequest
from src.prod_m import HiddenStateExtractor, ProDMPredictor, load_prod_m
from src.request_gateway import IncomingPrompt, RequestGateway
from src.scheduler import PairwiseLTRScheduler


class ServingPipeline:
  """
  End-to-end handler for incoming prompts.

  Usage:
    pipe = ServingPipeline(prod_m_path, pars_path, llm_model, device="cuda")
    req = pipe.accept(IncomingPrompt("Hello", priority="high"))
    batch = pipe.schedule_next()
  """

  def __init__(
    self,
    prod_m_checkpoint: str,
    pars_checkpoint: str,
    llm_model: str,
    device: str = "cpu",
    batch_size: int = 8,
    starvation_seconds: float = 120.0,
    priority_boosts: dict[str, float] | None = None,
  ):
    self.device = device
    self.gateway = RequestGateway()
    self.prod_m = load_prod_m(prod_m_checkpoint, device=device)
    self.pars = load_pars(pars_checkpoint, device=device)
    self.encoder = HiddenStateExtractor(llm_model, device=device)

    boosts = priority_boosts or {"high": -3.0, "normal": 0.0, "low": 3.0}
    self.scheduler = PairwiseLTRScheduler(
      batch_size=batch_size,
      starvation_seconds=starvation_seconds,
      priority_boosts=boosts,
    )

  def score_with_prod_m(self, requests: list[InferenceRequest]) -> None:
    """Fill rank_score using ProD-M length predictions."""
    if not requests:
      return
    prompts = [r.prompt for r in requests]
    hidden = self.encoder.encode(prompts)
    lengths = self.prod_m.predict_lengths(hidden)
    for req, length in zip(requests, lengths):
      req.rank_score = float(length)
      req.predicted_length = int(round(length))

  def score_with_pars(self, requests: list[InferenceRequest]) -> None:
    """Fill rank_score using PARS pairwise ranker."""
    if not requests:
      return
    prompts = [r.prompt for r in requests]
    scores = self.pars.score_prompts(prompts)
    for req, score in zip(requests, scores):
      req.rank_score = float(score)

  def accept(self, incoming: IncomingPrompt, use_pars: bool = True) -> InferenceRequest:
    """Register a new incoming prompt and score it."""
    req = self.gateway.submit(incoming)
    if use_pars:
      self.score_with_pars([req])
    else:
      self.score_with_prod_m([req])
    self.scheduler.add_request(req)
    return req

  def schedule_next(self) -> list[InferenceRequest]:
    """Pick the next batch using PARS + priority + starvation prevention."""
    return self.scheduler.pick_next_batch()
