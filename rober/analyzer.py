"""
RoBERTa-based sentiment analyzer (transformer).

Model
-----
cardiffnlp/twitter-roberta-base-sentiment
  - Fine-tuned on ~58M tweets
  - Labels: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
  - ~500MB download on first use (cached to ~/.cache/huggingface)

Strengths:  context-aware, handles negation, slang, sarcasm.
Weaknesses: slow on CPU; requires ~500MB model weights; GPU strongly recommended for batch.

Usage
-----
    analyzer = RoBERTaAnalyzer()
    result   = analyzer.predict("This is surprisingly good!")
    print(result)   # [roberta] POSITIVE  confidence=0.956
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

_LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}


@dataclass
class RoBERTaResult:
    label: str          # 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'
    confidence: float   # softmax probability for the predicted label [0, 1]
    scores: dict        # raw softmax scores for all three labels
    method: str = "roberta"

    def __str__(self) -> str:
        return (
            f"[{self.method}] {self.label:>8}  confidence={self.confidence:.3f}  "
            f"scores={{{', '.join(f'{k}: {v:.2f}' for k, v in self.scores.items())}}}"
        )


class RoBERTaAnalyzer:
    """
    Transformer-based sentiment analyzer using a pretrained RoBERTa model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Defaults to the Cardiff NLP Twitter model.
    device : str | int | None
        Torch device string ('cpu', 'cuda', 'mps') or device index.
        Defaults to 'cuda' if available, else 'cpu'.
    batch_size : int
        Number of samples per inference batch in predict_batch(). Default 32.
    max_length : int
        Maximum token length. Texts longer than this are truncated. Default 512.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = self._resolve_device(device)
        self._pipeline = None  # lazy-loaded on first predict() call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> RoBERTaResult:
        """Return a RoBERTaResult for a single string."""
        pipeline = self._get_pipeline()
        raw = pipeline(text, truncation=True, max_length=self.max_length)[0]

        # pipeline returns the top label; run all_scores=True for full distribution
        all_raw = pipeline(
            text,
            truncation=True,
            max_length=self.max_length,
            return_all_scores=True,
        )[0]

        scores = {_LABEL_MAP[item["label"]]: round(item["score"], 4) for item in all_raw}
        top_label = _LABEL_MAP[raw["label"]]
        confidence = round(raw["score"], 4)

        return RoBERTaResult(top_label, confidence, scores)

    def predict_batch(self, texts: list[str]) -> list[RoBERTaResult]:
        """
        Efficiently predict sentiment for a list of strings.
        Processes in batches for GPU utilization.
        """
        pipeline = self._get_pipeline()
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_outputs = pipeline(
                batch,
                truncation=True,
                max_length=self.max_length,
                return_all_scores=True,
            )
            for output in all_outputs:
                scores = {_LABEL_MAP[item["label"]]: round(item["score"], 4) for item in output}
                top = max(scores, key=scores.__getitem__)
                results.append(RoBERTaResult(top, scores[top], scores))

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        """Lazy-load the HuggingFace pipeline on first call."""
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
                logger.info("Loading RoBERTa model '%s' on %s …", self.model_name, self._device)
                self._pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=self._device,
                    tokenizer=self.model_name,
                )
                logger.info("Model loaded successfully.")
            except ImportError as exc:
                raise ImportError(
                    "transformers package not found. Install with: pip install transformers torch"
                ) from exc
        return self._pipeline

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is not None:
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
