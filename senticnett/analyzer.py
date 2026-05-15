"""
SenticNet sentiment analyzer.

Approach
--------
SenticNet is a commonsense knowledge base that maps concepts
(words and multi-word expressions) to polarity values in [-1, 1].
Unlike SentiWordNet, it understands phrases like "not good" directly
as a concept with negative polarity.

Pipeline:
1. Lowercase and tokenize the input.
2. Try multi-word concept lookup (bi-grams and uni-grams).
3. Aggregate polarity values; derive label via threshold.

Strengths:  handles multi-word concepts, commonsense reasoning.
Weaknesses: limited vocabulary; requires the senticnet package.

Install: pip install senticnet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SenticResult:
    label: str          # 'positive' | 'negative' | 'neutral'
    polarity: float     # aggregate polarity [-1.0, 1.0]
    concepts_found: list[str] = field(default_factory=list)
    method: str = "senticnet"

    def __str__(self) -> str:
        sign = "+" if self.polarity >= 0 else ""
        return (
            f"[{self.method}] {self.label:>8}  polarity={sign}{self.polarity:.3f}  "
            f"concepts={self.concepts_found}"
        )


class SenticNetAnalyzer:
    """
    Sentiment analyzer backed by the SenticNet 6.0 knowledge base.

    Parameters
    ----------
    pos_threshold : float
        Minimum polarity to be labelled 'positive'. Default 0.1.
    neg_threshold : float
        Maximum polarity to be labelled 'negative'. Default -0.1.
    use_bigrams : bool
        If True (default), try bigram concept lookup before unigrams.
    """

    def __init__(
        self,
        pos_threshold: float = 0.1,
        neg_threshold: float = -0.1,
        use_bigrams: bool = True,
    ) -> None:
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.use_bigrams = use_bigrams
        self._sn = self._load_senticnet()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> SenticResult:
        """Return a SenticResult for a single string."""
        tokens = text.lower().split()
        polarities: list[float] = []
        concepts_found: list[str] = []

        i = 0
        while i < len(tokens):
            # Try bigram first
            if self.use_bigrams and i + 1 < len(tokens):
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                polarity = self._lookup(bigram)
                if polarity is not None:
                    polarities.append(polarity)
                    concepts_found.append(bigram)
                    i += 2
                    continue

            # Fall back to unigram
            polarity = self._lookup(tokens[i])
            if polarity is not None:
                polarities.append(polarity)
                concepts_found.append(tokens[i])
            i += 1

        if not polarities:
            logger.debug("No SenticNet concepts found in: %r", text)
            return SenticResult("neutral", 0.0, [])

        avg_polarity = sum(polarities) / len(polarities)
        label = self._label(avg_polarity)
        return SenticResult(label, round(avg_polarity, 4), concepts_found)

    def predict_batch(self, texts: list[str]) -> list[SenticResult]:
        """Predict sentiment for a list of strings."""
        return [self.predict(t) for t in texts]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_senticnet(self):
        """Import SenticNet; return None if not installed (graceful degradation)."""
        try:
            from senticnet.senticnet import SenticNet
            return SenticNet()
        except ImportError:
            logger.warning(
                "senticnet package not found. Install with: pip install senticnet\n"
                "SenticNetAnalyzer will return neutral for all inputs."
            )
            return None

    def _lookup(self, concept: str) -> Optional[float]:
        """Return the polarity for a concept string, or None if not found."""
        if self._sn is None:
            return None
        try:
            polarity = self._sn.polarity_value(concept)
            return float(polarity)
        except Exception:
            return None

    def _label(self, polarity: float) -> str:
        if polarity >= self.pos_threshold:
            return "positive"
        if polarity <= self.neg_threshold:
            return "negative"
        return "neutral"
