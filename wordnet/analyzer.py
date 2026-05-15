"""
WordNet / SentiWordNet sentiment analyzer.

Approach
--------
1. POS-tag each token (NLTK averaged perceptron tagger).
2. Map Penn Treebank tags → WordNet POS constants.
3. Lemmatize each content word.
4. Look up the first SentiWordNet synset for (lemma, pos).
5. Aggregate (pos_score - neg_score) across all content words.
6. Threshold the aggregate to produce a label.

Strengths:  fast, interpretable, no internet/GPU needed.
Weaknesses: ignores word order, negation, and context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Download required NLTK corpora once
_REQUIRED = [
    ("corpora/wordnet",               "wordnet"),
    ("corpora/sentiwordnet",          "sentiwordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("tokenizers/punkt",              "punkt"),
]

def _ensure_nltk_data() -> None:
    for path, name in _REQUIRED:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_ensure_nltk_data()

# Penn Treebank → WordNet POS
_POS_MAP = {
    "J": wn.ADJ,
    "V": wn.VERB,
    "N": wn.NOUN,
    "R": wn.ADV,
}


@dataclass
class SentimentResult:
    label: str          # 'positive' | 'negative' | 'neutral'
    score: float        # net polarity  [-1.0, 1.0]
    pos_score: float    # raw positive accumulation
    neg_score: float    # raw negative accumulation
    method: str = "wordnet"

    def __str__(self) -> str:
        sign = "+" if self.score >= 0 else ""
        return f"[{self.method}] {self.label:>8}  score={sign}{self.score:.3f}"


class WordNetAnalyzer:
    """
    Lexicon-based sentiment analyzer using SentiWordNet 3.0.

    Parameters
    ----------
    pos_threshold : float
        Minimum net score to be labelled 'positive'. Default 0.05.
    neg_threshold : float
        Maximum (most negative) net score to be labelled 'negative'. Default -0.05.
    use_first_synset_only : bool
        If True (default), use only the most common synset per word.
        If False, average over all matching synsets (slower, sometimes noisier).
    """

    def __init__(
        self,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
        use_first_synset_only: bool = True,
    ) -> None:
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.use_first_synset_only = use_first_synset_only
        self._lemmatizer = WordNetLemmatizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> SentimentResult:
        """Return a SentimentResult for a single string."""
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)

        pos_total = 0.0
        neg_total = 0.0
        n_scored = 0

        for word, tag in tagged:
            wn_pos = _POS_MAP.get(tag[0])
            if wn_pos is None:
                continue  # skip punctuation, determiners, etc.

            lemma = self._lemmatizer.lemmatize(word, pos=wn_pos)
            synsets = list(swn.senti_synsets(lemma, wn_pos))
            if not synsets:
                continue

            if self.use_first_synset_only:
                syn = synsets[0]
                pos_total += syn.pos_score()
                neg_total += syn.neg_score()
            else:
                pos_total += sum(s.pos_score() for s in synsets) / len(synsets)
                neg_total += sum(s.neg_score() for s in synsets) / len(synsets)

            n_scored += 1

        if n_scored == 0:
            logger.debug("No scoreable tokens in: %r", text)
            return SentimentResult("neutral", 0.0, 0.0, 0.0)

        net = (pos_total - neg_total) / n_scored
        label = self._label(net)
        return SentimentResult(label, round(net, 4), round(pos_total / n_scored, 4), round(neg_total / n_scored, 4))

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Predict sentiment for a list of strings."""
        return [self.predict(t) for t in texts]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _label(self, score: float) -> str:
        if score >= self.pos_threshold:
            return "positive"
        if score <= self.neg_threshold:
            return "negative"
        return "neutral"
