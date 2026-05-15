"""
Unit tests for WordNetAnalyzer.

Run with:  pytest tests/test_wordnet.py -v
"""

import pytest
from wordnet.analyzer import WordNetAnalyzer, SentimentResult


@pytest.fixture(scope="module")
def analyzer():
    return WordNetAnalyzer()


class TestWordNetAnalyzer:

    def test_returns_sentiment_result(self, analyzer):
        result = analyzer.predict("The movie was great.")
        assert isinstance(result, SentimentResult)

    def test_clearly_positive(self, analyzer):
        result = analyzer.predict("Absolutely wonderful, fantastic, and joyful experience!")
        assert result.label == "positive"
        assert result.score > 0

    def test_clearly_negative(self, analyzer):
        result = analyzer.predict("Terrible, horrible, dreadful, and awful performance.")
        assert result.label == "negative"
        assert result.score < 0

    def test_neutral_empty_like(self, analyzer):
        # Only stopwords/punctuation → no scoreable tokens → neutral
        result = analyzer.predict("the and a in")
        assert result.label == "neutral"
        assert result.score == 0.0

    def test_score_in_range(self, analyzer):
        for text in [
            "I love this.",
            "I hate this.",
            "This is fine.",
            "Outstanding brilliance meets dismal failure.",
        ]:
            result = analyzer.predict(text)
            # Per-word scores each in [0,1]; net stays within reasonable bounds
            assert -2.0 <= result.score <= 2.0, f"Score out of range for: {text!r}"

    def test_batch_returns_correct_length(self, analyzer):
        texts = ["Good.", "Bad.", "Okay."]
        results = analyzer.predict_batch(texts)
        assert len(results) == 3

    def test_batch_types(self, analyzer):
        results = analyzer.predict_batch(["Excellent!", "Terrible!"])
        for r in results:
            assert isinstance(r, SentimentResult)
            assert r.label in ("positive", "negative", "neutral")

    def test_method_field(self, analyzer):
        result = analyzer.predict("Nice.")
        assert result.method == "wordnet"

    def test_custom_threshold(self):
        strict = WordNetAnalyzer(pos_threshold=0.5, neg_threshold=-0.5)
        result = strict.predict("somewhat good")
        # With high threshold even mildly positive text may be neutral
        assert result.label in ("positive", "neutral")

    def test_str_representation(self, analyzer):
        result = analyzer.predict("Great!")
        s = str(result)
        assert "wordnet" in s
        assert any(lbl in s for lbl in ("positive", "negative", "neutral"))

    def test_pos_neg_scores_non_negative(self, analyzer):
        result = analyzer.predict("This is very good.")
        assert result.pos_score >= 0.0
        assert result.neg_score >= 0.0

    def test_empty_string(self, analyzer):
        result = analyzer.predict("")
        assert result.label == "neutral"
        assert result.score == 0.0
