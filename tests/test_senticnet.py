"""
Unit tests for SenticNetAnalyzer.

Run with:  pytest tests/test_senticnet.py -v

Note: if senticnet is not installed, all tests gracefully check for
the neutral fallback rather than failing hard.
"""

import pytest
from senticnett.analyzer import SenticNetAnalyzer, SenticResult


@pytest.fixture(scope="module")
def analyzer():
    return SenticNetAnalyzer()


class TestSenticNetAnalyzer:

    def test_returns_sentic_result(self, analyzer):
        result = analyzer.predict("I love pizza.")
        assert isinstance(result, SenticResult)

    def test_label_is_valid(self, analyzer):
        result = analyzer.predict("What a wonderful day!")
        assert result.label in ("positive", "negative", "neutral")

    def test_polarity_in_range(self, analyzer):
        texts = [
            "Fantastic service!",
            "Terrible experience.",
            "It was okay.",
            "I am so happy today.",
        ]
        for text in texts:
            result = analyzer.predict(text)
            assert -1.0 <= result.polarity <= 1.0, f"Polarity out of range for: {text!r}"

    def test_method_field(self, analyzer):
        result = analyzer.predict("Nice.")
        assert result.method == "senticnet"

    def test_concepts_found_is_list(self, analyzer):
        result = analyzer.predict("Very happy and joyful.")
        assert isinstance(result.concepts_found, list)

    def test_empty_string(self, analyzer):
        result = analyzer.predict("")
        assert result.label == "neutral"
        assert result.polarity == 0.0
        assert result.concepts_found == []

    def test_batch_length(self, analyzer):
        texts = ["Great!", "Awful!", "Meh."]
        results = analyzer.predict_batch(texts)
        assert len(results) == 3

    def test_batch_types(self, analyzer):
        results = analyzer.predict_batch(["Excellent!", "Terrible!"])
        for r in results:
            assert isinstance(r, SenticResult)
            assert r.label in ("positive", "negative", "neutral")

    def test_str_representation(self, analyzer):
        result = analyzer.predict("Amazing!")
        s = str(result)
        assert "senticnet" in s

    def test_custom_threshold(self):
        strict = SenticNetAnalyzer(pos_threshold=0.5, neg_threshold=-0.5)
        result = strict.predict("somewhat good")
        assert result.label in ("positive", "neutral")

    def test_bigram_mode_toggle(self):
        no_bigram = SenticNetAnalyzer(use_bigrams=False)
        result = no_bigram.predict("not good")
        assert isinstance(result, SenticResult)

    def test_graceful_without_senticnet_package(self, monkeypatch):
        """If senticnet is not installed, analyzer returns neutral gracefully."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "senticnet.senticnet":
                raise ImportError("mocked missing package")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        analyzer = SenticNetAnalyzer()
        result = analyzer.predict("Good day!")
        assert result.label == "neutral"
