"""
Unit tests for RoBERTaAnalyzer.

These tests use unittest.mock to avoid downloading the model during CI.
To run real inference tests: pytest tests/test_roberta.py -v --real-model

Run with:  pytest tests/test_roberta.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from rober.analyzer import RoBERTaAnalyzer, RoBERTaResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_pipeline(label: str = "LABEL_2", score: float = 0.97):
    """Return a mock HuggingFace pipeline that yields predictable output."""
    single_out = [{"label": label, "score": score}]
    all_scores_out = [
        [
            {"label": "LABEL_0", "score": 0.01},
            {"label": "LABEL_1", "score": 0.02},
            {"label": "LABEL_2", "score": 0.97},
        ]
    ]

    mock = MagicMock()
    # Single call (no return_all_scores) → list of single dicts
    mock.side_effect = lambda text, **kwargs: (
        all_scores_out if isinstance(text, list) or kwargs.get("return_all_scores") else single_out
    )
    return mock


@pytest.fixture()
def analyzer_mocked():
    """RoBERTaAnalyzer with the HF pipeline mocked out."""
    with patch("rober.analyzer.RoBERTaAnalyzer._get_pipeline") as mock_get:
        mock_get.return_value = _make_mock_pipeline()
        a = RoBERTaAnalyzer.__new__(RoBERTaAnalyzer)
        a.model_name = "mock"
        a.batch_size = 32
        a.max_length = 512
        a._device = "cpu"
        a._pipeline = mock_get.return_value
        yield a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRoBERTaResult:
    def test_str_contains_method(self):
        r = RoBERTaResult("POSITIVE", 0.97, {"POSITIVE": 0.97, "NEUTRAL": 0.02, "NEGATIVE": 0.01})
        assert "roberta" in str(r)

    def test_str_contains_label(self):
        r = RoBERTaResult("NEGATIVE", 0.88, {"NEGATIVE": 0.88, "NEUTRAL": 0.1, "POSITIVE": 0.02})
        assert "NEGATIVE" in str(r)


class TestRoBERTaAnalyzer:

    def test_predict_returns_result(self, analyzer_mocked):
        result = analyzer_mocked.predict("This is amazing!")
        assert isinstance(result, RoBERTaResult)

    def test_label_is_valid(self, analyzer_mocked):
        result = analyzer_mocked.predict("Some text.")
        assert result.label in ("POSITIVE", "NEGATIVE", "NEUTRAL")

    def test_confidence_in_range(self, analyzer_mocked):
        result = analyzer_mocked.predict("Test.")
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_keys_complete(self, analyzer_mocked):
        result = analyzer_mocked.predict("Test.")
        assert set(result.scores.keys()) == {"POSITIVE", "NEGATIVE", "NEUTRAL"}

    def test_scores_sum_to_one(self, analyzer_mocked):
        result = analyzer_mocked.predict("Test.")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01

    def test_method_field(self, analyzer_mocked):
        result = analyzer_mocked.predict("Test.")
        assert result.method == "roberta"

    def test_batch_length(self, analyzer_mocked):
        # Mock pipeline returns a list of all_scores for a batch
        batch_out = [
            [
                {"label": "LABEL_0", "score": 0.01},
                {"label": "LABEL_1", "score": 0.02},
                {"label": "LABEL_2", "score": 0.97},
            ]
        ] * 3
        analyzer_mocked._pipeline.side_effect = lambda texts, **kw: batch_out

        texts = ["Good.", "Bad.", "Meh."]
        results = analyzer_mocked.predict_batch(texts)
        assert len(results) == 3

    def test_device_resolution_cpu(self):
        device = RoBERTaAnalyzer._resolve_device("cpu")
        assert device == "cpu"

    def test_device_resolution_explicit(self):
        device = RoBERTaAnalyzer._resolve_device("cuda:0")
        assert device == "cuda:0"

    def test_import_error_raised_gracefully(self):
        """If transformers is missing, ImportError should be clear and helpful."""
        with patch.dict("sys.modules", {"transformers": None}):
            a = RoBERTaAnalyzer.__new__(RoBERTaAnalyzer)
            a._pipeline = None
            a.model_name = "mock"
            a._device = "cpu"
            a.batch_size = 32
            a.max_length = 512

            with pytest.raises(ImportError, match="transformers"):
                a._get_pipeline()
