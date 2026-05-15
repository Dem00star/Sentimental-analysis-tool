# Sentiment Analysis Tool

A Python toolkit that benchmarks **three fundamentally different NLP approaches** to sentiment analysis on the same dataset, so you can directly compare their accuracy, speed, and tradeoffs.

| Module | Approach | Requires training? | Best for |
|---|---|---|---|
| `wordnet/` | Lexicon-based (SentiWordNet) | No | Interpretable baselines, any domain |
| `senticnett/` | Knowledge-graph (SenticNet) | No | Commonsense, multi-word expressions |
| `rober/` | Transformer (RoBERTa) | No (pretrained) | Best accuracy, social media text |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Dem00star/Sentimental-analysis-tool.git
cd Sentimental-analysis-tool

# 2. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('wordnet'); nltk.download('sentiwordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt')"

# 5. Run a quick demo
python main.py --text "This movie was absolutely fantastic!"
```

Expected output:
```
Input : "This movie was absolutely fantastic!"
WordNet  : positive  (score:  0.625)
SenticNet: positive  (polarity: 0.72)
RoBERTa  : POSITIVE  (confidence: 98.3%)
```

---

## Project structure

```
.
├── main.py                  # CLI entry point — run any or all methods
├── requirements.txt         # Pinned dependencies
├── setup.py                 # Package install config
├── data/
│   ├── sample_reviews.csv   # Example dataset (text, label columns)
│   └── README.md
├── wordnet/
│   ├── __init__.py
│   └── analyzer.py          # SentiWordNet lexicon-based scorer
├── senticnett/
│   ├── __init__.py
│   └── analyzer.py          # SenticNet knowledge-graph scorer
├── rober/
│   ├── __init__.py
│   └── analyzer.py          # RoBERTa transformer scorer
├── notebooks/
│   └── comparison.ipynb     # End-to-end benchmark across all three methods
└── tests/
    ├── test_wordnet.py
    ├── test_senticnet.py
    └── test_roberta.py
```

---

## Usage

### CLI

```bash
# Single sentence
python main.py --text "The service was terrible and slow."

# CSV file — runs all three methods and saves results
python main.py --file data/sample_reviews.csv --output results.csv

# Choose specific methods
python main.py --text "Great product!" --methods wordnet roberta

# Show scores (not just labels)
python main.py --text "It was okay I guess." --verbose
```

### Python API

```python
from wordnet.analyzer   import WordNetAnalyzer
from senticnett.analyzer import SenticNetAnalyzer
from rober.analyzer      import RoBERTaAnalyzer

text = "The film had breathtaking visuals but a weak storyline."

wn  = WordNetAnalyzer()
sn  = SenticNetAnalyzer()
rb  = RoBERTaAnalyzer()

print(wn.predict(text))   # {'label': 'positive', 'score': 0.41}
print(sn.predict(text))   # {'label': 'mixed',    'polarity': 0.12}
print(rb.predict(text))   # {'label': 'POSITIVE', 'confidence': 0.71}
```

---

## Methods explained

### WordNet / SentiWordNet
Uses NLTK's POS tagger and the SentiWordNet 3.0 lexicon. Each word is lemmatized, looked up in WordNet synsets, and scored for positivity/negativity. Simple, fast, and fully interpretable — but blind to word order and context.

### SenticNet
Uses the SenticNet 6.0 knowledge base, which maps concepts (including multi-word phrases like "not good") to polarity values using commonsense reasoning. Handles negation and compound expressions better than pure WordNet.

### RoBERTa
Uses the `cardiffnlp/twitter-roberta-base-sentiment` model from HuggingFace — a RoBERTa model fine-tuned on ~58M tweets. Context-aware, handles slang, negation, and sarcasm. Slowest but most accurate. GPU recommended for batch processing.

---

## Benchmark results (SST-2 test set)

| Method | Accuracy | F1 | Avg time/sample |
|---|---|---|---|
| WordNet | 67.2% | 0.64 | ~2ms |
| SenticNet | 72.4% | 0.71 | ~5ms |
| RoBERTa | 94.1% | 0.94 | ~45ms (CPU) |

*Run `notebooks/comparison.ipynb` to reproduce on your own dataset.*

---

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list
- RoBERTa module requires ~500MB disk for model weights (auto-downloaded on first use)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests welcome. Please open an issue first to discuss major changes. Run `pytest tests/` before submitting.
