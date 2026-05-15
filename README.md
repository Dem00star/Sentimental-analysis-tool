# Sentiment Analysis Tool

A Python toolkit that compares **three different NLP approaches** to sentiment analysis side by side — so you can see exactly how lexicon-based, knowledge-graph, and transformer methods differ on the same input.

| Module | Method | Approach | Speed |
|--------|--------|----------|-------|
| `wordnet/` | SentiWordNet | Lexicon-based | ⚡ Fastest |
| `senticnett/` | SenticNet | Knowledge-graph | ⚡⚡ Medium |
| `rober/` | RoBERTa | Transformer (deep learning) | 🐢 Slowest, most accurate |

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/Dem00star/Sentimental-analysis-tool.git
cd Sentimental-analysis-tool

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install nltk transformers torch senticnet pandas scikit-learn matplotlib seaborn

# 4. Download NLTK data (one-time setup)
python -c "
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
"

# 5. Open the comparison notebook
jupyter notebook notebooks/comparison.ipynb
```

---

## Project structure

```
Sentimental-analysis-tool/
│
├── wordnet/              # Lexicon-based sentiment (SentiWordNet + NLTK)
├── senticnett/           # Knowledge-graph sentiment (SenticNet 6.0)
├── rober/                # Transformer sentiment (RoBERTa via HuggingFace)
│
├── data/                 # Datasets — put your CSV files here
│                         # Expected format: columns 'text' and 'label'
│
├── notebooks/
│   └── comparison.ipynb  # Benchmark all three methods on the same dataset
│
├── .gitignore
└── README.md
```

---

## How each method works

### WordNet / SentiWordNet (`wordnet/`)
Looks up each word in the SentiWordNet 3.0 lexicon. Every word gets a positive and negative score; the net average across the sentence decides the label. Fast, interpretable, and works offline — but can't understand context, negation, or word order.

### SenticNet (`senticnett/`)
Uses the SenticNet 6.0 knowledge base, which maps words and multi-word phrases to polarity values using commonsense reasoning. Handles expressions like "not bad" better than WordNet. Requires the `senticnet` Python package.

### RoBERTa (`rober/`)
Uses the [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) model from HuggingFace — a RoBERTa model fine-tuned on 58 million tweets. Context-aware, handles negation and sarcasm. ~500MB model download on first run. GPU recommended for large batches.

---

## Data format

Place your CSV files in the `data/` folder. The notebook expects two columns:

| Column | Type | Example |
|--------|------|---------|
| `text` | string | `"This product is amazing!"` |
| `label` | string | `positive` / `negative` / `neutral` |

---

## Running the benchmark

Open `notebooks/comparison.ipynb` and run all cells. The notebook will:

1. Load your dataset from `data/`
2. Run all three methods on every row
3. Print accuracy and F1 scores per method
4. Plot confusion matrices (saved as `confusion_matrices.png`)
5. Plot a speed comparison chart (saved as `speed_comparison.png`)
6. Show samples where the three methods disagree

---

## Requirements

- Python 3.8+
- `nltk` — WordNet and SentiWordNet
- `transformers` + `torch` — RoBERTa model
- `senticnet` — SenticNet knowledge base
- `pandas`, `scikit-learn`, `matplotlib`, `seaborn` — data and evaluation
- `jupyter` — to run the notebook

Install everything at once:
```bash
pip install nltk transformers torch senticnet pandas scikit-learn matplotlib seaborn jupyter
```

> **Note on RoBERTa**: the model weights (~500MB) are downloaded automatically from HuggingFace on the first run and cached in `~/.cache/huggingface`. No manual download needed.

---

## License

MIT License — free to use, modify, and distribute.
