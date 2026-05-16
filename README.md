# Sentiment Analysis Tool

A Python toolkit that converts raw text reviews into **numerical star ratings (1–5)** using a novel pipeline combining three NLP backends with **Fuzzy Type-2 Interval Membership Functions** and **Shannon entropy** to model uncertainty in sentiment.

> Rather than simply labelling text as positive/negative, this tool produces a calibrated rating that reflects *how confident* the sentiment signal is — using token-level entropy to widen or narrow the fuzzy membership bands accordingly.

---

## Pipeline overview

```
Raw Review Text
      │
      ▼
  Tokenisation & Stop-word Filtering
      │
      ▼
  Sentiment Scoring per token        ──── Three parallel backends:
  + Shannon Entropy per token             wordnet/  |  senticnett/  |  rober/
      │
      ▼
  Aggregate:  mean_sentiment,  mean_entropy
      │
      ▼
  Map to rating scale [1–5]
  mapped_rating = 1 + ((mean_sentiment + 1) × 2)
      │
      ▼
  Build FOU (Footprint of Uncertainty)
  LSD = max(0.3,  1 − mean_entropy)
  USD = min(2.0,  1 + mean_entropy)
      │
      ▼
  Compute Gaussian Membership (LMF + UMF)
  across 4 linguistic centres: Low(1), Medium(2.5), High(4), Very High(5)
  weighted by proximity of each centre to mapped_rating
      │
      ▼
  Defuzzify over 100 keypoints across [1,5]
  rating = Σ[x·(LM(x)+UM(x))] / Σ[LM(x)+UM(x)]
      │
      ▼
  Final Rating  (1.0 – 5.0)
```

For the full mathematical derivation with a complete worked example, see [`METHODOLOGY.md`](METHODOLOGY.md).

---

## Modules

| Module | NLP Backend | Approach | Speed |
|--------|------------|----------|-------|
| `SentiWordNet/` | SentiWordNet 3.0 | Lexicon-based | ⚡ Fastest |
| `SenticNet/` | SenticNet 6.0 | Knowledge-graph | ⚡⚡ Medium |
| `RoBERTa/` | RoBERTa (HuggingFace) | Transformer | 🐢 Slowest, most accurate |

All three backends feed into the **same** FOU-based rating pipeline — so you can compare how the NLP backend choice affects the final defuzzified rating.

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
pip install nltk transformers torch senticnet pandas numpy scikit-learn matplotlib seaborn

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
├── wordnet/              # Lexicon-based sentiment scorer (SentiWordNet + NLTK)
├── senticnett/           # Knowledge-graph sentiment scorer (SenticNet 6.0)
├── rober/                # Transformer sentiment scorer (RoBERTa via HuggingFace)
│
├── data/                 # Place your CSV datasets here
│                           Required columns: 'text', 'label'
│
├── notebooks/
│   └── comparison.ipynb  # End-to-end benchmark across all three backends
│
├── METHODOLOGY.md        # Full mathematical explanation of the FOU pipeline
├── README.md
└── .gitignore
```

---

## Key concepts

### Entropy as uncertainty measure
Sentiment scores from lexicons carry varying levels of certainty. A review token like *"Excellent"* has a strong, unambiguous score (low entropy). An ambiguous token like *"material"* has a weaker signal (higher entropy). Shannon entropy across all scored tokens is used to measure *how uncertain* the overall review signal is.

### Fuzzy Type-2 membership
Standard (Type-1) fuzzy sets assign a single crisp membership value. **Interval Type-2 fuzzy sets** instead assign a *band* — a Lower Membership Function (LMF) and an Upper Membership Function (UMF). This band directly encodes uncertainty: high-entropy reviews produce wide bands; confident reviews produce narrow ones.

### FOU parameters
```
LSD = max(0.3,  1 − mean_entropy)   # Lower Standard Deviation → LMF width
USD = min(2.0,  1 + mean_entropy)   # Upper Standard Deviation → UMF width
```
The static clamps `0.3` and `2.0` prevent membership functions from collapsing (too narrow) or spanning the entire scale (too wide).

### Defuzzification
The final rating is computed as a weighted centroid across 100 evenly-spaced keypoints over [1, 5], using the sum of LMF and UMF membership values as weights. This produces a continuous, interpretable rating rather than a discrete label.

---

## Data format

CSV files placed in `data/` should have:

| Column | Type | Example |
|--------|------|---------|
| `text` | string | `"Excellent material, very passionate"` |
| `label` | string | `positive` / `negative` / `neutral` |

---

## Requirements

- Python 3.8+
- `nltk` — tokenisation and SentiWordNet
- `transformers` + `torch` — RoBERTa model
- `senticnet` — SenticNet knowledge base
- `pandas`, `numpy`, `scikit-learn` — data handling and evaluation
- `matplotlib`, `seaborn` — visualisation
- `jupyter` — notebook support

```bash
pip install nltk transformers torch senticnet pandas numpy scikit-learn matplotlib seaborn jupyter
```

> **Note on RoBERTa:** model weights (~500 MB) are downloaded automatically from HuggingFace on first run and cached in `~/.cache/huggingface`.

---

## References

- Zadeh, L.A. (1975). *The concept of a linguistic variable and its application to approximate reasoning.* Information Sciences.
- Mendel, J.M. (2001). *Uncertain Rule-Based Fuzzy Logic Systems.* Prentice Hall.
- Baccianella, Esuli, Sebastiani (2010). *SentiWordNet 3.0.* LREC.
- Cambria et al. (2022). *SenticNet 7.* AAAI.
- Liu et al. (2019). *RoBERTa.* arXiv:1907.11692.

---

## License

MIT License — free to use, modify, and distribute.
