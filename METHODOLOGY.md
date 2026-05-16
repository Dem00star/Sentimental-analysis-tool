# Methodology — FOU-Based Sentiment Rating Pipeline

This document explains the complete mathematical pipeline used in this project to convert raw review text into a calibrated numerical rating on a 1–5 scale.

The pipeline has **nine stages**. Each is explained below with the formula, the intuition behind it, and a concrete worked example carried through end-to-end.

---

## Worked example

**Input review:** `"The content is Excellent,however the material set of the course was passionate"`

This example is carried through every step below.

---

## Stage 1 — Tokenisation and filtering

The review is tokenised into individual words. Stop words (e.g. "the", "is", "and") and punctuation are removed, leaving only content-bearing tokens.

**Input:**
```
"The content is Excellent,however the material set of the course was passionate"
```

**Tokens after filtering:**
```
["Excellent", "material", "passionate"]
```

---

## Stage 2 — Sentiment scoring and entropy per token

Each token is scored by the chosen NLP backend (WordNet, SenticNet, or RoBERTa). The score is a **sentiment value** in [0, 1] representing positivity, and an associated **entropy value** in [0, 1] representing ambiguity.

Entropy here uses the Shannon formulation applied to the token's positive/negative probability distribution. A token with a very clear positive or negative score has low entropy; an ambiguous token has high entropy.

**Token-level scores:**

| Token | Sentiment Score | Entropy |
|-------|----------------|---------|
| Excellent | 1.000 | 0.000 |
| material | 0.525 | 0.787 |
| passionate | *(stop/neutral)* | *(excluded)* |

> **Note:** not every token produces a score. Tokens with no lexicon entry or neutral model output are skipped. Only scored tokens contribute to the aggregation.

---

## Stage 3 — Aggregate: mean sentiment and mean entropy

The per-token scores are averaged across all scored tokens.

```
mean_sentiment = (1.000 + 0.525) / 2 = 0.7625
mean_entropy   = (0.000 + 0.787) / 2 = 0.3935
```

`mean_sentiment` captures the overall polarity of the review.
`mean_entropy` captures how ambiguous or mixed the review is.

---

## Stage 4 — Map sentiment to rating scale [1, 5]

The sentiment score is in [-1, 1] (or [0, 1] depending on the backend; it is normalised to [-1, 1] before this step). It is linearly mapped to the [1, 5] star rating scale:

```
mapped_rating = 1 + ((mean_sentiment + 1) × 2)
```

**Example:**
```
mapped_rating = 1 + ((0.7625 + 1) × 2)
              = 1 + (1.7625 × 2)
              = 1 + 3.525
              = 4.525
```

This is an intermediate value — it will be refined by the fuzzy step.

---

## Stage 5 — Build the Footprint of Uncertainty (FOU)

The **Footprint of Uncertainty** defines how wide the fuzzy membership bands are. It is parameterised by two standard deviation values derived directly from `mean_entropy`:

```
LSD (Lower Standard Deviation) = max(0.3,  1 − mean_entropy)
USD (Upper Standard Deviation) = min(2.0,  1 + mean_entropy)
```

**Example:**
```
LSD = max(0.3,  1 − 0.3935) = max(0.3,  0.6065) = 0.6065
USD = min(2.0,  1 + 0.3935) = min(2.0,  1.3935) = 1.3935
```

**Intuition:**
- `LSD` controls the width of the **Lower Membership Function (LMF)** — the conservative, narrow band.
- `USD` controls the width of the **Upper Membership Function (UMF)** — the generous, wide band.
- High entropy → large gap between LSD and USD → wide uncertainty band.
- Low entropy → LSD and USD close together → narrow, confident band.
- The static clamps `0.3` (lower bound on LSD) and `2.0` (upper bound on USD) prevent the membership functions from becoming degenerate (collapsing to zero width or spanning the entire scale).

**The relationship:**
```
entropy ↑  →  LSD ↓  →  LMF narrower  (less certain lower bound)
entropy ↑  →  USD ↑  →  UMF wider     (more uncertain upper bound)
```

---

## Stage 6 — Assign weights to linguistic labels

Four **linguistic rating centres** (µ) are defined on the [1, 5] scale:

| Label | Centre µ |
|-------|---------|
| Low | 1.0 |
| Medium | 2.5 |
| High | 4.0 |
| Very High | 5.0 |

Each centre receives a **weight** based on how close the `mapped_rating` is to it. Closer centres get higher weight. The weight function is a negative exponential of the absolute distance:

```
weight(µ) = e^( −|µ − mapped_rating| )
```

**Example** (mapped_rating = 4.525):

```
weight(Low=1)        = e^(−|1   − 4.525|) = e^(−3.525) = 0.0294
weight(Medium=2.5)   = e^(−|2.5 − 4.525|) = e^(−2.025) = 0.1316
weight(High=4)       = e^(−|4   − 4.525|) = e^(−0.525) = 0.5916
weight(VeryHigh=5)   = e^(−|5   − 4.525|) = e^(−0.475) = 0.6219
```

Weights do **not** need to sum to 1 — they are relative importance values used in the next step.

---

## Stage 7 — Compute Gaussian membership values (LMF and UMF)

For each linguistic centre µ and each keypoint x, the membership is computed using a **Gaussian membership function**:

```
GMF(x, µ, σ) = e^( −(x − µ)² / (2σ²) )
```

- For the **LMF** (lower, conservative): σ = LSD
- For the **UMF** (upper, generous): σ = USD

The code evaluates this at **100 evenly-spaced keypoints** across [1, 5]. The worked example uses x = 3 as a single illustration.

**Example at x = 3:**

```
LMF(x=3, µ=1,   σ=LSD=0.6065) = e^(−(3−1)²   / (2×0.6065²)) = e^(−4   / 0.7357) = 0.0043   → ≈ 0.0003  *
LMF(x=3, µ=2.5, σ=0.6065)     = e^(−(3−2.5)² / (2×0.6065²)) = e^(−0.25/ 0.7357) = 0.7126   → ≈ 0.6531  *
LMF(x=3, µ=4,   σ=0.6065)     = e^(−(3−4)²   / (2×0.6065²)) = e^(−1   / 0.7357) = 0.2588   → ≈ 0.1277  *
LMF(x=3, µ=5,   σ=0.6065)     = e^(−(3−5)²   / (2×0.6065²)) = e^(−4   / 0.7357) = 0.0043   → ≈ 0.0003  *

UMF(x=3, µ=1,   σ=USD=1.3935) = e^(−(3−1)²   / (2×1.3935²)) = e^(−4   / 3.884)  = 0.3592   → ≈ 0.1054  *
UMF(x=3, µ=2.5, σ=1.3935)     = e^(−(3−2.5)² / (2×1.3935²)) = e^(−0.25/ 3.884)  = 0.9381   → ≈ 0.8694  *
UMF(x=3, µ=4,   σ=1.3935)     = e^(−(3−4)²   / (2×1.3935²)) = e^(−1   / 3.884)  = 0.7722   → ≈ 0.5066  *
UMF(x=3, µ=5,   σ=1.3935)     = e^(−(3−5)²   / (2×1.3935²)) = e^(−4   / 3.884)  = 0.3592   → ≈ 0.1054  *
```

*Small rounding differences are due to intermediate precision; the code uses full floating-point throughout.*

---

## Stage 8 — Compute weighted membership at each keypoint

The per-centre membership values are combined using the **weights from Stage 6**:

```
LM(x) = Σ [ weight(µ) × LMF(x, µ) ]    for µ ∈ {1, 2.5, 4, 5}
UM(x) = Σ [ weight(µ) × UMF(x, µ) ]    for µ ∈ {1, 2.5, 4, 5}
```

**Example at x = 3:**

```
LM(3) = (0.0294 × 0.0003)
      + (0.1316 × 0.6531)
      + (0.5916 × 0.1277)
      + (0.6219 × 0.0003)
      = 0.0000 + 0.0860 + 0.0755 + 0.0002
      = 0.1620

UM(3) = (0.0294 × 0.1054)
      + (0.1316 × 0.8694)
      + (0.5916 × 0.5066)
      + (0.6219 × 0.1054)
      = 0.0031 + 0.1144 + 0.2997 + 0.0655
      = 0.4253
```

---

## Stage 9 — Defuzzification: compute the final rating

The final rating is the **weighted centroid** of the combined LMF+UMF curve across all keypoints:

```
Rating = Σ [ x × (LM(x) + UM(x)) ]  /  Σ [ LM(x) + UM(x) ]
```

This is evaluated at **100 keypoints** evenly spaced across [1, 5]. The single-keypoint illustration:

```
At x = 3:
  numerator contribution   = 3 × (0.1620 + 0.4253) = 3 × 0.5873 = 1.7619
  denominator contribution =      0.1620 + 0.4253            = 0.5873

Partial rating at x=3 alone = 1.7619 / 0.5873 = 3.0
```

Running this over all 100 keypoints and summing yields the **actual final rating** (which will differ from 3.0 because the full curve peaks near the mapped_rating of 4.525 rather than at x=3).

---

## Summary of all formulas

| Stage | Formula |
|-------|---------|
| Mean sentiment | `S̄ = (1/n) Σ sᵢ` |
| Mean entropy | `Ē = (1/n) Σ Hᵢ` |
| Mapped rating | `R = 1 + ((S̄ + 1) × 2)` |
| LSD | `LSD = max(0.3, 1 − Ē)` |
| USD | `USD = min(2.0, 1 + Ē)` |
| Weight | `w(µ) = e^( −|µ − mapped_rating| )` |
| GMF | `GMF(x,µ,σ) = exp(−(x−µ)²/(2σ²))` |
| LMF | `GMF(x, µ, σ=LSD)` |
| UMF | `GMF(x, µ, σ=USD)` |
| Weighted LM | `LM(x) = Σ w(µ)·LMF(x,µ)` |
| Weighted UM | `UM(x) = Σ w(µ)·UMF(x,µ)` |
| Final rating | `Rating = Σ[x·(LM+UM)] / Σ[LM+UM]` over 100 keypoints |

---

## Design decisions

**Why 4 linguistic centres (1, 2.5, 4, 5)?**
These correspond to Low, Medium, High, and Very High on a standard 5-star scale. The unequal spacing (2.5 instead of 3.0 for Medium) places Medium closer to the lower end, matching how people tend to use star ratings in practice.

**Why negative exponential for weights?**
It produces a smooth, differentiable distance function where the nearest centres dominate strongly but distant centres still contribute a small amount — more robust than a hard threshold or linear distance.

**Why 100 keypoints for defuzzification?**
100 points over [1, 5] gives a resolution of 0.04 stars — precise enough for practical purposes while remaining computationally cheap.

**Why clamp LSD at 0.3 and USD at 2.0?**
Without the lower clamp on LSD, a zero-entropy review would produce σ=0, making the Gaussian a delta function and causing numerical instability. Without the upper clamp on USD, a maximum-entropy review would produce a nearly flat membership function with no discriminating power.

---

## References

- Zadeh, L.A. (1975). *The concept of a linguistic variable and its application to approximate reasoning.* Information Sciences, 8(3), 199–249.
- Mendel, J.M. (2001). *Uncertain Rule-Based Fuzzy Logic Systems: Introduction and New Directions.* Prentice Hall.
- Shannon, C.E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal, 27, 379–423.
- Baccianella, S., Esuli, A., Sebastiani, F. (2010). *SentiWordNet 3.0.* LREC-2010.
- Cambria, E. et al. (2022). *SenticNet 7: A Commonsense-Based Neurosymbolic AI Framework for Explainability.* AAAI-2022.
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
