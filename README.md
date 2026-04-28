# Earnings Call Sentiment Analysis: FinBERT vs. TF-IDF Baselines

**Course:** CS-4267-02 Deep Learning · Vanderbilt University  
**Research Question:** Does transformer-based financial sentiment extracted from earnings call transcripts improve prediction of short-term abnormal stock returns compared to classical NLP baselines?

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Results Summary](#results-summary)
3. [Project Structure](#project-structure)
4. [Quickstart](#quickstart)
5. [Data](#data)
6. [Pipeline Architecture](#pipeline-architecture)
   - [FinBERT Pipeline](#finbert-pipeline)
   - [Baseline Pipeline](#baseline-pipeline)
7. [Evaluation](#evaluation)
8. [Shared Contract](#shared-contract)
9. [Running the Code](#running-the-code)
10. [Key Findings & Interpretation](#key-findings--interpretation)
11. [Known Gotchas](#known-gotchas)
12. [Dependencies](#dependencies)

---

## Project Overview

Earnings calls are quarterly events where company executives present financial results and guidance to analysts and investors. The language used — management tone, hedging phrases, analyst Q&A dynamics — may carry forward-looking signals that drive short-term market reactions.

This project tests whether **FinBERT** (a BERT model pre-trained on financial text) can extract actionable sentiment signals from raw earnings call transcripts to predict whether a stock will outperform or underperform the S&P 500 market return in the two and five trading days following an earnings call.

We compare two approaches:

- **FinBERT-based** (two-stage): FinBERT extracts sentiment scores or CLS embeddings → logistic regression predicts the label
- **TF-IDF baselines** (one-stage): TF-IDF vectorizes the transcript → Naive Bayes or Logistic Regression predicts the label directly

Labels are derived from **Cumulative Abnormal Return (CAR)** — the stock's return minus the S&P 500's return — over two windows: 2-day and 5-day.

---

## Results Summary

All models were evaluated on the same 20% held-out test set (`random_state=42`). The full dataset is ~33,400 transcripts; ~26,720 were used for training after yfinance fetch failures were dropped.

### ML Metrics

| Model | Label | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| FinBERT-Sentiment + LogReg | CAR₂ | 0.501 | 0.000 | 0.510 |
| FinBERT-Embedding + LogReg | CAR₂ | 0.511 | 0.499 | 0.510 |
| TF-IDF + Naive Bayes | CAR₂ | 0.513 | 0.480 | **0.516** |
| TF-IDF + Logistic Regression | CAR₂ | 0.512 | 0.514 | **0.516** |
| FinBERT-Sentiment + LogReg | CAR₅ | 0.512 | 0.678 | 0.507 |
| FinBERT-Embedding + LogReg | CAR₅ | 0.509 | 0.541 | 0.512 |
| TF-IDF + Naive Bayes | CAR₅ | 0.516 | 0.584 | **0.522** |
| TF-IDF + Logistic Regression | CAR₅ | 0.517 | 0.528 | **0.524** |

### Financial Metrics

| Model | Label | Avg CAR (predicted +) | Avg CAR (predicted −) | p-value | Sharpe |
|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | CAR₂ | +0.137% | −0.027% | **0.019** ✓ | 0.052 |
| TF-IDF + Logistic Regression | CAR₂ | +0.102% | −0.014% | 0.093 | 0.039 |
| TF-IDF + Naive Bayes | CAR₅ | +0.222% | −0.022% | **0.027** ✓ | 0.058 |
| TF-IDF + Logistic Regression | CAR₅ | +0.167% | +0.106% | 0.553 | 0.044 |

> FinBERT financial metrics were not available at the time of the full run. CAR₂ sentiment model collapsed to predicting only the majority class (F1 = 0.000).

### Key Finding

**FinBERT does not outperform TF-IDF baselines on this task.** All models achieve ROC-AUC ≈ 0.51–0.52, barely above random chance. TF-IDF baselines edge out FinBERT on both windows. The primary bottleneck for FinBERT is that FinBERT sentiment scores are dominated by neutral (mean ≈ 0.66 across all transcripts), leaving insufficient discriminative signal in the 3-feature bottleneck. The 768-dim embedding does only marginally better.

---

## Project Structure

```
.
├── shared/
│   ├── __init__.py
│   ├── load_data.py        ← Load & clean the HuggingFace transcript dataset
│   └── compute_car.py      ← CAR computation, label assignment, canonical train/test split
│
├── finbert/
│   ├── __init__.py
│   ├── inference.py        ← FinBERT sliding-window inference + prediction head
│   └── finbert_pipeline.ipynb  ← End-to-end Colab notebook (runs on A100/H100)
│
├── baselines/
│   ├── __init__.py
│   └── tfidf_models.py     ← TF-IDF + Naive Bayes / Logistic Regression
│
├── evaluation/
│   ├── __init__.py
│   └── compare_models.py   ← Financial metrics (t-test, Sharpe) + comparison table
│
├── pyproject.toml          ← uv-managed dependencies
├── baseline_run.txt        ← Saved baseline metrics output
└── comparison_run.txt      ← Saved full comparison table output
```

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/simongydvandy/Earnings-Call-Sentiment-Analysis-FinBERT-vs.-Naive-Bayes-LR-TF-IDF.git
cd Earnings-Call-Sentiment-Analysis-FinBERT-vs.-Naive-Bayes-LR-TF-IDF

# Install all dependencies (one command — Python 3.11)
uv sync

# Verify the environment
uv run python -c "from shared.load_data import load_transcripts; print('OK')"
```

> **No GPU required for the baseline pipeline.** FinBERT inference runs on Google Colab (A100/H100 recommended). Everything else runs locally via `uv run python`.

> **`transcripts_with_sentiment.csv` is not in the repo** (too large for git). This file is produced by the FinBERT inference step. To run baselines without running inference first, generate a small labeled sample: `build_labeled_dataset(df, sample_n=200)`.

---

## Data

### Transcripts

- **Source:** [`kurry/sp500_earnings_transcripts`](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts) (HuggingFace)
- **Size:** ~33,400 earnings call transcripts
- **Coverage:** 685 S&P 500 tickers, 2005–2025
- **Loading:** `shared/load_data.py` → `load_transcripts()` — handles column renaming, null drops, and index resetting

### Stock Prices & Labels

- **Source:** `yfinance` — downloads adjusted close prices for each ticker and `^GSPC` (S&P 500)
- **Market timing:** Day 0 is the **next trading day after the earnings call** (calls happen after market close)
- **Abnormal return:** `stock_daily_return − S&P500_daily_return` for each day in the window
- **CAR:** Cumulative sum of abnormal returns over the window

Two label columns are computed simultaneously from a single `yfinance` fetch per row:

| Column | Description |
|---|---|
| `CAR_2` | Cumulative abnormal return over days [0, +2] |
| `CAR_5` | Cumulative abnormal return over days [0, +5] |
| `label_2` | 1 if CAR_2 > 0, else 0 (class balance ≈ 54% / 46%) |
| `label_5` | 1 if CAR_5 > 0, else 0 (class balance ≈ 50% / 50%) |

The `shared/compute_car.py` module handles all of this, including silent failure on delisted or data-missing tickers (rows are dropped).

---

## Pipeline Architecture

### FinBERT Pipeline

**Model:** `ProsusAI/finbert` — BERT fine-tuned on financial communications text.  
**Label order** (important): logit index `0 = positive`, `1 = negative`, `2 = neutral`.

#### Stage 1: Inference (`finbert/inference.py`)

Earnings transcripts are far longer than FinBERT's 512-token context limit. We use a **sliding window** approach:

```
Transcript → tokenize → [chunk_0, chunk_1, ..., chunk_n]  (512 tokens, 50-token overlap)
                                  ↓
                   Batched FinBERT forward pass (batch_size=32)
                                  ↓
                   Per-chunk: softmax probabilities (3,) + CLS embedding (768,)
                                  ↓
                   Average across all chunks
                                  ↓
           sent_positive, sent_negative, sent_neutral  +  emb_0 ... emb_767
```

The inference step outputs a CSV (`transcripts_with_sentiment.csv`) with three sentiment columns and 768 embedding columns appended to the original DataFrame. Checkpointing saves progress to Google Drive every 100 rows so Colab session disconnections don't lose work.

**Key implementation details:**
- Padding + attention masks for variable-length chunks within a batch
- `output_hidden_states=True` to extract the CLS token from the last hidden layer
- All inference runs under `torch.no_grad()` — FinBERT weights are frozen throughout

#### Stage 2: Prediction Head (`finbert/inference.py → train_prediction_head`)

A logistic regression classifier is trained on the CAR labels using FinBERT features. Two feature modes:

| Mode | Features | Preprocessing |
|---|---|---|
| `sentiment` | `[sent_positive, sent_negative, sent_neutral]` (3 dim) | None |
| `embedding` | `[emb_0, ..., emb_767]` (768 dim) | StandardScaler |

Hyperparameter tuning: 5-fold cross-validation grid search over `C ∈ {0.001, 0.01, 0.1, 1.0, 10.0}`, scoring by ROC-AUC. On the full dataset, all FinBERT variants selected `C=0.001` (maximum regularization), indicating the features provide weak signal relative to the dataset size.

This gives **4 FinBERT variants** in total: `{sentiment, embedding}` × `{label_2, label_5}`.

#### Running the FinBERT Pipeline

Open `finbert/finbert_pipeline.ipynb` in Google Colab (A100 or H100 recommended):

1. Mount Google Drive — outputs are saved there
2. Clone the repo and run `pip install -q transformers datasets yfinance scikit-learn scipy`
3. **Section 4** — Compute CAR labels (uses tqdm, ~1–3 sec/row via yfinance)
4. **Section 5** — FinBERT inference with checkpointing (~2–4 hours on A100 for full dataset)
5. **Section 7** — Train prediction head (4 variants, fast)
6. **Section 8** — Run TF-IDF baselines
7. **Section 9** — Full comparison table + charts

To resume inference after a disconnection: just re-run Section 5. It picks up from the last checkpoint automatically.

---

### Baseline Pipeline

**File:** `baselines/tfidf_models.py`  
**Entry points:** `run_naive_bayes()`, `run_logistic_regression()`, `run_all_label_variants()`

Both classifiers use the same TF-IDF vectorizer:

```
Transcript text
    → TfidfVectorizer(max_features=10_000, ngram_range=(1,2), min_df=2)
    → Naive Bayes  or  LogisticRegression(solver="liblinear", class_weight="balanced")
    → label_2 / label_5
```

**Configuration:**
- 10,000 features, unigrams + bigrams, minimum document frequency = 2
- Logistic Regression uses `class_weight="balanced"` to handle the slight label imbalance
- Both models are trained on raw transcript text with no preprocessing beyond TF-IDF normalization

**Running the baselines standalone:**

```bash
uv run python -m baselines.tfidf_models --csv transcripts_with_sentiment.csv
```

This prints metrics for all 4 variants (2 models × 2 label columns) and writes results to stdout.

---

## Evaluation

All evaluation is centralized in `evaluation/compare_models.py`.

### ML Metrics

Computed on the held-out test set for every model:

- **Accuracy** — fraction of correct predictions
- **Precision / Recall / F1** — especially important given class imbalance; always prefer F1 over accuracy
- **ROC-AUC** — primary ranking metric; measures discrimination ability regardless of threshold

### Financial Metrics

`compute_financial_metrics(df, test_idx, result)` computes:

- **Avg CAR (predicted positive):** mean CAR for stocks the model predicted would outperform
- **Avg CAR (predicted negative):** mean CAR for stocks the model predicted would underperform
- **T-test p-value:** Welch's two-sample t-test (`equal_var=False`) — is the CAR difference statistically significant? Threshold: p < 0.05
- **Sharpe Ratio:** `mean(CAR_positive) / std(CAR_positive)` — return per unit of risk for a simulated long-only strategy on predicted-positive stocks. Sharpe > 1.0 would be strong.

---

## Shared Contract

Both pipelines use the **exact same train/test split.** Never hardcode a split.

```python
from shared.compute_car import get_train_test_split

train_idx, test_idx, y_train, y_test = get_train_test_split(df, label_col="label_2")
```

- `test_size=0.2`, `random_state=42` — fixed, do not change
- `label_col` is `"label_2"` or `"label_5"` — passed at training time
- Returns **index arrays** so each pipeline slices its own feature matrix against the same rows

### Required Output Format (both pipelines)

Every pipeline function must return a dict in this exact format:

```python
{
    "model":     "Your Model Name",    # string displayed in comparison table
    "label_col": "label_2",            # "label_2" or "label_5"
    "accuracy":  float,
    "precision": float,
    "recall":    float,
    "f1":        float,
    "roc_auc":   float,
    "y_pred":    np.ndarray,           # shape (n_test,) — predicted labels
    "y_prob":    np.ndarray,           # shape (n_test,) — predicted probabilities
}
```

`y_pred` and `y_prob` are required by `compare_models.py` to compute financial metrics.

---

## Running the Code

### 1. Load and label the dataset locally (small sample)

```python
from shared.load_data import load_transcripts
from shared.compute_car import build_labeled_dataset

df = load_transcripts()
labeled = build_labeled_dataset(df, sample_n=200)   # 200 rows for local dev
```

### 2. Run baselines on a labeled CSV

```bash
uv run python -m baselines.tfidf_models --csv transcripts_with_sentiment.csv
```

### 3. Run full comparison

```python
import pandas as pd
from baselines.tfidf_models import run_all_label_variants
from evaluation.compare_models import compare_models

df = pd.read_csv("transcripts_with_sentiment.csv")
baseline_results = run_all_label_variants(df)

# Add FinBERT results to the list, then:
comparison = compare_models(df, all_results)
print(comparison.to_string())
```

### 4. Run the full FinBERT pipeline

Open `finbert/finbert_pipeline.ipynb` in Google Colab, set runtime to **A100 GPU**, and run top-to-bottom. For resuming after disconnect, re-run Section 5 only.

---

## Key Findings & Interpretation

### Why FinBERT didn't win

1. **Neutral sentiment dominance.** Across all ~33,400 transcripts, the mean FinBERT scores are approximately: positive ≈ 0.20, negative ≈ 0.05, neutral ≈ 0.66. When 66% of the sentiment signal is neutral, the 3-dimensional sentiment vector carries very little discriminative power for a downstream logistic regression head.

2. **The sentiment bottleneck.** Compressing a full transcript — sometimes 30,000+ tokens — into just 3 probability scores destroys most contextual information. The 768-dim CLS embedding preserves more, but still collapses across chunks by averaging.

3. **FinBERT wasn't trained for this task.** FinBERT was fine-tuned to classify financial sentences as positive/negative/neutral. It was not trained to predict stock returns. Without end-to-end fine-tuning on CAR labels, it functions as a general-purpose sentiment extractor — a poor proxy for the earnings-surprise signal that actually moves stocks.

4. **TF-IDF captures task-relevant vocabulary.** Baseline models learn directly which words and bigrams correlate with CAR. Terms like "beat", "guidance", "miss", "raised", "headwinds" are directly predictive — and TF-IDF weights them purely based on their correlation with the label, without any intermediate abstraction.

### Why all models are near-random

Post-earnings stock returns are notoriously hard to predict from text alone. Prices react to the *surprise* relative to analyst expectations — information not contained in the transcript itself. Analyst consensus estimates, prior guidance, sector context, and macro conditions all drive returns independently of what management said on the call. ROC-AUC ≈ 0.52 across the board reflects this fundamental difficulty, not a failure of implementation.

### What could do better

- **End-to-end FinBERT fine-tuning:** Train FinBERT directly on CAR labels (not intermediate sentiment), letting the model learn which features predict returns rather than which features indicate sentiment. This removes the information bottleneck.
- **Structured features:** Analyst EPS estimates, actual vs. consensus revenue, guidance changes — all of which are not in the transcript but are highly predictive.
- **Sector-stratified models:** Different sectors have different return dynamics around earnings. A single model across all sectors adds noise.
- **Longer or asymmetric windows:** 2-day and 5-day CAR may not capture the full post-earnings drift; some signals are priced in over weeks.

---

## Known Gotchas

- **yfinance flakiness:** Delisted tickers, tickers with insufficient price history, or API rate limits will fail silently. `compute_car.py` wraps all fetches in try/except and drops failed rows. Expect 5–15% dropout on the full dataset.
- **Market timing:** Day 0 = the *next* trading day after the earnings call date. Calls happen after market close, so same-day prices don't count.
- **Class imbalance:** Markets trend upward, so `label=1` (outperformed market) is slightly more common for short windows (~54% for CAR₂). Always report F1 and ROC-AUC — accuracy alone is misleading.
- **FinBERT label order:** The raw model logits are indexed `0=positive, 1=negative, 2=neutral` — the opposite of alphabetical order. This is handled in `inference.py` via `_LABEL_ORDER = ["positive", "negative", "neutral"]`.
- **Save `transcripts_with_sentiment.csv` immediately.** Full inference takes 2–4 hours on an A100. The file is gitignored (too large). If Colab deletes it, the checkpoint CSV in Google Drive lets you resume without restarting from scratch.
- **C=0.001 on the full dataset:** All FinBERT prediction heads selected maximum regularization during grid search. This is the model's way of saying the features provide minimal signal — it's shrinking weights toward zero.

---

## Dependencies

Managed via `uv`. Run `uv sync` to install everything from `pyproject.toml`.

| Package | Version | Purpose |
|---|---|---|
| `transformers` | ≥4.40 | FinBERT model + tokenizer |
| `torch` | ≥2.2 | PyTorch backend for GPU inference |
| `datasets` | ≥2.19 | HuggingFace dataset loading |
| `yfinance` | ≥0.2 | Stock price data retrieval |
| `scikit-learn` | ≥1.4 | TF-IDF, Logistic Regression, Naive Bayes, metrics |
| `scipy` | ≥1.13 | Welch's t-test for financial significance testing |
| `pandas` | ≥2.2 | DataFrame operations throughout |
| `numpy` | ≥1.26 | Array operations |
| `tqdm` | ≥4.66 | Progress bars during inference |

> Install a specific CUDA version of PyTorch if running locally with GPU — see [pytorch.org](https://pytorch.org/get-started/locally/). On Colab, the pre-installed torch version works fine.
