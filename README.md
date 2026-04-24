# Earnings Call Sentiment Analysis: FinBERT vs. Naive Bayes / LR + TF-IDF

**Course:** CS-4267-02 Deep Learning  
**Research Question:** Does transformer-based financial sentiment extracted from earnings call transcripts improve prediction of short-term abnormal stock returns compared to classical NLP baselines?

---

## Project Structure

```
.
├── shared/
│   ├── load_data.py        ← Load & clean the HuggingFace transcript dataset
│   └── compute_car.py      ← CAR (Cumulative Abnormal Return) computation, label assignment, train/test split
│
├── finbert/
│   ├── inference.py        ← FinBERT sliding window inference + prediction head
│   └── finbert_pipeline.ipynb  ← End-to-end notebook (run on Colab A100/H100)
│
├── baselines/
│   └── tfidf_models.py     ← TF-IDF + Naive Bayes / Logistic Regression pipelines
│
└── evaluation/
    └── compare_models.py   ← Shared evaluation format + final model comparison
```

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/simongydvandy/Earnings-Call-Sentiment-Analysis-FinBERT-vs.-Naive-Bayes-LR-TF-IDF.git
cd Earnings-Call-Sentiment-Analysis-FinBERT-vs.-Naive-Bayes-LR-TF-IDF

# Install all dependencies (one command)
uv sync

# Verify environment
uv run python -c "from shared.load_data import load_transcripts; print('OK')"
```

> **No GPU required for the baseline pipeline.** The FinBERT inference runs on Google Colab (A100/H100). Everything else runs locally via `uv run python`.

---

## Data

- **Transcripts:** `[kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts)` — ~33,400 earnings call transcripts across 685 S&P 500 tickers, 2005–2025
- **Stock prices:** Retrieved via `yfinance`
- **Labels:** Cumulative Abnormal Return (CAR) over two windows — **[0, +2]** and **[0, +5]** trading days after each earnings date, adjusted for S&P 500 market return
  - `label_2 = 1` if CAR_2 > 0, else 0
  - `label_5 = 1` if CAR_5 > 0, else 0

---

## Models

### Primary — FinBERT (`ProsusAI/finbert`)

Two-stage pipeline:

1. **FinBERT inference** — sliding window chunking (512 tokens, 50-token overlap) over each transcript; average probabilities/embeddings across chunks
2. **Prediction head** — logistic regression trained on CAR labels using FinBERT features

Two feature variants are evaluated:

- **FinBERT-Sentiment** — 3 features: `[positive, negative, neutral]` probability scores
- **FinBERT-Embedding** — 768-dim CLS token embedding from FinBERT's last hidden layer

### Baselines

One-stage pipelines trained end-to-end directly on CAR labels:

- **TF-IDF + Naive Bayes**
- **TF-IDF + Logistic Regression**

> Key distinction: FinBERT produces intermediate sentiment features; the baselines learn directly from raw word frequencies. Keep this clear in writeups.

---

## Shared Contract (read before writing any pipeline code)

Both pipelines must use the exact same train/test split. **Never hardcode a split — always use:**

```python
from shared.compute_car import get_train_test_split

train_idx, test_idx, y_train, y_test = get_train_test_split(df, label_col="label_2")
```

- `test_size=0.2`, `random_state=42` — baked in, do not change
- `label_col` can be `"label_2"` or `"label_5"` — choose at training time
- Returns **index arrays**, not data copies — each pipeline slices its own feature matrix against these indices

### Expected output format from each pipeline

```python
{
    "model":     "Your Model Name",
    "label_col": "label_2",          # or "label_5"
    "accuracy":  float,
    "precision": float,
    "recall":    float,
    "f1":        float,
    "roc_auc":   float,
    "y_pred":    np.ndarray,         # predicted labels on test set
    "y_prob":    np.ndarray,         # predicted probabilities on test set
}
```

`y_pred` and `y_prob` are needed by `evaluation/compare_models.py` for financial metrics (t-test, Sharpe ratio).

---

## Baseline Pipeline — Getting Started

The baseline pipeline works entirely on the enriched DataFrame produced after FinBERT inference. The relevant columns are:


| Column       | Description                             |
| ------------ | --------------------------------------- |
| `transcript` | Raw earnings call text                  |
| `label_2`    | Binary label — 2-day CAR > 0            |
| `label_5`    | Binary label — 5-day CAR > 0            |
| `CAR_2`      | Actual 2-day cumulative abnormal return |
| `CAR_5`      | Actual 5-day cumulative abnormal return |


**To get started:**

```python
import pandas as pd
from shared.compute_car import get_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load the shared labeled dataset (produced by FinBERT pipeline or compute_car directly)
df = pd.read_csv("transcripts_with_sentiment.csv")

# Get canonical train/test split — must match FinBERT's split exactly
train_idx, test_idx, y_train, y_test = get_train_test_split(df, label_col="label_2")

X_train_text = df["transcript"].iloc[train_idx].values
X_test_text  = df["transcript"].iloc[test_idx].values

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

# Train your classifiers in baselines/tfidf_models.py
```

Implement both classifiers in `baselines/tfidf_models.py` with a `run_pipeline(df, train_idx, test_idx, label_col)` function that returns the metrics dict format above.

### Baseline Action Items

Work through these in order:

- [ ] **1. Clone & install** — `git clone ...` then `uv sync`
- [ ] **2. Verify imports** — `uv run python -c "from shared.load_data import load_transcripts; print('OK')"`
- [ ] **3. Get the labeled dataset** — either receive `transcripts_with_sentiment.csv` from Simon, or run `shared/compute_car.py` directly on a sample to generate your own labels for dev
- [ ] **4. Explore the data** — load the CSV, check column names, class balance (`label_2` and `label_5`), and transcript length distribution
- [ ] **5. Implement `baselines/tfidf_models.py`** — two functions:
  - `run_naive_bayes(df, train_idx, test_idx, label_col)` → metrics dict
  - `run_logistic_regression(df, train_idx, test_idx, label_col)` → metrics dict
  - Both must return the exact dict format defined in the Shared Contract section above
- [ ] **6. Run both variants for both label columns** — `label_2` and `label_5`, so 4 results total (matching Simon's 4 FinBERT variants)
- [ ] **7. Hand results to `evaluation/compare_models.py`** — pass your metrics dicts alongside Simon's for the final comparison table

> **Note on `transcripts_with_sentiment.csv`:** this file is in `.gitignore` (too large for git). Simon will share it directly. In the meantime, you can generate a small labeled sample yourself using `shared/compute_car.py` — just call `build_labeled_dataset(df, sample_n=200)` for local dev.

---

## Evaluation Metrics

All models are evaluated on the same held-out test set.

**ML metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

**Financial metrics** (computed in `evaluation/compare_models.py`):

- Average CAR for predicted-positive vs predicted-negative groups
- **T-test** — is the CAR difference statistically significant? (threshold: p < 0.05)
- **Sharpe Ratio** — return per unit of risk for a simulated long strategy (Sharpe > 1.0 is strong)

---

## Known Gotchas

- **yfinance flakiness:** Delisted tickers will fail silently — `compute_car.py` handles this with try/except and drops failed rows
- **Market timing:** Day 0 is the open of the *next* trading day after the earnings call (calls happen after market close)
- **Class imbalance:** Markets trend upward — expect more `label=1` rows. Always report F1 and AUC, not just accuracy
- **FinBERT label order:** `positive=0, negative=1, neutral=2` in the model's raw logits
- **Save after inference:** `transcripts_with_sentiment.csv` takes hours to generate on GPU — never delete it

---

## Dependencies

Managed via `uv`. See `pyproject.toml` for the full list. Key packages:


| Package        | Purpose                      |
| -------------- | ---------------------------- |
| `transformers` | FinBERT model                |
| `torch`        | PyTorch backend              |
| `datasets`     | HuggingFace dataset loading  |
| `yfinance`     | Stock price data             |
| `scikit-learn` | Baselines + prediction head  |
| `scipy`        | T-test for financial metrics |


