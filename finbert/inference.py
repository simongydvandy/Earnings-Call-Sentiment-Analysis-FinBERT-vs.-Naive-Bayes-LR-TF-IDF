import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# FinBERT label order: 0=positive, 1=negative, 2=neutral
_LABEL_ORDER = ["positive", "negative", "neutral"]

_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device
    if _tokenizer is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model = _model.to(_device)
        _model.eval()


def _chunk_tokens(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    step = chunk_size - overlap
    return [tokens[i: i + chunk_size] for i in range(0, len(tokens), step)]


def _run_batched_forward(chunks: list, batch_size: int = 32) -> list:
    """
    Process chunks in batches through FinBERT.
    Returns list of (probs array, cls_embedding array) per chunk.
    Padding and attention masks handle variable-length chunks correctly.
    """
    results = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        max_len = max(len(c) for c in batch)

        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for j, chunk in enumerate(batch):
            input_ids[j, :len(chunk)] = torch.tensor(chunk)
            attention_mask[j, :len(chunk)] = 1

        input_ids = input_ids.to(_device)
        attention_mask = attention_mask.to(_device)

        with torch.no_grad():
            output = _model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        probs = F.softmax(output.logits, dim=-1).cpu().numpy()       # (batch, 3)
        cls = output.hidden_states[-1][:, 0, :].cpu().numpy()        # (batch, 768)

        for k in range(len(batch)):
            results.append((probs[k], cls[k]))

    return results


def _process_transcript(text: str, chunk_size: int = 512, overlap: int = 50, batch_size: int = 32):
    """
    Single forward pass returning both sentiment scores and CLS embedding.
    More efficient than calling get_finbert_sentiment and get_finbert_embedding separately.
    """
    chunks = _chunk_tokens(text, chunk_size, overlap)
    results = _run_batched_forward(chunks, batch_size=batch_size)

    avg_probs = np.mean([r[0] for r in results], axis=0)
    avg_emb = np.mean([r[1] for r in results], axis=0)

    sentiment = {label: float(avg_probs[i]) for i, label in enumerate(_LABEL_ORDER)}
    return sentiment, avg_emb


def get_finbert_sentiment(text: str, chunk_size: int = 512, overlap: int = 50, batch_size: int = 32) -> dict:
    """
    Extract sentiment probabilities from a full transcript using sliding window chunking.
    Averages softmax probabilities across all chunks.

    Returns:
        dict with keys "positive", "negative", "neutral"
    """
    _load_model()
    sentiment, _ = _process_transcript(text, chunk_size, overlap, batch_size)
    return sentiment


def get_finbert_embedding(text: str, chunk_size: int = 512, overlap: int = 50, batch_size: int = 32) -> np.ndarray:
    """
    Extract a 768-dim CLS token embedding from a full transcript.
    Preserves far more of FinBERT's contextual understanding than 3 sentiment scores.

    Returns:
        np.ndarray of shape (768,)
    """
    _load_model()
    _, embedding = _process_transcript(text, chunk_size, overlap, batch_size)
    return embedding


def run_pipeline(
    df: pd.DataFrame,
    output_path: str = "transcripts_with_sentiment.csv",
    checkpoint_path: str = "checkpoint_sentiment.csv",
    checkpoint_every: int = 100,
    batch_size: int = 32,
    chunk_size: int = 512,
    overlap: int = 50,
) -> pd.DataFrame:
    """
    Run FinBERT inference on every transcript in df.
    Computes both sentiment scores and CLS embeddings in one forward pass per transcript.

    Checkpointing: saves progress every `checkpoint_every` rows.
    On restart, automatically resumes from the last checkpoint.

    Args:
        df:               DataFrame with "transcript" column
        output_path:      final enriched CSV path
        checkpoint_path:  intermediate checkpoint CSV (auto-resume on restart)
        checkpoint_every: save checkpoint every N rows
        batch_size:       chunks per GPU batch (32 works well on A100/H100)
        chunk_size:       max tokens per chunk (FinBERT limit: 512)
        overlap:          token overlap between consecutive chunks

    Returns:
        df with added columns: sent_positive, sent_negative, sent_neutral, emb_0 ... emb_767
    """
    _load_model()
    df = df.copy().reset_index(drop=True)

    # Resume from checkpoint if one exists
    completed_rows = []
    start_idx = 0
    ckpt = Path(checkpoint_path)

    if ckpt.exists():
        ckpt_df = pd.read_csv(checkpoint_path)
        completed_rows = ckpt_df.to_dict("records")
        start_idx = len(completed_rows)
        print(f"Resuming from checkpoint — {start_idx}/{len(df)} rows already done.")

    total = len(df)

    with tqdm(total=total, initial=start_idx, desc="FinBERT inference", unit="transcript") as pbar:
        for i in range(start_idx, total):
            sentiment, embedding = _process_transcript(
                df.iloc[i]["transcript"], chunk_size, overlap, batch_size
            )

            record = {
                "sent_positive": sentiment["positive"],
                "sent_negative": sentiment["negative"],
                "sent_neutral":  sentiment["neutral"],
                **{f"emb_{j}": embedding[j] for j in range(768)},
            }
            completed_rows.append(record)
            pbar.update(1)
            pbar.set_postfix({
                "pos": f"{sentiment['positive']:.2f}",
                "neg": f"{sentiment['negative']:.2f}",
                "neu": f"{sentiment['neutral']:.2f}",
            })

            if (i + 1) % checkpoint_every == 0:
                pd.DataFrame(completed_rows).to_csv(checkpoint_path, index=False)

    results_df = pd.DataFrame(completed_rows)
    for col in results_df.columns:
        df[col] = results_df[col].values

    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched DataFrame ({len(df)} rows) to {output_path}")
    return df


def train_prediction_head(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_col: str = "label_2",
    mode: str = "sentiment",
) -> dict:
    """
    Train a logistic regression prediction head on FinBERT features.

    Args:
        df:        DataFrame with sentiment and embedding columns already filled in
        train_idx: row indices for training (from shared/compute_car.get_train_test_split)
        test_idx:  row indices for testing (same split Davis uses)
        label_col: "label_2" (2-day CAR) or "label_5" (5-day CAR)
        mode:      "sentiment" → 3 features | "embedding" → 768 features

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc, best_C, y_pred, y_prob
    """
    if mode == "sentiment":
        feature_cols = ["sent_positive", "sent_negative", "sent_neutral"]
        model_name = "FinBERT-Sentiment + LogReg"
    elif mode == "embedding":
        feature_cols = [f"emb_{i}" for i in range(768)]
        model_name = "FinBERT-Embedding + LogReg"
    else:
        raise ValueError(f"mode must be 'sentiment' or 'embedding', got '{mode}'")

    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if mode == "embedding":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid, cv=5, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f"[{model_name}] Best C: {grid.best_params_['C']}")

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "label_col": label_col,
        "mode": mode,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "best_C":    grid.best_params_["C"],
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }
