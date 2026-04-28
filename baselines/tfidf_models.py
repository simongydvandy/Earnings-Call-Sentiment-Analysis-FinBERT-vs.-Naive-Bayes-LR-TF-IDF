from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB

from shared.compute_car import get_train_test_split


REQUIRED_COLUMNS = {"transcript", "label_2", "label_5", "CAR_2", "CAR_5"}


def _validate_inputs(df: pd.DataFrame, label_col: str) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame is missing required columns: {missing_str}")

    if label_col not in {"label_2", "label_5"}:
        raise ValueError("label_col must be 'label_2' or 'label_5'")


def _build_text_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
):
    train_text = df["transcript"].fillna("").iloc[train_idx].astype(str).values
    test_text = df["transcript"].fillna("").iloc[test_idx].astype(str).values

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return vectorizer, X_train, X_test


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _format_metrics(
    *,
    model_name: str,
    label_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    return {
        "model": model_name,
        "label_col": label_col,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.asarray(y_prob),
    }


def run_naive_bayes(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_col: str,
) -> dict:
    _validate_inputs(df, label_col)
    y_train = df[label_col].iloc[train_idx].to_numpy()
    y_test = df[label_col].iloc[test_idx].to_numpy()

    _, X_train, X_test = _build_text_features(df, train_idx, test_idx)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return _format_metrics(
        model_name="TF-IDF + Naive Bayes",
        label_col=label_col,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_logistic_regression(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_col: str,
) -> dict:
    _validate_inputs(df, label_col)
    y_train = df[label_col].iloc[train_idx].to_numpy()
    y_test = df[label_col].iloc[test_idx].to_numpy()

    _, X_train, X_test = _build_text_features(df, train_idx, test_idx)

    clf = LogisticRegression(
        max_iter=1_000,
        random_state=42,
        class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return _format_metrics(
        model_name="TF-IDF + Logistic Regression",
        label_col=label_col,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def run_pipeline(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_col: str,
) -> list[dict]:
    return [
        run_naive_bayes(df, train_idx, test_idx, label_col),
        run_logistic_regression(df, train_idx, test_idx, label_col),
    ]


def run_all_label_variants(df: pd.DataFrame) -> list[dict]:
    results = []
    for label_col in ("label_2", "label_5"):
        train_idx, test_idx, _, _ = get_train_test_split(df, label_col=label_col)
        results.extend(run_pipeline(df, train_idx, test_idx, label_col))
    return results


def _load_default_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TF-IDF baseline models.")
    parser.add_argument(
        "--csv",
        default="transcripts_with_sentiment.csv",
        help="Path to the enriched transcripts CSV.",
    )
    args = parser.parse_args()

    df = _load_default_dataset(args.csv)
    for result in run_all_label_variants(df):
        summary = {k: v for k, v in result.items() if k not in {"y_pred", "y_prob"}}
        print(summary)
