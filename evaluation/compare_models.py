from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from baselines.tfidf_models import run_all_label_variants
from shared.compute_car import get_train_test_split


def _car_column_for_label(label_col: str) -> str:
    mapping = {"label_2": "CAR_2", "label_5": "CAR_5"}
    if label_col not in mapping:
        raise ValueError("label_col must be 'label_2' or 'label_5'")
    return mapping[label_col]


def compute_financial_metrics(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    result: dict,
) -> dict:
    car_col = _car_column_for_label(result["label_col"])
    test_cars = df.iloc[test_idx][car_col].to_numpy(dtype=float)
    y_pred = np.asarray(result["y_pred"])

    positive_returns = test_cars[y_pred == 1]
    negative_returns = test_cars[y_pred == 0]

    avg_car_positive = float(np.mean(positive_returns)) if len(positive_returns) else float("nan")
    avg_car_negative = float(np.mean(negative_returns)) if len(negative_returns) else float("nan")

    if len(positive_returns) > 1 and len(negative_returns) > 1:
        _, p_value = ttest_ind(positive_returns, negative_returns, equal_var=False)
        ttest_p_value = float(p_value)
    else:
        ttest_p_value = float("nan")

    if len(positive_returns) > 1:
        std = float(np.std(positive_returns, ddof=1))
        sharpe_ratio = float(np.mean(positive_returns) / std) if std > 0 else float("nan")
    else:
        sharpe_ratio = float("nan")

    return {
        "avg_car_positive": avg_car_positive,
        "avg_car_negative": avg_car_negative,
        "ttest_p_value": ttest_p_value,
        "sharpe_ratio": sharpe_ratio,
    }


def compare_models(df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    rows = []
    split_cache: dict[str, np.ndarray] = {}

    for result in results:
        label_col = result["label_col"]
        if label_col not in split_cache:
            _, test_idx, _, _ = get_train_test_split(df, label_col=label_col)
            split_cache[label_col] = test_idx

        financial = compute_financial_metrics(df, split_cache[label_col], result)
        rows.append(
            {
                "model": result["model"],
                "label_col": label_col,
                "accuracy": float(result["accuracy"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
                "roc_auc": float(result["roc_auc"]),
                **financial,
            }
        )

    return pd.DataFrame(rows).sort_values(["label_col", "f1", "roc_auc"], ascending=[True, False, False])


def _load_default_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model results on an enriched transcripts CSV.")
    parser.add_argument(
        "--csv",
        default="transcripts_with_sentiment.csv",
        help="Path to the enriched transcripts CSV.",
    )
    args = parser.parse_args()

    df = _load_default_dataset(args.csv)
    baseline_results = run_all_label_variants(df)
    comparison = compare_models(df, baseline_results)
    print(comparison.to_string(index=False))
