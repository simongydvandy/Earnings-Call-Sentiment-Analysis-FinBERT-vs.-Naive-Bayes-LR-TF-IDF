import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _fetch_abnormal_returns(ticker: str, earnings_date: str, window: int = 5):
    import yfinance as yf

    # Day 0 is the next trading day after the earnings call (calls happen after close)
    day0 = pd.Timestamp(earnings_date) + pd.offsets.BDay(1)
    end = day0 + pd.offsets.BDay(window + 2)

    stock = yf.download(ticker, start=day0, end=end, progress=False, auto_adjust=True)
    market = yf.download("^GSPC", start=day0, end=end, progress=False, auto_adjust=True)

    if stock.empty or market.empty:
        return None

    stock_ret = stock["Close"].pct_change().dropna()
    market_ret = market["Close"].pct_change().dropna()

    # Align on shared trading days then take first `window` days
    combined = pd.concat([stock_ret, market_ret], axis=1, join="inner").dropna()
    combined.columns = ["stock", "market"]
    combined = combined.iloc[:window]

    if len(combined) < window:
        return None

    abnormal = (combined["stock"] - combined["market"]).values
    return abnormal


def compute_car(ticker: str, earnings_date: str, window: int = 5):
    """Return CAR over [0, +window] trading days. Returns None on data failure."""
    try:
        abnormal = _fetch_abnormal_returns(ticker, earnings_date, window)
        if abnormal is None:
            return None
        return float(abnormal[:window].sum())
    except Exception:
        return None


def build_labeled_dataset(df: pd.DataFrame, sample_n: int = None) -> pd.DataFrame:
    """
    Add CAR_2, CAR_5, label_2, label_5 columns to df.
    Fetches price data once per row (window=5) and derives both windows from it.
    Use sample_n for local dev (e.g. sample_n=200).
    """
    if sample_n is not None:
        df = df.head(sample_n).copy()
    else:
        df = df.copy()

    car2_list, car5_list = [], []

    for _, row in df.iterrows():
        try:
            abnormal = _fetch_abnormal_returns(row["ticker"], str(row["date"]), window=5)
            if abnormal is None:
                car2_list.append(None)
                car5_list.append(None)
            else:
                car2_list.append(float(abnormal[:2].sum()))
                car5_list.append(float(abnormal[:5].sum()))
        except Exception:
            car2_list.append(None)
            car5_list.append(None)

    df["CAR_2"] = car2_list
    df["CAR_5"] = car5_list

    df = df.dropna(subset=["CAR_2", "CAR_5"]).reset_index(drop=True)

    df["label_2"] = (df["CAR_2"] > 0).astype(int)
    df["label_5"] = (df["CAR_5"] > 0).astype(int)

    print(f"Labeled dataset: {len(df)} rows after dropping failed fetches.")
    print(f"label_2 balance: {df['label_2'].value_counts().to_dict()}")
    print(f"label_5 balance: {df['label_5'].value_counts().to_dict()}")

    return df


def get_train_test_split(df: pd.DataFrame, label_col: str = "label_2"):
    """
    Canonical train/test split — always use this, never hardcode.
    Returns index arrays so Simon and Davis can slice their own feature matrices
    against the same rows.

    label_col: "label_2" (2-day CAR) or "label_5" (5-day CAR)
    """
    indices = np.arange(len(df))
    labels = df[label_col].values

    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, labels, test_size=0.2, random_state=42
    )
    return train_idx, test_idx, y_train, y_test
