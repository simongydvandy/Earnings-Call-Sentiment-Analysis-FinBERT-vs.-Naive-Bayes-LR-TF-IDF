from datasets import load_dataset
import pandas as pd


def load_transcripts() -> pd.DataFrame:
    ds = load_dataset("kurry/sp500_earnings_transcripts")
    df = ds["train"].to_pandas()

    print("Raw columns:", df.columns.tolist())
    print("Shape:", df.shape)

    df = df.rename(columns={
        "symbol": "ticker",
        "earnings_date": "date",
        "content": "transcript",
    })

    df = df[["ticker", "date", "transcript"]].copy()
    df["transcript_id"] = df.index
    df = df.dropna(subset=["ticker", "date", "transcript"])
    df = df[df["transcript"].str.strip() != ""]
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} transcripts after cleaning.")
    return df
