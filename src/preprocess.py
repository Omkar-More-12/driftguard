"""
Cleans the raw Electricity dataset and prepares it for streaming simulation:
- Converts the text label (UP/DOWN) into 0/1
- Drops missing rows
- Sorts by date and period so it can be replayed in true chronological order
  (this ordering is what makes the streaming simulation realistic)

Run once, after download_data.py:
    python src/preprocess.py
"""

import os
import pandas as pd

FEATURE_COLS = [
    "day",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]


def preprocess(
    input_path="data/raw/electricity.csv",
    output_path="data/processed/electricity_clean.csv",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)
    df["class"] = df["class"].map({"UP": 1, "DOWN": 0})
    df = df.dropna()
    df = df.sort_values(by=["date", "period"]).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset with {df.shape[0]} rows to {output_path}")
    return df


if __name__ == "__main__":
    preprocess()
