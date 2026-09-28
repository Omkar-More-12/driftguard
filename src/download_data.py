"""
Downloads the Electricity Market (ELEC2) dataset from OpenML and saves it
locally as a raw CSV. This dataset is a standard benchmark in concept-drift
research (Gama et al.) and contains real historical electricity market data
from New South Wales, Australia, with known distribution shifts over time.

Run once:
    python src/download_data.py
"""

import os
from sklearn.datasets import fetch_openml


def download_electricity_dataset(output_path="data/raw/electricity.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Downloading Electricity (ELEC2) dataset from OpenML...")
    data = fetch_openml(name="electricity", version=1, as_frame=True)
    df = data.frame

    df.to_csv(output_path, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {output_path}")
    return df


if __name__ == "__main__":
    download_electricity_dataset()
