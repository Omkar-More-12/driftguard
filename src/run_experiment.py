"""
End-to-end experiment runner: loads the cleaned dataset, builds all four
models, runs the rolling-accuracy evaluation, and saves both the results
CSV and the comparison plot.

Run from the project root, after download_data.py and preprocess.py:
    python -m src.run_experiment
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluator import rolling_accuracy_eval
from src.models.drift_hybrid_model import DriftHybridModel
from src.models.online_model import OnlineModel
from src.models.periodic_dl_model import PeriodicDLModel
from src.models.static_model import StaticModel
from src.preprocess import FEATURE_COLS

INIT_SIZE = 500
WINDOW = 500


def main():
    df = pd.read_csv("data/processed/electricity_clean.csv")
    input_dim = len(FEATURE_COLS)

    # Scale features so no single column (e.g. nswdemand, in the thousands)
    # dominates the others (e.g. period, 0-47). The scaler is fit only on
    # the initial warm-up rows, then applied to the whole stream, so no
    # information from "the future" leaks into the fit.
    scaler = StandardScaler()
    scaler.fit(df.loc[: INIT_SIZE - 1, FEATURE_COLS])
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    models = {
        "Static": StaticModel(),
        "Online": OnlineModel(),
        "PeriodicDL": PeriodicDLModel(input_dim=input_dim, retrain_every=500),
        "DriftHybrid": DriftHybridModel(input_dim=input_dim),
    }

    results_df = rolling_accuracy_eval(
        models, df, FEATURE_COLS, window=WINDOW, init_size=INIT_SIZE
    )

    os.makedirs("results/plots", exist_ok=True)
    results_df.to_csv("results/accuracy_log.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in results_df.columns:
        style = "--" if col == "DriftHybrid" else "-"
        ax.plot(results_df[col], style, label=col, linewidth=1.3)
    ax.set_title("Rolling accuracy over time")
    ax.set_xlabel("Rows processed")
    ax.set_ylabel("Rolling accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/plots/rolling_accuracy.png", dpi=150)

    print("Saved results/accuracy_log.csv and results/plots/rolling_accuracy.png")

    if hasattr(models["DriftHybrid"], "drift_log"):
        print(f"Drift events detected at rows: {models['DriftHybrid'].drift_log}")


if __name__ == "__main__":
    main()
