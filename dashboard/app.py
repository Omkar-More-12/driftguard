"""
DriftGuard dashboard: shows rolling accuracy for all four models over time,
plus a log of detected drift events.

Run:
    streamlit run dashboard/app.py
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DriftGuard", layout="wide")

st.title("DriftGuard")
st.caption("Watching how models decay - and recover - as data drifts over time")

try:
    results_df = pd.read_csv("results/accuracy_log.csv")
except FileNotFoundError:
    st.error(
        "No results found yet. Run `python -m src.run_experiment` first "
        "to generate results/accuracy_log.csv."
    )
    st.stop()

st.subheader("Rolling accuracy over time")

# Downsample for display: rendering all ~45,000 points as an interactive
# chart is slow in the browser. Every Nth row still shows the full shape
# of the trend clearly, and loads almost instantly.
max_points = 1000
step = max(1, len(results_df) // max_points)
display_df = results_df.iloc[::step]

st.line_chart(display_df)
st.caption(f"Showing every {step} rows ({len(display_df)} points) for faster rendering.")

st.subheader("Final accuracy snapshot")
final_scores = results_df.iloc[-1].sort_values(ascending=False)
cols = st.columns(len(final_scores))
for col, (name, score) in zip(cols, final_scores.items()):
    col.metric(name, f"{score:.1%}")

st.subheader("About this comparison")
st.write(
    """
- **Static** - trained once, never updated. Shows how accuracy decays when a model ignores drift.
- **Online** - updates after every row using incremental logistic regression.
- **PeriodicDL** - a small neural network retrained from scratch on a fixed schedule.
- **DriftHybrid** - online predictor paired with an autoencoder that flags drift and triggers targeted retraining.
"""
)
