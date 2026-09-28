"""
Runs every model against the same simulated stream, in the same order, and
tracks each model's rolling accuracy over time - the "rolling scoreboard"
that is the whole point of this project.
"""

from collections import deque

import pandas as pd


def rolling_accuracy_eval(models_dict, df, feature_cols, window=200, init_size=500):
    """
    models_dict: {"name": model_object}, each implementing predict(x) and
        learn_one(x, y). Models with initial_train / fit_baseline get warmed
        up on the first `init_size` rows before the timed stream begins.

    Returns a DataFrame with one column per model: rolling accuracy over
    the stream, computed over a trailing window of `window` predictions.
    """
    results = {name: [] for name in models_dict}
    windows = {name: deque(maxlen=window) for name in models_dict}

    init_df = df.iloc[:init_size]
    X_init = init_df[feature_cols].values
    y_init = init_df["class"].values

    for name, model in models_dict.items():
        if hasattr(model, "initial_train"):
            model.initial_train(X_init, y_init)
        if hasattr(model, "fit_baseline"):
            model.fit_baseline(X_init)

    stream_df = df.iloc[init_size:].reset_index(drop=True)

    for _, row in stream_df.iterrows():
        x = row[feature_cols].to_dict()
        y = row["class"]

        for name, model in models_dict.items():
            pred = model.predict(x)
            correct = int(pred == y)
            windows[name].append(correct)
            results[name].append(sum(windows[name]) / len(windows[name]))
            model.learn_one(x, y)

    return pd.DataFrame(results)
