"""
Basic sanity tests for each model: confirm they can be trained (where
applicable) and produce a valid prediction (0 or 1) on a single sample.

Run from the project root:
    pytest tests/
"""

import numpy as np

from src.models.drift_hybrid_model import DriftHybridModel
from src.models.online_model import OnlineModel
from src.models.periodic_dl_model import PeriodicDLModel
from src.models.static_model import StaticModel

FEATURE_COLS = ["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]


def _dummy_batch(n=50):
    X = np.random.rand(n, len(FEATURE_COLS))
    y = np.random.randint(0, 2, size=n)
    return X, y


def _dummy_row():
    return {col: float(np.random.rand()) for col in FEATURE_COLS}


def test_static_model_predicts_after_training():
    model = StaticModel()
    X, y = _dummy_batch()
    model.initial_train(X, y)
    pred = model.predict(_dummy_row())
    assert pred in (0, 1)


def test_online_model_learns_and_predicts():
    model = OnlineModel()
    x = _dummy_row()
    model.learn_one(x, 1)
    pred = model.predict(x)
    assert pred in (0, 1)


def test_periodic_dl_model_predicts():
    model = PeriodicDLModel(input_dim=len(FEATURE_COLS), retrain_every=10)
    for _ in range(15):
        model.learn_one(_dummy_row(), np.random.randint(0, 2))
    pred = model.predict(_dummy_row())
    assert pred in (0, 1)


def test_drift_hybrid_model_predicts():
    model = DriftHybridModel(input_dim=len(FEATURE_COLS))
    X, _ = _dummy_batch()
    model.fit_baseline(X)
    pred = model.predict(_dummy_row())
    assert pred in (0, 1)
