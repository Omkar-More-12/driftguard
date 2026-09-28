"""
Online model: updates its weights after every single row using river's
incremental Logistic Regression. Adapts continuously, with no explicit
drift detection - it just keeps nudging its weights toward recent data.
"""

from river import linear_model


class OnlineModel:
    def __init__(self):
        self.model = linear_model.LogisticRegression()

    def predict(self, x):
        pred = self.model.predict_one(x)
        return int(pred) if pred is not None else 0

    def learn_one(self, x, y):
        self.model.learn_one(x, y)
