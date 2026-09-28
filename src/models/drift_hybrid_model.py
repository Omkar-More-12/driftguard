"""
Drift-aware hybrid model: pairs an online Logistic Regression predictor with
an autoencoder that watches for drift. The autoencoder is trained on "normal"
early data; when a new row's reconstruction error crosses a threshold
(baseline mean + k * std), that is treated as a drift signal and logged.

This is the model the project's core claim rests on: react quickly when the
data distribution actually changes, and otherwise get out of the way.
"""

import numpy as np
import torch
import torch.nn as nn
from river import linear_model


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 4), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(4, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DriftHybridModel:
    def __init__(self, input_dim, threshold_multiplier=3.0, lr=0.01):
        self.input_dim = input_dim
        self.predictor = linear_model.LogisticRegression()
        self.autoencoder = Autoencoder(input_dim)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.threshold_multiplier = threshold_multiplier
        self.baseline_mean = None
        self.baseline_std = None
        self.drift_log = []
        self.row_count = 0

    def fit_baseline(self, X_init):
        """Train the autoencoder on early, presumed-normal data, and record
        the baseline reconstruction-error distribution."""
        X_tensor = torch.tensor(X_init, dtype=torch.float32)
        for _ in range(30):
            self.optimizer.zero_grad()
            recon = self.autoencoder(X_tensor)
            loss = self.loss_fn(recon, X_tensor)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            recon = self.autoencoder(X_tensor)
            errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
        self.baseline_mean = float(np.mean(errors))
        self.baseline_std = float(np.std(errors))

    def _reconstruction_error(self, x):
        x_arr = torch.tensor(list(x.values()), dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            recon = self.autoencoder(x_arr)
            error = self.loss_fn(recon, x_arr).item()
        return error, x_arr

    def predict(self, x):
        pred = self.predictor.predict_one(x)
        return int(pred) if pred is not None else 0

    def learn_one(self, x, y):
        self.row_count += 1
        error, x_arr = self._reconstruction_error(x)

        if self.baseline_mean is not None:
            threshold = self.baseline_mean + self.threshold_multiplier * self.baseline_std
            if error > threshold:
                self.drift_log.append(self.row_count)
                # Nudge the autoencoder toward the new pattern so it stops
                # over-firing on the same shift going forward.
                self.optimizer.zero_grad()
                recon = self.autoencoder(x_arr)
                loss = self.loss_fn(recon, x_arr)
                loss.backward()
                self.optimizer.step()

        self.predictor.learn_one(x, y)
