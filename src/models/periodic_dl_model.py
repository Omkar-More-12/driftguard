"""
Periodic deep learning model: a small feedforward neural network that
retrains from scratch on the most recent buffer of data every N rows,
rather than updating continuously. Shows the tradeoff of deep learning
in a streaming setting - more expressive, but costlier to keep fresh.
"""

import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PeriodicDLModel:
    def __init__(self, input_dim, retrain_every=500, buffer_size=2000, epochs=20, lr=0.01):
        self.input_dim = input_dim
        self.retrain_every = retrain_every
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.model = SimpleNet(input_dim)
        self.buffer_X = []
        self.buffer_y = []
        self.row_count = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()

    def predict(self, x):
        x_arr = torch.tensor(list(x.values()), dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            pred = self.model(x_arr).item()
        return 1 if pred > 0.5 else 0

    def learn_one(self, x, y):
        self.buffer_X.append(list(x.values()))
        self.buffer_y.append(y)
        self.buffer_X = self.buffer_X[-self.buffer_size:]
        self.buffer_y = self.buffer_y[-self.buffer_size:]
        self.row_count += 1

        if self.row_count % self.retrain_every == 0:
            self._retrain()

    def _retrain(self):
        X = torch.tensor(self.buffer_X, dtype=torch.float32)
        y = torch.tensor(self.buffer_y, dtype=torch.float32).reshape(-1, 1)

        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()
