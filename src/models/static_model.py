"""
Static model: trained once on the first batch of data, then frozen forever.
This is the control group - it shows what happens if you never adapt to
changing data. Expect its accuracy to decay over time as drift occurs.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class StaticModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def initial_train(self, X_init, y_init):
        self.model.fit(X_init, y_init)
        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            return 0
        x_arr = np.array(list(x.values())).reshape(1, -1)
        return int(self.model.predict(x_arr)[0])

    def learn_one(self, x, y):
        # Intentionally does nothing - this model never updates.
        pass
