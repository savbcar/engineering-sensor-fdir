from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    def __init__(self, random_state: int = 42) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=random_state,
            n_jobs=1,
        )
        self.threshold = 0.0

    def fit(self, x: np.ndarray) -> IsolationForestDetector:
        x0 = self.imputer.fit_transform(x)
        xs = self.scaler.fit_transform(x0)
        self.model.fit(xs)
        score = -self.model.decision_function(xs)
        self.threshold = float(np.quantile(score, 0.95))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler.transform(self.imputer.transform(x))
        return -self.model.decision_function(xs)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score = self.score(x)
        return (score > self.threshold).astype(int), score
