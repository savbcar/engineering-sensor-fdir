from __future__ import annotations

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoencoderDetector:
    def __init__(
        self,
        hidden_dim: int = 24,
        latent_dim: int = 6,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model: _Autoencoder | None = None
        self.threshold = 0.0
        self.loss_history: list[float] = []

    def fit(self, x: np.ndarray) -> AutoencoderDetector:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        x0 = self.imputer.fit_transform(x)
        xs = self.scaler.fit_transform(x0).astype(np.float32)
        tensor = torch.from_numpy(xs)
        self.model = _Autoencoder(xs.shape[1], self.hidden_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            recon = self.model(tensor)
            loss = loss_fn(recon, tensor)
            loss.backward()
            optimizer.step()
            self.loss_history.append(float(loss.detach().cpu()))
        train_error = self.score(x)
        self.threshold = float(np.quantile(train_error, 0.99))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Detector must be fitted before score")
        xs = self.scaler.transform(self.imputer.transform(x)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(xs)
            recon = self.model(tensor)
            error = torch.mean((recon - tensor) ** 2, dim=1)
        return error.cpu().numpy()

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score = self.score(x)
        return (score > self.threshold).astype(int), score
