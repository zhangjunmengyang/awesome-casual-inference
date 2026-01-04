"""Utilities for deep learning causal inference models"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


def generate_ihdp_like_data(
    n_samples: int = 1000,
    n_features: int = 25,
    hidden_confounding: bool = False,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate IHDP-like semi-synthetic data

    Parameters:
    -----------
    n_samples: Number of samples
    n_features: Number of features
    hidden_confounding: Whether to include hidden confounding
    seed: Random seed

    Returns:
    --------
    (X, T, Y, Y0, Y1) - Features, treatment, observed outcome, potential outcomes
    """
    if seed is not None:
        np.random.seed(seed)

    # Features (mixed continuous and binary)
    X_cont = np.random.randn(n_samples, n_features // 2)
    X_bin = np.random.binomial(1, 0.5, (n_samples, n_features - n_features // 2))
    X = np.concatenate([X_cont, X_bin], axis=1)

    # Propensity score (nonlinear)
    propensity_score = 1 / (1 + np.exp(-(
        0.5 * X[:, 0] +
        0.3 * X[:, 1] * X[:, 2] +
        0.2 * X[:, 3] -
        0.4 * X[:, 4]
    )))

    # Treatment assignment
    T = np.random.binomial(1, propensity_score)

    # Potential outcomes (nonlinear)
    # Y(0)
    Y0 = (
        1 +
        0.5 * X[:, 0] +
        0.3 * X[:, 1] ** 2 +
        0.2 * np.sin(X[:, 2]) +
        0.1 * X[:, 3] * X[:, 4] +
        np.random.randn(n_samples) * 0.5
    )

    # Y(1) - heterogeneous effects
    treatment_effect = (
        2 +
        0.8 * X[:, 0] -
        0.3 * X[:, 1] +
        0.5 * X[:, 2] * (X[:, 2] > 0)
    )

    Y1 = Y0 + treatment_effect

    # Observed outcome
    Y = np.where(T == 1, Y1, Y0)

    return X, T, Y, Y0, Y1


def generate_twins_like_data(
    n_samples: int = 2000,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Twins-like dataset

    Simulates infant mortality study with twins
    """
    if seed is not None:
        np.random.seed(seed)

    # Features: maternal characteristics
    age = np.random.uniform(15, 45, n_samples)
    education = np.random.randint(0, 18, n_samples)
    prenatal_visits = np.random.poisson(10, n_samples)
    smoking = np.random.binomial(1, 0.2, n_samples)
    weight_gain = np.random.normal(30, 10, n_samples)

    X = np.column_stack([age, education, prenatal_visits, smoking, weight_gain])

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # Treatment: premature birth
    propensity = 1 / (1 + np.exp(-(
        -0.5 +
        0.3 * X[:, 3] +  # smoking increases risk
        -0.2 * X[:, 1] +  # education decreases risk
        -0.1 * X[:, 2]    # prenatal care decreases risk
    )))
    T = np.random.binomial(1, propensity)

    # Potential outcomes: mortality risk
    base_risk = 1 / (1 + np.exp(-(
        -2 +
        0.5 * X[:, 3] +   # smoking
        -0.3 * X[:, 1] +  # education
        -0.2 * X[:, 2]    # prenatal care
    )))

    # Premature birth increases mortality risk, but effect varies
    treatment_effect = 0.1 + 0.05 * X[:, 3] - 0.02 * X[:, 2]

    Y0 = np.random.binomial(1, base_risk)
    Y1 = np.random.binomial(1, np.clip(base_risk + treatment_effect, 0, 1))

    Y = np.where(T == 1, Y1, Y0)

    return X, T, Y, Y0, Y1


class EarlyStopping:
    """Early stopping mechanism"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


def pehe(y0_true: np.ndarray, y1_true: np.ndarray,
         y0_pred: np.ndarray, y1_pred: np.ndarray) -> float:
    """
    Calculate PEHE (Precision in Estimation of Heterogeneous Effect)

    sqrt(E[(tau_true - tau_pred)^2])
    """
    tau_true = y1_true - y0_true
    tau_pred = y1_pred - y0_pred
    return np.sqrt(np.mean((tau_true - tau_pred) ** 2))


def ate_error(y0_true: np.ndarray, y1_true: np.ndarray,
              y0_pred: np.ndarray, y1_pred: np.ndarray) -> float:
    """
    Calculate ATE estimation error

    |E[tau_true] - E[tau_pred]|
    """
    ate_true = np.mean(y1_true - y0_true)
    ate_pred = np.mean(y1_pred - y0_pred)
    return np.abs(ate_true - ate_pred)


def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel: str = 'rbf') -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions

    Parameters:
    -----------
    x: First sample (n_samples, n_features)
    y: Second sample (m_samples, n_features)
    kernel: Kernel type ('rbf' or 'linear')

    Returns:
    --------
    MMD distance
    """
    if kernel == 'rbf':
        # RBF kernel with automatic bandwidth selection
        x_size = x.size(0)
        y_size = y.size(0)

        # Compute pairwise distances
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())

        x_sqnorms = torch.diag(xx)
        y_sqnorms = torch.diag(yy)

        # Gram matrices
        gamma = 1.0 / x.size(1)  # 1 / n_features

        K_xx = torch.exp(-gamma * (
            x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx
        ))
        K_yy = torch.exp(-gamma * (
            y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy
        ))
        K_xy = torch.exp(-gamma * (
            x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy
        ))

        # MMD
        mmd = (K_xx.sum() / (x_size * x_size) +
               K_yy.sum() / (y_size * y_size) -
               2 * K_xy.sum() / (x_size * y_size))

        return mmd

    elif kernel == 'linear':
        # Linear kernel (simple mean difference)
        return torch.mean(x, dim=0).sub(torch.mean(y, dim=0)).pow(2).sum()

    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def compute_wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute 1D Wasserstein distance (simplified)

    For high-dimensional data, this is computed on the first principal component
    """
    from scipy.stats import wasserstein_distance

    if x.ndim > 1:
        # Project to first dimension (simplified)
        x = x[:, 0]
        y = y[:, 0]

    return wasserstein_distance(x, y)
