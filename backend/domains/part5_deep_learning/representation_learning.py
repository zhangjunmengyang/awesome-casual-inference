"""Representation Learning for Causal Inference

Key concepts:
- Balanced Representations: Learning features that are similar across treatment groups
- IPM (Integral Probability Metric): Measuring distribution distance
- CFR (Counterfactual Regression): TARNet + IPM regularization

References:
- Johansson et al. (2016): Learning Representations for Counterfactual Inference
- Shalit et al. (2017): Estimating individual treatment effect
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import generate_ihdp_like_data, compute_mmd


class SimpleRepresentationNet(nn.Module):
    """Simple neural network for learning representations"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        repr_dim: int = 32,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, repr_dim))
        self.encoder = nn.Sequential(*layers)

        # Outcome prediction head
        self.outcome_head = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (representation, outcome_pred)
        """
        phi = self.encoder(x)
        y_pred = self.outcome_head(phi).squeeze(-1)
        return phi, y_pred


def train_with_ipm_regularization(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int = 64,
    repr_dim: int = 32,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    alpha: float = 0.0,  # IPM regularization weight
    device: str = 'cpu'
) -> Tuple[nn.Module, dict]:
    """
    Train representation learning model with optional IPM regularization

    Parameters:
    -----------
    alpha: IPM regularization weight
           0 = no regularization (standard supervised learning)
           >0 = CFR (Counterfactual Regression)
    """
    # Data preparation
    X_tensor = torch.FloatTensor(X).to(device)
    T_tensor = torch.FloatTensor(T).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = SimpleRepresentationNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        repr_dim=repr_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training history
    history = {
        'total_loss': [],
        'prediction_loss': [],
        'ipm_loss': [],
        'balance_metric': []
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_total = 0
        epoch_pred = 0
        epoch_ipm = 0

        for batch_x, batch_t, batch_y in dataloader:
            optimizer.zero_grad()

            phi, y_pred = model(batch_x)

            # Prediction loss
            pred_loss = criterion(y_pred, batch_y)

            # IPM regularization (MMD between treatment groups)
            if alpha > 0:
                phi_t1 = phi[batch_t == 1]
                phi_t0 = phi[batch_t == 0]

                if len(phi_t1) > 0 and len(phi_t0) > 0:
                    ipm_loss = compute_mmd(phi_t1, phi_t0, kernel='linear')
                else:
                    ipm_loss = torch.tensor(0.0).to(device)
            else:
                ipm_loss = torch.tensor(0.0).to(device)

            # Total loss
            total_loss = pred_loss + alpha * ipm_loss

            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_pred += pred_loss.item()
            epoch_ipm += ipm_loss.item()

        n_batches = len(dataloader)
        history['total_loss'].append(epoch_total / n_batches)
        history['prediction_loss'].append(epoch_pred / n_batches)
        history['ipm_loss'].append(epoch_ipm / n_batches)

        # Compute balance metric (MMD on full dataset)
        model.eval()
        with torch.no_grad():
            phi_all, _ = model(X_tensor)
            phi_t1 = phi_all[T_tensor == 1]
            phi_t0 = phi_all[T_tensor == 0]
            if len(phi_t1) > 0 and len(phi_t0) > 0:
                balance = compute_mmd(phi_t1, phi_t0, kernel='linear').item()
            else:
                balance = 0.0
            history['balance_metric'].append(balance)

    return model, history


def visualize_representations(
    X: np.ndarray,
    T: np.ndarray,
    phi: np.ndarray,
    title: str = "Learned Representations"
) -> go.Figure:
    """Visualize learned representations (using first 2 dimensions)"""

    fig = go.Figure()

    # Control group
    phi_t0 = phi[T == 0]
    fig.add_trace(go.Scatter(
        x=phi_t0[:, 0],
        y=phi_t0[:, 1] if phi.shape[1] > 1 else np.zeros(len(phi_t0)),
        mode='markers',
        name='Control (T=0)',
        marker=dict(color='#2D9CDB', opacity=0.6, size=5)
    ))

    # Treatment group
    phi_t1 = phi[T == 1]
    fig.add_trace(go.Scatter(
        x=phi_t1[:, 0],
        y=phi_t1[:, 1] if phi.shape[1] > 1 else np.zeros(len(phi_t1)),
        mode='markers',
        name='Treatment (T=1)',
        marker=dict(color='#EB5757', opacity=0.6, size=5)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        template='plotly_white',
        height=400
    )

    return fig


def compare_balance_metrics(
    X: np.ndarray,
    T: np.ndarray,
    phi_no_reg: np.ndarray,
    phi_with_reg: np.ndarray
) -> dict:
    """Compare balance metrics before and after IPM regularization"""

    # Original feature space balance
    X_t0 = X[T == 0]
    X_t1 = X[T == 1]
    original_balance = np.linalg.norm(X_t0.mean(axis=0) - X_t1.mean(axis=0))

    # Representation space balance (no regularization)
    phi_t0_no_reg = phi_no_reg[T == 0]
    phi_t1_no_reg = phi_no_reg[T == 1]
    balance_no_reg = np.linalg.norm(phi_t0_no_reg.mean(axis=0) - phi_t1_no_reg.mean(axis=0))

    # Representation space balance (with regularization)
    phi_t0_reg = phi_with_reg[T == 0]
    phi_t1_reg = phi_with_reg[T == 1]
    balance_with_reg = np.linalg.norm(phi_t0_reg.mean(axis=0) - phi_t1_reg.mean(axis=0))

    return {
        'original': float(original_balance),
        'no_regularization': float(balance_no_reg),
        'with_regularization': float(balance_with_reg),
        'improvement': float((balance_no_reg - balance_with_reg) / balance_no_reg * 100)
    }
