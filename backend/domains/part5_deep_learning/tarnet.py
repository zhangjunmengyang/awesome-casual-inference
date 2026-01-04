"""TARNet (Treatment-Agnostic Representation Network)

Architecture:
- Shared representation layer: Learns treatment-agnostic features
- Two output heads: Predict Y(0) and Y(1) separately

Reference:
- Shalit et al., "Estimating individual treatment effect: generalization bounds
  and algorithms" (ICML 2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple

from .utils import generate_ihdp_like_data, pehe, ate_error, compute_mmd


class TARNet(nn.Module):
    """
    TARNet Model

    Architecture:
    X -> [Shared Representation] -> Phi(X)
                                      |
                    +----------------+----------------+
                    |                                 |
                [Head 0]                         [Head 1]
                    |                                 |
                  Y(0)                              Y(1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        repr_dim: int = 50,
        n_repr_layers: int = 3,
        n_head_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Shared representation layer
        repr_layers = []
        prev_dim = input_dim
        for _ in range(n_repr_layers):
            repr_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        repr_layers.append(nn.Linear(prev_dim, repr_dim))
        self.representation = nn.Sequential(*repr_layers)

        # Control group output head (Y0)
        head0_layers = []
        prev_dim = repr_dim
        for _ in range(n_head_layers):
            head0_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        head0_layers.append(nn.Linear(prev_dim, 1))
        self.head0 = nn.Sequential(*head0_layers)

        # Treatment group output head (Y1)
        head1_layers = []
        prev_dim = repr_dim
        for _ in range(n_head_layers):
            head1_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        head1_layers.append(nn.Linear(prev_dim, 1))
        self.head1 = nn.Sequential(*head1_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            (y0_pred, y1_pred, representation)
        """
        phi = self.representation(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        return y0, y1, phi

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """Predict individual treatment effect"""
        y0, y1, _ = self.forward(x)
        return y1 - y0


def train_tarnet(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int = 100,
    repr_dim: int = 50,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    alpha: float = 0.0,  # IPM regularization weight
    device: str = 'cpu'
) -> Tuple[TARNet, dict]:
    """
    Train TARNet

    Parameters:
    -----------
    alpha: IPM regularization weight (for representation balancing)
           0 = TARNet, >0 = CFR (Counterfactual Regression)
    """
    # Data preparation
    X_tensor = torch.FloatTensor(X).to(device)
    T_tensor = torch.FloatTensor(T).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = TARNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        repr_dim=repr_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training history
    history = {'loss': [], 'factual_loss': [], 'ipm_loss': []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_factual = 0
        epoch_ipm = 0

        for batch_x, batch_t, batch_y in dataloader:
            optimizer.zero_grad()

            y0_pred, y1_pred, phi = model(batch_x)

            # Factual loss: only compute loss on observed outcomes
            y_pred = torch.where(batch_t == 1, y1_pred, y0_pred)
            factual_loss = criterion(y_pred, batch_y)

            # IPM regularization (optional): balance treatment and control representations
            if alpha > 0:
                phi_t = phi[batch_t == 1]
                phi_c = phi[batch_t == 0]
                if len(phi_t) > 0 and len(phi_c) > 0:
                    # Simplified MMD (Maximum Mean Discrepancy)
                    ipm_loss = compute_mmd(phi_t, phi_c, kernel='linear')
                else:
                    ipm_loss = torch.tensor(0.0).to(device)
            else:
                ipm_loss = torch.tensor(0.0).to(device)

            loss = factual_loss + alpha * ipm_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_factual += factual_loss.item()
            epoch_ipm += ipm_loss.item()

        n_batches = len(dataloader)
        history['loss'].append(epoch_loss / n_batches)
        history['factual_loss'].append(epoch_factual / n_batches)
        history['ipm_loss'].append(epoch_ipm / n_batches)

    return model, history


def visualize_tarnet(
    n_samples: int,
    hidden_dim: int,
    n_epochs: int,
    alpha: float
) -> Tuple[go.Figure, str]:
    """Visualize TARNet training and performance"""

    # Generate data
    X, T, Y, Y0_true, Y1_true = generate_ihdp_like_data(n_samples)

    # Train model
    model, history = train_tarnet(
        X, T, Y,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        alpha=alpha
    )

    # Predictions
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        Y0_pred, Y1_pred, phi = model(X_tensor)
        Y0_pred = Y0_pred.cpu().numpy()
        Y1_pred = Y1_pred.cpu().numpy()
        phi = phi.cpu().numpy()

    # Compute metrics
    pehe_val = pehe(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_err = ate_error(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_true = np.mean(Y1_true - Y0_true)
    ate_pred = np.mean(Y1_pred - Y0_pred)

    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Loss',
            'True vs Predicted ITE',
            'Representation Space (First 2 Dims)',
            'Y0/Y1 Predictions'
        )
    )

    # 1. Training loss
    fig.add_trace(go.Scatter(
        y=history['loss'], mode='lines',
        name='Total Loss', line=dict(color='#2D9CDB')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=history['factual_loss'], mode='lines',
        name='Factual Loss', line=dict(color='#27AE60')
    ), row=1, col=1)

    if alpha > 0:
        fig.add_trace(go.Scatter(
            y=history['ipm_loss'], mode='lines',
            name='IPM Loss', line=dict(color='#EB5757')
        ), row=1, col=1)

    # 2. ITE comparison
    ite_true = Y1_true - Y0_true
    ite_pred = Y1_pred - Y0_pred

    sample_idx = np.random.choice(len(ite_true), min(500, len(ite_true)), replace=False)
    fig.add_trace(go.Scatter(
        x=ite_true[sample_idx], y=ite_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=5, opacity=0.5),
        name='ITE'
    ), row=1, col=2)

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[ite_true.min(), ite_true.max()],
        y=[ite_true.min(), ite_true.max()],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect',
        showlegend=False
    ), row=1, col=2)

    # 3. Representation space (first 2 dimensions)
    if phi.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=phi[T == 0, 0], y=phi[T == 0, 1],
            mode='markers',
            marker=dict(color='#2D9CDB', size=4, opacity=0.5),
            name='Control'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=phi[T == 1, 0], y=phi[T == 1, 1],
            mode='markers',
            marker=dict(color='#EB5757', size=4, opacity=0.5),
            name='Treated'
        ), row=2, col=1)

    # 4. Y0/Y1 predictions
    fig.add_trace(go.Scatter(
        x=Y0_true[sample_idx], y=Y0_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=4, opacity=0.5),
        name='Y0'
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=Y1_true[sample_idx], y=Y1_pred[sample_idx],
        mode='markers',
        marker=dict(color='#EB5757', size=4, opacity=0.5),
        name='Y1'
    ), row=2, col=2)

    # Add diagonal
    y_min = min(Y0_true.min(), Y1_true.min())
    y_max = max(Y0_true.max(), Y1_true.max())
    fig.add_trace(go.Scatter(
        x=[y_min, y_max], y=[y_min, y_max],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='TARNet Training & Evaluation'
    )

    # Summary
    summary = f"""
## TARNet Training Results

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Sample Size | {n_samples} |
| Hidden Dimension | {hidden_dim} |
| Training Epochs | {n_epochs} |
| IPM Regularization (alpha) | {alpha} |

### Causal Effect Estimation

| Metric | Value |
|--------|-------|
| True ATE | {ate_true:.4f} |
| Predicted ATE | {ate_pred:.4f} |
| ATE Error | {ate_err:.4f} |
| PEHE (ITE Error) | {pehe_val:.4f} |

### Architecture

```
Input (X) --> [Shared Repr] --> Phi(X) --> [Head 0] --> Y(0)
                                      |
                                      +--> [Head 1] --> Y(1)
```

### Key Concepts

- **Shared Representation**: Learn treatment-agnostic features
- **Dual Heads**: Separately estimate two potential outcomes
- **Counterfactual Inference**: ITE = Y(1) - Y(0)
- **IPM Regularization**: When alpha > 0, encourages balanced representations (CFR)
"""

    return fig, summary
