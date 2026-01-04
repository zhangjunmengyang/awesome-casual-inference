"""DragonNet

Architecture:
- Shared representation layer
- Three output heads: Y(0), Y(1), propensity score e(X)
- End-to-end training with propensity head providing regularization

Reference:
- Shi et al., "Adapting Neural Networks for the Estimation of Treatment Effects" (NeurIPS 2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple

from .utils import generate_ihdp_like_data, pehe, ate_error


class DragonNet(nn.Module):
    """
    DragonNet Model

    Architecture:
    X -> [Shared Representation] -> Phi(X)
                                      |
              +----------------------+----------------------+
              |                      |                      |
          [Head 0]              [Head 1]          [Propensity Head]
              |                      |                      |
            Y(0)                   Y(1)                   e(X)

    Key Innovation:
    - Propensity score head provides "targeted regularization"
    - Uses epsilon layer for end-to-end optimization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 200,
        repr_dim: int = 100,
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
                nn.ELU(),  # DragonNet uses ELU
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        repr_layers.append(nn.Linear(prev_dim, repr_dim))
        repr_layers.append(nn.ELU())
        self.representation = nn.Sequential(*repr_layers)

        # Control group output head (Y0)
        self.head0 = self._build_head(repr_dim, hidden_dim, n_head_layers, dropout)

        # Treatment group output head (Y1)
        self.head1 = self._build_head(repr_dim, hidden_dim, n_head_layers, dropout)

        # Propensity score head
        self.propensity_head = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Epsilon layer (for targeted regularization)
        self.epsilon = nn.Parameter(torch.zeros(1))

    def _build_head(self, input_dim, hidden_dim, n_layers, dropout):
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim // 2
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            (y0_pred, y1_pred, propensity, epsilon, representation)
        """
        phi = self.representation(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        propensity = self.propensity_head(phi).squeeze(-1)

        return y0, y1, propensity, self.epsilon, phi

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """Predict individual treatment effect"""
        y0, y1, _, _, _ = self.forward(x)
        return y1 - y0


def dragonnet_loss(
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    y0_pred: torch.Tensor,
    y1_pred: torch.Tensor,
    propensity: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DragonNet Loss Function

    L = L_factual + alpha * L_propensity + beta * L_targeted

    Parameters:
    -----------
    alpha: Propensity loss weight
    beta: Targeted regularization weight
    """
    # Factual loss
    y_pred = torch.where(t_true == 1, y1_pred, y0_pred)
    factual_loss = nn.MSELoss()(y_pred, y_true)

    # Propensity loss (binary cross-entropy)
    propensity_loss = nn.BCELoss()(propensity, t_true)

    # Targeted regularization
    # This is DragonNet's innovation: uses TMLE-style regularization
    t_pred = propensity
    h = t_true / (t_pred + 1e-8) - (1 - t_true) / (1 - t_pred + 1e-8)

    # epsilon is learnable
    targeted_reg = torch.mean((y_true - y_pred - epsilon * h) ** 2)

    total_loss = factual_loss + alpha * propensity_loss + beta * targeted_reg

    return total_loss, factual_loss, propensity_loss, targeted_reg


def train_dragonnet(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int = 200,
    repr_dim: int = 100,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 1.0,
    device: str = 'cpu'
) -> Tuple[DragonNet, dict]:
    """Train DragonNet"""

    # Data preparation
    X_tensor = torch.FloatTensor(X).to(device)
    T_tensor = torch.FloatTensor(T).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = DragonNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        repr_dim=repr_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'total_loss': [],
        'factual_loss': [],
        'propensity_loss': [],
        'targeted_loss': []
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {k: 0 for k in history.keys()}

        for batch_x, batch_t, batch_y in dataloader:
            optimizer.zero_grad()

            y0_pred, y1_pred, propensity, epsilon, _ = model(batch_x)

            total, factual, prop, targeted = dragonnet_loss(
                batch_y, batch_t, y0_pred, y1_pred, propensity, epsilon,
                alpha=alpha, beta=beta
            )

            total.backward()
            optimizer.step()

            epoch_losses['total_loss'] += total.item()
            epoch_losses['factual_loss'] += factual.item()
            epoch_losses['propensity_loss'] += prop.item()
            epoch_losses['targeted_loss'] += targeted.item()

        n_batches = len(dataloader)
        for k in history.keys():
            history[k].append(epoch_losses[k] / n_batches)

    return model, history


def visualize_dragonnet(
    n_samples: int,
    hidden_dim: int,
    n_epochs: int,
    alpha: float,
    beta: float
) -> Tuple[go.Figure, str]:
    """Visualize DragonNet training and performance"""

    # Generate data
    X, T, Y, Y0_true, Y1_true = generate_ihdp_like_data(n_samples)

    # Train model
    model, history = train_dragonnet(
        X, T, Y,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        alpha=alpha,
        beta=beta
    )

    # Predictions
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        Y0_pred, Y1_pred, propensity_pred, epsilon, phi = model(X_tensor)
        Y0_pred = Y0_pred.cpu().numpy()
        Y1_pred = Y1_pred.cpu().numpy()
        propensity_pred = propensity_pred.cpu().numpy()

    # Compute metrics
    pehe_val = pehe(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_err = ate_error(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_true = np.mean(Y1_true - Y0_true)
    ate_pred = np.mean(Y1_pred - Y0_pred)

    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Losses',
            'True vs Predicted ITE',
            'Propensity Score Distribution',
            'Y0 vs Y1 Predictions'
        )
    )

    # 1. Training losses
    for name, color in [
        ('total_loss', '#2D9CDB'),
        ('factual_loss', '#27AE60'),
        ('propensity_loss', '#EB5757'),
        ('targeted_loss', '#9B59B6')
    ]:
        fig.add_trace(go.Scatter(
            y=history[name], mode='lines',
            name=name.replace('_', ' ').title(),
            line=dict(color=color)
        ), row=1, col=1)

    # 2. ITE comparison
    ite_true = Y1_true - Y0_true
    ite_pred = Y1_pred - Y0_pred
    sample_idx = np.random.choice(len(ite_true), min(500, len(ite_true)), replace=False)

    fig.add_trace(go.Scatter(
        x=ite_true[sample_idx], y=ite_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=5, opacity=0.5),
        name='ITE',
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[ite_true.min(), ite_true.max()],
        y=[ite_true.min(), ite_true.max()],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect',
        showlegend=False
    ), row=1, col=2)

    # 3. Propensity score distribution
    fig.add_trace(go.Histogram(
        x=propensity_pred[T == 0],
        name='Control',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=25
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=propensity_pred[T == 1],
        name='Treated',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=25
    ), row=2, col=1)

    # 4. Y0 vs Y1 predictions
    fig.add_trace(go.Scatter(
        x=Y0_true[sample_idx], y=Y0_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=4, opacity=0.5),
        name='Y0',
        showlegend=False
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=Y1_true[sample_idx], y=Y1_pred[sample_idx],
        mode='markers',
        marker=dict(color='#EB5757', size=4, opacity=0.5),
        name='Y1',
        showlegend=False
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
        title_text='DragonNet Training & Evaluation',
        barmode='overlay'
    )

    # Summary
    summary = f"""
## DragonNet Training Results

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Sample Size | {n_samples} |
| Hidden Dimension | {hidden_dim} |
| Training Epochs | {n_epochs} |
| Propensity Weight (alpha) | {alpha} |
| Targeted Weight (beta) | {beta} |

### Causal Effect Estimation

| Metric | Value |
|--------|-------|
| True ATE | {ate_true:.4f} |
| Predicted ATE | {ate_pred:.4f} |
| ATE Error | {ate_err:.4f} |
| PEHE | {pehe_val:.4f} |

### DragonNet Architecture

```
              +---> [Head 0] ---------> Y(0)
              |
X ---> [Shared Repr] ---> Phi(X) ---> [Head 1] ---------> Y(1)
              |
              +---> [Propensity Head] ---> e(X)
                          |
                          v
                     [Epsilon Layer] ---> Targeted Reg
```

### Key Innovations

1. **Propensity Score Head**: End-to-end learning of propensity scores for regularization
2. **Targeted Regularization**: TMLE-style unbiased estimation
3. **Epsilon Layer**: Learnable bias correction parameter

### Loss Function

The total loss is:
```
L = L_factual + α · L_propensity + β · L_targeted
```

Where:
- **L_factual**: Prediction loss on observed outcomes
- **L_propensity**: Binary cross-entropy for treatment prediction
- **L_targeted**: TMLE-style targeted regularization using h(X,T) weights
"""

    return fig, summary
