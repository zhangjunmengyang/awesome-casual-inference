"""Part 5 Deep Learning API Adapter Layer

Converts visualization functions to API-compatible format.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from .representation_learning import (
    train_with_ipm_regularization,
    visualize_representations,
    compare_balance_metrics
)
from .tarnet import visualize_tarnet
from .dragonnet import visualize_dragonnet
from .cevae import visualize_cevae
from .utils import generate_ihdp_like_data


def _fig_to_chart_data(fig: go.Figure) -> dict:
    """Convert Plotly Figure to frontend-compatible chart data"""
    return fig.to_dict()


def analyze_representation_learning(
    n_samples: int = 1000,
    imbalance_level: float = 0.5,
    alpha_no_reg: float = 0.0,
    alpha_with_reg: float = 1.0,
    n_epochs: int = 100
) -> dict:
    """
    Analyze representation learning with and without IPM regularization

    Parameters:
    -----------
    n_samples: Number of samples
    imbalance_level: Level of covariate imbalance (0-1)
    alpha_no_reg: IPM weight without regularization (should be 0)
    alpha_with_reg: IPM weight with regularization (>0)
    n_epochs: Training epochs
    """

    # Generate data with imbalance
    np.random.seed(42)
    X, T, Y, Y0_true, Y1_true = generate_ihdp_like_data(n_samples)

    # Introduce imbalance by biasing treatment assignment
    if imbalance_level > 0:
        # Make treatment assignment depend more on X
        propensity = 1 / (1 + np.exp(-imbalance_level * (X[:, 0] + X[:, 1])))
        T = np.random.binomial(1, propensity)
        Y = np.where(T == 1, Y1_true, Y0_true)

    # Train without regularization
    model_no_reg, history_no_reg = train_with_ipm_regularization(
        X, T, Y,
        alpha=alpha_no_reg,
        n_epochs=n_epochs
    )

    # Train with regularization
    model_with_reg, history_with_reg = train_with_ipm_regularization(
        X, T, Y,
        alpha=alpha_with_reg,
        n_epochs=n_epochs
    )

    # Get representations
    model_no_reg.eval()
    model_with_reg.eval()
    with torch.no_grad():
        device_no_reg = next(model_no_reg.parameters()).device
        device_with_reg = next(model_with_reg.parameters()).device

        X_tensor_no_reg = torch.FloatTensor(X).to(device_no_reg)
        X_tensor_with_reg = torch.FloatTensor(X).to(device_with_reg)

        phi_no_reg, _ = model_no_reg(X_tensor_no_reg)
        phi_with_reg, _ = model_with_reg(X_tensor_with_reg)

        phi_no_reg = phi_no_reg.cpu().numpy()
        phi_with_reg = phi_with_reg.cpu().numpy()

    # Visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Loss (No Regularization)',
            'Training Loss (With IPM Regularization)',
            'Representations (No Regularization)',
            'Representations (With IPM Regularization)'
        ),
        vertical_spacing=0.12
    )

    # 1. Training loss (no regularization)
    fig.add_trace(go.Scatter(
        y=history_no_reg['total_loss'],
        mode='lines',
        name='Total Loss',
        line=dict(color='#2D9CDB')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=history_no_reg['balance_metric'],
        mode='lines',
        name='Balance Metric',
        line=dict(color='#EB5757'),
        yaxis='y2'
    ), row=1, col=1)

    # 2. Training loss (with regularization)
    fig.add_trace(go.Scatter(
        y=history_with_reg['total_loss'],
        mode='lines',
        name='Total Loss',
        line=dict(color='#2D9CDB'),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        y=history_with_reg['balance_metric'],
        mode='lines',
        name='Balance Metric',
        line=dict(color='#EB5757'),
        showlegend=False
    ), row=1, col=2)

    # 3. Representations (no regularization)
    if phi_no_reg.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=phi_no_reg[T == 0, 0],
            y=phi_no_reg[T == 0, 1],
            mode='markers',
            name='Control',
            marker=dict(color='#2D9CDB', opacity=0.5, size=4)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=phi_no_reg[T == 1, 0],
            y=phi_no_reg[T == 1, 1],
            mode='markers',
            name='Treatment',
            marker=dict(color='#EB5757', opacity=0.5, size=4)
        ), row=2, col=1)

    # 4. Representations (with regularization)
    if phi_with_reg.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=phi_with_reg[T == 0, 0],
            y=phi_with_reg[T == 0, 1],
            mode='markers',
            name='Control',
            marker=dict(color='#2D9CDB', opacity=0.5, size=4),
            showlegend=False
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=phi_with_reg[T == 1, 0],
            y=phi_with_reg[T == 1, 1],
            mode='markers',
            name='Treatment',
            marker=dict(color='#EB5757', opacity=0.5, size=4),
            showlegend=False
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='Representation Learning: IPM Regularization Effect'
    )

    # Balance metrics
    balance_metrics = compare_balance_metrics(X, T, phi_no_reg, phi_with_reg)

    summary = f"""
## Representation Learning Analysis

### Setup

| Parameter | Value |
|-----------|-------|
| Sample Size | {n_samples} |
| Imbalance Level | {imbalance_level:.2f} |
| IPM Weight (No Reg) | {alpha_no_reg} |
| IPM Weight (With Reg) | {alpha_with_reg} |
| Training Epochs | {n_epochs} |

### Balance Metrics (L2 Distance)

| Metric | Value |
|--------|-------|
| Original Features | {balance_metrics['original']:.4f} |
| No Regularization | {balance_metrics['no_regularization']:.4f} |
| With IPM Regularization | {balance_metrics['with_regularization']:.4f} |
| **Improvement** | **{balance_metrics['improvement']:.1f}%** |

### Key Concepts

**Representation Learning** aims to learn features Φ(X) that are:
1. **Predictive**: Useful for outcome prediction
2. **Balanced**: Similar distributions across treatment groups

**IPM (Integral Probability Metric)** measures distribution distance:
- Linear MMD: ||E[Φ(X)|T=1] - E[Φ(X)|T=0]||
- Goal: Minimize this distance while maintaining prediction accuracy

### Why Balance Matters

- **Unbalanced representations**: Treatment groups differ in learned features
- **Confounding**: Differences may reflect selection bias, not treatment effect
- **Balanced representations**: Enable better counterfactual prediction

### Regularization Trade-off

```
L = L_prediction + α · IPM(Φ_T=1, Φ_T=0)
```

- α = 0: Pure supervised learning (may be unbalanced)
- α > 0: Counterfactual Regression (CFR) - forces balance
- Higher α: More balance, but may sacrifice prediction accuracy
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "balance_original": balance_metrics['original'],
            "balance_no_reg": balance_metrics['no_regularization'],
            "balance_with_reg": balance_metrics['with_regularization'],
            "improvement_pct": balance_metrics['improvement']
        }
    }


def analyze_tarnet(
    n_samples: int = 1000,
    hidden_dim: int = 100,
    num_layers: int = 3,
    alpha: float = 0.0,
    n_epochs: int = 100
) -> dict:
    """
    Analyze TARNet model

    Parameters:
    -----------
    n_samples: Number of samples
    hidden_dim: Hidden layer dimension
    num_layers: Number of representation layers
    alpha: IPM regularization weight (0 = TARNet, >0 = CFR)
    n_epochs: Training epochs
    """

    fig, summary = visualize_tarnet(
        n_samples=n_samples,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        alpha=alpha
    )

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {}
    }


def analyze_dragonnet(
    n_samples: int = 1000,
    hidden_dim: int = 200,
    targeted_reg: float = 1.0,
    alpha: float = 1.0,
    n_epochs: int = 100
) -> dict:
    """
    Analyze DragonNet model

    Parameters:
    -----------
    n_samples: Number of samples
    hidden_dim: Hidden layer dimension
    targeted_reg: Targeted regularization weight (beta)
    alpha: Propensity loss weight
    n_epochs: Training epochs
    """

    fig, summary = visualize_dragonnet(
        n_samples=n_samples,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        alpha=alpha,
        beta=targeted_reg
    )

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {}
    }


def analyze_cevae(
    n_samples: int = 1000,
    latent_dim: int = 20,
    beta: float = 1.0,
    n_epochs: int = 100
) -> dict:
    """
    Analyze CEVAE model (simplified version)

    Parameters:
    -----------
    n_samples: Number of samples
    latent_dim: Latent space dimension
    beta: KL divergence weight
    n_epochs: Training epochs
    """

    fig, summary = visualize_cevae(
        n_samples=n_samples,
        latent_dim=latent_dim,
        n_epochs=n_epochs,
        beta=beta
    )

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {}
    }
