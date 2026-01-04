"""CEVAE (Causal Effect Variational Autoencoder)

Simplified implementation of CEVAE for causal effect estimation.

Architecture:
- Inference network (encoder): X, T, Y -> latent Z
- Generative network (decoder): Z, T -> Y
- Treatment model: X -> T

Reference:
- Louizos et al. (2017): Causal Effect Inference with Deep Latent-Variable Models
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


class CEVAE(nn.Module):
    """
    Simplified CEVAE Model

    Key components:
    1. Inference network q(z|x,t,y): Encodes to latent space
    2. Generative network p(y|z,t): Decodes from latent space
    3. Treatment model p(t|x): Models treatment assignment
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 20,
        hidden_dim: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Inference network q(z|x,t,y) - Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),  # +2 for t and y
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        # Generative network p(y|z,t) - Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for t
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Treatment model p(t|x)
        self.treatment_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to latent space

        Returns:
            (mu, logvar)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        h = self.encoder(inputs)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space

        Returns:
            y_pred
        """
        inputs = torch.cat([z, t.unsqueeze(-1)], dim=-1)
        return self.decoder(inputs).squeeze(-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            (y_pred, propensity, mu, logvar)
        """
        # Encode
        mu, logvar = self.encode(x, t, y)
        z = self.reparameterize(mu, logvar)

        # Decode
        y_pred = self.decode(z, t)

        # Treatment prediction
        propensity = self.treatment_model(x).squeeze(-1)

        return y_pred, propensity, mu, logvar

    def predict_ite(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Predict ITE using Monte Carlo sampling

        Since we need to marginalize over z:
        E[Y(1) - Y(0)] ≈ (1/n) Σ [decode(z_i, 1) - decode(z_i, 0)]
        where z_i ~ q(z|x, t, y)
        """
        self.eval()
        with torch.no_grad():
            ite_samples = []

            for _ in range(n_samples):
                # Sample from prior (approximation: use N(0,1))
                z = torch.randn(x.size(0), self.latent_dim).to(x.device)

                # Predict under both treatments
                y1 = self.decode(z, torch.ones(x.size(0)).to(x.device))
                y0 = self.decode(z, torch.zeros(x.size(0)).to(x.device))

                ite_samples.append(y1 - y0)

            # Average over samples
            ite = torch.stack(ite_samples).mean(dim=0)

        return ite


def cevae_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    t_true: torch.Tensor,
    propensity: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CEVAE Loss (ELBO + treatment prediction)

    L = Reconstruction_loss + KL_divergence + Treatment_loss
    """
    # Reconstruction loss
    recon_loss = nn.MSELoss()(y_pred, y_true)

    # KL divergence (regularization term)
    # KL(q(z|x,t,y) || p(z)) where p(z) = N(0, I)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Treatment prediction loss
    treatment_loss = nn.BCELoss()(propensity, t_true)

    # Total loss (ELBO with treatment)
    total_loss = recon_loss + beta * kl_loss + treatment_loss

    return total_loss, recon_loss, kl_loss, treatment_loss


def train_cevae(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    latent_dim: int = 20,
    hidden_dim: int = 100,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    beta: float = 1.0,  # KL weight
    device: str = 'cpu'
) -> Tuple[CEVAE, dict]:
    """Train CEVAE"""

    # Data preparation
    X_tensor = torch.FloatTensor(X).to(device)
    T_tensor = torch.FloatTensor(T).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = CEVAE(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'treatment_loss': []
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {k: 0 for k in history.keys()}

        for batch_x, batch_t, batch_y in dataloader:
            optimizer.zero_grad()

            y_pred, propensity, mu, logvar = model(batch_x, batch_t, batch_y)

            total, recon, kl, treatment = cevae_loss(
                batch_y, y_pred, batch_t, propensity, mu, logvar,
                beta=beta
            )

            total.backward()
            optimizer.step()

            epoch_losses['total_loss'] += total.item()
            epoch_losses['recon_loss'] += recon.item()
            epoch_losses['kl_loss'] += kl.item()
            epoch_losses['treatment_loss'] += treatment.item()

        n_batches = len(dataloader)
        for k in history.keys():
            history[k].append(epoch_losses[k] / n_batches)

    return model, history


def visualize_cevae(
    n_samples: int,
    latent_dim: int,
    n_epochs: int,
    beta: float
) -> Tuple[go.Figure, str]:
    """Visualize CEVAE training and performance"""

    # Generate data
    X, T, Y, Y0_true, Y1_true = generate_ihdp_like_data(n_samples)

    # Train model
    model, history = train_cevae(
        X, T, Y,
        latent_dim=latent_dim,
        n_epochs=n_epochs,
        beta=beta
    )

    # Predictions
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)

        # Predict ITE
        ite_pred = model.predict_ite(X_tensor, n_samples=50).cpu().numpy()

        # For Y0 and Y1, we approximate using the latent space
        # Sample z from prior
        z = torch.randn(X.shape[0], latent_dim).to(device)
        Y0_pred = model.decode(z, torch.zeros(X.shape[0]).to(device)).cpu().numpy()
        Y1_pred = model.decode(z, torch.ones(X.shape[0]).to(device)).cpu().numpy()

    # Compute metrics
    pehe_val = pehe(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_err = ate_error(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_true = np.mean(Y1_true - Y0_true)
    ate_pred = np.mean(Y1_pred - Y0_pred)

    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Losses (ELBO)',
            'True vs Predicted ITE',
            'Latent Space Visualization',
            'KL Divergence over Training'
        )
    )

    # 1. Training losses
    for name, color in [
        ('total_loss', '#2D9CDB'),
        ('recon_loss', '#27AE60'),
        ('treatment_loss', '#EB5757')
    ]:
        fig.add_trace(go.Scatter(
            y=history[name], mode='lines',
            name=name.replace('_', ' ').title(),
            line=dict(color=color)
        ), row=1, col=1)

    # 2. ITE comparison
    ite_true = Y1_true - Y0_true
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

    # 3. Latent space (encode all data)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        T_tensor = torch.FloatTensor(T).to(device)
        Y_tensor = torch.FloatTensor(Y).to(device)
        mu, _ = model.encode(X_tensor, T_tensor, Y_tensor)
        mu = mu.cpu().numpy()

    if latent_dim >= 2:
        fig.add_trace(go.Scatter(
            x=mu[T == 0, 0], y=mu[T == 0, 1],
            mode='markers',
            marker=dict(color='#2D9CDB', size=4, opacity=0.5),
            name='Control',
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=mu[T == 1, 0], y=mu[T == 1, 1],
            mode='markers',
            marker=dict(color='#EB5757', size=4, opacity=0.5),
            name='Treated',
            showlegend=False
        ), row=2, col=1)

    # 4. KL divergence
    fig.add_trace(go.Scatter(
        y=history['kl_loss'], mode='lines',
        name='KL Divergence',
        line=dict(color='#9B59B6'),
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='CEVAE Training & Evaluation'
    )

    # Summary
    summary = f"""
## CEVAE Training Results

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Sample Size | {n_samples} |
| Latent Dimension | {latent_dim} |
| Training Epochs | {n_epochs} |
| KL Weight (beta) | {beta} |

### Causal Effect Estimation

| Metric | Value |
|--------|-------|
| True ATE | {ate_true:.4f} |
| Predicted ATE | {ate_pred:.4f} |
| ATE Error | {ate_err:.4f} |
| PEHE | {pehe_val:.4f} |

### CEVAE Architecture

```
Inference Network (Encoder):
X, T, Y -> [Neural Net] -> μ(z), σ(z)
                           |
                           v
                        z ~ q(z|X,T,Y)

Generative Network (Decoder):
z, T -> [Neural Net] -> Y_pred

Treatment Model:
X -> [Neural Net] -> p(T|X)
```

### Key Concepts

1. **Variational Inference**: Uses VAE framework to learn latent confounders
2. **Latent Space Z**: Captures unobserved confounders
3. **ELBO Objective**: Evidence Lower Bound for probabilistic inference
4. **Monte Carlo ITE**: Marginalizes over latent space for counterfactual prediction

### Loss Function

```
L = L_reconstruction + β · KL(q(z|X,T,Y) || p(z)) + L_treatment
```

### Notes

This is a simplified implementation. Full CEVAE includes:
- Proximal causal inference for unobserved confounding
- More sophisticated generative models
- Importance weighting for treatment effect estimation
"""

    return fig, summary
