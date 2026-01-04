"""Part 5: Deep Learning for Causal Inference

This module implements deep learning methods for causal effect estimation:
- Representation Learning: Balanced representations and IPM distances
- TARNet: Treatment-Agnostic Representation Network
- DragonNet: Network with propensity head and targeted regularization
- CEVAE: Causal Effect Variational Autoencoder

Key References:
- Shalit et al. (2017): Estimating individual treatment effect: generalization bounds and algorithms
- Shi et al. (2019): Adapting Neural Networks for the Estimation of Treatment Effects
- Louizos et al. (2017): Causal Effect Inference with Deep Latent-Variable Models
"""

from .api import (
    analyze_representation_learning,
    analyze_tarnet,
    analyze_dragonnet,
    analyze_cevae,
)

__all__ = [
    "analyze_representation_learning",
    "analyze_tarnet",
    "analyze_dragonnet",
    "analyze_cevae",
]
