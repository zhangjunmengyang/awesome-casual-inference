"""
Datasets Module - 因果推断数据集

提供经典因果推断数据集和合成数据生成器

Available Datasets:
------------------
- LaLonde: 就业培训项目数据集
- IHDP: 婴儿健康发展计划数据集
- Synthetic: 多种合成数据生成器

Usage:
------
>>> from datasets import load_lalonde, generate_linear_dgp
>>> df = load_lalonde()
>>> X, T, Y, true_ite = generate_linear_dgp(n_samples=1000)
"""

from .lalonde import load_lalonde
from .ihdp import load_ihdp, generate_ihdp_semi_synthetic
from .synthetic import (
    generate_linear_dgp,
    generate_nonlinear_dgp,
    generate_heterogeneous_dgp
)
from .utils import train_test_split_causal, describe_dataset

__all__ = [
    # LaLonde
    'load_lalonde',

    # IHDP
    'load_ihdp',
    'generate_ihdp_semi_synthetic',

    # Synthetic
    'generate_linear_dgp',
    'generate_nonlinear_dgp',
    'generate_heterogeneous_dgp',

    # Utils
    'train_test_split_causal',
    'describe_dataset',
]

__version__ = '1.0.0'
