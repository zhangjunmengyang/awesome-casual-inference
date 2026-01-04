"""
因果推断基础工具函数

提供数据生成、可视化辅助等通用功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import plotly.graph_objects as go


def generate_potential_outcomes_data(
    n_samples: int = 1000,
    treatment_effect: float = 2.0,
    baseline_outcome: float = 5.0,
    noise_std: float = 1.0,
    treatment_prob: float = 0.5,
    confounding_strength: float = 0.0,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    生成潜在结果数据

    Parameters:
    -----------
    n_samples: 样本数量
    treatment_effect: 真实处理效应 (ATE)
    baseline_outcome: 基线结果
    noise_std: 噪声标准差
    treatment_prob: 处理概率 (无混淆时)
    confounding_strength: 混淆强度 (0-1)
    seed: 随机种子

    Returns:
    --------
    DataFrame with columns: X, T, Y0, Y1, Y, ITE
    """
    if seed is not None:
        np.random.seed(seed)

    # 协变量
    X = np.random.randn(n_samples)

    # 潜在结果
    # Y(0) = baseline + noise
    # Y(1) = baseline + treatment_effect + noise
    noise_0 = np.random.randn(n_samples) * noise_std
    noise_1 = np.random.randn(n_samples) * noise_std

    Y0 = baseline_outcome + 0.5 * X + noise_0
    Y1 = baseline_outcome + treatment_effect + 0.5 * X + noise_1

    # 处理分配 (考虑混淆)
    # 混淆: X 影响 T 的概率
    if confounding_strength > 0:
        propensity = 1 / (1 + np.exp(-confounding_strength * X))
        T = np.random.binomial(1, propensity)
    else:
        T = np.random.binomial(1, treatment_prob, n_samples)

    # 观测结果 (只能观测到一个)
    Y = np.where(T == 1, Y1, Y0)

    # 个体处理效应 (真实值，通常不可观测)
    ITE = Y1 - Y0

    return pd.DataFrame({
        'X': X,
        'T': T,
        'Y0': Y0,
        'Y1': Y1,
        'Y': Y,
        'ITE': ITE,
        'propensity': 1 / (1 + np.exp(-confounding_strength * X)) if confounding_strength > 0 else np.full(n_samples, treatment_prob)
    })


def generate_confounded_data(
    n_samples: int = 1000,
    true_ate: float = 2.0,
    confounding_strength: float = 2.0,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    生成带混淆的观测数据

    DAG: X -> T, X -> Y, T -> Y

    Returns:
    --------
    DataFrame and dict with true parameters
    """
    if seed is not None:
        np.random.seed(seed)

    # 混淆变量
    X = np.random.randn(n_samples)

    # 处理分配受 X 影响
    propensity = 1 / (1 + np.exp(-confounding_strength * X))
    T = np.random.binomial(1, propensity)

    # 结果受 X 和 T 影响
    noise = np.random.randn(n_samples) * 0.5
    Y = 1.0 + true_ate * T + confounding_strength * X + noise

    df = pd.DataFrame({
        'X': X,
        'T': T,
        'Y': Y,
        'propensity': propensity
    })

    params = {
        'true_ate': true_ate,
        'confounding_strength': confounding_strength,
        'n_samples': n_samples
    }

    return df, params


def calculate_naive_ate(df: pd.DataFrame) -> float:
    """计算朴素 ATE 估计 (简单差分)"""
    treated = df[df['T'] == 1]['Y'].mean()
    control = df[df['T'] == 0]['Y'].mean()
    return treated - control


def calculate_adjusted_ate(df: pd.DataFrame, adjustment_vars: list = ['X']) -> float:
    """计算调整后的 ATE (线性回归)"""
    from sklearn.linear_model import LinearRegression

    X_vars = df[adjustment_vars + ['T']].values
    y = df['Y'].values

    model = LinearRegression()
    model.fit(X_vars, y)

    # 处理效应系数
    return model.coef_[-1]


def fig_to_dict(fig: go.Figure) -> dict:
    """将 Plotly Figure 转换为字典格式"""
    return fig.to_dict()
