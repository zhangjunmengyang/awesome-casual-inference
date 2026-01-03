"""
异质性处理效应工具函数

提供数据生成、评估指标等通用功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_heterogeneous_data(
    n_samples: int = 5000,
    n_features: int = 10,
    effect_heterogeneity: str = 'moderate',
    confounding_strength: float = 0.5,
    noise_level: float = 0.5,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成具有异质性处理效应的数据

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    effect_heterogeneity: 效应异质性强度
        - 'weak': 弱异质性 (主要是常数效应)
        - 'moderate': 中等异质性 (线性依赖于特征)
        - 'strong': 强异质性 (非线性、复杂交互)
    confounding_strength: 混淆强度 (0-1)
    noise_level: 噪声水平
    seed: 随机种子

    Returns:
    --------
    (DataFrame, true_cate, Y0_true, Y1_true)
    DataFrame columns: X1...Xn, T, Y
    true_cate: 真实的条件平均处理效应
    Y0_true: 真实的潜在结果 Y(0)
    Y1_true: 真实的潜在结果 Y(1)
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成特征 (混合连续和二值特征)
    n_cont = n_features // 2
    n_binary = n_features - n_cont

    X_cont = np.random.randn(n_samples, n_cont)
    X_binary = np.random.binomial(1, 0.5, (n_samples, n_binary))
    X = np.concatenate([X_cont, X_binary], axis=1)

    # 倾向得分 (含混淆)
    propensity_base = 0.5
    if confounding_strength > 0:
        # 倾向得分依赖于前几个特征
        propensity_logit = (
            confounding_strength * (
                0.5 * X[:, 0] +
                0.3 * X[:, 1] +
                0.2 * X[:, 2] if n_features >= 3 else 0.5 * X[:, 0]
            )
        )
        propensity = 1 / (1 + np.exp(-propensity_logit))
    else:
        propensity = np.full(n_samples, propensity_base)

    # 处理分配
    T = np.random.binomial(1, propensity)

    # 基线结果 Y(0)
    baseline = (
        5.0 +
        1.0 * X[:, 0] +
        0.5 * X[:, 1] +
        0.3 * X[:, 2] if n_features >= 3 else 5.0 + 1.0 * X[:, 0]
    )

    # 异质性处理效应
    if effect_heterogeneity == 'weak':
        # 主要是常数效应 + 小的异质性
        tau = 3.0 + 0.5 * X[:, 0]

    elif effect_heterogeneity == 'moderate':
        # 线性异质性效应
        tau = (
            2.0 +
            1.5 * X[:, 0] -
            1.0 * X[:, 1] +
            0.5 * X[:, 2] if n_features >= 3 else 2.0 + 1.5 * X[:, 0]
        )

    elif effect_heterogeneity == 'strong':
        # 强非线性异质性
        tau = (
            2.0 +
            2.0 * np.sin(X[:, 0]) +
            1.5 * (X[:, 1] ** 2) +
            1.0 * X[:, 0] * X[:, 1] +
            0.8 * (X[:, 2] > 0) if n_features >= 3 else (
                2.0 + 2.0 * np.sin(X[:, 0]) + 1.5 * (X[:, 1] ** 2)
            )
        )

    else:
        raise ValueError(f"Unknown heterogeneity type: {effect_heterogeneity}")

    # 潜在结果
    noise = np.random.randn(n_samples) * noise_level
    Y0_true = baseline + noise
    Y1_true = baseline + tau + noise

    # 观测结果
    Y = np.where(T == 1, Y1_true, Y0_true)

    # 创建 DataFrame
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['T'] = T
    df['Y'] = Y

    return df, tau, Y0_true, Y1_true


def compute_pehe(
    y0_true: np.ndarray,
    y1_true: np.ndarray,
    y0_pred: np.ndarray,
    y1_pred: np.ndarray
) -> float:
    """
    计算 PEHE (Precision in Estimation of Heterogeneous Treatment Effect)

    PEHE = sqrt(E[(ITE_true - ITE_pred)^2])

    这是衡量个体处理效应估计精度的黄金标准。

    Parameters:
    -----------
    y0_true: 真实的 Y(0)
    y1_true: 真实的 Y(1)
    y0_pred: 预测的 Y(0)
    y1_pred: 预测的 Y(1)

    Returns:
    --------
    PEHE 值 (越小越好)
    """
    ite_true = y1_true - y0_true
    ite_pred = y1_pred - y0_pred
    return np.sqrt(np.mean((ite_true - ite_pred) ** 2))


def compute_ate_bias(
    y0_true: np.ndarray,
    y1_true: np.ndarray,
    y0_pred: np.ndarray,
    y1_pred: np.ndarray
) -> float:
    """
    计算 ATE 估计偏差

    ATE_bias = |E[ITE_true] - E[ITE_pred]|

    这衡量平均处理效应估计的偏差。

    Parameters:
    -----------
    y0_true: 真实的 Y(0)
    y1_true: 真实的 Y(1)
    y0_pred: 预测的 Y(0)
    y1_pred: 预测的 Y(1)

    Returns:
    --------
    ATE 偏差 (越小越好)
    """
    ate_true = np.mean(y1_true - y0_true)
    ate_pred = np.mean(y1_pred - y0_pred)
    return np.abs(ate_true - ate_pred)


def identify_subgroups(
    X: np.ndarray,
    cate: np.ndarray,
    n_groups: int = 4
) -> np.ndarray:
    """
    根据 CATE 大小将样本分为若干子群体

    Parameters:
    -----------
    X: 特征矩阵
    cate: 条件平均处理效应
    n_groups: 分组数量

    Returns:
    --------
    group_labels: 每个样本的组别标签 (0 to n_groups-1)
    """
    # 根据 CATE 分位数分组
    quantiles = np.linspace(0, 1, n_groups + 1)
    thresholds = np.quantile(cate, quantiles)

    group_labels = np.zeros(len(cate), dtype=int)
    for i in range(n_groups):
        if i == n_groups - 1:
            mask = cate >= thresholds[i]
        else:
            mask = (cate >= thresholds[i]) & (cate < thresholds[i + 1])
        group_labels[mask] = i

    return group_labels


def compute_policy_value(
    Y: np.ndarray,
    T: np.ndarray,
    cate_pred: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    计算基于 CATE 的策略价值

    策略: 只对预测 CATE > threshold 的样本进行处理

    Parameters:
    -----------
    Y: 观测结果
    T: 处理状态
    cate_pred: 预测的 CATE
    threshold: CATE 阈值

    Returns:
    --------
    策略下的平均结果
    """
    # 策略决策
    should_treat = cate_pred > threshold

    # 使用逆概率加权估计策略价值
    propensity = np.mean(T)

    # 简化版: 只用观测数据估计
    # 实际应用中应该用更严格的方法
    treated_mask = (T == 1) & should_treat
    control_mask = (T == 0) & (~should_treat)

    if treated_mask.sum() > 0 and control_mask.sum() > 0:
        treated_value = np.mean(Y[treated_mask])
        control_value = np.mean(Y[control_mask])

        # 策略价值 = 被处理样本的平均结果 * 处理比例 + 未处理样本的平均结果 * 未处理比例
        treat_fraction = should_treat.mean()
        policy_value = treated_value * treat_fraction + control_value * (1 - treat_fraction)
    else:
        policy_value = np.mean(Y)

    return policy_value


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 R² (决定系数)

    R² = 1 - SSE / SST

    Parameters:
    -----------
    y_true: 真实值
    y_pred: 预测值

    Returns:
    --------
    R² 值 (越接近 1 越好)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)
