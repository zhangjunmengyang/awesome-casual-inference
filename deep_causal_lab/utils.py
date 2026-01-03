"""
深度因果模型工具函数
"""

import numpy as np
import pandas as pd
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
    生成类似 IHDP 的半合成数据

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    hidden_confounding: 是否有隐藏混淆
    seed: 随机种子

    Returns:
    --------
    (X, T, Y, Y0, Y1) - 特征、处理、观测结果、潜在结果
    """
    if seed is not None:
        np.random.seed(seed)

    # 特征 (混合连续和二值)
    X_cont = np.random.randn(n_samples, n_features // 2)
    X_bin = np.random.binomial(1, 0.5, (n_samples, n_features - n_features // 2))
    X = np.concatenate([X_cont, X_bin], axis=1)

    # 倾向得分 (非线性)
    propensity_score = 1 / (1 + np.exp(-(
        0.5 * X[:, 0] +
        0.3 * X[:, 1] * X[:, 2] +
        0.2 * X[:, 3] -
        0.4 * X[:, 4]
    )))

    # 处理分配
    T = np.random.binomial(1, propensity_score)

    # 潜在结果 (非线性)
    # Y(0)
    Y0 = (
        1 +
        0.5 * X[:, 0] +
        0.3 * X[:, 1] ** 2 +
        0.2 * np.sin(X[:, 2]) +
        0.1 * X[:, 3] * X[:, 4] +
        np.random.randn(n_samples) * 0.5
    )

    # Y(1) - 异质性效应
    treatment_effect = (
        2 +
        0.8 * X[:, 0] -
        0.3 * X[:, 1] +
        0.5 * X[:, 2] * (X[:, 2] > 0)
    )

    Y1 = Y0 + treatment_effect

    # 观测结果
    Y = np.where(T == 1, Y1, Y0)

    return X, T, Y, Y0, Y1


def generate_twins_like_data(
    n_samples: int = 2000,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成类似 Twins 数据集的数据

    模拟双胞胎婴儿死亡率研究
    """
    if seed is not None:
        np.random.seed(seed)

    # 特征: 母亲特征
    age = np.random.uniform(15, 45, n_samples)
    education = np.random.randint(0, 18, n_samples)
    prenatal_visits = np.random.poisson(10, n_samples)
    smoking = np.random.binomial(1, 0.2, n_samples)
    weight_gain = np.random.normal(30, 10, n_samples)

    X = np.column_stack([age, education, prenatal_visits, smoking, weight_gain])

    # 标准化
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # 处理: 是否早产
    propensity = 1 / (1 + np.exp(-(
        -0.5 +
        0.3 * X[:, 3] +  # 吸烟增加早产风险
        -0.2 * X[:, 1] +  # 教育降低风险
        -0.1 * X[:, 2]    # 产检降低风险
    )))
    T = np.random.binomial(1, propensity)

    # 潜在结果: 死亡风险
    base_risk = 1 / (1 + np.exp(-(
        -2 +
        0.5 * X[:, 3] +   # 吸烟
        -0.3 * X[:, 1] +  # 教育
        -0.2 * X[:, 2]    # 产检
    )))

    # 早产增加死亡风险，但效应因人而异
    treatment_effect = 0.1 + 0.05 * X[:, 3] - 0.02 * X[:, 2]

    Y0 = np.random.binomial(1, base_risk)
    Y1 = np.random.binomial(1, np.clip(base_risk + treatment_effect, 0, 1))

    Y = np.where(T == 1, Y1, Y0)

    return X, T, Y, Y0, Y1


class EarlyStopping:
    """早停机制"""

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
    计算 PEHE (Precision in Estimation of Heterogeneous Effect)

    sqrt(E[(tau_true - tau_pred)^2])
    """
    tau_true = y1_true - y0_true
    tau_pred = y1_pred - y0_pred
    return np.sqrt(np.mean((tau_true - tau_pred) ** 2))


def ate_error(y0_true: np.ndarray, y1_true: np.ndarray,
              y0_pred: np.ndarray, y1_pred: np.ndarray) -> float:
    """
    计算 ATE 估计误差

    |E[tau_true] - E[tau_pred]|
    """
    ate_true = np.mean(y1_true - y0_true)
    ate_pred = np.mean(y1_pred - y0_pred)
    return np.abs(ate_true - ate_pred)
