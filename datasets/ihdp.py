"""
IHDP (Infant Health and Development Program) 数据集

经典的 CATE (Conditional Average Treatment Effect) 评估基准数据集。
用于评估异质性处理效应估计方法。

数据来源:
---------
Hill, J. L. (2011). "Bayesian Nonparametric Modeling for Causal Inference".
Journal of Computational and Graphical Statistics.

数据特征:
---------
- 样本量: 747 (随机化实验)
- 协变量: 25 个 (连续 + 离散)
- 处理: 早期儿童教育干预
- 结果: 儿童认知测试分数

特点:
-----
1. 真实数据基础: 基于真实 RCT 实验
2. 半合成: 通过非线性响应函数生成结果
3. 异质性效应: 处理效应随协变量变化
4. 评估友好: 已知真实 ITE，便于评估
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# IHDP 原始数据统计特征 (用于生成模拟数据)
IHDP_FEATURE_SPECS = {
    # 连续特征 (特征名, 均值, 标准差)
    'continuous': [
        ('birth_weight', 2900, 600),     # 出生体重 (克)
        ('head_circumference', 34, 2),    # 头围 (厘米)
        ('gestational_age', 38, 2.5),     # 孕周
        ('mother_age', 24, 6),            # 母亲年龄
        ('prenatal_visits', 10, 4),       # 产前检查次数
    ],
    # 离散特征 (特征名, 类别数, 概率分布)
    'categorical': [
        ('male', 2, [0.5, 0.5]),                    # 性别
        ('twin', 2, [0.96, 0.04]),                  # 是否双胞胎
        ('premature', 2, [0.75, 0.25]),             # 是否早产
        ('mother_education', 4, [0.20, 0.35, 0.30, 0.15]),  # 母亲教育程度
        ('father_education', 4, [0.25, 0.35, 0.25, 0.15]),  # 父亲教育程度
        ('race_white', 2, [0.50, 0.50]),            # 白人
        ('race_black', 2, [0.40, 0.60]),            # 黑人
        ('alcohol_use', 2, [0.85, 0.15]),           # 孕期饮酒
        ('smoking', 2, [0.70, 0.30]),               # 孕期吸烟
        ('low_income', 2, [0.60, 0.40]),            # 低收入家庭
    ]
}


def load_ihdp(
    n_samples: int = 747,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    加载 IHDP 数据集 (模拟版本)

    生成基于原始 IHDP 统计特征的模拟数据

    Parameters:
    -----------
    n_samples: 样本数量 (默认 747，接近原始数据)
    seed: 随机种子

    Returns:
    --------
    DataFrame with columns:
        - X1-X25: 协变量
        - treatment: 处理状态 (0/1)
        - y_factual: 观测结果
        - (用于评估时可通过 generate_ihdp_semi_synthetic 获取反事实结果)

    Examples:
    ---------
    >>> df = load_ihdp()
    >>> print(df.shape)  # (747, 27)
    >>> print(df['treatment'].value_counts())
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成协变量
    X = _generate_ihdp_covariates(n_samples)

    # 随机分配处理 (模拟 RCT，不完全平衡)
    # 原始 IHDP: 608 控制, 139 处理
    treatment_prob = 139 / 747
    treatment = np.random.binomial(1, treatment_prob, n_samples)

    # 添加到 DataFrame
    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    df['treatment'] = treatment

    # 生成观测结果 (简化版本)
    y0, y1 = _generate_ihdp_outcomes(X, seed=seed)
    df['y_factual'] = np.where(treatment == 1, y1, y0)

    return df


def generate_ihdp_semi_synthetic(
    n_samples: int = 747,
    setting: str = 'A',
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 IHDP 半合成数据 (用于 CATE 评估)

    基于 Hill (2011) 的半合成数据生成过程

    Parameters:
    -----------
    n_samples: 样本数量
    setting: 响应函数设置
        - 'A': 设置 A (中等非线性)
        - 'B': 设置 B (高度非线性)
    seed: 随机种子

    Returns:
    --------
    (X, T, Y, true_ite)
        X: 协变量矩阵 (n_samples, 25)
        T: 处理状态 (n_samples,)
        Y: 观测结果 (n_samples,)
        true_ite: 真实个体处理效应 (n_samples,)

    Examples:
    ---------
    >>> X, T, Y, true_ite = generate_ihdp_semi_synthetic(n_samples=747)
    >>> print(f"True ATE: {true_ite.mean():.3f}")
    >>> print(f"CATE std: {true_ite.std():.3f}")
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成协变量
    X = _generate_ihdp_covariates(n_samples)

    # 处理分配 (带轻微混淆以模拟真实情况)
    propensity = _compute_ihdp_propensity(X)
    T = np.random.binomial(1, propensity)

    # 生成潜在结果
    if setting == 'A':
        y0, y1 = _generate_ihdp_outcomes_setting_a(X, seed=seed)
    elif setting == 'B':
        y0, y1 = _generate_ihdp_outcomes_setting_b(X, seed=seed)
    else:
        raise ValueError(f"Unknown setting: {setting}. Choose 'A' or 'B'")

    # 观测结果
    Y = np.where(T == 1, y1, y0)

    # 真实 ITE
    true_ite = y1 - y0

    return X, T, Y, true_ite


def _generate_ihdp_covariates(n_samples: int) -> np.ndarray:
    """
    生成 IHDP 协变量 (25 维)

    基于原始数据的统计特征
    """
    features = []

    # 连续特征 (5个)
    for feat_name, mean, std in IHDP_FEATURE_SPECS['continuous']:
        feat = np.random.normal(mean, std, n_samples)
        features.append(feat)

    # 离散特征 (10个，每个转为 one-hot 或直接)
    for feat_name, n_categories, probs in IHDP_FEATURE_SPECS['categorical']:
        feat = np.random.choice(n_categories, n_samples, p=probs)
        features.append(feat)

    # 额外衍生特征 (达到 25 维)
    # 交互特征
    features.append(features[0] * features[2])  # birth_weight * gestational_age
    features.append(features[3] * features[8])  # mother_age * mother_education
    features.append(features[0] / (features[1] + 1))  # birth_weight / head_circumference
    features.append(features[4] ** 2)  # prenatal_visits^2
    features.append(np.sqrt(np.abs(features[0])))  # sqrt(birth_weight)

    # 二次特征
    features.append(features[0] ** 2)
    features.append(features[1] ** 2)
    features.append(features[2] ** 2)
    features.append(features[3] ** 2)
    features.append(features[4] ** 2)

    # 堆叠为矩阵
    X = np.column_stack(features)

    # 标准化 (保持在合理范围)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X


def _compute_ihdp_propensity(X: np.ndarray) -> np.ndarray:
    """
    计算倾向得分 (带轻微混淆)

    倾向得分轻微依赖于协变量，但仍接近 RCT
    """
    # 基础概率
    base_prob = 139 / 747  # ≈ 0.186

    # 轻微依赖前几个协变量
    logit = np.log(base_prob / (1 - base_prob)) + 0.1 * X[:, 0] - 0.05 * X[:, 1]

    propensity = 1 / (1 + np.exp(-logit))

    # 限制在合理范围
    propensity = np.clip(propensity, 0.1, 0.4)

    return propensity


def _generate_ihdp_outcomes(
    X: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 IHDP 结果 (简化版本)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = X.shape[0]

    # Y(0) - 控制组结果
    # 基于多个协变量的非线性组合
    y0 = (
        100 +
        2 * X[:, 0] +
        3 * X[:, 1] +
        1.5 * X[:, 2] +
        0.5 * X[:, 0] * X[:, 1] +
        0.3 * X[:, 2] ** 2
    )
    y0 += np.random.randn(n_samples) * 5  # 噪声

    # 处理效应 (异质性)
    # 效应取决于协变量
    tau = 4 + 2 * X[:, 0] - X[:, 1]

    # Y(1) - 处理组结果
    y1 = y0 + tau

    return y0, y1


def _generate_ihdp_outcomes_setting_a(
    X: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IHDP 设置 A: 中等非线性响应函数
    (基于 Hill 2011 的响应函数设计)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = X.shape[0]

    # Y(0) - 非线性基线响应
    y0 = (
        100 +
        np.exp(0.5 * X[:, 0]) +
        X[:, 1] / (1 + np.exp(-X[:, 2])) +
        2 * X[:, 3] +
        3 * np.sin(X[:, 4]) +
        X[:, 5] * X[:, 6]
    )

    # 处理效应 (异质性，中等非线性)
    tau = (
        4 +
        2 * X[:, 0] +
        X[:, 1] * X[:, 2] -
        1.5 * X[:, 3] +
        0.5 * X[:, 4] ** 2
    )

    # 噪声
    noise = np.random.randn(n_samples) * 5

    y0 += noise
    y1 = y0 + tau

    return y0, y1


def _generate_ihdp_outcomes_setting_b(
    X: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IHDP 设置 B: 高度非线性响应函数
    (更具挑战性)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = X.shape[0]

    # Y(0) - 高度非线性
    y0 = (
        100 +
        5 * np.sin(X[:, 0] * X[:, 1]) +
        np.exp(0.3 * X[:, 2]) +
        X[:, 3] ** 3 / 10 +
        np.log(np.abs(X[:, 4]) + 1) * X[:, 5] +
        (X[:, 6] + 1) / (np.abs(X[:, 7]) + 1)
    )

    # 处理效应 (高度异质性)
    tau = (
        5 +
        3 * np.tanh(X[:, 0]) +
        2 * X[:, 1] ** 2 +
        X[:, 2] * np.sin(X[:, 3]) -
        1.5 * np.abs(X[:, 4]) +
        0.5 * X[:, 5] * X[:, 6] * X[:, 7]
    )

    # 更大噪声
    noise = np.random.randn(n_samples) * 8

    y0 += noise
    y1 = y0 + tau

    return y0, y1


def get_ihdp_statistics(X: np.ndarray, T: np.ndarray, Y: np.ndarray, true_ite: np.ndarray) -> dict:
    """
    计算 IHDP 数据统计摘要

    Parameters:
    -----------
    X: 协变量矩阵
    T: 处理状态
    Y: 观测结果
    true_ite: 真实 ITE

    Returns:
    --------
    Dictionary with statistics
    """
    stats = {
        'n_samples': len(T),
        'n_features': X.shape[1],
        'n_treated': T.sum(),
        'n_control': (1 - T).sum(),
        'treatment_rate': T.mean(),
        'true_ate': true_ite.mean(),
        'true_ate_std': true_ite.std(),
        'naive_ate': Y[T == 1].mean() - Y[T == 0].mean(),
        'outcome_mean_treated': Y[T == 1].mean(),
        'outcome_mean_control': Y[T == 0].mean(),
        'outcome_std': Y.std(),
    }

    return stats


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("Loading IHDP Dataset")
    print("="*60)

    df = load_ihdp(n_samples=747, seed=42)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns[:5])} ... {list(df.columns[-3:])}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))

    print("\n" + "="*60)
    print("Generating IHDP Semi-Synthetic Data (Setting A)")
    print("="*60)

    X, T, Y, true_ite = generate_ihdp_semi_synthetic(n_samples=747, setting='A', seed=42)

    stats = get_ihdp_statistics(X, T, Y, true_ite)
    print(f"\nSample Size: {stats['n_samples']}")
    print(f"Features: {stats['n_features']}")
    print(f"Treatment Rate: {stats['treatment_rate']:.2%}")
    print(f"\nTreatment Group: {stats['n_treated']}")
    print(f"Control Group: {stats['n_control']}")
    print(f"\nTrue ATE: {stats['true_ate']:.3f} ± {stats['true_ate_std']:.3f}")
    print(f"Naive ATE: {stats['naive_ate']:.3f}")
    print(f"Bias: {abs(stats['naive_ate'] - stats['true_ate']):.3f}")

    print("\n" + "="*60)
    print("IHDP Setting B (High Nonlinearity)")
    print("="*60)

    X_b, T_b, Y_b, ite_b = generate_ihdp_semi_synthetic(n_samples=747, setting='B', seed=42)
    stats_b = get_ihdp_statistics(X_b, T_b, Y_b, ite_b)

    print(f"\nTrue ATE: {stats_b['true_ate']:.3f} ± {stats_b['true_ate_std']:.3f}")
    print(f"Naive ATE: {stats_b['naive_ate']:.3f}")
    print(f"Bias: {abs(stats_b['naive_ate'] - stats_b['true_ate']):.3f}")

    print("\n" + "="*60)
    print("ITE Distribution Summary")
    print("="*60)
    print(f"Min ITE: {true_ite.min():.3f}")
    print(f"25% ITE: {np.percentile(true_ite, 25):.3f}")
    print(f"Median ITE: {np.median(true_ite):.3f}")
    print(f"75% ITE: {np.percentile(true_ite, 75):.3f}")
    print(f"Max ITE: {true_ite.max():.3f}")
