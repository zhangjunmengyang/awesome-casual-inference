"""
Synthetic Data Generators - 合成数据生成器

提供多种因果推断场景的合成数据生成器，用于教学和算法评估

Data Generating Processes:
--------------------------
1. Linear DGP: 线性因果模型
2. Nonlinear DGP: 非线性因果模型
3. Heterogeneous DGP: 异质性处理效应模型
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable


def generate_linear_dgp(
    n_samples: int = 1000,
    n_features: int = 5,
    treatment_effect: float = 2.0,
    confounding: bool = True,
    noise_std: float = 1.0,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成线性数据生成过程 (Linear DGP)

    模型结构:
    ---------
    X ~ N(0, I)
    T ~ Bernoulli(σ(α'X)) if confounding else Bernoulli(0.5)
    Y(0) = β'X + ε₀
    Y(1) = β'X + τ + ε₁
    Y = T·Y(1) + (1-T)·Y(0)

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    treatment_effect: 平均处理效应 (ATE)
    confounding: 是否存在混淆 (X 影响 T)
    noise_std: 噪声标准差
    seed: 随机种子

    Returns:
    --------
    (X, T, Y, true_ite)
        X: 协变量矩阵 (n_samples, n_features)
        T: 处理状态 (n_samples,)
        Y: 观测结果 (n_samples,)
        true_ite: 真实个体处理效应 (n_samples,)

    Examples:
    ---------
    >>> X, T, Y, ite = generate_linear_dgp(n_samples=1000)
    >>> print(f"True ATE: {ite.mean():.3f}")
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression().fit(np.c_[X, T], Y)
    >>> estimated_ate = model.coef_[-1]
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成协变量
    X = np.random.randn(n_samples, n_features)

    # 处理分配
    if confounding:
        # 倾向得分依赖于协变量
        propensity_logit = 0.5 * X[:, 0] - 0.3 * X[:, 1] if n_features >= 2 else 0.5 * X[:, 0]
        propensity = 1 / (1 + np.exp(-propensity_logit))
        T = np.random.binomial(1, propensity)
    else:
        # 完全随机
        T = np.random.binomial(1, 0.5, n_samples)

    # 结果模型系数 (线性)
    beta = np.random.randn(n_features) * 0.5

    # 潜在结果
    baseline = X @ beta
    noise_0 = np.random.randn(n_samples) * noise_std
    noise_1 = np.random.randn(n_samples) * noise_std

    y0 = baseline + noise_0
    y1 = baseline + treatment_effect + noise_1

    # 观测结果
    Y = np.where(T == 1, y1, y0)

    # 真实 ITE (常数)
    true_ite = np.full(n_samples, treatment_effect)

    return X, T, Y, true_ite


def generate_nonlinear_dgp(
    n_samples: int = 1000,
    n_features: int = 5,
    complexity: str = 'medium',
    noise_std: float = 1.0,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成非线性数据生成过程 (Nonlinear DGP)

    模型结构:
    ---------
    X ~ N(0, I)
    T ~ Bernoulli(σ(f(X)))
    Y(0) = g(X) + ε₀
    Y(1) = g(X) + h(X) + ε₁

    其中 f, g, h 是非线性函数

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    complexity: 复杂度
        - 'low': 简单非线性 (平方项)
        - 'medium': 中等非线性 (三角函数)
        - 'high': 高度非线性 (指数、对数)
    noise_std: 噪声标准差
    seed: 随机种子

    Returns:
    --------
    (X, T, Y, true_ite)

    Examples:
    ---------
    >>> X, T, Y, ite = generate_nonlinear_dgp(complexity='high')
    >>> print(f"ATE: {ite.mean():.3f}, std: {ite.std():.3f}")
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成协变量
    X = np.random.randn(n_samples, n_features)

    # 选择非线性函数
    if complexity == 'low':
        # 低复杂度: 多项式
        propensity_score = 1 / (1 + np.exp(-(0.5 * X[:, 0] ** 2 - 0.3 * X[:, 1])))
        baseline = 2 * X[:, 0] + X[:, 1] ** 2 + 0.5 * X[:, 2] if n_features >= 3 else 2 * X[:, 0] + X[:, 1] ** 2
        treatment_effect = 2 + 0.5 * X[:, 0]

    elif complexity == 'medium':
        # 中等复杂度: 三角函数
        propensity_score = 1 / (1 + np.exp(-(np.sin(X[:, 0]) + 0.5 * X[:, 1])))
        baseline = (
            3 * np.sin(X[:, 0]) +
            2 * np.cos(X[:, 1]) +
            X[:, 2] if n_features >= 3 else 3 * np.sin(X[:, 0]) + 2 * np.cos(X[:, 1])
        )
        treatment_effect = 3 + np.sin(X[:, 0]) - 0.5 * X[:, 1]

    elif complexity == 'high':
        # 高复杂度: 混合非线性
        propensity_score = 1 / (
            1 + np.exp(-(
                np.sin(X[:, 0]) +
                0.5 * np.log(np.abs(X[:, 1]) + 1) +
                0.3 * X[:, 2] if n_features >= 3 else np.sin(X[:, 0]) + 0.5 * np.log(np.abs(X[:, 1]) + 1)
            ))
        )
        baseline = (
            np.exp(0.3 * X[:, 0]) +
            X[:, 1] / (1 + np.exp(-X[:, 2])) +
            np.sin(X[:, 3]) * X[:, 4] if n_features >= 5
            else np.exp(0.3 * X[:, 0]) + X[:, 1] / (1 + np.exp(-X[:, 1]))
        )
        treatment_effect = (
            4 +
            2 * np.tanh(X[:, 0]) +
            X[:, 1] ** 2 -
            np.abs(X[:, 2]) if n_features >= 3
            else 4 + 2 * np.tanh(X[:, 0]) + X[:, 1] ** 2
        )

    else:
        raise ValueError(f"Unknown complexity: {complexity}")

    # 处理分配
    T = np.random.binomial(1, propensity_score)

    # 潜在结果
    noise_0 = np.random.randn(n_samples) * noise_std
    noise_1 = np.random.randn(n_samples) * noise_std

    y0 = baseline + noise_0
    y1 = baseline + treatment_effect + noise_1

    # 观测结果
    Y = np.where(T == 1, y1, y0)

    # 真实 ITE
    true_ite = treatment_effect

    return X, T, Y, true_ite


def generate_heterogeneous_dgp(
    n_samples: int = 1000,
    n_features: int = 10,
    heterogeneity_type: str = 'linear',
    noise_std: float = 1.0,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成异质性处理效应数据 (Heterogeneous Treatment Effect)

    重点: 处理效应 τ(X) 随协变量变化

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    heterogeneity_type: 异质性类型
        - 'linear': τ(X) = α + β'X
        - 'interaction': τ(X) = α + β₁X₁ + β₂X₁X₂
        - 'threshold': τ(X) = α if X₁ > 0 else β
        - 'complex': 复杂非线性异质性
    noise_std: 噪声标准差
    seed: 随机种子

    Returns:
    --------
    (X, T, Y, true_ite)

    Examples:
    ---------
    >>> X, T, Y, ite = generate_heterogeneous_dgp(heterogeneity_type='threshold')
    >>> # 可视化异质性
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(X[:, 0], ite, alpha=0.5)
    >>> plt.xlabel('X1')
    >>> plt.ylabel('ITE')
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成协变量
    X = np.random.randn(n_samples, n_features)

    # 处理分配 (随机化，避免混淆)
    T = np.random.binomial(1, 0.5, n_samples)

    # 基线结果 (依赖协变量)
    baseline = (
        2 * X[:, 0] +
        1.5 * X[:, 1] +
        0.5 * X[:, 2] +
        0.3 * X[:, 0] * X[:, 1] if n_features >= 3
        else 2 * X[:, 0] + 1.5 * X[:, 1]
    )

    # 异质性处理效应
    if heterogeneity_type == 'linear':
        # 线性异质性
        tau = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1]

    elif heterogeneity_type == 'interaction':
        # 交互效应
        tau = (
            3 +
            2 * X[:, 0] +
            1.5 * X[:, 1] +
            1.0 * X[:, 0] * X[:, 1]
        )

    elif heterogeneity_type == 'threshold':
        # 阈值效应 (分段常数)
        tau = np.where(X[:, 0] > 0, 4.0, 1.0)

    elif heterogeneity_type == 'complex':
        # 复杂非线性异质性
        tau = (
            3 +
            2 * np.sin(np.pi * X[:, 0]) +
            1.5 * X[:, 1] ** 2 +
            X[:, 2] * np.tanh(X[:, 3]) if n_features >= 4
            else 3 + 2 * np.sin(np.pi * X[:, 0]) + 1.5 * X[:, 1] ** 2
        )

    else:
        raise ValueError(f"Unknown heterogeneity type: {heterogeneity_type}")

    # 潜在结果
    noise_0 = np.random.randn(n_samples) * noise_std
    noise_1 = np.random.randn(n_samples) * noise_std

    y0 = baseline + noise_0
    y1 = baseline + tau + noise_1

    # 观测结果
    Y = np.where(T == 1, y1, y0)

    # 真实 ITE
    true_ite = tau

    return X, T, Y, true_ite


def generate_marketing_dgp(
    n_samples: int = 5000,
    scenario: str = 'coupon',
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成营销场景数据

    Parameters:
    -----------
    n_samples: 样本数量
    scenario: 场景类型
        - 'coupon': 优惠券发放
        - 'email': 邮件营销
        - 'recommendation': 推荐系统
    seed: 随机种子

    Returns:
    --------
    (df, true_uplift)
        df: DataFrame with features, treatment, outcome
        true_uplift: 真实 uplift (增益)

    Examples:
    ---------
    >>> df, uplift = generate_marketing_dgp(scenario='coupon')
    >>> print(df.columns)
    >>> print(f"Average uplift: {uplift.mean():.4f}")
    """
    if seed is not None:
        np.random.seed(seed)

    if scenario == 'coupon':
        # 优惠券场景
        age = np.random.uniform(18, 70, n_samples)
        income = np.random.lognormal(10.5, 0.8, n_samples)
        purchase_freq = np.random.poisson(3, n_samples)
        days_since_last = np.random.exponential(30, n_samples)
        is_member = np.random.binomial(1, 0.3, n_samples)

        # 标准化
        age_norm = (age - 40) / 15
        income_norm = (np.log(income) - 10.5) / 0.8
        freq_norm = (purchase_freq - 3) / 2
        recency_norm = (days_since_last - 30) / 20

        # 处理分配
        T = np.random.binomial(1, 0.5, n_samples)

        # 基线转化率
        baseline_prob = 1 / (1 + np.exp(-(
            -2 +
            0.3 * freq_norm +
            0.2 * income_norm +
            0.3 * is_member -
            0.2 * recency_norm
        )))

        # Uplift (异质性)
        uplift = (
            0.10 +  # 基础提升
            0.05 * (1 - age_norm) +  # 年轻人更敏感
            0.04 * (1 - freq_norm) +  # 低频用户更敏感
            0.03 * is_member  # 会员更敏感
        )
        uplift = np.clip(uplift, 0, 0.3)

        # 转化
        prob = np.clip(baseline_prob + uplift * T, 0, 1)
        Y = np.random.binomial(1, prob)

        df = pd.DataFrame({
            'age': age,
            'income': income,
            'purchase_freq': purchase_freq,
            'days_since_last': days_since_last,
            'is_member': is_member,
            'treatment': T,
            'conversion': Y
        })

    elif scenario == 'email':
        # 邮件营销场景
        engagement_score = np.random.beta(2, 5, n_samples) * 100
        email_open_rate = np.random.beta(3, 7, n_samples)
        tenure_days = np.random.exponential(180, n_samples)
        segment = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])

        T = np.random.binomial(1, 0.5, n_samples)

        # 基线点击率
        baseline_prob = email_open_rate * 0.3

        # Uplift
        uplift = np.where(
            segment == 'A', 0.05,
            np.where(segment == 'B', 0.08, 0.12)
        )

        prob = np.clip(baseline_prob + uplift * T, 0, 1)
        Y = np.random.binomial(1, prob)

        df = pd.DataFrame({
            'engagement_score': engagement_score,
            'email_open_rate': email_open_rate,
            'tenure_days': tenure_days,
            'segment': segment,
            'treatment': T,
            'click': Y
        })

    elif scenario == 'recommendation':
        # 推荐系统场景
        browse_time = np.random.exponential(10, n_samples)
        past_purchases = np.random.poisson(2, n_samples)
        avg_basket = np.random.lognormal(3.5, 1, n_samples)
        category_affinity = np.random.beta(2, 5, n_samples)

        T = np.random.binomial(1, 0.5, n_samples)

        # 基线购买概率
        baseline_prob = 1 / (1 + np.exp(-(
            -3 +
            0.05 * browse_time +
            0.2 * past_purchases +
            0.3 * category_affinity
        )))

        # Uplift
        uplift = 0.08 + 0.05 * category_affinity

        prob = np.clip(baseline_prob + uplift * T, 0, 1)
        Y = np.random.binomial(1, prob)

        df = pd.DataFrame({
            'browse_time': browse_time,
            'past_purchases': past_purchases,
            'avg_basket': avg_basket,
            'category_affinity': category_affinity,
            'treatment': T,
            'purchase': Y
        })

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return df, uplift


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("1. Linear DGP")
    print("="*60)
    X, T, Y, ite = generate_linear_dgp(n_samples=1000, confounding=True)
    print(f"Shape: X={X.shape}, T={T.shape}, Y={Y.shape}")
    print(f"Treatment rate: {T.mean():.2%}")
    print(f"True ATE: {ite.mean():.3f}")
    print(f"Naive ATE: {Y[T==1].mean() - Y[T==0].mean():.3f}")

    print("\n" + "="*60)
    print("2. Nonlinear DGP (High Complexity)")
    print("="*60)
    X, T, Y, ite = generate_nonlinear_dgp(complexity='high')
    print(f"True ATE: {ite.mean():.3f} ± {ite.std():.3f}")
    print(f"ITE range: [{ite.min():.3f}, {ite.max():.3f}]")

    print("\n" + "="*60)
    print("3. Heterogeneous DGP")
    print("="*60)
    for het_type in ['linear', 'interaction', 'threshold', 'complex']:
        X, T, Y, ite = generate_heterogeneous_dgp(heterogeneity_type=het_type)
        print(f"\n{het_type.capitalize()}:")
        print(f"  ATE: {ite.mean():.3f}")
        print(f"  ITE std: {ite.std():.3f}")
        print(f"  ITE range: [{ite.min():.3f}, {ite.max():.3f}]")

    print("\n" + "="*60)
    print("4. Marketing DGP (Coupon Scenario)")
    print("="*60)
    df, uplift = generate_marketing_dgp(scenario='coupon', n_samples=5000)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nAverage uplift: {uplift.mean():.4f}")
    print(f"Uplift std: {uplift.std():.4f}")
    print(f"\nConversion rate (treated): {df[df['treatment']==1]['conversion'].mean():.2%}")
    print(f"Conversion rate (control): {df[df['treatment']==0]['conversion'].mean():.2%}")
    print(f"Observed uplift: {df[df['treatment']==1]['conversion'].mean() - df[df['treatment']==0]['conversion'].mean():.4f}")
