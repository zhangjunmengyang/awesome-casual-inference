"""
处理效应估计工具函数

提供数据生成、评估指标等通用功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


def generate_confounded_data(
    n_samples: int = 2000,
    n_features: int = 5,
    treatment_effect: float = 2.0,
    confounding_strength: float = 1.5,
    noise_std: float = 0.5,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    生成有混淆的观测数据

    DAG: X -> T, X -> Y, T -> Y

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    treatment_effect: 真实 ATE
    confounding_strength: 混淆强度
    noise_std: 噪声标准差
    seed: 随机种子

    Returns:
    --------
    DataFrame with columns: X1...Xn, T, Y, propensity, true_cate
    dict with true parameters
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成特征
    X = np.random.randn(n_samples, n_features)

    # 倾向得分: 受特征影响
    # logit(e(x)) = confounding_strength * (X1 + 0.5*X2)
    propensity_logit = confounding_strength * (X[:, 0] + 0.5 * X[:, 1])
    propensity = 1 / (1 + np.exp(-propensity_logit))

    # 处理分配
    T = np.random.binomial(1, propensity)

    # 真实 CATE (可以是异质性的)
    # tau(x) = treatment_effect + 0.5 * X1
    true_cate = treatment_effect + 0.5 * X[:, 0]

    # 基线结果: 受特征影响
    baseline = 1.0 + X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2]

    # 观测结果
    noise = np.random.randn(n_samples) * noise_std
    Y = baseline + true_cate * T + noise

    # 创建 DataFrame
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['T'] = T
    df['Y'] = Y
    df['propensity'] = propensity
    df['true_cate'] = true_cate

    params = {
        'n_samples': n_samples,
        'n_features': n_features,
        'true_ate': treatment_effect,
        'confounding_strength': confounding_strength,
        'noise_std': noise_std
    }

    return df, params


def compute_ate_oracle(df: pd.DataFrame) -> float:
    """
    计算真实 ATE (用于评估)

    Parameters:
    -----------
    df: DataFrame with 'true_cate' column

    Returns:
    --------
    float: 真实 ATE
    """
    if 'true_cate' in df.columns:
        return df['true_cate'].mean()
    else:
        raise ValueError("DataFrame must contain 'true_cate' column")


def compute_naive_ate(df: pd.DataFrame) -> float:
    """
    计算朴素 ATE 估计 (简单差分)

    Parameters:
    -----------
    df: DataFrame with 'T' and 'Y' columns

    Returns:
    --------
    float: 朴素 ATE 估计
    """
    treated = df[df['T'] == 1]['Y'].mean()
    control = df[df['T'] == 0]['Y'].mean()
    return treated - control


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    特征标准化

    Parameters:
    -----------
    X: 特征矩阵

    Returns:
    --------
    (标准化后的特征, scaler对象)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def compute_smd(X_t: np.ndarray, X_c: np.ndarray) -> np.ndarray:
    """
    计算标准化均值差 (Standardized Mean Difference)

    用于评估协变量平衡性

    Parameters:
    -----------
    X_t: 处理组特征
    X_c: 控制组特征

    Returns:
    --------
    SMD for each feature
    """
    mean_t = X_t.mean(axis=0)
    mean_c = X_c.mean(axis=0)

    var_t = X_t.var(axis=0)
    var_c = X_c.var(axis=0)

    pooled_std = np.sqrt((var_t + var_c) / 2)

    smd = (mean_t - mean_c) / (pooled_std + 1e-8)

    return smd


def compute_variance_ratio(X_t: np.ndarray, X_c: np.ndarray) -> np.ndarray:
    """
    计算方差比

    用于评估协变量平衡性

    Parameters:
    -----------
    X_t: 处理组特征
    X_c: 控制组特征

    Returns:
    --------
    Variance ratio for each feature
    """
    var_t = X_t.var(axis=0)
    var_c = X_c.var(axis=0)

    ratio = var_t / (var_c + 1e-8)

    return ratio


def compute_propensity_overlap(propensity: np.ndarray, treatment: np.ndarray) -> dict:
    """
    计算倾向得分重叠情况

    Parameters:
    -----------
    propensity: 倾向得分
    treatment: 处理状态

    Returns:
    --------
    dict with overlap statistics
    """
    prop_t = propensity[treatment == 1]
    prop_c = propensity[treatment == 0]

    stats = {
        'treated_min': prop_t.min(),
        'treated_max': prop_t.max(),
        'treated_mean': prop_t.mean(),
        'control_min': prop_c.min(),
        'control_max': prop_c.max(),
        'control_mean': prop_c.mean(),
        'overlap_min': max(prop_t.min(), prop_c.min()),
        'overlap_max': min(prop_t.max(), prop_c.max()),
        'non_overlap_fraction': np.mean((propensity < max(prop_t.min(), prop_c.min())) |
                                        (propensity > min(prop_t.max(), prop_c.max())))
    }

    return stats


def evaluate_ate_estimator(estimated_ate: float, true_ate: float, se: Optional[float] = None) -> dict:
    """
    评估 ATE 估计器性能

    Parameters:
    -----------
    estimated_ate: 估计的 ATE
    true_ate: 真实 ATE
    se: 标准误差 (可选)

    Returns:
    --------
    dict with evaluation metrics
    """
    bias = estimated_ate - true_ate
    abs_bias = abs(bias)
    percent_bias = (bias / true_ate) * 100 if true_ate != 0 else np.inf

    metrics = {
        'estimated_ate': estimated_ate,
        'true_ate': true_ate,
        'bias': bias,
        'abs_bias': abs_bias,
        'percent_bias': percent_bias
    }

    if se is not None:
        metrics['se'] = se
        metrics['ci_lower'] = estimated_ate - 1.96 * se
        metrics['ci_upper'] = estimated_ate + 1.96 * se
        metrics['ci_covers_truth'] = (metrics['ci_lower'] <= true_ate <= metrics['ci_upper'])

    return metrics
