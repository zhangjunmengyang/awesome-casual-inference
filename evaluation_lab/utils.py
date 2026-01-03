"""
EvaluationLab 工具函数

提供数据生成、统计计算等通用功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression


def generate_observational_data(
    n_samples: int = 2000,
    n_features: int = 5,
    treatment_assignment: str = 'confounded',
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成观测性数据 (用于评估)

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    treatment_assignment: 处理分配机制
        - 'random': 随机分配 (RCT)
        - 'confounded': 混淆分配 (观测数据)
        - 'severe_confounding': 严重混淆
    seed: 随机种子

    Returns:
    --------
    (DataFrame, true_cate)
    DataFrame columns: X1...Xn, T, Y
    true_cate: 真实的个体处理效应
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成特征
    X = np.random.randn(n_samples, n_features)

    # 计算倾向得分
    if treatment_assignment == 'random':
        # 随机分配，倾向得分为常数
        propensity = np.full(n_samples, 0.5)

    elif treatment_assignment == 'confounded':
        # 混淆分配，依赖于前两个特征
        logit = 0.5 * X[:, 0] + 0.3 * X[:, 1]
        propensity = 1 / (1 + np.exp(-logit))

    elif treatment_assignment == 'severe_confounding':
        # 严重混淆
        logit = 1.5 * X[:, 0] + 1.0 * X[:, 1] - 0.5 * X[:, 0] * X[:, 1]
        propensity = 1 / (1 + np.exp(-logit))

    else:
        raise ValueError(f"Unknown treatment assignment: {treatment_assignment}")

    # 分配处理
    T = np.random.binomial(1, propensity)

    # 基线结果 (依赖于特征)
    baseline = 2 + X[:, 0] + 0.5 * X[:, 1]

    # 处理效应 (异质性)
    tau = 1.5 + 0.8 * X[:, 0] - 0.4 * X[:, 1]

    # 生成结果
    noise = np.random.randn(n_samples) * 0.5
    Y = baseline + tau * T + noise

    # 创建 DataFrame
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['T'] = T
    df['Y'] = Y
    df['propensity'] = propensity  # 保存真实倾向得分

    return df, tau


def calculate_standardized_mean_difference(
    X_treatment: np.ndarray,
    X_control: np.ndarray
) -> np.ndarray:
    """
    计算标准化均值差 (SMD)

    SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2)

    Parameters:
    -----------
    X_treatment: 处理组特征 (n_t, n_features)
    X_control: 对照组特征 (n_c, n_features)

    Returns:
    --------
    smd: 每个特征的 SMD (n_features,)
    """
    mean_t = np.mean(X_treatment, axis=0)
    mean_c = np.mean(X_control, axis=0)

    var_t = np.var(X_treatment, axis=0)
    var_c = np.var(X_control, axis=0)

    pooled_std = np.sqrt((var_t + var_c) / 2)

    # 避免除以零
    smd = np.where(pooled_std > 0, (mean_t - mean_c) / pooled_std, 0)

    return smd


def calculate_variance_ratio(
    X_treatment: np.ndarray,
    X_control: np.ndarray
) -> np.ndarray:
    """
    计算方差比

    Variance Ratio = var_t / var_c

    Parameters:
    -----------
    X_treatment: 处理组特征
    X_control: 对照组特征

    Returns:
    --------
    variance_ratio: 每个特征的方差比
    """
    var_t = np.var(X_treatment, axis=0)
    var_c = np.var(X_control, axis=0)

    # 避免除以零
    variance_ratio = np.where(var_c > 0, var_t / var_c, 1.0)

    return variance_ratio


def estimate_propensity_score(
    X: np.ndarray,
    T: np.ndarray
) -> np.ndarray:
    """
    估计倾向得分 (使用 Logistic Regression)

    Parameters:
    -----------
    X: 特征矩阵 (n, n_features)
    T: 处理状态 (n,)

    Returns:
    --------
    propensity_scores: 估计的倾向得分 (n,)
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, T)
    propensity = model.predict_proba(X)[:, 1]

    return propensity


def calculate_overlap_statistics(
    propensity_treatment: np.ndarray,
    propensity_control: np.ndarray
) -> Dict[str, float]:
    """
    计算重叠统计量

    Parameters:
    -----------
    propensity_treatment: 处理组的倾向得分
    propensity_control: 对照组的倾向得分

    Returns:
    --------
    stats: 包含重叠统计量的字典
    """
    # 处理组倾向得分的范围
    t_min, t_max = propensity_treatment.min(), propensity_treatment.max()

    # 对照组倾向得分的范围
    c_min, c_max = propensity_control.min(), propensity_control.max()

    # 重叠区域
    overlap_min = max(t_min, c_min)
    overlap_max = min(t_max, c_max)
    overlap_range = max(0, overlap_max - overlap_min)

    # 计算在重叠区域内的样本比例
    all_propensity = np.concatenate([propensity_treatment, propensity_control])
    in_overlap = (all_propensity >= overlap_min) & (all_propensity <= overlap_max)
    overlap_fraction = in_overlap.mean()

    # 正性假设检验 (是否有足够的重叠)
    # 检查倾向得分是否在极端值 (< 0.1 或 > 0.9)
    extreme_low = (all_propensity < 0.1).sum()
    extreme_high = (all_propensity > 0.9).sum()
    extreme_fraction = (extreme_low + extreme_high) / len(all_propensity)

    stats = {
        'overlap_min': overlap_min,
        'overlap_max': overlap_max,
        'overlap_range': overlap_range,
        'overlap_fraction': overlap_fraction,
        'extreme_fraction': extreme_fraction,
        'treatment_min': t_min,
        'treatment_max': t_max,
        'control_min': c_min,
        'control_max': c_max
    }

    return stats


def perform_propensity_score_matching(
    X: np.ndarray,
    T: np.ndarray,
    propensity: np.ndarray,
    caliper: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行倾向得分匹配 (1:1 最近邻匹配)

    Parameters:
    -----------
    X: 特征矩阵
    T: 处理状态
    propensity: 倾向得分
    caliper: 卡钳宽度 (最大允许的倾向得分差异)

    Returns:
    --------
    matched_treatment_idx: 匹配的处理组索引
    matched_control_idx: 匹配的对照组索引
    """
    treatment_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    matched_treatment = []
    matched_control = []

    used_controls = set()

    for t_idx in treatment_idx:
        t_ps = propensity[t_idx]

        # 找到最近的对照组
        distances = np.abs(propensity[control_idx] - t_ps)

        # 排除已使用的对照组
        available = [i for i, c_idx in enumerate(control_idx) if c_idx not in used_controls]

        if len(available) == 0:
            continue

        min_idx = available[np.argmin(distances[available])]
        min_distance = distances[min_idx]

        # 检查是否在 caliper 内
        if min_distance <= caliper:
            c_idx = control_idx[min_idx]
            matched_treatment.append(t_idx)
            matched_control.append(c_idx)
            used_controls.add(c_idx)

    return np.array(matched_treatment), np.array(matched_control)


def calculate_balance_metrics(
    X: np.ndarray,
    T: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """
    计算平衡性指标

    Returns:
    --------
    DataFrame with columns: feature, smd, variance_ratio, mean_t, mean_c, var_t, var_c
    """
    X_t = X[T == 1]
    X_c = X[T == 0]

    smd = calculate_standardized_mean_difference(X_t, X_c)
    var_ratio = calculate_variance_ratio(X_t, X_c)

    mean_t = np.mean(X_t, axis=0)
    mean_c = np.mean(X_c, axis=0)
    var_t = np.var(X_t, axis=0)
    var_c = np.var(X_c, axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'smd': smd,
        'variance_ratio': var_ratio,
        'mean_t': mean_t,
        'mean_c': mean_c,
        'var_t': var_t,
        'var_c': var_c
    })

    return df


def suggest_trimming(
    propensity: np.ndarray,
    T: np.ndarray,
    lower_threshold: float = 0.1,
    upper_threshold: float = 0.9
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    建议修剪 (Trimming) 样本

    移除倾向得分过于极端的样本

    Parameters:
    -----------
    propensity: 倾向得分
    T: 处理状态
    lower_threshold: 下限阈值
    upper_threshold: 上限阈值

    Returns:
    --------
    keep_mask: 保留的样本掩码
    trimming_info: 修剪信息
    """
    keep_mask = (propensity >= lower_threshold) & (propensity <= upper_threshold)

    n_total = len(propensity)
    n_trimmed = (~keep_mask).sum()
    n_trimmed_treatment = ((~keep_mask) & (T == 1)).sum()
    n_trimmed_control = ((~keep_mask) & (T == 0)).sum()

    trimming_info = {
        'n_total': n_total,
        'n_trimmed': n_trimmed,
        'trimmed_fraction': n_trimmed / n_total,
        'n_trimmed_treatment': n_trimmed_treatment,
        'n_trimmed_control': n_trimmed_control,
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold
    }

    return keep_mask, trimming_info
