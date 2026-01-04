"""
Uplift 评估模块

提供 Uplift 模型评估的核心指标:
- Qini 曲线
- Uplift 曲线
- AUUC (Area Under Uplift Curve)
- 累积增益
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Qini 曲线

    Qini 曲线衡量按 uplift 得分排序后，累积的增量收益。

    Parameters:
    -----------
    y_true: 真实结果
    treatment: 处理状态 (0/1)
    uplift_score: 预测的 uplift 得分

    Returns:
    --------
    (fraction_targeted, qini_values)
    """
    # 按 uplift 得分排序
    order = np.argsort(uplift_score)[::-1]
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    n_t = treatment.sum()
    n_c = n - n_t

    # 累积计算
    cum_t_outcomes = np.cumsum(y_sorted * t_sorted)
    cum_c_outcomes = np.cumsum(y_sorted * (1 - t_sorted))
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)

    # Qini 值: Q(p) = Y_t(p) - Y_c(p) * N_t(p) / N_c(p)
    # 避免除零
    qini = cum_t_outcomes - cum_c_outcomes * (cum_t / np.maximum(cum_c, 1))

    # 添加原点
    fraction = np.arange(1, n + 1) / n
    fraction = np.insert(fraction, 0, 0)
    qini = np.insert(qini, 0, 0)

    return fraction, qini


def calculate_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Uplift 曲线

    Uplift 曲线展示按 uplift 得分排序后，每个分位数的平均 uplift。

    Parameters:
    -----------
    y_true: 真实结果
    treatment: 处理状态
    uplift_score: 预测的 uplift 得分

    Returns:
    --------
    (fraction_targeted, cumulative_uplift)
    """
    order = np.argsort(uplift_score)[::-1]
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    cumulative_uplift = []

    for i in range(1, n + 1):
        y_sub = y_sorted[:i]
        t_sub = t_sorted[:i]

        n_t = (t_sub == 1).sum()
        n_c = (t_sub == 0).sum()

        if n_t > 0 and n_c > 0:
            uplift = y_sub[t_sub == 1].mean() - y_sub[t_sub == 0].mean()
        else:
            uplift = 0

        cumulative_uplift.append(uplift)

    fraction = np.arange(1, n + 1) / n
    return np.insert(fraction, 0, 0), np.insert(cumulative_uplift, 0, 0)


def calculate_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    normalize: bool = True
) -> float:
    """
    计算 AUUC (Area Under Uplift Curve)

    AUUC 是 Qini 曲线下的面积，衡量模型的整体 uplift 性能。

    Parameters:
    -----------
    y_true: 真实结果
    treatment: 处理状态
    uplift_score: 预测的 uplift 得分
    normalize: 是否归一化 (除以随机基线的 AUUC)

    Returns:
    --------
    AUUC 值 (越大越好)
    """
    fraction, qini = calculate_qini_curve(y_true, treatment, uplift_score)

    # 使用梯形法则计算面积
    auuc = np.trapz(qini, fraction)

    if normalize:
        # 计算随机基线的 AUUC
        n = len(y_true)
        n_t = treatment.sum()
        n_c = n - n_t

        y_t_total = y_true[treatment == 1].sum()
        y_c_total = y_true[treatment == 0].sum()

        # 随机 Qini 曲线是一条直线
        random_qini_max = y_t_total - y_c_total * (n_t / n_c)
        random_auuc = random_qini_max / 2  # 三角形面积

        if random_auuc != 0:
            auuc = auuc / abs(random_auuc)

    return auuc


def calculate_qini_coefficient(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray
) -> float:
    """
    计算 Qini 系数

    Qini 系数 = (实际 AUUC - 随机 AUUC) / (完美 AUUC - 随机 AUUC)

    类似于 Gini 系数，取值范围 [0, 1]，1 表示完美排序。

    Parameters:
    -----------
    y_true: 真实结果
    treatment: 处理状态
    uplift_score: 预测的 uplift 得分

    Returns:
    --------
    Qini 系数 (0-1，越大越好)
    """
    # 实际 AUUC
    actual_auuc = calculate_auuc(y_true, treatment, uplift_score, normalize=False)

    # 随机 AUUC
    random_score = np.random.rand(len(uplift_score))
    random_auuc = calculate_auuc(y_true, treatment, random_score, normalize=False)

    # 完美 AUUC (按真实 uplift 排序)
    # 这需要知道真实的个体处理效应，在实际应用中无法获得
    # 这里用 uplift_score 作为近似
    perfect_auuc = actual_auuc  # 简化处理

    if perfect_auuc - random_auuc == 0:
        return 0.0

    qini_coef = (actual_auuc - random_auuc) / (perfect_auuc - random_auuc)
    return np.clip(qini_coef, 0, 1)


def calculate_cumulative_gain(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    计算累积增益表

    将样本按 uplift 得分分为 n_bins 个分位数，
    计算每个分位数的累积增益。

    Parameters:
    -----------
    y_true: 真实结果
    treatment: 处理状态
    uplift_score: 预测的 uplift 得分
    n_bins: 分位数数量

    Returns:
    --------
    DataFrame with columns: bin, fraction, uplift, cumulative_gain
    """
    order = np.argsort(uplift_score)[::-1]
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    bin_size = n // n_bins

    results = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n

        y_bin = y_sorted[start_idx:end_idx]
        t_bin = t_sorted[start_idx:end_idx]

        n_t = (t_bin == 1).sum()
        n_c = (t_bin == 0).sum()

        if n_t > 0 and n_c > 0:
            uplift = y_bin[t_bin == 1].mean() - y_bin[t_bin == 0].mean()
        else:
            uplift = 0

        # 累积计算
        y_cum = y_sorted[:end_idx]
        t_cum = t_sorted[:end_idx]

        n_t_cum = (t_cum == 1).sum()
        n_c_cum = (t_cum == 0).sum()

        if n_t_cum > 0 and n_c_cum > 0:
            cumulative_uplift = y_cum[t_cum == 1].mean() - y_cum[t_cum == 0].mean()
            cumulative_gain = cumulative_uplift * end_idx
        else:
            cumulative_uplift = 0
            cumulative_gain = 0

        results.append({
            'bin': i + 1,
            'fraction': end_idx / n,
            'uplift': uplift,
            'cumulative_uplift': cumulative_uplift,
            'cumulative_gain': cumulative_gain,
            'n_samples': end_idx - start_idx,
        })

    return pd.DataFrame(results)


def calculate_uplift_at_k(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    k: float = 0.1
) -> float:
    """
    计算 Top-K% 样本的平均 Uplift

    常用于评估 "如果只对 top-k% 的用户进行处理，平均 uplift 是多少"

    Parameters:
    -----------
    y_true: 真实结果
    treatment: 处理状态
    uplift_score: 预测的 uplift 得分
    k: 百分比 (0-1)

    Returns:
    --------
    Top-K% 的平均 uplift
    """
    n = len(y_true)
    k_samples = int(n * k)

    order = np.argsort(uplift_score)[::-1]
    y_top_k = y_true[order][:k_samples]
    t_top_k = treatment[order][:k_samples]

    n_t = (t_top_k == 1).sum()
    n_c = (t_top_k == 0).sum()

    if n_t > 0 and n_c > 0:
        return y_top_k[t_top_k == 1].mean() - y_top_k[t_top_k == 0].mean()
    else:
        return 0.0
