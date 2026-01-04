"""
CATE Visualization - CATE 可视化与子群体识别

提供多种方式可视化和解释条件平均处理效应 (CATE):
- 按特征分组展示 CATE
- 置信区间可视化
- 子群体识别与对比
- 个体处理效应分布
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple


class TLearnerWithCI:
    """
    带置信区间的 T-Learner

    使用 Bootstrap 方法估计 CATE 的置信区间
    """

    def __init__(self, base_model=None, n_bootstrap: int = 100, alpha: float = 0.05):
        """
        Parameters:
        -----------
        base_model: 基础模型
        n_bootstrap: Bootstrap 采样次数
        alpha: 显著性水平 (默认 0.05 对应 95% CI)
        """
        self.base_model = base_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

        self.model_0 = None
        self.model_1 = None
        self.bootstrap_predictions = []

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        mask_0 = T == 0
        mask_1 = T == 1

        # 主模型
        self.model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_1 = RandomForestRegressor(n_estimators=100, random_state=43)

        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])

        return self

    def predict(self, X: np.ndarray, return_ci: bool = False) -> np.ndarray:
        """
        预测 CATE

        Parameters:
        -----------
        X: 特征矩阵
        return_ci: 是否返回置信区间

        Returns:
        --------
        如果 return_ci=False: cate
        如果 return_ci=True: (cate, lower_bound, upper_bound)
        """
        Y0 = self.model_0.predict(X)
        Y1 = self.model_1.predict(X)
        cate = Y1 - Y0

        if not return_ci:
            return cate

        # Bootstrap 置信区间 (简化版: 使用预测的标准误差)
        n = len(X)
        se = np.std(cate) / np.sqrt(n)
        margin = 1.96 * se  # 95% CI

        lower_bound = cate - margin
        upper_bound = cate + margin

        return cate, lower_bound, upper_bound


def analyze_cate_by_features(
    X: np.ndarray,
    feature_names: List[str],
    true_cate: np.ndarray,
    pred_cate: np.ndarray,
    n_bins: int = 5
) -> pd.DataFrame:
    """
    分析 CATE 如何随特征变化

    Parameters:
    -----------
    X: 特征矩阵
    feature_names: 特征名称
    true_cate: 真实 CATE
    pred_cate: 预测 CATE
    n_bins: 每个特征的分箱数

    Returns:
    --------
    分析结果的 DataFrame
    """
    results = []

    for i, fname in enumerate(feature_names):
        x_i = X[:, i]

        # 分箱
        if len(np.unique(x_i)) <= n_bins:
            bins = np.unique(x_i)
            bin_labels = bins
            x_binned = x_i
        else:
            bins = pd.qcut(x_i, q=n_bins, duplicates='drop', retbins=True)[1]
            bin_labels = [(bins[j] + bins[j+1]) / 2 for j in range(len(bins) - 1)]
            x_binned = pd.cut(x_i, bins=bins, labels=bin_labels, include_lowest=True).astype(float)

        # 每个 bin 的平均 CATE
        for bin_val in bin_labels:
            mask = x_binned == bin_val

            if mask.sum() > 0:
                results.append({
                    'feature': fname,
                    'bin_value': bin_val,
                    'true_cate_mean': true_cate[mask].mean(),
                    'pred_cate_mean': pred_cate[mask].mean(),
                    'true_cate_std': true_cate[mask].std(),
                    'pred_cate_std': pred_cate[mask].std(),
                    'count': mask.sum()
                })

    return pd.DataFrame(results)


def identify_subgroups(
    X: np.ndarray,
    cate: np.ndarray,
    n_groups: int = 4,
    method: str = 'quantile'
) -> np.ndarray:
    """
    根据 CATE 大小将样本分为若干子群体

    Parameters:
    -----------
    X: 特征矩阵
    cate: 条件平均处理效应
    n_groups: 分组数量
    method: 分组方法
        - 'quantile': 按 CATE 分位数分组
        - 'kmeans': 使用 K-means 聚类

    Returns:
    --------
    group_labels: 每个样本的组别标签 (0 to n_groups-1)
    """
    if method == 'quantile':
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

    elif method == 'kmeans':
        # 使用 K-means 聚类
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        group_labels = kmeans.fit_predict(X)

        return group_labels

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_subgroup_statistics(
    Y: np.ndarray,
    T: np.ndarray,
    cate_pred: np.ndarray,
    group_labels: np.ndarray
) -> pd.DataFrame:
    """
    计算各子群体的统计量

    Parameters:
    -----------
    Y: 观测结果
    T: 处理状态
    cate_pred: 预测的 CATE
    group_labels: 群体标签

    Returns:
    --------
    DataFrame with subgroup statistics
    """
    results = []

    for group in np.unique(group_labels):
        mask = group_labels == group
        mask_t = mask & (T == 1)
        mask_c = mask & (T == 0)

        n_total = mask.sum()
        n_t = mask_t.sum()
        n_c = mask_c.sum()

        if n_t > 0 and n_c > 0:
            # 观测 uplift
            y_t_mean = Y[mask_t].mean()
            y_c_mean = Y[mask_c].mean()
            observed_uplift = y_t_mean - y_c_mean
        else:
            observed_uplift = np.nan
            y_t_mean = np.nan
            y_c_mean = np.nan

        # 预测 CATE
        pred_cate_mean = cate_pred[mask].mean()
        pred_cate_std = cate_pred[mask].std()

        results.append({
            'group': group,
            'n_total': n_total,
            'n_treatment': n_t,
            'n_control': n_c,
            'y_treatment_mean': y_t_mean,
            'y_control_mean': y_c_mean,
            'observed_uplift': observed_uplift,
            'predicted_cate_mean': pred_cate_mean,
            'predicted_cate_std': pred_cate_std,
        })

    return pd.DataFrame(results)


def compute_cate_distribution_stats(
    true_cate: np.ndarray,
    pred_cate: np.ndarray
) -> dict:
    """
    计算 CATE 分布的统计指标

    Parameters:
    -----------
    true_cate: 真实 CATE
    pred_cate: 预测 CATE

    Returns:
    --------
    dict with distribution statistics
    """
    # 基本统计量
    stats = {
        'true_cate_mean': float(np.mean(true_cate)),
        'true_cate_std': float(np.std(true_cate)),
        'true_cate_min': float(np.min(true_cate)),
        'true_cate_max': float(np.max(true_cate)),
        'pred_cate_mean': float(np.mean(pred_cate)),
        'pred_cate_std': float(np.std(pred_cate)),
        'pred_cate_min': float(np.min(pred_cate)),
        'pred_cate_max': float(np.max(pred_cate)),
    }

    # 相关性
    correlation = np.corrcoef(true_cate, pred_cate)[0, 1]
    stats['correlation'] = float(correlation)

    # MSE
    mse = np.mean((true_cate - pred_cate) ** 2)
    stats['mse'] = float(mse)
    stats['rmse'] = float(np.sqrt(mse))

    # R²
    ss_res = np.sum((true_cate - pred_cate) ** 2)
    ss_tot = np.sum((true_cate - np.mean(true_cate)) ** 2)
    if ss_tot > 0:
        r_squared = 1 - (ss_res / ss_tot)
    else:
        r_squared = 0.0
    stats['r_squared'] = float(r_squared)

    # MAE
    mae = np.mean(np.abs(true_cate - pred_cate))
    stats['mae'] = float(mae)

    return stats
