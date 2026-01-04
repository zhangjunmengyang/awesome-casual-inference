"""
Uplift Tree 模块

实现 Uplift 决策树的核心算法和分裂准则

核心概念:
- Uplift Tree: 专门用于识别高 Uplift 子群体的决策树
- 分裂准则: KL 散度、欧氏距离、卡方统计量
- 目标: 最大化子节点间的 Uplift 差异
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_uplift_gain(
    y: np.ndarray,
    t: np.ndarray,
    criterion: str = 'KL'
) -> float:
    """
    计算 Uplift 增益

    Parameters:
    -----------
    y: 结果
    t: 处理状态
    criterion: 分裂准则
        - 'KL': KL 散度 (Kullback-Leibler)
        - 'ED': 欧氏距离
        - 'Chi': 卡方统计量
        - 'DDP': Delta-Delta P (差异的差异)

    Returns:
    --------
    uplift gain value
    """
    # 处理组和控制组
    mask_t = t == 1
    mask_c = t == 0

    n_t = mask_t.sum()
    n_c = mask_c.sum()

    if n_t == 0 or n_c == 0:
        return 0.0

    # 转化率
    p_t = y[mask_t].mean() if n_t > 0 else 0.5
    p_c = y[mask_c].mean() if n_c > 0 else 0.5

    # 避免 log(0)
    p_t = np.clip(p_t, 0.001, 0.999)
    p_c = np.clip(p_c, 0.001, 0.999)

    if criterion == 'KL':
        # KL 散度: 衡量处理组和控制组分布的差异
        kl = p_t * np.log(p_t / p_c) + (1 - p_t) * np.log((1 - p_t) / (1 - p_c))
        return kl

    elif criterion == 'ED':
        # 欧氏距离
        return (p_t - p_c) ** 2

    elif criterion == 'Chi':
        # 卡方统计量
        chi = ((p_t - p_c) ** 2) / (p_c * (1 - p_c) + 1e-10)
        return chi

    elif criterion == 'DDP':
        # Delta-Delta P: 简单的差异
        return abs(p_t - p_c)

    else:
        # 默认返回 uplift
        return p_t - p_c


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    feature_idx: int,
    criterion: str = 'KL',
    min_samples_leaf: int = 100
) -> Tuple[Optional[float], float, float, float]:
    """
    找到单个特征的最佳分裂点

    Parameters:
    -----------
    X: 特征矩阵
    y: 结果变量
    t: 处理状态
    feature_idx: 特征索引
    criterion: 分裂准则
    min_samples_leaf: 叶节点最小样本数

    Returns:
    --------
    (best_threshold, best_gain, left_uplift, right_uplift)
    """
    feature = X[:, feature_idx]
    unique_values = np.unique(feature)

    if len(unique_values) < 2:
        return None, -np.inf, 0, 0

    # 候选分裂点
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    best_gain = -np.inf
    best_threshold = None
    best_left_uplift = 0
    best_right_uplift = 0

    # 当前节点的 uplift
    current_gain = calculate_uplift_gain(y, t, criterion)

    for threshold in thresholds:
        left_mask = feature <= threshold
        right_mask = ~left_mask

        # 检查最小样本要求
        if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
            continue

        # 计算左右子节点的增益
        left_gain = calculate_uplift_gain(y[left_mask], t[left_mask], criterion)
        right_gain = calculate_uplift_gain(y[right_mask], t[right_mask], criterion)

        # 加权增益
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        n_total = len(y)

        weighted_gain = (n_left / n_total * left_gain + n_right / n_total * right_gain)
        gain_improvement = weighted_gain - current_gain

        if gain_improvement > best_gain:
            best_gain = gain_improvement
            best_threshold = threshold

            # 计算 uplift
            left_uplift = y[left_mask & (t == 1)].mean() - y[left_mask & (t == 0)].mean() \
                if (left_mask & (t == 1)).sum() > 0 and (left_mask & (t == 0)).sum() > 0 else 0
            right_uplift = y[right_mask & (t == 1)].mean() - y[right_mask & (t == 0)].mean() \
                if (right_mask & (t == 1)).sum() > 0 and (right_mask & (t == 0)).sum() > 0 else 0

            best_left_uplift = left_uplift
            best_right_uplift = right_uplift

    return best_threshold, best_gain, best_left_uplift, best_right_uplift


class SimpleUpliftTree:
    """
    简化版 Uplift Tree

    用于演示 Uplift Tree 的基本原理
    """

    def __init__(
        self,
        criterion: str = 'KL',
        max_depth: int = 3,
        min_samples_leaf: int = 100,
        min_samples_split: int = 200
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练 Uplift Tree"""
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, Y, T, depth=0)
        return self

    def _grow_tree(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray, depth: int) -> dict:
        """递归构建树"""
        n_samples = len(Y)

        # 终止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return self._create_leaf(Y, T)

        # 寻找最佳分裂
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        best_left_uplift = 0
        best_right_uplift = 0

        for feature_idx in range(self.n_features_):
            threshold, gain, left_uplift, right_uplift = find_best_split(
                X, Y, T, feature_idx, self.criterion, self.min_samples_leaf
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
                best_left_uplift = left_uplift
                best_right_uplift = right_uplift

        # 如果没有找到有效分裂，返回叶节点
        if best_feature is None or best_threshold is None:
            return self._create_leaf(Y, T)

        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # 递归构建子树
        return {
            'type': 'split',
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': best_gain,
            'left': self._grow_tree(X[left_mask], Y[left_mask], T[left_mask], depth + 1),
            'right': self._grow_tree(X[right_mask], Y[right_mask], T[right_mask], depth + 1),
        }

    def _create_leaf(self, Y: np.ndarray, T: np.ndarray) -> dict:
        """创建叶节点"""
        mask_t = T == 1
        mask_c = T == 0

        if mask_t.sum() > 0 and mask_c.sum() > 0:
            uplift = Y[mask_t].mean() - Y[mask_c].mean()
        else:
            uplift = 0.0

        return {
            'type': 'leaf',
            'uplift': uplift,
            'n_samples': len(Y),
            'n_treatment': mask_t.sum(),
            'n_control': mask_c.sum(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 Uplift"""
        if self.tree_ is None:
            raise ValueError("Tree not fitted. Call fit() first.")

        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x: np.ndarray, node: dict) -> float:
        """预测单个样本"""
        if node['type'] == 'leaf':
            return node['uplift']

        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
