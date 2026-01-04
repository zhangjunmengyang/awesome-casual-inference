"""
匹配方法模块

实现多种匹配算法:
- 最近邻匹配 (Nearest Neighbor Matching)
- 精确匹配 (Covariate Exact Matching, CEM)
- 马氏距离匹配 (Mahalanobis Distance Matching)
- 核匹配 (Kernel Matching)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde
from typing import Tuple, Optional, List


class NearestNeighborMatching:
    """
    最近邻匹配

    在协变量空间中寻找最近邻
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = 'euclidean',
        caliper: Optional[float] = None,
        replace: bool = False
    ):
        """
        Parameters:
        -----------
        n_neighbors: 匹配的邻居数量
        metric: 距离度量 ('euclidean', 'manhattan', 'chebyshev')
        caliper: 卡尺宽度
        replace: 是否有放回匹配
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.caliper = caliper
        self.replace = replace
        self.matches = None

    def match(
        self,
        X: np.ndarray,
        treatment: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行最近邻匹配

        Parameters:
        -----------
        X: 特征矩阵
        treatment: 处理状态

        Returns:
        --------
        (treated_indices, control_indices)
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        X_treated = X[treated_idx]
        X_control = X[control_idx]

        # KNN 匹配
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        knn.fit(X_control)

        distances, indices = knn.kneighbors(X_treated)

        # 应用卡尺
        matched_treated = []
        matched_control = []

        for i, t_idx in enumerate(treated_idx):
            for j in range(self.n_neighbors):
                dist = distances[i, j]

                if self.caliper is None or dist <= self.caliper:
                    c_idx = control_idx[indices[i, j]]
                    matched_treated.append(t_idx)
                    matched_control.append(c_idx)

        self.matches = (np.array(matched_treated), np.array(matched_control))
        return self.matches

    def estimate_ate(self, Y: np.ndarray) -> Tuple[float, float]:
        """估计 ATE"""
        if self.matches is None:
            raise ValueError("Must call match() first")

        treated_idx, control_idx = self.matches

        if len(treated_idx) == 0:
            return 0.0, 0.0

        # 1:1 或 1:N 匹配
        if self.n_neighbors == 1:
            pair_diffs = Y[treated_idx] - Y[control_idx]
            ate = pair_diffs.mean()
            se = pair_diffs.std() / np.sqrt(len(pair_diffs))
        else:
            unique_treated = np.unique(treated_idx)
            pair_diffs = []
            for t_idx in unique_treated:
                mask = treated_idx == t_idx
                y_t = Y[t_idx]
                y_c_mean = Y[control_idx[mask]].mean()
                pair_diffs.append(y_t - y_c_mean)
            pair_diffs = np.array(pair_diffs)
            ate = pair_diffs.mean()
            se = pair_diffs.std() / np.sqrt(len(pair_diffs))

        return float(ate), float(se)


class CovariateExactMatching:
    """
    粗糙精确匹配 (Coarsened Exact Matching, CEM)

    将连续协变量分层，然后在每层内精确匹配
    """

    def __init__(self, n_bins: int = 5):
        """
        Parameters:
        -----------
        n_bins: 分层数量
        """
        self.n_bins = n_bins
        self.matches = None
        self.weights = None

    def _coarsen(self, X: np.ndarray) -> np.ndarray:
        """将连续变量粗糙化为离散层"""
        n_samples, n_features = X.shape
        X_coarsened = np.zeros_like(X, dtype=int)

        for j in range(n_features):
            # 按分位数分层
            bins = np.percentile(X[:, j], np.linspace(0, 100, self.n_bins + 1))
            bins = np.unique(bins)  # 去重
            X_coarsened[:, j] = np.digitize(X[:, j], bins[1:-1])

        return X_coarsened

    def match(
        self,
        X: np.ndarray,
        treatment: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行 CEM 匹配

        Returns:
        --------
        (treated_indices, control_indices, weights)
        """
        # 粗糙化
        X_coarsened = self._coarsen(X)

        # 创建层标识
        strata = ['_'.join(map(str, row)) for row in X_coarsened]
        strata = np.array(strata)

        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        matched_treated = []
        matched_control = []
        weights = []

        # 在每层内匹配
        for stratum in np.unique(strata):
            stratum_mask = strata == stratum

            t_in_stratum = treated_idx[stratum_mask[treated_idx]]
            c_in_stratum = control_idx[stratum_mask[control_idx]]

            if len(t_in_stratum) > 0 and len(c_in_stratum) > 0:
                # 有匹配的层
                n_t = len(t_in_stratum)
                n_c = len(c_in_stratum)

                # 每个处理单元匹配所有控制单元
                for t_idx in t_in_stratum:
                    for c_idx in c_in_stratum:
                        matched_treated.append(t_idx)
                        matched_control.append(c_idx)
                        # CEM 权重
                        weights.append(n_t / (n_t * n_c))

        self.matches = (
            np.array(matched_treated),
            np.array(matched_control)
        )
        self.weights = np.array(weights)

        return self.matches[0], self.matches[1], self.weights

    def estimate_ate(self, Y: np.ndarray) -> Tuple[float, float]:
        """估计 ATE (使用 CEM 权重)"""
        if self.matches is None or self.weights is None:
            raise ValueError("Must call match() first")

        treated_idx, control_idx = self.matches

        if len(treated_idx) == 0:
            return 0.0, 0.0

        # 加权估计
        treated_outcomes = Y[treated_idx]
        control_outcomes = Y[control_idx]

        weighted_diffs = (treated_outcomes - control_outcomes) * self.weights
        ate = weighted_diffs.sum() / self.weights.sum()

        # 简化的标准误
        se = np.sqrt(np.sum(self.weights * (treated_outcomes - control_outcomes - ate) ** 2)) / self.weights.sum()

        return float(ate), float(se)


class MahalanobisMatching:
    """
    马氏距离匹配

    考虑协变量之间的相关性
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: Optional[float] = None
    ):
        """
        Parameters:
        -----------
        n_neighbors: 匹配的邻居数量
        caliper: 卡尺宽度 (马氏距离)
        """
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.matches = None
        self.cov_matrix = None

    def match(
        self,
        X: np.ndarray,
        treatment: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行马氏距离匹配

        Parameters:
        -----------
        X: 特征矩阵
        treatment: 处理状态

        Returns:
        --------
        (treated_indices, control_indices)
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        X_treated = X[treated_idx]
        X_control = X[control_idx]

        # 计算协方差矩阵 (使用全部数据)
        self.cov_matrix = np.cov(X.T)
        cov_inv = np.linalg.pinv(self.cov_matrix)  # 使用伪逆以应对奇异矩阵

        matched_treated = []
        matched_control = []

        # 为每个处理单元找最近邻
        for i, t_idx in enumerate(treated_idx):
            x_t = X_treated[i]

            # 计算与所有控制单元的马氏距离
            distances = []
            for x_c in X_control:
                try:
                    dist = mahalanobis(x_t, x_c, cov_inv)
                except:
                    dist = np.inf
                distances.append(dist)

            distances = np.array(distances)

            # 找最近的 n_neighbors 个
            nearest_indices = np.argsort(distances)[:self.n_neighbors]

            for j in nearest_indices:
                dist = distances[j]

                if self.caliper is None or dist <= self.caliper:
                    c_idx = control_idx[j]
                    matched_treated.append(t_idx)
                    matched_control.append(c_idx)

        self.matches = (np.array(matched_treated), np.array(matched_control))
        return self.matches

    def estimate_ate(self, Y: np.ndarray) -> Tuple[float, float]:
        """估计 ATE"""
        if self.matches is None:
            raise ValueError("Must call match() first")

        treated_idx, control_idx = self.matches

        if len(treated_idx) == 0:
            return 0.0, 0.0

        if self.n_neighbors == 1:
            pair_diffs = Y[treated_idx] - Y[control_idx]
            ate = pair_diffs.mean()
            se = pair_diffs.std() / np.sqrt(len(pair_diffs))
        else:
            unique_treated = np.unique(treated_idx)
            pair_diffs = []
            for t_idx in unique_treated:
                mask = treated_idx == t_idx
                y_t = Y[t_idx]
                y_c_mean = Y[control_idx[mask]].mean()
                pair_diffs.append(y_t - y_c_mean)
            pair_diffs = np.array(pair_diffs)
            ate = pair_diffs.mean()
            se = pair_diffs.std() / np.sqrt(len(pair_diffs))

        return float(ate), float(se)


class KernelMatching:
    """
    核匹配

    使用核函数对控制组进行加权
    """

    def __init__(
        self,
        kernel: str = 'gaussian',
        bandwidth: Optional[float] = None
    ):
        """
        Parameters:
        -----------
        kernel: 核函数类型 ('gaussian', 'epanechnikov')
        bandwidth: 带宽 (None 表示自动选择)
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.weights_matrix = None

    def _kernel_function(self, distance: float, h: float) -> float:
        """核函数"""
        u = distance / h

        if self.kernel == 'gaussian':
            return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        elif self.kernel == 'epanechnikov':
            return 0.75 * (1 - u ** 2) if np.abs(u) <= 1 else 0.0
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def match(
        self,
        propensity: np.ndarray,
        treatment: np.ndarray
    ) -> np.ndarray:
        """
        执行核匹配

        Parameters:
        -----------
        propensity: 倾向得分
        treatment: 处理状态

        Returns:
        --------
        weights_matrix: (n_treated, n_control) 权重矩阵
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        prop_treated = propensity[treated_idx]
        prop_control = propensity[control_idx]

        # 自动选择带宽
        if self.bandwidth is None:
            # Silverman's rule of thumb
            std = np.std(propensity)
            n = len(propensity)
            self.bandwidth = 1.06 * std * (n ** (-1/5))

        # 计算权重矩阵
        n_treated = len(treated_idx)
        n_control = len(control_idx)
        weights_matrix = np.zeros((n_treated, n_control))

        for i in range(n_treated):
            for j in range(n_control):
                distance = np.abs(prop_treated[i] - prop_control[j])
                weights_matrix[i, j] = self._kernel_function(distance, self.bandwidth)

            # 归一化权重
            if weights_matrix[i].sum() > 0:
                weights_matrix[i] /= weights_matrix[i].sum()

        self.weights_matrix = weights_matrix
        self.treated_idx = treated_idx
        self.control_idx = control_idx

        return weights_matrix

    def estimate_ate(self, Y: np.ndarray) -> Tuple[float, float]:
        """估计 ATE"""
        if self.weights_matrix is None:
            raise ValueError("Must call match() first")

        Y_treated = Y[self.treated_idx]
        Y_control = Y[self.control_idx]

        # 加权控制组结果
        Y_control_weighted = self.weights_matrix @ Y_control

        # ATE
        individual_effects = Y_treated - Y_control_weighted
        ate = individual_effects.mean()
        se = individual_effects.std() / np.sqrt(len(individual_effects))

        return float(ate), float(se)
