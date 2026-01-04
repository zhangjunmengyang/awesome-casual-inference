"""
倾向得分方法模块

实现倾向得分估计和倾向得分匹配 (Propensity Score Matching, PSM)

核心概念:
- 倾向得分: e(X) = P(T=1|X) - 接受处理的概率
- Rosenbaum & Rubin (1983): 在倾向得分上条件化可以平衡协变量
- 匹配: 为每个处理组个体找到倾向得分相似的控制组个体
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Union
from sklearn.base import BaseEstimator


class PropensityScoreEstimator:
    """
    倾向得分估计器

    使用逻辑回归或其他分类器估计 P(T=1|X)
    """

    def __init__(self, model: Optional[BaseEstimator] = None):
        """
        Parameters:
        -----------
        model: sklearn分类器，默认为LogisticRegression
        """
        self.model = model or LogisticRegression(max_iter=1000, random_state=42)
        self.propensity = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """训练倾向得分模型"""
        self.model.fit(X, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测倾向得分"""
        self.propensity = self.model.predict_proba(X)[:, 1]
        return self.propensity

    def fit_predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """训练并预测"""
        self.fit(X, T)
        return self.predict(X)


class PropensityScoreMatching:
    """
    倾向得分匹配 (PSM)

    方法:
    - 1:1 或 1:N 最近邻匹配
    - 卡尺匹配 (Caliper matching)
    - 有放回/无放回匹配
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: Optional[float] = None,
        replace: bool = False
    ):
        """
        Parameters:
        -----------
        n_neighbors: 匹配的邻居数量
        caliper: 卡尺宽度 (最大允许的倾向得分差异)
        replace: 是否有放回匹配
        """
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replace = replace
        self.matches = None

    def match(
        self,
        propensity: np.ndarray,
        treatment: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行倾向得分匹配

        Parameters:
        -----------
        propensity: 倾向得分
        treatment: 处理状态

        Returns:
        --------
        (treated_indices, control_indices)
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        # 使用 KNN 进行匹配
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        knn.fit(propensity[control_idx].reshape(-1, 1))

        # 为每个处理组个体找到最近的控制组个体
        distances, indices = knn.kneighbors(
            propensity[treated_idx].reshape(-1, 1)
        )

        # 应用卡尺
        matched_treated = []
        matched_control = []

        for i, t_idx in enumerate(treated_idx):
            for j in range(self.n_neighbors):
                dist = distances[i, j]

                # 检查卡尺约束
                if self.caliper is None or dist <= self.caliper:
                    c_idx = control_idx[indices[i, j]]
                    matched_treated.append(t_idx)
                    matched_control.append(c_idx)

        self.matches = (np.array(matched_treated), np.array(matched_control))

        return self.matches

    def estimate_ate(self, Y: np.ndarray) -> Tuple[float, float]:
        """
        估计 ATE

        Parameters:
        -----------
        Y: 结果变量

        Returns:
        --------
        (ATE估计, 标准误)
        """
        if self.matches is None:
            raise ValueError("Must call match() first")

        treated_idx, control_idx = self.matches

        if len(treated_idx) == 0:
            return 0.0, 0.0

        treated_outcomes = Y[treated_idx]
        control_outcomes = Y[control_idx]

        # 当 n_neighbors > 1 时，需要按配对计算
        if self.n_neighbors == 1:
            # 1:1 匹配，直接计算配对差异
            pair_diffs = treated_outcomes - control_outcomes
            ate = pair_diffs.mean()
            se = pair_diffs.std() / np.sqrt(len(pair_diffs))
        else:
            # n_neighbors > 1 时，每个处理样本有多个对照匹配
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

    def estimate_att(self, Y: np.ndarray) -> Tuple[float, float]:
        """
        估计 ATT (Average Treatment Effect on the Treated)

        与 ATE 相同，因为我们匹配的是处理组
        """
        return self.estimate_ate(Y)
