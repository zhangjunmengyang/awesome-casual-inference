"""
加权方法模块

实现多种倾向得分加权方法:
- 逆概率加权 (IPW)
- 稳定权重 (Stabilized IPW)
- 重叠权重 (Overlap Weighting)
- 截断权重 (Trimmed IPW)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Optional
from sklearn.base import BaseEstimator


class IPWEstimator:
    """
    逆概率加权 (Inverse Probability Weighting, IPW) 估计器

    ATE = E[Y(1)] - E[Y(0)]
        = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        clip_propensity: Tuple[float, float] = (0.01, 0.99),
        clip_weights: bool = False,
        weight_percentile: float = 99.0
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        clip_propensity: 倾向得分截断范围
        clip_weights: 是否截断极端权重
        weight_percentile: 权重截断百分位数
        """
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=42
        )
        self.clip_propensity = clip_propensity
        self.clip_weights = clip_weights
        self.weight_percentile = weight_percentile
        self.propensity = None
        self.weights = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """训练倾向得分模型"""
        self.propensity_model.fit(X, T)
        return self

    def _compute_weights(
        self,
        propensity: np.ndarray,
        treatment: np.ndarray
    ) -> np.ndarray:
        """
        计算 IPW 权重

        w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))
        """
        # 截断倾向得分
        propensity = np.clip(propensity, *self.clip_propensity)

        # 计算权重
        weights = np.where(
            treatment == 1,
            1.0 / propensity,
            1.0 / (1.0 - propensity)
        )

        # 截断极端权重
        if self.clip_weights:
            max_weight = np.percentile(weights, self.weight_percentile)
            weights = np.clip(weights, 0, max_weight)

        return weights

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        估计 ATE

        Returns:
        --------
        (ATE估计, 标准误, 权重)
        """
        # 估计倾向得分
        self.propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 计算权重
        self.weights = self._compute_weights(self.propensity, T)

        # 加权估计
        treated_mask = T == 1
        control_mask = T == 0

        # E[Y(1)]
        y1_weighted = np.sum(Y[treated_mask] * self.weights[treated_mask]) / \
                      np.sum(self.weights[treated_mask])

        # E[Y(0)]
        y0_weighted = np.sum(Y[control_mask] * self.weights[control_mask]) / \
                      np.sum(self.weights[control_mask])

        ate = y1_weighted - y0_weighted

        # 计算标准误 (Hajek estimator variance)
        n = len(Y)

        # 残差
        residuals_1 = np.zeros(n)
        residuals_1[treated_mask] = (Y[treated_mask] - y1_weighted) * \
                                    self.weights[treated_mask]

        residuals_0 = np.zeros(n)
        residuals_0[control_mask] = (Y[control_mask] - y0_weighted) * \
                                    self.weights[control_mask]

        influence_fn = residuals_1 - residuals_0
        variance = np.var(influence_fn) / n
        se = np.sqrt(variance)

        return float(ate), float(se), self.weights


class StabilizedIPW:
    """
    稳定权重 (Stabilized IPW)

    使用边际处理概率稳定权重，减少方差

    w_i = P(T=t) / P(T=t|X)
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        clip_propensity: Tuple[float, float] = (0.01, 0.99)
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        clip_propensity: 倾向得分截断范围
        """
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=42
        )
        self.clip_propensity = clip_propensity
        self.propensity = None
        self.weights = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """训练倾向得分模型"""
        self.propensity_model.fit(X, T)
        return self

    def _compute_weights(
        self,
        propensity: np.ndarray,
        treatment: np.ndarray
    ) -> np.ndarray:
        """
        计算稳定权重

        w_i = P(T=1) / e(X_i) for treated
        w_i = P(T=0) / (1-e(X_i)) for control
        """
        # 截断倾向得分
        propensity = np.clip(propensity, *self.clip_propensity)

        # 边际概率
        p_t = treatment.mean()

        # 稳定权重
        weights = np.where(
            treatment == 1,
            p_t / propensity,
            (1 - p_t) / (1 - propensity)
        )

        return weights

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """估计 ATE"""
        # 估计倾向得分
        self.propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 计算稳定权重
        self.weights = self._compute_weights(self.propensity, T)

        # 加权估计
        treated_mask = T == 1
        control_mask = T == 0

        y1_weighted = np.sum(Y[treated_mask] * self.weights[treated_mask]) / \
                      np.sum(self.weights[treated_mask])
        y0_weighted = np.sum(Y[control_mask] * self.weights[control_mask]) / \
                      np.sum(self.weights[control_mask])

        ate = y1_weighted - y0_weighted

        # 标准误
        n = len(Y)
        residuals_1 = np.zeros(n)
        residuals_1[treated_mask] = (Y[treated_mask] - y1_weighted) * \
                                    self.weights[treated_mask]
        residuals_0 = np.zeros(n)
        residuals_0[control_mask] = (Y[control_mask] - y0_weighted) * \
                                    self.weights[control_mask]

        influence_fn = residuals_1 - residuals_0
        variance = np.var(influence_fn) / n
        se = np.sqrt(variance)

        return float(ate), float(se), self.weights


class OverlapWeighting:
    """
    重叠权重 (Overlap Weighting)

    Li, Morgan & Zaslavsky (2018)
    重点关注倾向得分重叠区域的样本

    w_i = (1 - e(X_i)) for treated
    w_i = e(X_i) for control

    估计 ATO (Average Treatment Effect in the Overlap population)
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        clip_propensity: Tuple[float, float] = (0.01, 0.99)
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        clip_propensity: 倾向得分截断范围
        """
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=42
        )
        self.clip_propensity = clip_propensity
        self.propensity = None
        self.weights = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """训练倾向得分模型"""
        self.propensity_model.fit(X, T)
        return self

    def _compute_weights(
        self,
        propensity: np.ndarray,
        treatment: np.ndarray
    ) -> np.ndarray:
        """
        计算重叠权重

        w_i = (1 - e(X_i)) for treated
        w_i = e(X_i) for control
        """
        # 截断倾向得分
        propensity = np.clip(propensity, *self.clip_propensity)

        # 重叠权重
        weights = np.where(
            treatment == 1,
            1.0 - propensity,  # 处理组: 低倾向得分 -> 高权重
            propensity          # 控制组: 高倾向得分 -> 高权重
        )

        return weights

    def estimate_ato(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        估计 ATO (Average Treatment Effect in Overlap population)

        Returns:
        --------
        (ATO估计, 标准误, 权重)
        """
        # 估计倾向得分
        self.propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 计算重叠权重
        self.weights = self._compute_weights(self.propensity, T)

        # 加权估计
        treated_mask = T == 1
        control_mask = T == 0

        y1_weighted = np.sum(Y[treated_mask] * self.weights[treated_mask]) / \
                      np.sum(self.weights[treated_mask])
        y0_weighted = np.sum(Y[control_mask] * self.weights[control_mask]) / \
                      np.sum(self.weights[control_mask])

        ato = y1_weighted - y0_weighted

        # 标准误
        n = len(Y)
        residuals_1 = np.zeros(n)
        residuals_1[treated_mask] = (Y[treated_mask] - y1_weighted) * \
                                    self.weights[treated_mask]
        residuals_0 = np.zeros(n)
        residuals_0[control_mask] = (Y[control_mask] - y0_weighted) * \
                                    self.weights[control_mask]

        influence_fn = residuals_1 - residuals_0
        variance = np.var(influence_fn) / n
        se = np.sqrt(variance)

        return float(ato), float(se), self.weights


class TrimmedIPW:
    """
    截断 IPW (Trimmed IPW)

    Crump et al. (2009)
    移除极端倾向得分的样本，专注于"共同支撑"区域
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        trim_threshold: float = 0.1
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        trim_threshold: 截断阈值 (移除 e < α 或 e > 1-α 的样本)
        """
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=42
        )
        self.trim_threshold = trim_threshold
        self.propensity = None
        self.weights = None
        self.trimmed_mask = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """训练倾向得分模型"""
        self.propensity_model.fit(X, T)
        return self

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        估计 ATE (在截断后的样本上)

        Returns:
        --------
        (ATE估计, 标准误, 权重, 截断mask)
        """
        # 估计倾向得分
        self.propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 截断: 只保留 [α, 1-α] 内的样本
        self.trimmed_mask = (
            (self.propensity >= self.trim_threshold) &
            (self.propensity <= 1 - self.trim_threshold)
        )

        # 在截断后的样本上计算 IPW
        X_trimmed = X[self.trimmed_mask]
        T_trimmed = T[self.trimmed_mask]
        Y_trimmed = Y[self.trimmed_mask]
        propensity_trimmed = self.propensity[self.trimmed_mask]

        # 计算权重
        weights_trimmed = np.where(
            T_trimmed == 1,
            1.0 / propensity_trimmed,
            1.0 / (1.0 - propensity_trimmed)
        )

        # 加权估计
        treated_mask = T_trimmed == 1
        control_mask = T_trimmed == 0

        y1_weighted = np.sum(Y_trimmed[treated_mask] * weights_trimmed[treated_mask]) / \
                      np.sum(weights_trimmed[treated_mask])
        y0_weighted = np.sum(Y_trimmed[control_mask] * weights_trimmed[control_mask]) / \
                      np.sum(weights_trimmed[control_mask])

        ate = y1_weighted - y0_weighted

        # 标准误
        n_trimmed = len(Y_trimmed)
        residuals_1 = np.zeros(n_trimmed)
        residuals_1[treated_mask] = (Y_trimmed[treated_mask] - y1_weighted) * \
                                    weights_trimmed[treated_mask]
        residuals_0 = np.zeros(n_trimmed)
        residuals_0[control_mask] = (Y_trimmed[control_mask] - y0_weighted) * \
                                    weights_trimmed[control_mask]

        influence_fn = residuals_1 - residuals_0
        variance = np.var(influence_fn) / n_trimmed
        se = np.sqrt(variance)

        # 全样本权重 (未截断的为0)
        self.weights = np.zeros(len(Y))
        self.weights[self.trimmed_mask] = weights_trimmed

        return float(ate), float(se), self.weights, self.trimmed_mask
