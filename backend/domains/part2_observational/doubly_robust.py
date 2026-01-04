"""
双重稳健估计模块

实现多种双重稳健方法:
- 标准 AIPW (Augmented IPW)
- 双重稳健估计器
- TMLE (Targeted Maximum Likelihood Estimation)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import BaseEstimator, clone
from typing import Tuple, Optional


class DoublyRobustEstimator:
    """
    双重稳健估计器 (Doubly Robust Estimator)

    结合倾向得分模型和结果模型
    只要两个模型中有一个正确，估计就是一致的
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        outcome_model: Optional[BaseEstimator] = None
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型 (分类器)
        outcome_model: 结果模型 (回归器)
        """
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=42
        )

        if outcome_model is None:
            self.outcome_model_0 = Ridge(alpha=1.0, random_state=42)
            self.outcome_model_1 = Ridge(alpha=1.0, random_state=43)
        else:
            self.outcome_model_0 = clone(outcome_model)
            self.outcome_model_1 = clone(outcome_model)

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        propensity_correct: bool = True,
        outcome_correct: bool = True
    ) -> Tuple[float, float]:
        """
        估计 ATE

        Parameters:
        -----------
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量
        propensity_correct: 倾向得分模型是否使用全部特征
        outcome_correct: 结果模型是否使用全部特征

        Returns:
        --------
        (ATE估计, 标准误)
        """
        n = len(Y)

        # 1. 估计倾向得分 (可能误设定)
        if propensity_correct:
            X_prop = X
        else:
            # 人为误设定: 只使用部分特征
            X_prop = X[:, :2]

        self.propensity_model.fit(X_prop, T)
        propensity = self.propensity_model.predict_proba(X_prop)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)

        # 2. 估计结果模型 (可能误设定)
        treated_mask = T == 1
        control_mask = T == 0

        if outcome_correct:
            X_outcome = X
        else:
            # 人为误设定: 只使用部分特征
            X_outcome = X[:, :2]

        # mu_1(X) = E[Y|X, T=1]
        self.outcome_model_1.fit(X_outcome[treated_mask], Y[treated_mask])
        mu_1 = self.outcome_model_1.predict(X_outcome)

        # mu_0(X) = E[Y|X, T=0]
        self.outcome_model_0.fit(X_outcome[control_mask], Y[control_mask])
        mu_0 = self.outcome_model_0.predict(X_outcome)

        # 3. 双重稳健估计
        # DR score = (mu_1 - mu_0) + T*(Y - mu_1)/e - (1-T)*(Y - mu_0)/(1-e)
        term1 = mu_1 - mu_0
        term2 = T * (Y - mu_1) / propensity
        term3 = (1 - T) * (Y - mu_0) / (1 - propensity)

        dr_scores = term1 + term2 - term3

        ate = dr_scores.mean()
        se = dr_scores.std() / np.sqrt(n)

        return float(ate), float(se)


class AIPWEstimator:
    """
    增强逆概率加权 (Augmented IPW) 估计器

    与 DoublyRobustEstimator 相同，但提供更多灵活性
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        outcome_model: Optional[BaseEstimator] = None,
        clip_propensity: Tuple[float, float] = (0.01, 0.99)
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        outcome_model: 结果模型
        clip_propensity: 倾向得分截断范围
        """
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=42
        )

        if outcome_model is None:
            self.outcome_model_0 = Ridge(alpha=1.0, random_state=42)
            self.outcome_model_1 = Ridge(alpha=1.0, random_state=43)
        else:
            self.outcome_model_0 = clone(outcome_model)
            self.outcome_model_1 = clone(outcome_model)

        self.clip_propensity = clip_propensity
        self.propensity = None
        self.mu_0 = None
        self.mu_1 = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练倾向得分模型和结果模型

        Parameters:
        -----------
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量
        """
        # 训练倾向得分模型
        self.propensity_model.fit(X, T)
        self.propensity = self.propensity_model.predict_proba(X)[:, 1]
        self.propensity = np.clip(self.propensity, *self.clip_propensity)

        # 训练结果模型
        treated_mask = T == 1
        control_mask = T == 0

        self.outcome_model_1.fit(X[treated_mask], Y[treated_mask])
        self.mu_1 = self.outcome_model_1.predict(X)

        self.outcome_model_0.fit(X[control_mask], Y[control_mask])
        self.mu_0 = self.outcome_model_0.predict(X)

        return self

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, float]:
        """
        估计 ATE

        Returns:
        --------
        (ATE估计, 标准误)
        """
        # 如果还未训练，先训练
        if self.propensity is None:
            self.fit(X, T, Y)

        n = len(Y)

        # AIPW 分数
        term1 = self.mu_1 - self.mu_0
        term2 = T * (Y - self.mu_1) / self.propensity
        term3 = (1 - T) * (Y - self.mu_0) / (1 - self.propensity)

        aipw_scores = term1 + term2 - term3

        ate = aipw_scores.mean()
        se = aipw_scores.std() / np.sqrt(n)

        return float(ate), float(se)

    def predict_individual_effects(self, X: np.ndarray) -> np.ndarray:
        """
        预测个体处理效应

        Returns:
        --------
        个体处理效应估计 (仅基于结果模型)
        """
        if self.mu_0 is None or self.mu_1 is None:
            raise ValueError("Must call fit() first")

        return self.mu_1 - self.mu_0


class TMLEEstimator:
    """
    目标最大似然估计 (Targeted Maximum Likelihood Estimation)

    van der Laan & Rubin (2006)
    一种双重稳健方法，通过迭代更新结果模型来减少偏差
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        outcome_model: Optional[BaseEstimator] = None,
        clip_propensity: Tuple[float, float] = (0.025, 0.975),
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        outcome_model: 结果模型
        clip_propensity: 倾向得分截断范围
        max_iter: 最大迭代次数
        tol: 收敛阈值
        """
        self.propensity_model = propensity_model or LogisticRegression(
            max_iter=1000, random_state=42
        )

        if outcome_model is None:
            self.outcome_model_0 = Ridge(alpha=1.0, random_state=42)
            self.outcome_model_1 = Ridge(alpha=1.0, random_state=43)
        else:
            self.outcome_model_0 = clone(outcome_model)
            self.outcome_model_1 = clone(outcome_model)

        self.clip_propensity = clip_propensity
        self.max_iter = max_iter
        self.tol = tol

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, float]:
        """
        估计 ATE using TMLE

        TMLE 步骤:
        1. 估计初始结果模型 Q0(X, T)
        2. 估计倾向得分 g(X)
        3. 计算 clever covariate: H(T, X) = T/g(X) - (1-T)/(1-g(X))
        4. 更新 Q0 使用 logistic regression: logit(Q) = logit(Q0) + ε*H
        5. 计算 ATE = mean(Q(X, 1) - Q(X, 0))

        Returns:
        --------
        (ATE估计, 标准误)
        """
        n = len(Y)

        # Step 1: 估计初始结果模型
        treated_mask = T == 1
        control_mask = T == 0

        self.outcome_model_1.fit(X[treated_mask], Y[treated_mask])
        mu_1 = self.outcome_model_1.predict(X)

        self.outcome_model_0.fit(X[control_mask], Y[control_mask])
        mu_0 = self.outcome_model_0.predict(X)

        # 确保结果在 [0, 1] 区间 (如果不是，进行转换)
        y_min, y_max = Y.min(), Y.max()
        if y_min < 0 or y_max > 1:
            # 线性缩放到 [0.01, 0.99]
            Y_scaled = 0.01 + 0.98 * (Y - y_min) / (y_max - y_min + 1e-8)
            mu_1_scaled = 0.01 + 0.98 * (mu_1 - y_min) / (y_max - y_min + 1e-8)
            mu_0_scaled = 0.01 + 0.98 * (mu_0 - y_min) / (y_max - y_min + 1e-8)
        else:
            Y_scaled = np.clip(Y, 0.01, 0.99)
            mu_1_scaled = np.clip(mu_1, 0.01, 0.99)
            mu_0_scaled = np.clip(mu_0, 0.01, 0.99)

        # Step 2: 估计倾向得分
        self.propensity_model.fit(X, T)
        propensity = self.propensity_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, *self.clip_propensity)

        # Step 3: 计算 clever covariate
        H_1 = 1.0 / propensity
        H_0 = -1.0 / (1.0 - propensity)

        # Step 4: 目标更新 (简化版: 一步更新)
        # 计算 Q_bar = Q0 + ε*H
        # 使用观测数据拟合 ε

        # Clever covariate for observed treatment
        H = np.where(T == 1, H_1, H_0)

        # 观测的 Q0
        Q0_obs = np.where(T == 1, mu_1_scaled, mu_0_scaled)

        # Logit transform
        logit_Q0 = np.log(Q0_obs / (1 - Q0_obs))

        # 拟合 epsilon (使用简单的加权最小二乘)
        # logit(Y) ≈ logit(Q0) + ε*H
        logit_Y = np.log(Y_scaled / (1 - Y_scaled))
        residuals = logit_Y - logit_Q0

        # 估计 epsilon
        epsilon = np.sum(H * residuals) / np.sum(H ** 2)

        # 更新 Q
        logit_Q1_updated = np.log(mu_1_scaled / (1 - mu_1_scaled)) + epsilon * H_1
        logit_Q0_updated = np.log(mu_0_scaled / (1 - mu_0_scaled)) + epsilon * H_0

        Q1_updated = 1 / (1 + np.exp(-logit_Q1_updated))
        Q0_updated = 1 / (1 + np.exp(-logit_Q0_updated))

        # 转换回原始尺度
        if y_min < 0 or y_max > 1:
            Q1_updated = y_min + (Q1_updated - 0.01) / 0.98 * (y_max - y_min)
            Q0_updated = y_min + (Q0_updated - 0.01) / 0.98 * (y_max - y_min)

        # Step 5: 计算 ATE
        ate = (Q1_updated - Q0_updated).mean()

        # 标准误 (使用影响函数)
        D_i = (Q1_updated - Q0_updated) + \
              T * (Y - Q1_updated) / propensity - \
              (1 - T) * (Y - Q0_updated) / (1 - propensity)

        se = D_i.std() / np.sqrt(n)

        return float(ate), float(se)
