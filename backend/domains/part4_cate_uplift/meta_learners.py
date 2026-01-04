"""
Meta-Learners 模块

实现 S-Learner, T-Learner, X-Learner, R-Learner, DR-Learner

核心概念:
- S-Learner: 单一模型，处理作为特征
- T-Learner: 两个独立模型，分别建模处理/控制组
- X-Learner: 利用反事实估计的两阶段方法
- R-Learner: 基于残差的双重稳健方法
- DR-Learner: 双重稳健 Meta-Learner
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict
from typing import Optional


class SLearner:
    """
    S-Learner (Single Model)

    将处理 T 作为特征，训练单一模型:
    Y = f(X, T)

    CATE 估计: tau(x) = f(x, 1) - f(x, 0)
    """

    def __init__(self, base_model=None):
        self.base_model = base_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        X_with_T = np.column_stack([X, T])
        # 使用 clone 避免修改原始 base_model
        self.model = clone(self.base_model)
        self.model.fit(X_with_T, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        n = X.shape[0]

        # 预测 Y(1) 和 Y(0)
        X_1 = np.column_stack([X, np.ones(n)])
        X_0 = np.column_stack([X, np.zeros(n)])

        Y_1 = self.model.predict(X_1)
        Y_0 = self.model.predict(X_0)

        return Y_1 - Y_0


class TLearner:
    """
    T-Learner (Two Models)

    分别为处理组和控制组训练模型:
    - mu_0(x) = E[Y|X=x, T=0]
    - mu_1(x) = E[Y|X=x, T=1]

    CATE 估计: tau(x) = mu_1(x) - mu_0(x)
    """

    def __init__(self, base_model_0=None, base_model_1=None):
        self.model_0 = base_model_0 or RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_1 = base_model_1 or RandomForestRegressor(n_estimators=100, random_state=43)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        # 控制组模型
        mask_0 = T == 0
        self.model_0.fit(X[mask_0], Y[mask_0])

        # 处理组模型
        mask_1 = T == 1
        self.model_1.fit(X[mask_1], Y[mask_1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        Y_0 = self.model_0.predict(X)
        Y_1 = self.model_1.predict(X)

        return Y_1 - Y_0


class XLearner:
    """
    X-Learner

    两阶段方法:
    阶段 1: 分别估计 mu_0 和 mu_1 (同 T-Learner)
    阶段 2: 估计伪处理效应
        - D_1 = Y_1 - mu_0(X_1)  (处理组的反事实)
        - D_0 = mu_1(X_0) - Y_0  (控制组的反事实)

    训练 tau_1(x) ~ D_1 和 tau_0(x) ~ D_0
    最终: tau(x) = g(x) * tau_0(x) + (1-g(x)) * tau_1(x)

    其中 g(x) = P(T=1|X=x) 是倾向得分
    """

    def __init__(self, outcome_model=None, effect_model=None, propensity_model=None):
        # 使用 clone 确保每个模型是独立实例，避免共享状态
        default_outcome = RandomForestRegressor(n_estimators=100, random_state=42)
        default_effect = RandomForestRegressor(n_estimators=100, random_state=44)

        self.model_0 = clone(outcome_model) if outcome_model else clone(default_outcome)
        self.model_1 = clone(outcome_model) if outcome_model else RandomForestRegressor(n_estimators=100, random_state=43)
        self.tau_0 = clone(effect_model) if effect_model else clone(default_effect)
        self.tau_1 = clone(effect_model) if effect_model else RandomForestRegressor(n_estimators=100, random_state=45)
        self.propensity = clone(propensity_model) if propensity_model else LogisticRegression(random_state=42)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        mask_0 = T == 0
        mask_1 = T == 1

        # 阶段 1: 结果模型
        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])

        # 阶段 2: 伪处理效应
        # 处理组: D_1 = Y - mu_0(X)
        D_1 = Y[mask_1] - self.model_0.predict(X[mask_1])
        self.tau_1.fit(X[mask_1], D_1)

        # 控制组: D_0 = mu_1(X) - Y
        D_0 = self.model_1.predict(X[mask_0]) - Y[mask_0]
        self.tau_0.fit(X[mask_0], D_0)

        # 倾向得分
        self.propensity.fit(X, T)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        tau_0_pred = self.tau_0.predict(X)
        tau_1_pred = self.tau_1.predict(X)

        # 倾向得分作为权重
        g = self.propensity.predict_proba(X)[:, 1]

        # 加权组合
        return g * tau_0_pred + (1 - g) * tau_1_pred


class RLearner:
    """
    R-Learner (Residual/Robinson Learner)

    基于 Robinson 分解的双重稳健方法:
    Y - m(X) = tau(X) * (T - e(X)) + epsilon

    其中:
    - m(x) = E[Y|X]
    - e(x) = P(T=1|X) 倾向得分

    通过最小化加权残差来估计 tau
    """

    def __init__(self, outcome_model=None, propensity_model=None, effect_model=None):
        self.outcome_model = outcome_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.propensity_model = propensity_model or LogisticRegression(random_state=42)
        self.effect_model = effect_model or GradientBoostingRegressor(n_estimators=100, random_state=42)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        # 估计 m(x) = E[Y|X]
        m_hat = cross_val_predict(self.outcome_model, X, Y, cv=5)

        # 估计 e(x) = P(T=1|X)
        e_hat = cross_val_predict(self.propensity_model, X, T, cv=5, method='predict_proba')[:, 1]

        # 先 clip propensity scores 以避免极端值
        e_hat = np.clip(e_hat, 0.01, 0.99)

        # 计算残差
        Y_residual = Y - m_hat
        T_residual = T - e_hat

        # 再次 clip 以确保数值稳定性
        T_residual = np.clip(T_residual, -0.95, 0.95)

        # 伪结果 (避免除以接近零的值)
        pseudo_outcome = Y_residual / np.where(np.abs(T_residual) < 0.05,
                                                np.sign(T_residual) * 0.05,
                                                T_residual)

        # 加权回归
        weights = T_residual ** 2

        self.effect_model.fit(X, pseudo_outcome, sample_weight=weights)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        return self.effect_model.predict(X)


class DRLearner:
    """
    DR-Learner (Doubly Robust Learner)

    双重稳健的 Meta-Learner，结合倾向得分和结果回归:

    伪结果: Y_DR = mu_1(X) - mu_0(X) + T/e(X) * (Y - mu_1(X)) - (1-T)/(1-e(X)) * (Y - mu_0(X))

    然后使用机器学习模型预测 Y_DR ~ X
    """

    def __init__(self, outcome_model=None, propensity_model=None, effect_model=None):
        self.outcome_model_0 = outcome_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.outcome_model_1 = clone(self.outcome_model_0) if outcome_model else RandomForestRegressor(n_estimators=100, random_state=43)
        self.propensity_model = propensity_model or LogisticRegression(random_state=42)
        self.effect_model = effect_model or GradientBoostingRegressor(n_estimators=100, random_state=42)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        # 估计结果模型
        mask_0 = T == 0
        mask_1 = T == 1

        self.outcome_model_0.fit(X[mask_0], Y[mask_0])
        self.outcome_model_1.fit(X[mask_1], Y[mask_1])

        mu_0 = self.outcome_model_0.predict(X)
        mu_1 = self.outcome_model_1.predict(X)

        # 估计倾向得分
        self.propensity_model.fit(X, T)
        e = self.propensity_model.predict_proba(X)[:, 1]
        e = np.clip(e, 0.01, 0.99)  # 避免除零

        # 计算双重稳健伪结果
        Y_DR = (
            (mu_1 - mu_0) +
            (T / e) * (Y - mu_1) -
            ((1 - T) / (1 - e)) * (Y - mu_0)
        )

        # 训练效应模型
        self.effect_model.fit(X, Y_DR)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        return self.effect_model.predict(X)
