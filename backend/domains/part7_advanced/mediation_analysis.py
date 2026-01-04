"""中介分析模块

实现中介效应分解
- 直接效应（Direct Effect）
- 间接效应（Indirect Effect）
- 总效应（Total Effect）
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression


class MediationAnalyzer:
    """
    中介分析器

    分解总效应为直接效应和间接效应
    """

    def __init__(self):
        self.mediator_model = None
        self.outcome_model = None
        self.treatment_binary = False

    def fit(
        self,
        T: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        X: np.ndarray = None
    ):
        """
        拟合中介模型

        Args:
            T: 处理变量 (n,)
            M: 中介变量 (n,)
            Y: 结果变量 (n,)
            X: 协变量 (n, p)，可选
        """
        # 检查处理是否为二元
        self.treatment_binary = len(np.unique(T)) == 2

        # 拟合中介模型: M ~ T + X
        if X is not None:
            features_m = np.column_stack([T.reshape(-1, 1), X])
        else:
            features_m = T.reshape(-1, 1)

        self.mediator_model = LinearRegression()
        self.mediator_model.fit(features_m, M)

        # 拟合结果模型: Y ~ T + M + X
        if X is not None:
            features_y = np.column_stack([T.reshape(-1, 1), M.reshape(-1, 1), X])
        else:
            features_y = np.column_stack([T.reshape(-1, 1), M.reshape(-1, 1)])

        self.outcome_model = LinearRegression()
        self.outcome_model.fit(features_y, Y)

        return self

    def decompose_effects(
        self,
        T: np.ndarray,
        X: np.ndarray = None,
        t0: float = 0,
        t1: float = 1
    ) -> Dict[str, float]:
        """
        分解总效应为直接和间接效应

        使用潜在结果框架：
        - 总效应 (TE): E[Y(1, M(1))] - E[Y(0, M(0))]
        - 自然直接效应 (NDE): E[Y(1, M(0))] - E[Y(0, M(0))]
        - 自然间接效应 (NIE): E[Y(1, M(1))] - E[Y(1, M(0))]

        Args:
            T: 处理变量
            X: 协变量
            t0: 对照水平
            t1: 处理水平

        Returns:
            效应分解字典
        """
        n = len(T)

        # 预测 M(0) 和 M(1)
        T0 = np.full((n, 1), t0)
        T1 = np.full((n, 1), t1)

        if X is not None:
            features_m0 = np.column_stack([T0, X])
            features_m1 = np.column_stack([T1, X])
        else:
            features_m0 = T0
            features_m1 = T1

        M0 = self.mediator_model.predict(features_m0)
        M1 = self.mediator_model.predict(features_m1)

        # 预测潜在结果
        # Y(0, M(0))
        if X is not None:
            features_y00 = np.column_stack([T0, M0.reshape(-1, 1), X])
        else:
            features_y00 = np.column_stack([T0, M0.reshape(-1, 1)])
        Y00 = self.outcome_model.predict(features_y00)

        # Y(1, M(0)) - 用于 NDE
        if X is not None:
            features_y10 = np.column_stack([T1, M0.reshape(-1, 1), X])
        else:
            features_y10 = np.column_stack([T1, M0.reshape(-1, 1)])
        Y10 = self.outcome_model.predict(features_y10)

        # Y(1, M(1))
        if X is not None:
            features_y11 = np.column_stack([T1, M1.reshape(-1, 1), X])
        else:
            features_y11 = np.column_stack([T1, M1.reshape(-1, 1)])
        Y11 = self.outcome_model.predict(features_y11)

        # 计算效应
        total_effect = (Y11 - Y00).mean()
        natural_direct_effect = (Y10 - Y00).mean()
        natural_indirect_effect = (Y11 - Y10).mean()

        # 中介比例
        if abs(total_effect) > 1e-10:
            proportion_mediated = natural_indirect_effect / total_effect
        else:
            proportion_mediated = 0.0

        return {
            'total_effect': total_effect,
            'natural_direct_effect': natural_direct_effect,
            'natural_indirect_effect': natural_indirect_effect,
            'proportion_mediated': proportion_mediated
        }


def baron_kenny_test(
    T: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    X: np.ndarray = None
) -> Dict:
    """
    Baron & Kenny (1986) 经典中介检验

    三步检验：
    1. T 显著影响 Y（总效应）
    2. T 显著影响 M
    3. 控制 M 后，T 对 Y 的效应减弱（部分中介）或消失（完全中介）

    Args:
        T: 处理变量
        M: 中介变量
        Y: 结果变量
        X: 协变量

    Returns:
        包含三步检验结果的字典
    """
    # 步骤 1: Y ~ T (+ X)
    if X is not None:
        features_1 = np.column_stack([T.reshape(-1, 1), X])
    else:
        features_1 = T.reshape(-1, 1)

    model_1 = LinearRegression()
    model_1.fit(features_1, Y)
    coef_t_on_y = model_1.coef_[0]

    # 步骤 2: M ~ T (+ X)
    model_2 = LinearRegression()
    model_2.fit(features_1, M)
    coef_t_on_m = model_2.coef_[0]

    # 步骤 3: Y ~ T + M (+ X)
    if X is not None:
        features_3 = np.column_stack([T.reshape(-1, 1), M.reshape(-1, 1), X])
    else:
        features_3 = np.column_stack([T.reshape(-1, 1), M.reshape(-1, 1)])

    model_3 = LinearRegression()
    model_3.fit(features_3, Y)
    coef_t_on_y_controlled = model_3.coef_[0]
    coef_m_on_y = model_3.coef_[1]

    # 判断中介类型
    if abs(coef_t_on_y_controlled) < 0.01 * abs(coef_t_on_y):
        mediation_type = "full"  # 完全中介
    elif abs(coef_t_on_y_controlled) < abs(coef_t_on_y):
        mediation_type = "partial"  # 部分中介
    else:
        mediation_type = "none"  # 无中介

    return {
        'step1_total_effect': coef_t_on_y,
        'step2_t_to_m': coef_t_on_m,
        'step3_direct_effect': coef_t_on_y_controlled,
        'step3_m_to_y': coef_m_on_y,
        'mediation_type': mediation_type,
        'indirect_effect_estimate': coef_t_on_m * coef_m_on_y
    }


def sensitivity_analysis_mediation(
    observed_indirect: float,
    rho_range: np.ndarray = None
) -> Dict:
    """
    中介分析的敏感性分析

    评估未观测混淆对间接效应的影响

    Args:
        observed_indirect: 观测到的间接效应
        rho_range: 混淆强度范围

    Returns:
        敏感性分析结果
    """
    if rho_range is None:
        rho_range = np.linspace(0, 0.5, 20)

    # 简化的敏感性分析
    # 间接效应在不同混淆水平下的可能值
    adjusted_effects = []

    for rho in rho_range:
        # 简化模型：假设混淆使效应偏离原值
        bias = rho * observed_indirect
        adjusted = observed_indirect - bias
        adjusted_effects.append(adjusted)

    return {
        'rho_range': rho_range,
        'adjusted_effects': np.array(adjusted_effects),
        'observed_effect': observed_indirect
    }


def compute_mediation_bootstrap_ci(
    T: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    X: np.ndarray = None,
    n_bootstrap: int = 100,
    alpha: float = 0.05
) -> Dict:
    """
    Bootstrap 置信区间

    Args:
        T, M, Y, X: 数据
        n_bootstrap: Bootstrap 重复次数
        alpha: 显著性水平

    Returns:
        置信区间
    """
    n = len(T)
    indirect_effects = []

    for _ in range(n_bootstrap):
        # 重采样
        idx = np.random.choice(n, n, replace=True)
        T_boot = T[idx]
        M_boot = M[idx]
        Y_boot = Y[idx]
        X_boot = X[idx] if X is not None else None

        # 拟合并分解
        analyzer = MediationAnalyzer()
        analyzer.fit(T_boot, M_boot, Y_boot, X_boot)
        effects = analyzer.decompose_effects(T_boot, X_boot)

        indirect_effects.append(effects['natural_indirect_effect'])

    # 计算置信区间
    lower = np.percentile(indirect_effects, alpha/2 * 100)
    upper = np.percentile(indirect_effects, (1 - alpha/2) * 100)

    return {
        'indirect_effect_mean': np.mean(indirect_effects),
        'indirect_effect_std': np.std(indirect_effects),
        'ci_lower': lower,
        'ci_upper': upper,
        'alpha': alpha
    }
