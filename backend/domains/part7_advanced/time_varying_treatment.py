"""时变处理效应模块

实现时间序列因果推断方法
- 边际结构模型（MSM）
- G-computation
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.linear_model import LogisticRegression, LinearRegression


def estimate_time_varying_weights(
    data: pd.DataFrame,
    treatment_col: str = 'T',
    covariate_cols: List[str] = None,
    stabilized: bool = True
) -> np.ndarray:
    """
    估计时变处理的逆概率权重（IPW）

    Args:
        data: 长格式数据（包含 subject_id, period）
        treatment_col: 处理变量列名
        covariate_cols: 协变量列名
        stabilized: 是否使用稳定权重

    Returns:
        每个观测的权重
    """
    if covariate_cols is None:
        covariate_cols = ['X']

    subjects = data['subject_id'].unique()
    n_subjects = len(subjects)
    periods = sorted(data['period'].unique())

    weights = np.ones(len(data))

    for subject_id in subjects:
        subject_data = data[data['subject_id'] == subject_id].sort_values('period')
        subject_weight = 1.0

        for idx, row in subject_data.iterrows():
            period = row['period']

            if period == 0:
                # 第一期：使用边缘概率
                if stabilized:
                    # 稳定权重的分子
                    overall_prob = data[data['period'] == period][treatment_col].mean()
                    numerator = overall_prob if row[treatment_col] == 1 else (1 - overall_prob)
                else:
                    numerator = 1.0
            else:
                # 后续期：使用条件概率
                # 获取历史数据
                history_mask = (data['subject_id'] == subject_id) & (data['period'] < period)
                history = data[history_mask]

                if stabilized and len(history) > 0:
                    # 稳定权重：仅基于历史处理
                    hist_treatments = history[treatment_col].values
                    same_history_mask = True
                    for t_idx, t_val in enumerate(hist_treatments):
                        same_history_mask &= (data['period'] == t_idx) & (data[treatment_col] == t_val)

                    if same_history_mask.sum() > 0:
                        numerator = data[same_history_mask & (data['period'] == period)][treatment_col].mean()
                        if row[treatment_col] == 0:
                            numerator = 1 - numerator
                    else:
                        numerator = 0.5
                else:
                    numerator = 1.0

            # 分母：给定协变量和历史处理的概率
            # 简化实现：使用当前协变量
            X_current = row[covariate_cols].values.reshape(1, -1)

            # 拟合当前期的倾向得分模型（使用所有该期数据）
            period_data = data[data['period'] == period]
            if len(period_data) > 10 and period_data[treatment_col].nunique() > 1:
                try:
                    ps_model = LogisticRegression(max_iter=1000, solver='lbfgs')
                    ps_model.fit(period_data[covariate_cols], period_data[treatment_col])
                    prob_t = ps_model.predict_proba(X_current)[0, 1]
                except:
                    prob_t = period_data[treatment_col].mean()
            else:
                prob_t = period_data[treatment_col].mean()

            denominator = prob_t if row[treatment_col] == 1 else (1 - prob_t)
            denominator = np.clip(denominator, 0.01, 0.99)

            # 累积权重
            subject_weight *= numerator / denominator
            weights[data.index == idx] = subject_weight

    # 截断极端权重
    weights = np.clip(weights, 0.1, 10)

    return weights


def estimate_msm(
    data: pd.DataFrame,
    outcome_col: str = 'Y',
    treatment_col: str = 'T',
    weights: np.ndarray = None
) -> Dict:
    """
    估计边际结构模型（MSM）

    Args:
        data: 长格式数据
        outcome_col: 结果变量列名
        treatment_col: 处理变量列名
        weights: IPW 权重

    Returns:
        MSM 估计结果
    """
    if weights is None:
        weights = np.ones(len(data))

    # 计算累积处理
    data_copy = data.copy()
    data_copy['cumulative_treatment'] = data_copy.groupby('subject_id')[treatment_col].cumsum()

    # 拟合加权回归
    # E[Y_t] = β0 + β1 * cumulative_treatment_t
    X = data_copy[['cumulative_treatment']].values
    y = data_copy[outcome_col].values

    # 加权最小二乘
    model = LinearRegression()
    # 手动实现加权，处理 NaN
    weights_clean = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights_clean = np.clip(weights_clean, 0.1, 10.0)

    X_weighted = X * np.sqrt(weights_clean).reshape(-1, 1)
    y_weighted = y * np.sqrt(weights_clean)

    model.fit(X_weighted, y_weighted)

    # 提取系数
    treatment_effect = model.coef_[0]
    baseline = model.intercept_

    # 预测不同累积处理水平的结果
    cumulative_range = np.arange(0, data_copy['cumulative_treatment'].max() + 1)
    predicted_outcomes = baseline + treatment_effect * cumulative_range

    return {
        'treatment_effect': treatment_effect,
        'baseline': baseline,
        'cumulative_range': cumulative_range,
        'predicted_outcomes': predicted_outcomes,
        'model': model
    }


def g_computation(
    data: pd.DataFrame,
    outcome_col: str = 'Y',
    treatment_col: str = 'T',
    covariate_cols: List[str] = None,
    intervention: str = 'always_treat'
) -> Dict:
    """
    G-computation 估计

    Args:
        data: 长格式数据
        outcome_col: 结果变量列名
        treatment_col: 处理变量列名
        covariate_cols: 协变量列名
        intervention: 干预策略 ('always_treat', 'never_treat', 'natural')

    Returns:
        干预效应估计
    """
    if covariate_cols is None:
        covariate_cols = ['X']

    periods = sorted(data['period'].unique())
    subjects = data['subject_id'].unique()

    # 拟合结果模型
    # Y_t = f(T_t, X_t, history)
    all_features = covariate_cols + [treatment_col]

    # 简化：仅使用当前期变量
    outcome_model = LinearRegression()
    X_train = data[all_features].values
    y_train = data[outcome_col].values
    outcome_model.fit(X_train, y_train)

    # 模拟干预
    potential_outcomes = []

    for subject_id in subjects:
        subject_data = data[data['subject_id'] == subject_id].sort_values('period')

        for period in periods:
            period_data = subject_data[subject_data['period'] == period]

            if len(period_data) == 0:
                continue

            # 构造干预下的特征
            X_intervened = period_data[covariate_cols].values

            if intervention == 'always_treat':
                T_intervened = np.ones((len(period_data), 1))
            elif intervention == 'never_treat':
                T_intervened = np.zeros((len(period_data), 1))
            else:  # natural
                T_intervened = period_data[[treatment_col]].values

            X_full = np.hstack([X_intervened, T_intervened])

            # 预测潜在结果
            Y_potential = outcome_model.predict(X_full)
            potential_outcomes.extend(Y_potential)

    mean_outcome = np.mean(potential_outcomes)

    # 计算自然结果（无干预）
    natural_outcomes = outcome_model.predict(data[all_features].values)
    natural_mean = natural_outcomes.mean()

    return {
        'intervention': intervention,
        'mean_outcome': mean_outcome,
        'natural_mean': natural_mean,
        'effect': mean_outcome - natural_mean,
        'potential_outcomes': potential_outcomes
    }


def compute_cumulative_effect(
    data: pd.DataFrame,
    treatment_col: str = 'T',
    outcome_col: str = 'Y'
) -> pd.DataFrame:
    """
    计算累积处理效应

    Args:
        data: 长格式数据
        treatment_col: 处理变量列名
        outcome_col: 结果变量列名

    Returns:
        包含累积效应的数据框
    """
    result = data.copy()

    # 按个体计算累积处理
    result['cumulative_treatment'] = result.groupby('subject_id')[treatment_col].cumsum()

    # 计算不同累积水平的平均结果
    cumulative_summary = result.groupby('cumulative_treatment')[outcome_col].agg([
        'mean', 'std', 'count'
    ]).reset_index()

    return cumulative_summary
