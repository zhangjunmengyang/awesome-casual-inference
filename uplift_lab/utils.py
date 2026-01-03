"""
Uplift Modeling 工具函数

提供数据生成、评估指标等通用功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_uplift_data(
    n_samples: int = 5000,
    n_features: int = 10,
    treatment_effect_type: str = 'heterogeneous',
    noise_level: float = 0.5,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成 Uplift Modeling 数据

    Parameters:
    -----------
    n_samples: 样本数量
    n_features: 特征数量
    treatment_effect_type: 处理效应类型
        - 'constant': 常数效应 (ATE = 2)
        - 'heterogeneous': 异质性效应 (依赖特征)
        - 'complex': 复杂非线性效应
    noise_level: 噪声水平
    seed: 随机种子

    Returns:
    --------
    (DataFrame, true_cate)
    DataFrame columns: X1...Xn, T, Y
    true_cate: 真实的个体处理效应
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成特征
    X = np.random.randn(n_samples, n_features)

    # 处理分配 (随机化)
    propensity = 0.5
    T = np.random.binomial(1, propensity, n_samples)

    # 基线结果
    baseline = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1]

    # 处理效应
    if treatment_effect_type == 'constant':
        tau = np.full(n_samples, 2.0)

    elif treatment_effect_type == 'heterogeneous':
        # CATE 依赖于前两个特征
        # tau = 2 + X1 - 0.5 * X2
        tau = 2 + X[:, 0] - 0.5 * X[:, 1]

    elif treatment_effect_type == 'complex':
        # 非线性效应
        tau = 2 * np.sin(X[:, 0]) + X[:, 1] ** 2 - 1

    else:
        raise ValueError(f"Unknown effect type: {treatment_effect_type}")

    # 生成结果
    noise = np.random.randn(n_samples) * noise_level
    Y = baseline + tau * T + noise

    # 创建 DataFrame
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['T'] = T
    df['Y'] = Y

    return df, tau


def generate_marketing_uplift_data(
    n_samples: int = 10000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成营销场景的 Uplift 数据 (发券场景)

    特征:
    - age: 年龄
    - income: 收入水平
    - frequency: 购买频率
    - recency: 最近购买天数
    - is_member: 是否会员

    处理: 发放优惠券
    结果: 是否购买 (转化)

    Returns:
    --------
    (DataFrame, true_uplift)
    """
    if seed is not None:
        np.random.seed(seed)

    # 特征
    age = np.random.uniform(18, 65, n_samples)
    income = np.random.exponential(50000, n_samples)
    frequency = np.random.poisson(5, n_samples)
    recency = np.random.exponential(30, n_samples)
    is_member = np.random.binomial(1, 0.3, n_samples)

    # 标准化
    age_norm = (age - 40) / 15
    income_norm = (np.log(income + 1) - 10) / 1.5
    freq_norm = (frequency - 5) / 3
    recency_norm = (recency - 30) / 20

    # 处理分配 (随机)
    T = np.random.binomial(1, 0.5, n_samples)

    # Uplift 效应 (异质性)
    # 年轻人 + 低频用户对优惠券更敏感
    # 会员对优惠券响应更高
    uplift = (
        0.1 +  # 基础 uplift
        0.05 * (1 - age_norm) +  # 年轻人更敏感
        0.03 * (1 - freq_norm) +  # 低频用户更敏感
        0.04 * is_member +  # 会员更敏感
        0.02 * income_norm  # 高收入略微敏感
    )
    uplift = np.clip(uplift, 0, 0.5)  # 限制在合理范围

    # 基线转化率
    baseline_prob = (
        0.1 +  # 基础转化
        0.03 * freq_norm +  # 高频用户更容易购买
        0.02 * income_norm +  # 高收入更容易购买
        0.02 * is_member -  # 会员更容易购买
        0.01 * recency_norm  # 最近购买过的更容易再次购买
    )
    baseline_prob = np.clip(baseline_prob, 0.02, 0.5)

    # 转化概率
    prob = baseline_prob + uplift * T
    prob = np.clip(prob, 0, 1)

    # 生成转化结果
    Y = np.random.binomial(1, prob)

    # 创建 DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'frequency': frequency,
        'recency': recency,
        'is_member': is_member,
        'T': T,
        'Y': Y
    })

    return df, uplift


def calculate_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Qini 曲线

    Parameters:
    -----------
    y_true: 真实结果
    treatment: 处理状态
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

    # Qini 值
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

        if t_sub.sum() > 0 and (1 - t_sub).sum() > 0:
            uplift = y_sub[t_sub == 1].mean() - y_sub[t_sub == 0].mean()
        else:
            uplift = 0

        cumulative_uplift.append(uplift)

    fraction = np.arange(1, n + 1) / n
    return np.insert(fraction, 0, 0), np.insert(cumulative_uplift, 0, 0)


def calculate_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray
) -> float:
    """
    计算 AUUC (Area Under Uplift Curve)
    """
    fraction, qini = calculate_qini_curve(y_true, treatment, uplift_score)
    # 使用梯形法则计算面积
    auuc = np.trapz(qini, fraction)
    return auuc


def plot_uplift_curves(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: dict,  # {'model_name': scores}
    title: str = "Uplift Curves"
) -> go.Figure:
    """
    绘制多个模型的 Uplift 曲线对比
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Qini Curve', 'Uplift Curve')
    )

    colors = ['#2D9CDB', '#27AE60', '#EB5757', '#9B59B6', '#F2994A']

    for i, (name, scores) in enumerate(uplift_scores.items()):
        color = colors[i % len(colors)]

        # Qini Curve
        fraction, qini = calculate_qini_curve(y_true, treatment, scores)
        auuc = calculate_auuc(y_true, treatment, scores)

        fig.add_trace(go.Scatter(
            x=fraction, y=qini,
            mode='lines',
            name=f'{name} (AUUC={auuc:.4f})',
            line=dict(color=color, width=2)
        ), row=1, col=1)

        # Uplift Curve
        fraction_u, uplift = calculate_uplift_curve(y_true, treatment, scores)
        fig.add_trace(go.Scatter(
            x=fraction_u, y=uplift,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            showlegend=False
        ), row=1, col=2)

    # 添加随机基线
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, qini[-1]],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)

    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white'
    )

    fig.update_xaxes(title_text='Fraction Targeted', row=1, col=1)
    fig.update_xaxes(title_text='Fraction Targeted', row=1, col=2)
    fig.update_yaxes(title_text='Qini Value', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Uplift', row=1, col=2)

    return fig


def plot_cate_distribution(
    true_cate: np.ndarray,
    predicted_cate: dict,  # {'model_name': predictions}
    title: str = "CATE Distribution"
) -> go.Figure:
    """
    绘制 CATE 分布对比
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('CATE Distribution', 'True vs Predicted')
    )

    colors = ['#2D9CDB', '#27AE60', '#EB5757', '#9B59B6']

    # 真实 CATE 分布
    fig.add_trace(go.Histogram(
        x=true_cate,
        name='True CATE',
        marker_color='gray',
        opacity=0.7,
        nbinsx=30
    ), row=1, col=1)

    # 预测 CATE 分布
    for i, (name, pred) in enumerate(predicted_cate.items()):
        color = colors[i % len(colors)]

        fig.add_trace(go.Histogram(
            x=pred,
            name=name,
            marker_color=color,
            opacity=0.5,
            nbinsx=30
        ), row=1, col=1)

        # 散点图: 真实 vs 预测
        # 采样以避免过多点
        sample_idx = np.random.choice(len(true_cate), min(500, len(true_cate)), replace=False)
        fig.add_trace(go.Scatter(
            x=true_cate[sample_idx],
            y=pred[sample_idx],
            mode='markers',
            name=name,
            marker=dict(color=color, size=4, opacity=0.5),
            showlegend=False
        ), row=1, col=2)

    # 添加对角线
    min_val = min(true_cate.min(), min(p.min() for p in predicted_cate.values()))
    max_val = max(true_cate.max(), max(p.max() for p in predicted_cate.values()))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white',
        barmode='overlay'
    )

    fig.update_xaxes(title_text='CATE', row=1, col=1)
    fig.update_xaxes(title_text='True CATE', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Predicted CATE', row=1, col=2)

    return fig
