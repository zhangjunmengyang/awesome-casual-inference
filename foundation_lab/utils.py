"""
因果推断基础工具函数

提供数据生成、可视化辅助等通用功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px


def generate_potential_outcomes_data(
    n_samples: int = 1000,
    treatment_effect: float = 2.0,
    baseline_outcome: float = 5.0,
    noise_std: float = 1.0,
    treatment_prob: float = 0.5,
    confounding_strength: float = 0.0,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    生成潜在结果数据

    Parameters:
    -----------
    n_samples: 样本数量
    treatment_effect: 真实处理效应 (ATE)
    baseline_outcome: 基线结果
    noise_std: 噪声标准差
    treatment_prob: 处理概率 (无混淆时)
    confounding_strength: 混淆强度 (0-1)
    seed: 随机种子

    Returns:
    --------
    DataFrame with columns: X, T, Y0, Y1, Y, ITE
    """
    if seed is not None:
        np.random.seed(seed)

    # 协变量
    X = np.random.randn(n_samples)

    # 潜在结果
    # Y(0) = baseline + noise
    # Y(1) = baseline + treatment_effect + noise
    noise_0 = np.random.randn(n_samples) * noise_std
    noise_1 = np.random.randn(n_samples) * noise_std

    Y0 = baseline_outcome + 0.5 * X + noise_0
    Y1 = baseline_outcome + treatment_effect + 0.5 * X + noise_1

    # 处理分配 (考虑混淆)
    # 混淆: X 影响 T 的概率
    if confounding_strength > 0:
        propensity = 1 / (1 + np.exp(-confounding_strength * X))
        T = np.random.binomial(1, propensity)
    else:
        T = np.random.binomial(1, treatment_prob, n_samples)

    # 观测结果 (只能观测到一个)
    Y = np.where(T == 1, Y1, Y0)

    # 个体处理效应 (真实值，通常不可观测)
    ITE = Y1 - Y0

    return pd.DataFrame({
        'X': X,
        'T': T,
        'Y0': Y0,
        'Y1': Y1,
        'Y': Y,
        'ITE': ITE,
        'propensity': 1 / (1 + np.exp(-confounding_strength * X)) if confounding_strength > 0 else np.full(n_samples, treatment_prob)
    })


def generate_confounded_data(
    n_samples: int = 1000,
    true_ate: float = 2.0,
    confounding_strength: float = 2.0,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    生成带混淆的观测数据

    DAG: X -> T, X -> Y, T -> Y

    Returns:
    --------
    DataFrame and dict with true parameters
    """
    if seed is not None:
        np.random.seed(seed)

    # 混淆变量
    X = np.random.randn(n_samples)

    # 处理分配受 X 影响
    propensity = 1 / (1 + np.exp(-confounding_strength * X))
    T = np.random.binomial(1, propensity)

    # 结果受 X 和 T 影响
    noise = np.random.randn(n_samples) * 0.5
    Y = 1.0 + true_ate * T + confounding_strength * X + noise

    df = pd.DataFrame({
        'X': X,
        'T': T,
        'Y': Y,
        'propensity': propensity
    })

    params = {
        'true_ate': true_ate,
        'confounding_strength': confounding_strength,
        'n_samples': n_samples
    }

    return df, params


def generate_selection_bias_data(
    n_samples: int = 2000,
    selection_strength: float = 1.0,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成选择偏差数据 (Berkson's Paradox 示例)

    场景: 医院住院数据
    - X1: 疾病A严重程度
    - X2: 疾病B严重程度
    - S: 是否住院 (选择变量)

    在总体中 X1 和 X2 独立，但在住院人群中负相关

    Returns:
    --------
    full_data: 全部数据
    selected_data: 选择后的数据
    """
    if seed is not None:
        np.random.seed(seed)

    # 两个独立的疾病严重程度
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)

    # 选择机制: 任一疾病严重都会住院
    selection_prob = 1 / (1 + np.exp(-(selection_strength * (X1 + X2) - 1)))
    S = np.random.binomial(1, selection_prob)

    full_data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'S': S
    })

    selected_data = full_data[full_data['S'] == 1].copy()

    return full_data, selected_data


def calculate_naive_ate(df: pd.DataFrame) -> float:
    """计算朴素 ATE 估计 (简单差分)"""
    treated = df[df['T'] == 1]['Y'].mean()
    control = df[df['T'] == 0]['Y'].mean()
    return treated - control


def calculate_adjusted_ate(df: pd.DataFrame, adjustment_vars: list = ['X']) -> float:
    """计算调整后的 ATE (线性回归)"""
    from sklearn.linear_model import LinearRegression

    X_vars = df[adjustment_vars + ['T']].values
    y = df['Y'].values

    model = LinearRegression()
    model.fit(X_vars, y)

    # 处理效应系数
    return model.coef_[-1]


# ==================== 可视化工具 ====================

def plot_potential_outcomes(df: pd.DataFrame) -> go.Figure:
    """绘制潜在结果散点图"""
    fig = go.Figure()

    # Y(0)
    fig.add_trace(go.Scatter(
        x=df['X'],
        y=df['Y0'],
        mode='markers',
        name='Y(0) - 未处理结果',
        marker=dict(color='blue', opacity=0.5, size=6)
    ))

    # Y(1)
    fig.add_trace(go.Scatter(
        x=df['X'],
        y=df['Y1'],
        mode='markers',
        name='Y(1) - 处理结果',
        marker=dict(color='red', opacity=0.5, size=6)
    ))

    fig.update_layout(
        title='潜在结果框架: Y(0) vs Y(1)',
        xaxis_title='协变量 X',
        yaxis_title='结果 Y',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def plot_observed_outcomes(df: pd.DataFrame) -> go.Figure:
    """绘制观测结果 (按处理状态着色)"""
    fig = go.Figure()

    # 控制组
    control = df[df['T'] == 0]
    fig.add_trace(go.Scatter(
        x=control['X'],
        y=control['Y'],
        mode='markers',
        name='控制组 (T=0)',
        marker=dict(color='blue', opacity=0.6, size=6)
    ))

    # 处理组
    treated = df[df['T'] == 1]
    fig.add_trace(go.Scatter(
        x=treated['X'],
        y=treated['Y'],
        mode='markers',
        name='处理组 (T=1)',
        marker=dict(color='red', opacity=0.6, size=6)
    ))

    fig.update_layout(
        title='观测数据: 我们实际看到的',
        xaxis_title='协变量 X',
        yaxis_title='结果 Y',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def plot_treatment_effect_distribution(df: pd.DataFrame) -> go.Figure:
    """绘制个体处理效应 (ITE) 分布"""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['ITE'],
        nbinsx=30,
        name='ITE 分布',
        marker_color='green',
        opacity=0.7
    ))

    # 添加均值线
    mean_ite = df['ITE'].mean()
    fig.add_vline(
        x=mean_ite,
        line_dash="dash",
        line_color="red",
        annotation_text=f"ATE = {mean_ite:.3f}"
    )

    fig.update_layout(
        title='个体处理效应 (ITE) 分布',
        xaxis_title='ITE = Y(1) - Y(0)',
        yaxis_title='频数',
        template='plotly_white'
    )

    return fig


def plot_propensity_distribution(df: pd.DataFrame) -> go.Figure:
    """绘制倾向得分分布"""
    fig = go.Figure()

    # 控制组
    control = df[df['T'] == 0]
    fig.add_trace(go.Histogram(
        x=control['propensity'],
        name='控制组',
        marker_color='blue',
        opacity=0.6,
        nbinsx=20
    ))

    # 处理组
    treated = df[df['T'] == 1]
    fig.add_trace(go.Histogram(
        x=treated['propensity'],
        name='处理组',
        marker_color='red',
        opacity=0.6,
        nbinsx=20
    ))

    fig.update_layout(
        title='倾向得分分布 (按处理状态)',
        xaxis_title='倾向得分 P(T=1|X)',
        yaxis_title='频数',
        barmode='overlay',
        template='plotly_white'
    )

    return fig


def plot_confounding_effect(
    confounding_values: list,
    naive_estimates: list,
    true_ate: float
) -> go.Figure:
    """绘制混淆强度对估计的影响"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=confounding_values,
        y=naive_estimates,
        mode='lines+markers',
        name='朴素估计',
        line=dict(color='blue', width=2)
    ))

    fig.add_hline(
        y=true_ate,
        line_dash="dash",
        line_color="green",
        annotation_text=f"真实 ATE = {true_ate}"
    )

    fig.update_layout(
        title='混淆强度对 ATE 估计的影响',
        xaxis_title='混淆强度',
        yaxis_title='估计的 ATE',
        template='plotly_white'
    )

    return fig
