"""
偏差类型分析 (Bias Types)

统一处理各种因果推断中的偏差:
1. 混淆偏差 (Confounding Bias)
2. 选择偏差 (Selection Bias)
3. 测量偏差 (Measurement Bias)
4. 碰撞偏差 (Collider Bias / Berkson's Paradox)
5. 辛普森悖论 (Simpson's Paradox)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple
from sklearn.linear_model import LinearRegression

from .utils import generate_confounded_data


def analyze_confounding_bias(
    n_samples: int = 1000,
    confounding_strength: float = 1.0,
    treatment_effect: float = 2.0
) -> Tuple[go.Figure, dict]:
    """
    分析混淆偏差

    Returns:
    --------
    figure: 可视化图表
    stats: 统计信息
    """
    # 生成数据
    df, params = generate_confounded_data(
        n_samples=n_samples,
        true_ate=treatment_effect,
        confounding_strength=confounding_strength
    )

    # 计算估计
    naive_ate = df[df['T'] == 1]['Y'].mean() - df[df['T'] == 0]['Y'].mean()

    # 调整估计
    model = LinearRegression()
    model.fit(df[['T', 'X']], df['Y'])
    adjusted_ate = model.coef_[0]

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '协变量分布 (X)',
            '结果 vs 协变量',
            '倾向得分分布',
            '估计对比'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    control = df[df['T'] == 0]
    treated = df[df['T'] == 1]

    # 1. 协变量分布
    fig.add_trace(go.Histogram(
        x=control['X'],
        name='控制组',
        marker_color='#2D9CDB',
        opacity=0.7,
        nbinsx=25
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=treated['X'],
        name='处理组',
        marker_color='#EB5757',
        opacity=0.7,
        nbinsx=25
    ), row=1, col=1)

    # 2. 结果散点图
    fig.add_trace(go.Scatter(
        x=control['X'], y=control['Y'],
        mode='markers',
        name='控制组',
        marker=dict(color='#2D9CDB', opacity=0.5, size=4),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=treated['X'], y=treated['Y'],
        mode='markers',
        name='处理组',
        marker=dict(color='#EB5757', opacity=0.5, size=4),
        showlegend=False
    ), row=1, col=2)

    # 3. 倾向得分分布
    fig.add_trace(go.Histogram(
        x=control['propensity'],
        name='控制组 PS',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=20,
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=treated['propensity'],
        name='处理组 PS',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=20,
        showlegend=False
    ), row=2, col=1)

    # 4. 估计对比
    estimates = ['真实 ATE', '朴素估计', '调整估计']
    values = [treatment_effect, naive_ate, adjusted_ate]
    colors = ['#27AE60', '#EB5757', '#2D9CDB']

    fig.add_trace(go.Bar(
        x=estimates,
        y=values,
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='混淆偏差分析',
        barmode='overlay'
    )

    fig.update_xaxes(title_text="协变量 X", row=1, col=1)
    fig.update_xaxes(title_text="协变量 X", row=1, col=2)
    fig.update_xaxes(title_text="倾向得分", row=2, col=1)
    fig.update_yaxes(title_text="频数", row=1, col=1)
    fig.update_yaxes(title_text="结果 Y", row=1, col=2)
    fig.update_yaxes(title_text="频数", row=2, col=1)
    fig.update_yaxes(title_text="ATE 估计", row=2, col=2)

    stats = {
        'true_ate': float(treatment_effect),
        'naive_ate': float(naive_ate),
        'adjusted_ate': float(adjusted_ate),
        'bias': float(naive_ate - treatment_effect),
        'confounding_strength': float(confounding_strength)
    }

    return fig, stats


def analyze_selection_bias(
    n_samples: int = 1000,
    selection_strength: float = 1.0,
    treatment_effect: float = 2.0
) -> Tuple[go.Figure, dict]:
    """
    分析选择偏差 (样本选择与结果相关)

    Returns:
    --------
    figure: 可视化图表
    stats: 统计信息
    """
    np.random.seed(42)

    # 生成完整数据
    X = np.random.normal(0, 1, n_samples)
    T = np.random.binomial(1, 0.5, n_samples)
    Y0 = 1 + 0.5 * X + np.random.normal(0, 1, n_samples)
    Y1 = Y0 + treatment_effect
    Y = np.where(T == 1, Y1, Y0)

    # 选择偏差: 结果好的样本更可能被观测到
    selection_prob = 1 / (1 + np.exp(-selection_strength * Y))
    selected = np.random.binomial(1, selection_prob).astype(bool)

    # 计算统计
    full_ate = treatment_effect
    full_naive = Y[T == 1].mean() - Y[T == 0].mean()

    Y_sel = Y[selected]
    T_sel = T[selected]
    if T_sel.sum() > 0 and (1 - T_sel).sum() > 0:
        selected_naive = Y_sel[T_sel == 1].mean() - Y_sel[T_sel == 0].mean()
    else:
        selected_naive = np.nan

    # 可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('完整样本', '选择后样本')
    )

    # 完整样本
    fig.add_trace(go.Scatter(
        x=X[T == 0], y=Y[T == 0],
        mode='markers',
        name='控制组',
        marker=dict(color='#2D9CDB', opacity=0.5, size=5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=X[T == 1], y=Y[T == 1],
        mode='markers',
        name='处理组',
        marker=dict(color='#EB5757', opacity=0.5, size=5)
    ), row=1, col=1)

    # 选择后样本
    fig.add_trace(go.Scatter(
        x=X[selected & (T == 0)], y=Y[selected & (T == 0)],
        mode='markers',
        name='控制组 (选择后)',
        marker=dict(color='#2D9CDB', opacity=0.7, size=6),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=X[selected & (T == 1)], y=Y[selected & (T == 1)],
        mode='markers',
        name='处理组 (选择后)',
        marker=dict(color='#EB5757', opacity=0.7, size=6),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text='选择偏差分析'
    )

    fig.update_xaxes(title_text="协变量 X", row=1, col=1)
    fig.update_xaxes(title_text="协变量 X", row=1, col=2)
    fig.update_yaxes(title_text="结果 Y", row=1, col=1)
    fig.update_yaxes(title_text="结果 Y", row=1, col=2)

    stats = {
        'true_ate': float(full_ate),
        'full_naive': float(full_naive),
        'selected_naive': float(selected_naive) if not np.isnan(selected_naive) else None,
        'selection_bias': float(selected_naive - full_ate) if not np.isnan(selected_naive) else None,
        'selection_rate': float(selected.mean())
    }

    return fig, stats


def analyze_measurement_bias(
    n_samples: int = 1000,
    measurement_error_std: float = 1.0
) -> Tuple[go.Figure, dict]:
    """
    分析测量偏差 (变量测量存在误差)

    Returns:
    --------
    figure: 可视化图表
    stats: 统计信息
    """
    np.random.seed(42)

    # 真实变量
    X_true = np.random.randn(n_samples)
    T = np.random.binomial(1, 0.5, n_samples)
    true_ate = 2.0
    Y = 1 + true_ate * T + 0.5 * X_true + np.random.randn(n_samples) * 0.5

    # 测量误差
    X_measured = X_true + np.random.randn(n_samples) * measurement_error_std

    # 使用真实X和测量X的估计对比
    model_true = LinearRegression()
    model_true.fit(np.column_stack([T, X_true]), Y)
    ate_true = model_true.coef_[0]

    model_measured = LinearRegression()
    model_measured.fit(np.column_stack([T, X_measured]), Y)
    ate_measured = model_measured.coef_[0]

    # 可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('真实 X vs 测量 X', '估计对比')
    )

    # 1. 真实vs测量
    fig.add_trace(go.Scatter(
        x=X_true, y=X_measured,
        mode='markers',
        marker=dict(color='#2D9CDB', opacity=0.5, size=4),
        name='数据点',
        showlegend=False
    ), row=1, col=1)

    # 添加 y=x 参考线
    x_range = [X_true.min(), X_true.max()]
    fig.add_trace(go.Scatter(
        x=x_range, y=x_range,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='y=x',
        showlegend=False
    ), row=1, col=1)

    # 2. 估计对比
    estimates = ['真实 ATE', '使用真实X', '使用测量X']
    values = [true_ate, ate_true, ate_measured]
    colors = ['#27AE60', '#2D9CDB', '#EB5757']

    fig.add_trace(go.Bar(
        x=estimates,
        y=values,
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text=f'测量偏差分析 (误差标准差 = {measurement_error_std})'
    )

    fig.update_xaxes(title_text="真实 X", row=1, col=1)
    fig.update_xaxes(title_text="估计方法", row=1, col=2)
    fig.update_yaxes(title_text="测量 X", row=1, col=1)
    fig.update_yaxes(title_text="ATE 估计", row=1, col=2)

    stats = {
        'true_ate': float(true_ate),
        'ate_with_true_x': float(ate_true),
        'ate_with_measured_x': float(ate_measured),
        'attenuation_bias': float(ate_measured - true_ate),
        'measurement_error_std': float(measurement_error_std)
    }

    return fig, stats


def demonstrate_simpsons_paradox() -> Tuple[go.Figure, dict]:
    """
    演示辛普森悖论: 整体趋势与分组趋势相反

    Returns:
    --------
    figure: 可视化图表
    stats: 统计信息
    """
    np.random.seed(42)
    n_per_group = 200

    # 组 A: 高 X, 高处理率, 高结果
    X_A = np.random.randn(n_per_group) + 2
    T_A = np.random.binomial(1, 0.8, n_per_group)
    Y_A = 10 - 1.5 * T_A + 0.5 * X_A + np.random.randn(n_per_group) * 0.5

    # 组 B: 低 X, 低处理率, 低结果
    X_B = np.random.randn(n_per_group) - 2
    T_B = np.random.binomial(1, 0.2, n_per_group)
    Y_B = 10 - 1.5 * T_B + 0.5 * X_B + np.random.randn(n_per_group) * 0.5

    # 合并数据
    df = pd.DataFrame({
        'X': np.concatenate([X_A, X_B]),
        'T': np.concatenate([T_A, T_B]),
        'Y': np.concatenate([Y_A, Y_B]),
        'Group': ['A'] * n_per_group + ['B'] * n_per_group
    })

    # 计算各种估计
    overall_effect = df[df['T'] == 1]['Y'].mean() - df[df['T'] == 0]['Y'].mean()

    df_A = df[df['Group'] == 'A']
    effect_A = df_A[df_A['T'] == 1]['Y'].mean() - df_A[df_A['T'] == 0]['Y'].mean()

    df_B = df[df['Group'] == 'B']
    effect_B = df_B[df_B['T'] == 1]['Y'].mean() - df_B[df_B['T'] == 0]['Y'].mean()

    # 可视化
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'整体数据 (效应: {overall_effect:.2f})',
            f'组 A (效应: {effect_A:.2f})',
            f'组 B (效应: {effect_B:.2f})'
        )
    )

    # 整体
    for t_val, color, name in [(0, '#2D9CDB', '控制'), (1, '#EB5757', '处理')]:
        subset = df[df['T'] == t_val]
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'],
            mode='markers',
            marker=dict(color=color, opacity=0.5, size=5),
            name=f'{name}组',
            legendgroup=name
        ), row=1, col=1)

    # 组 A
    for t_val, color in [(0, '#2D9CDB'), (1, '#EB5757')]:
        subset = df_A[df_A['T'] == t_val]
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'],
            mode='markers',
            marker=dict(color=color, opacity=0.5, size=5),
            showlegend=False
        ), row=1, col=2)

    # 组 B
    for t_val, color in [(0, '#2D9CDB'), (1, '#EB5757')]:
        subset = df_B[df_B['T'] == t_val]
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'],
            mode='markers',
            marker=dict(color=color, opacity=0.5, size=5),
            showlegend=False
        ), row=1, col=3)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="Simpson's Paradox 演示"
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="协变量 X", row=1, col=i)
        fig.update_yaxes(title_text="结果 Y", row=1, col=i)

    stats = {
        'overall_effect': float(overall_effect),
        'effect_group_a': float(effect_A),
        'effect_group_b': float(effect_B),
        'true_effect': -1.5,
        'paradox': overall_effect > 0 and effect_A < 0 and effect_B < 0
    }

    return fig, stats


def demonstrate_berksons_paradox(
    n_samples: int = 2000,
    selection_strength: float = 1.5
) -> Tuple[go.Figure, dict]:
    """
    演示 Berkson's Paradox (碰撞偏差)

    场景: 医院住院研究
    - X1: 疾病A严重程度 (独立于 X2)
    - X2: 疾病B严重程度 (独立于 X1)
    - 住院概率取决于 X1 + X2

    在住院人群中，X1 和 X2 会呈现负相关!

    Returns:
    --------
    figure: 可视化图表
    stats: 统计信息
    """
    np.random.seed(42)

    # 两个独立的疾病严重程度
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)

    # 选择 (住院) 机制
    selection_score = selection_strength * (X1 + X2)
    selection_prob = 1 / (1 + np.exp(-selection_score + 2))
    S = np.random.binomial(1, selection_prob).astype(bool)

    # 计算相关系数
    full_corr = np.corrcoef(X1, X2)[0, 1]
    selected_corr = np.corrcoef(X1[S], X2[S])[0, 1]

    # 可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'总体数据 (相关系数: {full_corr:.3f})',
            f'住院人群 (相关系数: {selected_corr:.3f})'
        )
    )

    # 1. 总体数据
    fig.add_trace(go.Scatter(
        x=X1, y=X2,
        mode='markers',
        marker=dict(
            color=S.astype(int),
            colorscale=[[0, 'lightgray'], [1, '#EB5757']],
            size=4,
            opacity=0.5
        ),
        name='总体',
        showlegend=False
    ), row=1, col=1)

    # 添加回归线
    z_full = np.polyfit(X1, X2, 1)
    p_full = np.poly1d(z_full)
    x_line = np.linspace(X1.min(), X1.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p_full(x_line),
        mode='lines',
        line=dict(color='#2D9CDB', width=2),
        showlegend=False
    ), row=1, col=1)

    # 2. 选择后数据
    fig.add_trace(go.Scatter(
        x=X1[S], y=X2[S],
        mode='markers',
        marker=dict(color='#EB5757', size=5, opacity=0.5),
        showlegend=False
    ), row=1, col=2)

    # 添加选择后回归线
    z_sel = np.polyfit(X1[S], X2[S], 1)
    p_sel = np.poly1d(z_sel)
    fig.add_trace(go.Scatter(
        x=x_line, y=p_sel(x_line),
        mode='lines',
        line=dict(color='#27AE60', width=2),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="Berkson's Paradox: 住院偏差"
    )

    fig.update_xaxes(title_text="疾病 A 严重程度", row=1, col=1)
    fig.update_xaxes(title_text="疾病 A 严重程度", row=1, col=2)
    fig.update_yaxes(title_text="疾病 B 严重程度", row=1, col=1)
    fig.update_yaxes(title_text="疾病 B 严重程度", row=1, col=2)

    stats = {
        'full_correlation': float(full_corr),
        'selected_correlation': float(selected_corr),
        'sample_size_full': int(n_samples),
        'sample_size_selected': int(S.sum()),
        'selection_rate': float(S.mean())
    }

    return fig, stats
