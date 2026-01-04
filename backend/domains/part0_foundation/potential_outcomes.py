"""
潜在结果框架 (Potential Outcomes Framework)

核心概念:
- Rubin Causal Model (RCM)
- Y(0): 不接受处理时的潜在结果
- Y(1): 接受处理时的潜在结果
- ITE = Y(1) - Y(0): 个体处理效应
- ATE = E[Y(1) - Y(0)]: 平均处理效应
- 基本问题: 每个个体只能观测到一个潜在结果 (反事实问题)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple

from .utils import generate_potential_outcomes_data, fig_to_dict


def visualize_potential_outcomes(
    n_samples: int = 500,
    treatment_effect: float = 2.0,
    noise_std: float = 1.0,
    confounding_strength: float = 0.0
) -> Tuple[go.Figure, dict]:
    """
    可视化潜在结果框架

    Returns:
    --------
    figure: Plotly 图表
    stats: 统计信息字典
    """
    # 生成数据
    df = generate_potential_outcomes_data(
        n_samples=n_samples,
        treatment_effect=treatment_effect,
        noise_std=noise_std,
        confounding_strength=confounding_strength
    )

    # 计算统计量
    true_ate = df['ITE'].mean()
    naive_ate = df[df['T'] == 1]['Y'].mean() - df[df['T'] == 0]['Y'].mean()
    ite_std = df['ITE'].std()

    # 创建多子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '潜在结果: Y(0) vs Y(1)',
            '个体处理效应 (ITE) 分布',
            '观测数据 (我们实际看到的)',
            '反事实: 观测 vs 未观测'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. 潜在结果散点图
    fig.add_trace(go.Scatter(
        x=df['X'], y=df['Y0'],
        mode='markers',
        name='Y(0)',
        marker=dict(color='#2D9CDB', opacity=0.5, size=5),
        legendgroup='potential'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['X'], y=df['Y1'],
        mode='markers',
        name='Y(1)',
        marker=dict(color='#EB5757', opacity=0.5, size=5),
        legendgroup='potential'
    ), row=1, col=1)

    # 2. ITE 分布
    fig.add_trace(go.Histogram(
        x=df['ITE'],
        nbinsx=30,
        name='ITE',
        marker_color='#27AE60',
        opacity=0.7,
        showlegend=False
    ), row=1, col=2)

    # ATE 参考线
    fig.add_vline(
        x=true_ate,
        line_dash="dash",
        line_color="red",
        row=1, col=2,
        annotation_text=f"ATE={true_ate:.2f}"
    )

    # 3. 观测数据
    control = df[df['T'] == 0]
    treated = df[df['T'] == 1]

    fig.add_trace(go.Scatter(
        x=control['X'], y=control['Y'],
        mode='markers',
        name='控制组 (T=0)',
        marker=dict(color='#2D9CDB', opacity=0.6, size=5),
        legendgroup='observed'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=treated['X'], y=treated['Y'],
        mode='markers',
        name='处理组 (T=1)',
        marker=dict(color='#EB5757', opacity=0.6, size=5),
        legendgroup='observed'
    ), row=2, col=1)

    # 4. 反事实可视化
    fig.add_trace(go.Scatter(
        x=treated['X'], y=treated['Y1'],
        mode='markers',
        name='观测: Y(1)',
        marker=dict(color='#EB5757', opacity=0.8, size=6, symbol='circle'),
        legendgroup='cf'
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=treated['X'], y=treated['Y0'],
        mode='markers',
        name='反事实: Y(0)',
        marker=dict(color='#EB5757', opacity=0.3, size=6, symbol='x'),
        legendgroup='cf'
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=control['X'], y=control['Y0'],
        mode='markers',
        name='观测: Y(0)',
        marker=dict(color='#2D9CDB', opacity=0.8, size=6, symbol='circle'),
        legendgroup='cf',
        showlegend=False
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=control['X'], y=control['Y1'],
        mode='markers',
        name='反事实: Y(1)',
        marker=dict(color='#2D9CDB', opacity=0.3, size=6, symbol='x'),
        legendgroup='cf',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='潜在结果框架 (Potential Outcomes Framework)',
        showlegend=True
    )

    stats = {
        'n_samples': n_samples,
        'true_ate': float(true_ate),
        'naive_ate': float(naive_ate),
        'bias': float(naive_ate - true_ate),
        'ite_std': float(ite_std)
    }

    return fig, stats


def demonstrate_fundamental_problem() -> go.Figure:
    """演示因果推断的基本问题: 反事实不可观测"""

    # 创建简单示例数据
    individuals = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
    y0 = [5, 3, 7, 4, 6]
    y1 = [8, 4, 9, 6, 8]
    treatment = [1, 0, 1, 0, 1]

    fig = go.Figure()

    # Y(0) - 不处理的结果
    fig.add_trace(go.Bar(
        name='Y(0) - 不处理',
        x=individuals,
        y=y0,
        marker_color=['rgba(45,156,219,0.3)' if t == 1 else '#2D9CDB' for t in treatment],
        text=[f'{v} (反事实)' if t == 1 else f'{v} (观测)' for v, t in zip(y0, treatment)],
        textposition='outside'
    ))

    # Y(1) - 处理的结果
    fig.add_trace(go.Bar(
        name='Y(1) - 处理',
        x=individuals,
        y=y1,
        marker_color=['#EB5757' if t == 1 else 'rgba(235,87,87,0.3)' for t in treatment],
        text=[f'{v} (观测)' if t == 1 else f'{v} (反事实)' for v, t in zip(y1, treatment)],
        textposition='outside'
    ))

    fig.update_layout(
        title='因果推断的基本问题: 反事实不可观测',
        xaxis_title='个体',
        yaxis_title='结果 Y',
        barmode='group',
        template='plotly_white',
        annotations=[
            dict(
                x=0.5, y=-0.15,
                xref='paper', yref='paper',
                text='深色 = 实际观测, 浅色 = 反事实 (不可观测)',
                showarrow=False,
                font=dict(size=12)
            )
        ],
        height=500
    )

    return fig
