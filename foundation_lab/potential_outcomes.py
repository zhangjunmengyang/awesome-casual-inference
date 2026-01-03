"""
潜在结果框架 (Potential Outcomes Framework) 可视化模块

核心概念:
- Rubin Causal Model (RCM)
- Y(0): 不接受处理时的潜在结果
- Y(1): 接受处理时的潜在结果
- ITE = Y(1) - Y(0): 个体处理效应
- ATE = E[Y(1) - Y(0)]: 平均处理效应
- 基本问题: 每个个体只能观测到一个潜在结果 (反事实问题)
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import generate_potential_outcomes_data


def create_potential_outcomes_visualization(
    n_samples: int,
    treatment_effect: float,
    noise_std: float,
    show_counterfactual: bool
) -> tuple:
    """生成潜在结果可视化"""

    # 生成数据
    df = generate_potential_outcomes_data(
        n_samples=n_samples,
        treatment_effect=treatment_effect,
        noise_std=noise_std,
        confounding_strength=0.0
    )

    # 计算统计量
    true_ate = df['ITE'].mean()
    naive_ate = df[df['T'] == 1]['Y'].mean() - df[df['T'] == 0]['Y'].mean()

    # 创建主图
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
        marker=dict(color='blue', opacity=0.5, size=5),
        legendgroup='potential'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['X'], y=df['Y1'],
        mode='markers',
        name='Y(1)',
        marker=dict(color='red', opacity=0.5, size=5),
        legendgroup='potential'
    ), row=1, col=1)

    # 2. ITE 分布
    fig.add_trace(go.Histogram(
        x=df['ITE'],
        nbinsx=30,
        name='ITE',
        marker_color='green',
        opacity=0.7,
        showlegend=False
    ), row=1, col=2)

    fig.add_vline(
        x=true_ate, row=1, col=2,
        line_dash="dash", line_color="red",
        annotation_text=f"ATE={true_ate:.2f}"
    )

    # 3. 观测数据
    control = df[df['T'] == 0]
    treated = df[df['T'] == 1]

    fig.add_trace(go.Scatter(
        x=control['X'], y=control['Y'],
        mode='markers',
        name='控制组 (T=0)',
        marker=dict(color='blue', opacity=0.6, size=5),
        legendgroup='observed'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=treated['X'], y=treated['Y'],
        mode='markers',
        name='处理组 (T=1)',
        marker=dict(color='red', opacity=0.6, size=5),
        legendgroup='observed'
    ), row=2, col=2)

    # 4. 反事实可视化 (可选)
    if show_counterfactual:
        # 显示观测到的和未观测到的
        # 处理组: 观测到 Y(1), 未观测到 Y(0)
        fig.add_trace(go.Scatter(
            x=treated['X'], y=treated['Y1'],
            mode='markers',
            name='观测: Y(1)',
            marker=dict(color='red', opacity=0.8, size=6, symbol='circle'),
            legendgroup='cf'
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=treated['X'], y=treated['Y0'],
            mode='markers',
            name='反事实: Y(0)',
            marker=dict(color='red', opacity=0.3, size=6, symbol='x'),
            legendgroup='cf'
        ), row=2, col=2)

        # 控制组: 观测到 Y(0), 未观测到 Y(1)
        fig.add_trace(go.Scatter(
            x=control['X'], y=control['Y0'],
            mode='markers',
            name='观测: Y(0)',
            marker=dict(color='blue', opacity=0.8, size=6, symbol='circle'),
            legendgroup='cf'
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=control['X'], y=control['Y1'],
            mode='markers',
            name='反事实: Y(1)',
            marker=dict(color='blue', opacity=0.3, size=6, symbol='x'),
            legendgroup='cf'
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='潜在结果框架 (Potential Outcomes Framework)',
        showlegend=True
    )

    # 统计信息
    stats_md = f"""
### 统计量

| 指标 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 真实 ATE | {true_ate:.4f} |
| 朴素估计 | {naive_ate:.4f} |
| 估计偏差 | {naive_ate - true_ate:.4f} |
| ITE 标准差 | {df['ITE'].std():.4f} |

### 关键洞察

- **真实 ATE**: 在无混淆的随机实验中，朴素估计 ≈ 真实 ATE
- **ITE 异质性**: 标准差反映个体间处理效应的变异
- **反事实问题**: 每个个体只能观测到一个潜在结果
"""

    return fig, stats_md


def create_fundamental_problem_demo() -> go.Figure:
    """演示因果推断的基本问题"""

    # 创建一个简单的个体示例
    individuals = ['Alice', 'Bob', 'Carol', 'Dave']
    y0 = [5, 3, 7, 4]
    y1 = [8, 4, 9, 6]
    treatment = [1, 0, 1, 0]

    fig = go.Figure()

    # Y(0)
    fig.add_trace(go.Bar(
        name='Y(0) - 不处理',
        x=individuals,
        y=y0,
        marker_color=['rgba(0,0,255,0.3)' if t == 1 else 'blue' for t in treatment],
        text=[f'{v} (反事实)' if t == 1 else f'{v} (观测)' for v, t in zip(y0, treatment)],
        textposition='outside'
    ))

    # Y(1)
    fig.add_trace(go.Bar(
        name='Y(1) - 处理',
        x=individuals,
        y=y1,
        marker_color=['red' if t == 1 else 'rgba(255,0,0,0.3)' for t in treatment],
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
                text='淡色 = 反事实 (不可观测), 深色 = 实际观测值',
                showarrow=False,
                font=dict(size=12)
            )
        ],
        height=500
    )

    return fig


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 潜在结果框架 (Potential Outcomes Framework)

**Rubin Causal Model (RCM)** 是因果推断的基础框架，由 Donald Rubin 提出。

### 核心概念

- **Y(0)**: 不接受处理时的潜在结果
- **Y(1)**: 接受处理时的潜在结果
- **ITE = Y(1) - Y(0)**: 个体处理效应 (Individual Treatment Effect)
- **ATE = E[Y(1) - Y(0)]**: 平均处理效应 (Average Treatment Effect)

### 基本问题 (Fundamental Problem of Causal Inference)

每个个体在同一时刻只能处于一种状态 (处理或不处理)，因此我们永远无法同时观测到同一个体的 Y(0) 和 Y(1)。

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=100, maximum=2000, value=500, step=100,
                    label="样本量"
                )
                treatment_effect = gr.Slider(
                    minimum=-5, maximum=5, value=2.0, step=0.5,
                    label="真实处理效应 (ATE)"
                )
                noise_std = gr.Slider(
                    minimum=0.1, maximum=3, value=1.0, step=0.1,
                    label="噪声标准差"
                )
                show_cf = gr.Checkbox(
                    value=False,
                    label="显示反事实 (通常不可观测)"
                )
                run_btn = gr.Button("生成可视化", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="潜在结果可视化")

        with gr.Row():
            stats_output = gr.Markdown()

        # 基本问题演示
        gr.Markdown("---")
        gr.Markdown("### 因果推断的基本问题演示")

        with gr.Row():
            demo_plot = gr.Plot(
                value=create_fundamental_problem_demo(),
                label="反事实不可观测"
            )

        gr.Markdown("""
### 思考题

1. 如果我们能观测到所有潜在结果，因果推断会有什么不同？
2. 为什么随机实验可以识别平均处理效应？
3. ITE 的异质性对营销策略有什么启示？

### 练习

完成 `exercises/chapter1_foundation/ex1_potential_outcomes.py` 中的练习。
        """)

        run_btn.click(
            fn=create_potential_outcomes_visualization,
            inputs=[n_samples, treatment_effect, noise_std, show_cf],
            outputs=[plot_output, stats_output]
        )

    return {'load_fn': lambda: create_potential_outcomes_visualization(500, 2.0, 1.0, False),
            'load_outputs': [plot_output, stats_output]}
