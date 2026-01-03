"""
Uplift 评估模块

Qini 曲线、Uplift 曲线等评估工具
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    generate_marketing_uplift_data,
    calculate_qini_curve,
    calculate_uplift_curve,
    calculate_auuc
)


def demonstrate_uplift_evaluation(
    n_samples: int,
    model_quality: str
) -> tuple:
    """演示 Uplift 评估方法"""

    # 生成数据
    df, true_uplift = generate_marketing_uplift_data(n_samples)

    Y = df['Y'].values
    T = df['T'].values

    # 生成不同质量的预测
    noise_levels = {
        'perfect': 0.0,
        'good': 0.3,
        'medium': 0.6,
        'poor': 1.0
    }

    noise = noise_levels.get(model_quality, 0.5)
    predicted_uplift = true_uplift + np.random.randn(n_samples) * noise * true_uplift.std()

    # 计算曲线
    fraction_q, qini = calculate_qini_curve(Y, T, predicted_uplift)
    fraction_u, uplift_curve = calculate_uplift_curve(Y, T, predicted_uplift)

    # 随机基线
    random_pred = np.random.randn(n_samples)
    fraction_r, qini_random = calculate_qini_curve(Y, T, random_pred)

    # AUUC
    auuc_model = calculate_auuc(Y, T, predicted_uplift)
    auuc_random = calculate_auuc(Y, T, random_pred)
    auuc_perfect = calculate_auuc(Y, T, true_uplift)

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Qini Curve',
            'Uplift Curve',
            '按 Uplift 分组的转化率',
            '最优干预比例分析'
        )
    )

    # 1. Qini Curve
    fig.add_trace(go.Scatter(
        x=fraction_q, y=qini,
        mode='lines', name=f'Model (AUUC={auuc_model:.4f})',
        line=dict(color='#2D9CDB', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fraction_r, y=qini_random,
        mode='lines', name=f'Random (AUUC={auuc_random:.4f})',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)

    # 2. Uplift Curve
    fig.add_trace(go.Scatter(
        x=fraction_u, y=uplift_curve,
        mode='lines', name='Cumulative Uplift',
        line=dict(color='#27AE60', width=2)
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # 3. 分组转化率
    # 按预测 uplift 分成 10 组
    deciles = pd.qcut(predicted_uplift, 10, labels=False, duplicates='drop')
    group_stats = []

    for d in range(10):
        mask = deciles == d
        if mask.sum() > 0:
            y_sub = Y[mask]
            t_sub = T[mask]
            if (t_sub == 1).sum() > 0 and (t_sub == 0).sum() > 0:
                conv_t = y_sub[t_sub == 1].mean()
                conv_c = y_sub[t_sub == 0].mean()
                uplift = conv_t - conv_c
                group_stats.append({
                    'decile': d + 1,
                    'conv_treatment': conv_t,
                    'conv_control': conv_c,
                    'uplift': uplift
                })

    # 初始化 stats_df 以避免作用域问题
    stats_df = pd.DataFrame(group_stats) if group_stats else pd.DataFrame(columns=['decile', 'conv_treatment', 'conv_control', 'uplift'])

    if len(stats_df) > 0:

        fig.add_trace(go.Bar(
            x=stats_df['decile'], y=stats_df['conv_treatment'],
            name='Treatment', marker_color='#EB5757', opacity=0.7
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=stats_df['decile'], y=stats_df['conv_control'],
            name='Control', marker_color='#2D9CDB', opacity=0.7
        ), row=2, col=1)

    # 4. 最优干预比例
    # 计算累积 uplift 和成本收益
    sorted_idx = np.argsort(predicted_uplift)[::-1]
    cumulative_benefit = np.cumsum(true_uplift[sorted_idx])

    # 假设每次干预成本为 1，每次转化收益为 10
    cost_per_treatment = 1
    revenue_per_conversion = 10

    n_points = 100
    fractions = np.linspace(0, 1, n_points)
    roi_values = []

    for frac in fractions:
        n_treat = int(frac * n_samples)
        if n_treat > 0:
            top_idx = sorted_idx[:n_treat]
            expected_conversions = true_uplift[top_idx].sum()
            revenue = expected_conversions * revenue_per_conversion
            cost = n_treat * cost_per_treatment
            roi = (revenue - cost) / cost if cost > 0 else 0
            roi_values.append(roi)
        else:
            roi_values.append(0)

    fig.add_trace(go.Scatter(
        x=fractions * 100, y=roi_values,
        mode='lines', name='ROI',
        line=dict(color='#9B59B6', width=2)
    ), row=2, col=2)

    # 找最优点
    best_idx = np.argmax(roi_values)
    best_fraction = fractions[best_idx]
    best_roi = roi_values[best_idx]

    fig.add_trace(go.Scatter(
        x=[best_fraction * 100], y=[best_roi],
        mode='markers', name=f'Optimal ({best_fraction*100:.1f}%)',
        marker=dict(color='red', size=12, symbol='star')
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='Uplift 评估与优化',
        barmode='group'
    )

    fig.update_xaxes(title_text='Fraction Targeted', row=1, col=1)
    fig.update_xaxes(title_text='Fraction Targeted', row=1, col=2)
    fig.update_xaxes(title_text='Decile (1=highest uplift)', row=2, col=1)
    fig.update_xaxes(title_text='% Targeted', row=2, col=2)

    fig.update_yaxes(title_text='Qini Value', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Uplift', row=1, col=2)
    fig.update_yaxes(title_text='Conversion Rate', row=2, col=1)
    fig.update_yaxes(title_text='ROI', row=2, col=2)

    # 摘要
    overall_uplift = Y[T == 1].mean() - Y[T == 0].mean()
    # 安全地获取 top 10% uplift
    top_10_data = stats_df[stats_df['decile'] == 1]['uplift'].values if len(stats_df) > 0 else []
    top_10_uplift = top_10_data[0] if len(top_10_data) > 0 else 0

    summary = f"""
### 评估结果

| 指标 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 整体 Uplift | {overall_uplift:.4f} |
| Top 10% Uplift | {top_10_uplift:.4f} |
| Model AUUC | {auuc_model:.4f} |
| Random AUUC | {auuc_random:.4f} |
| AUUC 提升 | {(auuc_model - auuc_random) / abs(auuc_random) * 100:.1f}% |
| 最优干预比例 | {best_fraction * 100:.1f}% |
| 最优 ROI | {best_roi:.2f} |

### Qini 曲线解读

- **高于随机线**: 模型有效
- **面积 (AUUC)**: 越大越好
- **曲线形状**: 陡峭上升表示高 uplift 用户集中

### 实践建议

1. **不要干预所有人**: 最优比例约 {best_fraction * 100:.0f}%
2. **关注负 uplift**: 部分用户干预后反而更差
3. **分层策略**: 高 uplift 用户重点干预
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## Uplift 评估方法

评估 Uplift 模型的效果，指导营销决策。

### 核心评估指标

| 指标 | 含义 | 用途 |
|------|------|------|
| **Qini Curve** | 累积增益曲线 | 评估排序能力 |
| **AUUC** | Qini 曲线下面积 | 模型整体效果 |
| **Uplift by Decile** | 分组 Uplift | 理解用户分层 |
| **Optimal Targeting** | 最优干预比例 | 指导策略 |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=2000, maximum=20000, value=10000, step=1000,
                    label="样本量"
                )
                model_quality = gr.Radio(
                    choices=['perfect', 'good', 'medium', 'poor'],
                    value='good',
                    label="模型质量"
                )
                run_btn = gr.Button("运行评估", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="Uplift 评估")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=demonstrate_uplift_evaluation,
            inputs=[n_samples, model_quality],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### Qini 曲线公式

对于按预测 uplift 排序后的前 k 个样本:

$$Qini(k) = \\sum_{i=1}^{k} Y_i \\cdot T_i - \\sum_{i=1}^{k} Y_i \\cdot (1-T_i) \\cdot \\frac{n_{T,k}}{n_{C,k}}$$

其中:
- $Y_i$: 第 i 个样本的结果
- $T_i$: 第 i 个样本的处理状态
- $n_{T,k}$, $n_{C,k}$: 前 k 个样本中处理/控制组的数量

### 实践练习

计算不同模型的 AUUC 并进行对比分析。
        """)

    return None
