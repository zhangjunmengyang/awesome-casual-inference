"""
协变量平衡检查模块

检查处理组和对照组之间的协变量平衡性
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    generate_observational_data,
    calculate_balance_metrics,
    estimate_propensity_score,
    perform_propensity_score_matching
)


def perform_balance_check(
    n_samples: int,
    treatment_assignment: str,
    apply_matching: bool
) -> tuple:
    """
    执行协变量平衡检查

    Parameters:
    -----------
    n_samples: 样本量
    treatment_assignment: 处理分配机制
    apply_matching: 是否应用倾向得分匹配

    Returns:
    --------
    (figure, summary_markdown)
    """
    # 生成数据
    df, true_tau = generate_observational_data(
        n_samples=n_samples,
        n_features=5,
        treatment_assignment=treatment_assignment
    )

    # 提取特征和处理
    feature_cols = [col for col in df.columns if col.startswith('X')]
    X = df[feature_cols].values
    T = df['T'].values

    # 匹配前的平衡性
    balance_before = calculate_balance_metrics(X, T, feature_cols)

    # 如果应用匹配
    if apply_matching:
        # 估计倾向得分
        ps = estimate_propensity_score(X, T)

        # 执行匹配
        matched_t_idx, matched_c_idx = perform_propensity_score_matching(X, T, ps)

        # 匹配后的样本
        matched_idx = np.concatenate([matched_t_idx, matched_c_idx])
        X_matched = X[matched_idx]
        T_matched = T[matched_idx]

        # 匹配后的平衡性
        balance_after = calculate_balance_metrics(X_matched, T_matched, feature_cols)
    else:
        balance_after = None

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Love Plot - SMD',
            'Variance Ratio',
            'Mean Difference by Feature',
            'Distribution Example (X1)'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "box"}]
        ]
    )

    # 1. Love Plot (SMD)
    features = balance_before['feature'].values
    smd_before = balance_before['smd'].values

    fig.add_trace(go.Scatter(
        y=features,
        x=smd_before,
        mode='markers',
        name='Before Matching',
        marker=dict(color='#EB5757', size=10, symbol='circle')
    ), row=1, col=1)

    if balance_after is not None:
        smd_after = balance_after['smd'].values
        fig.add_trace(go.Scatter(
            y=features,
            x=smd_after,
            mode='markers',
            name='After Matching',
            marker=dict(color='#27AE60', size=10, symbol='diamond')
        ), row=1, col=1)

    # 添加阈值线 (SMD < 0.1 认为平衡良好)
    for threshold in [0.1, -0.1]:
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold,
            y0=0, y1=1,
            yref="y domain",
            line=dict(dash="dash", color="gray"),
            row=1, col=1
        )

    # 2. Variance Ratio
    var_ratio_before = balance_before['variance_ratio'].values

    fig.add_trace(go.Scatter(
        y=features,
        x=var_ratio_before,
        mode='markers',
        name='Before Matching',
        marker=dict(color='#EB5757', size=10, symbol='circle'),
        showlegend=False
    ), row=1, col=2)

    if balance_after is not None:
        var_ratio_after = balance_after['variance_ratio'].values
        fig.add_trace(go.Scatter(
            y=features,
            x=var_ratio_after,
            mode='markers',
            name='After Matching',
            marker=dict(color='#27AE60', size=10, symbol='diamond'),
            showlegend=False
        ), row=1, col=2)

    # 添加阈值线 (0.5 < VR < 2 认为平衡良好)
    for threshold, dash_style in [(0.5, "dash"), (2.0, "dash"), (1.0, "dot")]:
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold,
            y0=0, y1=1,
            yref="y2 domain",
            line=dict(dash=dash_style, color="gray"),
            row=1, col=2
        )

    # 3. Mean Difference by Feature
    mean_diff_before = balance_before['mean_t'].values - balance_before['mean_c'].values

    fig.add_trace(go.Bar(
        x=features,
        y=mean_diff_before,
        name='Before Matching',
        marker_color='#EB5757',
        opacity=0.7
    ), row=2, col=1)

    if balance_after is not None:
        mean_diff_after = balance_after['mean_t'].values - balance_after['mean_c'].values
        fig.add_trace(go.Bar(
            x=features,
            y=mean_diff_after,
            name='After Matching',
            marker_color='#27AE60',
            opacity=0.7
        ), row=2, col=1)

    # 4. Distribution Example (X1)
    X1_treatment = X[T == 1, 0]
    X1_control = X[T == 0, 0]

    fig.add_trace(go.Box(
        y=X1_treatment,
        name='Treatment',
        marker_color='#2D9CDB',
        boxmean='sd'
    ), row=2, col=2)

    fig.add_trace(go.Box(
        y=X1_control,
        name='Control',
        marker_color='#9B59B6',
        boxmean='sd'
    ), row=2, col=2)

    if balance_after is not None:
        X1_matched = X_matched[T_matched == 1, 0]
        fig.add_trace(go.Box(
            y=X1_matched,
            name='Treatment (Matched)',
            marker_color='#27AE60',
            boxmean='sd'
        ), row=2, col=2)

    # 更新布局
    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Covariate Balance Check',
        showlegend=True
    )

    fig.update_xaxes(title_text='SMD', row=1, col=1)
    fig.update_xaxes(title_text='Variance Ratio', row=1, col=2)
    fig.update_xaxes(title_text='Feature', row=2, col=1)

    fig.update_yaxes(title_text='Feature', row=1, col=1)
    fig.update_yaxes(title_text='Feature', row=1, col=2)
    fig.update_yaxes(title_text='Mean Difference', row=2, col=1)
    fig.update_yaxes(title_text='Value', row=2, col=2)

    # 统计摘要
    n_treatment = (T == 1).sum()
    n_control = (T == 0).sum()

    # 判断平衡性
    smd_threshold = 0.1
    var_ratio_low = 0.5
    var_ratio_high = 2.0

    n_balanced_smd_before = (np.abs(smd_before) < smd_threshold).sum()
    n_balanced_var_before = ((var_ratio_before > var_ratio_low) & (var_ratio_before < var_ratio_high)).sum()

    summary = f"""
### 平衡性检查结果

#### 样本信息
| 指标 | 值 |
|------|-----|
| 总样本量 | {n_samples} |
| 处理组样本量 | {n_treatment} ({n_treatment/n_samples*100:.1f}%) |
| 对照组样本量 | {n_control} ({n_control/n_samples*100:.1f}%) |

#### 匹配前平衡性
| 指标 | 值 |
|------|-----|
| SMD < 0.1 的特征数 | {n_balanced_smd_before} / {len(features)} |
| 平均 SMD | {np.abs(smd_before).mean():.4f} |
| 最大 SMD | {np.abs(smd_before).max():.4f} |
| 平衡良好的特征数 (Variance Ratio) | {n_balanced_var_before} / {len(features)} |
"""

    if balance_after is not None:
        n_matched = len(matched_idx)
        smd_after_vals = balance_after['smd'].values
        var_ratio_after_vals = balance_after['variance_ratio'].values

        n_balanced_smd_after = (np.abs(smd_after_vals) < smd_threshold).sum()
        n_balanced_var_after = ((var_ratio_after_vals > var_ratio_low) & (var_ratio_after_vals < var_ratio_high)).sum()

        summary += f"""
#### 匹配后平衡性
| 指标 | 值 |
|------|-----|
| 匹配成功样本量 | {n_matched} ({n_matched/n_samples*100:.1f}%) |
| SMD < 0.1 的特征数 | {n_balanced_smd_after} / {len(features)} |
| 平均 SMD | {np.abs(smd_after_vals).mean():.4f} |
| 最大 SMD | {np.abs(smd_after_vals).max():.4f} |
| 平衡良好的特征数 (Variance Ratio) | {n_balanced_var_after} / {len(features)} |

**改进**: SMD 从 {np.abs(smd_before).mean():.4f} 降至 {np.abs(smd_after_vals).mean():.4f}
"""

    summary += f"""
---

### 平衡性评估标准

#### SMD (Standardized Mean Difference)
- **< 0.1**: 平衡良好
- **0.1 - 0.25**: 可接受
- **> 0.25**: 不平衡

#### Variance Ratio
- **0.5 - 2.0**: 方差平衡良好
- 偏离此范围: 方差不平衡

### Love Plot 解读

Love Plot 展示了匹配前后的 SMD:
- **横轴**: SMD 值
- **纵轴**: 特征
- **红色圆点**: 匹配前
- **绿色菱形**: 匹配后
- **目标**: 所有点都接近 0 (在灰色虚线内)

### 实践建议

1. **SMD > 0.25**: 考虑使用倾向得分匹配或加权
2. **方差比异常**: 检查数据质量，可能需要转换特征
3. **匹配后样本量减少**: 权衡平衡性与样本量
"""

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 协变量平衡检查

评估处理组和对照组之间的协变量平衡性，这是因果推断有效性的重要前提。

### 核心概念

| 指标 | 含义 | 阈值 |
|------|------|------|
| **SMD** | 标准化均值差 | < 0.1 为平衡良好 |
| **Variance Ratio** | 方差比 | 0.5 - 2.0 为平衡良好 |
| **Love Plot** | SMD 可视化 | 直观展示平衡性 |

在观测性数据中，处理组和对照组的协变量分布可能不同（混淆），导致偏差。
平衡检查帮助我们识别和解决这个问题。

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=500, maximum=5000, value=2000, step=500,
                    label="样本量"
                )
                treatment_assignment = gr.Radio(
                    choices=['random', 'confounded', 'severe_confounding'],
                    value='confounded',
                    label="处理分配机制",
                    info="random: 随机分配; confounded: 混淆; severe_confounding: 严重混淆"
                )
                apply_matching = gr.Checkbox(
                    value=False,
                    label="应用倾向得分匹配",
                    info="匹配后查看平衡性改善"
                )
                run_btn = gr.Button("运行平衡检查", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="平衡性检查")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=perform_balance_check,
            inputs=[n_samples, treatment_assignment, apply_matching],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### SMD 公式

$$SMD = \\frac{\\bar{X}_T - \\bar{X}_C}{\\sqrt{(s_T^2 + s_C^2) / 2}}$$

其中:
- $\\bar{X}_T$, $\\bar{X}_C$: 处理组和对照组的均值
- $s_T^2$, $s_C^2$: 处理组和对照组的方差

### Variance Ratio 公式

$$VR = \\frac{s_T^2}{s_C^2}$$

理想情况下，VR 应接近 1。

### 什么时候需要平衡检查?

- **观测性研究**: 必须检查
- **RCT (随机实验)**: 可选，但建议检查以验证随机化效果
- **倾向得分方法后**: 验证匹配/加权是否改善了平衡性

### 练习

思考以下问题:
1. 为什么 SMD < 0.1 被认为是平衡良好?
2. 如果匹配后样本量减少很多，如何权衡?
3. 除了匹配，还有什么方法可以改善平衡性?
        """)

    return None
