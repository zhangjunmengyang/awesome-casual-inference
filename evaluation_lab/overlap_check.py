"""
重叠假设检验模块

检查处理组和对照组在倾向得分分布上的重叠
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    generate_observational_data,
    estimate_propensity_score,
    calculate_overlap_statistics,
    suggest_trimming
)


def perform_overlap_check(
    n_samples: int,
    treatment_assignment: str,
    show_trimming: bool
) -> tuple:
    """
    执行重叠假设检验

    Parameters:
    -----------
    n_samples: 样本量
    treatment_assignment: 处理分配机制
    show_trimming: 是否显示修剪建议

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

    # 估计倾向得分
    ps_estimated = estimate_propensity_score(X, T)

    # 真实倾向得分（如果可用）
    ps_true = df['propensity'].values if 'propensity' in df.columns else None

    # 分离处理组和对照组的倾向得分
    ps_treatment = ps_estimated[T == 1]
    ps_control = ps_estimated[T == 0]

    # 计算重叠统计量
    overlap_stats = calculate_overlap_statistics(ps_treatment, ps_control)

    # 修剪建议
    if show_trimming:
        keep_mask, trimming_info = suggest_trimming(ps_estimated, T)
        ps_trimmed = ps_estimated[keep_mask]
        T_trimmed = T[keep_mask]
        ps_treatment_trimmed = ps_trimmed[T_trimmed == 1]
        ps_control_trimmed = ps_trimmed[T_trimmed == 0]
    else:
        trimming_info = None

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Propensity Score Distribution',
            'Overlap Region',
            'Propensity Score by Treatment',
            'Common Support'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "box"}, {"type": "scatter"}]
        ]
    )

    # 1. Propensity Score Distribution (Histogram)
    fig.add_trace(go.Histogram(
        x=ps_treatment,
        name='Treatment',
        marker_color='#2D9CDB',
        opacity=0.7,
        nbinsx=40
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=ps_control,
        name='Control',
        marker_color='#9B59B6',
        opacity=0.7,
        nbinsx=40
    ), row=1, col=1)

    # 2. Overlap Region (Density Plot)
    # 使用核密度估计
    ps_range = np.linspace(0, 1, 200)

    # 简单的核密度估计（使用高斯核）
    def kde(data, x_range, bandwidth=0.05):
        density = np.zeros_like(x_range)
        for x in data:
            density += np.exp(-0.5 * ((x_range - x) / bandwidth) ** 2)
        density /= (len(data) * bandwidth * np.sqrt(2 * np.pi))
        return density

    density_treatment = kde(ps_treatment, ps_range)
    density_control = kde(ps_control, ps_range)

    fig.add_trace(go.Scatter(
        x=ps_range,
        y=density_treatment,
        mode='lines',
        name='Treatment Density',
        line=dict(color='#2D9CDB', width=2),
        fill='tozeroy',
        opacity=0.5
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=ps_range,
        y=density_control,
        mode='lines',
        name='Control Density',
        line=dict(color='#9B59B6', width=2),
        fill='tozeroy',
        opacity=0.5
    ), row=1, col=2)

    # 标记重叠区域
    overlap_min = overlap_stats['overlap_min']
    overlap_max = overlap_stats['overlap_max']

    fig.add_vrect(
        x0=overlap_min, x1=overlap_max,
        fillcolor='#27AE60', opacity=0.2,
        layer='below', line_width=0,
        row=1, col=2
    )

    # 3. Propensity Score by Treatment (Box Plot)
    fig.add_trace(go.Box(
        y=ps_treatment,
        name='Treatment',
        marker_color='#2D9CDB',
        boxmean='sd'
    ), row=2, col=1)

    fig.add_trace(go.Box(
        y=ps_control,
        name='Control',
        marker_color='#9B59B6',
        boxmean='sd'
    ), row=2, col=1)

    # 添加阈值线
    fig.add_hline(y=0.1, line_dash="dash", line_color="red", row=2, col=1,
                  annotation_text="Lower Threshold (0.1)")
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", row=2, col=1,
                  annotation_text="Upper Threshold (0.9)")

    # 4. Common Support (Scatter Plot)
    # 显示每个样本的倾向得分
    sample_size_plot = min(500, n_samples)  # 限制点数
    idx_sample = np.random.choice(n_samples, sample_size_plot, replace=False)

    ps_sample = ps_estimated[idx_sample]
    T_sample = T[idx_sample]

    # 添加 jitter 以便更好地可视化
    y_jitter = np.random.randn(sample_size_plot) * 0.02

    colors = ['#9B59B6' if t == 0 else '#2D9CDB' for t in T_sample]

    fig.add_trace(go.Scatter(
        x=ps_sample,
        y=T_sample + y_jitter,
        mode='markers',
        marker=dict(
            color=colors,
            size=5,
            opacity=0.6
        ),
        name='Samples',
        showlegend=False
    ), row=2, col=2)

    # 标记 common support 区域
    fig.add_vrect(
        x0=overlap_min, x1=overlap_max,
        fillcolor='#27AE60', opacity=0.2,
        layer='below', line_width=0,
        row=2, col=2,
        annotation_text="Common Support",
        annotation_position="top left"
    )

    # 修剪阈值线
    if show_trimming:
        fig.add_vline(x=0.1, line_dash="dash", line_color="red", row=2, col=2)
        fig.add_vline(x=0.9, line_dash="dash", line_color="red", row=2, col=2)

    # 更新布局
    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Overlap Assumption Check',
        showlegend=True,
        barmode='overlay'
    )

    fig.update_xaxes(title_text='Propensity Score', row=1, col=1)
    fig.update_xaxes(title_text='Propensity Score', row=1, col=2)
    fig.update_xaxes(title_text='Propensity Score', row=2, col=2)

    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Density', row=1, col=2)
    fig.update_yaxes(title_text='Propensity Score', row=2, col=1)
    fig.update_yaxes(title_text='Treatment (0=Control, 1=Treatment)', row=2, col=2)

    # 统计摘要
    n_treatment = (T == 1).sum()
    n_control = (T == 0).sum()

    summary = f"""
### 重叠假设检验结果

#### 样本信息
| 指标 | 值 |
|------|-----|
| 总样本量 | {n_samples} |
| 处理组样本量 | {n_treatment} ({n_treatment/n_samples*100:.1f}%) |
| 对照组样本量 | {n_control} ({n_control/n_samples*100:.1f}%) |

#### 倾向得分范围
| 组别 | 最小值 | 最大值 | 均值 | 标准差 |
|------|--------|--------|------|--------|
| 处理组 | {overlap_stats['treatment_min']:.4f} | {overlap_stats['treatment_max']:.4f} | {ps_treatment.mean():.4f} | {ps_treatment.std():.4f} |
| 对照组 | {overlap_stats['control_min']:.4f} | {overlap_stats['control_max']:.4f} | {ps_control.mean():.4f} | {ps_control.std():.4f} |

#### 重叠区域
| 指标 | 值 |
|------|-----|
| 重叠区间 | [{overlap_stats['overlap_min']:.4f}, {overlap_stats['overlap_max']:.4f}] |
| 重叠范围 | {overlap_stats['overlap_range']:.4f} |
| 在重叠区域内的样本比例 | {overlap_stats['overlap_fraction']*100:.1f}% |
| 极端值样本比例 (< 0.1 或 > 0.9) | {overlap_stats['extreme_fraction']*100:.1f}% |

#### 重叠质量评估
"""

    # 判断重叠质量
    if overlap_stats['overlap_fraction'] > 0.9:
        quality = "优秀"
        color = "green"
    elif overlap_stats['overlap_fraction'] > 0.75:
        quality = "良好"
        color = "blue"
    elif overlap_stats['overlap_fraction'] > 0.5:
        quality = "一般"
        color = "orange"
    else:
        quality = "较差"
        color = "red"

    summary += f"**重叠质量**: {quality}\n\n"

    # 正性假设检验
    if overlap_stats['extreme_fraction'] > 0.1:
        summary += f"""
**警告**: {overlap_stats['extreme_fraction']*100:.1f}% 的样本倾向得分极端 (< 0.1 或 > 0.9)。
这违反了**正性假设** (Positivity Assumption)，可能导致估计不稳定。

**建议**: 考虑修剪 (Trimming) 这些样本。
"""

    # 修剪信息
    if trimming_info is not None:
        summary += f"""
---

### 修剪 (Trimming) 建议

如果移除倾向得分 < 0.1 或 > 0.9 的样本:

| 指标 | 值 |
|------|-----|
| 移除样本数 | {trimming_info['n_trimmed']} ({trimming_info['trimmed_fraction']*100:.1f}%) |
| 移除的处理组样本 | {trimming_info['n_trimmed_treatment']} |
| 移除的对照组样本 | {trimming_info['n_trimmed_control']} |
| 保留样本数 | {trimming_info['n_total'] - trimming_info['n_trimmed']} |

**权衡**: 修剪改善了估计的稳定性，但减少了样本量和外推性。
"""

    summary += f"""
---

### 正性假设 (Positivity Assumption)

对于所有协变量值 $X$，必须满足:

$$0 < P(T=1|X) < 1$$

即：每个个体都有非零概率被分配到处理组或对照组。

### 为什么重叠很重要?

- **缺乏重叠**: 某些协变量模式下只有处理组或只有对照组
- **后果**: 无法进行有效的因果推断
- **解决方案**:
  1. 修剪 (Trimming): 移除倾向得分极端的样本
  2. 重新定义研究人群
  3. 使用更灵活的模型

### 重叠区域图解读

- **绿色区域**: Common Support (公共支持域)
- **目标**: 两组的倾向得分分布应有充分重叠
- **红色虚线**: 建议的修剪阈值 (0.1 和 0.9)

### 实践建议

1. **重叠良好** (> 90%): 可以安全进行因果推断
2. **重叠一般** (50% - 90%): 考虑修剪或加权
3. **重叠较差** (< 50%): 重新考虑研究设计或数据收集

### 练习

思考以下问题:
1. 为什么倾向得分接近 0 或 1 会导致问题?
2. 修剪样本后，因果效应的目标人群发生了什么变化?
3. 除了修剪，还有什么方法可以处理重叠不足的问题?
"""

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 重叠假设检验

检查处理组和对照组在倾向得分上的重叠，验证正性假设 (Positivity Assumption)。

### 核心概念

| 概念 | 含义 |
|------|------|
| **Positivity** | 每个个体都有非零概率被分配到任一组 |
| **Common Support** | 处理组和对照组倾向得分的重叠区域 |
| **Trimming** | 移除倾向得分极端的样本 |

正性假设是因果推断的三大核心假设之一（与无混淆性、一致性并列）。

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
                    value='severe_confounding',
                    label="处理分配机制",
                    info="severe_confounding 会导致重叠不足"
                )
                show_trimming = gr.Checkbox(
                    value=True,
                    label="显示修剪建议",
                    info="显示移除极端倾向得分后的效果"
                )
                run_btn = gr.Button("运行重叠检查", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="重叠假设检验")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=perform_overlap_check,
            inputs=[n_samples, treatment_assignment, show_trimming],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 倾向得分公式

$$e(X) = P(T=1|X)$$

即给定协变量 $X$ 的条件下，接受处理的概率。

### 重叠质量评分

- **优秀** (> 90%): 几乎所有样本都在公共支持域内
- **良好** (75% - 90%): 大部分样本在公共支持域内
- **一般** (50% - 75%): 需要考虑修剪
- **较差** (< 50%): 重叠严重不足，需要重新设计研究

### 修剪 (Trimming) 的权衡

**优点**:
- 提高估计的稳定性
- 减少外推误差
- 改善匹配质量

**缺点**:
- 减少样本量
- 改变目标人群（外部效度下降）
- 可能引入选择偏差

### 相关方法

- **Crump 规则**: 修剪倾向得分 < 0.1 或 > 0.9
- **对称修剪**: 基于两组倾向得分的共同范围
- **数据驱动修剪**: 根据估计方差优化修剪阈值
        """)

    return None
