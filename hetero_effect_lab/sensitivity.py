"""
Sensitivity Analysis - 敏感性分析

评估因果推断结果对未观测混淆的敏感性。

核心概念:
- 无混淆假设 (Unconfoundedness) 通常无法验证
- 敏感性分析评估: 如果存在未观测混淆，结论会如何改变
- Rosenbaum Bounds: 量化对未观测混淆的鲁棒性
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple

from .utils import generate_heterogeneous_data


def compute_rosenbaum_bounds(
    Y: np.ndarray,
    T: np.ndarray,
    gamma_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Rosenbaum 敏感性边界

    Rosenbaum (2002) 提出的敏感性分析方法，用于评估
    未观测混淆对因果效应估计的影响。

    Parameters:
    -----------
    Y: 结果变量
    T: 处理变量
    gamma_values: 敏感性参数 Γ 的取值范围
        Γ = 1: 无未观测混淆
        Γ > 1: 允许一定程度的未观测混淆

    Returns:
    --------
    (lower_bounds, upper_bounds): ATE 的敏感性边界
    """
    n = len(Y)
    n_t = T.sum()
    n_c = n - n_t

    # 观测的 ATE
    ate_obs = Y[T == 1].mean() - Y[T == 0].mean()

    lower_bounds = []
    upper_bounds = []

    # 计算标准误（用于构建边界）
    var_t = Y[T == 1].var() / n_t if n_t > 0 else 0
    var_c = Y[T == 0].var() / n_c if n_c > 0 else 0
    se_ate = np.sqrt(var_t + var_c)

    for gamma in gamma_values:
        if gamma == 1:
            # 无偏情况
            lower_bounds.append(ate_obs)
            upper_bounds.append(ate_obs)
        else:
            # Rosenbaum bounds 的改进实现
            # 基于 Rosenbaum (2002) "Observational Studies" 的思想
            #
            # 当存在未观测混淆时，真实的倾向得分 p_true 与观测的 p_obs 之间满足:
            # 1/gamma <= p_true(1-p_obs) / (p_obs(1-p_true)) <= gamma
            #
            # 这里使用基于标准误的近似边界：
            # 边界宽度 = se * sqrt(2 * log(gamma)) * adjustment_factor
            # 其中 adjustment_factor 考虑样本量和 gamma 的非线性关系

            # 对数变换使边界增长更合理
            log_gamma = np.log(gamma)

            # 基于正态分位数的边界宽度
            # 当 gamma 增大时，边界扩展符合统计直觉
            z_adjustment = np.sqrt(2 * log_gamma) if log_gamma > 0 else 0

            # 边界宽度还与效应量和标准误有关
            bound_width = se_ate * z_adjustment + abs(ate_obs) * (gamma - 1) / (gamma + 1)

            lower_bounds.append(ate_obs - bound_width)
            upper_bounds.append(ate_obs + bound_width)

    return np.array(lower_bounds), np.array(upper_bounds)


def simulate_unobserved_confounding(
    n_samples: int,
    confounder_strength: float,
    correlation_with_x: float
) -> Tuple[pd.DataFrame, np.ndarray, float, float]:
    """
    模拟未观测混淆的影响

    生成数据时包含一个未观测的混淆因子 U，观察它如何影响效应估计。

    Parameters:
    -----------
    n_samples: 样本量
    confounder_strength: 未观测混淆的强度 (0-1)
    correlation_with_x: U 与观测特征 X 的相关性 (0-1)

    Returns:
    --------
    (df, true_ate, ate_biased, ate_true)
    """
    np.random.seed(42)

    # 观测特征
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)

    # 未观测混淆因子 U
    # U 与 X1 有一定相关性
    U = correlation_with_x * X1 + np.sqrt(1 - correlation_with_x**2) * np.random.randn(n_samples)

    # 倾向得分 (受 X 和 U 影响)
    propensity_logit = (
        0.3 * X1 +
        0.2 * X2 +
        confounder_strength * 1.5 * U  # U 的混淆效应
    )
    propensity = 1 / (1 + np.exp(-propensity_logit))
    T = np.random.binomial(1, propensity)

    # 潜在结果 (也受 U 影响)
    tau = 2.0  # 真实 ATE

    Y0 = (
        5.0 +
        1.0 * X1 +
        0.5 * X2 +
        confounder_strength * 2.0 * U +  # U 影响基线结果
        np.random.randn(n_samples) * 0.5
    )

    Y1 = Y0 + tau

    Y = np.where(T == 1, Y1, Y0)

    # 创建数据集 (不包含 U)
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'T': T,
        'Y': Y
    })

    # 真实 ATE
    ate_true = tau

    # 有偏估计 (仅用观测数据)
    ate_biased = Y[T == 1].mean() - Y[T == 0].mean()

    return df, U, ate_true, ate_biased


def visualize_sensitivity_analysis(
    n_samples: int,
    confounder_strength: float,
    correlation_with_x: float,
    max_gamma: float
) -> Tuple[go.Figure, str]:
    """可视化敏感性分析"""

    # 模拟数据
    df, U, ate_true, ate_biased = simulate_unobserved_confounding(
        n_samples=n_samples,
        confounder_strength=confounder_strength,
        correlation_with_x=correlation_with_x
    )

    X = df[['X1', 'X2']].values
    T = df['T'].values
    Y = df['Y'].values

    # Rosenbaum 边界
    gamma_values = np.linspace(1, max_gamma, 50)
    lower_bounds, upper_bounds = compute_rosenbaum_bounds(Y, T, gamma_values)

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rosenbaum Sensitivity Bounds',
            'Unobserved Confounder U Distribution',
            'Treatment Assignment by U',
            'Outcome Y by T and U'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )

    # 1. Rosenbaum 边界
    fig.add_trace(go.Scatter(
        x=gamma_values, y=upper_bounds,
        mode='lines', name='Upper Bound',
        line=dict(color='#EB5757', width=2),
        fill=None
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=gamma_values, y=lower_bounds,
        mode='lines', name='Lower Bound',
        line=dict(color='#2D9CDB', width=2),
        fill='tonexty', fillcolor='rgba(45, 156, 219, 0.2)'
    ), row=1, col=1)

    # 真实 ATE
    fig.add_trace(go.Scatter(
        x=[1, max_gamma], y=[ate_true, ate_true],
        mode='lines', name='True ATE',
        line=dict(color='#27AE60', width=2, dash='dash')
    ), row=1, col=1)

    # 观测 ATE
    fig.add_trace(go.Scatter(
        x=[1, max_gamma], y=[ate_biased, ate_biased],
        mode='lines', name='Observed ATE',
        line=dict(color='gray', width=2, dash='dot')
    ), row=1, col=1)

    # 零线
    fig.add_trace(go.Scatter(
        x=[1, max_gamma], y=[0, 0],
        mode='lines', name='Null Effect',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False
    ), row=1, col=1)

    # 2. U 的分布
    fig.add_trace(go.Histogram(
        x=U, name='Unobserved U',
        marker_color='#9B59B6', opacity=0.7, nbinsx=30
    ), row=1, col=2)

    # 3. 处理分配 by U
    U_quartiles = pd.qcut(U, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    treat_rate_by_u = [T[U_quartiles == q].mean() for q in ['Q1', 'Q2', 'Q3', 'Q4']]

    fig.add_trace(go.Bar(
        x=['Q1', 'Q2', 'Q3', 'Q4'], y=treat_rate_by_u,
        name='Treatment Rate',
        marker_color='#2D9CDB'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=['Q1', 'Q2', 'Q3', 'Q4'], y=[0.5, 0.5, 0.5, 0.5],
        mode='lines', name='Random (0.5)',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=2, col=1)

    # 4. 结果 Y by T and U
    U_groups = U_quartiles.astype(str).values
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask_q = U_groups == q
        Y_treat_q = Y[mask_q & (T == 1)]
        Y_control_q = Y[mask_q & (T == 0)]

        if len(Y_treat_q) > 0 and len(Y_control_q) > 0:
            ate_q = Y_treat_q.mean() - Y_control_q.mean()
        else:
            ate_q = 0

        # 箱线图数据
        fig.add_trace(go.Box(
            y=Y[mask_q & (T == 0)], name=f'{q} Control',
            marker_color='#2D9CDB', boxmean='sd',
            showlegend=False
        ), row=2, col=2)

        fig.add_trace(go.Box(
            y=Y[mask_q & (T == 1)], name=f'{q} Treated',
            marker_color='#EB5757', boxmean='sd',
            showlegend=False
        ), row=2, col=2)

    fig.update_xaxes(title_text='Gamma (Γ)', row=1, col=1)
    fig.update_yaxes(title_text='ATE Estimate', row=1, col=1)

    fig.update_xaxes(title_text='Unobserved Confounder U', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)

    fig.update_xaxes(title_text='U Quartile', row=2, col=1)
    fig.update_yaxes(title_text='Treatment Rate', row=2, col=1)

    fig.update_xaxes(title_text='U Quartile & Treatment', row=2, col=2)
    fig.update_yaxes(title_text='Outcome Y', row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Sensitivity Analysis: Impact of Unobserved Confounding'
    )

    # 摘要
    bias = ate_biased - ate_true
    bias_pct = (bias / ate_true * 100) if ate_true != 0 else 0

    # 找到置信区间包含 0 的最小 gamma
    gamma_threshold = None
    for i, gamma in enumerate(gamma_values):
        if lower_bounds[i] <= 0 <= upper_bounds[i]:
            gamma_threshold = gamma
            break

    summary = f"""
### 敏感性分析结果

#### 数据设置

| 参数 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 未观测混淆强度 | {confounder_strength:.2f} |
| U 与 X 的相关性 | {correlation_with_x:.2f} |

#### 效应估计

| 指标 | 值 |
|------|-----|
| 真实 ATE | {ate_true:.4f} |
| 观测 ATE (有偏) | {ate_biased:.4f} |
| 偏差 | {bias:.4f} ({bias_pct:.1f}%) |

#### Rosenbaum Bounds

**解读 Γ (Gamma)**:
- Γ = 1: 无未观测混淆 (无混淆假设成立)
- Γ = 2: 两个样本的处理概率可以相差 2 倍 (即使协变量相同)
- Γ 越大: 允许的未观测混淆越强

**敏感性阈值**:
{f"- 当 Γ ≥ {gamma_threshold:.2f} 时，置信区间包含 0 (效应可能不显著)" if gamma_threshold else "- 在 Γ ≤ " + str(max_gamma) + " 范围内，效应始终显著"}

### 关键洞察

1. **混淆的影响**:
   - 未观测混淆导致 ATE 估计偏差 {abs(bias):.3f}
   - 混淆越强，偏差越大

2. **相关性的作用**:
   - U 与 X 相关性: {correlation_with_x:.2f}
   - 相关性越高，用 X 调整后的残余偏差越小

3. **鲁棒性评估**:
   - 检查敏感性曲线的宽度
   - 曲线越窄，结论对未观测混淆越鲁棒

### 实践建议

1. **报告敏感性分析**:
   - 总是报告结论对未观测混淆的敏感性
   - "如果存在使倾向得分相差 2 倍的未观测混淆，结论仍然成立"

2. **寻找强工具变量**:
   - 工具变量不受未观测混淆影响
   - 可以作为鲁棒性检验

3. **收集更多协变量**:
   - 更多的协变量减少未观测混淆
   - 领域知识指导变量选择
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## Sensitivity Analysis - 敏感性分析

### 为什么需要敏感性分析？

因果推断的核心假设之一是 **无混淆假设 (Unconfoundedness)**:

$$(Y(0), Y(1)) \\perp T | X$$

即: 给定观测协变量 X，处理分配与潜在结果独立。

**问题**: 这个假设 **无法从数据中验证**！

可能存在未观测的混淆因子 U，使得:
- U 同时影响处理选择 T 和结果 Y
- U 未被包含在 X 中

### Rosenbaum Bounds

Rosenbaum (2002) 提出的敏感性分析方法：

**核心思想**: 量化"如果无混淆假设被违背，结论会改变多少"

**敏感性参数 Γ (Gamma)**:

对于两个协变量相同的个体 i 和 j (X_i = X_j):

$$\\frac{1}{\\Gamma} \\leq \\frac{P(T_i=1)}{P(T_j=1)} \\leq \\Gamma$$

- Γ = 1: 无未观测混淆（倾向得分相同）
- Γ = 2: 倾向得分可相差 2 倍
- Γ 越大: 允许的未观测混淆越强

**解读**:

1. 计算不同 Γ 下的 ATE 置信区间
2. 找到使结论改变的最小 Γ
3. 判断这样的混淆是否合理

### 示例

如果发现:
- Γ = 1.5 时效应仍显著
- Γ = 2.0 时效应变为不显著

解读: "如果存在一个未观测混淆使倾向得分相差 2 倍，则结论可能改变"

然后判断: 这样的混淆在实践中是否合理？

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=1000, maximum=10000, value=3000, step=500,
                    label="样本量"
                )
                confounder_strength = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.1,
                    label="未观测混淆强度 (0=无混淆, 1=强混淆)"
                )
                correlation_with_x = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.1,
                    label="U 与观测变量 X 的相关性"
                )
                max_gamma = gr.Slider(
                    minimum=1.5, maximum=5, value=3, step=0.5,
                    label="最大 Gamma 值"
                )
                run_btn = gr.Button("运行敏感性分析", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="敏感性分析可视化")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=visualize_sensitivity_analysis,
            inputs=[n_samples, confounder_strength, correlation_with_x, max_gamma],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 其他敏感性分析方法

#### 1. E-value (VanderWeele & Ding, 2017)

**定义**: 使观测关联完全被混淆解释所需的最小风险比。

$$E = RR + \\sqrt{RR \\times (RR - 1)}$$

其中 RR 是观测到的风险比。

**优点**:
- 简单直观
- 不需要额外假设
- 可用于已发表的研究

#### 2. Regression-based Sensitivity

改变未观测混淆的强度，观察估计如何变化:

$$Y = \\beta_0 + \\beta_T T + \\beta_X X + \\rho \\cdot U + \\epsilon$$

其中 U ~ N(0,1) 是假设的未观测混淆。

通过改变 ρ 来评估敏感性。

#### 3. Partial Identification Bounds

不对未观测混淆做参数假设，直接计算可识别的效应范围。

### 实践指南

#### 何时需要敏感性分析？

**总是需要！** 尤其是:
- 观测性研究（非随机化）
- 高风险决策（医疗、政策）
- 可能存在重要未观测因素

#### 如何报告？

1. **定量评估**: "如果存在使倾向得分相差 2 倍的未观测混淆，95% CI 仍不包含 0"

2. **定性判断**: "我们认为这样强的混淆不太可能，因为..."

3. **透明性**: 承认局限性，讨论可能的混淆来源

### 补充阅读

- Rosenbaum (2002). "Observational Studies"
- VanderWeele & Ding (2017). "Sensitivity Analysis in Observational Research"
- Ding & VanderWeele (2016). "Sensitivity Analysis Without Assumptions"

### 实践练习

对你的因果分析结果进行敏感性检验，评估结论的稳健性。
        """)

    return None
