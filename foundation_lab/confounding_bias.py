"""
混淆偏差 (Confounding Bias) 可视化模块

核心概念:
- 混淆变量 (Confounder): 同时影响处理和结果的变量
- 混淆偏差: 由于未控制混淆变量导致的估计偏差
- Simpson's Paradox: 整体趋势与分组趋势相反的现象
- 后门准则 (Backdoor Criterion): 识别混淆变量的图准则
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_confounded_data(
    n_samples: int = 1000,
    true_ate: float = 2.0,
    confounding_strength: float = 2.0,
    seed: int = 42
) -> pd.DataFrame:
    """生成带混淆的数据"""
    np.random.seed(seed)

    # 混淆变量 X
    X = np.random.randn(n_samples)

    # 处理分配受 X 影响 (正向混淆)
    propensity = 1 / (1 + np.exp(-confounding_strength * X))
    T = np.random.binomial(1, propensity)

    # 结果受 X 和 T 影响
    Y = 5 + true_ate * T + confounding_strength * X + np.random.randn(n_samples) * 0.8

    return pd.DataFrame({
        'X': X,
        'T': T,
        'Y': Y,
        'propensity': propensity
    })


def visualize_confounding(
    n_samples: int,
    true_ate: float,
    confounding_strength: float
) -> tuple:
    """可视化混淆偏差"""

    df = generate_confounded_data(n_samples, true_ate, confounding_strength)

    # 计算估计
    naive_ate = df[df['T'] == 1]['Y'].mean() - df[df['T'] == 0]['Y'].mean()

    # 调整估计 (控制 X)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df[['T', 'X']], df['Y'])
    adjusted_ate = model.coef_[0]

    # 创建多子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '观测数据 (处理组 vs 控制组)',
            '混淆变量 X 的分布',
            '结果 Y vs 混淆变量 X',
            '估计对比'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. 观测数据散点图
    control = df[df['T'] == 0]
    treated = df[df['T'] == 1]

    fig.add_trace(go.Scatter(
        x=control['X'], y=control['Y'],
        mode='markers',
        name='控制组',
        marker=dict(color='blue', opacity=0.5, size=5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=treated['X'], y=treated['Y'],
        mode='markers',
        name='处理组',
        marker=dict(color='red', opacity=0.5, size=5)
    ), row=1, col=1)

    # 2. X 的分布 (按组)
    fig.add_trace(go.Histogram(
        x=control['X'],
        name='控制组 X',
        marker_color='blue',
        opacity=0.6,
        nbinsx=25
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=treated['X'],
        name='处理组 X',
        marker_color='red',
        opacity=0.6,
        nbinsx=25
    ), row=1, col=2)

    # 3. Y vs X (展示混淆)
    fig.add_trace(go.Scatter(
        x=df['X'], y=df['Y'],
        mode='markers',
        marker=dict(
            color=df['T'],
            colorscale=[[0, 'blue'], [1, 'red']],
            size=5,
            opacity=0.5
        ),
        name='Y vs X',
        showlegend=False
    ), row=2, col=1)

    # 添加趋势线
    z = np.polyfit(df['X'], df['Y'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['X'].min(), df['X'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode='lines',
        name='趋势线',
        line=dict(color='green', width=2, dash='dash'),
        showlegend=False
    ), row=2, col=1)

    # 4. 估计对比柱状图
    estimates = ['真实 ATE', '朴素估计', '调整估计']
    values = [true_ate, naive_ate, adjusted_ate]
    colors = ['green', 'red', 'blue']

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
        showlegend=True,
        legend=dict(x=1.02, y=1)
    )

    # 统计摘要
    bias = naive_ate - true_ate
    relative_bias = bias / true_ate * 100 if true_ate != 0 else 0

    summary = f"""
### 混淆偏差分析

| 指标 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 真实 ATE | {true_ate:.4f} |
| 混淆强度 | {confounding_strength:.2f} |
| 朴素估计 | {naive_ate:.4f} |
| 调整估计 | {adjusted_ate:.4f} |
| **偏差** | {bias:+.4f} ({relative_bias:+.1f}%) |

### 解读

- **混淆变量分布不平衡**: 观察第二张图，处理组和控制组的 X 分布不同
- **正向混淆**: X 同时正向影响 T 和 Y，导致朴素估计偏高
- **调整后**: 控制 X 后，估计接近真实值

### 公式

朴素估计:
$$\\hat{{\\tau}}_{{naive}} = E[Y|T=1] - E[Y|T=0] = {naive_ate:.4f}$$

调整估计 (后门调整):
$$\\hat{{\\tau}}_{{adj}} = \\sum_x \\{{ E[Y|T=1,X=x] - E[Y|T=0,X=x] \\}} P(X=x) = {adjusted_ate:.4f}$$
    """

    return fig, summary


def simulate_simpson_paradox() -> tuple:
    """模拟 Simpson's Paradox"""
    np.random.seed(42)

    # 生成数据: 整体趋势与分组趋势相反
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
    for t_val, color, name in [(0, 'blue', '控制'), (1, 'red', '处理')]:
        subset = df[df['T'] == t_val]
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'],
            mode='markers',
            marker=dict(color=color, opacity=0.5, size=6),
            name=f'{name}组',
            legendgroup=name
        ), row=1, col=1)

    # 组 A
    for t_val, color in [(0, 'blue'), (1, 'red')]:
        subset = df_A[df_A['T'] == t_val]
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'],
            mode='markers',
            marker=dict(color=color, opacity=0.5, size=6),
            showlegend=False
        ), row=1, col=2)

    # 组 B
    for t_val, color in [(0, 'blue'), (1, 'red')]:
        subset = df_B[df_B['T'] == t_val]
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'],
            mode='markers',
            marker=dict(color=color, opacity=0.5, size=6),
            showlegend=False
        ), row=1, col=3)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="Simpson's Paradox 演示"
    )

    explanation = f"""
### Simpson's Paradox

| 数据集 | 处理效应 |
|--------|----------|
| 整体数据 | {overall_effect:+.2f} (看起来正向!) |
| 组 A | {effect_A:+.2f} (负向) |
| 组 B | {effect_B:+.2f} (负向) |
| **真实效应** | **-1.50** (负向) |

### 发生了什么?

1. **真实效应是负的** (-1.5): 处理实际上降低了结果
2. **混淆**: 组 A 有更高的 X 值，导致更高的结果和更高的处理率
3. **整体趋势反转**: 因为处理组更多来自组 A (高结果组)

### 教训

**永远不要只看整体数据!** 在可能存在混淆的情况下，分层或调整分析是必要的。
    """

    return fig, explanation


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 混淆偏差 (Confounding Bias)

混淆是因果推断中最常见的问题，发生在一个变量同时影响处理和结果时。

### 核心概念

- **混淆变量**: 同时是处理和结果的原因
- **混淆偏差**: 未控制混淆导致的估计偏差
- **后门路径**: 从处理到结果的非因果关联

---
        """)

        gr.Markdown("### 混淆偏差模拟")

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=200, maximum=2000, value=1000, step=100,
                    label="样本量"
                )
                true_ate = gr.Slider(
                    minimum=-3, maximum=3, value=2.0, step=0.5,
                    label="真实 ATE"
                )
                confounding = gr.Slider(
                    minimum=0, maximum=3, value=1.5, step=0.1,
                    label="混淆强度"
                )
                run_btn = gr.Button("运行模拟", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="混淆偏差可视化")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=visualize_confounding,
            inputs=[n_samples, true_ate, confounding],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("---")
        gr.Markdown("### Simpson's Paradox")

        with gr.Row():
            simpson_btn = gr.Button("演示 Simpson's Paradox", variant="secondary")

        with gr.Row():
            simpson_plot = gr.Plot()

        with gr.Row():
            simpson_explain = gr.Markdown()

        simpson_btn.click(
            fn=simulate_simpson_paradox,
            inputs=[],
            outputs=[simpson_plot, simpson_explain]
        )

        gr.Markdown("""
---

### 思考题

1. 混淆强度如何影响偏差大小？
2. 为什么 Simpson's Paradox 在实际中很危险？
3. 如何判断是否存在未观测混淆？

### 练习

完成 `exercises/chapter1_foundation/ex3_confounding.py` 中的练习。
        """)

    return None
