"""
选择偏差 (Selection Bias) 可视化模块

核心概念:
- 选择偏差: 样本选择与研究变量相关导致的偏差
- Berkson's Paradox: 在碰撞变量上条件化导致的虚假关联
- 存活偏差 (Survivorship Bias): 只观察到"存活"样本
- 样本选择偏差: 样本不代表目标总体
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_berkson_data(
    n_samples: int = 2000,
    selection_strength: float = 1.5,
    seed: int = 42
) -> tuple:
    """
    生成 Berkson's Paradox 数据

    场景: 医院住院研究
    - X1: 疾病A严重程度 (独立于 X2)
    - X2: 疾病B严重程度 (独立于 X1)
    - 住院概率取决于 X1 + X2

    在住院人群中，X1 和 X2 会呈现负相关!
    """
    np.random.seed(seed)

    # 两个独立的疾病严重程度
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)

    # 选择 (住院) 机制: 任一疾病严重都会住院
    selection_score = selection_strength * (X1 + X2)
    selection_prob = 1 / (1 + np.exp(-selection_score + 2))
    S = np.random.binomial(1, selection_prob)

    full_data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'S': S,
        'selection_prob': selection_prob
    })

    selected_data = full_data[full_data['S'] == 1].copy()

    return full_data, selected_data


def visualize_berkson_paradox(
    n_samples: int,
    selection_strength: float
) -> tuple:
    """可视化 Berkson's Paradox"""

    full_data, selected_data = generate_berkson_data(n_samples, selection_strength)

    # 计算相关系数
    full_corr = np.corrcoef(full_data['X1'], full_data['X2'])[0, 1]
    selected_corr = np.corrcoef(selected_data['X1'], selected_data['X2'])[0, 1]

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'总体数据 (相关系数: {full_corr:.3f})',
            f'住院人群 (相关系数: {selected_corr:.3f})',
            '选择概率热力图',
            '相关系数对比'
        ),
        vertical_spacing=0.15
    )

    # 1. 总体数据
    fig.add_trace(go.Scatter(
        x=full_data['X1'], y=full_data['X2'],
        mode='markers',
        marker=dict(
            color=full_data['S'],
            colorscale=[[0, 'lightgray'], [1, 'red']],
            size=4,
            opacity=0.5
        ),
        name='总体',
        showlegend=False
    ), row=1, col=1)

    # 添加总体回归线
    z_full = np.polyfit(full_data['X1'], full_data['X2'], 1)
    p_full = np.poly1d(z_full)
    x_line = np.linspace(full_data['X1'].min(), full_data['X1'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p_full(x_line),
        mode='lines',
        line=dict(color='blue', width=2),
        name='总体趋势',
        showlegend=False
    ), row=1, col=1)

    # 2. 选择后数据
    fig.add_trace(go.Scatter(
        x=selected_data['X1'], y=selected_data['X2'],
        mode='markers',
        marker=dict(color='red', size=5, opacity=0.5),
        name='住院',
        showlegend=False
    ), row=1, col=2)

    # 添加选择后回归线
    z_sel = np.polyfit(selected_data['X1'], selected_data['X2'], 1)
    p_sel = np.poly1d(z_sel)
    fig.add_trace(go.Scatter(
        x=x_line, y=p_sel(x_line),
        mode='lines',
        line=dict(color='green', width=2),
        name='选择后趋势',
        showlegend=False
    ), row=1, col=2)

    # 3. 选择概率热力图
    x_grid = np.linspace(-3, 3, 50)
    y_grid = np.linspace(-3, 3, 50)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    Z_prob = 1 / (1 + np.exp(-(selection_strength * (X_mesh + Y_mesh) - 2)))

    fig.add_trace(go.Heatmap(
        x=x_grid, y=y_grid, z=Z_prob,
        colorscale='RdBu_r',
        colorbar=dict(title='P(住院)', x=1.02),
        name='选择概率'
    ), row=2, col=1)

    # 4. 相关系数对比
    fig.add_trace(go.Bar(
        x=['总体', '住院人群'],
        y=[full_corr, selected_corr],
        marker_color=['blue', 'red'],
        text=[f'{full_corr:.3f}', f'{selected_corr:.3f}'],
        textposition='outside',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text="Berkson's Paradox: 住院偏差"
    )

    # 添加 y=0 参考线
    fig.add_hline(y=0, row=2, col=2, line_dash="dash", line_color="gray")

    summary = f"""
### Berkson's Paradox 分析

| 指标 | 总体 | 住院人群 |
|------|------|----------|
| 样本量 | {len(full_data)} | {len(selected_data)} ({len(selected_data)/len(full_data)*100:.1f}%) |
| X1-X2 相关 | {full_corr:.4f} | {selected_corr:.4f} |

### 发生了什么?

1. **总体中 X1 和 X2 独立** (相关系数 ≈ 0)
2. **住院是碰撞变量**: 住院 ← X1, 住院 ← X2
3. **条件化打开路径**: 在住院人群中，X1 和 X2 变成负相关

### 直觉解释

如果一个人因为疾病 A 住院 (X1 高)，他不太需要疾病 B 严重才能住院 (X2 可以低)。
反过来也成立。这就产生了虚假的负相关。

### 实际影响

- 医院数据研究常见这个问题
- 某些疾病在医院研究中显示"保护作用"，但实际上是选择偏差
    """

    return fig, summary


def visualize_survivorship_bias(
    n_companies: int = 500,
    risk_factor: float = 0.3,
    years: int = 10
) -> tuple:
    """
    可视化存活偏差

    场景: 分析"成功公司"的特点
    - 高风险策略可能带来高收益或倒闭
    - 只观察存活公司会高估风险策略的收益
    """
    np.random.seed(42)

    # 生成公司数据
    # risk_level: 公司的风险策略程度
    risk_level = np.random.uniform(0, 1, n_companies)

    # 模拟多年后的结果
    # 高风险公司: 高方差收益，但也高倒闭率
    # 收益 = 基础收益 + 风险bonus - 随机冲击
    base_return = 0.05  # 5% 基础年化
    risk_bonus = risk_level * 0.1  # 高风险最多额外 10%
    random_shock = np.random.randn(n_companies) * (risk_level * risk_factor)

    final_return = (1 + base_return + risk_bonus + random_shock) ** years - 1

    # 存活判断: 年化收益低于 -50% 视为倒闭
    survived = final_return > -0.5

    df = pd.DataFrame({
        'risk_level': risk_level,
        'final_return': final_return,
        'survived': survived,
        'annual_return': (1 + final_return) ** (1/years) - 1
    })

    survivors = df[df['survived']]
    failed = df[~df['survived']]

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '所有公司的风险-收益关系',
            '只看存活公司 (存活偏差!)',
            '存活率 vs 风险水平',
            '风险-收益相关性对比'
        )
    )

    # 1. 所有公司
    fig.add_trace(go.Scatter(
        x=df['risk_level'], y=df['final_return'],
        mode='markers',
        marker=dict(
            color=df['survived'].map({True: 'green', False: 'red'}),
            size=5,
            opacity=0.5
        ),
        name='所有公司',
        showlegend=False
    ), row=1, col=1)

    # 2. 只有存活公司
    fig.add_trace(go.Scatter(
        x=survivors['risk_level'], y=survivors['final_return'],
        mode='markers',
        marker=dict(color='green', size=5, opacity=0.6),
        name='存活公司',
        showlegend=False
    ), row=1, col=2)

    # 添加回归线
    z_surv = np.polyfit(survivors['risk_level'], survivors['final_return'], 1)
    p_surv = np.poly1d(z_surv)
    x_line = np.linspace(0, 1, 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p_surv(x_line),
        mode='lines',
        line=dict(color='blue', width=2),
        showlegend=False
    ), row=1, col=2)

    # 3. 存活率 vs 风险
    risk_bins = np.linspace(0, 1, 11)
    survival_rates = []
    bin_centers = []
    for i in range(len(risk_bins) - 1):
        mask = (df['risk_level'] >= risk_bins[i]) & (df['risk_level'] < risk_bins[i+1])
        if mask.sum() > 0:
            survival_rates.append(df.loc[mask, 'survived'].mean())
            bin_centers.append((risk_bins[i] + risk_bins[i+1]) / 2)

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=survival_rates,
        marker_color='steelblue',
        showlegend=False
    ), row=2, col=1)

    # 4. 相关性对比
    all_corr = np.corrcoef(df['risk_level'], df['final_return'])[0, 1]
    surv_corr = np.corrcoef(survivors['risk_level'], survivors['final_return'])[0, 1]

    fig.add_trace(go.Bar(
        x=['所有公司', '存活公司'],
        y=[all_corr, surv_corr],
        marker_color=['gray', 'green'],
        text=[f'{all_corr:.3f}', f'{surv_corr:.3f}'],
        textposition='outside',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text="存活偏差: 分析'成功公司'的陷阱"
    )

    survival_rate = df['survived'].mean() * 100

    summary = f"""
### 存活偏差分析

| 指标 | 所有公司 | 存活公司 |
|------|----------|----------|
| 样本量 | {len(df)} | {len(survivors)} |
| 存活率 | 100% | {survival_rate:.1f}% |
| 风险-收益相关 | {all_corr:.4f} | {surv_corr:.4f} |
| 平均收益 | {df['final_return'].mean()*100:.1f}% | {survivors['final_return'].mean()*100:.1f}% |

### 存活偏差的影响

1. **真实关系**: 风险与收益的真实相关性 = {all_corr:.3f}
2. **偏差后关系**: 只看存活公司，相关性 = {surv_corr:.3f}
3. **偏差方向**: 存活偏差**高估**了风险策略的收益

### 为什么会这样?

高风险公司有两种命运:
- 成功: 高收益 → 进入"存活"样本
- 失败: 倒闭 → 从样本中消失

我们只看到成功的高风险公司，错误地认为高风险 = 高收益。

### 经典例子

- 二战飞机装甲研究 (Abraham Wald)
- 对冲基金收益分析
- 创业公司研究
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 选择偏差 (Selection Bias)

选择偏差发生在样本不代表目标总体时，是因果推断中常见但容易被忽视的问题。

### 主要类型

- **Berkson's Paradox**: 在碰撞变量上条件化导致虚假关联
- **存活偏差**: 只观察到"存活"的样本
- **自选择偏差**: 处理分配受结果影响

---
        """)

        with gr.Tabs():
            with gr.Tab("Berkson's Paradox"):
                gr.Markdown("### Berkson's Paradox - 住院偏差")

                with gr.Row():
                    n_berkson = gr.Slider(
                        minimum=500, maximum=3000, value=2000, step=100,
                        label="样本量"
                    )
                    sel_strength = gr.Slider(
                        minimum=0.5, maximum=3, value=1.5, step=0.1,
                        label="选择强度"
                    )
                    berkson_btn = gr.Button("运行模拟", variant="primary")

                with gr.Row():
                    berkson_plot = gr.Plot()

                with gr.Row():
                    berkson_summary = gr.Markdown()

                berkson_btn.click(
                    fn=visualize_berkson_paradox,
                    inputs=[n_berkson, sel_strength],
                    outputs=[berkson_plot, berkson_summary]
                )

            with gr.Tab("存活偏差"):
                gr.Markdown("### 存活偏差 - 成功公司的陷阱")

                with gr.Row():
                    n_companies = gr.Slider(
                        minimum=200, maximum=1000, value=500, step=50,
                        label="公司数量"
                    )
                    risk_factor = gr.Slider(
                        minimum=0.1, maximum=0.5, value=0.3, step=0.05,
                        label="风险波动系数"
                    )
                    n_years = gr.Slider(
                        minimum=5, maximum=20, value=10, step=1,
                        label="模拟年数"
                    )
                    survival_btn = gr.Button("运行模拟", variant="primary")

                with gr.Row():
                    survival_plot = gr.Plot()

                with gr.Row():
                    survival_summary = gr.Markdown()

                survival_btn.click(
                    fn=visualize_survivorship_bias,
                    inputs=[n_companies, risk_factor, n_years],
                    outputs=[survival_plot, survival_summary]
                )

        gr.Markdown("""
---

### 思考题

1. Berkson's Paradox 为什么会产生负相关？
2. 如何在实际研究中避免存活偏差？
3. 选择偏差和混淆偏差有什么区别？

### 练习

完成 `exercises/chapter1_foundation/ex4_selection_bias.py` 中的练习。
        """)

    return None
