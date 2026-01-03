"""
样本量计算器

业务场景：
---------
- 实验需要跑多久？需要多少用户？
- 能检测到多小的效果？
- 如何平衡实验成本和检验功效？

核心公式：
---------
二项指标（转化率等）：
n = 2 * (Z_α + Z_β)² * p(1-p) / δ²

连续指标（收入等）：
n = 2 * (Z_α + Z_β)² * σ² / δ²

其中：
- Z_α: 显著性水平对应的 Z 值（通常 1.96）
- Z_β: 功效对应的 Z 值（通常 0.84）
- p: 基线转化率
- σ: 标准差
- δ: 最小可检测效应 (MDE)

面试考点：
---------
- 为什么需要样本量计算？
- MDE 和业务意义的关系？
- 如何处理多重检验？
- 方差缩减技术？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SampleSizeResult:
    """样本量计算结果"""
    sample_size_per_group: int
    total_sample_size: int
    mde: float
    baseline: float
    alpha: float
    power: float
    metric_type: str
    days_needed: Optional[int] = None


def calculate_sample_size_proportion(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True
) -> SampleSizeResult:
    """
    计算比例指标的样本量

    Parameters:
    -----------
    baseline_rate: 基线转化率 (e.g., 0.05 = 5%)
    mde_relative: 相对提升 (e.g., 0.10 = 10% 相对提升)
    alpha: 显著性水平
    power: 统计功效
    two_sided: 是否双侧检验

    Returns:
    --------
    SampleSizeResult
    """
    # 效应量（绝对提升）
    mde_absolute = baseline_rate * mde_relative
    treatment_rate = baseline_rate + mde_absolute

    # Z 值
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)

    # 合并方差
    pooled_var = baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)

    # 样本量公式
    n = (z_alpha + z_beta) ** 2 * pooled_var / (mde_absolute ** 2)
    n = int(np.ceil(n))

    return SampleSizeResult(
        sample_size_per_group=n,
        total_sample_size=n * 2,
        mde=mde_relative,
        baseline=baseline_rate,
        alpha=alpha,
        power=power,
        metric_type='proportion'
    )


def calculate_sample_size_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True
) -> SampleSizeResult:
    """
    计算连续指标的样本量

    Parameters:
    -----------
    baseline_mean: 基线均值
    baseline_std: 基线标准差
    mde_relative: 相对提升
    alpha: 显著性水平
    power: 统计功效
    two_sided: 是否双侧检验

    Returns:
    --------
    SampleSizeResult
    """
    # 效应量
    mde_absolute = baseline_mean * mde_relative

    # Z 值
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)

    # 样本量公式
    n = 2 * (z_alpha + z_beta) ** 2 * (baseline_std ** 2) / (mde_absolute ** 2)
    n = int(np.ceil(n))

    return SampleSizeResult(
        sample_size_per_group=n,
        total_sample_size=n * 2,
        mde=mde_relative,
        baseline=baseline_mean,
        alpha=alpha,
        power=power,
        metric_type='continuous'
    )


def calculate_mde(
    sample_size_per_group: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    metric_type: str = 'proportion',
    baseline_std: Optional[float] = None
) -> float:
    """
    给定样本量，计算最小可检测效应

    Parameters:
    -----------
    sample_size_per_group: 每组样本量
    baseline_rate: 基线值（转化率或均值）
    alpha: 显著性水平
    power: 统计功效
    metric_type: 'proportion' 或 'continuous'
    baseline_std: 连续指标的标准差

    Returns:
    --------
    最小可检测效应（相对值）
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    if metric_type == 'proportion':
        # MDE = sqrt(2 * var * (z_alpha + z_beta)^2 / n) / baseline
        var = baseline_rate * (1 - baseline_rate)
        mde_absolute = np.sqrt(2 * var * (z_alpha + z_beta) ** 2 / sample_size_per_group)
        mde_relative = mde_absolute / baseline_rate
    else:
        # 连续指标
        if baseline_std is None:
            baseline_std = baseline_rate * 0.5  # 假设 CV = 0.5
        mde_absolute = np.sqrt(2 * (z_alpha + z_beta) ** 2 * baseline_std ** 2 / sample_size_per_group)
        mde_relative = mde_absolute / baseline_rate

    return mde_relative


def calculate_experiment_duration(
    total_sample_size: int,
    daily_traffic: int,
    traffic_allocation: float = 1.0
) -> int:
    """
    计算实验所需天数

    Parameters:
    -----------
    total_sample_size: 总样本量
    daily_traffic: 日均流量
    traffic_allocation: 实验流量占比

    Returns:
    --------
    所需天数
    """
    effective_daily = daily_traffic * traffic_allocation
    days = int(np.ceil(total_sample_size / effective_daily))
    return days


def plot_sample_size_curves(
    baseline_rate: float,
    mde_range: Tuple[float, float] = (0.01, 0.30),
    power_levels: list = [0.7, 0.8, 0.9],
    alpha: float = 0.05
) -> go.Figure:
    """绘制样本量曲线"""
    mde_values = np.linspace(mde_range[0], mde_range[1], 50)

    fig = go.Figure()

    colors = ['#2D9CDB', '#27AE60', '#9B59B6']

    for i, power in enumerate(power_levels):
        sample_sizes = []
        for mde in mde_values:
            result = calculate_sample_size_proportion(baseline_rate, mde, alpha, power)
            sample_sizes.append(result.total_sample_size)

        fig.add_trace(go.Scatter(
            x=mde_values * 100,
            y=sample_sizes,
            mode='lines',
            name=f'Power = {power*100:.0f}%',
            line=dict(color=colors[i], width=2)
        ))

    fig.update_layout(
        title=f'样本量 vs MDE (基线转化率 = {baseline_rate*100:.1f}%)',
        xaxis_title='最小可检测效应 (相对提升 %)',
        yaxis_title='总样本量',
        yaxis_type='log',
        template='plotly_white',
        height=400
    )

    return fig


def plot_power_curves(
    baseline_rate: float,
    sample_sizes: list = [1000, 5000, 10000, 50000],
    mde_range: Tuple[float, float] = (0.01, 0.30),
    alpha: float = 0.05
) -> go.Figure:
    """绘制功效曲线"""
    mde_values = np.linspace(mde_range[0], mde_range[1], 50)

    fig = go.Figure()
    colors = ['#2D9CDB', '#27AE60', '#9B59B6', '#F2994A']

    for i, n in enumerate(sample_sizes):
        powers = []
        for mde in mde_values:
            mde_absolute = baseline_rate * mde
            treatment_rate = baseline_rate + mde_absolute

            # 计算功效
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            pooled_var = baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
            se = np.sqrt(pooled_var / n)

            z_beta = mde_absolute / se - z_alpha
            power = stats.norm.cdf(z_beta)
            powers.append(power)

        fig.add_trace(go.Scatter(
            x=mde_values * 100,
            y=np.array(powers) * 100,
            mode='lines',
            name=f'n = {n:,}',
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    # 添加 80% 功效参考线
    fig.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="80% 功效")

    fig.update_layout(
        title=f'功效曲线 (基线转化率 = {baseline_rate*100:.1f}%)',
        xaxis_title='最小可检测效应 (相对提升 %)',
        yaxis_title='统计功效 (%)',
        template='plotly_white',
        height=400
    )

    return fig


def run_sample_size_calculator(
    metric_type: str,
    baseline_value: float,
    mde_relative: float,
    alpha: float,
    power: float,
    daily_traffic: int,
    traffic_allocation: float,
    baseline_std: Optional[float] = None
) -> Tuple[go.Figure, str]:
    """运行样本量计算"""

    # 计算样本量
    if metric_type == 'proportion':
        result = calculate_sample_size_proportion(
            baseline_value / 100,  # 转为小数
            mde_relative / 100,
            alpha,
            power
        )
    else:
        std = baseline_std if baseline_std else baseline_value * 0.5
        result = calculate_sample_size_continuous(
            baseline_value,
            std,
            mde_relative / 100,
            alpha,
            power
        )

    # 计算实验天数
    days_needed = calculate_experiment_duration(
        result.total_sample_size,
        daily_traffic,
        traffic_allocation / 100
    )

    # 可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('样本量 vs MDE', '功效曲线')
    )

    if metric_type == 'proportion':
        baseline_for_plot = baseline_value / 100
    else:
        baseline_for_plot = baseline_value

    # 样本量曲线
    mde_values = np.linspace(0.02, 0.30, 50)
    for p in [0.7, 0.8, 0.9]:
        sizes = []
        for m in mde_values:
            if metric_type == 'proportion':
                r = calculate_sample_size_proportion(baseline_for_plot, m, alpha, p)
            else:
                std = baseline_std if baseline_std else baseline_value * 0.5
                r = calculate_sample_size_continuous(baseline_value, std, m, alpha, p)
            sizes.append(r.total_sample_size)

        fig.add_trace(
            go.Scatter(x=mde_values * 100, y=sizes, name=f'Power={p*100:.0f}%',
                       mode='lines'),
            row=1, col=1
        )

    # 标记当前点
    fig.add_trace(
        go.Scatter(
            x=[mde_relative],
            y=[result.total_sample_size],
            mode='markers',
            marker=dict(size=12, color='red', symbol='star'),
            name='当前设置'
        ),
        row=1, col=1
    )

    # 功效曲线
    sample_sizes = [1000, 5000, 10000, result.sample_size_per_group]
    for n in sample_sizes:
        powers = []
        for m in mde_values:
            if metric_type == 'proportion':
                mde_calc = calculate_mde(n, baseline_for_plot, alpha, power, 'proportion')
                # 简化功效计算
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                mde_abs = baseline_for_plot * m
                var = 2 * baseline_for_plot * (1 - baseline_for_plot)
                se = np.sqrt(var / n)
                z_beta = mde_abs / se - z_alpha
                p_val = stats.norm.cdf(z_beta)
            else:
                std = baseline_std if baseline_std else baseline_value * 0.5
                mde_abs = baseline_value * m
                se = std * np.sqrt(2 / n)
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = mde_abs / se - z_alpha
                p_val = stats.norm.cdf(z_beta)
            powers.append(p_val * 100)

        fig.add_trace(
            go.Scatter(x=mde_values * 100, y=powers, name=f'n={n:,}',
                       mode='lines'),
            row=1, col=2
        )

    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=True
    )

    fig.update_xaxes(title_text='MDE (%)', row=1, col=1)
    fig.update_yaxes(title_text='总样本量', type='log', row=1, col=1)
    fig.update_xaxes(title_text='MDE (%)', row=1, col=2)
    fig.update_yaxes(title_text='功效 (%)', row=1, col=2)

    # 生成报告
    report = f"""
### 样本量计算结果

#### 输入参数
| 参数 | 值 |
|-----|-----|
| 指标类型 | {'转化率' if metric_type == 'proportion' else '连续指标'} |
| 基线值 | {baseline_value}{'%' if metric_type == 'proportion' else ''} |
| 最小可检测效应 (MDE) | {mde_relative}% 相对提升 |
| 显著性水平 (α) | {alpha} |
| 统计功效 (1-β) | {power*100:.0f}% |

#### 计算结果
| 指标 | 值 |
|-----|-----|
| **每组样本量** | **{result.sample_size_per_group:,}** |
| **总样本量** | **{result.total_sample_size:,}** |
| 日均流量 | {daily_traffic:,} |
| 实验流量占比 | {traffic_allocation}% |
| **预计实验天数** | **{days_needed} 天** |

#### 解读

- 需要 **{result.total_sample_size:,}** 个样本才能以 **{power*100:.0f}%** 的概率检测到 **{mde_relative}%** 的相对提升
- 按当前流量，实验需要运行约 **{days_needed} 天**
- 如果效果实际更大，可以更快得到显著结果
- 如果效果实际更小，可能无法检测到（需要更多样本）

#### 建议

1. **MDE 选择**: 确保 {mde_relative}% 的提升对业务有实际意义
2. **实验周期**: {days_needed} 天{'需要覆盖完整周末效应' if days_needed < 7 else '已覆盖至少一个完整周'}
3. **提前终止**: 不要在达到样本量前因为看到显著结果就停止实验
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 样本量计算器

### 为什么需要样本量计算？

在实验开始前，需要回答：
- 需要多少样本才能检测到效果？
- 实验要跑多久？
- 能检测到的最小效果是多少？

**不做样本量计算的后果**：
- 样本太少 → 真有效果也检测不到（假阴性）
- 样本太多 → 浪费流量和时间
- 实验周期不足 → 结果不稳定

---

### 核心概念

| 概念 | 定义 | 典型值 |
|-----|------|--------|
| **α (显著性水平)** | 假阳性率，无效果时误判为有效果的概率 | 0.05 (5%) |
| **β (假阴性率)** | 有效果时误判为无效果的概率 | 0.20 (20%) |
| **Power (功效)** | 1-β，有效果时正确检测出的概率 | 0.80 (80%) |
| **MDE** | 最小可检测效应，能检测到的最小变化 | 业务决定 |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                metric_type = gr.Radio(
                    choices=['proportion', 'continuous'],
                    value='proportion',
                    label="指标类型"
                )
                baseline_value = gr.Number(value=5.0, label="基线值 (转化率填百分比，如5表示5%)")
                baseline_std = gr.Number(value=None, label="标准差 (仅连续指标需要)", visible=False)
                mde_relative = gr.Slider(1, 30, 10, step=1, label="最小可检测效应 (相对提升 %)")

            with gr.Column(scale=1):
                alpha = gr.Slider(0.01, 0.10, 0.05, step=0.01, label="显著性水平 (α)")
                power = gr.Slider(0.7, 0.95, 0.8, step=0.05, label="统计功效")
                daily_traffic = gr.Number(value=10000, label="日均流量", precision=0)
                traffic_allocation = gr.Slider(10, 100, 100, step=10, label="实验流量占比 (%)")

        run_btn = gr.Button("计算样本量", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="可视化")

        with gr.Row():
            report_output = gr.Markdown()

        # 切换指标类型时显示/隐藏标准差输入
        def toggle_std_input(metric_type):
            return gr.update(visible=(metric_type == 'continuous'))

        metric_type.change(
            fn=toggle_std_input,
            inputs=[metric_type],
            outputs=[baseline_std]
        )

        run_btn.click(
            fn=run_sample_size_calculator,
            inputs=[metric_type, baseline_value, mde_relative, alpha, power,
                    daily_traffic, traffic_allocation, baseline_std],
            outputs=[plot_output, report_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: MDE 怎么选？**
> MDE 应该由业务决定，而非统计决定：
> - 多大的提升才值得上线？
> - 多小的提升不值得实验成本？
> 例：转化率从 5% 提升到 5.5% (10% 相对提升)，能带来多少收入？

**Q2: 为什么不能看到显著就停止？**
> 这是"偷看问题" (Peeking Problem)：
> - 多次检验增加假阳性率
> - α=0.05 但看 10 次，实际假阳性率 > 20%
> 解决方案：序贯检验 (Sequential Testing)

**Q3: 如何减少所需样本量？**
> 方差缩减技术：
> - CUPED：用实验前数据调整
> - 分层抽样：按特征分层
> - 配对设计：前后对比
> 可减少 20-50% 样本量

**Q4: 多个指标怎么办？**
> 多重检验校正：
> - Bonferroni：α' = α/k
> - Holm-Bonferroni：阶梯校正
> - FDR：控制假发现率
        """)

    return None


if __name__ == "__main__":
    # 测试
    result = calculate_sample_size_proportion(0.05, 0.10, 0.05, 0.8)
    print(f"每组样本量: {result.sample_size_per_group:,}")
    print(f"总样本量: {result.total_sample_size:,}")
