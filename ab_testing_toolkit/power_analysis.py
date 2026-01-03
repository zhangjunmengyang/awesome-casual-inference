"""
功效分析模块

功能：
-----
1. 事前功效分析：设计阶段确定样本量
2. 事后功效分析：解读实验结果（慎用）
3. 敏感性分析：参数变化对结果的影响

面试考点：
---------
- 什么是统计功效？
- 事后功效分析有什么问题？
- 如何提高实验功效？
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class PowerAnalysisResult:
    """功效分析结果"""
    power: float
    sample_size: int
    effect_size: float
    alpha: float
    analysis_type: str  # 'priori' or 'post_hoc'
    warnings: List[str]


def calculate_power_proportion(
    n_per_group: int,
    baseline_rate: float,
    effect_size: float,
    alpha: float = 0.05,
    two_sided: bool = True
) -> float:
    """
    计算比例检验的功效

    Parameters:
    -----------
    n_per_group: 每组样本量
    baseline_rate: 基线转化率
    effect_size: 相对效应大小
    alpha: 显著性水平
    two_sided: 是否双侧检验

    Returns:
    --------
    统计功效 (0-1)
    """
    # 效应量
    mde_absolute = baseline_rate * effect_size
    treatment_rate = baseline_rate + mde_absolute

    # Z 值
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))

    # 标准误
    pooled_var = baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
    se = np.sqrt(pooled_var / n_per_group)

    # 功效
    z_beta = mde_absolute / se - z_alpha
    power = stats.norm.cdf(z_beta)

    return power


def calculate_power_continuous(
    n_per_group: int,
    effect_size: float,
    std: float,
    alpha: float = 0.05,
    two_sided: bool = True
) -> float:
    """
    计算连续指标检验的功效

    Parameters:
    -----------
    n_per_group: 每组样本量
    effect_size: Cohen's d (效应量/标准差)
    std: 标准差
    alpha: 显著性水平
    two_sided: 是否双侧检验

    Returns:
    --------
    统计功效 (0-1)
    """
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))

    # 标准误
    se = std * np.sqrt(2 / n_per_group)

    # 非中心参数
    ncp = effect_size * std / se

    # 功效
    power = stats.norm.cdf(ncp - z_alpha)

    return power


def sensitivity_analysis(
    baseline_rate: float,
    n_range: Tuple[int, int] = (1000, 100000),
    mde_range: Tuple[float, float] = (0.01, 0.30),
    alpha: float = 0.05,
    target_power: float = 0.80
) -> go.Figure:
    """
    敏感性分析：展示样本量、MDE、功效的关系

    生成功效热力图
    """
    n_values = np.logspace(np.log10(n_range[0]), np.log10(n_range[1]), 50).astype(int)
    mde_values = np.linspace(mde_range[0], mde_range[1], 50)

    power_matrix = np.zeros((len(mde_values), len(n_values)))

    for i, mde in enumerate(mde_values):
        for j, n in enumerate(n_values):
            power_matrix[i, j] = calculate_power_proportion(n, baseline_rate, mde, alpha)

    fig = go.Figure(data=go.Heatmap(
        z=power_matrix * 100,
        x=n_values,
        y=mde_values * 100,
        colorscale='RdYlGn',
        zmin=0,
        zmax=100,
        colorbar=dict(title='功效 (%)')
    ))

    # 添加 80% 功效等高线
    fig.add_trace(go.Contour(
        z=power_matrix * 100,
        x=n_values,
        y=mde_values * 100,
        contours=dict(
            start=80,
            end=80,
            size=0,
            showlabels=True
        ),
        line=dict(color='black', width=2),
        showscale=False,
        name='80% 功效'
    ))

    fig.update_layout(
        title=f'功效敏感性分析 (基线 = {baseline_rate*100:.1f}%)',
        xaxis_title='每组样本量',
        yaxis_title='MDE (相对提升 %)',
        xaxis_type='log',
        template='plotly_white',
        height=500
    )

    return fig


def post_hoc_power_analysis(
    observed_effect: float,
    se: float,
    alpha: float = 0.05
) -> PowerAnalysisResult:
    """
    事后功效分析（仅用于解读，不应指导决策）

    WARNING: 事后功效分析存在严重问题：
    1. 循环论证：显著结果的事后功效总是 > 50%
    2. 无法改变实验：实验已经结束
    3. 误导解读：低事后功效不意味着"需要更多样本"

    正确用法：仅用于解释为什么某些效应未被检测到
    """
    # 计算功效
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = observed_effect / se - z_alpha
    power = stats.norm.cdf(z_beta)

    warnings = [
        "警告：事后功效分析结果应谨慎解读",
        "事后功效与 p 值直接相关，不提供额外信息",
        "不应用于证明'需要更多样本'",
        "正确做法：事前进行功效分析"
    ]

    return PowerAnalysisResult(
        power=power,
        sample_size=0,  # 事后分析不涉及样本量决策
        effect_size=observed_effect,
        alpha=alpha,
        analysis_type='post_hoc',
        warnings=warnings
    )


def minimum_detectable_effect(
    n_per_group: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> float:
    """
    给定样本量，计算最小可检测效应

    Parameters:
    -----------
    n_per_group: 每组样本量
    baseline_rate: 基线转化率
    alpha: 显著性水平
    power: 目标功效

    Returns:
    --------
    最小可检测效应（相对值）
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # 近似求解（假设效应较小）
    var = baseline_rate * (1 - baseline_rate)
    se = np.sqrt(2 * var / n_per_group)

    mde_absolute = (z_alpha + z_beta) * se
    mde_relative = mde_absolute / baseline_rate

    return mde_relative


def plot_power_analysis(
    baseline_rate: float,
    current_n: int,
    current_mde: float,
    alpha: float = 0.05
) -> go.Figure:
    """绘制功效分析图"""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('功效 vs 样本量', '功效 vs 效应量')
    )

    # 图1: 功效 vs 样本量
    n_values = np.logspace(2, 6, 100).astype(int)
    powers = [calculate_power_proportion(n, baseline_rate, current_mde, alpha)
              for n in n_values]

    fig.add_trace(
        go.Scatter(x=n_values, y=np.array(powers) * 100, mode='lines',
                   name='功效曲线', line=dict(color='#2D9CDB', width=2)),
        row=1, col=1
    )

    # 标记当前点
    current_power = calculate_power_proportion(current_n, baseline_rate, current_mde, alpha)
    fig.add_trace(
        go.Scatter(x=[current_n], y=[current_power * 100], mode='markers',
                   marker=dict(size=12, color='red', symbol='star'),
                   name=f'当前 (n={current_n:,})'),
        row=1, col=1
    )

    # 80% 参考线
    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=1, col=1)

    # 图2: 功效 vs 效应量
    mde_values = np.linspace(0.01, 0.30, 100)
    powers_mde = [calculate_power_proportion(current_n, baseline_rate, m, alpha)
                  for m in mde_values]

    fig.add_trace(
        go.Scatter(x=mde_values * 100, y=np.array(powers_mde) * 100, mode='lines',
                   name='功效曲线', line=dict(color='#27AE60', width=2)),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=[current_mde * 100], y=[current_power * 100], mode='markers',
                   marker=dict(size=12, color='red', symbol='star'),
                   name=f'当前 (MDE={current_mde*100:.1f}%)'),
        row=1, col=2
    )

    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=True
    )

    fig.update_xaxes(title_text='每组样本量', type='log', row=1, col=1)
    fig.update_yaxes(title_text='功效 (%)', row=1, col=1)
    fig.update_xaxes(title_text='MDE (相对提升 %)', row=1, col=2)
    fig.update_yaxes(title_text='功效 (%)', row=1, col=2)

    return fig


class PowerAnalyzer:
    """功效分析器"""

    def __init__(self, metric_type: str = 'proportion'):
        self.metric_type = metric_type

    def analyze(
        self,
        baseline: float,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05,
        std: Optional[float] = None
    ) -> Dict:
        """
        执行功效分析

        Returns:
        --------
        分析结果字典
        """
        if self.metric_type == 'proportion':
            power = calculate_power_proportion(
                sample_size, baseline, effect_size, alpha
            )
            mde = minimum_detectable_effect(sample_size, baseline, alpha, 0.80)
        else:
            if std is None:
                std = baseline * 0.5
            power = calculate_power_continuous(
                sample_size, effect_size * baseline / std, std, alpha
            )
            mde = effect_size  # 简化

        return {
            'power': power,
            'sample_size': sample_size,
            'effect_size': effect_size,
            'mde_at_80_power': mde,
            'alpha': alpha,
            'metric_type': self.metric_type,
            'recommendation': self._get_recommendation(power, effect_size, mde)
        }

    def _get_recommendation(self, power: float, effect_size: float, mde: float) -> str:
        """生成建议"""
        if power >= 0.80:
            return "功效充足，可以检测到预期效应"
        elif power >= 0.50:
            return f"功效偏低 ({power*100:.0f}%)，建议增加样本量或接受更大的 MDE ({mde*100:.1f}%)"
        else:
            return f"功效不足 ({power*100:.0f}%)，强烈建议增加样本量"


def render():
    """渲染 Gradio 界面"""
    import gradio as gr

    with gr.Blocks() as block:
        gr.Markdown("""
## 功效分析

### 什么是统计功效？

统计功效 = 当效果真实存在时，正确检测出来的概率

```
功效 = 1 - β = P(拒绝 H₀ | H₀ 为假)
```

**功效的影响因素**：
1. 效应量：效果越大，越容易检测
2. 样本量：样本越多，功效越高
3. 方差：方差越小，功效越高
4. 显著性水平：α 越大，功效越高

---
        """)

        with gr.Row():
            with gr.Column():
                baseline = gr.Number(value=5.0, label="基线转化率 (%)")
                effect_size = gr.Slider(1, 30, 10, step=1, label="预期效应 (相对提升 %)")
                sample_size = gr.Number(value=10000, label="每组样本量", precision=0)
                alpha = gr.Slider(0.01, 0.10, 0.05, step=0.01, label="显著性水平 (α)")

            with gr.Column():
                run_btn = gr.Button("分析功效", variant="primary")
                power_output = gr.Number(label="统计功效 (%)", precision=1)
                mde_output = gr.Number(label="80% 功效下的 MDE (%)", precision=1)
                recommendation = gr.Textbox(label="建议")

        with gr.Row():
            plot_output = gr.Plot()

        def run_analysis(baseline, effect_size, sample_size, alpha):
            analyzer = PowerAnalyzer('proportion')
            result = analyzer.analyze(
                baseline / 100,
                effect_size / 100,
                int(sample_size),
                alpha
            )

            fig = plot_power_analysis(
                baseline / 100,
                int(sample_size),
                effect_size / 100,
                alpha
            )

            return (
                result['power'] * 100,
                result['mde_at_80_power'] * 100,
                result['recommendation'],
                fig
            )

        run_btn.click(
            fn=run_analysis,
            inputs=[baseline, effect_size, sample_size, alpha],
            outputs=[power_output, mde_output, recommendation, plot_output]
        )

        gr.Markdown("""
---

### 事后功效分析的陷阱

**不要做事后功效分析！**

常见错误：
> "实验不显著，事后功效只有 30%，说明需要更多样本"

这是错误的，因为：
1. 事后功效与 p 值一一对应，不提供新信息
2. 显著结果的事后功效总是 > 50%
3. 无法改变已完成的实验

**正确做法**：
- 实验前进行功效分析
- 实验后报告置信区间而非功效
        """)

    return None


if __name__ == "__main__":
    # 测试
    power = calculate_power_proportion(10000, 0.05, 0.10, 0.05)
    print(f"功效: {power*100:.1f}%")

    mde = minimum_detectable_effect(10000, 0.05, 0.05, 0.80)
    print(f"80% 功效下的 MDE: {mde*100:.1f}%")
