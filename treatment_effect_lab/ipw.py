"""
逆概率加权 (Inverse Probability Weighting, IPW) 模块

实现 IPW 和增强 IPW (AIPW)

核心概念:
- IPW: 使用倾向得分的逆作为权重，创造"伪总体"
- 权重: w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))
- AIPW: 双重稳健估计器，结合结果回归和 IPW
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression, Ridge
from typing import Tuple, Optional

from .utils import (
    generate_confounded_data,
    compute_ate_oracle,
    compute_naive_ate,
    compute_propensity_overlap
)


class IPWEstimator:
    """
    逆概率加权 (IPW) 估计器

    ATE = E[Y(1)] - E[Y(0)]
        = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
    """

    def __init__(self, propensity_model=None, clip_weights: bool = True, weight_percentile: float = 99):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        clip_weights: 是否裁剪极端权重
        weight_percentile: 权重裁剪百分位数
        """
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000, random_state=42)
        self.clip_weights = clip_weights
        self.weight_percentile = weight_percentile
        self.propensity = None
        self.weights = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """训练倾向得分模型"""
        self.propensity_model.fit(X, T)
        return self

    def _compute_weights(self, propensity: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """
        计算 IPW 权重

        w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))
        """
        # 避免除零，裁剪倾向得分
        propensity = np.clip(propensity, 0.01, 0.99)

        weights = treatment / propensity + (1 - treatment) / (1 - propensity)

        # 裁剪极端权重
        if self.clip_weights:
            max_weight = np.percentile(weights, self.weight_percentile)
            weights = np.clip(weights, 0, max_weight)

        return weights

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        估计 ATE

        Parameters:
        -----------
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量

        Returns:
        --------
        (ATE估计, 标准误, 权重)
        """
        # 估计倾向得分
        self.propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 计算权重
        self.weights = self._compute_weights(self.propensity, T)

        # 加权估计 E[Y(1)] 和 E[Y(0)]
        treated_mask = T == 1
        control_mask = T == 0

        # E[Y(1)]
        y1_weighted = np.sum(Y[treated_mask] * self.weights[treated_mask]) / np.sum(self.weights[treated_mask])

        # E[Y(0)]
        y0_weighted = np.sum(Y[control_mask] * self.weights[control_mask]) / np.sum(self.weights[control_mask])

        ate = y1_weighted - y0_weighted

        # 计算标准误 (简化版)
        # 使用 Hajek 估计器的渐进方差
        n = len(Y)

        # 处理组残差
        residuals_1 = np.zeros(n)
        residuals_1[treated_mask] = (Y[treated_mask] - y1_weighted) * self.weights[treated_mask]

        # 控制组残差
        residuals_0 = np.zeros(n)
        residuals_0[control_mask] = (Y[control_mask] - y0_weighted) * self.weights[control_mask]

        # 影响函数
        influence_fn = residuals_1 - residuals_0

        variance = np.var(influence_fn) / n
        se = np.sqrt(variance)

        return ate, se, self.weights


class AIPWEstimator:
    """
    增强逆概率加权 (Augmented IPW, AIPW) 估计器

    双重稳健估计器: 只要倾向得分模型或结果模型有一个正确，估计就是一致的

    ATE = E[(mu_1(X) - mu_0(X)) +
            T*(Y - mu_1(X))/e(X) -
            (1-T)*(Y - mu_0(X))/(1-e(X))]

    其中:
    - mu_1(X) = E[Y|X, T=1]
    - mu_0(X) = E[Y|X, T=0]
    - e(X) = P(T=1|X)
    """

    def __init__(
        self,
        propensity_model=None,
        outcome_model=None,
        clip_weights: bool = True
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型
        outcome_model: 结果模型
        clip_weights: 是否裁剪权重
        """
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000, random_state=42)
        if outcome_model is None:
            self.outcome_model_0 = Ridge(alpha=1.0, random_state=42)
            self.outcome_model_1 = Ridge(alpha=1.0, random_state=43)
        else:
            from sklearn.base import clone
            self.outcome_model_0 = clone(outcome_model)
            self.outcome_model_1 = clone(outcome_model)
        self.clip_weights = clip_weights

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        估计 ATE

        Parameters:
        -----------
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量

        Returns:
        --------
        (ATE估计, 标准误)
        """
        n = len(Y)

        # 1. 估计倾向得分
        self.propensity_model.fit(X, T)
        propensity = self.propensity_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)

        # 2. 估计结果模型
        treated_mask = T == 1
        control_mask = T == 0

        # mu_1(X)
        self.outcome_model_1.fit(X[treated_mask], Y[treated_mask])
        mu_1 = self.outcome_model_1.predict(X)

        # mu_0(X)
        self.outcome_model_0.fit(X[control_mask], Y[control_mask])
        mu_0 = self.outcome_model_0.predict(X)

        # 3. AIPW 估计
        # 第一项: 结果模型预测的差异
        term1 = mu_1 - mu_0

        # 第二项: 处理组的 IPW 修正
        term2 = T * (Y - mu_1) / propensity

        # 第三项: 控制组的 IPW 修正
        term3 = (1 - T) * (Y - mu_0) / (1 - propensity)

        # AIPW 估计
        aipw_scores = term1 + term2 - term3

        ate = aipw_scores.mean()

        # 标准误
        se = aipw_scores.std() / np.sqrt(n)

        return ate, se


def visualize_ipw_weights(
    propensity: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray
) -> go.Figure:
    """
    可视化 IPW 权重分布

    Parameters:
    -----------
    propensity: 倾向得分
    treatment: 处理状态
    weights: IPW 权重

    Returns:
    --------
    Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '倾向得分分布',
            'IPW 权重分布',
            '倾向得分 vs 权重 (处理组)',
            '倾向得分 vs 权重 (控制组)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    treated_mask = treatment == 1
    control_mask = treatment == 0

    # 1. 倾向得分分布
    fig.add_trace(go.Histogram(
        x=propensity[control_mask],
        name='控制组',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=30
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=propensity[treated_mask],
        name='处理组',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=30
    ), row=1, col=1)

    # 2. 权重分布
    fig.add_trace(go.Histogram(
        x=weights[control_mask],
        name='控制组权重',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=30,
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=weights[treated_mask],
        name='处理组权重',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=30,
        showlegend=False
    ), row=1, col=2)

    # 3. 处理组: 倾向得分 vs 权重
    fig.add_trace(go.Scatter(
        x=propensity[treated_mask],
        y=weights[treated_mask],
        mode='markers',
        name='处理组',
        marker=dict(color='#EB5757', size=4, opacity=0.5),
        showlegend=False
    ), row=2, col=1)

    # 理论曲线: w = 1/e
    prop_range = np.linspace(0.01, 0.99, 100)
    theoretical_weight_t = 1 / prop_range
    fig.add_trace(go.Scatter(
        x=prop_range,
        y=theoretical_weight_t,
        mode='lines',
        name='理论权重: 1/e(X)',
        line=dict(color='black', dash='dash'),
        showlegend=False
    ), row=2, col=1)

    # 4. 控制组: 倾向得分 vs 权重
    fig.add_trace(go.Scatter(
        x=propensity[control_mask],
        y=weights[control_mask],
        mode='markers',
        name='控制组',
        marker=dict(color='#2D9CDB', size=4, opacity=0.5),
        showlegend=False
    ), row=2, col=2)

    # 理论曲线: w = 1/(1-e)
    theoretical_weight_c = 1 / (1 - prop_range)
    fig.add_trace(go.Scatter(
        x=prop_range,
        y=theoretical_weight_c,
        mode='lines',
        name='理论权重: 1/(1-e(X))',
        line=dict(color='black', dash='dash'),
        showlegend=False
    ), row=2, col=2)

    # 更新布局
    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='IPW 权重诊断',
        barmode='overlay'
    )

    fig.update_xaxes(title_text='倾向得分', row=1, col=1)
    fig.update_xaxes(title_text='IPW 权重', row=1, col=2)
    fig.update_xaxes(title_text='倾向得分', row=2, col=1)
    fig.update_xaxes(title_text='倾向得分', row=2, col=2)

    fig.update_yaxes(title_text='频数', row=1, col=1)
    fig.update_yaxes(title_text='频数', row=1, col=2)
    fig.update_yaxes(title_text='权重', row=2, col=1)
    fig.update_yaxes(title_text='权重', row=2, col=2)

    return fig


def run_ipw_demo(
    n_samples: int,
    confounding_strength: float,
    clip_weights: bool
) -> Tuple[go.Figure, str]:
    """
    运行 IPW/AIPW 演示

    Parameters:
    -----------
    n_samples: 样本数
    confounding_strength: 混淆强度
    clip_weights: 是否裁剪权重

    Returns:
    --------
    (figure, markdown_report)
    """
    # 生成数据
    df, params = generate_confounded_data(
        n_samples=n_samples,
        confounding_strength=confounding_strength,
        seed=42
    )

    feature_names = [col for col in df.columns if col.startswith('X')]
    X = df[feature_names].values
    T = df['T'].values
    Y = df['Y'].values

    # 真实 ATE 和朴素估计
    true_ate = params['true_ate']
    naive_ate = compute_naive_ate(df)

    # IPW 估计
    ipw = IPWEstimator(clip_weights=clip_weights)
    ipw.fit(X, T)
    ipw_ate, ipw_se, weights = ipw.estimate_ate(X, T, Y)

    # AIPW 估计
    aipw = AIPWEstimator(clip_weights=clip_weights)
    aipw_ate, aipw_se = aipw.estimate_ate(X, T, Y)

    # 可视化
    fig = visualize_ipw_weights(ipw.propensity, T, weights)

    # 计算有效样本量
    # ESS = (sum(w))^2 / sum(w^2)
    ess = np.sum(weights) ** 2 / np.sum(weights ** 2)
    ess_fraction = ess / n_samples

    # 重叠统计
    overlap_stats = compute_propensity_overlap(ipw.propensity, T)

    # 生成报告
    report = f"""
### 逆概率加权 (IPW & AIPW) 结果

#### 样本统计
| 指标 | 值 |
|------|-----|
| 总样本数 | {n_samples} |
| 处理组样本 | {(T == 1).sum()} |
| 控制组样本 | {(T == 0).sum()} |
| 有效样本量 (ESS) | {ess:.1f} ({ess_fraction*100:.1f}%) |
| 权重裁剪 | {'是' if clip_weights else '否'} |

#### 倾向得分重叠
| 统计量 | 处理组 | 控制组 |
|--------|--------|--------|
| 最小值 | {overlap_stats['treated_min']:.4f} | {overlap_stats['control_min']:.4f} |
| 最大值 | {overlap_stats['treated_max']:.4f} | {overlap_stats['control_max']:.4f} |
| 均值 | {overlap_stats['treated_mean']:.4f} | {overlap_stats['control_mean']:.4f} |

重叠区间: [{overlap_stats['overlap_min']:.4f}, {overlap_stats['overlap_max']:.4f}]

#### ATE 估计
| 方法 | 估计值 | 标准误 | 95% CI | 偏差 |
|------|--------|--------|--------|------|
| 真实 ATE | {true_ate:.4f} | - | - | - |
| 朴素估计 | {naive_ate:.4f} | - | - | {naive_ate - true_ate:.4f} |
| IPW | {ipw_ate:.4f} | {ipw_se:.4f} | [{ipw_ate - 1.96*ipw_se:.4f}, {ipw_ate + 1.96*ipw_se:.4f}] | {ipw_ate - true_ate:.4f} |
| AIPW | {aipw_ate:.4f} | {aipw_se:.4f} | [{aipw_ate - 1.96*aipw_se:.4f}, {aipw_ate + 1.96*aipw_se:.4f}] | {aipw_ate - true_ate:.4f} |

#### 关键洞察

- **IPW 原理**: 通过重加权创造"伪总体"，其中处理分配与协变量独立
- **权重意义**:
  - 处理组: w = 1/e(X) - 低倾向得分个体权重高
  - 控制组: w = 1/(1-e(X)) - 高倾向得分个体权重高
- **有效样本量**: {ess:.1f} ({ess_fraction*100:.1f}%) - 衡量权重分散程度
- **AIPW 优势**: 双重稳健，只要倾向得分或结果模型之一正确即可

#### IPW vs AIPW

- **IPW**: 依赖倾向得分模型正确
- **AIPW**: 双重稳健，效率更高，方差更小
- 观察到 AIPW 的标准误通常小于 IPW

#### 注意事项

⚠️ **极端权重**: 倾向得分接近 0 或 1 时，权重会非常大，导致估计不稳定
⚠️ **共同支撑**: 需要处理组和控制组的倾向得分有充分重叠
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 逆概率加权 (Inverse Probability Weighting, IPW)

IPW 通过重加权观测数据，创造一个"伪总体"，使得处理分配与协变量独立。

### 核心思想

**IPW 权重**:
```
w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))
```

**ATE 估计**:
```
ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
```

### 增强 IPW (AIPW)

AIPW 是**双重稳健**估计器，结合了结果回归和 IPW:

```
ATE = E[(mu_1(X) - mu_0(X)) +
        T*(Y - mu_1(X))/e(X) -
        (1-T)*(Y - mu_0(X))/(1-e(X))]
```

**双重稳健性质**: 只要倾向得分模型或结果模型有一个正确，估计就是一致的！

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=500, maximum=5000, value=2000, step=500,
                    label="样本量"
                )
                confounding_strength = gr.Slider(
                    minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                    label="混淆强度"
                )
                clip_weights = gr.Checkbox(
                    value=True,
                    label="裁剪极端权重"
                )
                run_btn = gr.Button("运行 IPW/AIPW", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="IPW 权重诊断")

        with gr.Row():
            report_output = gr.Markdown()

        run_btn.click(
            fn=run_ipw_demo,
            inputs=[n_samples, confounding_strength, clip_weights],
            outputs=[plot_output, report_output]
        )

        gr.Markdown("""
---

### 权重诊断

**有效样本量 (Effective Sample Size)**:
```
ESS = (sum(w))^2 / sum(w^2)
```

ESS 衡量权重分散程度。ESS 越接近 n，权重越均匀。

### 共同支撑假设

倾向得分分布需要有充分重叠。如果处理组和控制组的倾向得分几乎不重叠，IPW 会产生极端权重。

### IPW vs PSM

| 特性 | IPW | PSM |
|------|-----|-----|
| 样本利用 | 使用所有样本 | 可能丢弃未匹配样本 |
| 权重 | 连续权重 | 0/1 权重 |
| 方差 | 可能较大 (极端权重) | 相对稳定 |
| 估计目标 | ATE (全总体) | ATT/ATC (匹配总体) |

### AIPW 的双重稳健性

即使倾向得分模型错误，只要结果模型正确，AIPW 仍然一致 (反之亦然)。

这使得 AIPW 成为实践中的首选方法。

### 实践练习

对比不同倾向得分模型（Logistic Regression vs GBM）对 IPW 估计的影响。
        """)

    return None
