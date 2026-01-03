"""
Uplift Tree 可视化模块

展示 Uplift 决策树的原理和分裂准则
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_uplift_gain(
    y: np.ndarray,
    t: np.ndarray,
    criterion: str = 'KL'
) -> float:
    """
    计算 Uplift 增益

    Parameters:
    -----------
    y: 结果
    t: 处理状态
    criterion: 分裂准则
        - 'KL': KL 散度 (Kullback-Leibler)
        - 'ED': 欧氏距离
        - 'Chi': 卡方统计量

    Returns:
    --------
    uplift gain value
    """
    # 处理组和控制组
    mask_t = t == 1
    mask_c = t == 0

    n_t = mask_t.sum()
    n_c = mask_c.sum()

    if n_t == 0 or n_c == 0:
        return 0.0

    # 转化率
    p_t = y[mask_t].mean() if n_t > 0 else 0.5
    p_c = y[mask_c].mean() if n_c > 0 else 0.5

    # 避免 log(0)
    p_t = np.clip(p_t, 0.001, 0.999)
    p_c = np.clip(p_c, 0.001, 0.999)

    if criterion == 'KL':
        # KL 散度: 衡量处理组和控制组分布的差异
        kl = p_t * np.log(p_t / p_c) + (1 - p_t) * np.log((1 - p_t) / (1 - p_c))
        return kl

    elif criterion == 'ED':
        # 欧氏距离
        return (p_t - p_c) ** 2

    elif criterion == 'Chi':
        # 卡方统计量
        chi = ((p_t - p_c) ** 2) / (p_c * (1 - p_c) + 1e-10)
        return chi

    else:
        return p_t - p_c  # 简单 uplift


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    feature_idx: int,
    criterion: str = 'KL',
    min_samples_leaf: int = 100
) -> tuple:
    """
    找到单个特征的最佳分裂点

    Returns:
    --------
    (best_threshold, best_gain, left_uplift, right_uplift)
    """
    feature = X[:, feature_idx]
    unique_values = np.unique(feature)

    if len(unique_values) < 2:
        return None, -np.inf, 0, 0

    # 候选分裂点
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    best_gain = -np.inf
    best_threshold = None
    best_left_uplift = 0
    best_right_uplift = 0

    # 当前节点的 uplift
    current_gain = calculate_uplift_gain(y, t, criterion)

    for threshold in thresholds:
        left_mask = feature <= threshold
        right_mask = ~left_mask

        # 检查最小样本要求
        if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
            continue

        # 计算左右子节点的增益
        left_gain = calculate_uplift_gain(y[left_mask], t[left_mask], criterion)
        right_gain = calculate_uplift_gain(y[right_mask], t[right_mask], criterion)

        # 加权增益
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        n_total = len(y)

        weighted_gain = (n_left / n_total * left_gain + n_right / n_total * right_gain)
        gain_improvement = weighted_gain - current_gain

        if gain_improvement > best_gain:
            best_gain = gain_improvement
            best_threshold = threshold

            # 计算 uplift
            t_left, c_left = t[left_mask], y[left_mask]
            t_right, c_right = t[right_mask], y[right_mask]

            left_uplift = y[left_mask & (t == 1)].mean() - y[left_mask & (t == 0)].mean() \
                if (left_mask & (t == 1)).sum() > 0 and (left_mask & (t == 0)).sum() > 0 else 0
            right_uplift = y[right_mask & (t == 1)].mean() - y[right_mask & (t == 0)].mean() \
                if (right_mask & (t == 1)).sum() > 0 and (right_mask & (t == 0)).sum() > 0 else 0

            best_left_uplift = left_uplift
            best_right_uplift = right_uplift

    return best_threshold, best_gain, best_left_uplift, best_right_uplift


def visualize_uplift_split(
    n_samples: int,
    feature_effect: float,
    criterion: str
) -> tuple:
    """
    可视化 Uplift 分裂过程
    """
    np.random.seed(42)

    # 生成数据
    X = np.random.randn(n_samples, 2)
    T = np.random.binomial(1, 0.5, n_samples)

    # 异质性效应: X1 影响 uplift
    base_uplift = 0.1
    heterogeneous_uplift = base_uplift + feature_effect * X[:, 0]
    heterogeneous_uplift = np.clip(heterogeneous_uplift, 0, 0.5)

    # 基线转化
    baseline = 0.2 + 0.1 * X[:, 1]
    baseline = np.clip(baseline, 0.05, 0.5)

    # 转化概率
    prob = baseline + heterogeneous_uplift * T
    Y = np.random.binomial(1, prob)

    # 找最佳分裂
    threshold, gain, left_uplift, right_uplift = find_best_split(
        X, Y, T, feature_idx=0, criterion=criterion
    )

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '数据分布与分裂点',
            '分裂前后 Uplift 对比',
            '分裂准则增益',
            '不同阈值的增益'
        )
    )

    # 1. 数据分布
    left_mask = X[:, 0] <= threshold if threshold else np.ones(n_samples, dtype=bool)
    colors = np.where(Y == 1, np.where(T == 1, 'red', 'blue'),
                     np.where(T == 1, 'lightcoral', 'lightblue'))

    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(
            color=['rgba(255,0,0,0.5)' if y == 1 and t == 1 else
                   'rgba(0,0,255,0.5)' if y == 1 and t == 0 else
                   'rgba(255,200,200,0.3)' if t == 1 else
                   'rgba(200,200,255,0.3)'
                   for y, t in zip(Y, T)],
            size=4
        ),
        showlegend=False
    ), row=1, col=1)

    # 分裂线
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="green",
                     annotation_text=f"Split: X1 <= {threshold:.2f}", row=1, col=1)

    # 2. Uplift 对比
    overall_uplift = Y[T == 1].mean() - Y[T == 0].mean()

    fig.add_trace(go.Bar(
        x=['Overall', 'Left (X1 <= threshold)', 'Right (X1 > threshold)'],
        y=[overall_uplift, left_uplift if threshold else 0, right_uplift if threshold else 0],
        marker_color=['gray', 'green', 'orange'],
        text=[f'{overall_uplift:.3f}',
              f'{left_uplift:.3f}' if threshold else 'N/A',
              f'{right_uplift:.3f}' if threshold else 'N/A'],
        textposition='outside'
    ), row=1, col=2)

    # 3. 分裂准则说明
    criteria_gains = []
    for crit in ['KL', 'ED', 'Chi']:
        _, g, _, _ = find_best_split(X, Y, T, 0, criterion=crit)
        criteria_gains.append(g)

    fig.add_trace(go.Bar(
        x=['KL Divergence', 'Euclidean Distance', 'Chi-Square'],
        y=criteria_gains,
        marker_color=['#2D9CDB', '#27AE60', '#EB5757'],
        text=[f'{g:.4f}' for g in criteria_gains],
        textposition='outside'
    ), row=2, col=1)

    # 4. 不同阈值的增益曲线
    thresholds = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    gains = []
    for th in thresholds:
        left = X[:, 0] <= th
        right = ~left
        if left.sum() > 50 and right.sum() > 50:
            lg = calculate_uplift_gain(Y[left], T[left], criterion)
            rg = calculate_uplift_gain(Y[right], T[right], criterion)
            gains.append((left.sum() / n_samples * lg + right.sum() / n_samples * rg))
        else:
            gains.append(0)

    fig.add_trace(go.Scatter(
        x=thresholds, y=gains,
        mode='lines',
        line=dict(color='#2D9CDB', width=2)
    ), row=2, col=2)

    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="green", row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='Uplift Tree 分裂可视化',
        showlegend=False
    )

    # 摘要
    threshold_str = f"X1 <= {threshold:.4f}" if threshold is not None else "N/A"
    summary = f"""
### Uplift Tree 分裂分析

**分裂准则**: {criterion}

| 指标 | 值 |
|------|-----|
| 最佳分裂点 | {threshold_str} |
| 分裂增益 | {gain:.6f} |
| 整体 Uplift | {overall_uplift:.4f} |
| 左子节点 Uplift | {left_uplift:.4f} |
| 右子节点 Uplift | {right_uplift:.4f} |

### 分裂准则解释

- **KL Divergence**: 衡量处理组和控制组分布差异，信息论基础
- **Euclidean Distance**: 简单直观，计算转化率差异的平方
- **Chi-Square**: 统计检验视角，考虑样本量

### 为什么需要特殊的分裂准则?

传统决策树优化**预测准确性**，而 Uplift Tree 优化**处理效应的异质性**。
我们希望分裂后的子节点有**不同的 uplift**，而不是预测更准。
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## Uplift Tree - 增益决策树

Uplift Tree 是专门为估计异质性处理效应设计的决策树变体。

### 核心思想

传统决策树: 最大化预测准确性 (如信息增益、基尼系数)
Uplift Tree: 最大化处理效应的异质性 (如 KL 散度、欧氏距离)

### 分裂准则

| 准则 | 公式 | 特点 |
|------|------|------|
| KL Divergence | D_KL(P_T || P_C) | 信息论基础，对小概率敏感 |
| Euclidean | (p_T - p_C)^2 | 简单直观 |
| Chi-Square | (p_T - p_C)^2 / (p_C(1-p_C)) | 统计检验视角 |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=1000, maximum=5000, value=2000, step=500,
                    label="样本量"
                )
                feature_effect = gr.Slider(
                    minimum=0, maximum=0.3, value=0.15, step=0.01,
                    label="特征对 Uplift 的影响强度"
                )
                criterion = gr.Radio(
                    choices=['KL', 'ED', 'Chi'],
                    value='KL',
                    label="分裂准则"
                )
                run_btn = gr.Button("可视化分裂", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="Uplift Tree 分裂")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=visualize_uplift_split,
            inputs=[n_samples, feature_effect, criterion],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 练习

完成 `exercises/chapter3_uplift/ex2_uplift_tree.py` 中的练习。
        """)

    return None
