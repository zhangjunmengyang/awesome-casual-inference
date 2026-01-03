"""
模型对比评估模块

对比多个因果推断模型的估计结果，进行稳健性检查
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

from .utils import generate_observational_data


def estimate_ate_naive(Y: np.ndarray, T: np.ndarray) -> Tuple[float, float]:
    """
    朴素估计 (简单差异)

    Returns:
    --------
    (ate_estimate, std_error)
    """
    y_t = Y[T == 1]
    y_c = Y[T == 0]

    ate = y_t.mean() - y_c.mean()

    # 标准误
    se = np.sqrt(y_t.var() / len(y_t) + y_c.var() / len(y_c))

    return ate, se


def estimate_ate_regression(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    回归调整估计

    Returns:
    --------
    (ate_estimate, std_error)
    """
    from sklearn.linear_model import LinearRegression

    # 拟合回归模型
    XT = np.column_stack([X, T])
    model = LinearRegression()
    model.fit(XT, Y)

    # ATE 是处理变量的系数
    ate = model.coef_[-1]

    # 简化的标准误计算
    residuals = Y - model.predict(XT)
    n = len(Y)
    k = XT.shape[1]
    mse = np.sum(residuals ** 2) / (n - k)

    # 计算 (X'X)^-1
    XTX_inv = np.linalg.inv(XT.T @ XT)
    se = np.sqrt(mse * XTX_inv[-1, -1])

    return ate, se


def estimate_ate_ipw(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    逆概率加权 (IPW) 估计

    Returns:
    --------
    (ate_estimate, std_error)
    """
    from sklearn.linear_model import LogisticRegression

    # 估计倾向得分
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]

    # 避免极端权重
    ps = np.clip(ps, 0.01, 0.99)

    # IPW 估计
    ate = (Y * T / ps).mean() - (Y * (1 - T) / (1 - ps)).mean()

    # 标准误 (简化计算)
    weights_t = T / ps
    weights_c = (1 - T) / (1 - ps)

    var_t = np.var(Y * weights_t) / (T.sum())
    var_c = np.var(Y * weights_c) / ((1 - T).sum())

    se = np.sqrt(var_t + var_c)

    return ate, se


def estimate_ate_dr(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    双重稳健 (Doubly Robust) 估计

    Returns:
    --------
    (ate_estimate, std_error)
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression

    # 估计倾向得分
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    # 估计结果回归
    # 对照组结果模型
    model_c = LinearRegression()
    model_c.fit(X[T == 0], Y[T == 0])
    mu0 = model_c.predict(X)

    # 处理组结果模型
    model_t = LinearRegression()
    model_t.fit(X[T == 1], Y[T == 1])
    mu1 = model_t.predict(X)

    # DR 估计
    ate = (
        np.mean(T * (Y - mu1) / ps + mu1) -
        np.mean((1 - T) * (Y - mu0) / (1 - ps) + mu0)
    )

    # 标准误 (简化计算)
    influence_t = T * (Y - mu1) / ps + mu1
    influence_c = (1 - T) * (Y - mu0) / (1 - ps) + mu0

    se = np.sqrt(np.var(influence_t - influence_c) / len(Y))

    return ate, se


def estimate_ate_matching(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    倾向得分匹配估计

    Returns:
    --------
    (ate_estimate, std_error)
    """
    from sklearn.linear_model import LogisticRegression

    # 估计倾向得分
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]

    # 简单的最近邻匹配
    treatment_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    matched_effects = []

    for t_idx in treatment_idx:
        t_ps = ps[t_idx]

        # 找到最近的对照
        distances = np.abs(ps[control_idx] - t_ps)
        nearest_c_idx = control_idx[np.argmin(distances)]

        # 计算配对效应
        effect = Y[t_idx] - Y[nearest_c_idx]
        matched_effects.append(effect)

    ate = np.mean(matched_effects)
    se = np.std(matched_effects) / np.sqrt(len(matched_effects))

    return ate, se


def perform_model_comparison(
    n_samples: int,
    treatment_assignment: str,
    show_confidence_intervals: bool
) -> tuple:
    """
    对比多个因果推断模型

    Parameters:
    -----------
    n_samples: 样本量
    treatment_assignment: 处理分配机制
    show_confidence_intervals: 是否显示置信区间

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
    Y = df['Y'].values

    # 真实 ATE
    true_ate = true_tau.mean()

    # 估计 ATE (使用多种方法)
    models = {
        'Naive (Simple Diff)': estimate_ate_naive,
        'Regression Adjustment': estimate_ate_regression,
        'IPW': estimate_ate_ipw,
        'Doubly Robust': estimate_ate_dr,
        'PSM': estimate_ate_matching
    }

    results = {}

    for name, estimator in models.items():
        if name == 'Naive (Simple Diff)':
            ate, se = estimator(Y, T)
        else:
            ate, se = estimator(Y, T, X)

        results[name] = {
            'ate': ate,
            'se': se,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'bias': ate - true_ate,
            'abs_bias': abs(ate - true_ate)
        }

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'ATE Estimates Comparison',
            'Bias Analysis',
            'Confidence Intervals',
            'Estimates Distribution'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "box"}]
        ]
    )

    model_names = list(results.keys())
    ate_estimates = [results[m]['ate'] for m in model_names]
    biases = [results[m]['bias'] for m in model_names]

    # 1. ATE Estimates Comparison
    colors = ['#EB5757' if abs(b) > 0.2 else '#27AE60' for b in biases]

    fig.add_trace(go.Bar(
        x=model_names,
        y=ate_estimates,
        marker_color=colors,
        name='ATE Estimate',
        text=[f'{ate:.3f}' for ate in ate_estimates],
        textposition='outside'
    ), row=1, col=1)

    # 添加真实 ATE 参考线
    fig.add_hline(y=true_ate, line_dash="dash", line_color="gray",
                  annotation_text=f"True ATE: {true_ate:.3f}",
                  row=1, col=1)

    # 2. Bias Analysis
    fig.add_trace(go.Bar(
        x=model_names,
        y=biases,
        marker_color=['#EB5757' if b > 0 else '#2D9CDB' for b in biases],
        name='Bias',
        text=[f'{b:.3f}' for b in biases],
        textposition='outside'
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # 3. Confidence Intervals
    for i, name in enumerate(model_names):
        ate = results[name]['ate']
        ci_lower = results[name]['ci_lower']
        ci_upper = results[name]['ci_upper']

        # 检查置信区间是否包含真实值
        contains_true = ci_lower <= true_ate <= ci_upper
        color = '#27AE60' if contains_true else '#EB5757'

        fig.add_trace(go.Scatter(
            x=[ate],
            y=[i],
            mode='markers',
            marker=dict(color=color, size=12, symbol='diamond'),
            name=name,
            showlegend=False
        ), row=2, col=1)

        # 添加置信区间
        if show_confidence_intervals:
            fig.add_trace(go.Scatter(
                x=[ci_lower, ci_upper],
                y=[i, i],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ), row=2, col=1)

    # 添加真实 ATE 参考线
    fig.add_vline(x=true_ate, line_dash="dash", line_color="gray",
                  annotation_text=f"True: {true_ate:.3f}",
                  row=2, col=1)

    # 4. Estimates Distribution (Box Plot)
    # 使用 Bootstrap 生成分布
    n_bootstrap = 100
    bootstrap_estimates = {name: [] for name in model_names}

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Bootstrap 采样
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[idx]
        T_boot = T[idx]
        Y_boot = Y[idx]

        for name, estimator in models.items():
            try:
                if name == 'Naive (Simple Diff)':
                    ate_boot, _ = estimator(Y_boot, T_boot)
                else:
                    ate_boot, _ = estimator(Y_boot, T_boot, X_boot)
                bootstrap_estimates[name].append(ate_boot)
            except:
                # 如果估计失败，跳过
                pass

    for name in model_names:
        if len(bootstrap_estimates[name]) > 0:
            fig.add_trace(go.Box(
                y=bootstrap_estimates[name],
                name=name,
                boxmean='sd'
            ), row=2, col=2)

    fig.add_hline(y=true_ate, line_dash="dash", line_color="gray", row=2, col=2)

    # 更新布局
    fig.update_layout(
        height=900,
        template='plotly_white',
        title_text='Causal Model Comparison',
        showlegend=False
    )

    fig.update_xaxes(title_text='Model', row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text='Model', row=1, col=2, tickangle=45)
    fig.update_xaxes(title_text='ATE Estimate', row=2, col=1)

    fig.update_yaxes(title_text='ATE Estimate', row=1, col=1)
    fig.update_yaxes(title_text='Bias', row=1, col=2)
    fig.update_yaxes(title_text='Model', row=2, col=1)
    fig.update_yaxes(title_text='ATE Estimate (Bootstrap)', row=2, col=2)

    # 更新 y 轴显示模型名称
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(range(len(model_names))),
        ticktext=model_names,
        row=2, col=1
    )

    # 统计摘要
    summary = f"""
### 模型对比结果

#### 数据信息
| 指标 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 真实 ATE | {true_ate:.4f} |
| 处理分配机制 | {treatment_assignment} |

#### 各模型估计结果
"""

    # 创建结果表格
    results_data = []
    for name in model_names:
        r = results[name]
        results_data.append({
            'Model': name,
            'ATE': f"{r['ate']:.4f}",
            'Bias': f"{r['bias']:.4f}",
            'SE': f"{r['se']:.4f}",
            '95% CI': f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]",
            'Contains True': '✓' if r['ci_lower'] <= true_ate <= r['ci_upper'] else '✗'
        })

    results_df = pd.DataFrame(results_data)

    summary += "\n" + results_df.to_markdown(index=False) + "\n"

    # 模型排名
    sorted_models = sorted(results.items(), key=lambda x: x[1]['abs_bias'])

    summary += f"""
---

#### 模型排名 (按偏差绝对值)

"""

    for i, (name, r) in enumerate(sorted_models, 1):
        summary += f"{i}. **{name}**: Bias = {r['bias']:.4f}, SE = {r['se']:.4f}\n"

    # 稳健性检查
    summary += f"""
---

### 稳健性检查

#### 一致性检验
"""

    # 检查不同模型估计是否一致
    ate_range = max(ate_estimates) - min(ate_estimates)
    ate_std = np.std(ate_estimates)

    if ate_range < 0.3:
        consistency = "高"
        summary += f"- 不同模型的估计结果一致性**高** (范围: {ate_range:.4f})\n"
    elif ate_range < 0.5:
        consistency = "中等"
        summary += f"- 不同模型的估计结果一致性**中等** (范围: {ate_range:.4f})\n"
    else:
        consistency = "低"
        summary += f"- 不同模型的估计结果一致性**低** (范围: {ate_range:.4f})\n"
        summary += "- **警告**: 模型间差异较大，需要进一步检查数据质量和模型假设\n"

    # 置信区间覆盖
    n_covered = sum(1 for r in results.values() if r['ci_lower'] <= true_ate <= r['ci_upper'])
    coverage_rate = n_covered / len(results)

    summary += f"\n#### 置信区间覆盖率\n"
    summary += f"- {n_covered}/{len(results)} 个模型的 95% 置信区间包含真实 ATE ({coverage_rate*100:.0f}%)\n"

    if coverage_rate < 0.8:
        summary += "- **警告**: 置信区间覆盖率较低，可能存在模型误设或估计偏差\n"

    summary += f"""
---

### 方法说明

| 方法 | 优点 | 缺点 | 假设 |
|------|------|------|------|
| **Naive** | 简单易懂 | 混淆偏差严重 | 随机分配 |
| **Regression** | 控制混淆 | 模型依赖 | 线性正确设定 |
| **IPW** | 无需结果模型 | 权重不稳定 | 倾向得分正确 |
| **Doubly Robust** | 双重保护 | 复杂度高 | 两模型至少一个正确 |
| **PSM** | 直观易懂 | 样本量减少 | 平衡假设 |

### 实践建议

1. **一致性高**: 结果可信，选择偏差最小的模型
2. **一致性低**:
   - 检查数据质量
   - 检查模型假设
   - 考虑使用双重稳健方法
3. **置信区间不包含真实值**: 可能存在未观测混淆或模型误设

### 双重稳健方法的优势

双重稳健 (Doubly Robust) 方法结合了回归和 IPW:
- **只需一个模型正确**: 倾向得分模型或结果模型
- **稳健性更强**: 降低模型误设风险
- **推荐使用**: 特别是在观测性研究中

### 练习

思考以下问题:
1. 为什么在混淆分配下，朴素估计有较大偏差?
2. 双重稳健方法如何实现"双重保护"?
3. 如果所有模型估计都不一致，应该怎么办?
"""

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 模型对比评估

对比多个因果推断模型的估计结果，进行稳健性检查。

### 核心方法

| 方法 | 类型 | 特点 |
|------|------|------|
| **Naive** | 简单差异 | 基线方法，忽略混淆 |
| **Regression** | 回归调整 | 控制协变量 |
| **IPW** | 加权 | 基于倾向得分 |
| **Doubly Robust** | 混合 | 双重保护 |
| **PSM** | 匹配 | 构造平衡样本 |

通过对比不同方法的结果，评估估计的稳健性。

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
                    info="观察混淆对不同方法的影响"
                )
                show_confidence_intervals = gr.Checkbox(
                    value=True,
                    label="显示置信区间",
                    info="在估计图中显示 95% 置信区间"
                )
                run_btn = gr.Button("运行模型对比", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="模型对比")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=perform_model_comparison,
            inputs=[n_samples, treatment_assignment, show_confidence_intervals],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 各方法公式

#### 1. Naive (Simple Difference)

$$\\hat{\\tau}_{naive} = \\bar{Y}_T - \\bar{Y}_C$$

#### 2. Regression Adjustment

$$\\hat{\\tau}_{reg} = \\hat{\\beta}_T \\text{ from } Y = \\alpha + X\\beta + T\\beta_T + \\epsilon$$

#### 3. IPW (Inverse Probability Weighting)

$$\\hat{\\tau}_{IPW} = \\frac{1}{n}\\sum_{i=1}^n \\frac{Y_i T_i}{e(X_i)} - \\frac{Y_i (1-T_i)}{1-e(X_i)}$$

#### 4. Doubly Robust

$$\\hat{\\tau}_{DR} = \\frac{1}{n}\\sum_{i=1}^n \\left[\\frac{T_i(Y_i - \\hat{\\mu}_1(X_i))}{e(X_i)} + \\hat{\\mu}_1(X_i) - \\frac{(1-T_i)(Y_i - \\hat{\\mu}_0(X_i))}{1-e(X_i)} - \\hat{\\mu}_0(X_i)\\right]$$

#### 5. PSM (Propensity Score Matching)

$$\\hat{\\tau}_{PSM} = \\frac{1}{n_T}\\sum_{i:T_i=1} (Y_i - Y_{j(i)})$$

其中 $j(i)$ 是与 $i$ 倾向得分最接近的对照样本。

### 稳健性检查清单

1. **估计一致性**: 不同方法的估计是否接近?
2. **置信区间**: 是否覆盖真实值?
3. **偏差方向**: 是否有系统性偏差?
4. **敏感性**: 对模型设定的敏感程度?

### 选择建议

- **RCT 数据**: Naive 或 Regression 即可
- **观测性数据，确信模型正确**: Regression 或 IPW
- **观测性数据，模型不确定**: Doubly Robust (推荐)
- **需要可解释性**: PSM + 平衡检查
        """)

    return None
