"""
双重稳健估计模块

展示双重稳健估计的性质和优势

核心概念:
- 双重稳健性: 倾向得分模型或结果模型有一个正确即可
- 对模型误设定的鲁棒性
- 效率提升
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Tuple, Dict

from .utils import (
    generate_confounded_data,
    compute_ate_oracle,
    compute_naive_ate,
    evaluate_ate_estimator
)


class DoublyRobustEstimator:
    """
    双重稳健估计器

    实现标准的 AIPW 估计器，并提供灵活的模型选择
    """

    def __init__(
        self,
        propensity_model=None,
        outcome_model=None,
        model_type: str = 'linear'
    ):
        """
        Parameters:
        -----------
        propensity_model: 倾向得分模型 (可选)
        outcome_model: 结果模型 (可选)
        model_type: 模型类型 ('linear', 'rf')
        """
        self.model_type = model_type

        # 默认模型
        if propensity_model is None:
            if model_type == 'linear':
                self.propensity_model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                self.propensity_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.propensity_model = propensity_model

        if outcome_model is None:
            if model_type == 'linear':
                self.outcome_model_0 = Ridge(alpha=1.0, random_state=42)
                self.outcome_model_1 = Ridge(alpha=1.0, random_state=43)
            else:
                self.outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
                self.outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=43)
        else:
            from sklearn.base import clone
            self.outcome_model_0 = clone(outcome_model)
            self.outcome_model_1 = clone(outcome_model)

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        propensity_correct: bool = True,
        outcome_correct: bool = True
    ) -> Tuple[float, float]:
        """
        估计 ATE，可以人为引入模型误设定

        Parameters:
        -----------
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量
        propensity_correct: 倾向得分模型是否正确
        outcome_correct: 结果模型是否正确

        Returns:
        --------
        (ATE估计, 标准误)
        """
        n = len(Y)

        # 1. 估计倾向得分 (可能误设定)
        if propensity_correct:
            X_prop = X
        else:
            # 人为误设定: 只使用部分特征
            X_prop = X[:, :2]  # 只用前两个特征

        self.propensity_model.fit(X_prop, T)

        if hasattr(self.propensity_model, 'predict_proba'):
            if propensity_correct:
                propensity = self.propensity_model.predict_proba(X)[:, 1]
            else:
                propensity = self.propensity_model.predict_proba(X[:, :2])[:, 1]
        else:
            propensity = self.propensity_model.predict(X_prop)

        propensity = np.clip(propensity, 0.01, 0.99)

        # 2. 估计结果模型 (可能误设定)
        treated_mask = T == 1
        control_mask = T == 0

        if outcome_correct:
            X_outcome = X
        else:
            # 人为误设定: 只使用部分特征或错误的函数形式
            X_outcome = X[:, :2]

        # mu_1(X)
        self.outcome_model_1.fit(X_outcome[treated_mask], Y[treated_mask])
        if outcome_correct:
            mu_1 = self.outcome_model_1.predict(X)
        else:
            mu_1 = self.outcome_model_1.predict(X[:, :2])

        # mu_0(X)
        self.outcome_model_0.fit(X_outcome[control_mask], Y[control_mask])
        if outcome_correct:
            mu_0 = self.outcome_model_0.predict(X)
        else:
            mu_0 = self.outcome_model_0.predict(X[:, :2])

        # 3. AIPW 估计
        term1 = mu_1 - mu_0
        term2 = T * (Y - mu_1) / propensity
        term3 = (1 - T) * (Y - mu_0) / (1 - propensity)

        aipw_scores = term1 + term2 - term3

        ate = aipw_scores.mean()
        se = aipw_scores.std() / np.sqrt(n)

        return ate, se


def demonstrate_double_robustness(
    n_samples: int = 3000,
    confounding_strength: float = 1.5,
    seed: int = 42
) -> Tuple[go.Figure, str]:
    """
    演示双重稳健性质

    测试四种情况:
    1. 两个模型都正确
    2. 只有倾向得分模型正确
    3. 只有结果模型正确
    4. 两个模型都错误

    Parameters:
    -----------
    n_samples: 样本数
    confounding_strength: 混淆强度
    seed: 随机种子

    Returns:
    --------
    (figure, markdown_report)
    """
    # 生成数据
    df, params = generate_confounded_data(
        n_samples=n_samples,
        confounding_strength=confounding_strength,
        seed=seed
    )

    feature_names = [col for col in df.columns if col.startswith('X')]
    X = df[feature_names].values
    T = df['T'].values
    Y = df['Y'].values

    true_ate = params['true_ate']
    naive_ate = compute_naive_ate(df)

    # 测试四种情况
    scenarios = [
        ('两模型都正确', True, True),
        ('只有倾向得分正确', True, False),
        ('只有结果模型正确', False, True),
        ('两模型都错误', False, False)
    ]

    results = []

    for name, prop_correct, outcome_correct in scenarios:
        dr = DoublyRobustEstimator(model_type='linear')
        ate_est, se_est = dr.estimate_ate(X, T, Y, prop_correct, outcome_correct)

        eval_metrics = evaluate_ate_estimator(ate_est, true_ate, se_est)

        results.append({
            'scenario': name,
            'propensity_correct': prop_correct,
            'outcome_correct': outcome_correct,
            'ate': ate_est,
            'se': se_est,
            'bias': eval_metrics['bias'],
            'ci_lower': eval_metrics['ci_lower'],
            'ci_upper': eval_metrics['ci_upper'],
            'covers_truth': eval_metrics['ci_covers_truth']
        })

    # 可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'ATE 估计对比',
            '估计偏差对比'
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )

    # 1. ATE 估计对比 (带置信区间)
    scenarios_names = [r['scenario'] for r in results]
    ate_estimates = [r['ate'] for r in results]
    ci_lowers = [r['ci_lower'] for r in results]
    ci_uppers = [r['ci_upper'] for r in results]

    colors = ['#27AE60', '#2D9CDB', '#9B59B6', '#EB5757']

    for i, (name, ate, ci_l, ci_u, color) in enumerate(zip(
        scenarios_names, ate_estimates, ci_lowers, ci_uppers, colors
    )):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[ate],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_u - ate],
                arrayminus=[ate - ci_l]
            ),
            mode='markers',
            marker=dict(size=12, color=color),
            name=name,
            showlegend=False
        ), row=1, col=1)

    # 真实 ATE 线
    fig.add_hline(
        y=true_ate,
        line_dash="dash",
        line_color="green",
        annotation_text=f"真实 ATE = {true_ate:.4f}",
        row=1, col=1
    )

    # 2. 偏差对比
    biases = [abs(r['bias']) for r in results]

    fig.add_trace(go.Bar(
        x=scenarios_names,
        y=biases,
        marker_color=colors,
        showlegend=False
    ), row=1, col=2)

    # 更新布局
    fig.update_layout(
        height=500,
        template='plotly_white',
        title_text='双重稳健估计器: 对模型误设定的鲁棒性'
    )

    fig.update_xaxes(ticktext=scenarios_names, tickvals=list(range(len(scenarios_names))), row=1, col=1)
    fig.update_xaxes(tickangle=-30, row=1, col=2)

    fig.update_yaxes(title_text='ATE 估计', row=1, col=1)
    fig.update_yaxes(title_text='|偏差|', row=1, col=2)

    # 生成报告
    report = f"""
### 双重稳健性演示

#### 数据设置
- 样本量: {n_samples}
- 真实 ATE: {true_ate:.4f}
- 朴素估计: {naive_ate:.4f} (偏差: {abs(naive_ate - true_ate):.4f})
- 混淆强度: {confounding_strength}

#### 四种情况下的估计结果

| 场景 | 倾向得分 | 结果模型 | ATE估计 | 标准误 | 偏差 | 95% CI | 覆盖真值 |
|------|----------|----------|---------|--------|------|--------|----------|
"""

    for r in results:
        prop_mark = '✓' if r['propensity_correct'] else '✗'
        outcome_mark = '✓' if r['outcome_correct'] else '✗'
        cover_mark = '✓' if r['covers_truth'] else '✗'

        report += f"| {r['scenario']} | {prop_mark} | {outcome_mark} | "
        report += f"{r['ate']:.4f} | {r['se']:.4f} | {r['bias']:.4f} | "
        report += f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] | {cover_mark} |\n"

    report += f"""

#### 关键洞察

**双重稳健性质**:

1. **两模型都正确**: 估计最优，偏差最小 (偏差: {abs(results[0]['bias']):.4f})

2. **只有倾向得分正确**:
   - 估计仍然一致! (偏差: {abs(results[1]['bias']):.4f})
   - 虽然结果模型错误，但 IPW 项修正了误差

3. **只有结果模型正确**:
   - 估计仍然一致! (偏差: {abs(results[2]['bias']):.4f})
   - 虽然倾向得分错误，但结果模型的预测是对的

4. **两模型都错误**:
   - 估计有偏 (偏差: {abs(results[3]['bias']):.4f})
   - 两个错误的模型无法相互修正

#### 为什么双重稳健有效?

AIPW 估计器的形式:
```
ATE = E[(mu_1(X) - mu_0(X)) + T*(Y - mu_1(X))/e(X) - (1-T)*(Y - mu_0(X))/(1-e(X))]
```

- **结果模型正确时**: 即使 e(X) 错误，E[T*(Y - mu_1(X))/e(X)] ≈ 0 (残差的期望为0)
- **倾向得分正确时**: IPW 项创造伪总体，使得 E[mu_1(X) - mu_0(X)] ≈ ATE

#### 实践建议

1. **优先使用 AIPW**: 相比单独的 IPW 或结果回归，AIPW 提供双重保护
2. **使用灵活模型**: 机器学习模型可以提高至少一个模型正确的概率
3. **检查模型**: 虽然有双重保护，但两个模型都错误时仍会失败
4. **交叉验证**: 使用交叉拟合避免过拟合偏差

#### 效率提升

即使两个模型都正确，AIPW 的方差也通常小于单独的 IPW 或回归估计器。
"""

    return fig, report


def compare_estimators(
    n_samples: int,
    confounding_strength: float
) -> Tuple[go.Figure, str]:
    """
    对比不同估计方法

    Parameters:
    -----------
    n_samples: 样本数
    confounding_strength: 混淆强度

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

    true_ate = params['true_ate']

    # 不同方法
    from .ipw import IPWEstimator, AIPWEstimator

    methods = {}

    # 朴素估计
    naive_ate = compute_naive_ate(df)
    methods['朴素估计'] = (naive_ate, None)

    # IPW
    ipw = IPWEstimator()
    ipw.fit(X, T)
    ipw_ate, ipw_se, _ = ipw.estimate_ate(X, T, Y)
    methods['IPW'] = (ipw_ate, ipw_se)

    # AIPW (双重稳健)
    aipw = AIPWEstimator()
    aipw_ate, aipw_se = aipw.estimate_ate(X, T, Y)
    methods['AIPW'] = (aipw_ate, aipw_se)

    # 结果回归 (线性)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X_with_t = np.column_stack([X, T])
    lr.fit(X_with_t, Y)
    reg_ate = lr.coef_[-1]
    methods['回归调整'] = (reg_ate, None)

    # 可视化
    fig = go.Figure()

    method_names = list(methods.keys())
    estimates = [methods[m][0] for m in method_names]
    ses = [methods[m][1] for m in method_names]

    colors = ['#EB5757', '#F2994A', '#27AE60', '#2D9CDB']

    for i, (name, est, se, color) in enumerate(zip(method_names, estimates, ses, colors)):
        if se is not None:
            fig.add_trace(go.Scatter(
                x=[i],
                y=[est],
                error_y=dict(type='data', array=[1.96*se]),
                mode='markers',
                marker=dict(size=12, color=color),
                name=name
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[i],
                y=[est],
                mode='markers',
                marker=dict(size=12, color=color),
                name=name
            ))

    # 真实 ATE
    fig.add_hline(
        y=true_ate,
        line_dash="dash",
        line_color="green",
        annotation_text=f"真实 ATE = {true_ate:.4f}"
    )

    fig.update_layout(
        title='因果效应估计方法对比',
        xaxis=dict(
            ticktext=method_names,
            tickvals=list(range(len(method_names)))
        ),
        yaxis_title='ATE 估计',
        template='plotly_white',
        height=500,
        showlegend=False
    )

    # 报告
    report = f"""
### 因果效应估计方法对比

| 方法 | 估计值 | 标准误 | 偏差 | 绝对偏差 |
|------|--------|--------|------|----------|
| 真实 ATE | {true_ate:.4f} | - | - | - |
"""

    for name in method_names:
        est, se = methods[name]
        bias = est - true_ate
        report += f"| {name} | {est:.4f} | {se:.4f if se else '-'} | {bias:.4f} | {abs(bias):.4f} |\n"

    report += f"""

#### 方法总结

- **朴素估计**: 简单差分，有混淆偏差
- **回归调整**: 控制协变量，假设线性关系
- **IPW**: 倾向得分加权，依赖倾向得分模型
- **AIPW**: 双重稳健，结合两者优势

**推荐**: 在实践中优先使用 AIPW，它提供双重保护且效率更高。
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 双重稳健估计 (Doubly Robust Estimation)

双重稳健估计器是因果推断中的重要工具，具有独特的鲁棒性。

### 核心性质

**双重稳健性**: 只要倾向得分模型或结果模型有一个正确，估计就是一致的！

### AIPW 估计器

```
ATE = E[(mu_1(X) - mu_0(X)) +
        T*(Y - mu_1(X))/e(X) -
        (1-T)*(Y - mu_0(X))/(1-e(X))]
```

- **第一项**: 结果模型的预测差异
- **第二项**: 处理组的 IPW 修正项
- **第三项**: 控制组的 IPW 修正项

### 为什么叫"双重"稳健?

1. **结果模型正确** → 即使倾向得分错误，IPW 修正项的期望为 0
2. **倾向得分正确** → IPW 创造伪总体，结果模型不需要完美

---
        """)

        with gr.Tab("双重稳健性演示"):
            gr.Markdown("### 测试四种情况: 两模型都对/只有一个对/都错")

            with gr.Row():
                with gr.Column(scale=1):
                    n_samples_dr = gr.Slider(
                        minimum=1000, maximum=5000, value=3000, step=500,
                        label="样本量"
                    )
                    confounding_dr = gr.Slider(
                        minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                        label="混淆强度"
                    )
                    run_dr_btn = gr.Button("演示双重稳健性", variant="primary")

            with gr.Row():
                dr_plot = gr.Plot(label="双重稳健性演示")

            with gr.Row():
                dr_report = gr.Markdown()

            run_dr_btn.click(
                fn=demonstrate_double_robustness,
                inputs=[n_samples_dr, confounding_dr],
                outputs=[dr_plot, dr_report]
            )

        with gr.Tab("方法对比"):
            gr.Markdown("### 对比不同因果推断方法")

            with gr.Row():
                with gr.Column(scale=1):
                    n_samples_comp = gr.Slider(
                        minimum=1000, maximum=5000, value=2000, step=500,
                        label="样本量"
                    )
                    confounding_comp = gr.Slider(
                        minimum=0.5, maximum=3.0, value=1.5, step=0.1,
                        label="混淆强度"
                    )
                    run_comp_btn = gr.Button("对比方法", variant="primary")

            with gr.Row():
                comp_plot = gr.Plot(label="方法对比")

            with gr.Row():
                comp_report = gr.Markdown()

            run_comp_btn.click(
                fn=compare_estimators,
                inputs=[n_samples_comp, confounding_comp],
                outputs=[comp_plot, comp_report]
            )

        gr.Markdown("""
---

### 理论基础

双重稳健性来自于估计方程的特殊结构:

**影响函数**:
```
ψ(O; θ) = (mu_1(X) - mu_0(X)) +
          T*(Y - mu_1(X))/e(X) -
          (1-T)*(Y - mu_0(X))/(1-e(X)) - θ
```

当倾向得分正确时: E[ψ(O; ATE)] = 0
当结果模型正确时: E[ψ(O; ATE)] = 0

### 实践优势

1. **鲁棒性**: 对模型误设定更鲁棒
2. **效率**: 即使两模型都正确，方差也通常更小
3. **灵活性**: 可以使用机器学习模型
4. **可诊断**: 可以检查两个模型的拟合质量

### 现代发展

- **交叉拟合 (Cross-fitting)**: 避免过拟合偏差
- **目标化学习 (Targeted Learning)**: 进一步优化效率
- **去偏机器学习 (Debiased ML)**: 结合高维机器学习

### 实践练习

使用真实数据验证双重稳健性：故意错误指定一个模型，观察估计的鲁棒性。
        """)

    return None
