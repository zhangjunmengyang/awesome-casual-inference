"""
Meta-Learners 模块

实现 S-Learner, T-Learner, X-Learner, R-Learner

核心概念:
- S-Learner: 单一模型，处理作为特征
- T-Learner: 两个独立模型，分别建模处理/控制组
- X-Learner: 利用反事实估计的两阶段方法
- R-Learner: 基于残差的双重稳健方法
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict
from typing import Optional

from .utils import (
    generate_uplift_data,
    calculate_qini_curve,
    calculate_auuc,
    plot_uplift_curves,
    plot_cate_distribution
)


class SLearner:
    """
    S-Learner (Single Model)

    将处理 T 作为特征，训练单一模型:
    Y = f(X, T)

    CATE 估计: tau(x) = f(x, 1) - f(x, 0)
    """

    def __init__(self, base_model=None):
        self.base_model = base_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        X_with_T = np.column_stack([X, T])
        self.model = self.base_model
        self.model.fit(X_with_T, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        n = X.shape[0]

        # 预测 Y(1) 和 Y(0)
        X_1 = np.column_stack([X, np.ones(n)])
        X_0 = np.column_stack([X, np.zeros(n)])

        Y_1 = self.model.predict(X_1)
        Y_0 = self.model.predict(X_0)

        return Y_1 - Y_0


class TLearner:
    """
    T-Learner (Two Models)

    分别为处理组和控制组训练模型:
    - mu_0(x) = E[Y|X=x, T=0]
    - mu_1(x) = E[Y|X=x, T=1]

    CATE 估计: tau(x) = mu_1(x) - mu_0(x)
    """

    def __init__(self, base_model_0=None, base_model_1=None):
        self.model_0 = base_model_0 or RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_1 = base_model_1 or RandomForestRegressor(n_estimators=100, random_state=43)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        # 控制组模型
        mask_0 = T == 0
        self.model_0.fit(X[mask_0], Y[mask_0])

        # 处理组模型
        mask_1 = T == 1
        self.model_1.fit(X[mask_1], Y[mask_1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        Y_0 = self.model_0.predict(X)
        Y_1 = self.model_1.predict(X)

        return Y_1 - Y_0


class XLearner:
    """
    X-Learner

    两阶段方法:
    阶段 1: 分别估计 mu_0 和 mu_1 (同 T-Learner)
    阶段 2: 估计伪处理效应
        - D_1 = Y_1 - mu_0(X_1)  (处理组的反事实)
        - D_0 = mu_1(X_0) - Y_0  (控制组的反事实)

    训练 tau_1(x) ~ D_1 和 tau_0(x) ~ D_0
    最终: tau(x) = g(x) * tau_0(x) + (1-g(x)) * tau_1(x)

    其中 g(x) = P(T=1|X=x) 是倾向得分
    """

    def __init__(self, outcome_model=None, effect_model=None, propensity_model=None):
        self.model_0 = outcome_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_1 = outcome_model or RandomForestRegressor(n_estimators=100, random_state=43)
        self.tau_0 = effect_model or RandomForestRegressor(n_estimators=100, random_state=44)
        self.tau_1 = effect_model or RandomForestRegressor(n_estimators=100, random_state=45)
        self.propensity = propensity_model or LogisticRegression(random_state=42)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        mask_0 = T == 0
        mask_1 = T == 1

        # 阶段 1: 结果模型
        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])

        # 阶段 2: 伪处理效应
        # 处理组: D_1 = Y - mu_0(X)
        D_1 = Y[mask_1] - self.model_0.predict(X[mask_1])
        self.tau_1.fit(X[mask_1], D_1)

        # 控制组: D_0 = mu_1(X) - Y
        D_0 = self.model_1.predict(X[mask_0]) - Y[mask_0]
        self.tau_0.fit(X[mask_0], D_0)

        # 倾向得分
        self.propensity.fit(X, T)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        tau_0_pred = self.tau_0.predict(X)
        tau_1_pred = self.tau_1.predict(X)

        # 倾向得分作为权重
        g = self.propensity.predict_proba(X)[:, 1]

        # 加权组合
        return g * tau_0_pred + (1 - g) * tau_1_pred


class RLearner:
    """
    R-Learner (Residual/Robinson Learner)

    基于 Robinson 分解的双重稳健方法:
    Y - m(X) = tau(X) * (T - e(X)) + epsilon

    其中:
    - m(x) = E[Y|X]
    - e(x) = P(T=1|X) 倾向得分

    通过最小化加权残差来估计 tau
    """

    def __init__(self, outcome_model=None, propensity_model=None, effect_model=None):
        self.outcome_model = outcome_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.propensity_model = propensity_model or LogisticRegression(random_state=42)
        self.effect_model = effect_model or GradientBoostingRegressor(n_estimators=100, random_state=42)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        # 估计 m(x) = E[Y|X]
        m_hat = cross_val_predict(self.outcome_model, X, Y, cv=5)

        # 估计 e(x) = P(T=1|X)
        e_hat = cross_val_predict(self.propensity_model, X, T, cv=5, method='predict_proba')[:, 1]

        # 计算残差
        Y_residual = Y - m_hat
        T_residual = T - e_hat

        # 避免除零
        T_residual = np.clip(T_residual, -0.9, 0.9)

        # 伪结果
        pseudo_outcome = Y_residual / T_residual

        # 加权回归
        weights = T_residual ** 2

        self.effect_model.fit(X, pseudo_outcome, sample_weight=weights)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        return self.effect_model.predict(X)


def compare_meta_learners(
    n_samples: int,
    effect_type: str,
    noise_level: float
) -> tuple:
    """比较不同 Meta-Learner 的性能"""

    # 生成数据
    df, true_cate = generate_uplift_data(
        n_samples=n_samples,
        n_features=5,
        treatment_effect_type=effect_type,
        noise_level=noise_level
    )

    X = df[[f'X{i+1}' for i in range(5)]].values
    T = df['T'].values
    Y = df['Y'].values

    # 训练模型
    models = {
        'S-Learner': SLearner(),
        'T-Learner': TLearner(),
        'X-Learner': XLearner(),
        # 'R-Learner': RLearner()  # R-Learner 可能较慢
    }

    predictions = {}
    metrics = []

    for name, model in models.items():
        model.fit(X, T, Y)
        pred = model.predict(X)
        predictions[name] = pred

        # 计算指标
        mse = np.mean((pred - true_cate) ** 2)
        corr = np.corrcoef(pred, true_cate)[0, 1]
        auuc = calculate_auuc(Y, T, pred)

        metrics.append({
            'Model': name,
            'MSE': mse,
            'Correlation': corr,
            'AUUC': auuc
        })

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'CATE Distribution',
            'True vs Predicted (S-Learner)',
            'True vs Predicted (T-Learner)',
            'Qini Curves'
        )
    )

    colors = {'S-Learner': '#2D9CDB', 'T-Learner': '#27AE60', 'X-Learner': '#EB5757'}

    # 1. CATE 分布
    fig.add_trace(go.Histogram(
        x=true_cate, name='True CATE',
        marker_color='gray', opacity=0.7, nbinsx=30
    ), row=1, col=1)

    for name, pred in predictions.items():
        fig.add_trace(go.Histogram(
            x=pred, name=name,
            marker_color=colors.get(name, 'blue'),
            opacity=0.4, nbinsx=30
        ), row=1, col=1)

    # 2-3. 散点图
    sample_idx = np.random.choice(len(true_cate), min(500, len(true_cate)), replace=False)

    fig.add_trace(go.Scatter(
        x=true_cate[sample_idx], y=predictions['S-Learner'][sample_idx],
        mode='markers', marker=dict(color='#2D9CDB', size=4, opacity=0.5),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=true_cate[sample_idx], y=predictions['T-Learner'][sample_idx],
        mode='markers', marker=dict(color='#27AE60', size=4, opacity=0.5),
        showlegend=False
    ), row=2, col=1)

    # 对角线
    for row, col in [(1, 2), (2, 1)]:
        fig.add_trace(go.Scatter(
            x=[true_cate.min(), true_cate.max()],
            y=[true_cate.min(), true_cate.max()],
            mode='lines', line=dict(color='gray', dash='dash'),
            showlegend=False
        ), row=row, col=col)

    # 4. Qini Curves
    for name, pred in predictions.items():
        fraction, qini = calculate_qini_curve(Y, T, pred)
        fig.add_trace(go.Scatter(
            x=fraction, y=qini, mode='lines',
            name=f'{name} Qini',
            line=dict(color=colors.get(name, 'blue'), width=2)
        ), row=2, col=2)

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, qini[-1]], mode='lines',
        name='Random', line=dict(color='gray', dash='dash')
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='Meta-Learners Comparison',
        barmode='overlay'
    )

    # 指标摘要
    metrics_df = pd.DataFrame(metrics)
    metrics_md = f"""
### 模型性能对比

{metrics_df.to_markdown(index=False)}

### 关键洞察

- **MSE**: 均方误差，越小越好
- **Correlation**: 与真实 CATE 的相关性，越高越好
- **AUUC**: Qini 曲线下面积，越大越好

### 不同效应类型的影响

- **constant**: 所有模型表现相似
- **heterogeneous**: T-Learner 和 X-Learner 通常更好
- **complex**: X-Learner 通常最优
    """

    return fig, metrics_md


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## Meta-Learners: S/T/X-Learner

Meta-Learner 是一类使用标准机器学习模型估计 CATE 的方法。

### 方法对比

| 方法 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **S-Learner** | 单一模型，T 作为特征 | 简单，样本效率高 | 可能忽略处理效应 |
| **T-Learner** | 两个独立模型 | 直观，灵活 | 样本效率低，不一致 |
| **X-Learner** | 利用反事实，加权组合 | 在不平衡时表现好 | 较复杂 |
| **R-Learner** | 残差回归，双重稳健 | 理论保证好 | 计算较慢 |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=1000, maximum=10000, value=3000, step=500,
                    label="样本量"
                )
                effect_type = gr.Radio(
                    choices=['constant', 'heterogeneous', 'complex'],
                    value='heterogeneous',
                    label="处理效应类型"
                )
                noise_level = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                    label="噪声水平"
                )
                run_btn = gr.Button("运行对比", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="Meta-Learners 对比")

        with gr.Row():
            metrics_output = gr.Markdown()

        run_btn.click(
            fn=compare_meta_learners,
            inputs=[n_samples, effect_type, noise_level],
            outputs=[plot_output, metrics_output]
        )

        gr.Markdown("""
---

### 算法细节

**S-Learner**:
```
Y = f(X, T)
tau(x) = f(x, 1) - f(x, 0)
```

**T-Learner**:
```
mu_0(x) = E[Y|X=x, T=0]  (控制组模型)
mu_1(x) = E[Y|X=x, T=1]  (处理组模型)
tau(x) = mu_1(x) - mu_0(x)
```

**X-Learner**:
```
Stage 1: 估计 mu_0, mu_1
Stage 2: D_1 = Y - mu_0(X), D_0 = mu_1(X) - Y
         tau_1 ~ D_1, tau_0 ~ D_0
Final:   tau(x) = e(x)*tau_0(x) + (1-e(x))*tau_1(x)
```

### 练习

完成 `exercises/chapter3_uplift/ex1_meta_learners.py` 中的练习。
        """)

    return None
