"""
倾向得分方法模块

实现倾向得分匹配 (Propensity Score Matching, PSM)

核心概念:
- 倾向得分: e(X) = P(T=1|X) - 接受处理的概率
- Rosenbaum & Rubin (1983): 在倾向得分上条件化可以平衡协变量
- 匹配: 为每个处理组个体找到倾向得分相似的控制组个体
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional

from .utils import (
    generate_confounded_data,
    compute_ate_oracle,
    compute_naive_ate,
    compute_smd,
    compute_variance_ratio
)


class PropensityScoreEstimator:
    """
    倾向得分估计器

    使用逻辑回归估计 P(T=1|X)
    """

    def __init__(self, model=None):
        """
        Parameters:
        -----------
        model: sklearn分类器，默认为LogisticRegression
        """
        self.model = model or LogisticRegression(max_iter=1000, random_state=42)
        self.propensity = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """训练倾向得分模型"""
        self.model.fit(X, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测倾向得分"""
        self.propensity = self.model.predict_proba(X)[:, 1]
        return self.propensity

    def fit_predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """训练并预测"""
        self.fit(X, T)
        return self.predict(X)


class PropensityScoreMatching:
    """
    倾向得分匹配 (PSM)

    方法:
    - 1:1 最近邻匹配
    - 卡尺匹配 (Caliper matching)
    - 有放回/无放回匹配
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: Optional[float] = None,
        replace: bool = False
    ):
        """
        Parameters:
        -----------
        n_neighbors: 匹配的邻居数量
        caliper: 卡尺宽度 (最大允许的倾向得分差异)
        replace: 是否有放回匹配
        """
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replace = replace
        self.matches = None

    def match(
        self,
        propensity: np.ndarray,
        treatment: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行倾向得分匹配

        Parameters:
        -----------
        propensity: 倾向得分
        treatment: 处理状态

        Returns:
        --------
        (treated_indices, control_indices)
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        # 使用 KNN 进行匹配
        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        knn.fit(propensity[control_idx].reshape(-1, 1))

        # 为每个处理组个体找到最近的控制组个体
        distances, indices = knn.kneighbors(
            propensity[treated_idx].reshape(-1, 1)
        )

        # 应用卡尺
        matched_treated = []
        matched_control = []

        for i, t_idx in enumerate(treated_idx):
            for j in range(self.n_neighbors):
                dist = distances[i, j]

                # 检查卡尺约束
                if self.caliper is None or dist <= self.caliper:
                    c_idx = control_idx[indices[i, j]]
                    matched_treated.append(t_idx)
                    matched_control.append(c_idx)

        self.matches = (np.array(matched_treated), np.array(matched_control))

        return self.matches

    def estimate_ate(self, Y: np.ndarray) -> Tuple[float, float]:
        """
        估计 ATE

        Parameters:
        -----------
        Y: 结果变量

        Returns:
        --------
        (ATE估计, 标准误)
        """
        if self.matches is None:
            raise ValueError("Must call match() first")

        treated_idx, control_idx = self.matches

        if len(treated_idx) == 0:
            return 0.0, 0.0

        treated_outcomes = Y[treated_idx]
        control_outcomes = Y[control_idx]

        # 当 n_neighbors > 1 时，需要按配对计算
        # 每个处理组样本可能有多个控制组匹配
        # 使用分组聚合的方式计算正确的 ATE
        if self.n_neighbors == 1:
            # 1:1 匹配，直接计算配对差异
            pair_diffs = treated_outcomes - control_outcomes
            ate = pair_diffs.mean()
            # 配对 t-test 标准误
            se = pair_diffs.std() / np.sqrt(len(pair_diffs))
        else:
            # n_neighbors > 1 时，每个处理样本有多个对照匹配
            # 按处理样本分组，取对照组均值后再计算差异
            unique_treated = np.unique(treated_idx)
            pair_diffs = []
            for t_idx in unique_treated:
                mask = treated_idx == t_idx
                y_t = Y[t_idx]
                y_c_mean = Y[control_idx[mask]].mean()
                pair_diffs.append(y_t - y_c_mean)
            pair_diffs = np.array(pair_diffs)
            ate = pair_diffs.mean()
            se = pair_diffs.std() / np.sqrt(len(pair_diffs))

        return ate, se


def visualize_matching(
    df: pd.DataFrame,
    propensity: np.ndarray,
    matched_treated: np.ndarray,
    matched_control: np.ndarray,
    feature_names: list
) -> go.Figure:
    """
    可视化匹配前后的协变量平衡

    Parameters:
    -----------
    df: 原始数据
    propensity: 倾向得分
    matched_treated: 匹配的处理组索引
    matched_control: 匹配的控制组索引
    feature_names: 特征名称列表

    Returns:
    --------
    Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '倾向得分分布 (匹配前)',
            '倾向得分分布 (匹配后)',
            '协变量平衡: SMD (匹配前)',
            '协变量平衡: SMD (匹配后)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. 匹配前倾向得分分布
    treated_mask = df['T'] == 1
    control_mask = df['T'] == 0

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

    # 2. 匹配后倾向得分分布
    fig.add_trace(go.Histogram(
        x=propensity[matched_control],
        name='控制组 (匹配)',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=30,
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=propensity[matched_treated],
        name='处理组 (匹配)',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=30,
        showlegend=False
    ), row=1, col=2)

    # 3. SMD 匹配前
    X = df[feature_names].values
    X_t_before = X[treated_mask]
    X_c_before = X[control_mask]
    smd_before = compute_smd(X_t_before, X_c_before)

    fig.add_trace(go.Bar(
        x=feature_names,
        y=np.abs(smd_before),
        name='匹配前',
        marker_color='#F2994A'
    ), row=2, col=1)

    # 阈值线 (0.1 是常用的平衡阈值)
    fig.add_hline(y=0.1, row=2, col=1, line_dash="dash",
                  line_color="green", annotation_text="良好平衡阈值 (0.1)")

    # 4. SMD 匹配后
    X_t_after = X[matched_treated]
    X_c_after = X[matched_control]
    smd_after = compute_smd(X_t_after, X_c_after)

    fig.add_trace(go.Bar(
        x=feature_names,
        y=np.abs(smd_after),
        name='匹配后',
        marker_color='#27AE60',
        showlegend=False
    ), row=2, col=2)

    fig.add_hline(y=0.1, row=2, col=2, line_dash="dash",
                  line_color="green", annotation_text="良好平衡阈值 (0.1)")

    # 更新布局
    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='倾向得分匹配: 平衡性诊断',
        barmode='overlay'
    )

    fig.update_xaxes(title_text='倾向得分', row=1, col=1)
    fig.update_xaxes(title_text='倾向得分', row=1, col=2)
    fig.update_xaxes(title_text='特征', row=2, col=1)
    fig.update_xaxes(title_text='特征', row=2, col=2)

    fig.update_yaxes(title_text='频数', row=1, col=1)
    fig.update_yaxes(title_text='频数', row=1, col=2)
    fig.update_yaxes(title_text='标准化均值差 (SMD)', row=2, col=1)
    fig.update_yaxes(title_text='标准化均值差 (SMD)', row=2, col=2)

    return fig


def run_psm_demo(
    n_samples: int,
    confounding_strength: float,
    caliper: float,
    use_caliper: bool
) -> Tuple[go.Figure, str]:
    """
    运行 PSM 演示

    Parameters:
    -----------
    n_samples: 样本数
    confounding_strength: 混淆强度
    caliper: 卡尺宽度
    use_caliper: 是否使用卡尺

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

    # 估计倾向得分
    ps_model = PropensityScoreEstimator()
    propensity = ps_model.fit_predict(X, T)

    # PSM 匹配
    psm = PropensityScoreMatching(
        n_neighbors=1,
        caliper=caliper if use_caliper else None,
        replace=False
    )

    matched_treated, matched_control = psm.match(propensity, T)

    # 估计 ATE
    psm_ate, psm_se = psm.estimate_ate(Y)

    # 可视化
    fig = visualize_matching(
        df, propensity, matched_treated, matched_control, feature_names
    )

    # 生成报告
    n_matched = len(matched_treated)
    n_treated = (T == 1).sum()
    match_rate = n_matched / n_treated * 100

    report = f"""
### 倾向得分匹配 (PSM) 结果

#### 匹配统计
| 指标 | 值 |
|------|-----|
| 总样本数 | {n_samples} |
| 处理组样本 | {n_treated} |
| 成功匹配 | {n_matched} ({match_rate:.1f}%) |
| 控制组样本 | {(T == 0).sum()} |
| 使用卡尺 | {'是 (宽度=' + str(caliper) + ')' if use_caliper else '否'} |

#### ATE 估计
| 方法 | 估计值 | 标准误 | 偏差 |
|------|--------|--------|------|
| 真实 ATE | {true_ate:.4f} | - | - |
| 朴素估计 | {naive_ate:.4f} | - | {naive_ate - true_ate:.4f} |
| PSM | {psm_ate:.4f} | {psm_se:.4f} | {psm_ate - true_ate:.4f} |

#### 关键洞察

- **朴素估计偏差**: {abs(naive_ate - true_ate):.4f} - 由于混淆导致的偏差
- **PSM 偏差**: {abs(psm_ate - true_ate):.4f} - 匹配后的偏差
- **偏差降低**: {(abs(naive_ate - true_ate) - abs(psm_ate - true_ate)):.4f}

PSM 通过平衡协变量来减少混淆偏差。观察 SMD 图，匹配后的 SMD 应该接近 0。

#### 倾向得分匹配的假设

1. **强可忽略性** (Strong Ignorability): 给定协变量 X，处理分配与潜在结果独立
2. **共同支撑** (Common Support): 处理组和控制组的倾向得分分布有重叠
3. **SUTVA**: 稳定单元处理值假设
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 倾向得分匹配 (Propensity Score Matching, PSM)

倾向得分是个体接受处理的概率: **e(X) = P(T=1|X)**

### 核心思想

Rosenbaum & Rubin (1983) 证明: **在倾向得分上条件化可以平衡协变量分布**

即: 如果 (Y(0), Y(1)) ⊥ T | X，则 (Y(0), Y(1)) ⊥ T | e(X)

### PSM 步骤

1. **估计倾向得分**: 使用逻辑回归估计 e(X) = P(T=1|X)
2. **匹配**: 为每个处理组个体找到倾向得分相近的控制组个体
3. **估计 ATE**: 计算匹配样本中的平均处理效应

### 匹配方法

- **最近邻匹配**: 找倾向得分最接近的个体
- **卡尺匹配**: 只匹配倾向得分差异小于卡尺宽度的个体
- **有放回/无放回**: 控制组个体是否可以被多次匹配

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
                use_caliper = gr.Checkbox(
                    value=False,
                    label="使用卡尺匹配"
                )
                caliper = gr.Slider(
                    minimum=0.01, maximum=0.5, value=0.1, step=0.01,
                    label="卡尺宽度 (倾向得分差异)"
                )
                run_btn = gr.Button("运行 PSM", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="PSM 平衡性诊断")

        with gr.Row():
            report_output = gr.Markdown()

        run_btn.click(
            fn=run_psm_demo,
            inputs=[n_samples, confounding_strength, caliper, use_caliper],
            outputs=[plot_output, report_output]
        )

        gr.Markdown("""
---

### 评估平衡性

**标准化均值差 (SMD)**:
```
SMD = (mean_treated - mean_control) / pooled_std
```

一般认为 |SMD| < 0.1 表示良好平衡。

### PSM 的优缺点

**优点**:
- 直观易懂
- 非参数方法，不依赖函数形式
- 可以检查共同支撑假设

**缺点**:
- 需要足够的重叠区域
- 匹配质量依赖倾向得分模型
- 可能丢失未匹配样本，损失效率
- 只能控制观测到的混淆

### 实践练习

尝试使用 scikit-learn 实现自己的倾向得分匹配方法，并与本模块对比结果。
        """)

    return None
