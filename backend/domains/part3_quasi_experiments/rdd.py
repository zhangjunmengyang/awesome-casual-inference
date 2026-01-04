"""断点回归 (Regression Discontinuity Design) 分析"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

from .utils import generate_rdd_data, fig_to_chart_data


class SharpRDD:
    """Sharp RDD估计器（局部多项式回归）"""

    def __init__(self, cutoff: float = 0, bandwidth: float = None, polynomial_order: int = 1):
        """初始化

        Args:
            cutoff: 门槛值
            bandwidth: 带宽（None表示自动选择）
            polynomial_order: 多项式阶数
        """
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.polynomial_order = polynomial_order
        self.tau_ = None
        self.se_ = None

    def _select_bandwidth_ik(self, X, Y, D):
        """Imbens-Kalyanaraman (2012) 带宽选择（简化版）"""
        X_centered = X - self.cutoff

        # 分别拟合左右两侧
        left_mask = D == 0
        right_mask = D == 1

        # 估计方差
        var_left = np.var(Y[left_mask]) if left_mask.sum() > 0 else 1.0
        var_right = np.var(Y[right_mask]) if right_mask.sum() > 0 else 1.0

        # 简化公式
        n = len(X)
        range_x = np.max(X) - np.min(X)

        # 经验公式
        h_ik = 1.84 * np.sqrt(var_left + var_right) * n**(-1/5) * range_x / 10

        return h_ik

    def fit(self, X, Y, D=None):
        """拟合RDD模型

        Args:
            X: 驱动变量 (n,)
            Y: 结果变量 (n,)
            D: 处理状态 (n,), 如果为None则根据cutoff自动生成
        """
        X = np.array(X)
        Y = np.array(Y)

        if D is None:
            D = (X >= self.cutoff).astype(int)
        else:
            D = np.array(D)

        # 自动选择带宽
        if self.bandwidth is None:
            self.bandwidth = self._select_bandwidth_ik(X, Y, D)

        # 筛选带宽内的样本
        mask = np.abs(X - self.cutoff) <= self.bandwidth
        X_bw = X[mask]
        Y_bw = Y[mask]
        D_bw = D[mask]

        if len(X_bw) < 10:
            raise ValueError(f"带宽内样本量过少: {len(X_bw)}")

        # 构建特征矩阵
        X_centered = (X_bw - self.cutoff).reshape(-1, 1)

        # 特征矩阵: [1, D, X-c, D*(X-c)]
        features = [np.ones(len(X_bw)), D_bw, X_centered.flatten(), D_bw * X_centered.flatten()]
        X_design = np.column_stack(features)

        # 最小二乘
        model = LinearRegression(fit_intercept=False)
        model.fit(X_design, Y_bw)

        # 提取处理效应（第二个系数）
        self.tau_ = model.coef_[1]

        # 标准误（简化）
        residuals = Y_bw - model.predict(X_design)
        sigma2 = np.sum(residuals**2) / (len(Y_bw) - len(features))
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        self.se_ = np.sqrt(sigma2 * XtX_inv[1, 1])

        return self

    def summary(self):
        """输出估计结果"""
        if self.tau_ is None:
            raise ValueError("模型未拟合")

        z = 1.96
        ci_lower = self.tau_ - z * self.se_
        ci_upper = self.tau_ + z * self.se_
        t_stat = self.tau_ / self.se_
        p_value = 2 * (1 - 0.975) if abs(t_stat) > z else 0.5

        return {
            'tau': self.tau_,
            'se': self.se_,
            'ci': (ci_lower, ci_upper),
            'p_value': p_value,
            'bandwidth': self.bandwidth
        }


def analyze_rdd_sharp(
    n_samples: int = 500,
    bandwidth: float = 50.0,
    cutoff: float = 200.0,
    effect_size: float = 15.0
) -> dict:
    """Sharp RDD分析

    Args:
        n_samples: 样本量
        bandwidth: 带宽
        cutoff: 门槛值
        effect_size: 处理效应大小

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成数据
    df = generate_rdd_data(
        n_samples=n_samples,
        cutoff=cutoff,
        treatment_effect=effect_size,
        noise_std=20.0
    )

    # 估计RDD
    rdd = SharpRDD(cutoff=cutoff, bandwidth=bandwidth, polynomial_order=1)
    rdd.fit(df['running_var'], df['outcome'])
    results = rdd.summary()

    # 创建可视化
    fig = go.Figure()

    # 散点图
    fig.add_trace(go.Scatter(
        x=df[df['treatment'] == 0]['running_var'],
        y=df[df['treatment'] == 0]['outcome'],
        mode='markers',
        name='未处理 (X < cutoff)',
        marker=dict(color='#EB5757', size=5, opacity=0.5)
    ))

    fig.add_trace(go.Scatter(
        x=df[df['treatment'] == 1]['running_var'],
        y=df[df['treatment'] == 1]['outcome'],
        mode='markers',
        name='处理 (X ≥ cutoff)',
        marker=dict(color='#27AE60', size=5, opacity=0.5)
    ))

    # 拟合线（分段线性）
    left = df[df['running_var'] < cutoff]
    right = df[df['running_var'] >= cutoff]

    if len(left) > 0:
        X_left = left['running_var'].values.reshape(-1, 1)
        y_left = left['outcome'].values
        model_left = LinearRegression().fit(X_left, y_left)
        x_left_line = np.linspace(df['running_var'].min(), cutoff, 100).reshape(-1, 1)
        y_left_pred = model_left.predict(x_left_line)

        fig.add_trace(go.Scatter(
            x=x_left_line.flatten(),
            y=y_left_pred,
            mode='lines',
            name='左侧拟合',
            line=dict(color='#EB5757', width=3)
        ))

    if len(right) > 0:
        X_right = right['running_var'].values.reshape(-1, 1)
        y_right = right['outcome'].values
        model_right = LinearRegression().fit(X_right, y_right)
        x_right_line = np.linspace(cutoff, df['running_var'].max(), 100).reshape(-1, 1)
        y_right_pred = model_right.predict(x_right_line)

        fig.add_trace(go.Scatter(
            x=x_right_line.flatten(),
            y=y_right_pred,
            mode='lines',
            name='右侧拟合',
            line=dict(color='#27AE60', width=3)
        ))

    # 门槛线
    fig.add_vline(
        x=cutoff,
        line_dash="dash",
        line_color="black",
        annotation_text=f"门槛: {cutoff}"
    )

    fig.update_layout(
        title='Sharp RDD：门槛处的跳跃即为因果效应',
        xaxis_title='驱动变量 (Running Variable)',
        yaxis_title='结果变量',
        template='plotly_white',
        hovermode='closest',
        height=500
    )

    summary = f"""
## Sharp RDD 分析结果

### 核心思想

RDD利用政策/规则产生的**门槛**来识别因果效应:
- 在门槛附近，个体特征是连续的
- 只有处理状态发生跳跃
- 门槛处的跳跃即为因果效应（局部）

### 关键指标

| 指标 | 值 |
|------|-----|
| 门槛值 | {cutoff} |
| 带宽 | {results['bandwidth']:.2f} |
| **RDD估计量** | **{results['tau']:.2f}** |
| 标准误 | {results['se']:.2f} |
| 95% 置信区间 | [{results['ci'][0]:.2f}, {results['ci'][1]:.2f}] |
| p值 | {results['p_value']:.4f} |
| 真实效应 | {effect_size:.2f} |

### 关键假设

1. **连续性假设**: 如果没有处理，结果在门槛处应该连续
2. **不可操纵**: 个体无法精确控制驱动变量跨越门槛
3. **局部效应**: 只能估计门槛附近的效应（LATE）

### 方法要点

- **带宽选择**: 平衡偏差和方差
  - 小带宽：低偏差，高方差
  - 大带宽：高偏差，低方差
- **稳健性检验**:
  - McCrary密度检验（操纵检验）
  - 协变量连续性检验
  - Placebo检验

### 业务应用

Sharp RDD适用于:
- 满减活动（满200减50）
- 会员等级（消费5000升金卡）
- 考试及格线（60分及格）
- 信用评分门槛（600分以上低利率）
"""

    return {
        "charts": [fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "tau": float(results['tau']),
            "se": float(results['se']),
            "ci_lower": float(results['ci'][0]),
            "ci_upper": float(results['ci'][1]),
            "p_value": float(results['p_value']),
            "bandwidth": float(results['bandwidth']),
            "true_effect": float(effect_size),
            "cutoff": float(cutoff)
        }
    }


def analyze_rdd_fuzzy(
    n_samples: int = 1000,
    bandwidth: float = 50.0,
    cutoff: float = 200.0,
    compliance_rate: float = 0.7
) -> dict:
    """Fuzzy RDD分析

    Args:
        n_samples: 样本量
        bandwidth: 带宽
        cutoff: 门槛值
        compliance_rate: 顺从率

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成Fuzzy RDD数据
    np.random.seed(42)

    # 驱动变量
    running_var = np.random.uniform(100, 300, n_samples)

    # 工具变量（是否超过门槛）
    eligible = (running_var >= cutoff).astype(int)

    # 处理状态（不完全顺从）
    treatment = np.zeros(n_samples)
    treatment[eligible == 1] = np.random.binomial(1, compliance_rate, (eligible == 1).sum())
    treatment[eligible == 0] = np.random.binomial(1, 0.1, (eligible == 0).sum())

    # 结果变量
    true_effect = 15.0
    y0 = 30 + 0.1 * (running_var - cutoff) + np.random.normal(0, 20, n_samples)
    y1 = y0 + true_effect
    outcome = treatment * y1 + (1 - treatment) * y0

    df = pd.DataFrame({
        'running_var': running_var,
        'eligible': eligible,
        'treatment': treatment,
        'outcome': outcome
    })

    # 带宽内样本
    mask_bw = np.abs(df['running_var'] - cutoff) <= bandwidth
    df_bw = df[mask_bw].copy()

    # 2SLS估计
    # 第一阶段：eligible -> treatment
    X_first = df_bw[['eligible']].values
    D_bw = df_bw['treatment'].values
    first_stage = LinearRegression().fit(X_first, D_bw)
    D_hat = first_stage.predict(X_first)

    # 第二阶段：D_hat -> outcome
    X_second = D_hat.reshape(-1, 1)
    Y_bw = df_bw['outcome'].values
    second_stage = LinearRegression().fit(X_second, Y_bw)
    tau_fuzzy = second_stage.coef_[0]

    # Wald估计（等价）
    left_mask = df_bw['running_var'] < cutoff
    right_mask = df_bw['running_var'] >= cutoff

    # 结果的跳跃（Reduced Form）
    if right_mask.sum() > 0 and left_mask.sum() > 0:
        reduced_form = df_bw[right_mask]['outcome'].mean() - df_bw[left_mask]['outcome'].mean()
        # 处理的跳跃（First Stage）
        first_stage_jump = df_bw[right_mask]['treatment'].mean() - df_bw[left_mask]['treatment'].mean()
        tau_wald = reduced_form / first_stage_jump if first_stage_jump != 0 else 0
    else:
        reduced_form = 0
        first_stage_jump = 0
        tau_wald = 0

    # 创建可视化
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('第一阶段：处理概率跳跃', '约化式：结果跳跃')
    )

    # 左图：处理概率
    bins_x = np.linspace(100, 300, 30)
    bin_centers = (bins_x[:-1] + bins_x[1:]) / 2
    treatment_prob = []

    for i in range(len(bins_x) - 1):
        mask = (df['running_var'] >= bins_x[i]) & (df['running_var'] < bins_x[i+1])
        if mask.sum() > 0:
            treatment_prob.append(df[mask]['treatment'].mean())
        else:
            treatment_prob.append(np.nan)

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=treatment_prob,
            mode='markers',
            marker=dict(size=8, color='#2D9CDB'),
            name='处理概率'
        ),
        row=1, col=1
    )

    fig.add_vline(x=cutoff, line_dash="dash", line_color="black", row=1, col=1)

    # 右图：结果跳跃
    avg_outcome = []
    for i in range(len(bins_x) - 1):
        mask = (df['running_var'] >= bins_x[i]) & (df['running_var'] < bins_x[i+1])
        if mask.sum() > 0:
            avg_outcome.append(df[mask]['outcome'].mean())
        else:
            avg_outcome.append(np.nan)

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=avg_outcome,
            mode='markers',
            marker=dict(size=8, color='#27AE60'),
            name='平均结果'
        ),
        row=1, col=2
    )

    fig.add_vline(x=cutoff, line_dash="dash", line_color="black", row=1, col=2)

    fig.update_xaxes(title_text="驱动变量", row=1, col=1)
    fig.update_xaxes(title_text="驱动变量", row=1, col=2)
    fig.update_yaxes(title_text="处理概率", row=1, col=1)
    fig.update_yaxes(title_text="平均结果", row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False,
        title_text='Fuzzy RDD：门槛处处理概率和结果的跳跃'
    )

    summary = f"""
## Fuzzy RDD 分析结果

### 核心思想

Fuzzy RDD中，门槛不是完全决定处理，而是影响处理概率:
- 门槛处处理概率跳跃，但不是0→1
- 本质上是**工具变量**设计
- 估计的是**LATE**（顺从者的效应）

### 关键指标

| 指标 | 值 |
|------|-----|
| 处理概率跳跃 | {first_stage_jump:.3f} |
| 结果跳跃（Reduced Form） | {reduced_form:.3f} |
| **2SLS估计** | **{tau_fuzzy:.3f}** |
| **Wald估计** | **{tau_wald:.3f}** |
| 真实效应 | {true_effect:.2f} |
| 顺从率设定 | {compliance_rate:.1%} |

### Fuzzy vs Sharp

| 维度 | Sharp RDD | Fuzzy RDD |
|------|-----------|-----------|
| 处理跳跃 | 0 → 1（完全） | 部分跳跃 |
| 估计方法 | 局部回归 | 2SLS / IV |
| 估计对象 | 门槛处ATE | LATE（顺从者） |
| 适用场景 | 规则严格执行 | 存在不顺从 |

### LATE解释

Fuzzy RDD估计的是**顺从者**的效应:
- **顺从者**: 超过门槛就接受处理，低于门槛就不接受
- **Always-takers**: 无论如何都接受处理
- **Never-takers**: 无论如何都不接受处理

### 业务应用

Fuzzy RDD适用于:
- 满减活动（有人忘记用券）
- 会员等级（有人达标但未升级）
- 考试及格（有人及格但未申请奖学金）
"""

    return {
        "charts": [fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "first_stage_jump": float(first_stage_jump),
            "reduced_form": float(reduced_form),
            "tau_2sls": float(tau_fuzzy),
            "tau_wald": float(tau_wald),
            "true_effect": float(true_effect),
            "compliance_rate": float(compliance_rate)
        }
    }
