"""工具变量 (Instrumental Variables) 分析"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from .utils import generate_iv_data, fig_to_chart_data


def two_stage_least_squares(Z, X, Y):
    """两阶段最小二乘估计

    Args:
        Z: 工具变量 (n,)
        X: 内生处理变量 (n,)
        Y: 结果变量 (n,)

    Returns:
        包含2SLS估计结果的字典
    """
    # 第一阶段：Z -> X
    first_stage = LinearRegression()
    first_stage.fit(Z.reshape(-1, 1), X)
    X_hat = first_stage.predict(Z.reshape(-1, 1))

    # 第二阶段：X_hat -> Y
    second_stage = LinearRegression()
    second_stage.fit(X_hat.reshape(-1, 1), Y)
    beta_2sls = second_stage.coef_[0]

    # Wald估计（等价）
    cov_zy = np.cov(Z, Y)[0, 1]
    cov_zx = np.cov(Z, X)[0, 1]
    beta_wald = cov_zy / cov_zx if cov_zx != 0 else 0

    # 第一阶段F统计量
    y_pred = first_stage.predict(Z.reshape(-1, 1))
    y_true = X
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - rss / tss
    n = len(Z)
    k = 1
    f_stat = (r2 / k) / ((1 - r2) / (n - k - 1)) if (1 - r2) > 0 else 0

    return {
        'beta_2sls': beta_2sls,
        'beta_wald': beta_wald,
        'first_stage': first_stage,
        'second_stage': second_stage,
        'X_hat': X_hat,
        'f_stat': f_stat,
        'r2_first_stage': r2
    }


def analyze_iv(
    n_samples: int = 1000,
    instrument_strength: float = 0.5,
    compliance_rate: float = 0.7
) -> dict:
    """工具变量分析

    Args:
        n_samples: 样本量
        instrument_strength: 工具变量强度
        compliance_rate: 顺从率（用于解释）

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成数据
    df = generate_iv_data(
        n_samples=n_samples,
        instrument_strength=instrument_strength,
        treatment_effect=-2.0
    )

    # OLS估计（有偏）
    ols = LinearRegression()
    ols.fit(df[['treatment']], df['outcome'])
    beta_ols = ols.coef_[0]

    # 2SLS估计（无偏）
    results_2sls = two_stage_least_squares(
        df['instrument'].values,
        df['treatment'].values,
        df['outcome'].values
    )

    # 检验工具变量假设
    # 1. 相关性
    corr_zx = df['instrument'].corr(df['treatment'])

    # 2. 排他性（不可直接检验，显示相关性作为参考）
    corr_zy = df['instrument'].corr(df['outcome'])

    # 3. 外生性（不可直接检验，显示与混淆因素的相关性）
    corr_zu = df['instrument'].corr(df['confounder'])

    # 创建可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('第一阶段: Z → X', '第二阶段: X̂ → Y')
    )

    # 第一阶段
    fig.add_trace(
        go.Scatter(
            x=df['instrument'],
            y=df['treatment'],
            mode='markers',
            marker=dict(color='lightblue', size=6, opacity=0.6),
            name='观测数据'
        ),
        row=1, col=1
    )

    # 第一阶段拟合线
    z_range = np.linspace(df['instrument'].min(), df['instrument'].max(), 100)
    x_pred = results_2sls['first_stage'].predict(z_range.reshape(-1, 1))
    fig.add_trace(
        go.Scatter(
            x=z_range,
            y=x_pred,
            mode='lines',
            line=dict(color='green', width=3),
            name='第一阶段拟合'
        ),
        row=1, col=1
    )

    # 第二阶段
    fig.add_trace(
        go.Scatter(
            x=results_2sls['X_hat'],
            y=df['outcome'],
            mode='markers',
            marker=dict(color='lightcoral', size=6, opacity=0.6),
            name='预测的 X'
        ),
        row=1, col=2
    )

    # 第二阶段拟合线
    x_hat_range = np.linspace(results_2sls['X_hat'].min(), results_2sls['X_hat'].max(), 100)
    y_pred = results_2sls['second_stage'].predict(x_hat_range.reshape(-1, 1))
    fig.add_trace(
        go.Scatter(
            x=x_hat_range,
            y=y_pred,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'第二阶段拟合 (β={results_2sls["beta_2sls"]:.2f})'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="工具变量 (Z)", row=1, col=1)
    fig.update_xaxes(title_text="预测的处理变量 (X̂)", row=1, col=2)
    fig.update_yaxes(title_text="处理变量 (X)", row=1, col=1)
    fig.update_yaxes(title_text="结果变量 (Y)", row=1, col=2)

    fig.update_layout(
        height=500,
        title_text="两阶段最小二乘 (2SLS) 估计过程",
        showlegend=True,
        template='plotly_white'
    )

    # 创建对比图
    fig_comparison = go.Figure()

    methods = ['真实值', 'OLS (有偏)', '2SLS (无偏)']
    estimates = [-2.0, beta_ols, results_2sls['beta_2sls']]
    colors = ['gray', '#EB5757', '#27AE60']

    fig_comparison.add_trace(go.Bar(
        x=methods,
        y=estimates,
        marker_color=colors,
        text=[f'{e:.2f}' for e in estimates],
        textposition='auto'
    ))

    fig_comparison.update_layout(
        title='估计方法对比',
        yaxis_title='估计值',
        template='plotly_white',
        height=400
    )

    # 判断工具变量强度
    if results_2sls['f_stat'] > 10:
        iv_strength = "✅ 强工具变量"
    else:
        iv_strength = "⚠️ 弱工具变量"

    summary = f"""
## 工具变量分析结果

### 核心思想

工具变量通过找到**外生冲击**来解决内生性问题:
- 第一阶段：提取处理变量的外生部分
- 第二阶段：用外生部分估计因果效应

### 估计结果

| 方法 | 估计值 | 备注 |
|------|--------|------|
| 真实值 | -2.00 | 真实因果效应 |
| OLS | {beta_ols:.2f} | ❌ 有偏（内生性） |
| **2SLS** | **{results_2sls['beta_2sls']:.2f}** | ✅ 无偏 |
| Wald | {results_2sls['beta_wald']:.2f} | 与2SLS等价 |

### 工具变量检验

#### 1️⃣ 相关性假设 (可检验)

| 指标 | 值 | 判断 |
|------|-----|------|
| Corr(Z, X) | {corr_zx:.3f} | {'✅ 满足' if abs(corr_zx) > 0.3 else '❌ 弱IV'} |
| 第一阶段 F 统计量 | {results_2sls['f_stat']:.2f} | {iv_strength} |
| 第一阶段 R² | {results_2sls['r2_first_stage']:.3f} | - |

**经验法则**: F > 10 表示强工具变量

#### 2️⃣ 排他性假设 (不可检验)

- Corr(Z, Y) = {corr_zy:.3f}
- ⚠️ 需要理论支撑，无法直接检验
- 工具变量只能通过处理变量影响结果

#### 3️⃣ 外生性假设 (不可检验)

- Corr(Z, U) = {corr_zu:.3f}
- ⚠️ 实际中混淆因素U不可观测
- 需要依赖制度背景和经济逻辑

### LATE 解释

工具变量估计的是**局部平均处理效应 (LATE)**:
- 估计对象：**顺从者** (Compliers)
- 顺从者：因工具变量变化而改变处理状态的个体
- LATE ≠ ATE（全体平均处理效应）

### 方法选择

| 场景 | 推荐方法 |
|------|----------|
| 有强工具变量 | IV / 2SLS |
| 有门槛 | RDD |
| 有时间维度 | DID |
| 可随机分配 | RCT |

### 业务应用

工具变量适用于:
1. **价格弹性**: 成本冲击作为价格的IV
2. **广告效果**: 服务器故障作为曝光的IV
3. **教育回报**: 到学校的距离作为教育年限的IV
4. **内容推荐**: 随机展示位置作为点击的IV
"""

    return {
        "charts": [
            fig_to_chart_data(fig),
            fig_to_chart_data(fig_comparison)
        ],
        "tables": [
            {
                "title": "工具变量假设检验",
                "headers": ["假设", "检验方法", "结果", "是否满足"],
                "rows": [
                    ["相关性", f"Corr(Z,X) = {corr_zx:.3f}", f"F = {results_2sls['f_stat']:.2f}", "✅" if results_2sls['f_stat'] > 10 else "❌"],
                    ["排他性", "不可直接检验", f"Corr(Z,Y) = {corr_zy:.3f}", "⚠️ 需理论支撑"],
                    ["外生性", "不可直接检验", f"Corr(Z,U) = {corr_zu:.3f}", "⚠️ 需制度背景"]
                ]
            }
        ],
        "summary": summary,
        "metrics": {
            "ols_estimate": float(beta_ols),
            "iv_estimate": float(results_2sls['beta_2sls']),
            "wald_estimate": float(results_2sls['beta_wald']),
            "true_effect": -2.0,
            "ols_bias": float(beta_ols - (-2.0)),
            "iv_bias": float(results_2sls['beta_2sls'] - (-2.0)),
            "f_statistic": float(results_2sls['f_stat']),
            "r2_first_stage": float(results_2sls['r2_first_stage']),
            "corr_zx": float(corr_zx),
            "corr_zy": float(corr_zy),
            "corr_zu": float(corr_zu),
            "is_weak_iv": results_2sls['f_stat'] < 10
        }
    }
