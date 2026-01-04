"""双重差分 (Difference-in-Differences) 分析"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from .utils import generate_did_data, fig_to_chart_data


def analyze_did_basic(
    n_periods: int = 10,
    treatment_period: int = 6,
    parallel_trend_violation: float = 0.0,
    effect_size: float = 10.0
) -> dict:
    """基础双重差分分析

    Args:
        n_periods: 总时期数
        treatment_period: 处理开始时期
        parallel_trend_violation: 平行趋势违反程度
        effect_size: 处理效应大小

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成多期数据
    np.random.seed(42)
    n_treated = 500
    n_control = 500

    data_list = []

    # 处理组
    for i in range(n_treated):
        user_id = f"treated_{i}"
        baseline = np.random.normal(100, 15)
        individual_trend = np.random.normal(0, 2)

        for t in range(n_periods):
            time_effect = 4 * t
            is_post = 1 if t >= treatment_period else 0
            treatment = effect_size if is_post else 0

            data_list.append({
                'user_id': user_id,
                'group': 'treated',
                'period': t,
                'treat': 1,
                'post': is_post,
                'outcome': baseline + time_effect + individual_trend * t + treatment + np.random.normal(0, 5)
            })

    # 对照组
    for i in range(n_control):
        user_id = f"control_{i}"
        baseline = np.random.normal(80, 15)
        individual_trend = np.random.normal(0, 2)

        for t in range(n_periods):
            time_effect = 4 * t + parallel_trend_violation * t  # 添加趋势违反

            data_list.append({
                'user_id': user_id,
                'group': 'control',
                'period': t,
                'treat': 0,
                'post': 0,
                'outcome': baseline + time_effect + individual_trend * t + np.random.normal(0, 5)
            })

    df = pd.DataFrame(data_list)

    # 计算DID估计量
    # 手动计算
    treat_post = df[(df['treat'] == 1) & (df['post'] == 1)]['outcome'].mean()
    treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)]['outcome'].mean()
    control_post = df[(df['treat'] == 0) & (df['period'] >= treatment_period)]['outcome'].mean()
    control_pre = df[(df['treat'] == 0) & (df['period'] < treatment_period)]['outcome'].mean()

    diff_treat = treat_post - treat_pre
    diff_control = control_post - control_pre
    did_estimate = diff_treat - diff_control

    # 回归估计
    df['treat_post'] = df['treat'] * df['post']
    X = df[['treat', 'post', 'treat_post']].values
    y = df['outcome'].values
    model = LinearRegression().fit(X, y)
    did_regression = model.coef_[2]

    # 计算每期平均值用于可视化
    trends = df.groupby(['period', 'group'])['outcome'].mean().reset_index()

    # 创建可视化
    fig = go.Figure()

    # 处理组趋势
    treated_data = trends[trends['group'] == 'treated']
    fig.add_trace(go.Scatter(
        x=treated_data['period'],
        y=treated_data['outcome'],
        mode='lines+markers',
        name='处理组',
        line=dict(color='#27AE60', width=3),
        marker=dict(size=8)
    ))

    # 对照组趋势
    control_data = trends[trends['group'] == 'control']
    fig.add_trace(go.Scatter(
        x=control_data['period'],
        y=control_data['outcome'],
        mode='lines+markers',
        name='对照组',
        line=dict(color='#2D9CDB', width=3),
        marker=dict(size=8)
    ))

    # 添加处理时间线
    fig.add_vline(
        x=treatment_period - 0.5,
        line_dash="dash",
        line_color='#EB5757',
        line_width=2,
        annotation_text="处理开始",
        annotation_position="top"
    )

    # 添加背景色
    fig.add_vrect(
        x0=-0.5, x1=treatment_period - 0.5,
        fillcolor="lightgray", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="处理前", annotation_position="top left"
    )

    fig.add_vrect(
        x0=treatment_period - 0.5, x1=n_periods - 0.5,
        fillcolor="lightgreen", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="处理后", annotation_position="top right"
    )

    fig.update_layout(
        title='双重差分 (DID) 趋势图',
        xaxis_title='时期',
        yaxis_title='平均结果',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    # 构建摘要
    summary = f"""
## 双重差分分析结果

### 关键指标

| 指标 | 值 |
|------|-----|
| 处理组变化 | {diff_treat:.2f} |
| 对照组变化 | {diff_control:.2f} |
| **DID估计量 (手动)** | **{did_estimate:.2f}** |
| **DID估计量 (回归)** | **{did_regression:.2f}** |
| 真实效应 | {effect_size:.2f} |
| 估计偏差 | {(did_estimate - effect_size):.2f} |

### 核心思想

DID通过两次差分消除偏差:
1. **第一次差分 (时间)**: 消除时间趋势
2. **第二次差分 (组间)**: 消除组间固有差异

### 关键假设

**平行趋势假设**: 如果没有处理，两组的趋势应该平行
- 当前违反程度: {parallel_trend_violation}
- 违反程度越大，估计偏差越大

### 解读

- 处理组在处理后增长 {diff_treat:.2f} 单位
- 对照组同期增长 {diff_control:.2f} 单位
- 额外的 {did_estimate:.2f} 单位增长归因于处理效应
"""

    return {
        "charts": [fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "did_estimate_manual": float(did_estimate),
            "did_estimate_regression": float(did_regression),
            "true_effect": float(effect_size),
            "bias": float(did_estimate - effect_size),
            "treat_change": float(diff_treat),
            "control_change": float(diff_control),
        }
    }


def analyze_did_event_study(
    n_periods: int = 10,
    treatment_period: int = 6,
    effect_size: float = 10.0
) -> dict:
    """Event Study设计分析

    Args:
        n_periods: 总时期数
        treatment_period: 处理开始时期
        effect_size: 处理效应大小

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成多期数据
    np.random.seed(42)
    n_treated = 500
    n_control = 500

    data_list = []

    for i in range(n_treated):
        user_id = f"treated_{i}"
        baseline = np.random.normal(100, 15)

        for t in range(n_periods):
            time_effect = 4 * t
            is_post = 1 if t >= treatment_period else 0
            treatment = effect_size if is_post else 0

            data_list.append({
                'user_id': user_id,
                'group': 'treated',
                'period': t,
                'treat': 1,
                'outcome': baseline + time_effect + treatment + np.random.normal(0, 5)
            })

    for i in range(n_control):
        user_id = f"control_{i}"
        baseline = np.random.normal(80, 15)

        for t in range(n_periods):
            time_effect = 4 * t

            data_list.append({
                'user_id': user_id,
                'group': 'control',
                'period': t,
                'treat': 0,
                'outcome': baseline + time_effect + np.random.normal(0, 5)
            })

    df = pd.DataFrame(data_list)

    # 计算相对时间
    df['rel_time'] = df['period'] - treatment_period

    # Event study系数估计 (简化版本)
    coeffs = []
    for rel_t in sorted(df['rel_time'].unique()):
        if rel_t == -1:  # 基准期
            coeffs.append({
                'rel_time': rel_t,
                'coef': 0,
                'ci_lower': 0,
                'ci_upper': 0
            })
        else:
            # 计算该相对时期的处理效应
            period_data = df[df['rel_time'] == rel_t]
            treat_mean = period_data[period_data['treat'] == 1]['outcome'].mean()
            control_mean = period_data[period_data['treat'] == 0]['outcome'].mean()
            coef = treat_mean - control_mean

            # 简化的标准误
            se = 2.0
            coeffs.append({
                'rel_time': rel_t,
                'coef': coef - (df[df['rel_time'] == -1].groupby('treat')['outcome'].mean().diff().iloc[-1]),
                'ci_lower': coef - 1.96 * se,
                'ci_upper': coef + 1.96 * se
            })

    event_df = pd.DataFrame(coeffs)

    # 创建Event Study图
    fig = go.Figure()

    # 点估计
    fig.add_trace(go.Scatter(
        x=event_df['rel_time'],
        y=event_df['coef'],
        mode='markers+lines',
        name='DID估计量',
        marker=dict(size=10, color='#2D9CDB'),
        line=dict(color='#2D9CDB', width=2)
    ))

    # 置信区间
    fig.add_trace(go.Scatter(
        x=list(event_df['rel_time']) + list(event_df['rel_time'][::-1]),
        y=list(event_df['ci_upper']) + list(event_df['ci_lower'][::-1]),
        fill='toself',
        fillcolor='rgba(45, 156, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% 置信区间',
        showlegend=True
    ))

    # 零线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    # 处理时间线
    fig.add_vline(
        x=-0.5,
        line_dash="dash",
        line_color='#EB5757',
        line_width=2,
        annotation_text="处理开始",
        annotation_position="top"
    )

    fig.update_layout(
        title='Event Study 图：政策效应的动态演变',
        xaxis_title='相对时间（相对于处理时间）',
        yaxis_title='处理效应',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    summary = f"""
## Event Study 分析结果

### 方法说明

Event Study允许我们:
1. **检验平行趋势**: 观察处理前系数是否接近0
2. **估计动态效应**: 观察处理效应如何随时间变化
3. **检测预期效应**: 处理前是否已有反应

### 关键发现

- **处理前**: 系数应接近0且不显著（支持平行趋势）
- **处理时**: 开始出现正向效应
- **处理后**: 观察效应是否持续、增强或减弱

### 解读建议

1. 如果处理前系数显著不为0 → 违反平行趋势假设
2. 如果处理后效应逐渐增大 → 效应存在累积
3. 如果处理后效应逐渐减小 → 效应存在衰减
"""

    return {
        "charts": [fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "n_periods": int(n_periods),
            "treatment_period": int(treatment_period),
            "effect_size": float(effect_size)
        }
    }


def analyze_did_staggered(
    n_cities: int = 4,
    n_users_per_city: int = 200,
    n_periods: int = 12,
    effect_size: float = 15.0
) -> dict:
    """交错DID分析

    Args:
        n_cities: 城市数量
        n_users_per_city: 每个城市的用户数
        n_periods: 总时期数
        effect_size: 基础处理效应

    Returns:
        包含图表、表格和摘要的字典
    """
    np.random.seed(42)

    cities = ['Beijing', 'Shanghai', 'Shenzhen', 'Guangzhou']
    treatment_times = [3, 6, 9, None]  # None表示始终不处理

    data_list = []

    for city, treat_time in zip(cities[:n_cities], treatment_times[:n_cities]):
        for i in range(n_users_per_city):
            user_id = f"{city}_{i}"
            baseline = np.random.normal(100, 15)

            for t in range(n_periods):
                time_effect = 3 * t

                # 是否已处理
                is_treated = 0
                if treat_time is not None and t >= treat_time:
                    is_treated = 1

                # 处理效应（随时间衰减）
                if is_treated:
                    time_since_treatment = t - treat_time
                    treatment_effect = effect_size * np.exp(-0.1 * time_since_treatment)
                else:
                    treatment_effect = 0

                data_list.append({
                    'user_id': user_id,
                    'city': city,
                    'period': t,
                    'treat_time': treat_time if treat_time is not None else 999,
                    'is_treated': is_treated,
                    'outcome': baseline + time_effect + treatment_effect + np.random.normal(0, 5)
                })

    df = pd.DataFrame(data_list)

    # 计算每个城市每期的平均值
    trends = df.groupby(['period', 'city'])['outcome'].mean().reset_index()

    # 创建可视化
    fig = go.Figure()

    colors_map = {
        'Beijing': '#27AE60',
        'Shanghai': '#2D9CDB',
        'Shenzhen': '#F2994A',
        'Guangzhou': '#EB5757'
    }

    for city in trends['city'].unique():
        city_data = trends[trends['city'] == city]
        fig.add_trace(go.Scatter(
            x=city_data['period'],
            y=city_data['outcome'],
            mode='lines+markers',
            name=city,
            line=dict(color=colors_map.get(city, 'gray'), width=2),
            marker=dict(size=6)
        ))

    # 添加处理时间线
    treatment_times_valid = [t for t in treatment_times[:n_cities] if t is not None]
    for city, treat_time in zip(cities[:n_cities], treatment_times[:n_cities]):
        if treat_time is not None and treat_time < 999:
            fig.add_vline(
                x=treat_time - 0.5,
                line_dash="dot",
                line_color=colors_map.get(city, 'gray'),
                line_width=1,
                annotation_text=city,
                annotation_position="top"
            )

    fig.update_layout(
        title='交错DID：不同城市在不同时间接受处理',
        xaxis_title='时期',
        yaxis_title='平均结果',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    summary = f"""
## 交错DID分析结果

### 方法说明

交错DID适用于:
- 不同单位在不同时间接受处理
- 需要避免已处理单位作为对照组（禁忌比较）

### 关键发现

- **处理时点**: 各城市在不同时间点接受处理
- **效应异质性**: 不同城市可能有不同的处理效应
- **对照组**: 始终未处理的单位提供干净的对照

### 现代方法

传统的TWFE（双向固定效应）在交错设计中可能有偏，推荐使用:
1. **Callaway-Sant'Anna (2021)**: 避免禁忌比较
2. **Sun-Abraham (2021)**: 交互加权估计
3. **De Chaisemartin-D'Haultfoeuille (2020)**: DID_M估计量

### 注意事项

- 已处理组不应作为未处理组的对照
- 处理效应可能随时间变化（动态效应）
- 需要检验平行趋势假设
"""

    return {
        "charts": [fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "n_cities": int(n_cities),
            "n_periods": int(n_periods),
            "effect_size": float(effect_size)
        }
    }
