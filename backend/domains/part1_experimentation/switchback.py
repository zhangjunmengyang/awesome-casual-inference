"""
Switchback Experiments
Switchback实验设计

核心思想:
--------
在时间维度上切换处理，同一单元在不同时间段接受不同处理
适用于供给侧实验和存在网络效应的场景

典型应用:
--------
1. 网约车：司机端实验（避免市场溢出）
2. 外卖配送：配送策略实验
3. 广告竞价：出价策略测试
4. 定价实验：动态定价策略

优势:
----
1. 减少组间差异（同一单元自己做对照）
2. 缓解网络效应
3. 提高统计功效
4. 适合供给侧实验

挑战:
----
1. Carryover效应（残留效应）
2. 时间效应（趋势、周期性）
3. 学习效应
4. 切换频率选择

面试考点:
--------
- Switchback vs A/B testing的区别？
- 如何处理carryover效应？
- 时间粒度如何选择？
- Switchback的统计模型？
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import welch_t_test


def generate_switchback_data(
    n_units: int = 50,
    n_time_periods: int = 100,
    treatment_effect: float = 0.10,
    carryover_effect: float = 0.03,
    time_trend: float = 0.001,
    baseline_mean: float = 100,
    noise_std: float = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成Switchback实验数据

    Parameters:
    -----------
    n_units: 单元数（如司机数）
    n_time_periods: 时间段数
    treatment_effect: 处理效应
    carryover_effect: 残留效应强度
    time_trend: 时间趋势
    baseline_mean: 基线均值
    noise_std: 噪声标准差
    seed: 随机种子

    Returns:
    --------
    实验数据DataFrame
    """
    np.random.seed(seed)

    data = []

    for unit in range(n_units):
        # 单元固定效应
        unit_effect = np.random.normal(0, 5)

        # 生成处理序列（随机切换）
        treatment_sequence = np.random.binomial(1, 0.5, n_time_periods)

        for t in range(n_time_periods):
            # 基线值
            base_value = baseline_mean + unit_effect

            # 时间趋势
            time_effect = time_trend * t

            # 当前处理
            current_treatment = treatment_sequence[t]

            # 处理效应
            treat_effect = current_treatment * treatment_effect * base_value

            # Carryover效应（上一期的处理影响）
            carryover = 0
            if t > 0:
                prev_treatment = treatment_sequence[t-1]
                carryover = prev_treatment * carryover_effect * base_value

            # 噪声
            noise = np.random.normal(0, noise_std)

            # 最终观测值
            outcome = base_value + time_effect + treat_effect + carryover + noise

            data.append({
                'unit_id': unit,
                'time_period': t,
                'treatment': current_treatment,
                'outcome': outcome,
                'base_value': base_value,
                'time_effect': time_effect
            })

    return pd.DataFrame(data)


def naive_switchback_analysis(
    df: pd.DataFrame,
    outcome_col: str = 'outcome',
    treatment_col: str = 'treatment'
) -> Dict:
    """
    朴素Switchback分析

    简单比较所有treatment vs control时间段

    Parameters:
    -----------
    df: 实验数据
    outcome_col: 结果列名
    treatment_col: 处理列名

    Returns:
    --------
    分析结果
    """
    treated = df[df[treatment_col] == 1][outcome_col].values
    control = df[df[treatment_col] == 0][outcome_col].values

    result = welch_t_test(control, treated)

    return {
        'ate_naive': result['effect'],
        'relative_effect': result['relative_effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'n_treated_periods': len(treated),
        'n_control_periods': len(control)
    }


def fixed_effects_analysis(
    df: pd.DataFrame,
    outcome_col: str = 'outcome',
    treatment_col: str = 'treatment',
    unit_col: str = 'unit_id',
    time_col: str = 'time_period'
) -> Dict:
    """
    固定效应分析

    控制单元固定效应和时间固定效应

    模型: Y_it = α_i + γ_t + β*Treatment_it + ε_it

    Parameters:
    -----------
    df: 实验数据
    outcome_col: 结果列名
    treatment_col: 处理列名
    unit_col: 单元列名
    time_col: 时间列名

    Returns:
    --------
    固定效应估计结果
    """
    # 去均值（Within transformation）
    df = df.copy()

    # 按单元去均值（去除单元固定效应）
    df['outcome_demeaned_unit'] = df.groupby(unit_col)[outcome_col].transform(
        lambda x: x - x.mean()
    )
    df['treatment_demeaned_unit'] = df.groupby(unit_col)[treatment_col].transform(
        lambda x: x - x.mean()
    )

    # 按时间去均值（去除时间固定效应）
    df['outcome_demeaned_time'] = df.groupby(time_col)[outcome_col].transform(
        lambda x: x - x.mean()
    )

    # 双向去均值（同时去除两个固定效应）
    unit_means = df.groupby(unit_col)[outcome_col].transform('mean')
    time_means = df.groupby(time_col)[outcome_col].transform('mean')
    grand_mean = df[outcome_col].mean()

    df['outcome_fe'] = df[outcome_col] - unit_means - time_means + grand_mean

    treatment_unit_means = df.groupby(unit_col)[treatment_col].transform('mean')
    treatment_time_means = df.groupby(time_col)[treatment_col].transform('mean')
    treatment_grand_mean = df[treatment_col].mean()

    df['treatment_fe'] = (
        df[treatment_col] - treatment_unit_means -
        treatment_time_means + treatment_grand_mean
    )

    # 简单线性回归（已去除固定效应）
    from scipy.stats import linregress

    X = df['treatment_fe'].values
    Y = df['outcome_fe'].values

    # 去除NaN
    mask = ~(np.isnan(X) | np.isnan(Y))
    X = X[mask]
    Y = Y[mask]

    slope, intercept, r_value, p_value, std_err = linregress(X, Y)

    # 置信区间
    ci_lower = slope - 1.96 * std_err
    ci_upper = slope + 1.96 * std_err

    return {
        'ate_fe': slope,
        'se': std_err,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'r_squared': r_value ** 2,
        'note': 'Fixed effects model controlling for unit and time'
    }


def detect_carryover(
    df: pd.DataFrame,
    outcome_col: str = 'outcome',
    treatment_col: str = 'treatment',
    unit_col: str = 'unit_id',
    lag: int = 1
) -> Dict:
    """
    检测残留效应

    看上一期的处理是否影响当前期的结果

    Parameters:
    -----------
    df: 实验数据
    outcome_col: 结果列名
    treatment_col: 处理列名
    unit_col: 单元列名
    lag: 滞后期数

    Returns:
    --------
    残留效应检测结果
    """
    df = df.copy()

    # 创建滞后处理变量
    df['treatment_lag'] = df.groupby(unit_col)[treatment_col].shift(lag)

    # 去除缺失值
    df_clean = df.dropna(subset=['treatment_lag'])

    # 仅看对照组（当前期未处理）
    control_df = df_clean[df_clean[treatment_col] == 0]

    if len(control_df) < 100:
        return {
            'carryover_effect': np.nan,
            'error': 'Insufficient data'
        }

    # 比较：上期treated vs 上期control
    prev_treated = control_df[control_df['treatment_lag'] == 1][outcome_col].values
    prev_control = control_df[control_df['treatment_lag'] == 0][outcome_col].values

    if len(prev_treated) == 0 or len(prev_control) == 0:
        return {
            'carryover_effect': np.nan,
            'error': 'Insufficient data in groups'
        }

    result = welch_t_test(prev_control, prev_treated)

    return {
        'carryover_effect': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'is_significant': result['p_value'] < 0.05,
        'interpretation': (
            'Significant carryover detected!' if result['p_value'] < 0.05
            else 'No significant carryover'
        )
    }


def cluster_by_time(
    df: pd.DataFrame,
    outcome_col: str = 'outcome',
    treatment_col: str = 'treatment',
    time_col: str = 'time_period'
) -> Dict:
    """
    时间聚类分析

    聚合到时间段层面，考虑时间内相关性

    Parameters:
    -----------
    df: 实验数据
    outcome_col: 结果列名
    treatment_col: 处理列名
    time_col: 时间列名

    Returns:
    --------
    时间聚类分析结果
    """
    # 聚合到时间层面
    time_summary = df.groupby([time_col, treatment_col]).agg({
        outcome_col: ['mean', 'count']
    }).reset_index()

    time_summary.columns = [time_col, treatment_col, 'mean_outcome', 'n']

    # 分离treated和control时间段
    treated_times = time_summary[time_summary[treatment_col] == 1]
    control_times = time_summary[time_summary[treatment_col] == 0]

    if len(treated_times) == 0 or len(control_times) == 0:
        return {'error': 'Insufficient time periods'}

    result = welch_t_test(
        control_times['mean_outcome'].values,
        treated_times['mean_outcome'].values
    )

    return {
        'ate_time_clustered': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'n_treated_times': len(treated_times),
        'n_control_times': len(control_times),
        'note': 'Time-clustered analysis'
    }


def plot_switchback_timeline(
    df: pd.DataFrame,
    unit_id: int,
    outcome_col: str = 'outcome',
    treatment_col: str = 'treatment',
    time_col: str = 'time_period',
    unit_col: str = 'unit_id'
) -> go.Figure:
    """
    绘制单个单元的switchback时间线

    Parameters:
    -----------
    df: 实验数据
    unit_id: 单元ID
    outcome_col: 结果列名
    treatment_col: 处理列名
    time_col: 时间列名
    unit_col: 单元列名

    Returns:
    --------
    Plotly图表
    """
    unit_df = df[df[unit_col] == unit_id].sort_values(time_col)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Treatment Assignment', 'Outcome Over Time'),
        row_heights=[0.3, 0.7],
        vertical_spacing=0.15
    )

    # 1. 处理分配
    fig.add_trace(
        go.Scatter(
            x=unit_df[time_col],
            y=unit_df[treatment_col],
            mode='lines',
            line=dict(shape='hv', color='#2D9CDB', width=2),
            fill='tozeroy',
            fillcolor='rgba(45, 156, 219, 0.3)',
            name='Treatment',
            showlegend=False
        ),
        row=1, col=1
    )

    # 2. 结果
    colors = ['#27AE60' if t == 1 else '#2D9CDB' for t in unit_df[treatment_col]]

    fig.add_trace(
        go.Scatter(
            x=unit_df[time_col],
            y=unit_df[outcome_col],
            mode='markers+lines',
            marker=dict(size=6, color=colors),
            line=dict(color='gray', width=1),
            name='Outcome',
            showlegend=False
        ),
        row=2, col=1
    )

    # 标注治疗期
    for idx, row in unit_df.iterrows():
        if row[treatment_col] == 1:
            fig.add_vrect(
                x0=row[time_col] - 0.5,
                x1=row[time_col] + 0.5,
                fillcolor='rgba(39, 174, 96, 0.1)',
                layer='below',
                line_width=0,
                row=2, col=1
            )

    fig.update_layout(
        title=f'Switchback Timeline - Unit {unit_id}',
        template='plotly_white',
        height=600
    )

    fig.update_xaxes(title_text='Time Period', row=2, col=1)
    fig.update_yaxes(title_text='Treatment', row=1, col=1)
    fig.update_yaxes(title_text='Outcome', row=2, col=1)

    return fig


def plot_treatment_effect_over_time(
    df: pd.DataFrame,
    outcome_col: str = 'outcome',
    treatment_col: str = 'treatment',
    time_col: str = 'time_period'
) -> go.Figure:
    """
    绘制处理效应随时间变化

    Parameters:
    -----------
    df: 实验数据
    outcome_col: 结果列名
    treatment_col: 处理列名
    time_col: 时间列名

    Returns:
    --------
    Plotly图表
    """
    # 按时间聚合
    time_summary = df.groupby([time_col, treatment_col]).agg({
        outcome_col: ['mean', 'sem']
    }).reset_index()

    time_summary.columns = [time_col, treatment_col, 'mean', 'sem']

    treated = time_summary[time_summary[treatment_col] == 1]
    control = time_summary[time_summary[treatment_col] == 0]

    fig = go.Figure()

    # 对照组
    fig.add_trace(go.Scatter(
        x=control[time_col],
        y=control['mean'],
        error_y=dict(type='data', array=control['sem'] * 1.96),
        mode='markers+lines',
        name='Control',
        marker=dict(size=6, color='#2D9CDB'),
        line=dict(color='#2D9CDB')
    ))

    # 实验组
    fig.add_trace(go.Scatter(
        x=treated[time_col],
        y=treated['mean'],
        error_y=dict(type='data', array=treated['sem'] * 1.96),
        mode='markers+lines',
        name='Treatment',
        marker=dict(size=6, color='#27AE60'),
        line=dict(color='#27AE60')
    ))

    fig.update_layout(
        title='Treatment Effect Over Time',
        xaxis_title='Time Period',
        yaxis_title='Mean Outcome',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )

    return fig


if __name__ == "__main__":
    # 测试代码
    df = generate_switchback_data(
        n_units=50,
        n_time_periods=100,
        treatment_effect=0.10,
        carryover_effect=0.03
    )

    print("Switchback Data Generated:")
    print(f"  Units: {df['unit_id'].nunique()}")
    print(f"  Time Periods: {df['time_period'].nunique()}")
    print(f"  Total Observations: {len(df)}")

    # 朴素分析
    naive = naive_switchback_analysis(df)
    print(f"\nNaive ATE: {naive['ate_naive']:.2f} (p={naive['p_value']:.4f})")

    # 固定效应分析
    fe = fixed_effects_analysis(df)
    print(f"Fixed Effects ATE: {fe['ate_fe']:.2f} (p={fe['p_value']:.4f})")

    # 残留效应检测
    carryover = detect_carryover(df)
    print(f"\nCarryover Detection: {carryover['interpretation']}")
