"""
Long-term Effects Estimation
长期效应估计

核心问题:
--------
A/B测试通常只运行几周，但业务关心长期影响
短期效应 ≠ 长期效应

典型场景:
--------
1. 新手引导：短期转化 vs 长期留存
2. 推荐算法：短期点击 vs 长期满意度
3. 定价策略：短期收入 vs 长期用户价值
4. 产品改版：短期适应 vs 长期习惯

解决方案:
--------
1. 长期运行实验（成本高）
2. Surrogate指标（代理指标）
3. Holdout组（长期对照组）
4. 时间序列外推

面试考点:
--------
- 为什么短期效应可能误导？
- Surrogate指标的有效性条件？
- Holdout的权衡？
- 如何预测长期效应？
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import welch_t_test


def simulate_long_term_data(
    n_users: int = 10000,
    n_days: int = 180,
    short_term_effect: float = 0.15,
    long_term_effect: float = 0.05,
    decay_rate: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """
    模拟长期效应数据

    效应随时间衰减：
    Effect(t) = long_term + (short_term - long_term) * exp(-decay_rate * t)

    Parameters:
    -----------
    n_users: 用户数
    n_days: 观察天数
    short_term_effect: 短期效应
    long_term_effect: 长期效应（稳态）
    decay_rate: 衰减率
    seed: 随机种子

    Returns:
    --------
    用户日志数据
    """
    np.random.seed(seed)

    data = []

    for user_id in range(n_users):
        # 随机分配
        is_treatment = np.random.binomial(1, 0.5)

        # 基线值（用户异质性）
        baseline = np.random.normal(50, 10)

        # 模拟每天的行为
        for day in range(n_days):
            # 时间衰减的效应
            if is_treatment:
                time_varying_effect = (
                    long_term_effect +
                    (short_term_effect - long_term_effect) * np.exp(-decay_rate * day)
                )
                effect = time_varying_effect * baseline
            else:
                effect = 0

            # 加噪声
            outcome = baseline + effect + np.random.normal(0, 5)

            data.append({
                'user_id': user_id,
                'day': day,
                'is_treatment': is_treatment,
                'outcome': outcome,
                'baseline': baseline
            })

    return pd.DataFrame(data)


def short_term_analysis(
    df: pd.DataFrame,
    short_term_window: int = 14,
    outcome_col: str = 'outcome'
) -> Dict:
    """
    短期分析（前N天）

    Parameters:
    -----------
    df: 用户数据
    short_term_window: 短期窗口（天）
    outcome_col: 结果列名

    Returns:
    --------
    短期效应估计
    """
    # 筛选短期数据
    short_df = df[df['day'] < short_term_window].copy()

    # 按用户聚合
    user_summary = short_df.groupby('user_id').agg({
        outcome_col: 'mean',
        'is_treatment': 'first'
    }).reset_index()

    treated = user_summary[user_summary['is_treatment'] == 1][outcome_col].values
    control = user_summary[user_summary['is_treatment'] == 0][outcome_col].values

    result = welch_t_test(control, treated)

    return {
        'short_term_effect': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'window_days': short_term_window
    }


def long_term_analysis(
    df: pd.DataFrame,
    long_term_window: Tuple[int, int] = (90, 180),
    outcome_col: str = 'outcome'
) -> Dict:
    """
    长期分析（后期稳态）

    Parameters:
    -----------
    df: 用户数据
    long_term_window: 长期窗口（天）- (开始, 结束)
    outcome_col: 结果列名

    Returns:
    --------
    长期效应估计
    """
    # 筛选长期数据
    long_df = df[
        (df['day'] >= long_term_window[0]) &
        (df['day'] < long_term_window[1])
    ].copy()

    # 按用户聚合
    user_summary = long_df.groupby('user_id').agg({
        outcome_col: 'mean',
        'is_treatment': 'first'
    }).reset_index()

    treated = user_summary[user_summary['is_treatment'] == 1][outcome_col].values
    control = user_summary[user_summary['is_treatment'] == 0][outcome_col].values

    result = welch_t_test(control, treated)

    return {
        'long_term_effect': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'window_days': long_term_window
    }


def time_varying_effect(
    df: pd.DataFrame,
    window_size: int = 7,
    outcome_col: str = 'outcome'
) -> pd.DataFrame:
    """
    估计时变效应

    滑动窗口估计效应随时间的变化

    Parameters:
    -----------
    df: 用户数据
    window_size: 窗口大小（天）
    outcome_col: 结果列名

    Returns:
    --------
    时变效应DataFrame
    """
    max_day = df['day'].max()
    results = []

    for start_day in range(0, max_day - window_size + 1, window_size // 2):
        end_day = start_day + window_size

        window_df = df[
            (df['day'] >= start_day) &
            (df['day'] < end_day)
        ].copy()

        # 按用户聚合
        user_summary = window_df.groupby('user_id').agg({
            outcome_col: 'mean',
            'is_treatment': 'first'
        }).reset_index()

        if len(user_summary) == 0:
            continue

        treated = user_summary[user_summary['is_treatment'] == 1][outcome_col].values
        control = user_summary[user_summary['is_treatment'] == 0][outcome_col].values

        if len(treated) == 0 or len(control) == 0:
            continue

        result = welch_t_test(control, treated)

        results.append({
            'window_start': start_day,
            'window_end': end_day,
            'window_mid': (start_day + end_day) / 2,
            'effect': result['effect'],
            'se': result['se'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'p_value': result['p_value']
        })

    return pd.DataFrame(results)


def surrogate_analysis(
    df: pd.DataFrame,
    short_term_metric: str,
    long_term_metric: str,
    short_term_window: int = 14,
    long_term_window: Tuple[int, int] = (90, 180)
) -> Dict:
    """
    代理指标分析

    检验短期指标能否预测长期指标

    Parameters:
    -----------
    df: 用户数据（需包含两个指标）
    short_term_metric: 短期指标列名
    long_term_metric: 长期指标列名
    short_term_window: 短期窗口
    long_term_window: 长期窗口

    Returns:
    --------
    代理指标有效性分析
    """
    # 短期指标
    short_df = df[df['day'] < short_term_window].copy()
    short_user = short_df.groupby('user_id')[short_term_metric].mean()

    # 长期指标
    long_df = df[
        (df['day'] >= long_term_window[0]) &
        (df['day'] < long_term_window[1])
    ].copy()
    long_user = long_df.groupby('user_id')[long_term_metric].mean()

    # 合并
    combined = pd.DataFrame({
        'short_term': short_user,
        'long_term': long_user
    }).dropna()

    if len(combined) == 0:
        return {'error': 'No overlapping users'}

    # 计算相关性
    correlation = combined['short_term'].corr(combined['long_term'])

    # 简单线性回归：long_term ~ short_term
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        combined['short_term'],
        combined['long_term']
    )

    return {
        'correlation': correlation,
        'r_squared': r_value ** 2,
        'slope': slope,
        'intercept': intercept,
        'p_value': p_value,
        'interpretation': (
            f"Strong surrogate (r²={r_value**2:.2f})" if r_value ** 2 > 0.5
            else f"Weak surrogate (r²={r_value**2:.2f})"
        )
    }


def holdout_analysis(
    df: pd.DataFrame,
    holdout_ratio: float = 0.1,
    ramp_up_period: int = 30,
    measurement_period: Tuple[int, int] = (90, 180),
    outcome_col: str = 'outcome',
    seed: int = 42
) -> Dict:
    """
    Holdout组分析

    保留一部分用户长期作为对照组

    Parameters:
    -----------
    df: 用户数据
    holdout_ratio: Holdout比例
    ramp_up_period: 全量上线的天数
    measurement_period: 测量窗口
    outcome_col: 结果列名
    seed: 随机种子

    Returns:
    --------
    Holdout分析结果
    """
    np.random.seed(seed)

    # 选择holdout用户
    unique_users = df['user_id'].unique()
    n_holdout = int(len(unique_users) * holdout_ratio)
    holdout_users = np.random.choice(unique_users, n_holdout, replace=False)

    df = df.copy()
    df['is_holdout'] = df['user_id'].isin(holdout_users)

    # 模拟：非holdout用户在ramp_up_period后全部切换到treatment
    # （这里简化处理，假设原本treatment的保持treatment）
    # 实际应用中，这里会模拟逐步全量上线

    # 在measurement_period测量
    measurement_df = df[
        (df['day'] >= measurement_period[0]) &
        (df['day'] < measurement_period[1])
    ].copy()

    # 聚合用户
    user_summary = measurement_df.groupby('user_id').agg({
        outcome_col: 'mean',
        'is_holdout': 'first',
        'is_treatment': 'first'
    }).reset_index()

    # Holdout (仍为control) vs Non-holdout (已全量)
    holdout_data = user_summary[user_summary['is_holdout'] == True][outcome_col].values
    non_holdout_data = user_summary[user_summary['is_holdout'] == False][outcome_col].values

    result = welch_t_test(holdout_data, non_holdout_data)

    return {
        'long_term_effect_holdout': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'n_holdout': len(holdout_data),
        'n_non_holdout': len(non_holdout_data),
        'note': 'Effect of ramped-up treatment vs holdout'
    }


def plot_time_varying_effect(
    time_effects_df: pd.DataFrame
) -> go.Figure:
    """
    绘制时变效应

    Parameters:
    -----------
    time_effects_df: 时变效应DataFrame

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_effects_df['window_mid'],
        y=time_effects_df['effect'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=time_effects_df['ci_upper'] - time_effects_df['effect'],
            arrayminus=time_effects_df['effect'] - time_effects_df['ci_lower']
        ),
        mode='markers+lines',
        marker=dict(size=8, color='#2D9CDB'),
        line=dict(color='#2D9CDB', width=2),
        name='Treatment Effect'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Treatment Effect Over Time',
        xaxis_title='Day',
        yaxis_title='Treatment Effect',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )

    return fig


def plot_short_vs_long_term(
    short_result: Dict,
    long_result: Dict
) -> go.Figure:
    """
    对比短期和长期效应

    Parameters:
    -----------
    short_result: 短期分析结果
    long_result: 长期分析结果

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    categories = ['Short-term', 'Long-term']
    effects = [short_result['short_term_effect'], long_result['long_term_effect']]
    ci_lower = [short_result['ci_lower'], long_result['ci_lower']]
    ci_upper = [short_result['ci_upper'], long_result['ci_upper']]

    fig.add_trace(go.Bar(
        x=categories,
        y=effects,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[u - e for u, e in zip(ci_upper, effects)],
            arrayminus=[e - l for e, l in zip(effects, ci_lower)]
        ),
        marker_color=['#2D9CDB', '#27AE60'],
        text=[f'{e:.2f}' for e in effects],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title='Short-term vs Long-term Effects',
        yaxis_title='Treatment Effect',
        template='plotly_white',
        height=400
    )

    return fig


if __name__ == "__main__":
    # 测试代码
    df = simulate_long_term_data(
        n_users=10000,
        n_days=180,
        short_term_effect=0.15,
        long_term_effect=0.05
    )

    print("Long-term Data Generated:")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Days: {df['day'].nunique()}")

    # 短期分析
    short = short_term_analysis(df, short_term_window=14)
    print(f"\nShort-term Effect (Day 0-14): {short['short_term_effect']:.2f}")

    # 长期分析
    long = long_term_analysis(df, long_term_window=(90, 180))
    print(f"Long-term Effect (Day 90-180): {long['long_term_effect']:.2f}")

    print(f"\nEffect Decay: {(short['short_term_effect'] - long['long_term_effect']):.2f}")
