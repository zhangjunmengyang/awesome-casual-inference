"""
Utility functions for experimentation methods
实验方法工具函数

功能：
-----
1. 数据生成
2. 统计检验
3. 可视化辅助
4. 常用计算
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_ab_data(
    n_control: int = 10000,
    n_treatment: int = 10000,
    baseline_rate: float = 0.05,
    treatment_effect: float = 0.10,
    add_covariates: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成A/B测试数据

    Parameters:
    -----------
    n_control: 对照组样本量
    n_treatment: 实验组样本量
    baseline_rate: 基线转化率
    treatment_effect: 处理效应（相对提升）
    add_covariates: 是否添加协变量
    seed: 随机种子

    Returns:
    --------
    实验数据 DataFrame
    """
    np.random.seed(seed)

    n_total = n_control + n_treatment

    # 用户特征
    if add_covariates:
        user_activity = np.random.beta(2, 5, n_total)
        historical_conversion = np.random.beta(2, 8, n_total)
        age = np.random.normal(35, 10, n_total).clip(18, 65)
        is_mobile = np.random.binomial(1, 0.6, n_total)
    else:
        user_activity = np.zeros(n_total)
        historical_conversion = np.zeros(n_total)
        age = np.ones(n_total) * 35
        is_mobile = np.zeros(n_total)

    # 分组
    group = np.array(['control'] * n_control + ['treatment'] * n_treatment)
    np.random.shuffle(group)
    is_treatment = (group == 'treatment').astype(int)

    # 基线转化概率
    base_logit = np.log(baseline_rate / (1 - baseline_rate))

    if add_covariates:
        user_effect = (
            1.5 * (user_activity - 0.3) +
            1.0 * (historical_conversion - 0.2) +
            0.005 * (age - 35)
        )
    else:
        user_effect = 0

    base_prob = 1 / (1 + np.exp(-(base_logit + user_effect)))

    # 处理效应
    treatment_effect_individual = base_prob * treatment_effect

    # 实际转化概率
    prob = base_prob + is_treatment * treatment_effect_individual
    prob = np.clip(prob, 0, 1)

    # 转化
    converted = np.random.binomial(1, prob)

    # 收入（条件于转化）
    revenue = np.where(
        converted == 1,
        np.random.lognormal(4, 0.8, n_total),
        0
    )

    # 构建DataFrame
    df = pd.DataFrame({
        'user_id': range(n_total),
        'group': group,
        'is_treatment': is_treatment,
        'converted': converted,
        'revenue': revenue,
    })

    if add_covariates:
        df['user_activity'] = user_activity
        df['historical_conversion'] = historical_conversion
        df['age'] = age
        df['is_mobile'] = is_mobile

    return df


def two_proportion_test(
    p1: float,
    n1: int,
    p2: float,
    n2: int,
    alpha: float = 0.05
) -> Dict:
    """
    两比例 Z 检验

    Parameters:
    -----------
    p1: 组1的比例
    n1: 组1的样本量
    p2: 组2的比例
    n2: 组2的样本量
    alpha: 显著性水平

    Returns:
    --------
    检验结果字典
    """
    # 效应
    effect = p2 - p1
    relative_effect = effect / p1 if p1 > 0 else 0

    # 标准误
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    # Z统计量
    z_stat = effect / se

    # p值（双侧）
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 置信区间
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = effect - z_crit * se
    ci_upper = effect + z_crit * se

    return {
        'effect': effect,
        'relative_effect': relative_effect,
        'se': se,
        'z_stat': z_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'is_significant': p_value < alpha
    }


def welch_t_test(
    data1: np.ndarray,
    data2: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Welch's t 检验（不假设方差齐性）

    Parameters:
    -----------
    data1: 组1数据
    data2: 组2数据
    alpha: 显著性水平

    Returns:
    --------
    检验结果字典
    """
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = data1.mean(), data2.mean()
    var1, var2 = data1.var(ddof=1), data2.var(ddof=1)

    # 效应
    effect = mean2 - mean1
    relative_effect = effect / mean1 if mean1 > 0 else 0

    # 标准误
    se = np.sqrt(var1 / n1 + var2 / n2)

    # t统计量
    t_stat = effect / se

    # Welch-Satterthwaite自由度
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )

    # p值（双侧）
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # 置信区间
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = effect - t_crit * se
    ci_upper = effect + t_crit * se

    return {
        'effect': effect,
        'relative_effect': relative_effect,
        'se': se,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'is_significant': p_value < alpha
    }


def bootstrap_ci(
    data1: np.ndarray,
    data2: np.ndarray,
    stat_func=lambda x: x.mean(),
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap置信区间

    Parameters:
    -----------
    data1: 对照组数据
    data2: 实验组数据
    stat_func: 统计量函数
    n_bootstrap: Bootstrap次数
    alpha: 显著性水平
    seed: 随机种子

    Returns:
    --------
    (点估计, 置信区间下界, 置信区间上界)
    """
    np.random.seed(seed)

    # 观测效应
    observed_effect = stat_func(data2) - stat_func(data1)

    # Bootstrap
    bootstrap_effects = []
    n1, n2 = len(data1), len(data2)

    for _ in range(n_bootstrap):
        boot_data1 = np.random.choice(data1, n1, replace=True)
        boot_data2 = np.random.choice(data2, n2, replace=True)
        boot_effect = stat_func(boot_data2) - stat_func(boot_data1)
        bootstrap_effects.append(boot_effect)

    bootstrap_effects = np.array(bootstrap_effects)

    # 百分位置信区间
    ci_lower = np.percentile(bootstrap_effects, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_effects, (1 - alpha / 2) * 100)

    return observed_effect, ci_lower, ci_upper


def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0
) -> int:
    """
    计算所需样本量（每组）

    Parameters:
    -----------
    baseline_rate: 基线转化率
    mde: 最小可检测效应（绝对值）
    alpha: 显著性水平
    power: 统计功效
    ratio: 实验组/对照组比例

    Returns:
    --------
    每组所需样本量
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + mde

    pooled_p = (p1 + ratio * p2) / (1 + ratio)

    n = (z_alpha * np.sqrt((1 + 1/ratio) * pooled_p * (1 - pooled_p)) +
         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2 / mde ** 2

    return int(np.ceil(n))


def plot_distribution_comparison(
    control_data: np.ndarray,
    treatment_data: np.ndarray,
    metric_name: str = 'Metric',
    bins: int = 50
) -> go.Figure:
    """
    绘制分布对比图

    Parameters:
    -----------
    control_data: 对照组数据
    treatment_data: 实验组数据
    metric_name: 指标名称
    bins: 直方图分箱数

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    # 对照组
    fig.add_trace(go.Histogram(
        x=control_data,
        name='Control',
        opacity=0.6,
        marker_color='#2D9CDB',
        nbinsx=bins,
        histnorm='probability density'
    ))

    # 实验组
    fig.add_trace(go.Histogram(
        x=treatment_data,
        name='Treatment',
        opacity=0.6,
        marker_color='#27AE60',
        nbinsx=bins,
        histnorm='probability density'
    ))

    # 添加均值线
    fig.add_vline(
        x=control_data.mean(),
        line_dash="dash",
        line_color="#2D9CDB",
        annotation_text=f"Control Mean: {control_data.mean():.3f}"
    )

    fig.add_vline(
        x=treatment_data.mean(),
        line_dash="dash",
        line_color="#27AE60",
        annotation_text=f"Treatment Mean: {treatment_data.mean():.3f}"
    )

    fig.update_layout(
        title=f'{metric_name} Distribution Comparison',
        xaxis_title=metric_name,
        yaxis_title='Density',
        barmode='overlay',
        template='plotly_white',
        height=400
    )

    return fig


def plot_ci_comparison(
    results: List[Dict],
    metric_names: List[str]
) -> go.Figure:
    """
    绘制置信区间对比图

    Parameters:
    -----------
    results: 检验结果列表
    metric_names: 指标名称列表

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    effects = [r['relative_effect'] * 100 for r in results]
    ci_lower = [(r['ci_lower'] / r.get('baseline', 1)) * 100 for r in results]
    ci_upper = [(r['ci_upper'] / r.get('baseline', 1)) * 100 for r in results]

    colors = ['#27AE60' if r['is_significant'] and r['relative_effect'] > 0
              else '#EB5757' if r['is_significant']
              else '#6B7280' for r in results]

    fig.add_trace(go.Bar(
        x=metric_names,
        y=effects,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[u - e for u, e in zip(ci_upper, effects)],
            arrayminus=[e - l for e, l in zip(effects, ci_lower)]
        ),
        marker_color=colors,
        text=[f'{e:.1f}%' for e in effects],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title='Treatment Effects with Confidence Intervals',
        xaxis_title='Metric',
        yaxis_title='Relative Effect (%)',
        template='plotly_white',
        height=400
    )

    return fig


def format_api_response(
    charts: List[go.Figure],
    tables: List[pd.DataFrame],
    summary: str,
    metrics: Dict
) -> Dict:
    """
    格式化API响应

    Parameters:
    -----------
    charts: Plotly图表列表
    tables: 数据表格列表
    summary: 文字总结
    metrics: 关键指标字典

    Returns:
    --------
    标准化的API响应字典
    """
    return {
        'charts': [chart.to_json() for chart in charts],
        'tables': [table.to_dict(orient='records') for table in tables],
        'summary': summary,
        'metrics': metrics
    }


def calculate_power(
    n: int,
    baseline_rate: float,
    effect: float,
    alpha: float = 0.05
) -> float:
    """
    计算统计功效

    Parameters:
    -----------
    n: 每组样本量
    baseline_rate: 基线转化率
    effect: 效应大小（绝对值）
    alpha: 显著性水平

    Returns:
    --------
    统计功效
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    p1 = baseline_rate
    p2 = baseline_rate + effect

    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)

    z_beta = effect / se - z_alpha
    power = stats.norm.cdf(z_beta)

    return max(0, min(1, power))
