"""
Stratified Analysis
分层分析

核心思想:
--------
在不同用户群体中分别分析实验效果
目的：发现异质性处理效应 (Heterogeneous Treatment Effects)

应用场景:
--------
1. 不同用户群体效果不同
2. 精准定向投放
3. 个性化决策
4. 理解效应机制

面试考点:
--------
- 什么是异质性处理效应？
- 分层分析 vs CATE估计的区别？
- 如何避免过度拟合？
- 多重检验如何处理？
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import two_proportion_test, welch_t_test
from .ab_testing import multiple_testing_correction


def stratified_analysis(
    df: pd.DataFrame,
    metric_col: str,
    strata_col: str,
    group_col: str = 'group',
    min_samples_per_stratum: int = 100
) -> pd.DataFrame:
    """
    分层分析

    Parameters:
    -----------
    df: 实验数据
    metric_col: 目标指标列名
    strata_col: 分层变量列名
    group_col: 分组列名
    min_samples_per_stratum: 每层最小样本量

    Returns:
    --------
    分层结果DataFrame
    """
    results = []

    for stratum_value in sorted(df[strata_col].unique()):
        stratum_df = df[df[strata_col] == stratum_value]

        control = stratum_df[stratum_df[group_col] == 'control'][metric_col].values
        treatment = stratum_df[stratum_df[group_col] == 'treatment'][metric_col].values

        # 样本量过小则跳过
        if len(control) < min_samples_per_stratum or len(treatment) < min_samples_per_stratum:
            continue

        # 判断指标类型
        is_binary = set(stratum_df[metric_col].unique()).issubset({0, 1})

        if is_binary:
            test_result = two_proportion_test(
                control.mean(), len(control),
                treatment.mean(), len(treatment)
            )
        else:
            test_result = welch_t_test(control, treatment)

        results.append({
            'stratum': stratum_value,
            'n_control': len(control),
            'n_treatment': len(treatment),
            'control_mean': control.mean(),
            'treatment_mean': treatment.mean(),
            'absolute_effect': test_result['effect'],
            'relative_effect': test_result['relative_effect'],
            'se': test_result['se'],
            'ci_lower': test_result['ci_lower'],
            'ci_upper': test_result['ci_upper'],
            'p_value': test_result['p_value'],
            'is_significant': test_result['is_significant']
        })

    return pd.DataFrame(results)


def create_quantile_strata(
    df: pd.DataFrame,
    feature_col: str,
    n_quantiles: int = 4
) -> pd.DataFrame:
    """
    创建分位数分层

    Parameters:
    -----------
    df: 数据
    feature_col: 特征列名
    n_quantiles: 分位数数量

    Returns:
    --------
    添加了分层列的DataFrame
    """
    df = df.copy()

    # 计算分位数
    quantiles = pd.qcut(
        df[feature_col],
        q=n_quantiles,
        labels=[f'Q{i+1}' for i in range(n_quantiles)],
        duplicates='drop'
    )

    df[f'{feature_col}_quantile'] = quantiles

    return df


def test_treatment_heterogeneity(
    stratified_results: pd.DataFrame,
    alpha: float = 0.05
) -> Dict:
    """
    检验处理效应异质性

    使用交互效应检验：是否不同层的效应显著不同？

    Parameters:
    -----------
    stratified_results: 分层分析结果
    alpha: 显著性水平

    Returns:
    --------
    异质性检验结果
    """
    # 方法1: 卡方异质性检验（类似meta分析）
    effects = stratified_results['absolute_effect'].values
    ses = stratified_results['se'].values
    weights = 1 / (ses ** 2)

    # 加权平均效应
    pooled_effect = np.sum(weights * effects) / np.sum(weights)

    # Q统计量
    Q = np.sum(weights * (effects - pooled_effect) ** 2)

    # 自由度
    df_Q = len(effects) - 1

    # p值
    p_value = 1 - stats.chi2.cdf(Q, df_Q)

    # I²统计量（异质性程度）
    I_squared = max(0, (Q - df_Q) / Q) if Q > 0 else 0

    return {
        'Q_statistic': Q,
        'df': df_Q,
        'p_value': p_value,
        'has_heterogeneity': p_value < alpha,
        'I_squared': I_squared,
        'interpretation': (
            f"显著异质性 (I² = {I_squared*100:.1f}%)" if p_value < alpha
            else f"无显著异质性 (I² = {I_squared*100:.1f}%)"
        )
    }


def pooled_estimate(
    stratified_results: pd.DataFrame,
    method: str = 'inverse_variance'
) -> Dict:
    """
    汇总分层估计

    Parameters:
    -----------
    stratified_results: 分层分析结果
    method: 汇总方法
        - 'inverse_variance': 逆方差加权
        - 'sample_size': 样本量加权
        - 'simple': 简单平均

    Returns:
    --------
    汇总估计结果
    """
    effects = stratified_results['absolute_effect'].values
    ses = stratified_results['se'].values

    if method == 'inverse_variance':
        weights = 1 / (ses ** 2)
    elif method == 'sample_size':
        weights = (stratified_results['n_control'] + stratified_results['n_treatment']).values
    else:  # simple
        weights = np.ones(len(effects))

    # 归一化权重
    weights = weights / weights.sum()

    # 汇总效应
    pooled_effect = np.sum(weights * effects)

    # 汇总标准误
    if method == 'inverse_variance':
        pooled_se = np.sqrt(1 / np.sum(1 / (ses ** 2)))
    else:
        pooled_se = np.sqrt(np.sum(weights ** 2 * ses ** 2))

    # 置信区间
    z = stats.norm.ppf(0.975)
    ci_lower = pooled_effect - z * pooled_se
    ci_upper = pooled_effect + z * pooled_se

    # p值
    z_stat = pooled_effect / pooled_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        'pooled_effect': pooled_effect,
        'pooled_se': pooled_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'weights': weights,
        'method': method
    }


def plot_stratified_effects(
    stratified_results: pd.DataFrame,
    strata_col: str = 'stratum',
    title: str = 'Stratified Treatment Effects'
) -> go.Figure:
    """
    绘制分层效应图

    Parameters:
    -----------
    stratified_results: 分层分析结果
    strata_col: 分层列名
    title: 图表标题

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    # 提取数据
    strata = stratified_results[strata_col].astype(str).values
    effects = stratified_results['relative_effect'].values * 100

    # 计算相对置信区间
    ci_lower = []
    ci_upper = []
    for _, row in stratified_results.iterrows():
        if row['control_mean'] > 0:
            ci_lower.append(row['ci_lower'] / row['control_mean'] * 100)
            ci_upper.append(row['ci_upper'] / row['control_mean'] * 100)
        else:
            ci_lower.append(0)
            ci_upper.append(0)

    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # 颜色编码
    colors = []
    for _, row in stratified_results.iterrows():
        if row['is_significant'] and row['relative_effect'] > 0:
            colors.append('#27AE60')
        elif row['is_significant'] and row['relative_effect'] < 0:
            colors.append('#EB5757')
        else:
            colors.append('#6B7280')

    # 添加点和置信区间
    fig.add_trace(go.Scatter(
        x=strata,
        y=effects,
        error_y=dict(
            type='data',
            symmetric=False,
            array=ci_upper - effects,
            arrayminus=effects - ci_lower
        ),
        mode='markers',
        marker=dict(size=12, color=colors),
        name='Effect'
    ))

    # 添加参考线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # 添加汇总估计
    pooled = pooled_estimate(stratified_results)
    pooled_effect = pooled['pooled_effect']

    # 计算汇总的相对效应（近似）
    avg_control_mean = stratified_results['control_mean'].mean()
    if avg_control_mean > 0:
        pooled_relative = pooled_effect / avg_control_mean * 100
        fig.add_hline(
            y=pooled_relative,
            line_dash="dot",
            line_color="#2D9CDB",
            annotation_text=f"Pooled: {pooled_relative:.1f}%",
            annotation_position="right"
        )

    fig.update_layout(
        title=title,
        xaxis_title='Stratum',
        yaxis_title='Relative Effect (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def plot_stratum_sizes(
    stratified_results: pd.DataFrame,
    strata_col: str = 'stratum'
) -> go.Figure:
    """
    绘制各层样本量分布

    Parameters:
    -----------
    stratified_results: 分层分析结果
    strata_col: 分层列名

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    strata = stratified_results[strata_col].astype(str).values

    fig.add_trace(go.Bar(
        x=strata,
        y=stratified_results['n_control'],
        name='Control',
        marker_color='#2D9CDB'
    ))

    fig.add_trace(go.Bar(
        x=strata,
        y=stratified_results['n_treatment'],
        name='Treatment',
        marker_color='#27AE60'
    ))

    fig.update_layout(
        title='Sample Sizes by Stratum',
        xaxis_title='Stratum',
        yaxis_title='Sample Size',
        barmode='group',
        template='plotly_white',
        height=400
    )

    return fig


def multi_dimensional_stratification(
    df: pd.DataFrame,
    metric_col: str,
    strata_cols: List[str],
    group_col: str = 'group',
    min_samples: int = 100
) -> pd.DataFrame:
    """
    多维分层分析

    Parameters:
    -----------
    df: 实验数据
    metric_col: 目标指标列名
    strata_cols: 分层变量列名列表
    group_col: 分组列名
    min_samples: 最小样本量

    Returns:
    --------
    多维分层结果DataFrame
    """
    # 创建组合分层
    df = df.copy()
    df['_combined_stratum'] = df[strata_cols].astype(str).agg('_'.join, axis=1)

    # 执行分层分析
    results = stratified_analysis(
        df,
        metric_col=metric_col,
        strata_col='_combined_stratum',
        group_col=group_col,
        min_samples_per_stratum=min_samples
    )

    # 拆分组合分层
    for i, col in enumerate(strata_cols):
        results[col] = results['stratum'].str.split('_').str[i]

    return results


if __name__ == "__main__":
    # 测试代码
    from .utils import generate_ab_data

    df = generate_ab_data(n_control=10000, n_treatment=10000, add_covariates=True)

    # 创建分位数分层
    df = create_quantile_strata(df, 'user_activity', n_quantiles=4)

    # 分层分析
    results = stratified_analysis(
        df,
        metric_col='converted',
        strata_col='user_activity_quantile'
    )

    print("Stratified Analysis Results:")
    print(results[['stratum', 'relative_effect', 'p_value', 'is_significant']])

    # 异质性检验
    hetero = test_treatment_heterogeneity(results)
    print(f"\nHeterogeneity Test: {hetero['interpretation']}")

    # 汇总估计
    pooled = pooled_estimate(results)
    print(f"\nPooled Effect: {pooled['pooled_effect']:.4f} ± {pooled['pooled_se']:.4f}")
