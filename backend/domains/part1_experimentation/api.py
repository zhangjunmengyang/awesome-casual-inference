"""
API Adapter Layer for Part 1: Experimentation Methods
实验方法API适配层

统一API接口，所有函数返回标准格式:
{
    "charts": [...],      # Plotly图表JSON列表
    "tables": [...],      # 数据表格列表
    "summary": "...",     # 文字总结
    "metrics": {...}      # 关键指标字典
}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from . import ab_testing
from . import cuped
from . import stratified_analysis as strat
from . import network_effects as network
from . import switchback as sb
from . import long_term_effects as lte
from . import multi_armed_bandits as mab
from . import utils


def analyze_ab_test(
    n_control: int = 10000,
    n_treatment: int = 10000,
    baseline_rate: float = 0.05,
    treatment_effect: float = 0.10,
    metrics: List[str] = None,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict:
    """
    A/B测试完整分析

    Parameters:
    -----------
    n_control: 对照组样本量
    n_treatment: 实验组样本量
    baseline_rate: 基线转化率
    treatment_effect: 处理效应
    metrics: 要分析的指标列表
    alpha: 显著性水平
    seed: 随机种子

    Returns:
    --------
    标准API响应
    """
    # 生成数据
    df = utils.generate_ab_data(
        n_control=n_control,
        n_treatment=n_treatment,
        baseline_rate=baseline_rate,
        treatment_effect=treatment_effect,
        add_covariates=True,
        seed=seed
    )

    # 默认指标
    if metrics is None:
        metrics = ['converted', 'revenue']

    # 分析器
    analyzer = ab_testing.ABTestAnalyzer(alpha=alpha)

    # 分析每个指标
    results = []
    for metric in metrics:
        if metric in df.columns:
            result = analyzer.analyze_metric(df, metric)
            results.append(result)

    # SRM检查
    srm = ab_testing.check_srm(df)

    # 平衡性检查
    covariate_cols = ['user_activity', 'historical_conversion', 'age', 'is_mobile']
    balance_df = analyzer.check_balance(df, covariate_cols)

    # 生成图表
    charts = []

    # 1. 结果总览
    fig_results = ab_testing.plot_ab_results(results, title='A/B Test Results')
    charts.append(fig_results.to_json())

    # 2. 样本量曲线
    fig_sample_size = ab_testing.plot_sample_size_curve(
        baseline_rate=baseline_rate,
        mde_range=(0.01, 0.30),
        power_levels=[0.70, 0.80, 0.90]
    )
    charts.append(fig_sample_size.to_json())

    # 生成表格
    tables = []

    # 结果表
    results_table = pd.DataFrame([{
        'metric': r.metric_name,
        'control_mean': f"{r.control_mean:.4f}",
        'treatment_mean': f"{r.treatment_mean:.4f}",
        'relative_effect': f"{r.relative_effect*100:.2f}%",
        'p_value': f"{r.p_value:.4f}",
        'significant': '✓' if r.is_significant else '✗'
    } for r in results])
    tables.append(results_table.to_dict(orient='records'))

    # 平衡性表
    tables.append(balance_df.to_dict(orient='records'))

    # 生成总结
    primary_result = results[0] if results else None
    if primary_result:
        if primary_result.is_significant:
            decision = "**建议上线**" if primary_result.relative_effect > 0 else "**不建议上线**"
            effect_desc = f"显著的{'正向' if primary_result.relative_effect > 0 else '负向'}效应"
        else:
            decision = "**继续观察或调整**"
            effect_desc = "未检测到显著效应"

        summary = f"""
## A/B测试分析报告

### 实验配置
- 对照组样本量: {n_control:,}
- 实验组样本量: {n_treatment:,}
- 显著性水平: {alpha}

### 核心发现
主指标 **{primary_result.metric_name}**:
- 对照组: {primary_result.control_mean:.4f}
- 实验组: {primary_result.treatment_mean:.4f}
- 相对变化: **{primary_result.relative_effect*100:+.2f}%**
- p值: {primary_result.p_value:.4f}

### 决策建议
{decision}

{effect_desc}，置信区间为 [{primary_result.ci_lower*100:.3f}%, {primary_result.ci_upper*100:.3f}%]

### 数据质量
- SRM检查: {srm['message']}
- 平衡性: {'通过' if balance_df['is_balanced'].all() else '部分协变量不平衡'}
"""
    else:
        summary = "未能完成分析"

    # 关键指标
    metrics_dict = {}
    if primary_result:
        metrics_dict = {
            'effect': primary_result.relative_effect,
            'p_value': primary_result.p_value,
            'is_significant': primary_result.is_significant,
            'n_total': n_control + n_treatment,
            'srm_ok': not srm['has_srm']
        }

    return {
        'charts': charts,
        'tables': tables,
        'summary': summary,
        'metrics': metrics_dict
    }


def apply_cuped(
    metric_col: str = 'converted',
    covariate_col: str = 'historical_conversion',
    n_samples: int = 10000,
    seed: int = 42
) -> Dict:
    """
    CUPED方差缩减分析

    Parameters:
    -----------
    metric_col: 目标指标
    covariate_col: 协变量
    n_samples: 样本量
    seed: 随机种子

    Returns:
    --------
    标准API响应
    """
    # 生成数据
    df = utils.generate_ab_data(
        n_control=n_samples // 2,
        n_treatment=n_samples // 2,
        add_covariates=True,
        seed=seed
    )

    # 比较CUPED前后
    comparison = cuped.compare_with_without_cuped(df, metric_col, covariate_col)

    # 协变量选择
    candidate_covariates = ['user_activity', 'historical_conversion', 'age']
    best_cov, selection_df = cuped.select_best_covariate(df, metric_col, candidate_covariates)

    # 生成图表
    charts = []

    # 1. CUPED对比图
    fig_cuped = cuped.plot_cuped_comparison(df, metric_col, covariate_col)
    charts.append(fig_cuped.to_json())

    # 2. 协变量选择
    fig_selection = cuped.plot_variance_reduction_curve(df, metric_col, candidate_covariates)
    charts.append(fig_selection.to_json())

    # 生成表格
    tables = []

    # 对比表
    comparison_table = pd.DataFrame([
        {
            'method': 'Original',
            'effect': f"{comparison['original']['effect']:.4f}",
            'se': f"{comparison['original']['se']:.5f}",
            'p_value': f"{comparison['original']['p_value']:.4f}"
        },
        {
            'method': 'CUPED',
            'effect': f"{comparison['cuped']['effect']:.4f}",
            'se': f"{comparison['cuped']['se']:.5f}",
            'p_value': f"{comparison['cuped']['p_value']:.4f}"
        }
    ])
    tables.append(comparison_table.to_dict(orient='records'))

    # 协变量选择表
    tables.append(selection_df.to_dict(orient='records'))

    # 生成总结
    var_reduction = comparison['improvement']['variance_reduction']
    se_reduction = comparison['improvement']['se_reduction']

    summary = f"""
## CUPED方差缩减分析

### 效果总结
使用协变量 **{covariate_col}** 进行CUPED调整:

- 方差缩减: **{var_reduction*100:.1f}%**
- 标准误缩减: **{se_reduction*100:.1f}%**
- 置信区间宽度缩减: **{comparison['improvement']['ci_width_reduction']*100:.1f}%**

### 原始 vs CUPED
- 原始p值: {comparison['original']['p_value']:.4f}
- CUPED p值: {comparison['cuped']['p_value']:.4f}

### 最佳协变量
推荐使用 **{best_cov}**，预计方差缩减 {selection_df.iloc[0]['variance_reduction']*100:.1f}%

### 优势
1. 更高的统计功效
2. 更小的样本量需求
3. 更短的实验周期
"""

    metrics_dict = {
        'variance_reduction': var_reduction,
        'se_reduction': se_reduction,
        'best_covariate': best_cov
    }

    return {
        'charts': charts,
        'tables': tables,
        'summary': summary,
        'metrics': metrics_dict
    }


def stratified_analysis(
    metric_col: str = 'converted',
    strata_col: str = 'user_activity',
    n_quantiles: int = 4,
    n_samples: int = 10000,
    seed: int = 42
) -> Dict:
    """
    分层分析

    Parameters:
    -----------
    metric_col: 目标指标
    strata_col: 分层变量
    n_quantiles: 分位数数量
    n_samples: 样本量
    seed: 随机种子

    Returns:
    --------
    标准API响应
    """
    # 生成数据
    df = utils.generate_ab_data(
        n_control=n_samples // 2,
        n_treatment=n_samples // 2,
        add_covariates=True,
        seed=seed
    )

    # 创建分位数分层
    df = strat.create_quantile_strata(df, strata_col, n_quantiles)

    # 分层分析 (调整最小样本量要求)
    min_samples = max(30, n_samples // (n_quantiles * 10))
    results_df = strat.stratified_analysis(
        df,
        metric_col=metric_col,
        strata_col=f'{strata_col}_quantile',
        min_samples_per_stratum=min_samples
    )

    # 检查是否有有效的分层结果
    if len(results_df) == 0:
        return {
            'charts': [],
            'tables': [],
            'summary': f"样本量过小（{n_samples}），无法进行分层分析。建议至少 {n_quantiles * 200} 个样本。",
            'metrics': {'has_heterogeneity': False, 'pooled_effect': 0, 'n_strata': 0}
        }

    # 异质性检验
    hetero_test = strat.test_treatment_heterogeneity(results_df)

    # 汇总估计
    pooled = strat.pooled_estimate(results_df)

    # 生成图表
    charts = []

    # 1. 分层效应图
    fig_effects = strat.plot_stratified_effects(
        results_df,
        strata_col='stratum',
        title=f'Treatment Effects by {strata_col}'
    )
    charts.append(fig_effects.to_json())

    # 2. 样本量分布
    fig_sizes = strat.plot_stratum_sizes(results_df)
    charts.append(fig_sizes.to_json())

    # 生成表格
    tables = []
    tables.append(results_df[['stratum', 'relative_effect', 'p_value', 'is_significant']].to_dict(orient='records'))

    # 生成总结
    summary = f"""
## 分层分析报告

### 异质性检验
{hetero_test['interpretation']}

### 分层结果
在 {n_quantiles} 个分层中:
- 显著正向效应: {(results_df['is_significant'] & (results_df['relative_effect'] > 0)).sum()} 层
- 显著负向效应: {(results_df['is_significant'] & (results_df['relative_effect'] < 0)).sum()} 层
- 不显著: {(~results_df['is_significant']).sum()} 层

### 汇总估计
- 汇总效应: {pooled['pooled_effect']:.4f}
- 标准误: {pooled['pooled_se']:.5f}
- p值: {pooled['p_value']:.4f}

### 建议
{'发现显著的异质性效应！可以考虑针对不同群体采取不同策略。' if hetero_test['has_heterogeneity'] else '各群体效应相对一致，可以统一上线。'}
"""

    metrics_dict = {
        'has_heterogeneity': hetero_test['has_heterogeneity'],
        'pooled_effect': pooled['pooled_effect'],
        'n_strata': len(results_df)
    }

    return {
        'charts': charts,
        'tables': tables,
        'summary': summary,
        'metrics': metrics_dict
    }


def analyze_network_effects(
    n_users: int = 1000,
    avg_degree: int = 10,
    direct_effect: float = 0.15,
    spillover_effect: float = 0.05,
    seed: int = 42
) -> Dict:
    """
    网络效应分析

    Returns:
    --------
    标准API响应
    """
    # 模拟数据
    df, adjacency = network.simulate_network_data(
        n_users=n_users,
        avg_degree=avg_degree,
        treatment_effect_direct=direct_effect,
        treatment_effect_spillover=spillover_effect,
        seed=seed
    )

    # 朴素估计
    naive = network.naive_ate_biased(df)

    # 效应分解
    decomp = network.decompose_total_effect(df)

    # 溢出效应
    spillover = network.estimate_spillover_effect(df)

    # 生成图表
    charts = []
    fig = network.plot_network_effects(df)
    charts.append(fig.to_json())

    # 生成表格
    tables = []
    results_table = pd.DataFrame([
        {'component': 'Direct Effect', 'estimate': f"{decomp.get('direct_effect', 0):.4f}"},
        {'component': 'Spillover Effect', 'estimate': f"{decomp.get('spillover_effect', 0):.4f}"},
        {'component': 'Total (Naive)', 'estimate': f"{decomp.get('total_effect_naive', 0):.4f}"}
    ])
    tables.append(results_table.to_dict(orient='records'))

    summary = f"""
## 网络效应分析

### 效应分解
- 直接效应: {decomp.get('direct_effect', 0):.4f}
- 溢出效应: {spillover.get('spillover_effect', 0):.4f}
- 朴素总效应: {naive['ate_naive']:.4f} (有偏！)

### 警告
朴素A/B测试假设违反SUTVA，结果可能有偏。
建议使用Cluster随机化或Switchback实验设计。
"""

    return {
        'charts': charts,
        'tables': tables,
        'summary': summary,
        'metrics': decomp
    }


def analyze_switchback(
    n_units: int = 50,
    n_periods: int = 100,
    treatment_effect: float = 0.10,
    carryover: float = 0.03,
    seed: int = 42
) -> Dict:
    """
    Switchback实验分析

    Returns:
    --------
    标准API响应
    """
    # 生成数据
    df = sb.generate_switchback_data(
        n_units=n_units,
        n_time_periods=n_periods,
        treatment_effect=treatment_effect,
        carryover_effect=carryover,
        seed=seed
    )

    # 分析
    naive = sb.naive_switchback_analysis(df)
    fe = sb.fixed_effects_analysis(df)
    carryover_test = sb.detect_carryover(df)

    # 生成图表
    charts = []

    # 时间线图
    fig_timeline = sb.plot_switchback_timeline(df, unit_id=0)
    charts.append(fig_timeline.to_json())

    # 效应时间图
    fig_time = sb.plot_treatment_effect_over_time(df)
    charts.append(fig_time.to_json())

    # 生成表格
    tables = []
    comparison_table = pd.DataFrame([
        {'method': 'Naive', 'effect': f"{naive['ate_naive']:.2f}", 'p_value': f"{naive['p_value']:.4f}"},
        {'method': 'Fixed Effects', 'effect': f"{fe['ate_fe']:.2f}", 'p_value': f"{fe['p_value']:.4f}"}
    ])
    tables.append(comparison_table.to_dict(orient='records'))

    summary = f"""
## Switchback实验分析

### 效应估计
- 朴素估计: {naive['ate_naive']:.2f}
- 固定效应估计: {fe['ate_fe']:.2f}

### 残留效应检测
{carryover_test.get('interpretation', 'N/A')}

### 建议
{'检测到残留效应，建议增加切换间隔或使用更长的wash-out期。' if carryover_test.get('is_significant') else '未检测到显著残留效应，当前设计合理。'}
"""

    return {
        'charts': charts,
        'tables': tables,
        'summary': summary,
        'metrics': {'ate_fe': fe['ate_fe'], 'has_carryover': carryover_test.get('is_significant', False)}
    }


def estimate_long_term_effects(
    n_users: int = 10000,
    n_days: int = 180,
    short_term_effect: float = 0.15,
    long_term_effect: float = 0.05,
    seed: int = 42
) -> Dict:
    """
    长期效应估计

    Returns:
    --------
    标准API响应
    """
    # 生成数据
    df = lte.simulate_long_term_data(
        n_users=n_users,
        n_days=n_days,
        short_term_effect=short_term_effect,
        long_term_effect=long_term_effect,
        seed=seed
    )

    # 短期分析
    short = lte.short_term_analysis(df, short_term_window=14)

    # 长期分析
    long = lte.long_term_analysis(df, long_term_window=(90, 180))

    # 时变效应
    time_effects = lte.time_varying_effect(df, window_size=7)

    # 生成图表
    charts = []

    fig_time = lte.plot_time_varying_effect(time_effects)
    charts.append(fig_time.to_json())

    fig_compare = lte.plot_short_vs_long_term(short, long)
    charts.append(fig_compare.to_json())

    # 生成表格
    tables = []
    tables.append(time_effects[['window_start', 'window_end', 'effect', 'p_value']].to_dict(orient='records'))

    summary = f"""
## 长期效应分析

### 短期 vs 长期
- 短期效应 (Day 0-14): {short['short_term_effect']:.2f}
- 长期效应 (Day 90-180): {long['long_term_effect']:.2f}
- 效应衰减: {(short['short_term_effect'] - long['long_term_effect']):.2f}

### 建议
{'短期效应高于长期，注意长期监控。' if short['short_term_effect'] > long['long_term_effect'] else '效应稳定或增长。'}
"""

    return {
        'charts': charts,
        'tables': tables,
        'summary': summary,
        'metrics': {'short_term': short['short_term_effect'], 'long_term': long['long_term_effect']}
    }


def run_bandit_simulation(
    arm_means: List[float] = None,
    algorithm: str = 'thompson',
    n_rounds: int = 1000,
    seed: int = 42
) -> Dict:
    """
    多臂老虎机模拟

    Parameters:
    -----------
    arm_means: 各臂的真实均值
    algorithm: 算法 ('epsilon', 'thompson', 'ucb')
    n_rounds: 模拟轮数
    seed: 随机种子

    Returns:
    --------
    标准API响应
    """
    if arm_means is None:
        arm_means = [0.5, 0.6, 0.55, 0.7]

    # 创建臂
    arms = [mab.Arm(true_mean=m, name=f"Arm {chr(65+i)} (μ={m:.2f})") for i, m in enumerate(arm_means)]

    # 选择算法
    if algorithm == 'epsilon':
        algo = mab.EpsilonGreedy(epsilon=0.1)
    elif algorithm == 'thompson':
        algo = mab.ThompsonSampling()
    else:  # ucb
        algo = mab.UCB(confidence=2.0)

    # 模拟
    history, total_regret = mab.simulate_bandit(arms, algo, n_rounds=n_rounds, seed=seed)

    # 比较算法
    algorithms = [
        mab.EpsilonGreedy(epsilon=0.1),
        mab.ThompsonSampling(),
        mab.UCB(confidence=2.0)
    ]

    arms_reset = [mab.Arm(true_mean=m, name=f"Arm {chr(65+i)}") for i, m in enumerate(arm_means)]
    comparison_df = mab.compare_algorithms(arms_reset, algorithms, n_rounds=n_rounds, n_simulations=50, seed=seed)

    # 生成图表
    charts = []

    fig_sim = mab.plot_bandit_simulation(history, arms)
    charts.append(fig_sim.to_json())

    fig_compare = mab.plot_algorithm_comparison(comparison_df)
    charts.append(fig_compare.to_json())

    # 生成表格
    tables = []
    tables.append(comparison_df.to_dict(orient='records'))

    summary = f"""
## 多臂老虎机模拟

### 算法: {algo.name}
- 总轮数: {n_rounds}
- 累积Regret: {total_regret:.2f}

### 最终拉动次数
{chr(10).join([f"- {arm.name}: {arm.pulls} 次" for arm in arms])}

### 最优臂
最优臂是 Arm {chr(65 + np.argmax(arm_means))} (μ={max(arm_means):.2f})
"""

    return {
        'charts': charts,
        'tables': tables,
        'summary': summary,
        'metrics': {'total_regret': total_regret, 'best_arm': int(np.argmax(arm_means))}
    }
