"""
A/B Testing Basics
A/B测试基础

核心内容:
--------
1. 实验设计 - 样本量计算、功效分析
2. 随机化 - 分流策略、平衡检查
3. 统计检验 - t检验、z检验
4. SRM检测 - 样本比例不匹配
5. 多重检验校正 - Bonferroni、FDR

面试考点:
--------
- A/B测试的基本原理
- 如何计算样本量
- 什么是SRM，如何检测
- 如何处理多重检验问题
- t检验和z检验的区别
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    generate_ab_data,
    two_proportion_test,
    welch_t_test,
    plot_distribution_comparison,
    plot_ci_comparison
)


@dataclass
class ABTestResult:
    """A/B测试结果"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    n_control: int
    n_treatment: int
    test_type: str  # 'proportion' or 'continuous'


class ABTestAnalyzer:
    """A/B测试分析器"""

    def __init__(self, alpha: float = 0.05):
        """
        Parameters:
        -----------
        alpha: 显著性水平
        """
        self.alpha = alpha
        self.results = {}

    def analyze_metric(
        self,
        df: pd.DataFrame,
        metric_col: str,
        group_col: str = 'group',
        control_label: str = 'control',
        treatment_label: str = 'treatment',
        test_type: str = 'auto'
    ) -> ABTestResult:
        """
        分析单个指标

        Parameters:
        -----------
        df: 实验数据
        metric_col: 指标列名
        group_col: 分组列名
        control_label: 对照组标签
        treatment_label: 实验组标签
        test_type: 检验类型 ('auto', 'proportion', 'continuous')

        Returns:
        --------
        ABTestResult对象
        """
        control_data = df[df[group_col] == control_label][metric_col].values
        treatment_data = df[df[group_col] == treatment_label][metric_col].values

        n_control = len(control_data)
        n_treatment = len(treatment_data)

        # 自动判断检验类型
        if test_type == 'auto':
            unique_values = df[metric_col].nunique()
            if unique_values == 2 and set(df[metric_col].unique()).issubset({0, 1}):
                test_type = 'proportion'
            else:
                test_type = 'continuous'

        # 执行检验
        if test_type == 'proportion':
            p_control = control_data.mean()
            p_treatment = treatment_data.mean()

            result_dict = two_proportion_test(
                p_control, n_control,
                p_treatment, n_treatment,
                self.alpha
            )

            result = ABTestResult(
                metric_name=metric_col,
                control_mean=p_control,
                treatment_mean=p_treatment,
                absolute_effect=result_dict['effect'],
                relative_effect=result_dict['relative_effect'],
                se=result_dict['se'],
                ci_lower=result_dict['ci_lower'],
                ci_upper=result_dict['ci_upper'],
                p_value=result_dict['p_value'],
                is_significant=result_dict['is_significant'],
                n_control=n_control,
                n_treatment=n_treatment,
                test_type='proportion'
            )
        else:
            result_dict = welch_t_test(control_data, treatment_data, self.alpha)

            result = ABTestResult(
                metric_name=metric_col,
                control_mean=control_data.mean(),
                treatment_mean=treatment_data.mean(),
                absolute_effect=result_dict['effect'],
                relative_effect=result_dict['relative_effect'],
                se=result_dict['se'],
                ci_lower=result_dict['ci_lower'],
                ci_upper=result_dict['ci_upper'],
                p_value=result_dict['p_value'],
                is_significant=result_dict['is_significant'],
                n_control=n_control,
                n_treatment=n_treatment,
                test_type='continuous'
            )

        self.results[metric_col] = result
        return result

    def check_balance(
        self,
        df: pd.DataFrame,
        covariate_cols: List[str],
        group_col: str = 'group'
    ) -> pd.DataFrame:
        """
        平衡性检查

        检查协变量在两组之间是否平衡
        平衡是随机化成功的标志

        Parameters:
        -----------
        df: 实验数据
        covariate_cols: 协变量列名列表
        group_col: 分组列名

        Returns:
        --------
        平衡性检查结果DataFrame
        """
        results = []

        for col in covariate_cols:
            if col not in df.columns:
                continue

            control = df[df[group_col] == 'control'][col].values
            treatment = df[df[group_col] == 'treatment'][col].values

            # 执行t检验
            result = welch_t_test(control, treatment, self.alpha)

            results.append({
                'covariate': col,
                'control_mean': control.mean(),
                'treatment_mean': treatment.mean(),
                'diff': result['effect'],
                'p_value': result['p_value'],
                'is_balanced': result['p_value'] > 0.05  # 不显著=平衡
            })

        return pd.DataFrame(results)


def check_srm(
    df: pd.DataFrame,
    group_col: str = 'group',
    expected_ratio: float = 0.5,
    alpha: float = 0.001
) -> Dict:
    """
    Sample Ratio Mismatch (SRM) 检测

    SRM是实验分流问题的信号，可能导致结果偏差

    Parameters:
    -----------
    df: 实验数据
    group_col: 分组列名
    expected_ratio: 期望的实验组比例
    alpha: 检验水平（使用严格阈值）

    Returns:
    --------
    SRM检测结果字典
    """
    n_treatment = (df[group_col] == 'treatment').sum()
    n_control = (df[group_col] == 'control').sum()
    n_total = n_treatment + n_control

    observed_ratio = n_treatment / n_total

    # 卡方检验
    expected_treatment = n_total * expected_ratio
    expected_control = n_total * (1 - expected_ratio)

    chi2, p_value = stats.chisquare(
        [n_treatment, n_control],
        [expected_treatment, expected_control]
    )

    has_srm = p_value < alpha

    return {
        'n_control': n_control,
        'n_treatment': n_treatment,
        'n_total': n_total,
        'observed_ratio': observed_ratio,
        'expected_ratio': expected_ratio,
        'chi2': chi2,
        'p_value': p_value,
        'has_srm': has_srm,
        'severity': 'critical' if has_srm else 'ok',
        'message': (
            "警告：检测到SRM！实验结果可能不可信，请检查分流逻辑。"
            if has_srm else "样本比例正常"
        )
    }


def multiple_testing_correction(
    p_values: List[float],
    method: str = 'bonferroni',
    alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """
    多重检验校正

    Parameters:
    -----------
    p_values: 原始p值列表
    method: 校正方法
        - 'bonferroni': Bonferroni校正 (最保守)
        - 'holm': Holm-Bonferroni校正
        - 'fdr': Benjamini-Hochberg FDR控制
    alpha: 全局显著性水平

    Returns:
    --------
    (校正后的p值, 是否显著的布尔列表)
    """
    n = len(p_values)

    if method == 'bonferroni':
        # Bonferroni: p_adj = min(p * n, 1.0)
        adjusted_p = [min(p * n, 1.0) for p in p_values]
        significant = [p < alpha / n for p in p_values]

    elif method == 'holm':
        # Holm-Bonferroni: 逐步校正
        sorted_indices = np.argsort(p_values)
        adjusted_p = [0.0] * n
        significant = [False] * n

        for rank, idx in enumerate(sorted_indices):
            threshold = alpha / (n - rank)
            adjusted_p[idx] = min(p_values[idx] * (n - rank), 1.0)

            if p_values[idx] < threshold:
                significant[idx] = True
            else:
                # 一旦不显著，后续都不显著
                break

    elif method == 'fdr':
        # Benjamini-Hochberg FDR控制
        sorted_indices = np.argsort(p_values)
        adjusted_p = [0.0] * n
        significant = [False] * n

        for rank, idx in enumerate(sorted_indices):
            threshold = (rank + 1) * alpha / n
            adjusted_p[idx] = min(p_values[idx] * n / (rank + 1), 1.0)
            significant[idx] = p_values[idx] <= threshold

        # 确保单调性
        for i in range(n - 2, -1, -1):
            idx = sorted_indices[i]
            next_idx = sorted_indices[i + 1]
            adjusted_p[idx] = min(adjusted_p[idx], adjusted_p[next_idx])

    else:
        raise ValueError(f"Unknown method: {method}")

    return adjusted_p, significant


def calculate_sample_size(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0
) -> Dict:
    """
    样本量计算

    Parameters:
    -----------
    baseline_rate: 基线转化率
    mde_relative: 最小可检测效应（相对提升）
    alpha: 显著性水平
    power: 统计功效
    ratio: 实验组/对照组样本量比例

    Returns:
    --------
    样本量计算结果
    """
    # 绝对效应
    mde_absolute = baseline_rate * mde_relative
    treatment_rate = baseline_rate + mde_absolute

    # Z值
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # 样本量计算
    pooled_var = baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate)
    n_per_group = (z_alpha + z_beta) ** 2 * pooled_var / (mde_absolute ** 2)
    n_per_group = int(np.ceil(n_per_group))

    n_total = n_per_group * (1 + ratio)

    return {
        'n_control': n_per_group,
        'n_treatment': int(n_per_group * ratio),
        'n_total': int(n_total),
        'baseline_rate': baseline_rate,
        'mde_relative': mde_relative,
        'mde_absolute': mde_absolute,
        'alpha': alpha,
        'power': power,
        'ratio': ratio
    }


def plot_ab_results(
    results: List[ABTestResult],
    title: str = 'A/B Test Results'
) -> go.Figure:
    """
    绘制A/B测试结果

    Parameters:
    -----------
    results: ABTestResult对象列表
    title: 图表标题

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    metrics = [r.metric_name for r in results]
    effects = [r.relative_effect * 100 for r in results]

    # 计算相对置信区间
    ci_lower = []
    ci_upper = []
    for r in results:
        if r.control_mean > 0:
            ci_lower.append(r.ci_lower / r.control_mean * 100)
            ci_upper.append(r.ci_upper / r.control_mean * 100)
        else:
            ci_lower.append(0)
            ci_upper.append(0)

    # 颜色编码：绿色=显著正向，红色=显著负向，灰色=不显著
    colors = []
    for r in results:
        if r.is_significant and r.relative_effect > 0:
            colors.append('#27AE60')
        elif r.is_significant and r.relative_effect < 0:
            colors.append('#EB5757')
        else:
            colors.append('#6B7280')

    fig.add_trace(go.Bar(
        x=metrics,
        y=effects,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[u - e for u, e in zip(ci_upper, effects)],
            arrayminus=[e - l for e, l in zip(effects, ci_lower)]
        ),
        marker_color=colors,
        text=[f'{e:.1f}%' for e in effects],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      'Effect: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title='Metric',
        yaxis_title='Relative Effect (%)',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def plot_sample_size_curve(
    baseline_rate: float,
    mde_range: Tuple[float, float] = (0.01, 0.30),
    power_levels: List[float] = [0.70, 0.80, 0.90],
    alpha: float = 0.05
) -> go.Figure:
    """
    绘制样本量曲线

    展示MDE与样本量的关系

    Parameters:
    -----------
    baseline_rate: 基线转化率
    mde_range: MDE范围（相对值）
    power_levels: 功效水平列表
    alpha: 显著性水平

    Returns:
    --------
    Plotly图表
    """
    mde_values = np.linspace(mde_range[0], mde_range[1], 50)

    fig = go.Figure()

    colors = ['#2D9CDB', '#27AE60', '#9B59B6']

    for i, power in enumerate(power_levels):
        sample_sizes = []

        for mde in mde_values:
            result = calculate_sample_size(baseline_rate, mde, alpha, power)
            sample_sizes.append(result['n_total'])

        fig.add_trace(go.Scatter(
            x=mde_values * 100,
            y=sample_sizes,
            mode='lines',
            name=f'Power = {power*100:.0f}%',
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title=f'Sample Size vs MDE (Baseline = {baseline_rate*100:.1f}%)',
        xaxis_title='Minimum Detectable Effect (%)',
        yaxis_title='Total Sample Size',
        yaxis_type='log',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )

    return fig


if __name__ == "__main__":
    # 测试代码
    df = generate_ab_data(n_control=10000, n_treatment=10000)

    analyzer = ABTestAnalyzer()
    result = analyzer.analyze_metric(df, 'converted')

    print(f"Conversion Rate:")
    print(f"  Control: {result.control_mean*100:.2f}%")
    print(f"  Treatment: {result.treatment_mean*100:.2f}%")
    print(f"  Effect: {result.relative_effect*100:+.1f}%")
    print(f"  p-value: {result.p_value:.4f}")
    print(f"  Significant: {result.is_significant}")

    # SRM检查
    srm = check_srm(df)
    print(f"\nSRM Check: {srm['message']}")
