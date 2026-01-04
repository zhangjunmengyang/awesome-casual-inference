"""
CUPED: Controlled-experiment Using Pre-Experiment Data
CUPED方差缩减技术

核心思想:
--------
使用实验前数据作为协变量，减少实验指标的方差，提高统计功效

公式:
----
Y_adj = Y - θ(X - X̄)
其中 θ = Cov(Y, X) / Var(X)

优势:
----
1. 方差缩减 -> 更高的统计功效
2. 更小的样本量需求
3. 更短的实验周期
4. 无偏估计（θ不依赖于处理）

面试考点:
--------
- CUPED的原理是什么？
- 为什么CUPED能减少方差？
- CUPED的假设条件是什么？
- 如何选择协变量？
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import welch_t_test, plot_distribution_comparison


class CUPED:
    """CUPED方差缩减"""

    def __init__(self):
        self.theta = None
        self.covariate_mean = None

    def fit_transform(
        self,
        Y: np.ndarray,
        X: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        拟合并转换数据

        Parameters:
        -----------
        Y: 目标指标（实验期间）
        X: 协变量（实验前）

        Returns:
        --------
        (调整后的Y, 统计信息字典)
        """
        # 计算theta
        self.theta = np.cov(Y, X)[0, 1] / np.var(X)
        self.covariate_mean = X.mean()

        # 调整Y
        Y_adjusted = Y - self.theta * (X - self.covariate_mean)

        # 计算方差缩减
        var_original = np.var(Y)
        var_adjusted = np.var(Y_adjusted)
        variance_reduction = 1 - var_adjusted / var_original

        # 计算相关系数
        correlation = np.corrcoef(Y, X)[0, 1]

        stats_info = {
            'theta': self.theta,
            'variance_original': var_original,
            'variance_adjusted': var_adjusted,
            'variance_reduction': variance_reduction,
            'correlation': correlation,
            'covariate_mean': self.covariate_mean
        }

        return Y_adjusted, stats_info

    def transform(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        使用已拟合的theta转换新数据

        Parameters:
        -----------
        Y: 目标指标
        X: 协变量

        Returns:
        --------
        调整后的Y
        """
        if self.theta is None:
            raise ValueError("Must call fit_transform first")

        return Y - self.theta * (X - self.covariate_mean)


def apply_cuped(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    group_col: str = 'group'
) -> Tuple[pd.DataFrame, Dict]:
    """
    应用CUPED进行方差缩减

    Parameters:
    -----------
    df: 实验数据
    metric_col: 目标指标列名
    covariate_col: 协变量列名（实验前指标）
    group_col: 分组列名

    Returns:
    --------
    (包含调整后指标的DataFrame, CUPED统计信息)
    """
    df = df.copy()

    Y = df[metric_col].values
    X = df[covariate_col].values

    # 应用CUPED
    cuped = CUPED()
    Y_adjusted, stats_info = cuped.fit_transform(Y, X)

    df[f'{metric_col}_cuped'] = Y_adjusted

    return df, stats_info


def compare_with_without_cuped(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    group_col: str = 'group'
) -> Dict:
    """
    比较使用和不使用CUPED的结果

    Parameters:
    -----------
    df: 实验数据
    metric_col: 目标指标列名
    covariate_col: 协变量列名
    group_col: 分组列名

    Returns:
    --------
    比较结果字典
    """
    # 应用CUPED
    df_cuped, cuped_stats = apply_cuped(df, metric_col, covariate_col, group_col)

    # 原始分析
    control_original = df[df[group_col] == 'control'][metric_col].values
    treatment_original = df[df[group_col] == 'treatment'][metric_col].values
    result_original = welch_t_test(control_original, treatment_original)

    # CUPED分析
    control_cuped = df_cuped[df_cuped[group_col] == 'control'][f'{metric_col}_cuped'].values
    treatment_cuped = df_cuped[df_cuped[group_col] == 'treatment'][f'{metric_col}_cuped'].values
    result_cuped = welch_t_test(control_cuped, treatment_cuped)

    # SE缩减比例
    se_reduction = 1 - result_cuped['se'] / result_original['se']

    # 功效提升（近似）
    # 功效与SE成反比，SE缩减意味着功效提升
    power_gain_approx = se_reduction

    return {
        'original': {
            'effect': result_original['effect'],
            'se': result_original['se'],
            'p_value': result_original['p_value'],
            'ci_lower': result_original['ci_lower'],
            'ci_upper': result_original['ci_upper']
        },
        'cuped': {
            'effect': result_cuped['effect'],
            'se': result_cuped['se'],
            'p_value': result_cuped['p_value'],
            'ci_lower': result_cuped['ci_lower'],
            'ci_upper': result_cuped['ci_upper']
        },
        'improvement': {
            'variance_reduction': cuped_stats['variance_reduction'],
            'se_reduction': se_reduction,
            'ci_width_reduction': 1 - (result_cuped['ci_upper'] - result_cuped['ci_lower']) /
                                      (result_original['ci_upper'] - result_original['ci_lower']),
            'power_gain_approx': power_gain_approx,
            'correlation': cuped_stats['correlation']
        }
    }


def select_best_covariate(
    df: pd.DataFrame,
    metric_col: str,
    candidate_covariates: List[str]
) -> Tuple[str, pd.DataFrame]:
    """
    从候选协变量中选择最佳的

    最佳协变量 = 与目标指标相关性最高的

    Parameters:
    -----------
    df: 数据
    metric_col: 目标指标列名
    candidate_covariates: 候选协变量列名列表

    Returns:
    --------
    (最佳协变量名, 评估结果DataFrame)
    """
    results = []

    for cov in candidate_covariates:
        if cov not in df.columns:
            continue

        # 计算相关系数
        corr = df[[metric_col, cov]].corr().iloc[0, 1]

        # 模拟CUPED
        Y = df[metric_col].values
        X = df[cov].values

        theta = np.cov(Y, X)[0, 1] / np.var(X)
        Y_adj = Y - theta * (X - X.mean())

        var_reduction = 1 - np.var(Y_adj) / np.var(Y)

        results.append({
            'covariate': cov,
            'correlation': abs(corr),
            'variance_reduction': var_reduction,
            'score': var_reduction  # 用方差缩减作为评分
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    best_covariate = results_df.iloc[0]['covariate'] if len(results_df) > 0 else None

    return best_covariate, results_df


def plot_cuped_comparison(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    group_col: str = 'group'
) -> go.Figure:
    """
    可视化CUPED效果

    Parameters:
    -----------
    df: 实验数据
    metric_col: 目标指标列名
    covariate_col: 协变量列名
    group_col: 分组列名

    Returns:
    --------
    Plotly图表
    """
    # 应用CUPED
    df_cuped, cuped_stats = apply_cuped(df, metric_col, covariate_col, group_col)

    # 比较结果
    comparison = compare_with_without_cuped(df, metric_col, covariate_col, group_col)

    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Original Metric Distribution',
            'CUPED Adjusted Distribution',
            'Metric vs Covariate',
            'Variance Reduction'
        ),
        specs=[
            [{'type': 'box'}, {'type': 'box'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ]
    )

    # 1. 原始分布
    for group in ['control', 'treatment']:
        data = df[df[group_col] == group][metric_col]
        fig.add_trace(
            go.Box(y=data, name=group.capitalize(), marker_color='#2D9CDB' if group == 'control' else '#27AE60'),
            row=1, col=1
        )

    # 2. CUPED调整后分布
    for group in ['control', 'treatment']:
        data = df_cuped[df_cuped[group_col] == group][f'{metric_col}_cuped']
        fig.add_trace(
            go.Box(y=data, name=f'{group.capitalize()} (CUPED)', marker_color='#2D9CDB' if group == 'control' else '#27AE60', showlegend=False),
            row=1, col=2
        )

    # 3. 散点图：指标 vs 协变量
    fig.add_trace(
        go.Scatter(
            x=df[covariate_col],
            y=df[metric_col],
            mode='markers',
            marker=dict(size=3, opacity=0.3, color='#2D9CDB'),
            name='Data',
            showlegend=False
        ),
        row=2, col=1
    )

    # 添加回归线
    X = df[covariate_col].values
    Y = df[metric_col].values
    theta = np.cov(Y, X)[0, 1] / np.var(X)
    X_mean = X.mean()
    Y_mean = Y.mean()

    x_range = np.array([X.min(), X.max()])
    y_range = Y_mean + theta * (x_range - X_mean)

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            line=dict(color='red', width=2),
            name='Regression',
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. 方差缩减对比
    categories = ['Original SE', 'CUPED SE']
    values = [
        comparison['original']['se'],
        comparison['cuped']['se']
    ]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=['#EB5757', '#27AE60'],
            text=[f'{v:.4f}' for v in values],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=2
    )

    # 更新布局
    fig.update_layout(
        title=f'CUPED Analysis: {metric_col}',
        template='plotly_white',
        height=800,
        showlegend=True
    )

    # 更新坐标轴
    fig.update_xaxes(title_text=covariate_col, row=2, col=1)
    fig.update_yaxes(title_text=metric_col, row=2, col=1)
    fig.update_yaxes(title_text='Standard Error', row=2, col=2)

    return fig


def plot_variance_reduction_curve(
    df: pd.DataFrame,
    metric_col: str,
    candidate_covariates: List[str]
) -> go.Figure:
    """
    绘制不同协变量的方差缩减效果

    Parameters:
    -----------
    df: 数据
    metric_col: 目标指标列名
    candidate_covariates: 候选协变量列名列表

    Returns:
    --------
    Plotly图表
    """
    _, results_df = select_best_covariate(df, metric_col, candidate_covariates)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=results_df['covariate'],
        y=results_df['variance_reduction'] * 100,
        marker_color='#27AE60',
        text=[f'{v:.1f}%' for v in results_df['variance_reduction'] * 100],
        textposition='outside'
    ))

    # 添加相关系数作为第二条
    fig.add_trace(go.Scatter(
        x=results_df['covariate'],
        y=results_df['correlation'] * 100,
        mode='markers+lines',
        name='Correlation',
        marker=dict(size=10, color='#2D9CDB'),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Covariate Selection: Variance Reduction',
        xaxis_title='Covariate',
        yaxis_title='Variance Reduction (%)',
        yaxis2=dict(
            title='Correlation',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        template='plotly_white',
        height=400,
        showlegend=True
    )

    return fig


if __name__ == "__main__":
    # 测试代码
    from .utils import generate_ab_data

    df = generate_ab_data(n_control=5000, n_treatment=5000, add_covariates=True)

    # 比较CUPED前后
    comparison = compare_with_without_cuped(
        df,
        metric_col='converted',
        covariate_col='historical_conversion'
    )

    print("Original:")
    print(f"  SE: {comparison['original']['se']:.5f}")
    print(f"  p-value: {comparison['original']['p_value']:.4f}")

    print("\nCUPED:")
    print(f"  SE: {comparison['cuped']['se']:.5f}")
    print(f"  p-value: {comparison['cuped']['p_value']:.4f}")

    print("\nImprovement:")
    print(f"  Variance Reduction: {comparison['improvement']['variance_reduction']*100:.1f}%")
    print(f"  SE Reduction: {comparison['improvement']['se_reduction']*100:.1f}%")
    print(f"  Correlation: {comparison['improvement']['correlation']:.3f}")
