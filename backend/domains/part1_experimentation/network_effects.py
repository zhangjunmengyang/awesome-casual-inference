"""
Network Effects and Spillover
网络效应与溢出效应

核心问题:
--------
SUTVA (Stable Unit Treatment Value Assumption) 违反
- 一个用户的处理会影响其他用户
- 经典A/B测试假设失效

典型场景:
--------
1. 社交网络：用户A的行为影响好友B
2. 双边市场：司机增加影响乘客体验
3. 推荐系统：物品曝光影响其他物品
4. 平台效应：供给侧影响需求侧

解决方案:
--------
1. Cluster随机化
2. Geo实验
3. 时间切换实验
4. 网络效应模型

面试考点:
--------
- 什么是SUTVA？
- 如何检测网络效应？
- Cluster随机化的权衡？
- 如何估计溢出效应？
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import welch_t_test


def simulate_network_data(
    n_users: int = 1000,
    avg_degree: int = 10,
    treatment_effect_direct: float = 0.15,
    treatment_effect_spillover: float = 0.05,
    baseline_rate: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    模拟存在网络效应的数据

    Parameters:
    -----------
    n_users: 用户数
    avg_degree: 平均好友数
    treatment_effect_direct: 直接处理效应
    treatment_effect_spillover: 溢出效应
    baseline_rate: 基线转化率
    seed: 随机种子

    Returns:
    --------
    (用户数据DataFrame, 邻接矩阵)
    """
    np.random.seed(seed)

    # 生成随机图（Erdős-Rényi）
    p_edge = avg_degree / n_users
    adjacency = (np.random.random((n_users, n_users)) < p_edge).astype(int)
    adjacency = np.triu(adjacency, k=1)  # 上三角
    adjacency = adjacency + adjacency.T  # 对称化
    np.fill_diagonal(adjacency, 0)

    # 随机分配处理
    is_treated = np.random.binomial(1, 0.5, n_users)

    # 计算每个用户的treated好友比例
    n_friends = adjacency.sum(axis=1)
    n_treated_friends = adjacency.dot(is_treated)
    treated_friend_ratio = np.divide(
        n_treated_friends,
        n_friends,
        out=np.zeros_like(n_treated_friends, dtype=float),
        where=n_friends > 0
    )

    # 基线概率
    base_prob = np.full(n_users, baseline_rate)

    # 总效应 = 直接效应 + 溢出效应
    total_effect = (
        is_treated * treatment_effect_direct * base_prob +
        treated_friend_ratio * treatment_effect_spillover * base_prob
    )

    # 转化概率
    prob = base_prob + total_effect
    prob = np.clip(prob, 0, 1)

    # 转化
    converted = np.random.binomial(1, prob)

    # 构建DataFrame
    df = pd.DataFrame({
        'user_id': range(n_users),
        'is_treated': is_treated,
        'n_friends': n_friends,
        'n_treated_friends': n_treated_friends,
        'treated_friend_ratio': treated_friend_ratio,
        'converted': converted,
        'conversion_prob': prob
    })

    return df, adjacency


def naive_ate_biased(
    df: pd.DataFrame,
    outcome_col: str = 'converted'
) -> Dict:
    """
    朴素ATE估计（有偏）

    忽略网络效应，直接比较treated vs control

    Parameters:
    -----------
    df: 用户数据
    outcome_col: 结果列名

    Returns:
    --------
    估计结果（有偏）
    """
    treated = df[df['is_treated'] == 1][outcome_col].values
    control = df[df['is_treated'] == 0][outcome_col].values

    result = welch_t_test(control, treated)

    return {
        'ate_naive': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'note': 'This estimate is BIASED due to network spillover'
    }


def estimate_spillover_effect(
    df: pd.DataFrame,
    outcome_col: str = 'converted'
) -> Dict:
    """
    估计溢出效应

    比较：
    1. 对照组中，有treated好友 vs 无treated好友
    2. 这个差异就是溢出效应

    Parameters:
    -----------
    df: 用户数据
    outcome_col: 结果列名

    Returns:
    --------
    溢出效应估计
    """
    # 仅看对照组
    control_df = df[df['is_treated'] == 0].copy()

    # 有treated好友 vs 无treated好友
    has_treated_friends = control_df[control_df['n_treated_friends'] > 0]
    no_treated_friends = control_df[control_df['n_treated_friends'] == 0]

    if len(has_treated_friends) == 0 or len(no_treated_friends) == 0:
        return {
            'spillover_effect': 0,
            'error': 'Insufficient data for spillover estimation'
        }

    result = welch_t_test(
        no_treated_friends[outcome_col].values,
        has_treated_friends[outcome_col].values
    )

    return {
        'spillover_effect': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'n_with_treated_friends': len(has_treated_friends),
        'n_without_treated_friends': len(no_treated_friends)
    }


def decompose_total_effect(
    df: pd.DataFrame,
    outcome_col: str = 'converted'
) -> Dict:
    """
    分解总效应 = 直接效应 + 溢出效应

    方法：线性回归
    Y = β0 + β1*Treated + β2*TreatedFriendRatio + ε

    β1 = 直接效应
    β2 = 溢出效应

    Parameters:
    -----------
    df: 用户数据
    outcome_col: 结果列名

    Returns:
    --------
    效应分解结果
    """
    from scipy.stats import linregress

    # 准备数据
    X_treated = df['is_treated'].values
    X_spillover = df['treated_friend_ratio'].values
    Y = df[outcome_col].values

    # 方法1: 分别回归（近似）
    # 直接效应：控制spillover
    treated_only_df = df[df['treated_friend_ratio'] == 0]
    if len(treated_only_df) > 100:
        treated_data = treated_only_df[treated_only_df['is_treated'] == 1][outcome_col].values
        control_data = treated_only_df[treated_only_df['is_treated'] == 0][outcome_col].values
        if len(treated_data) > 0 and len(control_data) > 0:
            direct_effect = treated_data.mean() - control_data.mean()
        else:
            direct_effect = np.nan
    else:
        direct_effect = np.nan

    # 溢出效应
    spillover_result = estimate_spillover_effect(df, outcome_col)
    spillover_effect = spillover_result.get('spillover_effect', 0)

    # 朴素总效应
    naive_result = naive_ate_biased(df, outcome_col)
    total_effect_naive = naive_result['ate_naive']

    return {
        'direct_effect': direct_effect,
        'spillover_effect': spillover_effect,
        'total_effect_naive': total_effect_naive,
        'note': 'Direct effect isolated from spillover'
    }


def cluster_randomization_analysis(
    df: pd.DataFrame,
    cluster_col: str,
    outcome_col: str,
    treatment_col: str = 'is_treated'
) -> Dict:
    """
    Cluster随机化分析

    在cluster层面随机化，避免组内干扰

    Parameters:
    -----------
    df: 数据
    cluster_col: cluster列名
    outcome_col: 结果列名
    treatment_col: 处理列名

    Returns:
    --------
    cluster层面的分析结果
    """
    # 聚合到cluster层面
    cluster_summary = df.groupby(cluster_col).agg({
        outcome_col: 'mean',
        treatment_col: 'first',  # cluster内应该都一样
    }).reset_index()

    # cluster层面的t检验
    treated_clusters = cluster_summary[cluster_summary[treatment_col] == 1][outcome_col].values
    control_clusters = cluster_summary[cluster_summary[treatment_col] == 0][outcome_col].values

    result = welch_t_test(control_clusters, treated_clusters)

    return {
        'cluster_ate': result['effect'],
        'se': result['se'],
        'p_value': result['p_value'],
        'n_clusters_treatment': len(treated_clusters),
        'n_clusters_control': len(control_clusters),
        'note': 'Cluster-level analysis, accounts for within-cluster correlation'
    }


def plot_network_effects(
    df: pd.DataFrame,
    outcome_col: str = 'converted'
) -> go.Figure:
    """
    可视化网络效应

    Parameters:
    -----------
    df: 用户数据
    outcome_col: 结果列名

    Returns:
    --------
    Plotly图表
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Spillover Effect by Friend Treatment Ratio',
            'Direct vs Spillover Effect'
        )
    )

    # 1. 溢出效应 vs treated好友比例
    # 分箱
    df['ratio_bin'] = pd.cut(df['treated_friend_ratio'], bins=5)

    control_df = df[df['is_treated'] == 0].copy()
    binned = control_df.groupby('ratio_bin', observed=True).agg({
        outcome_col: ['mean', 'sem', 'count']
    }).reset_index()

    binned.columns = ['ratio_bin', 'mean', 'sem', 'count']
    binned = binned[binned['count'] >= 10]  # 过滤小样本

    x_labels = [str(b) for b in binned['ratio_bin']]

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=binned['mean'],
            error_y=dict(type='data', array=binned['sem'] * 1.96),
            mode='markers+lines',
            marker=dict(size=10, color='#2D9CDB'),
            name='Spillover'
        ),
        row=1, col=1
    )

    # 2. 效应分解
    decomposition = decompose_total_effect(df, outcome_col)

    categories = ['Direct Effect', 'Spillover Effect', 'Total Effect (Naive)']
    values = [
        decomposition.get('direct_effect', 0),
        decomposition.get('spillover_effect', 0),
        decomposition.get('total_effect_naive', 0)
    ]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=['#27AE60', '#F2994A', '#2D9CDB'],
            text=[f'{v:.4f}' for v in values],
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='Network Effects Analysis',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    fig.update_xaxes(title_text='Treated Friend Ratio', row=1, col=1)
    fig.update_yaxes(title_text='Conversion Rate', row=1, col=1)
    fig.update_yaxes(title_text='Effect Size', row=1, col=2)

    return fig


if __name__ == "__main__":
    # 测试代码
    df, adjacency = simulate_network_data(
        n_users=1000,
        treatment_effect_direct=0.15,
        treatment_effect_spillover=0.05
    )

    print("Network Data Generated:")
    print(f"  Users: {len(df)}")
    print(f"  Avg Friends: {df['n_friends'].mean():.1f}")

    # 朴素估计（有偏）
    naive = naive_ate_biased(df)
    print(f"\nNaive ATE (biased): {naive['ate_naive']:.4f}")

    # 效应分解
    decomp = decompose_total_effect(df)
    print(f"\nEffect Decomposition:")
    print(f"  Direct: {decomp['direct_effect']:.4f}")
    print(f"  Spillover: {decomp['spillover_effect']:.4f}")
    print(f"  Total (naive): {decomp['total_effect_naive']:.4f}")
