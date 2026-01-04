"""Part 6 Marketing - API 适配层

将营销应用模块转换为返回标准格式的 API 函数。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

from .utils import (
    generate_user_journey_data,
    generate_marketing_data,
    generate_driver_data,
)
from .attribution import MarketingAttribution
from .coupon_optimization import CouponOptimizer
from .user_targeting import TLearner, XLearner, PolicyLearner
from .budget_allocation import BudgetOptimizer


def _fig_to_chart_data(fig: go.Figure) -> dict:
    """将 Plotly Figure 转换为前端可用的图表数据"""
    return fig.to_dict()


def analyze_attribution(
    n_conversions: int = 500,
    channels: Optional[List[str]] = None,
    model_type: str = 'shapley'
) -> dict:
    """营销归因分析

    Args:
        n_conversions: 转化用户数量
        channels: 渠道列表
        model_type: 归因模型类型
            - 'last_touch': Last-touch 归因
            - 'first_touch': First-touch 归因
            - 'linear': Linear 归因
            - 'time_decay': Time-decay 归因
            - 'position_based': Position-based 归因
            - 'shapley': Shapley Value 归因
            - 'markov': Markov Chain 归因

    Returns:
        标准 API 响应格式
    """
    # 生成用户路径数据
    df = generate_user_journey_data(n_users=n_conversions * 5, channels=channels)

    # 创建归因模型
    attribution_model = MarketingAttribution(df)

    # 应用归因方法
    result_df = attribution_model.apply_attribution(method=model_type)

    # 创建可视化
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['各渠道归因收入', '收入占比'],
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )

    # 柱状图
    fig.add_trace(
        go.Bar(
            x=result_df['Channel'],
            y=result_df['Attributed_Revenue'],
            text=result_df['Attributed_Revenue'].apply(lambda x: f'${x:.0f}'),
            textposition='auto',
            marker_color='#2D9CDB',
            showlegend=False
        ),
        row=1, col=1
    )

    # 饼图
    fig.add_trace(
        go.Pie(
            labels=result_df['Channel'],
            values=result_df['Attributed_Revenue'],
            textinfo='label+percent',
            marker=dict(colors=['#2D9CDB', '#27AE60', '#F2C94C', '#EB5757', '#9B51E0'][:len(result_df)])
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="渠道", row=1, col=1)
    fig.update_yaxes(title_text="归因收入 ($)", row=1, col=1)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text=f"营销归因分析 ({model_type.replace('_', '-').title()})"
    )

    # 构建总结
    total_revenue = result_df['Attributed_Revenue'].sum()
    top_channel = result_df.iloc[0]

    summary = f"""
## 营销归因分析结果

### 使用方法
**{model_type.replace('_', '-').title()} Attribution**

### 关键指标

| 指标 | 值 |
|------|-----|
| 转化用户数 | {len(df)} |
| 总收入 | ${total_revenue:,.2f} |
| 平均客单价 | ${df['revenue'].mean():.2f} |
| 平均触点数 | {df['touchpoints'].mean():.1f} |

### 归因结果

**Top 渠道**: {top_channel['Channel']} 占 {top_channel['Revenue_Share']*100:.1f}% (${top_channel['Attributed_Revenue']:,.0f})

### 方法说明

"""

    if model_type == 'shapley':
        summary += """
**Shapley Value 归因**基于合作博弈论，公平地分配每个渠道的贡献。
- 满足效率性、对称性、虚拟性、可加性公理
- 考虑了渠道的边际贡献
"""
    elif model_type == 'markov':
        summary += """
**Markov Chain 归因**基于用户路径的转移概率。
- 计算移除某个渠道后转化率的下降
- 反映渠道在转化路径中的重要性
"""
    else:
        summary += f"""
**{model_type.replace('_', '-').title()} 归因**是基于规则的归因方法。
- 简单直观，易于理解
- 但可能无法准确反映渠道的真实贡献
"""

    # 构建表格数据
    table_data = result_df.to_dict('records')

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [table_data],
        "summary": summary,
        "metrics": {
            "total_revenue": float(total_revenue),
            "n_conversions": int(len(df)),
            "avg_touchpoints": float(df['touchpoints'].mean()),
            "top_channel": top_channel['Channel'],
            "top_channel_share": float(top_channel['Revenue_Share'])
        }
    }


def analyze_coupon_optimization(
    n_users: int = 2000,
    budget_constraint: float = 0.3
) -> dict:
    """智能发券优化分析

    Args:
        n_users: 用户数量
        budget_constraint: 预算约束（可发券比例）

    Returns:
        标准 API 响应格式
    """
    # 生成数据
    df = generate_marketing_data(n_samples=n_users)

    # 训练模型
    X = df[['age', 'order_freq', 'days_since_last']].values
    T = df['T'].values
    Y = df['conversion'].values

    optimizer = CouponOptimizer()
    optimizer.fit(X, T, Y)

    # 预测 Uplift
    uplift_scores = optimizer.predict_uplift(X)
    df['uplift_score'] = uplift_scores

    # 策略对比
    comparison = optimizer.compare_strategies(
        df, X, budget_fraction=budget_constraint
    )

    # 创建可视化
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['各策略 ROI 对比', 'Uplift 分数分布']
    )

    # ROI 对比
    fig.add_trace(
        go.Bar(
            x=comparison['strategy'],
            y=comparison['roi'],
            text=comparison['roi'].apply(lambda x: f'{x:.2f}'),
            textposition='auto',
            marker_color=['#95a5a6', '#3498db', '#2ecc71'][:len(comparison)],
            showlegend=False
        ),
        row=1, col=1
    )

    # Uplift 分布
    fig.add_trace(
        go.Histogram(
            x=uplift_scores,
            nbinsx=30,
            marker_color='#27AE60',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="策略", row=1, col=1)
    fig.update_yaxes(title_text="ROI", row=1, col=1)
    fig.update_xaxes(title_text="Uplift 分数", row=1, col=2)
    fig.update_yaxes(title_text="用户数", row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="智能发券优化分析"
    )

    # 找最佳策略
    best_idx = comparison['roi'].idxmax()
    best_strategy = comparison.loc[best_idx]

    # 用户类型统计
    user_type_stats = df['user_type'].value_counts().to_dict()

    summary = f"""
## 智能发券优化结果

### 策略对比

预算约束：可发券给 {budget_constraint*100:.0f}% 的用户

**最佳策略**: {best_strategy['strategy']}
- ROI: {best_strategy['roi']:.2f}
- 发券人数: {best_strategy['n_coupons']}
- 预期增量转化: {best_strategy['expected_conversions']:.1f}
- 净收益: ¥{best_strategy['revenue'] - best_strategy['cost']:.0f}

### 用户类型分布

- **Persuadables** (可说服者): {user_type_stats.get('Persuadables', 0)} 人 - 重点发券对象
- **Sure Things** (必转化者): {user_type_stats.get('Sure Things', 0)} 人 - 浪费预算
- **Sleeping Dogs** (睡狗): {user_type_stats.get('Sleeping Dogs', 0)} 人 - 千万别发
- **Lost Causes** (流失者): {user_type_stats.get('Lost Causes', 0)} 人 - 无效

### 关键洞察

1. **Uplift 建模优于传统方法**: 相比随机发券或高频用户发券，Uplift 模型能识别真正被优惠券影响的用户
2. **避免"Sure Things"**: 这些用户本来就会转化，发券是浪费
3. **警惕"Sleeping Dogs"**: 部分用户收到促销反而会降低转化率
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [comparison.to_dict('records')],
        "summary": summary,
        "metrics": {
            "best_strategy": best_strategy['strategy'],
            "best_roi": float(best_strategy['roi']),
            "avg_uplift": float(uplift_scores.mean()),
            "persuadables_count": int(user_type_stats.get('Persuadables', 0)),
            "sure_things_count": int(user_type_stats.get('Sure Things', 0))
        }
    }


def analyze_user_targeting(
    n_users: int = 2000,
    targeting_strategy: str = 't_learner'
) -> dict:
    """用户定向分析

    Args:
        n_users: 用户数量
        targeting_strategy: 定向策略
            - 't_learner': T-Learner CATE 估计
            - 'x_learner': X-Learner CATE 估计

    Returns:
        标准 API 响应格式
    """
    # 生成数据
    df = generate_driver_data(n_samples=n_users)

    # 训练模型
    X = df[['rating', 'order_history', 'is_fulltime']].values
    T = df['T'].values
    Y = df['online_hours'].values

    if targeting_strategy == 'x_learner':
        model = XLearner()
    else:
        model = TLearner()

    model.fit(X, T, Y)

    # 预测 CATE
    cate = model.predict_cate(X)
    df['cate'] = cate

    # 学习最优策略
    optimal_policy, metrics = PolicyLearner.learn_optimal_policy(
        cate,
        cost_per_treatment=100,
        value_per_unit=30
    )

    # 策略对比
    comparison = PolicyLearner.compare_targeting_strategies(
        df, cate, cost=100, value=30
    )

    # 用户分层
    segments = PolicyLearner.segment_by_cate(cate)
    df['segment'] = segments

    # 创建可视化
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['策略净收益对比', 'CATE 分布']
    )

    # 净收益对比
    fig.add_trace(
        go.Bar(
            x=comparison['strategy'],
            y=comparison['net_benefit'],
            text=comparison['net_benefit'].apply(lambda x: f'¥{x:.0f}'),
            textposition='auto',
            marker_color=['#95a5a6', '#e74c3c', '#3498db', '#2ecc71'][:len(comparison)],
            showlegend=False
        ),
        row=1, col=1
    )

    # CATE 分布
    fig.add_trace(
        go.Histogram(
            x=cate,
            nbinsx=30,
            marker_color='#2D9CDB',
            opacity=0.7,
            showlegend=False
        ),
        row=1, col=2
    )

    # 添加阈值线
    fig.add_vline(
        x=metrics['threshold'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"阈值={metrics['threshold']:.2f}",
        row=1, col=2
    )

    fig.update_xaxes(title_text="策略", row=1, col=1)
    fig.update_yaxes(title_text="净收益 (¥)", row=1, col=1)
    fig.update_xaxes(title_text="CATE (小时)", row=1, col=2)
    fig.update_yaxes(title_text="司机数", row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text=f"用户定向分析 ({targeting_strategy.upper()})"
    )

    # 分层统计
    segment_stats = pd.Series(segments).value_counts().to_dict()

    summary = f"""
## 用户定向分析结果

### 使用方法
**{targeting_strategy.upper()}** - {'分别训练处理组和控制组模型' if targeting_strategy == 't_learner' else '考虑伪处理效应的高级方法'}

### 最优策略

- **干预人数**: {metrics['n_treated']} ({metrics['treatment_rate']*100:.1f}%)
- **预期效应**: {metrics['expected_effect']:.1f} 小时
- **总成本**: ¥{metrics['total_cost']:,.0f}
- **总价值**: ¥{metrics['total_value']:,.0f}
- **净收益**: ¥{metrics['net_benefit']:,.0f}
- **ROI**: {metrics['roi']:.2f}

### 决策规则

只有当 CATE × 价值 > 成本 时才进行干预
- 阈值 = {metrics['threshold']:.2f} 小时
- 单位价值 = ¥30/小时
- 干预成本 = ¥100

### 用户分层

- **High Impact**: {segment_stats.get('High Impact', 0)} 人 - 重点激励
- **Medium Impact**: {segment_stats.get('Medium Impact', 0)} 人 - 可选
- **Low Impact**: {segment_stats.get('Low Impact', 0)} 人 - ROI较低
- **Negative Impact**: {segment_stats.get('Negative Impact', 0)} 人 - 避免干预

### 关键洞察

1. 最优策略只干预 {metrics['treatment_rate']*100:.1f}% 的用户，但能获得最高的净收益
2. 兼职司机通常有更高的 CATE（激励效果更好）
3. 全职司机已经很活跃，边际效应较小
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [comparison.to_dict('records')],
        "summary": summary,
        "metrics": {
            "treatment_rate": float(metrics['treatment_rate']),
            "net_benefit": float(metrics['net_benefit']),
            "roi": float(metrics['roi']),
            "avg_cate": float(cate.mean()),
            "high_impact_count": int(segment_stats.get('High Impact', 0))
        }
    }


def analyze_budget_allocation(
    total_budget: float = 1000,
    channels: Optional[Dict[str, Dict[str, float]]] = None,
    optimization_method: str = 'unconstrained'
) -> dict:
    """预算分配优化分析

    Args:
        total_budget: 总预算（万元）
        channels: 渠道参数（如果为 None 使用默认值）
        optimization_method: 优化方法
            - 'unconstrained': 无约束优化
            - 'constrained': 带约束优化
            - 'robust': 稳健优化

    Returns:
        标准 API 响应格式
    """
    # 默认渠道参数
    if channels is None:
        channels = {
            '搜索广告': {'a': 500, 'c': 150, 'alpha': 0.8},
            '信息流': {'a': 800, 'c': 300, 'alpha': 1.2},
            '短视频': {'a': 600, 'c': 200, 'alpha': 1.0}
        }

    # 创建优化器
    optimizer = BudgetOptimizer(channels)

    # 根据方法执行优化
    if optimization_method == 'constrained':
        constraints = {
            'min_budget': {ch: 50 for ch in channels.keys()},
            'max_share': {list(channels.keys())[0]: 0.5}  # 第一个渠道不超过50%
        }
        allocation, total_response = optimizer.optimize(
            total_budget,
            constraints=constraints
        )
    elif optimization_method == 'robust':
        param_uncertainty = {
            ch: {'a_std': params['a'] * 0.1, 'c_std': params['c'] * 0.1, 'alpha_std': 0.05}
            for ch, params in channels.items()
        }
        allocation, stats = optimizer.robust_optimization(
            total_budget,
            param_uncertainty,
            n_scenarios=200
        )
        total_response = stats['mean']
    else:  # unconstrained
        allocation, total_response = optimizer.optimize(total_budget)

    # 计算边际 ROI
    marginal_rois = optimizer.compute_marginal_rois(allocation)

    # 计算平均分配作为对比
    avg_allocation = {ch: total_budget / len(channels) for ch in channels.keys()}
    _, avg_response = optimizer.optimize(total_budget)  # 用优化后的作为对比基准

    # 创建可视化
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['预算分配', '边际 ROI'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    # 预算分配
    fig.add_trace(
        go.Bar(
            x=list(allocation.keys()),
            y=list(allocation.values()),
            text=[f'{v:.0f}万' for v in allocation.values()],
            textposition='auto',
            marker_color='#2D9CDB',
            showlegend=False
        ),
        row=1, col=1
    )

    # 边际 ROI
    fig.add_trace(
        go.Bar(
            x=list(marginal_rois.keys()),
            y=list(marginal_rois.values()),
            text=[f'{v:.3f}' for v in marginal_rois.values()],
            textposition='auto',
            marker_color='#27AE60',
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="渠道", row=1, col=1)
    fig.update_yaxes(title_text="预算 (万元)", row=1, col=1)
    fig.update_xaxes(title_text="渠道", row=1, col=2)
    fig.update_yaxes(title_text="边际 ROI", row=1, col=2)

    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text=f"预算分配优化 ({optimization_method.title()})"
    )

    # 构建分配表
    allocation_table = []
    for ch in channels.keys():
        allocation_table.append({
            '渠道': ch,
            '预算(万)': f'{allocation[ch]:.1f}',
            '占比': f'{allocation[ch]/total_budget*100:.1f}%',
            '边际ROI': f'{marginal_rois[ch]:.3f}'
        })

    summary = f"""
## 预算分配优化结果

### 优化方法
**{optimization_method.title()}**

### 最优分配

总预算: {total_budget} 万元
总收益: {total_response:.1f} 万元
平均 ROI: {total_response/total_budget:.2f}

### 优化原则

**边际 ROI 相等**: 最优分配时，所有渠道的边际 ROI 应该相等（或接近）
- 当前边际 ROI 范围: [{min(marginal_rois.values()):.3f}, {max(marginal_rois.values()):.3f}]

### 关键洞察

1. **不要平均分配**: 不同渠道有不同的响应曲线和饱和点
2. **边际收益递减**: 单一渠道投入过多会导致收益递减
3. **动态调整**: 应定期根据实际效果重新优化分配
"""

    if optimization_method == 'robust':
        summary += f"""

### 稳健性分析

考虑参数不确定性（±10%），在 200 个场景下：
- 期望收益: {stats['mean']:.1f} 万元
- 标准差: {stats['std']:.1f} 万元
- 90% 置信区间: [{stats['percentile_5']:.1f}, {stats['percentile_95']:.1f}]
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [allocation_table],
        "summary": summary,
        "metrics": {
            "total_budget": float(total_budget),
            "total_response": float(total_response),
            "avg_roi": float(total_response / total_budget),
            "marginal_roi_range": [float(min(marginal_rois.values())), float(max(marginal_rois.values()))]
        }
    }
