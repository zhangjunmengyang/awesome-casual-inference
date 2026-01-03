"""
IndustryCases 工具函数

提供真实行业场景的数据生成器、因果图定义、评估指标计算等功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_doordash_delivery_data(
    n_samples: int = 8000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成 DoorDash 配送时间优化数据

    场景: DoorDash 配送时间预估与司机调度优化
    - 特征: 距离、天气、时段、餐厅类型、司机经验等
    - 处理: 是否使用新调度算法
    - 结果: 配送时间、客户满意度
    - 混淆因素: 天气影响算法使用和配送时间

    业务背景:
    DoorDash 需要优化配送时间预估，提升客户体验。新算法考虑了更多实时因素
    (交通、天气、餐厅准备速度)，但需要评估其真实效果。

    Parameters:
    -----------
    n_samples: 订单数量
    seed: 随机种子

    Returns:
    --------
    (df, true_effect): 数据和真实处理效应
    """
    if seed is not None:
        np.random.seed(seed)

    # === 协变量 ===

    # 配送距离 (公里)
    distance_km = np.random.gamma(2, 1.5, n_samples)  # 大部分 2-5 公里

    # 天气状况 (0=晴天, 1=雨天, 2=恶劣天气)
    weather = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])

    # 时段 (0=非高峰, 1=午高峰, 2=晚高峰)
    time_period = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])

    # 餐厅类型 (0=快餐, 1=正餐, 2=高端)
    restaurant_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1])

    # 餐厅准备时间 (分钟) - 依赖于餐厅类型
    prep_time_base = [10, 15, 25]  # 快餐、正餐、高端
    prep_time = np.array([
        np.random.normal(prep_time_base[rt], 5) for rt in restaurant_type
    ])
    prep_time = np.clip(prep_time, 5, 60)

    # 司机经验 (月数)
    driver_exp_months = np.random.exponential(12, n_samples)

    # 历史准时率
    driver_ontime_rate = np.random.beta(8, 2, n_samples)

    # 订单金额
    order_value = np.random.lognormal(3.5, 0.6, n_samples)

    # 客户是会员
    is_dashpass = np.random.binomial(1, 0.25, n_samples)

    # 标准化特征
    distance_norm = (distance_km - 3) / 2
    weather_norm = weather / 2
    time_norm = time_period / 2
    prep_norm = (prep_time - 15) / 10
    exp_norm = (driver_exp_months - 12) / 8

    # === 处理分配 (非随机 - 受天气和时段影响) ===
    # 好天气和非高峰时段更倾向使用新算法 (系统负载低)
    propensity = (
        0.5 +
        0.15 * (weather == 0) +  # 好天气
        0.1 * (time_period == 0) +  # 非高峰
        0.05 * (driver_exp_months > 12)  # 经验丰富的司机
    )
    propensity = np.clip(propensity, 0.2, 0.8)
    T = np.random.binomial(1, propensity)

    # === 基线配送时间 (不使用新算法) ===
    baseline_time = (
        20 +  # 基础时间
        3 * distance_norm +  # 距离影响
        8 * weather_norm +  # 天气延误
        5 * time_norm +  # 高峰延误
        0.3 * prep_norm +  # 餐厅准备时间
        -2 * exp_norm -  # 经验丰富的司机更快
        3 * driver_ontime_rate  # 历史准时的司机更快
    )
    baseline_time = np.clip(baseline_time, 10, 60)

    # === 处理效应 (新算法的效果) ===
    # 新算法在复杂情况下效果更好
    effect = (
        -3 +  # 平均减少 3 分钟
        -2 * (weather > 0) +  # 恶劣天气下更有效
        -1.5 * (time_period > 0) +  # 高峰时段更有效
        -1 * (distance_km > 4) +  # 长距离更有效
        0.5 * (driver_exp_months < 6)  # 新司机获益较少
    )
    effect = np.clip(effect, -8, 0)

    # === 实际配送时间 ===
    delivery_time = baseline_time + effect * T + np.random.randn(n_samples) * 2
    delivery_time = np.clip(delivery_time, 8, 70)

    # === 客户满意度 (1-5 分) ===
    # 受配送时间、订单金额等影响
    satisfaction_score = (
        4.5 +
        -0.05 * (delivery_time - 25) +  # 超过 25 分钟满意度下降
        0.1 * (order_value > 50) +  # 大单客户期望高
        0.3 * is_dashpass +  # 会员更宽容
        np.random.randn(n_samples) * 0.3
    )
    satisfaction_score = np.clip(satisfaction_score, 1, 5)

    # === 是否准时 (预估时间的 90% 置信区间) ===
    estimated_time = delivery_time + np.random.randn(n_samples) * 3
    on_time = (delivery_time <= estimated_time).astype(int)

    # === 小费 (受满意度影响) ===
    tip_amount = np.where(
        satisfaction_score >= 4,
        np.random.gamma(2, 2, n_samples),
        np.random.gamma(1, 1, n_samples)
    )

    # === 创建 DataFrame ===
    df = pd.DataFrame({
        'distance_km': distance_km,
        'weather': weather,
        'time_period': time_period,
        'restaurant_type': restaurant_type,
        'prep_time': prep_time,
        'driver_exp_months': driver_exp_months,
        'driver_ontime_rate': driver_ontime_rate,
        'order_value': order_value,
        'is_dashpass': is_dashpass,
        'T': T,
        'propensity': propensity,
        'delivery_time': delivery_time,
        'satisfaction_score': satisfaction_score,
        'on_time': on_time,
        'tip_amount': tip_amount,
    })

    return df, effect


def generate_netflix_recommendation_data(
    n_samples: int = 10000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成 Netflix 推荐系统优化数据

    场景: Netflix 测试新推荐算法对用户留存的影响
    - 特征: 观看历史、偏好类型、设备、观看时段等
    - 处理: 是否使用新推荐算法
    - 结果: 30 天留存率、观看时长
    - 混淆因素: 用户活跃度影响算法分配和留存

    业务背景:
    Netflix 开发了基于深度学习的新推荐算法，能更好捕捉用户长期偏好。
    需要通过因果推断评估其对不同用户群的真实效果。

    Parameters:
    -----------
    n_samples: 用户数量
    seed: 随机种子

    Returns:
    --------
    (df, true_effect): 数据和真实处理效应
    """
    if seed is not None:
        np.random.seed(seed)

    # === 用户特征 ===

    # 用户年龄
    user_age = np.random.normal(35, 15, n_samples)
    user_age = np.clip(user_age, 13, 80)

    # 注册时长 (月)
    tenure_months = np.random.exponential(18, n_samples)

    # 月观看时长 (小时)
    monthly_watch_hours = np.random.gamma(3, 8, n_samples)

    # 观看内容多样性 (0-1, 越高越多样)
    content_diversity = np.random.beta(3, 2, n_samples)

    # 偏好类型 (0=剧集, 1=电影, 2=纪录片)
    preferred_genre = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.35, 0.15])

    # 主要设备 (0=手机, 1=平板, 2=电视, 3=电脑)
    primary_device = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.15, 0.3, 0.15])

    # 观看时段偏好 (0=白天, 1=晚上, 2=深夜)
    watch_time_pref = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.6, 0.2])

    # 是否有家庭共享
    has_family_sharing = np.random.binomial(1, 0.4, n_samples)

    # 历史 NPS (净推荐值, -100 到 100)
    nps_score = np.random.normal(40, 30, n_samples)
    nps_score = np.clip(nps_score, -100, 100)

    # 标准化
    age_norm = (user_age - 35) / 15
    tenure_norm = (tenure_months - 18) / 12
    watch_norm = (monthly_watch_hours - 24) / 15
    diversity_norm = (content_diversity - 0.6) / 0.2

    # === 处理分配 (非随机 - 活跃用户优先) ===
    # 高活跃度用户更可能分到新算法 (渐进式发布)
    propensity = (
        0.5 +
        0.15 * watch_norm +  # 高观看时长
        0.1 * diversity_norm +  # 高多样性
        0.05 * (tenure_months > 24) +  # 长期用户
        0.05 * (nps_score > 50)  # 高满意度
    )
    propensity = np.clip(propensity, 0.25, 0.75)
    T = np.random.binomial(1, propensity)

    # === 基线留存率 (不使用新算法) ===
    baseline_retention_prob = (
        0.75 +  # 基础留存率
        0.08 * tenure_norm +  # 老用户留存高
        0.10 * watch_norm +  # 活跃用户留存高
        0.05 * diversity_norm +  # 多样化用户留存高
        0.03 * has_family_sharing +  # 家庭用户留存高
        0.002 * nps_score  # 满意度影响
    )
    baseline_retention_prob = np.clip(baseline_retention_prob, 0.3, 0.95)

    # === 处理效应 (新算法的异质性效果) ===
    # 新算法对不同用户群效果不同
    effect = (
        0.05 +  # 平均提升 5%
        0.08 * (content_diversity < 0.5) +  # 偏好单一的用户获益更多
        0.06 * (monthly_watch_hours < 20) +  # 低活跃用户获益更多
        0.04 * (user_age < 25) +  # 年轻用户更喜欢新推荐
        -0.02 * (tenure_months > 36) +  # 老用户已有固定偏好
        0.03 * (preferred_genre == 0)  # 剧集用户获益更多
    )
    effect = np.clip(effect, -0.05, 0.2)

    # === 30 天留存 ===
    retention_prob = baseline_retention_prob + effect * T
    retention_prob = np.clip(retention_prob, 0, 1)
    retention_30d = np.random.binomial(1, retention_prob)

    # === 观看时长 (小时/月) ===
    # 新算法提升观看时长
    watch_hours_lift = effect * 10  # 效果转化为时长
    watch_hours = (
        monthly_watch_hours * (1 + watch_hours_lift * T) +
        np.random.randn(n_samples) * 3
    )
    watch_hours = np.clip(watch_hours, 0, 200)

    # === 内容完成率 (看完的比例) ===
    completion_rate = (
        0.65 +
        0.15 * effect * T +
        0.1 * watch_norm +
        np.random.randn(n_samples) * 0.1
    )
    completion_rate = np.clip(completion_rate, 0.2, 1.0)

    # === 用户价值 (LTV 估算) ===
    # 假设月费 15 美元，按留存计算
    subscription_fee = 15
    ltv = retention_30d * subscription_fee * (1 + tenure_months / 12)

    # === 创建 DataFrame ===
    df = pd.DataFrame({
        'user_age': user_age,
        'tenure_months': tenure_months,
        'monthly_watch_hours': monthly_watch_hours,
        'content_diversity': content_diversity,
        'preferred_genre': preferred_genre,
        'primary_device': primary_device,
        'watch_time_pref': watch_time_pref,
        'has_family_sharing': has_family_sharing,
        'nps_score': nps_score,
        'T': T,
        'propensity': propensity,
        'retention_30d': retention_30d,
        'watch_hours': watch_hours,
        'completion_rate': completion_rate,
        'ltv': ltv,
    })

    return df, effect


def generate_uber_surge_pricing_data(
    n_samples: int = 12000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成 Uber 动态定价数据

    场景: Uber Surge Pricing 对供需平衡的影响
    - 特征: 时段、地区、天气、事件等
    - 处理: Surge 倍数 (1.0, 1.5, 2.0, 2.5)
    - 结果: 司机供给、乘客需求、匹配率、等待时间
    - 混淆因素: 需求高峰时段更可能启动 Surge

    业务背景:
    Uber 的动态定价需要平衡供需，既要吸引足够司机，又不能让乘客流失。
    使用回归断点 + IPW 评估不同 Surge 倍数的效果。

    Parameters:
    -----------
    n_samples: 订单请求数量
    seed: 随机种子

    Returns:
    --------
    (df, true_effect): 数据和真实处理效应
    """
    if seed is not None:
        np.random.seed(seed)

    # === 时空特征 ===

    # 时段 (0-23 小时)
    hour = np.random.choice(range(24), n_samples)

    # 星期几 (0=周一, 6=周日)
    day_of_week = np.random.choice(range(7), n_samples)
    is_weekend = (day_of_week >= 5).astype(int)

    # 地区类型 (0=商务区, 1=居民区, 2=娱乐区, 3=机场)
    zone_type = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1])

    # 天气 (0=晴, 1=雨, 2=雪)
    weather = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.25, 0.05])

    # 是否有特殊事件 (演唱会、体育赛事等)
    has_event = np.random.binomial(1, 0.15, n_samples)

    # 基础需求强度 (每分钟订单数)
    base_demand = (
        10 +
        15 * ((hour >= 7) & (hour <= 9)) +  # 早高峰
        20 * ((hour >= 17) & (hour <= 19)) +  # 晚高峰
        10 * is_weekend * ((hour >= 21) & (hour <= 23)) +  # 周末夜生活
        15 * has_event +  # 事件需求
        8 * (weather > 0) +  # 恶劣天气需求增加
        5 * (zone_type == 3)  # 机场需求稳定
    )

    # 基础供给 (可用司机数)
    base_supply = (
        50 +
        20 * ((hour >= 7) & (hour <= 9)) +  # 早高峰司机多
        25 * ((hour >= 17) & (hour <= 19)) +  # 晚高峰司机多
        -10 * (weather == 2) +  # 下雪司机少
        -5 * ((hour >= 0) & (hour <= 5))  # 深夜司机少 (修复运算符优先级)
    )

    # 需求/供给比 (Demand-Supply Ratio)
    ds_ratio = base_demand / (base_supply + 1)

    # === Surge 倍数分配 (基于需求/供给比的断点回归) ===
    # 阈值: 1.0, 1.5, 2.0
    # 添加噪声模拟决策的随机性
    surge_multiplier = np.ones(n_samples)

    for i in range(n_samples):
        ratio = ds_ratio[i] + np.random.randn() * 0.1  # 添加噪声
        if ratio < 0.8:
            surge_multiplier[i] = 1.0
        elif ratio < 1.2:
            surge_multiplier[i] = np.random.choice([1.0, 1.5], p=[0.7, 0.3])
        elif ratio < 1.8:
            surge_multiplier[i] = np.random.choice([1.5, 2.0], p=[0.6, 0.4])
        else:
            surge_multiplier[i] = np.random.choice([2.0, 2.5], p=[0.7, 0.3])

    # === 司机供给响应 (Surge 吸引司机) ===
    # Surge 越高，司机越愿意上线
    supply_increase_pct = (
        0.15 * (surge_multiplier - 1) +  # 每 0.1 倍增加 1.5%
        0.1 * (surge_multiplier - 1) * (hour >= 22)  # 深夜更敏感
    )

    actual_supply = base_supply * (1 + supply_increase_pct) + np.random.randn(n_samples) * 3
    actual_supply = np.clip(actual_supply, 5, 200)

    # === 乘客需求响应 (Surge 抑制需求) ===
    # Surge 越高，部分乘客放弃
    demand_decrease_pct = (
        -0.20 * (surge_multiplier - 1) +  # 每 0.1 倍减少 2%
        -0.15 * (surge_multiplier - 1) * (zone_type == 1)  # 居民区更敏感
    )

    actual_demand = base_demand * (1 + demand_decrease_pct) + np.random.randn(n_samples) * 2
    actual_demand = np.clip(actual_demand, 1, 100)

    # === 匹配率 (司机够用的比例) ===
    match_rate = np.minimum(actual_supply / (actual_demand + 0.1), 1.0)
    match_rate = np.clip(match_rate + np.random.randn(n_samples) * 0.05, 0.1, 1.0)

    # === 等待时间 (分钟) ===
    # 供给越充足，等待越短
    wait_time = (
        5 +
        10 * (1 - match_rate) +  # 匹配率低则等待长
        3 * (weather > 0) +  # 恶劣天气等待长
        np.random.exponential(2, n_samples)
    )
    wait_time = np.clip(wait_time, 1, 30)

    # === 订单完成率 ===
    # 等待太久会取消
    completion_rate = (
        0.9 -
        0.02 * wait_time -
        0.1 * (surge_multiplier > 2.0) +  # 高 Surge 取消多
        np.random.randn(n_samples) * 0.05
    )
    completion_rate = np.clip(completion_rate, 0.3, 1.0)

    # === 收入 (司机 + 平台) ===
    base_fare = 15
    trip_fare = base_fare * surge_multiplier

    # 司机收入 (扣除平台抽成 25%)
    driver_earnings = trip_fare * 0.75 * completion_rate

    # 平台收入
    platform_revenue = trip_fare * 0.25 * completion_rate

    # === 用户满意度 (1-5) ===
    user_satisfaction = (
        4.0 -
        0.3 * (surge_multiplier - 1) -  # Surge 高满意度低
        0.15 * wait_time +  # 等待久满意度低
        0.5 * (completion_rate > 0.8) +
        np.random.randn(n_samples) * 0.3
    )
    user_satisfaction = np.clip(user_satisfaction, 1, 5)

    # === 处理效应 (Surge 对匹配率的因果效应) ===
    # 定义为 Surge 从 1.0 提升到 1.5 的效果
    effect = supply_increase_pct - demand_decrease_pct

    # === 创建 DataFrame ===
    df = pd.DataFrame({
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'zone_type': zone_type,
        'weather': weather,
        'has_event': has_event,
        'base_demand': base_demand,
        'base_supply': base_supply,
        'ds_ratio': ds_ratio,
        'surge_multiplier': surge_multiplier,
        'actual_supply': actual_supply,
        'actual_demand': actual_demand,
        'match_rate': match_rate,
        'wait_time': wait_time,
        'completion_rate': completion_rate,
        'trip_fare': trip_fare,
        'driver_earnings': driver_earnings,
        'platform_revenue': platform_revenue,
        'user_satisfaction': user_satisfaction,
    })

    return df, effect


def plot_causal_dag(case_name: str) -> go.Figure:
    """
    绘制因果图 (DAG)

    Parameters:
    -----------
    case_name: 'doordash', 'netflix', 'uber'

    Returns:
    --------
    Plotly Figure
    """
    fig = go.Figure()

    if case_name == 'doordash':
        # 节点定义
        nodes = {
            'Weather': (0, 2),
            'Distance': (0, 0),
            'Time': (0, 1),
            'Algorithm': (2, 1.5),
            'Delivery Time': (4, 1.5),
            'Satisfaction': (6, 1.5),
        }

        # 边定义 (from -> to)
        edges = [
            ('Weather', 'Algorithm'),
            ('Weather', 'Delivery Time'),
            ('Distance', 'Delivery Time'),
            ('Time', 'Algorithm'),
            ('Time', 'Delivery Time'),
            ('Algorithm', 'Delivery Time'),
            ('Delivery Time', 'Satisfaction'),
        ]

        title = 'DoorDash 配送优化因果图'

    elif case_name == 'netflix':
        nodes = {
            'Watch History': (0, 2),
            'User Activity': (0, 0),
            'Content Pref': (0, 1),
            'Algorithm': (2, 1.5),
            'Retention': (4, 1.5),
            'LTV': (6, 1.5),
        }

        edges = [
            ('Watch History', 'Algorithm'),
            ('Watch History', 'Retention'),
            ('User Activity', 'Algorithm'),
            ('User Activity', 'Retention'),
            ('Content Pref', 'Retention'),
            ('Algorithm', 'Retention'),
            ('Retention', 'LTV'),
        ]

        title = 'Netflix 推荐系统因果图'

    else:  # uber
        nodes = {
            'Hour': (0, 2),
            'Weather': (0, 0),
            'Event': (0, 1),
            'Demand/Supply': (2, 1),
            'Surge': (4, 1.5),
            'Match Rate': (6, 1.5),
            'Revenue': (8, 1.5),
        }

        edges = [
            ('Hour', 'Demand/Supply'),
            ('Weather', 'Demand/Supply'),
            ('Event', 'Demand/Supply'),
            ('Demand/Supply', 'Surge'),
            ('Surge', 'Match Rate'),
            ('Demand/Supply', 'Match Rate'),
            ('Match Rate', 'Revenue'),
        ]

        title = 'Uber Surge Pricing 因果图'

    # 绘制边
    for from_node, to_node in edges:
        x0, y0 = nodes[from_node]
        x1, y1 = nodes[to_node]

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(color='#95A5A6', width=2),
            hoverinfo='none',
            showlegend=False
        ))

        # 箭头
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#95A5A6'
        )

    # 绘制节点
    treatment_node = 'Algorithm' if case_name != 'uber' else 'Surge'
    outcome_node = 'Satisfaction' if case_name == 'doordash' else ('LTV' if case_name == 'netflix' else 'Revenue')

    for node_name, (x, y) in nodes.items():
        if node_name == treatment_node:
            color = '#2D9CDB'  # 处理节点
            size = 25
        elif node_name == outcome_node:
            color = '#27AE60'  # 结果节点
            size = 25
        else:
            color = '#E8E8E8'  # 协变量
            size = 20

        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=size, color=color, line=dict(width=2, color='white')),
            text=[node_name],
            textposition='middle center',
            textfont=dict(size=10, color='#2C3E50'),
            hoverinfo='text',
            hovertext=node_name,
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        template='plotly_white',
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def compute_ate_with_ci(
    Y: np.ndarray,
    T: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    计算 ATE 及其置信区间

    Parameters:
    -----------
    Y: 结果变量
    T: 处理变量 (0/1)
    alpha: 显著性水平

    Returns:
    --------
    (ate, ci_lower, ci_upper)
    """
    from scipy import stats

    y1 = Y[T == 1]
    y0 = Y[T == 0]

    ate = y1.mean() - y0.mean()

    # 计算标准误
    se = np.sqrt(y1.var() / len(y1) + y0.var() / len(y0))

    # 置信区间
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = ate - z * se
    ci_upper = ate + z * se

    return ate, ci_lower, ci_upper
