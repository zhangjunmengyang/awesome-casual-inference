"""
ApplicationLab 工具函数

提供行业场景数据生成、业务指标计算等功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import plotly.graph_objects as go


def generate_marketing_data(
    n_samples: int = 10000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    生成营销场景数据 (外卖/电商发券)

    场景: 外卖平台优惠券分配
    - 特征: 用户画像 (年龄、消费、频次、最近活跃度等)
    - 处理: 是否发券
    - 结果: 是否下单 (转化)

    Parameters:
    -----------
    n_samples: 样本数量
    seed: 随机种子

    Returns:
    --------
    (df, true_uplift, expected_gmv_lift)
    - df: 包含特征、处理、结果的 DataFrame
    - true_uplift: 真实转化率增量
    - expected_gmv_lift: 预期 GMV 增量 (考虑订单金额)
    """
    if seed is not None:
        np.random.seed(seed)

    # === 用户特征 ===
    age = np.random.uniform(18, 65, n_samples)
    avg_order_value = np.random.lognormal(3.5, 0.8, n_samples)  # 平均订单金额
    order_frequency = np.random.poisson(8, n_samples)  # 历史订单数
    days_since_last_order = np.random.exponential(15, n_samples)  # 距离上次下单天数
    is_member = np.random.binomial(1, 0.25, n_samples)  # 是否会员
    app_sessions = np.random.poisson(20, n_samples)  # APP 打开次数
    city_tier = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])  # 城市等级

    # === 标准化特征 ===
    age_norm = (age - 40) / 15
    aov_norm = (np.log(avg_order_value) - 3.5) / 0.8
    freq_norm = (order_frequency - 8) / 5
    recency_norm = (days_since_last_order - 15) / 10
    session_norm = (app_sessions - 20) / 15

    # === 用户分组 (4 种) ===
    # Persuadables: 对优惠敏感，发券有效
    # Sure Things: 本来就会买，发券浪费
    # Lost Causes: 无论如何都不会买
    # Sleeping Dogs: 发券反而反感，转化下降

    # 计算用户类型概率
    persuadable_score = (
        0.3 * (1 - age_norm) +  # 年轻人更容易被说服
        0.3 * (1 - freq_norm) +  # 低频用户更敏感
        0.2 * recency_norm +  # 流失风险高的用户
        0.2 * (1 - is_member)  # 非会员更敏感
    )

    sure_thing_score = (
        0.4 * freq_norm +  # 高频用户本来就会买
        0.3 * is_member +  # 会员忠诚度高
        0.3 * (1 - recency_norm)  # 最近活跃
    )

    lost_cause_score = (
        0.5 * recency_norm +  # 长期不活跃
        0.3 * (1 - session_norm) +  # APP 使用少
        0.2 * (city_tier - 2) / 1  # 低线城市用户
    )

    # 归一化得分
    total_score = persuadable_score + sure_thing_score + lost_cause_score + 1
    p_persuadable = np.clip(persuadable_score / total_score, 0, 1)
    p_sure_thing = np.clip(sure_thing_score / total_score, 0, 1)
    p_lost_cause = np.clip(lost_cause_score / total_score, 0, 1)
    p_sleeping_dog = np.clip(1 / total_score, 0, 0.15)  # 小部分负面反应

    # === 处理分配 (随机化实验) ===
    T = np.random.binomial(1, 0.5, n_samples)

    # === 基线转化率 (不发券) ===
    baseline_prob = (
        0.15 +  # 基础转化率
        0.05 * freq_norm +  # 高频用户更容易下单
        0.03 * aov_norm +  # 高客单价用户
        0.04 * is_member +  # 会员转化高
        0.02 * session_norm -  # 活跃用户
        0.03 * recency_norm  # 最近活跃的
    )
    baseline_prob = np.clip(baseline_prob, 0.02, 0.6)

    # === Uplift 效应 (异质性) ===
    uplift = (
        p_persuadable * 0.15 +  # Persuadables: +15% 转化
        p_sure_thing * 0.02 +  # Sure Things: +2% (浪费)
        p_lost_cause * 0.01 -  # Lost Causes: +1% (几乎无效)
        p_sleeping_dog * 0.05  # Sleeping Dogs: -5% (负面效应)
    )

    # 添加一些噪声
    uplift = uplift + np.random.randn(n_samples) * 0.02
    uplift = np.clip(uplift, -0.1, 0.3)

    # === 转化结果 ===
    prob = baseline_prob + uplift * T
    prob = np.clip(prob, 0, 1)
    conversion = np.random.binomial(1, prob)

    # === GMV 计算 ===
    # 订单金额 (只有转化的才有)
    order_value = np.where(conversion == 1, avg_order_value, 0)

    # 优惠券成本 (发券才有成本)
    coupon_cost_per_user = 15  # 每张券 15 元
    coupon_cost = T * coupon_cost_per_user

    # 毛利率 30%
    profit_margin = 0.3
    gmv = order_value
    profit = order_value * profit_margin - coupon_cost

    # GMV 增量 (相对于不发券)
    baseline_gmv = baseline_prob * avg_order_value * profit_margin
    gmv_lift = (prob - baseline_prob) * avg_order_value * profit_margin

    # === 创建 DataFrame ===
    df = pd.DataFrame({
        'age': age,
        'avg_order_value': avg_order_value,
        'order_frequency': order_frequency,
        'days_since_last_order': days_since_last_order,
        'is_member': is_member,
        'app_sessions': app_sessions,
        'city_tier': city_tier,
        'T': T,
        'conversion': conversion,
        'order_value': order_value,
        'coupon_cost': coupon_cost,
        'gmv': gmv,
        'profit': profit,
        # 用户分组 (仅用于分析，实际不可观测)
        'user_type_persuadable': p_persuadable,
        'user_type_sure_thing': p_sure_thing,
        'user_type_lost_cause': p_lost_cause,
        'user_type_sleeping_dog': p_sleeping_dog,
    })

    return df, uplift, gmv_lift


def generate_pricing_data(
    n_samples: int = 8000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成定价实验数据 (产品功能 A/B 测试)

    场景: 视频平台新功能测试
    - 处理: 是否看到新功能
    - 结果: 用户留存 / 观看时长

    Parameters:
    -----------
    n_samples: 样本数量
    seed: 随机种子

    Returns:
    --------
    (df, true_effect)
    """
    if seed is not None:
        np.random.seed(seed)

    # 用户特征
    user_age = np.random.uniform(15, 60, n_samples)
    tenure_days = np.random.exponential(180, n_samples)  # 注册天数
    daily_usage_minutes = np.random.lognormal(4, 1, n_samples)  # 日均使用时长
    is_premium = np.random.binomial(1, 0.15, n_samples)  # 是否付费用户
    device_type = np.random.choice(['mobile', 'tablet', 'desktop'], n_samples, p=[0.6, 0.2, 0.2])

    # 标准化
    age_norm = (user_age - 30) / 15
    tenure_norm = (tenure_days - 180) / 120
    usage_norm = (np.log(daily_usage_minutes) - 4) / 1

    # 处理分配 (层次随机化 - 考虑用户活跃度)
    propensity = 0.5 + 0.1 * usage_norm  # 活跃用户略高概率分到实验组
    propensity = np.clip(propensity, 0.3, 0.7)
    T = np.random.binomial(1, propensity)

    # 基线留存率
    baseline_retention_prob = (
        0.6 +
        0.1 * tenure_norm +
        0.15 * usage_norm +
        0.1 * is_premium
    )
    baseline_retention_prob = np.clip(baseline_retention_prob, 0.3, 0.9)

    # 处理效应 (异质性)
    # 新功能对不同用户的效应不同
    effect = (
        0.05 +  # 平均效应 +5%
        0.08 * (1 - age_norm) +  # 年轻用户更喜欢新功能
        0.05 * usage_norm +  # 活跃用户更受益
        0.03 * is_premium -  # 付费用户略微受益
        0.02 * (device_type == 'desktop')  # 桌面端效果略差
    )
    effect = np.clip(effect, -0.05, 0.2)

    # 留存结果
    retention_prob = baseline_retention_prob + effect * T
    retention_prob = np.clip(retention_prob, 0, 1)
    retention = np.random.binomial(1, retention_prob)

    # 观看时长 (分钟)
    watch_time = daily_usage_minutes * (1 + effect * T) * retention
    watch_time = np.maximum(watch_time, 0)

    df = pd.DataFrame({
        'user_age': user_age,
        'tenure_days': tenure_days,
        'daily_usage_minutes': daily_usage_minutes,
        'is_premium': is_premium,
        'device_type': device_type,
        'T': T,
        'retention': retention,
        'watch_time': watch_time,
        'propensity': propensity,
    })

    return df, effect


def generate_driver_incentive_data(
    n_samples: int = 15000,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成网约车司机激励数据

    场景: 网约车平台司机激励实验
    - 处理: 是否给予额外奖励
    - 结果: 司机在线时长 / 完成订单数

    Parameters:
    -----------
    n_samples: 样本数量
    seed: 随机种子

    Returns:
    --------
    (df, true_effect)
    """
    if seed is not None:
        np.random.seed(seed)

    # 司机特征
    driver_rating = np.random.beta(9, 1, n_samples) * 5  # 评分 4.5-5.0 集中
    years_on_platform = np.random.exponential(2, n_samples)
    avg_daily_hours = np.random.gamma(2, 2, n_samples)  # 日均在线时长
    completed_orders = np.random.poisson(500, n_samples)  # 历史完单数
    city_zone = np.random.choice(['downtown', 'suburb', 'airport'], n_samples, p=[0.4, 0.4, 0.2])
    is_fulltime = np.random.binomial(1, 0.6, n_samples)

    # 标准化
    rating_norm = (driver_rating - 4.5) / 0.5
    years_norm = (years_on_platform - 2) / 1.5
    hours_norm = (avg_daily_hours - 4) / 2
    orders_norm = (completed_orders - 500) / 300

    # 处理分配
    T = np.random.binomial(1, 0.5, n_samples)

    # 基线在线时长
    baseline_hours = (
        6 +
        2 * hours_norm +
        1 * is_fulltime +
        0.5 * rating_norm +
        0.5 * (city_zone == 'downtown')
    )
    baseline_hours = np.clip(baseline_hours, 2, 12)

    # 激励效应 (异质性)
    effect = (
        1.5 +  # 平均增加 1.5 小时
        0.5 * (1 - hours_norm) +  # 低活跃司机更敏感
        0.3 * (1 - is_fulltime) +  # 兼职司机更敏感
        0.2 * (1 - years_norm) +  # 新司机更敏感
        -0.3 * (hours_norm > 1)  # 已经很活跃的司机边际效应递减
    )
    effect = np.clip(effect, 0, 3)

    # 在线时长
    online_hours = baseline_hours + effect * T + np.random.randn(n_samples) * 0.5
    online_hours = np.clip(online_hours, 0, 16)

    # 完成订单数 (与在线时长正相关)
    orders_per_hour = 2 + 0.5 * rating_norm
    completed_orders_today = np.random.poisson(online_hours * orders_per_hour)

    # 收入估算 (假设每单 30 元)
    revenue_per_order = 30
    driver_revenue = completed_orders_today * revenue_per_order

    # 激励成本 (给司机的奖金)
    incentive_cost = T * 100  # 每人 100 元

    # 平台 GMV
    platform_gmv = completed_orders_today * revenue_per_order

    df = pd.DataFrame({
        'driver_rating': driver_rating,
        'years_on_platform': years_on_platform,
        'avg_daily_hours': avg_daily_hours,
        'completed_orders_history': completed_orders,
        'city_zone': city_zone,
        'is_fulltime': is_fulltime,
        'T': T,
        'online_hours': online_hours,
        'completed_orders_today': completed_orders_today,
        'driver_revenue': driver_revenue,
        'incentive_cost': incentive_cost,
        'platform_gmv': platform_gmv,
    })

    return df, effect


def compute_roi(
    df: pd.DataFrame,
    uplift_scores: np.ndarray,
    target_fraction: float = 0.3,
    revenue_per_conversion: float = 100,
    cost_per_treatment: float = 15
) -> Dict[str, float]:
    """
    计算 ROI 指标

    Parameters:
    -----------
    df: 包含 T, conversion, profit 等字段的 DataFrame
    uplift_scores: 预测的 uplift 得分
    target_fraction: 目标干预比例
    revenue_per_conversion: 每次转化的收入
    cost_per_treatment: 每次处理的成本

    Returns:
    --------
    ROI 指标字典
    """
    n_samples = len(df)
    n_target = int(target_fraction * n_samples)

    # 按 uplift 排序
    sorted_idx = np.argsort(uplift_scores)[::-1]
    top_idx = sorted_idx[:n_target]

    # 计算实际效果 (需要有 T 和 Y)
    if 'conversion' in df.columns:
        Y = df['conversion'].values
        T = df['T'].values

        # 在 top 用户中，实际的 uplift
        y_top = Y[top_idx]
        t_top = T[top_idx]

        if t_top.sum() > 0 and (1 - t_top).sum() > 0:
            actual_uplift = y_top[t_top == 1].mean() - y_top[t_top == 0].mean()
        else:
            actual_uplift = 0

        # 预期转化数
        expected_conversions = actual_uplift * n_target

        # 收入
        revenue = expected_conversions * revenue_per_conversion

        # 成本
        cost = n_target * cost_per_treatment

        # ROI
        roi = (revenue - cost) / cost if cost > 0 else 0

        # Incremental Profit
        incremental_profit = revenue - cost

        return {
            'roi': roi,
            'revenue': revenue,
            'cost': cost,
            'incremental_profit': incremental_profit,
            'expected_conversions': expected_conversions,
            'target_size': n_target,
            'actual_uplift': actual_uplift
        }

    return {}


def calculate_cuped_variance_reduction(
    Y: np.ndarray,
    T: np.ndarray,
    X_pre: np.ndarray
) -> Tuple[float, float, float]:
    """
    计算 CUPED 方差缩减

    CUPED (Controlled-experiment Using Pre-Experiment Data):
    使用实验前数据作为协变量，减少实验方差

    Y_adj = Y - theta * X_pre
    其中 theta = Cov(Y, X_pre) / Var(X_pre)

    Parameters:
    -----------
    Y: 实验结果
    T: 处理分配
    X_pre: 实验前协变量 (如历史指标)

    Returns:
    --------
    (ate_original, ate_cuped, variance_reduction)
    """
    # 原始 ATE
    ate_original = Y[T == 1].mean() - Y[T == 0].mean()
    var_original = Y[T == 1].var() / (T == 1).sum() + Y[T == 0].var() / (T == 0).sum()

    # CUPED 调整
    theta = np.cov(Y, X_pre)[0, 1] / np.var(X_pre)
    Y_adj = Y - theta * (X_pre - X_pre.mean())

    ate_cuped = Y_adj[T == 1].mean() - Y_adj[T == 0].mean()
    var_cuped = Y_adj[T == 1].var() / (T == 1).sum() + Y_adj[T == 0].var() / (T == 0).sum()

    variance_reduction = (var_original - var_cuped) / var_original

    return ate_original, ate_cuped, variance_reduction
