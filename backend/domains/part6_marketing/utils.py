"""Part 6 Marketing - 工具函数

提供营销场景的数据生成和辅助函数。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def generate_user_journey_data(
    n_users: int = 1000,
    channels: Optional[List[str]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """生成用户转化路径数据

    Args:
        n_users: 用户数量
        channels: 渠道列表，默认为 ['Search', 'Social', 'Email', 'Display', 'Affiliate']
        seed: 随机种子

    Returns:
        DataFrame with columns: user_id, path, path_list, converted, revenue, touchpoints
    """
    np.random.seed(seed)

    if channels is None:
        channels = ['Search', 'Social', 'Email', 'Display', 'Affiliate']

    # 常见路径模式
    common_paths = [
        ['Social', 'Search'],
        ['Search'],
        ['Social', 'Email', 'Search'],
        ['Display', 'Search'],
        ['Email', 'Search'],
        ['Social', 'Search', 'Email', 'Search'],
        ['Affiliate', 'Search'],
        ['Display', 'Social', 'Search'],
    ]

    # 每种路径的转化概率
    conversion_probs = [0.05, 0.03, 0.15, 0.08, 0.10, 0.25, 0.12, 0.18]

    data = []

    for user_id in range(n_users):
        # 随机选择路径模式
        path_idx = np.random.choice(
            len(common_paths),
            p=[0.15, 0.20, 0.10, 0.12, 0.13, 0.08, 0.10, 0.12]
        )
        path = common_paths[path_idx].copy()

        # 随机添加噪声（重复触点）
        if np.random.rand() < 0.3:
            path.append(np.random.choice(channels))

        # 判断是否转化
        converted = np.random.rand() < conversion_probs[path_idx]

        # 转化金额
        revenue = np.random.lognormal(4.5, 0.5) if converted else 0

        data.append({
            'user_id': user_id,
            'path': ' > '.join(path),
            'path_list': path,
            'converted': converted,
            'revenue': revenue,
            'touchpoints': len(path)
        })

    df = pd.DataFrame(data)

    # 只保留转化用户（归因只针对转化用户）
    df_converted = df[df['converted']].reset_index(drop=True)

    return df_converted


def generate_marketing_data(
    n_samples: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """生成发券场景数据

    场景：外卖平台发券实验
    - 特征：用户年龄、历史订单数、最近活跃度
    - 处理：是否发券 (T=1 发券, T=0 不发券)
    - 结果：是否下单 (conversion=1 下单, conversion=0 未下单)

    用户类型:
    - Persuadables: 年轻 + 低频用户（被优惠券吸引）
    - Sure Things: 高频用户（本来就爱点外卖）
    - Sleeping Dogs: 低频老用户（反感促销）
    - Lost Causes: 其他

    Returns:
        DataFrame with columns: age, order_freq, days_since_last, T, conversion, user_type
    """
    np.random.seed(seed)

    # 生成用户特征
    age = np.random.uniform(18, 65, n_samples)
    order_freq = np.random.poisson(5, n_samples)
    days_since_last = np.random.exponential(10, n_samples)

    # 随机分配处理 (50% 概率发券)
    T = np.random.binomial(1, 0.5, n_samples)

    # 确定用户类型和转化
    user_type = []
    conversion = []

    for i in range(n_samples):
        # 判断用户类型
        if age[i] < 30 and order_freq[i] < 5:
            user_type.append('Persuadables')
        elif order_freq[i] >= 8:
            user_type.append('Sure Things')
        elif order_freq[i] < 2 and age[i] >= 40:
            user_type.append('Sleeping Dogs')
        else:
            user_type.append('Lost Causes')

        # 生成转化结果
        base_prob = 0.1 + 0.02 * min(order_freq[i], 10)

        if user_type[i] == 'Persuadables':
            # 发券效果很好：+0.25
            prob = base_prob + (0.25 if T[i] == 1 else 0)
        elif user_type[i] == 'Sure Things':
            # 本来就会买：基线高，发券效果小
            prob = base_prob + 0.3 + (0.03 if T[i] == 1 else 0)
        elif user_type[i] == 'Sleeping Dogs':
            # 发券反而降低：-0.1
            prob = base_prob + 0.15 + (-0.1 if T[i] == 1 else 0)
        else:  # Lost Causes
            # 发不发都不买
            prob = base_prob + (0.01 if T[i] == 1 else 0)

        # 确保概率在 [0, 1] 范围内
        prob = np.clip(prob, 0, 1)
        conversion.append(np.random.binomial(1, prob))

    return pd.DataFrame({
        'age': age,
        'order_freq': order_freq,
        'days_since_last': days_since_last,
        'T': T,
        'conversion': conversion,
        'user_type': user_type
    })


def generate_driver_data(
    n_samples: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """生成网约车司机激励数据

    场景：平台给司机发放奖励，激励其增加在线时长

    Returns:
        DataFrame with columns: rating, order_history, is_fulltime, T, online_hours
    """
    np.random.seed(seed)

    # 生成司机特征
    rating = np.random.beta(8, 1, n_samples) * 1.0 + 4.0  # 4.0-5.0
    order_history = np.random.poisson(200, n_samples)
    is_fulltime = np.random.binomial(1, 0.3, n_samples)

    # 随机分配处理
    T = np.random.binomial(1, 0.5, n_samples)

    # 生成在线时长
    online_hours = []
    for i in range(n_samples):
        # 基线时长
        if is_fulltime[i] == 1:
            base = 6 + np.random.randn() * 1.5
        else:
            base = 3 + np.random.randn() * 1.0

        # 激励效应 (异质性!)
        if T[i] == 1:
            # 兼职效应更大
            fulltime_effect = 0.5 if is_fulltime[i] == 1 else 2.0
            # 低单量效应更大
            order_effect = 2.5 if order_history[i] < 150 else 1.0
            # 综合效应
            effect = (fulltime_effect + order_effect) / 2
        else:
            effect = 0

        online_hours.append(max(0, base + effect))

    return pd.DataFrame({
        'rating': rating,
        'order_history': order_history,
        'is_fulltime': is_fulltime,
        'T': T,
        'online_hours': online_hours
    })


def calculate_response_curve(
    x: np.ndarray,
    a: float,
    c: float,
    alpha: float
) -> np.ndarray:
    """计算响应曲线（Hill Equation）

    R(x) = a * x^alpha / (c^alpha + x^alpha)

    Args:
        x: 投入预算
        a: 饱和收益（最大收益）
        c: 半饱和点（达到 50% 最大收益时的投入）
        alpha: 形状参数

    Returns:
        响应值
    """
    return a * (x ** alpha) / (c ** alpha + x ** alpha)


def calculate_marginal_response(
    x: np.ndarray,
    a: float,
    c: float,
    alpha: float
) -> np.ndarray:
    """计算边际响应（响应曲线的导数）

    R'(x) = a * alpha * c^alpha * x^(alpha-1) / (c^alpha + x^alpha)^2

    Args:
        x: 投入预算
        a: 饱和收益
        c: 半饱和点
        alpha: 形状参数

    Returns:
        边际响应值
    """
    return a * alpha * (c ** alpha) * (x ** (alpha - 1)) / ((c ** alpha + x ** alpha) ** 2)
