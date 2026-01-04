"""准实验方法通用工具函数"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def generate_did_data(
    n_treated: int = 500,
    n_control: int = 500,
    treatment_effect: float = 10.0,
    seed: int = 42
) -> pd.DataFrame:
    """生成双重差分模拟数据

    Args:
        n_treated: 处理组样本量
        n_control: 对照组样本量
        treatment_effect: 真实处理效应
        seed: 随机种子

    Returns:
        包含DID数据的DataFrame
    """
    np.random.seed(seed)
    data_list = []

    # 处理组
    for i in range(n_treated):
        user_id = f"treated_{i}"
        baseline = np.random.normal(100, 15)

        # 政策前
        data_list.append({
            'user_id': user_id,
            'group': 'treated',
            'period': 'pre',
            'treat': 1,
            'post': 0,
            'outcome': baseline + np.random.normal(0, 5)
        })

        # 政策后
        time_trend = 40  # 共同时间趋势
        data_list.append({
            'user_id': user_id,
            'group': 'treated',
            'period': 'post',
            'treat': 1,
            'post': 1,
            'outcome': baseline + time_trend + treatment_effect + np.random.normal(0, 5)
        })

    # 对照组
    for i in range(n_control):
        user_id = f"control_{i}"
        baseline = np.random.normal(80, 15)

        # 政策前
        data_list.append({
            'user_id': user_id,
            'group': 'control',
            'period': 'pre',
            'treat': 0,
            'post': 0,
            'outcome': baseline + np.random.normal(0, 5)
        })

        # 政策后
        time_trend = 40
        data_list.append({
            'user_id': user_id,
            'group': 'control',
            'period': 'post',
            'treat': 0,
            'post': 1,
            'outcome': baseline + time_trend + np.random.normal(0, 5)
        })

    df = pd.DataFrame(data_list)
    df['treat_post'] = df['treat'] * df['post']

    return df


def generate_synthetic_control_data(
    n_control_units: int = 6,
    n_periods: int = 31,
    treatment_period: int = 18,
    treatment_effect: float = -15.0,
    seed: int = 42
) -> tuple[pd.DataFrame, int]:
    """生成合成控制模拟数据

    Args:
        n_control_units: 对照单位数量
        n_periods: 时期总数
        treatment_period: 处理开始时期
        treatment_effect: 真实处理效应
        seed: 随机种子

    Returns:
        (数据DataFrame, 处理时期)
    """
    np.random.seed(seed)

    # 时间序列
    years = np.arange(1970, 1970 + n_periods)
    pre_period = years < (1970 + treatment_period)
    post_period = years >= (1970 + treatment_period)

    # 处理单位数据
    treated_trend = -2.0
    treated_base = 120
    treated_pre = treated_base + treated_trend * (years[pre_period] - 1970) + np.random.normal(0, 2, pre_period.sum())
    treated_post = treated_base + treated_trend * (years[post_period] - 1970) + treatment_effect + np.random.normal(0, 2, post_period.sum())
    treated_outcome = np.concatenate([treated_pre, treated_post])

    # 对照单位数据
    control_names = [f'control_{i}' for i in range(n_control_units)]
    control_data = {}

    for i, name in enumerate(control_names):
        base = np.random.uniform(90, 130)
        trend = np.random.uniform(-2.5, -1.5)
        noise = np.random.normal(0, 3, len(years))
        control_data[name] = base + trend * (years - 1970) + noise

    # 构建DataFrame
    df = pd.DataFrame({
        'year': years,
        'treated': treated_outcome,
        **control_data
    })

    return df, treatment_period


def generate_rdd_data(
    n_samples: int = 500,
    cutoff: float = 200.0,
    treatment_effect: float = 15.0,
    noise_std: float = 20.0,
    seed: int = 42
) -> pd.DataFrame:
    """生成断点回归模拟数据

    Args:
        n_samples: 样本量
        cutoff: 门槛值
        treatment_effect: 真实处理效应
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        包含RDD数据的DataFrame
    """
    np.random.seed(seed)

    # 驱动变量
    running_var = np.random.uniform(100, 300, n_samples)

    # 处理状态
    treatment = (running_var >= cutoff).astype(int)

    # 潜在结果
    y0 = 30 + 0.1 * (running_var - cutoff) + np.random.normal(0, noise_std, n_samples)
    y1 = y0 + treatment_effect

    # 观测结果
    outcome = treatment * y1 + (1 - treatment) * y0

    return pd.DataFrame({
        'running_var': running_var,
        'treatment': treatment,
        'outcome': outcome,
        'y0': y0,
        'y1': y1
    })


def generate_iv_data(
    n_samples: int = 1000,
    instrument_strength: float = 0.5,
    treatment_effect: float = -2.0,
    seed: int = 42
) -> pd.DataFrame:
    """生成工具变量模拟数据

    Args:
        n_samples: 样本量
        instrument_strength: 工具变量强度
        treatment_effect: 真实因果效应
        seed: 随机种子

    Returns:
        包含IV数据的DataFrame
    """
    np.random.seed(seed)

    # 不可观测的混淆因素
    confounder = np.random.normal(0, 10, n_samples)

    # 外生的工具变量
    instrument = np.random.normal(0, 5, n_samples)

    # 内生处理变量
    treatment = 10 + instrument_strength * instrument + 0.2 * confounder + np.random.normal(0, 1, n_samples)

    # 结果变量
    outcome = 100 + treatment_effect * treatment + confounder + np.random.normal(0, 2, n_samples)

    return pd.DataFrame({
        'instrument': instrument,
        'treatment': treatment,
        'outcome': outcome,
        'confounder': confounder
    })


def fig_to_chart_data(fig) -> Dict[str, Any]:
    """将Plotly Figure转换为图表数据

    Args:
        fig: Plotly Figure对象

    Returns:
        图表字典数据
    """
    return fig.to_dict()
