"""Part 7 高级主题工具函数"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy import stats


def generate_causal_discovery_data(
    n_samples: int = 1000,
    n_variables: int = 6,
    graph_type: str = "chain",
    noise_std: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    生成因果发现数据

    Args:
        n_samples: 样本量
        n_variables: 变量数量
        graph_type: 图类型 ('chain', 'fork', 'collider', 'complex')
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        数据框和真实因果图
    """
    np.random.seed(seed)

    if graph_type == "chain":
        # X1 -> X2 -> X3 -> ... -> Xn
        X = np.zeros((n_samples, n_variables))
        X[:, 0] = np.random.randn(n_samples)

        edges = []
        for i in range(1, n_variables):
            X[:, i] = 0.7 * X[:, i-1] + np.random.randn(n_samples) * noise_std
            edges.append((f'X{i}', f'X{i+1}'))

        true_graph = {'type': 'chain', 'edges': edges}

    elif graph_type == "fork":
        # X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X5
        X = np.zeros((n_samples, n_variables))
        X[:, 0] = np.random.randn(n_samples)
        X[:, 1] = 0.8 * X[:, 0] + np.random.randn(n_samples) * noise_std
        X[:, 2] = 0.7 * X[:, 0] + np.random.randn(n_samples) * noise_std

        if n_variables > 3:
            X[:, 3] = 0.6 * X[:, 1] + np.random.randn(n_samples) * noise_std
        if n_variables > 4:
            X[:, 4] = 0.6 * X[:, 2] + np.random.randn(n_samples) * noise_std
        if n_variables > 5:
            X[:, 5] = 0.5 * X[:, 3] + 0.5 * X[:, 4] + np.random.randn(n_samples) * noise_std

        edges = [('X1', 'X2'), ('X1', 'X3'), ('X2', 'X4'), ('X3', 'X5'), ('X4', 'X6'), ('X5', 'X6')]
        true_graph = {'type': 'fork', 'edges': edges[:n_variables-1]}

    elif graph_type == "collider":
        # X1 -> X3 <- X2
        X = np.zeros((n_samples, n_variables))
        X[:, 0] = np.random.randn(n_samples)
        X[:, 1] = np.random.randn(n_samples)
        X[:, 2] = 0.7 * X[:, 0] + 0.7 * X[:, 1] + np.random.randn(n_samples) * noise_std

        if n_variables > 3:
            X[:, 3] = 0.8 * X[:, 2] + np.random.randn(n_samples) * noise_std
        if n_variables > 4:
            X[:, 4] = 0.6 * X[:, 0] + np.random.randn(n_samples) * noise_std
        if n_variables > 5:
            X[:, 5] = 0.6 * X[:, 1] + np.random.randn(n_samples) * noise_std

        edges = [('X1', 'X3'), ('X2', 'X3'), ('X3', 'X4'), ('X1', 'X5'), ('X2', 'X6')]
        true_graph = {'type': 'collider', 'edges': edges[:min(len(edges), n_variables-1)]}

    else:  # complex
        # 复杂图结构
        X = np.zeros((n_samples, n_variables))
        X[:, 0] = np.random.randn(n_samples)
        X[:, 1] = np.random.randn(n_samples)
        X[:, 2] = 0.6 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * noise_std
        X[:, 3] = 0.7 * X[:, 2] + np.random.randn(n_samples) * noise_std
        if n_variables > 4:
            X[:, 4] = 0.5 * X[:, 1] + 0.4 * X[:, 3] + np.random.randn(n_samples) * noise_std
        if n_variables > 5:
            X[:, 5] = 0.6 * X[:, 3] + 0.3 * X[:, 4] + np.random.randn(n_samples) * noise_std

        edges = [('X1', 'X3'), ('X2', 'X3'), ('X3', 'X4'), ('X2', 'X5'), ('X4', 'X5'), ('X4', 'X6'), ('X5', 'X6')]
        true_graph = {'type': 'complex', 'edges': edges[:min(len(edges), n_variables)]}

    df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(n_variables)])

    return df, true_graph


def generate_continuous_treatment_data(
    n_samples: int = 1000,
    treatment_distribution: str = "uniform",
    seed: int = 42
) -> pd.DataFrame:
    """
    生成连续处理效应数据

    Args:
        n_samples: 样本量
        treatment_distribution: 处理分布类型 ('uniform', 'normal', 'gamma')
        seed: 随机种子

    Returns:
        包含协变量、连续处理和结果的数据框
    """
    np.random.seed(seed)

    # 协变量
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)

    # 连续处理变量（根据分布类型）
    if treatment_distribution == "uniform":
        T = np.random.uniform(0, 10, n_samples)
    elif treatment_distribution == "normal":
        T_mean = X1 + 0.5 * X2
        T = np.random.normal(T_mean, 1, n_samples)
        T = np.clip(T, 0, 10)
    elif treatment_distribution == "gamma":
        shape = 2.0
        scale = 2.0 + 0.5 * X1 + 0.3 * X2
        T = np.random.gamma(shape, np.clip(scale, 0.1, 10), n_samples)
        T = np.clip(T, 0, 20)
    else:
        T = np.random.uniform(0, 10, n_samples)

    # 结果变量（二次剂量响应函数）
    # Y = 100 + 2*X1 - X2 + 3*T - 0.1*T^2 + noise
    Y = 100 + 2*X1 - X2 + 3*T - 0.1*T**2 + np.random.normal(0, 0.5, n_samples)

    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'T': T,
        'Y': Y
    })

    return df


def generate_time_varying_data(
    n_subjects: int = 500,
    n_periods: int = 5,
    treatment_pattern: str = "random",
    seed: int = 42
) -> pd.DataFrame:
    """
    生成时变处理效应数据

    Args:
        n_subjects: 个体数量
        n_periods: 时间周期数
        treatment_pattern: 处理模式 ('random', 'increasing', 'alternating')
        seed: 随机种子

    Returns:
        长格式数据框（每行代表一个个体-时间点）
    """
    np.random.seed(seed)

    data_list = []

    for i in range(n_subjects):
        # 个体固定效应
        individual_effect = np.random.normal(0, 1)

        # 初始协变量
        X_baseline = np.random.normal(0, 1)

        # 累积处理效应
        cumulative_effect = 0

        for t in range(n_periods):
            # 时变协变量
            X_t = X_baseline + 0.3 * t + np.random.normal(0, 0.5)

            # 时变处理
            if treatment_pattern == "random":
                T_t = np.random.binomial(1, 0.5)
            elif treatment_pattern == "increasing":
                prob = min(0.1 + 0.15 * t, 0.9)
                T_t = np.random.binomial(1, prob)
            elif treatment_pattern == "alternating":
                T_t = t % 2
            else:
                T_t = np.random.binomial(1, 0.5)

            # 累积效应
            cumulative_effect += T_t * 2.0

            # 结果变量（依赖于当前和历史处理）
            Y_t = (50 + individual_effect * 5 + 3 * X_t +
                   cumulative_effect + np.random.normal(0, 2))

            data_list.append({
                'subject_id': i,
                'period': t,
                'X': X_t,
                'T': T_t,
                'Y': Y_t,
                'cumulative_treatment': cumulative_effect / 2.0  # 累积处理次数
            })

    df = pd.DataFrame(data_list)
    return df


def generate_mediation_data(
    n_samples: int = 1000,
    direct_effect: float = 2.0,
    indirect_effect: float = 1.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成中介分析数据

    Args:
        n_samples: 样本量
        direct_effect: 直接效应大小
        indirect_effect: 间接效应大小
        seed: 随机种子

    Returns:
        包含处理、中介变量和结果的数据框
    """
    np.random.seed(seed)

    # 协变量
    X = np.random.normal(0, 1, n_samples)

    # 处理变量
    T = np.random.binomial(1, 0.5, n_samples)

    # 中介变量 M (受 T 和 X 影响)
    M = 0.5 * X + indirect_effect * T + np.random.normal(0, 0.5, n_samples)

    # 结果变量 Y (受 T, M, X 影响)
    # 直接效应: T -> Y
    # 间接效应: T -> M -> Y
    Y = (10 + 2 * X + direct_effect * T + 1.0 * M +
         np.random.normal(0, 1, n_samples))

    df = pd.DataFrame({
        'X': X,
        'T': T,
        'M': M,
        'Y': Y
    })

    return df


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """计算相关系数矩阵"""
    return df.corr()


def standardize_data(X: np.ndarray) -> np.ndarray:
    """标准化数据"""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def compute_partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    计算偏相关系数

    Args:
        x: 变量 X
        y: 变量 Y
        z: 控制变量 Z（可选）

    Returns:
        偏相关系数和 p 值
    """
    if z is None:
        # 边缘相关
        corr, pval = stats.pearsonr(x, y)
        return corr, pval

    # 偏相关：通过回归残差
    from sklearn.linear_model import LinearRegression

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # 回归 X ~ Z 和 Y ~ Z
    reg_x = LinearRegression().fit(z, x)
    reg_y = LinearRegression().fit(z, y)

    # 残差
    res_x = x - reg_x.predict(z)
    res_y = y - reg_y.predict(z)

    # 残差相关
    corr, pval = stats.pearsonr(res_x, res_y)

    return corr, pval
