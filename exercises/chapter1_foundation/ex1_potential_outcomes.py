"""
练习 1: 潜在结果框架

学习目标:
1. 理解 Y(0), Y(1), ITE, ATE 的概念
2. 理解反事实 (counterfactual) 的含义
3. 理解为什么需要因果推断方法

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ==================== 练习 1.1: 理解潜在结果 ====================

def generate_potential_outcomes(
    n: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成带有潜在结果的模拟数据

    假设真实的数据生成过程 (DGP) 为:
    - Y(0) = 5 + 0.5*X + noise
    - Y(1) = 7 + 0.5*X + noise  (即 ATE = 2)
    - T 是完全随机分配的 (50% 概率)

    TODO: 完成数据生成代码

    Returns:
        DataFrame with columns: X, T, Y0, Y1, Y, ITE
    """
    np.random.seed(seed)

    # TODO: 生成协变量 X (标准正态分布)
    X = None  # 你的代码

    # TODO: 生成潜在结果 Y(0) 和 Y(1)
    # Y(0) = 5 + 0.5*X + noise (noise ~ N(0, 1))
    # Y(1) = 7 + 0.5*X + noise
    Y0 = None  # 你的代码
    Y1 = None  # 你的代码

    # TODO: 随机分配处理 T (50% 概率)
    T = None  # 你的代码

    # TODO: 计算观测结果 Y = T * Y(1) + (1-T) * Y(0)
    Y = None  # 你的代码

    # TODO: 计算个体处理效应 ITE = Y(1) - Y(0)
    ITE = None  # 你的代码

    return pd.DataFrame({
        'X': X,
        'T': T,
        'Y0': Y0,
        'Y1': Y1,
        'Y': Y,
        'ITE': ITE
    })


# ==================== 练习 1.2: 计算处理效应 ====================

def calculate_true_ate(df: pd.DataFrame) -> float:
    """
    计算真实 ATE

    真实 ATE = E[Y(1) - Y(0)] = E[ITE]

    由于我们生成了完整的潜在结果，可以直接计算

    TODO: 计算并返回真实 ATE
    """
    # 你的代码
    pass


def calculate_naive_ate(df: pd.DataFrame) -> float:
    """
    计算朴素 ATE 估计

    朴素估计 = E[Y|T=1] - E[Y|T=0]

    TODO: 计算并返回朴素 ATE 估计
    """
    # 你的代码
    pass


# ==================== 练习 1.3: 反事实分析 ====================

def counterfactual_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个个体，找出其观测值和反事实

    TODO: 创建一个新 DataFrame，包含:
    - observed_outcome: 实际观测到的 Y
    - counterfactual_outcome: 未观测到的潜在结果
    - treatment_status: 处理状态
    - individual_te: 个体处理效应

    Returns:
        DataFrame with counterfactual analysis
    """
    # 你的代码
    pass


# ==================== 练习 1.4: 随机化的重要性 ====================

def compare_random_vs_confounded(
    n: int = 1000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    比较随机分配和混淆分配下的估计

    场景 1: 随机分配 (T 与 X 独立)
    场景 2: 混淆分配 (X 高的人更可能接受处理)

    TODO:
    1. 生成两种场景的数据
    2. 计算各自的朴素估计
    3. 比较与真实 ATE 的差异

    Returns:
        (true_ate, random_estimate, confounded_estimate)
    """
    np.random.seed(seed)

    # 真实 ATE
    true_ate = 2.0

    # 协变量
    X = np.random.randn(n)

    # 潜在结果
    Y0 = 5 + 0.5 * X + np.random.randn(n)
    Y1 = 7 + 0.5 * X + np.random.randn(n)

    # TODO: 场景 1 - 随机分配
    # T_random 完全随机 (50% 概率)
    T_random = None  # 你的代码

    # 观测结果
    Y_random = np.where(T_random == 1, Y1, Y0)

    # 朴素估计
    random_estimate = None  # 你的代码

    # TODO: 场景 2 - 混淆分配
    # T_confounded: X 越大越可能被处理
    # 使用 logistic: P(T=1|X) = 1 / (1 + exp(-2*X))
    propensity = None  # 你的代码
    T_confounded = None  # 你的代码

    Y_confounded = np.where(T_confounded == 1, Y1, Y0)
    confounded_estimate = None  # 你的代码

    return true_ate, random_estimate, confounded_estimate


# ==================== 练习 1.5: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 为什么我们无法同时观测到同一个人的 Y(0) 和 Y(1)?

你的答案:


2. 随机实验 (RCT) 为什么能识别 ATE?

你的答案:


3. 在混淆分配的场景中，朴素估计偏高还是偏低? 为什么?

你的答案:


4. ITE (个体处理效应) 在实践中为什么重要? 举一个例子。

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("练习 1: 潜在结果框架")
    print("=" * 50)

    # 测试 1.1
    print("\n1.1 生成潜在结果数据")
    df = generate_potential_outcomes()
    if df is not None:
        print(f"  样本量: {len(df)}")
        print(f"  列: {list(df.columns)}")
        print(df.head())
    else:
        print("  [未完成] 请完成 generate_potential_outcomes 函数")

    # 测试 1.2
    print("\n1.2 计算处理效应")
    if df is not None:
        true_ate = calculate_true_ate(df)
        naive_ate = calculate_naive_ate(df)
        if true_ate is not None and naive_ate is not None:
            print(f"  真实 ATE: {true_ate:.4f}")
            print(f"  朴素估计: {naive_ate:.4f}")
            print(f"  偏差: {naive_ate - true_ate:.4f}")
        else:
            print("  [未完成] 请完成 calculate_true_ate 和 calculate_naive_ate 函数")

    # 测试 1.3
    print("\n1.3 反事实分析")
    if df is not None:
        cf_df = counterfactual_analysis(df)
        if cf_df is not None:
            print(cf_df.head(10))
        else:
            print("  [未完成] 请完成 counterfactual_analysis 函数")

    # 测试 1.4
    print("\n1.4 随机化 vs 混淆")
    results = compare_random_vs_confounded()
    if results[1] is not None:
        true_ate, random_est, confounded_est = results
        print(f"  真实 ATE: {true_ate:.4f}")
        print(f"  随机分配估计: {random_est:.4f} (偏差: {random_est - true_ate:+.4f})")
        print(f"  混淆分配估计: {confounded_est:.4f} (偏差: {confounded_est - true_ate:+.4f})")
    else:
        print("  [未完成] 请完成 compare_random_vs_confounded 函数")

    print("\n" + "=" * 50)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 50)
