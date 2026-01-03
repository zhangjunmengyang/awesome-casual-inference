"""
练习 3: 混淆偏差

学习目标:
1. 理解混淆偏差的来源和影响
2. 掌握 Simpson's Paradox
3. 学习控制混淆的基本方法
4. 理解 omitted variable bias 公式

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.linear_model import LinearRegression


# ==================== 练习 3.1: 混淆偏差公式 ====================

def calculate_confounding_bias(
    df: pd.DataFrame,
    confound_var: str = 'X'
) -> dict:
    """
    计算混淆偏差的各个组成部分

    Omitted Variable Bias 公式:
    bias = gamma * delta

    其中:
    - gamma: 混淆变量 X 对结果 Y 的效应 (控制 T)
    - delta: 混淆变量 X 与处理 T 的关联

    TODO: 计算 gamma, delta, 和 bias

    Returns:
        dict with 'gamma', 'delta', 'bias', 'naive_estimate', 'adjusted_estimate'
    """
    # 朴素估计 (不控制 X)
    model_naive = LinearRegression()
    model_naive.fit(df[['T']], df['Y'])
    naive_estimate = model_naive.coef_[0]

    # 调整估计 (控制 X)
    model_adjusted = LinearRegression()
    model_adjusted.fit(df[['T', confound_var]], df['Y'])
    adjusted_estimate = model_adjusted.coef_[0]

    # TODO: 计算 gamma (X 对 Y 的效应，控制 T)
    # 提示: 这是 model_adjusted 中 X 的系数
    gamma = None  # 你的代码

    # TODO: 计算 delta (X 与 T 的关联)
    # 提示: 回归 T ~ X
    delta = None  # 你的代码

    # TODO: 计算理论偏差
    theoretical_bias = None  # 你的代码

    # 实际偏差
    actual_bias = naive_estimate - adjusted_estimate

    return {
        'gamma': gamma,
        'delta': delta,
        'theoretical_bias': theoretical_bias,
        'actual_bias': actual_bias,
        'naive_estimate': naive_estimate,
        'adjusted_estimate': adjusted_estimate
    }


# ==================== 练习 3.2: 混淆强度实验 ====================

def experiment_confounding_strength(
    n: int = 1000,
    true_ate: float = 2.0,
    confounding_strengths: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    实验不同混淆强度对估计的影响

    TODO:
    1. 对每个混淆强度生成数据
    2. 计算朴素估计和调整估计
    3. 记录偏差

    Returns:
        DataFrame with columns: confounding_strength, naive_estimate, adjusted_estimate, bias
    """
    if confounding_strengths is None:
        confounding_strengths = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    np.random.seed(seed)
    results = []

    for strength in confounding_strengths:
        # TODO: 生成混淆数据
        # X ~ N(0, 1)
        # P(T=1|X) = sigmoid(strength * X)
        # Y = 5 + true_ate * T + strength * X + noise

        X = np.random.randn(n)
        # 你的代码生成 T 和 Y

        T = None
        Y = None

        if T is not None and Y is not None:
            df = pd.DataFrame({'X': X, 'T': T, 'Y': Y})

            # 计算估计
            naive_est = df[df['T'] == 1]['Y'].mean() - df[df['T'] == 0]['Y'].mean()

            model = LinearRegression()
            model.fit(df[['T', 'X']], df['Y'])
            adjusted_est = model.coef_[0]

            results.append({
                'confounding_strength': strength,
                'naive_estimate': naive_est,
                'adjusted_estimate': adjusted_est,
                'bias': naive_est - true_ate
            })

    return pd.DataFrame(results) if results else None


# ==================== 练习 3.3: Simpson's Paradox ====================

def create_simpson_paradox_data(
    n_per_group: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    创建展示 Simpson's Paradox 的数据

    场景: 研究某药物对康复率的影响
    - 有两个医院 (A 和 B)
    - 医院 A 接收重症患者多，药物使用率高
    - 医院 B 接收轻症患者多，药物使用率低
    - 药物实际上有正效应

    TODO: 设计数据使得:
    - 整体: 用药组康复率 < 未用药组康复率 (看起来药物有害!)
    - 分医院: 用药组康复率 > 未用药组康复率 (药物实际有益)

    Returns:
        DataFrame with columns: Hospital, Treatment, Recovery, Severity
    """
    np.random.seed(seed)

    data = []

    # TODO: 医院 A (重症多，用药多)
    # 重症基础康复率低，用药提高康复率
    # 大部分重症患者在这里，大部分接受治疗

    # 你的代码

    # TODO: 医院 B (轻症多，用药少)
    # 轻症基础康复率高
    # 大部分轻症患者在这里，大部分不接受治疗

    # 你的代码

    return pd.DataFrame(data)


def analyze_simpson_paradox(df: pd.DataFrame) -> dict:
    """
    分析 Simpson's Paradox

    TODO:
    1. 计算整体的处理效应
    2. 分医院计算处理效应
    3. 识别 paradox

    Returns:
        dict with overall and stratified effects
    """
    results = {}

    # TODO: 整体效应
    # 你的代码

    # TODO: 医院 A 效应
    # 你的代码

    # TODO: 医院 B 效应
    # 你的代码

    return results


# ==================== 练习 3.4: 敏感性分析 ====================

def sensitivity_to_unmeasured_confounding(
    df: pd.DataFrame,
    gamma_range: np.ndarray = None,
    delta_range: np.ndarray = None
) -> pd.DataFrame:
    """
    敏感性分析: 如果存在未观测混淆，估计会如何变化?

    假设存在未观测混淆变量 U:
    - U 对 Y 的效应为 gamma_u
    - U 与 T 的关联为 delta_u
    - 偏差 = gamma_u * delta_u

    TODO:
    1. 计算当前调整估计
    2. 对不同的 (gamma_u, delta_u) 组合，计算可能的真实效应

    Returns:
        DataFrame with columns: gamma_u, delta_u, adjusted_effect, true_effect_range
    """
    if gamma_range is None:
        gamma_range = np.linspace(-2, 2, 9)
    if delta_range is None:
        delta_range = np.linspace(-1, 1, 9)

    # 当前调整估计
    model = LinearRegression()
    model.fit(df[['T', 'X']], df['Y'])
    current_estimate = model.coef_[0]

    results = []

    # TODO: 对每个 (gamma_u, delta_u) 组合
    for gamma_u in gamma_range:
        for delta_u in delta_range:
            # 可能的偏差
            possible_bias = gamma_u * delta_u
            # 可能的真实效应
            possible_true_effect = current_estimate - possible_bias

            results.append({
                'gamma_u': gamma_u,
                'delta_u': delta_u,
                'possible_bias': possible_bias,
                'possible_true_effect': possible_true_effect
            })

    return pd.DataFrame(results)


# ==================== 练习 3.5: 实际案例分析 ====================

def analyze_marketing_confounding() -> str:
    """
    分析营销场景中的混淆

    场景: 评估促销邮件对购买的影响
    - 处理 T: 是否收到促销邮件
    - 结果 Y: 是否购买
    - 潜在混淆: 用户活跃度、历史购买、偏好等

    TODO: 写出你的分析，回答以下问题:
    1. 列出可能的混淆变量
    2. 每个混淆变量如何同时影响 T 和 Y?
    3. 朴素估计会高估还是低估真实效应?
    4. 如何控制这些混淆?

    Returns:
        你的分析 (字符串)
    """
    analysis = """
    # 营销邮件效应的混淆分析

    ## 1. 潜在混淆变量

    TODO: 列出至少 3 个混淆变量


    ## 2. 混淆机制分析

    TODO: 分析每个变量如何影响 T 和 Y


    ## 3. 偏差方向

    TODO: 分析朴素估计的偏差方向


    ## 4. 控制策略

    TODO: 提出控制混淆的方法

    """

    return analysis


# ==================== 练习 3.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 为什么 Simpson's Paradox 不是悖论? 它告诉我们什么?

你的答案:


2. Omitted Variable Bias 公式中，什么情况下偏差为 0?

你的答案:


3. 在实际中，如何判断是否存在混淆?

你的答案:


4. 如果确信存在未观测混淆，除了敏感性分析还有什么方法?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("练习 3: 混淆偏差")
    print("=" * 50)

    # 生成测试数据
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n)
    T = (np.random.randn(n) + 1.5 * X > 0).astype(int)
    Y = 5 + 2 * T + 1.5 * X + np.random.randn(n) * 0.5
    df = pd.DataFrame({'X': X, 'T': T, 'Y': Y})

    # 测试 3.1
    print("\n3.1 混淆偏差分解")
    bias_result = calculate_confounding_bias(df)
    if bias_result.get('gamma') is not None:
        print(f"  gamma (X->Y): {bias_result['gamma']:.4f}")
        print(f"  delta (X-T): {bias_result['delta']:.4f}")
        print(f"  理论偏差: {bias_result['theoretical_bias']:.4f}")
        print(f"  实际偏差: {bias_result['actual_bias']:.4f}")
    else:
        print("  [未完成] 请完成 calculate_confounding_bias 函数")

    # 测试 3.2
    print("\n3.2 混淆强度实验")
    exp_results = experiment_confounding_strength()
    if exp_results is not None and not exp_results.empty:
        print(exp_results.to_string(index=False))
    else:
        print("  [未完成] 请完成 experiment_confounding_strength 函数")

    # 测试 3.3
    print("\n3.3 Simpson's Paradox")
    simpson_df = create_simpson_paradox_data()
    if simpson_df is not None and not simpson_df.empty:
        analysis = analyze_simpson_paradox(simpson_df)
        if analysis:
            for key, value in analysis.items():
                print(f"  {key}: {value}")
        else:
            print("  [未完成] 请完成 analyze_simpson_paradox 函数")
    else:
        print("  [未完成] 请完成 create_simpson_paradox_data 函数")

    # 测试 3.4
    print("\n3.4 敏感性分析")
    sensitivity = sensitivity_to_unmeasured_confounding(df)
    print(f"  生成了 {len(sensitivity)} 种敏感性场景")
    print(f"  真实效应范围: [{sensitivity['possible_true_effect'].min():.2f}, {sensitivity['possible_true_effect'].max():.2f}]")

    print("\n" + "=" * 50)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 50)
