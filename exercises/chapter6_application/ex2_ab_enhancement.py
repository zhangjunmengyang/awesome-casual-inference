"""
练习 2: A/B 测试增强

学习目标:
1. 理解传统 A/B 测试的局限性
2. 掌握 CUPED 方差缩减技术
3. 学习异质效应 (HTE) 分析
4. 理解统计功效 (Power) 和样本量规划

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats


# ==================== 练习 2.1: A/B 测试数据生成 ====================

def generate_ab_test_data(
    n_samples: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成 A/B 测试数据

    场景: 视频平台测试新推荐算法
    - 处理: 是否使用新算法 (T=1 新算法, T=0 旧算法)
    - 结果: 用户观看时长 (分钟)
    - 实验前数据: 历史观看时长 (用于 CUPED)

    TODO: 完成数据生成

    Returns:
        DataFrame with columns: user_age, historical_watch_time, T, watch_time
    """
    np.random.seed(seed)

    # TODO: 生成用户特征
    # user_age: 15-60 岁均匀分布
    # historical_watch_time: 历史观看时长，对数正态分布 (mean=4, std=1)
    user_age = None  # 你的代码
    historical_watch_time = None  # 你的代码

    # TODO: 随机分配处理 (50% 概率)
    T = None  # 你的代码

    # TODO: 生成观看时长
    # 基线观看时长: 0.8 * historical_watch_time + noise
    # 处理效应: 年轻用户 (age < 30) +10 分钟, 其他用户 +5 分钟

    watch_time = []
    for i in range(n_samples):
        # 你的代码
        pass

    return pd.DataFrame({
        'user_age': user_age,
        'historical_watch_time': historical_watch_time,
        'T': T,
        'watch_time': watch_time
    })


# ==================== 练习 2.2: CUPED 方差缩减 ====================

def apply_cuped_simple(
    Y: np.ndarray,
    X_pre: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    应用 CUPED (Controlled-experiment Using Pre-Experiment Data)

    核心思想: 使用实验前数据消除用户间差异，减少方差

    Y_adjusted = Y - theta * (X_pre - mean(X_pre))

    其中 theta = Cov(Y, X_pre) / Var(X_pre)

    TODO: 实现 CUPED

    Args:
        Y: 实验结果 (如观看时长)
        X_pre: 实验前协变量 (如历史观看时长)

    Returns:
        (Y_adjusted, theta, variance_reduction)
    """
    # TODO: 计算 theta
    # theta = Cov(Y, X_pre) / Var(X_pre)
    theta = None  # 你的代码: np.cov(Y, X_pre)[0, 1] / np.var(X_pre)

    # TODO: 调整 Y
    # Y_adjusted = Y - theta * (X_pre - mean(X_pre))
    Y_adjusted = None  # 你的代码

    # TODO: 计算方差缩减率
    # variance_reduction = (Var(Y) - Var(Y_adjusted)) / Var(Y)
    var_original = None  # 你的代码
    var_adjusted = None  # 你的代码
    variance_reduction = None  # 你的代码

    return Y_adjusted, theta, variance_reduction


# ==================== 练习 2.3: ATE 估计与显著性检验 ====================

def estimate_ate_with_test(
    Y: np.ndarray,
    T: np.ndarray,
    use_cuped: bool = False,
    X_pre: np.ndarray = None
) -> Dict[str, float]:
    """
    估计 ATE 并进行显著性检验

    TODO: 实现 ATE 估计和 t-检验

    Args:
        Y: 结果变量
        T: 处理指示
        use_cuped: 是否使用 CUPED
        X_pre: 实验前协变量 (如果 use_cuped=True)

    Returns:
        包含 ate, se, t_stat, p_value 的字典
    """
    # TODO: 如果使用 CUPED，先调整 Y
    if use_cuped and X_pre is not None:
        Y_adjusted, theta, var_reduction = apply_cuped_simple(Y, X_pre)
    else:
        Y_adjusted = Y
        var_reduction = 0

    # TODO: 计算 ATE
    # ATE = E[Y|T=1] - E[Y|T=0]
    ate = None  # 你的代码

    # TODO: 计算标准误差
    # SE = sqrt(Var(Y|T=1)/n1 + Var(Y|T=0)/n0)
    n1 = None  # T=1 的样本量
    n0 = None  # T=0 的样本量
    var1 = None  # T=1 的方差
    var0 = None  # T=0 的方差
    se = None  # 你的代码: np.sqrt(var1/n1 + var0/n0)

    # TODO: 计算 t-统计量和 p-值
    # t = ATE / SE
    t_stat = None  # 你的代码
    # p-value: 双边检验
    p_value = None  # 你的代码: 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        'ate': ate,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        'variance_reduction': var_reduction
    }


# ==================== 练习 2.4: 异质效应分析 ====================

def analyze_heterogeneous_effects_simple(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    简单的异质效应分析

    TODO: 按用户年龄分组，计算各组的处理效应

    Args:
        df: 包含 user_age, T, watch_time 的 DataFrame

    Returns:
        各年龄组的处理效应 DataFrame
    """
    # TODO: 将用户按年龄分组
    # 年龄组: [15-25), [25-35), [35-45), [45-60]
    age_groups = None  # 你的代码: pd.cut(df['user_age'], bins=[15, 25, 35, 45, 60])

    results = []
    for group in age_groups.cat.categories if age_groups is not None else []:
        # TODO: 计算该组的处理效应
        mask = None  # 你的代码: age_groups == group

        # 该组的数据
        group_df = df[mask]

        if len(group_df) < 10:
            continue

        # TODO: 计算该组的 ATE
        # ATE = E[Y|T=1] - E[Y|T=0]
        ate_group = None  # 你的代码

        # TODO: 计算标准误差
        se_group = None  # 你的代码

        results.append({
            'age_group': str(group),
            'ate': ate_group,
            'se': se_group,
            'sample_size': len(group_df)
        })

    return pd.DataFrame(results)


# ==================== 练习 2.5: 统计功效分析 ====================

def calculate_power(
    effect_size: float,
    sample_size: int,
    baseline_std: float,
    alpha: float = 0.05
) -> float:
    """
    计算统计功效 (Power)

    Power = P(拒绝 H0 | H0 为假)

    即: 当真实效应存在时，我们能检测到它的概率

    TODO: 实现功效计算

    Args:
        effect_size: 真实效应大小 (ATE)
        sample_size: 每组样本量
        baseline_std: 基线标准差
        alpha: 显著性水平 (通常 0.05)

    Returns:
        power: 统计功效 (0-1)
    """
    # TODO: 计算标准误差
    # SE = baseline_std * sqrt(2 / sample_size)
    se = None  # 你的代码

    # TODO: 计算 Cohen's d (标准化效应量)
    # d = effect_size / baseline_std
    cohen_d = None  # 你的代码

    # TODO: 计算非中心参数
    # ncp = d * sqrt(sample_size / 2)
    ncp = None  # 你的代码

    # TODO: 计算临界值 (双边检验)
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # TODO: 计算 Power
    # Power = P(|Z| > z_alpha | ncp)
    power = None  # 你的代码: 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    return power


def calculate_required_sample_size(
    effect_size: float,
    baseline_std: float,
    power: float = 0.8,
    alpha: float = 0.05
) -> int:
    """
    计算达到目标功效所需的样本量

    TODO: 实现样本量计算

    Args:
        effect_size: 预期效应大小
        baseline_std: 基线标准差
        power: 目标统计功效 (通常 0.8)
        alpha: 显著性水平 (通常 0.05)

    Returns:
        required_n: 每组所需样本量
    """
    # TODO: 使用二分搜索找到所需样本量
    # 提示: 从小到大尝试样本量，直到 power 达到目标

    for n in range(100, 100000, 100):
        current_power = calculate_power(effect_size, n, baseline_std, alpha)
        if current_power is not None and current_power >= power:
            return n

    return 100000  # 如果未找到，返回上限


# ==================== 练习 2.6: CUPED vs 传统方法对比 ====================

def compare_cuped_vs_traditional(
    n_samples: int = 2000,
    n_simulations: int = 100
) -> Dict[str, float]:
    """
    通过模拟对比 CUPED 和传统方法

    TODO: 运行多次模拟，对比两种方法的性能

    Args:
        n_samples: 每次模拟的样本量
        n_simulations: 模拟次数

    Returns:
        对比结果 (显著性检出率、平均 SE 等)
    """
    traditional_significant = 0
    cuped_significant = 0
    traditional_se_list = []
    cuped_se_list = []

    for sim in range(n_simulations):
        # TODO: 生成数据
        df = generate_ab_test_data(n_samples, seed=sim)

        if df is None or len(df) == 0:
            continue

        Y = df['watch_time'].values
        T = df['T'].values
        X_pre = df['historical_watch_time'].values

        # TODO: 传统方法
        traditional_result = estimate_ate_with_test(Y, T, use_cuped=False)
        if traditional_result['p_value'] is not None:
            if traditional_result['p_value'] < 0.05:
                traditional_significant += 1
            traditional_se_list.append(traditional_result['se'])

        # TODO: CUPED 方法
        cuped_result = estimate_ate_with_test(Y, T, use_cuped=True, X_pre=X_pre)
        if cuped_result['p_value'] is not None:
            if cuped_result['p_value'] < 0.05:
                cuped_significant += 1
            cuped_se_list.append(cuped_result['se'])

    return {
        'traditional_power': traditional_significant / n_simulations,
        'cuped_power': cuped_significant / n_simulations,
        'traditional_avg_se': np.mean(traditional_se_list) if traditional_se_list else 0,
        'cuped_avg_se': np.mean(cuped_se_list) if cuped_se_list else 0,
    }


# ==================== 练习 2.7: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. CUPED 为什么能减少方差? 核心原理是什么?

你的答案:


2. CUPED 中的协变量 X_pre 必须满足什么条件?

你的答案:


3. 如果实验前数据 X_pre 与结果 Y 完全不相关，CUPED 还有效吗?

你的答案:


4. 统计功效 (Power) 为 0.8 意味着什么? 为什么通常设为 0.8?

你的答案:


5. 异质效应分析的业务价值是什么? 举一个例子。

你的答案:


6. 在什么情况下，A/B 测试结果不显著但仍然值得上线?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("练习 2: A/B 测试增强")
    print("=" * 50)

    # 测试 2.1
    print("\n2.1 生成 A/B 测试数据")
    df = generate_ab_test_data()
    if df is not None and len(df) > 0 and df['user_age'].iloc[0] is not None:
        print(f"  样本量: {len(df)}")
        print(f"  处理组占比: {df['T'].mean():.2%}")
        print(f"  平均观看时长: {df['watch_time'].mean():.1f} 分钟")
    else:
        print("  [未完成] 请完成 generate_ab_test_data 函数")

    # 测试 2.2
    print("\n2.2 CUPED 方差缩减")
    if df is not None and len(df) > 0 and df['user_age'].iloc[0] is not None:
        Y = df['watch_time'].values
        X_pre = df['historical_watch_time'].values

        Y_adj, theta, var_reduction = apply_cuped_simple(Y, X_pre)
        if Y_adj is not None and theta is not None:
            print(f"  theta 系数: {theta:.4f}")
            print(f"  方差缩减率: {var_reduction:.2%}")
            print(f"  原始方差: {np.var(Y):.2f}")
            print(f"  调整后方差: {np.var(Y_adj):.2f}")
        else:
            print("  [未完成] 请完成 apply_cuped_simple 函数")

    # 测试 2.3
    print("\n2.3 ATE 估计与显著性检验")
    if df is not None and len(df) > 0 and df['user_age'].iloc[0] is not None:
        Y = df['watch_time'].values
        T = df['T'].values
        X_pre = df['historical_watch_time'].values

        # 传统方法
        traditional = estimate_ate_with_test(Y, T, use_cuped=False)
        if traditional['ate'] is not None:
            print(f"\n  传统方法:")
            print(f"    ATE: {traditional['ate']:.2f} 分钟")
            print(f"    标准误: {traditional['se']:.2f}")
            print(f"    t-统计量: {traditional['t_stat']:.2f}")
            print(f"    p-值: {traditional['p_value']:.4f}")
            print(f"    显著性: {'是' if traditional['p_value'] < 0.05 else '否'}")

            # CUPED 方法
            cuped = estimate_ate_with_test(Y, T, use_cuped=True, X_pre=X_pre)
            if cuped['ate'] is not None:
                print(f"\n  CUPED 方法:")
                print(f"    ATE: {cuped['ate']:.2f} 分钟")
                print(f"    标准误: {cuped['se']:.2f} (减少 {(1-cuped['se']/traditional['se'])*100:.1f}%)")
                print(f"    t-统计量: {cuped['t_stat']:.2f}")
                print(f"    p-值: {cuped['p_value']:.4f}")
                print(f"    显著性: {'是' if cuped['p_value'] < 0.05 else '否'}")
        else:
            print("  [未完成] 请完成 estimate_ate_with_test 函数")

    # 测试 2.4
    print("\n2.4 异质效应分析")
    if df is not None and len(df) > 0 and df['user_age'].iloc[0] is not None:
        hte_results = analyze_heterogeneous_effects_simple(df)
        if hte_results is not None and len(hte_results) > 0:
            print(f"\n  各年龄组的处理效应:")
            print(hte_results.to_string(index=False))
        else:
            print("  [未完成] 请完成 analyze_heterogeneous_effects_simple 函数")

    # 测试 2.5
    print("\n2.5 统计功效分析")
    effect_size = 7.0  # 预期效应 7 分钟
    baseline_std = 30.0  # 基线标准差 30 分钟

    power_500 = calculate_power(effect_size, 500, baseline_std)
    power_1000 = calculate_power(effect_size, 1000, baseline_std)
    power_2000 = calculate_power(effect_size, 2000, baseline_std)

    if power_500 is not None:
        print(f"  样本量 500/组: Power = {power_500:.2%}")
        print(f"  样本量 1000/组: Power = {power_1000:.2%}")
        print(f"  样本量 2000/组: Power = {power_2000:.2%}")

        required_n = calculate_required_sample_size(effect_size, baseline_std, power=0.8)
        print(f"\n  达到 80% Power 所需样本量: {required_n}/组")
    else:
        print("  [未完成] 请完成 calculate_power 函数")

    # 测试 2.6
    print("\n2.6 CUPED vs 传统方法对比 (运行 20 次模拟)")
    comparison = compare_cuped_vs_traditional(n_samples=1000, n_simulations=20)
    if comparison.get('traditional_power', 0) > 0:
        print(f"  传统方法检出率: {comparison['traditional_power']:.2%}")
        print(f"  CUPED 方法检出率: {comparison['cuped_power']:.2%}")
        print(f"  传统方法平均 SE: {comparison['traditional_avg_se']:.2f}")
        print(f"  CUPED 方法平均 SE: {comparison['cuped_avg_se']:.2f}")
        print(f"  SE 减少: {(1 - comparison['cuped_avg_se']/comparison['traditional_avg_se'])*100:.1f}%")
    else:
        print("  [未完成] 请完成 compare_cuped_vs_traditional 函数")

    print("\n" + "=" * 50)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 50)
