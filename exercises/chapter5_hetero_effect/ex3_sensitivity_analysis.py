"""
练习 3: 敏感性分析 (Sensitivity Analysis)

学习目标:
1. 理解未观测混淆的影响
2. 实现 Rosenbaum 敏感性边界
3. 计算 E-value
4. 进行稳健性检验
5. 理解敏感性分析在因果推断中的重要性

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats
import matplotlib.pyplot as plt


# ==================== 练习 3.1: 模拟未观测混淆 ====================

def simulate_unobserved_confounding(
    n: int = 1000,
    confounder_strength: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    模拟包含未观测混淆的数据

    数据生成过程:
    - X ~ N(0, 1)  # 观测协变量
    - U ~ N(0, 1)  # 未观测混淆因子
    - P(T=1|X,U) = logistic(0.5*X + strength*U)  # U 影响处理选择
    - Y = 10 + 2*T + 1.5*X + strength*2*U + noise  # U 也影响结果

    真实 ATE = 2.0

    TODO: 完成数据生成

    Args:
        n: 样本量
        confounder_strength: 未观测混淆的强度 (0-1)
        seed: 随机种子

    Returns:
        (df, U, params)
        df: 包含 X, T, Y 的 DataFrame (不包含 U!)
        U: 未观测混淆因子
        params: 真实参数
    """
    np.random.seed(seed)

    # TODO: 生成观测协变量 X
    X = None  # 你的代码

    # TODO: 生成未观测混淆因子 U
    U = None  # 你的代码

    # TODO: 生成处理 T (受 X 和 U 影响)
    # P(T=1) = 1 / (1 + exp(-(0.5*X + confounder_strength*U)))
    propensity_logit = None  # 你的代码
    propensity = None  # 你的代码: sigmoid
    T = None  # 你的代码: 伯努利抽样

    # TODO: 生成结果 Y (受 T, X, U 影响)
    # Y = 10 + 2*T + 1.5*X + confounder_strength*2*U + noise
    noise = np.random.randn(n) * 0.5
    Y = None  # 你的代码

    # 创建 DataFrame (不包含 U!)
    df = pd.DataFrame({
        'X': X,
        'T': T,
        'Y': Y
    })

    params = {
        'true_ate': 2.0,
        'confounder_strength': confounder_strength
    }

    return df, U, params


def compute_naive_ate(df: pd.DataFrame) -> float:
    """
    计算朴素的 ATE 估计 (忽略混淆)

    TODO: 计算 E[Y|T=1] - E[Y|T=0]
    """
    # 你的代码
    pass


def compute_adjusted_ate(df: pd.DataFrame) -> float:
    """
    调整 X 后的 ATE 估计

    使用线性回归: Y ~ T + X

    TODO: 使用回归估计 ATE
    """
    from sklearn.linear_model import LinearRegression

    # TODO: 构建特征矩阵 [T, X]
    # TODO: 拟合回归模型
    # TODO: 返回 T 的系数

    # 你的代码
    pass


# ==================== 练习 3.2: Rosenbaum 敏感性边界 ====================

def compute_rosenbaum_bounds(
    Y: np.ndarray,
    T: np.ndarray,
    gamma: float
) -> Tuple[float, float]:
    """
    计算 Rosenbaum 敏感性边界

    Rosenbaum Γ 参数:
    对于两个协变量相同的个体 i 和 j，
    1/Γ ≤ P(T_i=1)/P(T_j=1) ≤ Γ

    Γ = 1: 无未观测混淆
    Γ > 1: 允许倾向得分相差 Γ 倍

    TODO: 计算给定 Γ 下的 ATE 边界

    简化实现: 使用 ATE 的标准误差估计边界
    真实的 Rosenbaum bounds 更复杂，需要置换检验

    Args:
        Y: 结果变量
        T: 处理变量
        gamma: 敏感性参数 Γ

    Returns:
        (lower_bound, upper_bound)
    """
    # 观测的 ATE
    ate_obs = Y[T == 1].mean() - Y[T == 0].mean()

    if gamma == 1.0:
        # 无偏情况
        return ate_obs, ate_obs

    # TODO: 计算敏感性边界
    # 简化版本: 边界宽度与 gamma 相关

    # 提示: 可以使用 ATE 的标准误差
    n1 = (T == 1).sum()
    n0 = (T == 0).sum()
    var1 = Y[T == 1].var()
    var0 = Y[T == 0].var()

    # TODO: 计算标准误差
    se = None  # 你的代码: np.sqrt(var1/n1 + var0/n0)

    # TODO: 边界宽度随 gamma 增加
    # 简化: bound_width = se * (gamma - 1)
    bound_width = None  # 你的代码

    lower_bound = None  # 你的代码
    upper_bound = None  # 你的代码

    return lower_bound, upper_bound


def sensitivity_curve(
    Y: np.ndarray,
    T: np.ndarray,
    gamma_range: np.ndarray,
    true_ate: float = None
) -> pd.DataFrame:
    """
    计算敏感性曲线

    TODO: 对一系列 Γ 值计算边界

    Args:
        Y, T: 数据
        gamma_range: Γ 的取值范围
        true_ate: 真实 ATE (如果已知)

    Returns:
        DataFrame with columns: gamma, lower, upper, (true_ate)
    """
    results = []

    for gamma in gamma_range:
        # TODO: 计算边界
        lower, upper = compute_rosenbaum_bounds(Y, T, gamma)

        row = {
            'gamma': gamma,
            'lower': lower,
            'upper': upper
        }

        if true_ate is not None:
            row['true_ate'] = true_ate

        results.append(row)

    return pd.DataFrame(results)


# ==================== 练习 3.3: E-value ====================

def compute_e_value(
    observed_rr: float,
    ci_lower: float = None
) -> Dict[str, float]:
    """
    计算 E-value (VanderWeele & Ding 2017)

    E-value: 使观测关联完全被混淆解释所需的最小风险比

    公式: E = RR + sqrt(RR * (RR - 1))

    TODO: 计算 E-value

    Args:
        observed_rr: 观测到的风险比 (Risk Ratio)
        ci_lower: 置信区间下界 (可选)

    Returns:
        {'e_value': ..., 'e_value_ci': ...}
    """
    # TODO: 计算 E-value
    # E = RR + sqrt(RR * (RR - 1))
    e_value = None  # 你的代码

    result = {'e_value': e_value}

    # TODO: 如果有置信区间，计算 CI 的 E-value
    if ci_lower is not None:
        e_value_ci = None  # 你的代码
        result['e_value_ci'] = e_value_ci

    return result


def ate_to_risk_ratio(
    ate: float,
    baseline_mean: float
) -> float:
    """
    将 ATE 转换为风险比

    RR = (baseline + ATE) / baseline

    TODO: 计算风险比
    """
    # 你的代码
    pass


# ==================== 练习 3.4: 稳健性检验 ====================

def placebo_test(
    df: pd.DataFrame,
    outcome_col: str = 'Y',
    placebo_outcome_col: str = 'Y_placebo'
) -> Dict[str, float]:
    """
    Placebo 测试

    使用不应受处理影响的"假"结果变量进行测试
    如果在假结果上也发现显著效应，说明可能存在混淆

    TODO: 实现 placebo 测试

    Args:
        df: 包含 T, Y, Y_placebo 的 DataFrame
        outcome_col: 真实结果列名
        placebo_outcome_col: 假结果列名

    Returns:
        {'true_effect': ..., 'placebo_effect': ..., 'p_value_placebo': ...}
    """
    # TODO: 计算真实结果的效应
    true_effect = None  # 你的代码

    # TODO: 计算假结果的效应
    placebo_effect = None  # 你的代码

    # TODO: 测试假结果效应是否显著 (使用 t-test)
    # 提示: scipy.stats.ttest_ind
    # 你的代码

    return {
        'true_effect': true_effect,
        'placebo_effect': placebo_effect,
        'p_value_placebo': None  # 你的代码
    }


def subgroup_stability_test(
    df: pd.DataFrame,
    subgroup_var: str,
    n_bootstrap: int = 100
) -> pd.DataFrame:
    """
    子群体稳定性测试

    检查效应估计在不同子群体中是否稳定

    TODO: 在不同子群体中估计 ATE

    Args:
        df: 包含 T, Y, subgroup_var 的 DataFrame
        subgroup_var: 用于分组的变量
        n_bootstrap: Bootstrap 次数

    Returns:
        子群体效应的 DataFrame
    """
    # TODO: 对 subgroup_var 进行分组
    # TODO: 计算每个组的 ATE 和置信区间

    # 你的代码
    pass


# ==================== 练习 3.5: 可视化敏感性分析 ====================

def plot_sensitivity_analysis(
    sensitivity_df: pd.DataFrame,
    title: str = "Rosenbaum Sensitivity Analysis"
) -> None:
    """
    可视化敏感性分析结果

    TODO: 绘制敏感性曲线

    参数:
        sensitivity_df: 包含 gamma, lower, upper 的 DataFrame
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # TODO: 绘制边界曲线
    # 上界
    # 你的代码: ax.plot(...)

    # 下界
    # 你的代码: ax.plot(...)

    # 填充区域
    # 你的代码: ax.fill_between(...)

    # TODO: 如果有真实 ATE，绘制水平线
    if 'true_ate' in sensitivity_df.columns:
        # 你的代码
        pass

    # 零线
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Null Effect')

    ax.set_xlabel('Gamma (Γ)', fontsize=12)
    ax.set_ylabel('ATE Estimate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== 练习 3.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 为什么无混淆假设无法从数据中验证?

你的答案:


2. Rosenbaum Γ = 2 意味着什么? 在实践中这算强混淆还是弱混淆?

你的答案:


3. E-value 和 Rosenbaum bounds 有何异同?

你的答案:


4. 在什么情况下应该报告敏感性分析结果?

你的答案:


5. 如果敏感性分析显示结论对未观测混淆很敏感，应该怎么办?

你的答案:


6. Placebo 测试的原理是什么? 它能检测哪种类型的偏差?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 3: 敏感性分析")
    print("=" * 60)

    # 测试 3.1
    print("\n3.1 模拟未观测混淆")
    try:
        df, U, params = simulate_unobserved_confounding(
            n=2000,
            confounder_strength=0.6
        )

        if df is not None and U is not None:
            print(f"  样本量: {len(df)}")
            print(f"  真实 ATE: {params['true_ate']:.4f}")
            print(f"  混淆强度: {params['confounder_strength']:.2f}")

            # 计算不同估计
            naive_ate = compute_naive_ate(df)
            adjusted_ate = compute_adjusted_ate(df)

            if naive_ate is not None:
                print(f"\n  朴素估计 (无调整): {naive_ate:.4f}")
                print(f"  偏差: {naive_ate - params['true_ate']:+.4f}")

            if adjusted_ate is not None:
                print(f"\n  调整 X 后估计: {adjusted_ate:.4f}")
                print(f"  偏差: {adjusted_ate - params['true_ate']:+.4f}")
                print(f"\n  注意: 即使调整了 X，仍有偏差 (因为 U 未观测)")
        else:
            print("  [未完成] 请完成 simulate_unobserved_confounding 函数")
    except Exception as e:
        print(f"  [错误] {e}")

    # 测试 3.2
    print("\n3.2 Rosenbaum 敏感性边界")
    if df is not None:
        try:
            gamma_range = np.linspace(1.0, 3.0, 20)
            sens_df = sensitivity_curve(
                df['Y'].values,
                df['T'].values,
                gamma_range,
                true_ate=params['true_ate']
            )

            if sens_df is not None and len(sens_df) > 0:
                print("\n  Rosenbaum Bounds (部分结果):")
                print(sens_df.head(10).to_string(index=False))

                # 找到包含 0 的最小 gamma
                includes_zero = (sens_df['lower'] <= 0) & (sens_df['upper'] >= 0)
                if includes_zero.any():
                    gamma_threshold = sens_df.loc[includes_zero, 'gamma'].min()
                    print(f"\n  敏感性阈值: Γ = {gamma_threshold:.2f}")
                    print(f"  解读: 如果未观测混淆使倾向得分相差 {gamma_threshold:.1f} 倍，")
                    print(f"       效应可能变为不显著")
                else:
                    print(f"\n  在 Γ ≤ {gamma_range.max():.1f} 范围内，效应始终显著")
            else:
                print("  [未完成] 请完成 sensitivity_curve 函数")
        except Exception as e:
            print(f"  [错误] {e}")

    # 测试 3.3
    print("\n3.3 E-value")
    try:
        if naive_ate is not None:
            # 转换为风险比
            baseline_mean = df[df['T'] == 0]['Y'].mean()
            rr = ate_to_risk_ratio(naive_ate, baseline_mean)

            if rr is not None:
                e_values = compute_e_value(rr)

                if e_values['e_value'] is not None:
                    print(f"  观测风险比: {rr:.4f}")
                    print(f"  E-value: {e_values['e_value']:.4f}")
                    print(f"\n  解读: 需要风险比 ≥ {e_values['e_value']:.2f} 的未观测混淆")
                    print(f"       才能完全解释观测到的关联")
                else:
                    print("  [未完成] 请完成 compute_e_value 函数")
    except Exception as e:
        print(f"  [错误] {e}")

    # 测试 3.4
    print("\n3.4 稳健性检验 - Placebo 测试")
    try:
        # 生成假结果变量 (不应受 T 影响)
        np.random.seed(123)
        df['Y_placebo'] = df['X'] * 2 + np.random.randn(len(df))

        placebo_results = placebo_test(df, 'Y', 'Y_placebo')

        if placebo_results['true_effect'] is not None:
            print(f"  真实结果的效应: {placebo_results['true_effect']:.4f}")
            print(f"  假结果的效应: {placebo_results['placebo_effect']:.4f}")
            if placebo_results['p_value_placebo'] is not None:
                print(f"  假结果 p-value: {placebo_results['p_value_placebo']:.4f}")
                if placebo_results['p_value_placebo'] < 0.05:
                    print("  ⚠️ 警告: 假结果也显示显著效应，可能存在混淆!")
                else:
                    print("  ✓ 假结果无显著效应，通过 placebo 测试")
        else:
            print("  [未完成] 请完成 placebo_test 函数")
    except Exception as e:
        print(f"  [错误] {e}")

    # 测试 3.5
    print("\n3.5 可视化")
    try:
        if 'sens_df' in locals() and sens_df is not None:
            print("  生成敏感性曲线...")
            # plot_sensitivity_analysis(sens_df)
            print("  [提示] 取消注释上一行代码查看可视化")
    except Exception as e:
        print(f"  [错误] {e}")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 60)
    print("\n关键要点:")
    print("- 敏感性分析评估结论对未观测混淆的稳健性")
    print("- 总是报告敏感性分析结果，增加研究的可信度")
    print("- 结合领域知识判断假设的未观测混淆是否合理")
    print("=" * 60)
