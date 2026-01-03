"""
练习 2: 逆概率加权 (Inverse Probability Weighting, IPW)

学习目标:
1. 理解 IPW 的核心思想 - 创造"伪总体"
2. 实现 IPW 权重计算
3. 使用加权方法估计 ATE
4. 理解极端权重的问题和解决方法
5. 诊断权重分布和有效样本量

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.linear_model import LogisticRegression


# ==================== 练习 2.1: 理解 IPW 权重 ====================

def compute_ipw_weights(
    propensity: np.ndarray,
    treatment: np.ndarray
) -> np.ndarray:
    """
    计算 IPW 权重

    权重公式:
    - 处理组: w_i = 1 / e(X_i)
    - 控制组: w_i = 1 / (1 - e(X_i))

    完整权重: w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))

    TODO: 实现 IPW 权重计算

    Args:
        propensity: 倾向得分 e(X)
        treatment: 处理状态 T

    Returns:
        IPW 权重数组
    """
    # TODO: 避免除零，裁剪倾向得分到 [0.01, 0.99]
    propensity_clipped = None  # 你的代码

    # TODO: 计算权重
    # w = T/e + (1-T)/(1-e)
    weights = None  # 你的代码

    return weights


def clip_extreme_weights(
    weights: np.ndarray,
    percentile: float = 99
) -> np.ndarray:
    """
    裁剪极端权重

    极端权重会导致估计不稳定，裁剪到某个百分位数

    TODO: 实现权重裁剪

    Args:
        weights: 原始权重
        percentile: 裁剪的百分位数

    Returns:
        裁剪后的权重
    """
    # TODO: 计算权重的 percentile 分位数
    max_weight = None  # 你的代码

    # TODO: 裁剪权重
    clipped_weights = None  # 你的代码

    return clipped_weights


# ==================== 练习 2.2: IPW 估计 ATE ====================

def estimate_ate_ipw(
    Y: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray
) -> Tuple[float, float]:
    """
    使用 IPW 估计 ATE

    ATE = E[Y(1)] - E[Y(0)]
        = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]

    使用 Hajek 估计器 (归一化权重):
    E[Y(1)] = Σ(Y_i * w_i * T_i) / Σ(w_i * T_i)
    E[Y(0)] = Σ(Y_i * w_i * (1-T_i)) / Σ(w_i * (1-T_i))

    TODO: 实现 IPW 估计

    Args:
        Y: 结果变量
        treatment: 处理状态
        weights: IPW 权重

    Returns:
        (ATE估计, 标准误)
    """
    # 处理组和控制组 mask
    treated_mask = treatment == 1
    control_mask = treatment == 0

    # TODO: 计算加权的 E[Y(1)]
    y1_weighted = None  # 你的代码

    # TODO: 计算加权的 E[Y(0)]
    y0_weighted = None  # 你的代码

    # TODO: 计算 ATE
    ate = None  # 你的代码

    # TODO: 计算标准误 (简化版)
    # 使用影响函数方法
    n = len(Y)

    # 处理组残差
    residuals_1 = np.zeros(n)
    residuals_1[treated_mask] = (Y[treated_mask] - y1_weighted) * weights[treated_mask]

    # 控制组残差
    residuals_0 = np.zeros(n)
    residuals_0[control_mask] = (Y[control_mask] - y0_weighted) * weights[control_mask]

    # 影响函数
    influence_fn = residuals_1 - residuals_0
    variance = np.var(influence_fn) / n
    se = np.sqrt(variance)

    return ate, se


# ==================== 练习 2.3: 权重诊断 ====================

def diagnose_weights(
    weights: np.ndarray,
    treatment: np.ndarray
) -> dict:
    """
    诊断 IPW 权重分布

    TODO: 计算权重统计量

    Args:
        weights: IPW 权重
        treatment: 处理状态

    Returns:
        dict with weight diagnostics
    """
    treated_weights = weights[treatment == 1]
    control_weights = weights[treatment == 0]

    # TODO: 计算有效样本量 (Effective Sample Size)
    # ESS = (sum(w))^2 / sum(w^2)
    ess = None  # 你的代码
    ess_fraction = None  # ESS / n

    # TODO: 计算权重统计量
    stats = {
        'n_samples': len(weights),
        'ess': ess,
        'ess_fraction': ess_fraction,
        'treated_weight_mean': None,  # 你的代码
        'treated_weight_max': None,   # 你的代码
        'control_weight_mean': None,  # 你的代码
        'control_weight_max': None,   # 你的代码
        'max_weight': None,           # 你的代码
        'weight_cv': None             # coefficient of variation: std/mean
    }

    return stats


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    计算有效样本量

    ESS 衡量权重的分散程度
    ESS = (sum(w))^2 / sum(w^2)

    当所有权重相等时，ESS = n
    当权重差异很大时，ESS << n

    TODO: 计算 ESS

    Returns:
        有效样本量
    """
    # 你的代码
    pass


# ==================== 练习 2.4: 稳定权重 ====================

def compute_stabilized_weights(
    propensity: np.ndarray,
    treatment: np.ndarray
) -> np.ndarray:
    """
    计算稳定权重 (Stabilized Weights)

    稳定权重在分子上加入边际处理概率，减少权重的方差:

    w_stab = P(T) / P(T|X)  for treated
    w_stab = P(1-T) / P(1-T|X)  for control

    其中 P(T) 是边际处理概率 (样本中的处理比例)

    TODO: 实现稳定权重计算

    Args:
        propensity: 倾向得分
        treatment: 处理状态

    Returns:
        稳定权重数组
    """
    # TODO: 计算边际处理概率
    marginal_prob = None  # mean(T)

    # TODO: 裁剪倾向得分
    propensity_clipped = None  # 你的代码

    # TODO: 计算稳定权重
    weights_stab = np.zeros(len(treatment))

    # 处理组
    # weights_stab[treated] = marginal_prob / propensity[treated]

    # 控制组
    # weights_stab[control] = (1 - marginal_prob) / (1 - propensity[control])

    # 你的代码

    return weights_stab


# ==================== 练习 2.5: 比较不同权重方法 ====================

def compare_weighting_methods(
    propensity: np.ndarray,
    treatment: np.ndarray,
    Y: np.ndarray
) -> pd.DataFrame:
    """
    比较不同的加权方法

    TODO: 比较以下方法:
    1. 标准 IPW
    2. 裁剪权重的 IPW (99th percentile)
    3. 稳定权重
    4. 裁剪的稳定权重

    Args:
        propensity: 倾向得分
        treatment: 处理状态
        Y: 结果变量

    Returns:
        DataFrame with comparison results
    """
    results = []

    # TODO: 1. 标准 IPW
    weights_standard = None  # 你的代码
    ate_standard, se_standard = None, None  # 你的代码
    ess_standard = None  # 你的代码

    results.append({
        'method': 'Standard IPW',
        'ate': ate_standard,
        'se': se_standard,
        'ess': ess_standard,
        'max_weight': weights_standard.max() if weights_standard is not None else None
    })

    # TODO: 2. 裁剪权重
    # 你的代码

    # TODO: 3. 稳定权重
    # 你的代码

    # TODO: 4. 裁剪的稳定权重
    # 你的代码

    return pd.DataFrame(results)


# ==================== 练习 2.6: IPW 实验 ====================

def generate_ipw_data(
    n: int = 2000,
    confounding_strength: float = 1.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, float]:
    """
    生成用于 IPW 实验的数据

    TODO: 生成有混淆的数据

    Args:
        n: 样本量
        confounding_strength: 混淆强度
        seed: 随机种子

    Returns:
        (DataFrame, true_ate)
    """
    np.random.seed(seed)

    # TODO: 生成协变量
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)

    # TODO: 生成倾向得分
    # logit(e) = confounding_strength * (X1 + 0.5*X2)
    propensity_logit = None  # 你的代码
    propensity = None  # 1 / (1 + exp(-logit))

    # TODO: 生成处理
    T = None  # 你的代码

    # TODO: 生成结果
    # Y = 5 + 2*T + 1.5*X1 + X2 + noise
    true_ate = 2.0
    Y = None  # 你的代码

    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'T': T,
        'Y': Y
    })

    return df, true_ate


def run_ipw_experiment(
    confounding_strengths: list = [0.5, 1.0, 1.5, 2.0, 2.5]
) -> pd.DataFrame:
    """
    在不同混淆强度下运行 IPW 实验

    TODO: 对每个混淆强度，生成数据并估计 ATE

    Returns:
        DataFrame with experiment results
    """
    results = []

    for strength in confounding_strengths:
        # TODO: 生成数据
        df, true_ate = None, None  # 你的代码

        if df is None:
            continue

        # TODO: 估计倾向得分
        X = df[['X1', 'X2']].values
        T = df['T'].values
        Y = df['Y'].values

        # TODO: 计算朴素估计
        naive_ate = None  # 你的代码

        # TODO: IPW 估计
        # 你的代码

        results.append({
            'confounding_strength': strength,
            'true_ate': true_ate,
            'naive_ate': naive_ate,
            'naive_bias': None,  # naive_ate - true_ate
            'ipw_ate': None,
            'ipw_bias': None,
            'ipw_se': None
        })

    return pd.DataFrame(results)


# ==================== 练习 2.7: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. IPW 的核心思想是什么? 为什么重加权可以去除混淆偏差?

你的答案:


2. IPW 权重的含义是什么? 为什么处理组用 1/e(X)，控制组用 1/(1-e(X))?

你的答案:


3. 什么情况下会出现极端权重? 极端权重有什么问题?

你的答案:


4. 有效样本量 (ESS) 的含义是什么? ESS 小意味着什么?

你的答案:


5. 稳定权重相比标准权重有什么优势?

你的答案:


6. IPW 相比 PSM 有什么优缺点?

你的答案:


7. 如果倾向得分模型误设定，IPW 估计会怎样?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 2: 逆概率加权 (IPW)")
    print("=" * 60)

    # 测试 2.1
    print("\n2.1 生成数据")
    df, true_ate = generate_ipw_data(n=2000, confounding_strength=1.5)
    if df is not None and 'X1' in df.columns:
        print(f"  样本量: {len(df)}")
        print(f"  真实 ATE: {true_ate:.4f}")

        X = df[['X1', 'X2']].values
        T = df['T'].values
        Y = df['Y'].values

        # 朴素估计
        naive_ate = df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()
        print(f"  朴素估计: {naive_ate:.4f} (偏差: {naive_ate - true_ate:.4f})")
    else:
        print("  [未完成] 请完成 generate_ipw_data 函数")

    # 测试 2.2
    print("\n2.2 估计倾向得分")
    if df is not None:
        # 估计倾向得分
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, T)
        propensity = lr.predict_proba(X)[:, 1]

        print(f"  倾向得分范围: [{propensity.min():.4f}, {propensity.max():.4f}]")
        print(f"  处理组平均: {propensity[T==1].mean():.4f}")
        print(f"  控制组平均: {propensity[T==0].mean():.4f}")

    # 测试 2.3
    print("\n2.3 计算 IPW 权重")
    weights = compute_ipw_weights(propensity, T)
    if weights is not None:
        print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  平均权重: {weights.mean():.4f}")
        print(f"  最大权重: {weights.max():.4f}")
    else:
        print("  [未完成] 请完成 compute_ipw_weights 函数")

    # 测试 2.4
    print("\n2.4 权重诊断")
    if weights is not None:
        diag = diagnose_weights(weights, T)
        if diag['ess'] is not None:
            print(f"  有效样本量: {diag['ess']:.1f} ({diag['ess_fraction']*100:.1f}%)")
            print(f"  处理组权重: 均值={diag['treated_weight_mean']:.4f}, 最大={diag['treated_weight_max']:.4f}")
            print(f"  控制组权重: 均值={diag['control_weight_mean']:.4f}, 最大={diag['control_weight_max']:.4f}")
        else:
            print("  [未完成] 请完成 diagnose_weights 函数")

    # 测试 2.5
    print("\n2.5 IPW 估计 ATE")
    if weights is not None:
        ipw_ate, ipw_se = estimate_ate_ipw(Y, T, weights)
        if ipw_ate is not None:
            print(f"  IPW ATE: {ipw_ate:.4f} ± {ipw_se:.4f}")
            print(f"  95% CI: [{ipw_ate - 1.96*ipw_se:.4f}, {ipw_ate + 1.96*ipw_se:.4f}]")
            print(f"  偏差: {ipw_ate - true_ate:.4f}")
        else:
            print("  [未完成] 请完成 estimate_ate_ipw 函数")

    # 测试 2.6
    print("\n2.6 裁剪极端权重")
    weights_clipped = clip_extreme_weights(weights, percentile=99)
    if weights_clipped is not None:
        print(f"  裁剪前最大权重: {weights.max():.4f}")
        print(f"  裁剪后最大权重: {weights_clipped.max():.4f}")

        ipw_ate_clipped, ipw_se_clipped = estimate_ate_ipw(Y, T, weights_clipped)
        if ipw_ate_clipped is not None:
            print(f"  裁剪后 ATE: {ipw_ate_clipped:.4f} ± {ipw_se_clipped:.4f}")
    else:
        print("  [未完成] 请完成 clip_extreme_weights 函数")

    # 测试 2.7
    print("\n2.7 稳定权重")
    weights_stab = compute_stabilized_weights(propensity, T)
    if weights_stab is not None and weights_stab.sum() > 0:
        print(f"  稳定权重范围: [{weights_stab.min():.4f}, {weights_stab.max():.4f}]")
        print(f"  标准权重 max: {weights.max():.4f}")
        print(f"  稳定权重 max: {weights_stab.max():.4f}")

        ess_standard = compute_effective_sample_size(weights)
        ess_stab = compute_effective_sample_size(weights_stab)
        if ess_standard is not None:
            print(f"  标准权重 ESS: {ess_standard:.1f}")
            print(f"  稳定权重 ESS: {ess_stab:.1f}")
    else:
        print("  [未完成] 请完成 compute_stabilized_weights 函数")

    # 测试 2.8
    print("\n2.8 比较不同权重方法")
    comparison = compare_weighting_methods(propensity, T, Y)
    if comparison is not None and not comparison.empty:
        print(comparison.to_string(index=False))
    else:
        print("  [未完成] 请完成 compare_weighting_methods 函数")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("提示: 稳定权重和裁剪权重可以提高估计的稳定性")
    print("=" * 60)
