"""
练习 1: 倾向得分匹配 (Propensity Score Matching)

学习目标:
1. 理解倾向得分的概念和作用
2. 实现简单的倾向得分匹配 (PSM)
3. 评估匹配质量 (SMD, 方差比)
4. 理解 PSM 的假设和局限性

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


# ==================== 练习 1.1: 理解倾向得分 ====================

def estimate_propensity_score(
    X: np.ndarray,
    T: np.ndarray
) -> np.ndarray:
    """
    估计倾向得分 e(X) = P(T=1|X)

    使用逻辑回归估计个体接受处理的概率

    Args:
        X: 特征矩阵 (n, p)
        T: 处理状态 (n,)

    TODO: 完成倾向得分估计

    Returns:
        倾向得分数组 (n,)
    """
    # TODO: 使用 LogisticRegression 拟合 T ~ X
    # 提示: 使用 predict_proba 获取概率

    # 你的代码
    pass


def generate_confounded_data(
    n: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成有混淆的观测数据

    DAG: X -> T, X -> Y, T -> Y

    DGP:
    - X1, X2, X3 ~ N(0, 1)
    - P(T=1|X) = logistic(X1 + 0.5*X2)
    - Y = 5 + 2*T + 1.5*X1 + X2 + noise

    真实 ATE = 2

    TODO: 完成数据生成

    Returns:
        DataFrame with columns: X1, X2, X3, T, Y
    """
    np.random.seed(seed)

    # TODO: 生成三个协变量
    X1 = None  # 你的代码
    X2 = None  # 你的代码
    X3 = None  # 你的代码

    # TODO: 生成处理 T
    # P(T=1|X) = 1 / (1 + exp(-(X1 + 0.5*X2)))
    propensity = None  # 你的代码
    T = None  # 你的代码，使用 np.random.binomial

    # TODO: 生成结果 Y
    # Y = 5 + 2*T + 1.5*X1 + X2 + noise
    Y = None  # 你的代码

    return pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'T': T,
        'Y': Y
    })


# ==================== 练习 1.2: 实现 PSM ====================

def propensity_score_matching(
    propensity: np.ndarray,
    treatment: np.ndarray,
    n_neighbors: int = 1,
    caliper: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行倾向得分匹配

    为每个处理组个体找到倾向得分最接近的控制组个体

    Args:
        propensity: 倾向得分
        treatment: 处理状态
        n_neighbors: 匹配的邻居数量
        caliper: 卡尺宽度 (最大允许的倾向得分差异)

    TODO: 实现 PSM 匹配算法

    Returns:
        (matched_treated_indices, matched_control_indices)
    """
    # 获取处理组和控制组的索引
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    # TODO: 使用 NearestNeighbors 进行匹配
    # 提示:
    # 1. 创建 NearestNeighbors 对象，用控制组的倾向得分拟合
    # 2. 为每个处理组个体找到最近的控制组个体
    # 3. 如果使用 caliper，过滤掉距离超过 caliper 的匹配

    knn = None  # 你的代码

    # 找到最近邻
    distances = None  # 你的代码
    indices = None  # 你的代码

    # 应用卡尺约束
    matched_treated = []
    matched_control = []

    # TODO: 遍历处理组个体，应用卡尺约束
    # 你的代码

    return np.array(matched_treated), np.array(matched_control)


def estimate_ate_psm(
    Y: np.ndarray,
    matched_treated_idx: np.ndarray,
    matched_control_idx: np.ndarray
) -> Tuple[float, float]:
    """
    使用 PSM 估计 ATE

    TODO: 计算匹配样本的平均处理效应

    Args:
        Y: 结果变量
        matched_treated_idx: 匹配的处理组索引
        matched_control_idx: 匹配的控制组索引

    Returns:
        (ATE估计, 标准误)
    """
    # TODO: 计算匹配后的 ATE
    # ATE = mean(Y_treated) - mean(Y_control)

    # 你的代码
    pass


# ==================== 练习 1.3: 评估平衡性 ====================

def compute_smd(
    X_treated: np.ndarray,
    X_control: np.ndarray
) -> np.ndarray:
    """
    计算标准化均值差 (Standardized Mean Difference)

    SMD = (mean_treated - mean_control) / pooled_std

    TODO: 实现 SMD 计算

    Args:
        X_treated: 处理组特征
        X_control: 控制组特征

    Returns:
        每个特征的 SMD
    """
    # TODO: 计算均值差
    mean_diff = None  # 你的代码

    # TODO: 计算合并标准差
    # pooled_std = sqrt((var_t + var_c) / 2)
    pooled_std = None  # 你的代码

    # TODO: 计算 SMD
    smd = None  # 你的代码

    return smd


def evaluate_balance(
    X: np.ndarray,
    treatment: np.ndarray,
    matched_treated_idx: Optional[np.ndarray] = None,
    matched_control_idx: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    评估匹配前后的协变量平衡

    TODO: 计算匹配前后的 SMD

    Args:
        X: 特征矩阵
        treatment: 处理状态
        matched_treated_idx: 匹配的处理组索引 (可选)
        matched_control_idx: 匹配的控制组索引 (可选)

    Returns:
        (smd_before, smd_after)
    """
    # 匹配前的 SMD
    treated_mask = treatment == 1
    control_mask = treatment == 0

    smd_before = compute_smd(X[treated_mask], X[control_mask])

    # 匹配后的 SMD
    if matched_treated_idx is not None and matched_control_idx is not None:
        # TODO: 计算匹配后的 SMD
        smd_after = None  # 你的代码
    else:
        smd_after = None

    return smd_before, smd_after


def check_common_support(
    propensity: np.ndarray,
    treatment: np.ndarray
) -> dict:
    """
    检查共同支撑假设 (Common Support)

    处理组和控制组的倾向得分分布应该有重叠

    TODO: 计算重叠统计量

    Args:
        propensity: 倾向得分
        treatment: 处理状态

    Returns:
        dict with overlap statistics
    """
    prop_treated = propensity[treatment == 1]
    prop_control = propensity[treatment == 0]

    # TODO: 计算重叠区间
    overlap_min = None  # max(min(treated), min(control))
    overlap_max = None  # min(max(treated), max(control))

    # TODO: 计算在重叠区间外的样本比例
    outside_overlap = None  # 你的代码

    return {
        'treated_min': prop_treated.min(),
        'treated_max': prop_treated.max(),
        'control_min': prop_control.min(),
        'control_max': prop_control.max(),
        'overlap_min': overlap_min,
        'overlap_max': overlap_max,
        'outside_overlap_pct': outside_overlap * 100 if outside_overlap else 0
    }


# ==================== 练习 1.4: 卡尺匹配 ====================

def compare_caliper_widths(
    propensity: np.ndarray,
    treatment: np.ndarray,
    Y: np.ndarray,
    caliper_widths: List[float] = [0.01, 0.05, 0.1, 0.2, None]
) -> pd.DataFrame:
    """
    比较不同卡尺宽度的效果

    TODO: 对每个卡尺宽度执行 PSM，比较匹配率和 ATE 估计

    Args:
        propensity: 倾向得分
        treatment: 处理状态
        Y: 结果变量
        caliper_widths: 要测试的卡尺宽度列表

    Returns:
        DataFrame with results for each caliper
    """
    results = []

    for caliper in caliper_widths:
        # TODO: 执行 PSM
        matched_t, matched_c = None, None  # 你的代码

        if matched_t is None or len(matched_t) == 0:
            continue

        # TODO: 计算匹配率
        match_rate = None  # len(matched_t) / total_treated

        # TODO: 计算 ATE
        ate, se = None, None  # 你的代码

        results.append({
            'caliper': caliper if caliper else 'None',
            'matched_pairs': len(matched_t),
            'match_rate': match_rate,
            'ate': ate,
            'se': se
        })

    return pd.DataFrame(results)


# ==================== 练习 1.5: ATT vs ATE ====================

def estimate_att_psm(
    Y: np.ndarray,
    matched_treated_idx: np.ndarray,
    matched_control_idx: np.ndarray
) -> float:
    """
    估计 ATT (Average Treatment Effect on the Treated)

    ATT = E[Y(1) - Y(0) | T=1]

    PSM 默认估计的是 ATT，因为我们为处理组个体寻找匹配的控制组

    TODO: 计算 ATT

    Returns:
        ATT估计
    """
    # TODO: 计算 ATT (与 ATE 的计算相同)
    # 你的代码
    pass


# ==================== 练习 1.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 倾向得分的核心思想是什么? 为什么在倾向得分上匹配可以平衡协变量?

你的答案:


2. PSM 估计的是 ATE 还是 ATT? 为什么?

你的答案:


3. 什么是共同支撑假设? 为什么它很重要?

你的答案:


4. |SMD| < 0.1 被认为是良好平衡的阈值。如果匹配后某些协变量的 SMD 仍然很大，应该怎么办?

你的答案:


5. PSM 相比线性回归调整有什么优缺点?

你的答案:


6. 卡尺匹配的权衡是什么? (匹配质量 vs 样本量)

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 1: 倾向得分匹配 (PSM)")
    print("=" * 60)

    # 测试 1.1
    print("\n1.1 生成混淆数据")
    df = generate_confounded_data(n=2000, seed=42)
    if df is not None and 'X1' in df.columns:
        print(f"  样本量: {len(df)}")
        print(f"  处理组: {df['T'].sum()} ({df['T'].mean()*100:.1f}%)")
        print(f"  控制组: {(1-df['T']).sum()} ({(1-df['T']).mean()*100:.1f}%)")
        print(f"  前5行:\n{df.head()}")

        # 朴素估计
        naive_ate = df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()
        print(f"\n  朴素 ATE 估计: {naive_ate:.4f} (真实: 2.0)")
    else:
        print("  [未完成] 请完成 generate_confounded_data 函数")

    # 测试 1.2
    print("\n1.2 估计倾向得分")
    if df is not None and 'X1' in df.columns:
        X = df[['X1', 'X2', 'X3']].values
        T = df['T'].values
        Y = df['Y'].values

        propensity = estimate_propensity_score(X, T)
        if propensity is not None:
            print(f"  倾向得分范围: [{propensity.min():.4f}, {propensity.max():.4f}]")
            print(f"  处理组平均倾向: {propensity[T==1].mean():.4f}")
            print(f"  控制组平均倾向: {propensity[T==0].mean():.4f}")
        else:
            print("  [未完成] 请完成 estimate_propensity_score 函数")

    # 测试 1.3
    print("\n1.3 检查共同支撑")
    if propensity is not None:
        overlap_stats = check_common_support(propensity, T)
        if overlap_stats['overlap_min'] is not None:
            print(f"  处理组范围: [{overlap_stats['treated_min']:.4f}, {overlap_stats['treated_max']:.4f}]")
            print(f"  控制组范围: [{overlap_stats['control_min']:.4f}, {overlap_stats['control_max']:.4f}]")
            print(f"  重叠区间: [{overlap_stats['overlap_min']:.4f}, {overlap_stats['overlap_max']:.4f}]")
            print(f"  区间外样本: {overlap_stats['outside_overlap_pct']:.2f}%")
        else:
            print("  [未完成] 请完成 check_common_support 函数")

    # 测试 1.4
    print("\n1.4 执行 PSM (无卡尺)")
    if propensity is not None:
        matched_t, matched_c = propensity_score_matching(propensity, T, caliper=None)
        if matched_t is not None and len(matched_t) > 0:
            print(f"  匹配对数: {len(matched_t)}")
            print(f"  匹配率: {len(matched_t) / T.sum() * 100:.1f}%")

            # 估计 ATE
            psm_ate, psm_se = estimate_ate_psm(Y, matched_t, matched_c)
            if psm_ate is not None:
                print(f"  PSM ATE: {psm_ate:.4f} ± {psm_se:.4f}")
                print(f"  偏差: {psm_ate - 2.0:.4f}")
            else:
                print("  [未完成] 请完成 estimate_ate_psm 函数")
        else:
            print("  [未完成] 请完成 propensity_score_matching 函数")

    # 测试 1.5
    print("\n1.5 评估平衡性")
    if matched_t is not None and len(matched_t) > 0:
        smd_before, smd_after = evaluate_balance(X, T, matched_t, matched_c)
        if smd_before is not None:
            print(f"  匹配前 SMD: {np.abs(smd_before).mean():.4f} (平均)")
            print(f"  匹配前 SMD 详细: {smd_before}")
            if smd_after is not None:
                print(f"  匹配后 SMD: {np.abs(smd_after).mean():.4f} (平均)")
                print(f"  匹配后 SMD 详细: {smd_after}")
                print(f"  平衡改善: {(np.abs(smd_before).mean() - np.abs(smd_after).mean()):.4f}")
            else:
                print("  [未完成] 请完成匹配后 SMD 计算")
        else:
            print("  [未完成] 请完成 compute_smd 函数")

    # 测试 1.6
    print("\n1.6 比较不同卡尺宽度")
    if propensity is not None:
        caliper_results = compare_caliper_widths(propensity, T, Y)
        if caliper_results is not None and not caliper_results.empty:
            print(caliper_results.to_string(index=False))
        else:
            print("  [未完成] 请完成 compare_caliper_widths 函数")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("提示: 良好的匹配应使 |SMD| < 0.1")
    print("=" * 60)
