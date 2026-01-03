"""
练习 3: Uplift 模型评估

学习目标:
1. 理解 Qini 曲线的原理和计算方法
2. 掌握 AUUC (Area Under Uplift Curve) 指标
3. 学会使用累积增益分析
4. 理解最优干预比例的确定方法

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List


# ==================== 练习 3.1: Qini 曲线计算 ====================

def calculate_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Qini 曲线

    Qini 曲线评估模型对 uplift 的排序能力:
    - 将样本按预测 uplift 从高到低排序
    - 累积计算前 k 个样本的增益

    Qini(k) = Y_t(k) - Y_c(k) * (n_t(k) / n_c(k))

    其中:
    - Y_t(k): 前 k 个样本中处理组的总转化数
    - Y_c(k): 前 k 个样本中控制组的总转化数
    - n_t(k): 前 k 个样本中处理组的数量
    - n_c(k): 前 k 个样本中控制组的数量

    TODO: 实现 Qini 曲线计算

    Args:
        y_true: 真实结果 (0/1)
        treatment: 处理状态 (0/1)
        uplift_score: 预测的 uplift 得分

    Returns:
        (fraction_targeted, qini_values)
        fraction_targeted: 干预比例 [0, 1]
        qini_values: 对应的 Qini 值
    """
    # TODO: 按 uplift 得分从高到低排序
    order = None  # 你的代码: np.argsort(uplift_score)[::-1]

    y_sorted = None  # y_true[order]
    t_sorted = None  # treatment[order]

    n = len(y_true)

    # TODO: 计算累积统计量
    # 提示: 使用 np.cumsum
    cum_t_outcomes = None  # 累积处理组转化数
    cum_c_outcomes = None  # 累积控制组转化数
    cum_t = None           # 累积处理组样本数
    cum_c = None           # 累积控制组样本数

    # TODO: 计算 Qini 值
    # Qini = cum_t_outcomes - cum_c_outcomes * (cum_t / cum_c)
    # 注意避免除零
    qini = None  # 你的代码

    # TODO: 计算干预比例
    fraction = None  # 你的代码: np.arange(1, n+1) / n

    # 添加原点 (0, 0)
    fraction = np.insert(fraction, 0, 0)
    qini = np.insert(qini, 0, 0)

    return fraction, qini


# ==================== 练习 3.2: AUUC 计算 ====================

def calculate_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray
) -> float:
    """
    计算 AUUC (Area Under Uplift Curve)

    AUUC 是 Qini 曲线下的面积，衡量模型整体性能

    TODO: 实现 AUUC 计算
    提示: 可以使用 np.trapz 进行梯形积分

    Returns:
        AUUC value
    """
    # TODO: 先计算 Qini 曲线
    fraction, qini = calculate_qini_curve(y_true, treatment, uplift_score)

    # TODO: 计算曲线下面积
    auuc = None  # 你的代码: np.trapz(qini, fraction)

    return auuc


# ==================== 练习 3.3: Uplift by Decile ====================

def calculate_uplift_by_decile(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    按预测 uplift 分组，计算每组的实际 uplift

    这是一个重要的诊断工具:
    - 如果模型有效，高分组应该有更高的 uplift
    - 可以识别负 uplift 人群

    TODO: 实现分组统计

    Args:
        y_true: 真实结果
        treatment: 处理状态
        uplift_score: 预测得分
        n_deciles: 分组数量

    Returns:
        DataFrame with columns: decile, n_samples, uplift, treated_rate, control_rate
    """
    # TODO: 按 uplift_score 分成 n_deciles 组
    # 提示: 使用 pd.qcut
    deciles = None  # 你的代码

    results = []

    # TODO: 对每组计算统计量
    for d in range(n_deciles):
        mask = deciles == d

        if mask.sum() == 0:
            continue

        y_sub = y_true[mask]
        t_sub = treatment[mask]

        # TODO: 计算该组的统计量
        n_samples = mask.sum()

        # 处理组和控制组转化率
        treated_mask = t_sub == 1
        control_mask = t_sub == 0

        if treated_mask.sum() > 0 and control_mask.sum() > 0:
            treated_rate = None  # 你的代码
            control_rate = None  # 你的代码
            uplift = None        # 你的代码
        else:
            treated_rate = 0.0
            control_rate = 0.0
            uplift = 0.0

        results.append({
            'decile': d + 1,  # 1-indexed
            'n_samples': n_samples,
            'uplift': uplift,
            'treated_rate': treated_rate,
            'control_rate': control_rate
        })

    return pd.DataFrame(results)


# ==================== 练习 3.4: 累积增益分析 ====================

def calculate_cumulative_gain(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    true_uplift: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算累积增益曲线

    累积增益 = 按预测排序后，前 k% 人群的总 uplift

    TODO: 实现累积增益计算

    Args:
        y_true: 真实结果
        treatment: 处理状态
        uplift_score: 预测得分
        true_uplift: 真实 uplift (如果已知)

    Returns:
        (fraction, observed_gain, perfect_gain)
        fraction: 干预比例
        observed_gain: 模型的累积增益
        perfect_gain: 完美模型的累积增益 (如果 true_uplift 已知)
    """
    n = len(y_true)

    # TODO: 按预测 uplift 排序
    order = None  # 你的代码

    y_sorted = y_true[order]
    t_sorted = treatment[order]

    # TODO: 计算每个样本的观测 uplift
    # 这里用处理组和控制组的平均转化率差作为估计
    observed_uplift = []
    for i in range(1, n + 1):
        y_sub = y_sorted[:i]
        t_sub = t_sorted[:i]

        if (t_sub == 1).sum() > 0 and (t_sub == 0).sum() > 0:
            uplift = None  # 你的代码: 计算该段的 uplift
        else:
            uplift = 0

        observed_uplift.append(uplift * i)  # 累积增益

    fraction = np.arange(1, n + 1) / n
    observed_gain = np.array(observed_uplift)

    # TODO: 如果知道真实 uplift，计算完美排序的增益
    perfect_gain = None
    if true_uplift is not None:
        # 按真实 uplift 排序
        perfect_order = None  # 你的代码
        perfect_gain = np.cumsum(true_uplift[perfect_order])

    return fraction, observed_gain, perfect_gain


# ==================== 练习 3.5: 最优干预比例 ====================

def find_optimal_targeting_fraction(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    cost_per_treatment: float = 1.0,
    revenue_per_conversion: float = 10.0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    找到最优干预比例

    考虑成本和收益:
    - 每次干预成本: cost_per_treatment
    - 每次转化收益: revenue_per_conversion

    目标: 最大化 ROI = (revenue - cost) / cost

    TODO: 实现最优比例搜索

    Returns:
        (optimal_fraction, optimal_roi, all_fractions, all_rois)
    """
    n = len(y_true)

    # TODO: 按 uplift 排序
    order = None  # 你的代码

    y_sorted = y_true[order]
    t_sorted = treatment[order]

    # 测试不同的干预比例
    fractions = np.linspace(0.05, 1.0, 20)
    rois = []

    for frac in fractions:
        # TODO: 计算前 frac 比例人群的 ROI
        n_target = int(frac * n)

        if n_target == 0:
            rois.append(0)
            continue

        # 前 n_target 个样本
        y_top = y_sorted[:n_target]
        t_top = t_sorted[:n_target]

        # TODO: 估计 uplift
        if (t_top == 1).sum() > 0 and (t_top == 0).sum() > 0:
            uplift = None  # 你的代码
        else:
            uplift = 0

        # TODO: 计算收益和成本
        # 增量转化 = uplift * 目标人数
        incremental_conversions = None  # 你的代码
        revenue = None  # 你的代码
        cost = None     # 你的代码

        # ROI
        roi = (revenue - cost) / cost if cost > 0 else 0
        rois.append(roi)

    rois = np.array(rois)

    # TODO: 找最优点
    optimal_idx = None  # 你的代码
    optimal_fraction = fractions[optimal_idx]
    optimal_roi = rois[optimal_idx]

    return optimal_fraction, optimal_roi, fractions, rois


# ==================== 练习 3.6: 模型对比 ====================

def compare_uplift_models(
    y_true: np.ndarray,
    treatment: np.ndarray,
    model_predictions: dict  # {'model_name': uplift_scores}
) -> pd.DataFrame:
    """
    对比多个 Uplift 模型的性能

    TODO: 实现多模型对比

    Args:
        y_true: 真实结果
        treatment: 处理状态
        model_predictions: 字典，key 为模型名，value 为预测 uplift

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    for model_name, scores in model_predictions.items():
        # TODO: 计算各项指标
        auuc = None  # calculate_auuc(...)

        # Top 10% uplift
        n = len(y_true)
        order = np.argsort(scores)[::-1]
        top_10_idx = order[:int(0.1 * n)]

        y_top = y_true[top_10_idx]
        t_top = treatment[top_10_idx]

        if (t_top == 1).sum() > 0 and (t_top == 0).sum() > 0:
            top_10_uplift = None  # 你的代码
        else:
            top_10_uplift = 0

        results.append({
            'Model': model_name,
            'AUUC': auuc,
            'Top 10% Uplift': top_10_uplift
        })

    return pd.DataFrame(results)


# ==================== 练习 3.7: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. Qini 曲线和 ROC 曲线有什么相似和不同之处?

你的答案:
相似:
-

不同:
-


2. 为什么 Qini 曲线需要调整控制组人数 (n_t / n_c 的权重)?

你的答案:


3. AUUC 的值越大越好吗? AUUC 的取值范围是什么?

你的答案:


4. 如果某个模型的 Qini 曲线在某些区域低于随机基线，这意味着什么?

你的答案:


5. Top 10% Uplift 指标有什么实际意义? 什么时候应该关注这个指标?

你的答案:


6. 在确定最优干预比例时，除了 ROI，还应该考虑哪些因素?

你的答案:
-
-
-


7. 如果没有真实的 uplift (只有观测数据 Y, T)，如何评估 Uplift 模型?

你的答案:


"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 3: Uplift 模型评估")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 2000

    # 特征
    X = np.random.randn(n)

    # 处理分配
    T = np.random.binomial(1, 0.5, n)

    # 真实 uplift (异质性)
    true_uplift = 0.1 + 0.15 * (X > 0)

    # 基线转化率
    baseline_prob = 0.2

    # 生成结果
    prob = baseline_prob + true_uplift * T
    Y = np.random.binomial(1, prob)

    # 模拟两个模型的预测
    # Model 1: 好模型 (与真实 uplift 相关)
    model1_pred = true_uplift + np.random.randn(n) * 0.05

    # Model 2: 差模型 (随机)
    model2_pred = np.random.randn(n) * 0.1

    # 测试 3.1: Qini 曲线
    print("\n3.1 Qini 曲线计算")
    fraction, qini = calculate_qini_curve(Y, T, model1_pred)
    if fraction is not None and qini is not None:
        print(f"  计算完成: {len(fraction)} 个点")
        print(f"  最大 Qini 值: {qini.max():.4f}")
    else:
        print("  [未完成] 请完成 calculate_qini_curve 函数")

    # 测试 3.2: AUUC
    print("\n3.2 AUUC 计算")
    auuc1 = calculate_auuc(Y, T, model1_pred)
    auuc2 = calculate_auuc(Y, T, model2_pred)
    if auuc1 is not None:
        print(f"  Good Model AUUC: {auuc1:.4f}")
        print(f"  Poor Model AUUC: {auuc2:.4f}")
    else:
        print("  [未完成] 请完成 calculate_auuc 函数")

    # 测试 3.3: Uplift by Decile
    print("\n3.3 Uplift by Decile")
    decile_df = calculate_uplift_by_decile(Y, T, model1_pred)
    if decile_df is not None and len(decile_df) > 0:
        print(decile_df.to_string(index=False))
    else:
        print("  [未完成] 请完成 calculate_uplift_by_decile 函数")

    # 测试 3.4: 累积增益
    print("\n3.4 累积增益分析")
    fraction, obs_gain, perf_gain = calculate_cumulative_gain(
        Y, T, model1_pred, true_uplift
    )
    if obs_gain is not None:
        print(f"  计算完成")
        print(f"  观测增益 (50%): {obs_gain[int(0.5*len(obs_gain))]:.2f}")
        if perf_gain is not None:
            print(f"  完美增益 (50%): {perf_gain[int(0.5*len(perf_gain))]:.2f}")
    else:
        print("  [未完成] 请完成 calculate_cumulative_gain 函数")

    # 测试 3.5: 最优干预比例
    print("\n3.5 最优干预比例")
    opt_frac, opt_roi, fracs, rois = find_optimal_targeting_fraction(
        Y, T, model1_pred,
        cost_per_treatment=1.0,
        revenue_per_conversion=10.0
    )
    if opt_frac is not None:
        print(f"  最优干预比例: {opt_frac * 100:.1f}%")
        print(f"  最优 ROI: {opt_roi:.2f}")
    else:
        print("  [未完成] 请完成 find_optimal_targeting_fraction 函数")

    # 测试 3.6: 模型对比
    print("\n3.6 模型对比")
    comparison = compare_uplift_models(
        Y, T,
        {
            'Good Model': model1_pred,
            'Poor Model': model2_pred,
            'Random': np.random.randn(n)
        }
    )
    if comparison is not None and len(comparison) > 0:
        print(comparison.to_string(index=False))
    else:
        print("  [未完成] 请完成 compare_uplift_models 函数")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("思考题请在代码注释中回答")
    print("=" * 60)
