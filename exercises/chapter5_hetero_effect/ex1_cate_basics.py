"""
练习 1: CATE 基础

学习目标:
1. 理解条件平均处理效应 (CATE) 的概念
2. 识别和分析子群体效应差异
3. 可视化效应异质性
4. 理解 CATE 与 ATE 的关系

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestRegressor


# ==================== 练习 1.1: 理解 CATE ====================

def generate_heterogeneous_treatment_data(
    n: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成具有异质性处理效应的数据

    数据生成过程:
    - X1, X2 ~ N(0, 1)
    - Age ~ Uniform(20, 60)
    - T ~ Bernoulli(0.5)  # 随机分配
    - Y(0) = 10 + 2*X1 + 1*X2 + 0.1*Age + noise
    - Y(1) = Y(0) + tau(X1, Age)
    - tau(X1, Age) = 5 + 3*X1 - 0.05*Age  # 异质性效应

    TODO: 完成数据生成代码

    Returns:
        DataFrame with columns: X1, X2, Age, T, Y, tau_true
    """
    np.random.seed(seed)

    # TODO: 生成协变量
    X1 = None  # 你的代码: 标准正态分布
    X2 = None  # 你的代码: 标准正态分布
    Age = None  # 你的代码: [20, 60] 均匀分布

    # TODO: 随机分配处理 (50% 概率)
    T = None  # 你的代码

    # TODO: 计算异质性处理效应 tau
    # tau(X1, Age) = 5 + 3*X1 - 0.05*Age
    tau_true = None  # 你的代码

    # TODO: 生成潜在结果
    noise = np.random.randn(n) * 0.5
    # Y(0) = 10 + 2*X1 + 1*X2 + 0.1*Age + noise
    Y0 = None  # 你的代码
    # Y(1) = Y(0) + tau
    Y1 = None  # 你的代码

    # TODO: 计算观测结果 Y
    Y = None  # 你的代码

    return pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Age': Age,
        'T': T,
        'Y': Y,
        'tau_true': tau_true
    })


# ==================== 练习 1.2: 估计 CATE ====================

class SimpleTLearner:
    """
    T-Learner: 分别为处理组和对照组训练模型

    CATE(x) = E[Y|T=1,X=x] - E[Y|T=0,X=x]
    """

    def __init__(self):
        self.model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_1 = RandomForestRegressor(n_estimators=100, random_state=43)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练 T-Learner

        TODO: 完成训练逻辑
        """
        # TODO: 分离处理组和对照组
        mask_0 = None  # 你的代码: T == 0
        mask_1 = None  # 你的代码: T == 1

        # TODO: 分别训练两个模型
        # model_0 在对照组上训练
        # model_1 在处理组上训练
        # 你的代码

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        预测 CATE

        TODO: 完成 CATE 预测
        """
        # TODO: 预测 Y(0) 和 Y(1)
        Y0_pred = None  # 你的代码
        Y1_pred = None  # 你的代码

        # TODO: 计算 CATE = Y(1) - Y(0)
        cate_pred = None  # 你的代码

        return cate_pred


def compute_ate(tau: np.ndarray) -> float:
    """
    计算平均处理效应 (ATE)

    ATE = E[tau(X)] = E[Y(1) - Y(0)]

    TODO: 计算 ATE
    """
    # 你的代码
    pass


def compute_pehe(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    """
    计算 PEHE (Precision in Estimation of HTE)

    PEHE = sqrt(E[(tau_true - tau_pred)^2])

    这是衡量 CATE 估计精度的黄金标准

    TODO: 计算 PEHE
    """
    # 你的代码
    pass


# ==================== 练习 1.3: 子群体分析 ====================

def identify_subgroups_by_cate(
    tau: np.ndarray,
    n_groups: int = 4
) -> np.ndarray:
    """
    根据 CATE 大小将样本分为若干子群体

    TODO: 使用分位数将样本分为 n_groups 组

    Args:
        tau: CATE 值
        n_groups: 分组数量

    Returns:
        group_labels: 每个样本的组别 (0 to n_groups-1)
    """
    # TODO: 使用 np.quantile 和分位数分组
    # 提示: Q1 = 最低 25%, Q2 = 25-50%, Q3 = 50-75%, Q4 = 最高 25%

    # 你的代码
    pass


def analyze_subgroup_effects(
    df: pd.DataFrame,
    group_labels: np.ndarray,
    tau_pred: np.ndarray
) -> pd.DataFrame:
    """
    分析每个子群体的处理效应

    TODO: 计算每个子群体的:
    - 样本量
    - 真实平均 CATE
    - 预测平均 CATE
    - 观测的处理效应 (只用观测数据估计)

    Returns:
        汇总 DataFrame
    """
    results = []

    n_groups = len(np.unique(group_labels))

    for g in range(n_groups):
        mask = group_labels == g

        # TODO: 计算该组的统计量
        group_size = None  # 你的代码
        true_cate = None  # 你的代码: df.loc[mask, 'tau_true'].mean()
        pred_cate = None  # 你的代码: tau_pred[mask].mean()

        # TODO: 计算观测的处理效应
        # 在该子群体中，比较处理组和对照组的平均 Y
        df_group = df[mask]
        obs_effect = None  # 你的代码

        results.append({
            'Group': f'Q{g+1}',
            'Size': group_size,
            'True CATE': true_cate,
            'Pred CATE': pred_cate,
            'Obs Effect': obs_effect
        })

    return pd.DataFrame(results)


# ==================== 练习 1.4: 可视化异质性 ====================

def visualize_cate_distribution(
    tau_true: np.ndarray,
    tau_pred: np.ndarray,
    feature: np.ndarray,
    feature_name: str = 'X1'
) -> None:
    """
    可视化 CATE 的分布和与特征的关系

    TODO: 创建两个子图:
    1. CATE 的直方图 (真实 vs 预测)
    2. CATE vs 特征的散点图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # TODO: 子图 1 - CATE 分布
    # 绘制 tau_true 和 tau_pred 的直方图
    # 你的代码

    axes[0].set_xlabel('CATE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('CATE Distribution')
    axes[0].legend()

    # TODO: 子图 2 - CATE vs 特征
    # 绘制 feature vs tau_true 和 feature vs tau_pred 的散点图
    # 你的代码

    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel('CATE')
    axes[1].set_title(f'CATE vs {feature_name}')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ==================== 练习 1.5: CATE 用于决策 ====================

def optimal_treatment_policy(
    tau_pred: np.ndarray,
    cost: float = 1.0,
    threshold: float = None
) -> Tuple[np.ndarray, Dict]:
    """
    基于 CATE 的最优处理策略

    策略: 只对预测 CATE > threshold 的样本进行处理

    TODO: 实现最优策略

    Args:
        tau_pred: 预测的 CATE
        cost: 处理成本
        threshold: CATE 阈值 (如果为 None，使用 cost 作为阈值)

    Returns:
        (treatment_decision, policy_stats)
    """
    if threshold is None:
        threshold = cost

    # TODO: 决策规则
    # 如果 tau_pred > threshold，则处理 (=1)，否则不处理 (=0)
    treatment_decision = None  # 你的代码

    # TODO: 计算策略统计
    policy_stats = {
        'threshold': threshold,
        'treatment_rate': None,  # 你的代码: 处理比例
        'avg_expected_benefit': None,  # 你的代码: 被处理样本的平均 CATE
        'total_expected_value': None  # 你的代码: 总价值 - 总成本
    }

    return treatment_decision, policy_stats


# ==================== 练习 1.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. CATE 和 ATE 的关系是什么? ATE 能否反映异质性?

你的答案:


2. 在什么情况下，所有个体的 CATE 都相同 (即没有异质性)?

你的答案:


3. 为什么 PEHE 是评估 CATE 估计的好指标? 它与评估 ATE 的指标有何不同?

你的答案:


4. 在实践中，如何验证子群体分析的结果是可靠的 (而非过拟合)?

你的答案:


5. 基于 CATE 的处理策略有哪些实际应用场景?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 1: CATE 基础")
    print("=" * 60)

    # 测试 1.1
    print("\n1.1 生成异质性数据")
    df = generate_heterogeneous_treatment_data()
    if df is not None and df['X1'].iloc[0] is not None:
        print(f"  样本量: {len(df)}")
        print(f"  列: {list(df.columns)}")
        print(f"\n  前 5 行:")
        print(df.head())
        print(f"\n  真实 ATE: {df['tau_true'].mean():.4f}")
        print(f"  CATE 范围: [{df['tau_true'].min():.2f}, {df['tau_true'].max():.2f}]")
    else:
        print("  [未完成] 请完成 generate_heterogeneous_treatment_data 函数")

    # 测试 1.2
    print("\n1.2 估计 CATE (T-Learner)")
    if df is not None and df['X1'].iloc[0] is not None:
        X = df[['X1', 'X2', 'Age']].values
        T = df['T'].values
        Y = df['Y'].values
        tau_true = df['tau_true'].values

        try:
            learner = SimpleTLearner()
            learner.fit(X, T, Y)
            tau_pred = learner.predict_cate(X)

            if tau_pred is not None and not np.all(tau_pred == None):
                ate_true = compute_ate(tau_true)
                ate_pred = compute_ate(tau_pred)
                pehe = compute_pehe(tau_true, tau_pred)

                if ate_true is not None and pehe is not None:
                    print(f"  真实 ATE: {ate_true:.4f}")
                    print(f"  预测 ATE: {ate_pred:.4f}")
                    print(f"  PEHE: {pehe:.4f}")
                else:
                    print("  [未完成] 请完成 compute_ate 和 compute_pehe 函数")
            else:
                print("  [未完成] 请完成 SimpleTLearner.predict_cate 方法")
        except Exception as e:
            print(f"  [未完成] SimpleTLearner 实现有误: {e}")

    # 测试 1.3
    print("\n1.3 子群体分析")
    if df is not None and df['X1'].iloc[0] is not None:
        try:
            if 'tau_pred' in locals() and tau_pred is not None:
                groups = identify_subgroups_by_cate(tau_true, n_groups=4)

                if groups is not None and not np.all(groups == None):
                    subgroup_df = analyze_subgroup_effects(df, groups, tau_pred)

                    if subgroup_df is not None:
                        print(subgroup_df.to_string(index=False))
                    else:
                        print("  [未完成] 请完成 analyze_subgroup_effects 函数")
                else:
                    print("  [未完成] 请完成 identify_subgroups_by_cate 函数")
        except Exception as e:
            print(f"  [错误] {e}")

    # 测试 1.4
    print("\n1.4 可视化 CATE")
    if df is not None and df['X1'].iloc[0] is not None:
        try:
            if 'tau_pred' in locals() and tau_pred is not None:
                print("  生成可视化...")
                # visualize_cate_distribution(tau_true, tau_pred, df['X1'].values, 'X1')
                print("  [提示] 取消注释上一行代码查看可视化")
        except Exception as e:
            print(f"  [错误] {e}")

    # 测试 1.5
    print("\n1.5 最优处理策略")
    if df is not None and df['X1'].iloc[0] is not None:
        try:
            if 'tau_pred' in locals() and tau_pred is not None:
                treatment, stats = optimal_treatment_policy(tau_pred, cost=2.0)

                if treatment is not None and stats['treatment_rate'] is not None:
                    print(f"  成本阈值: {stats['threshold']:.2f}")
                    print(f"  处理比例: {stats['treatment_rate']:.2%}")
                    print(f"  平均预期收益: {stats['avg_expected_benefit']:.4f}")
                    print(f"  总预期价值: {stats['total_expected_value']:.4f}")

                    # 比较随机策略
                    random_value = tau_true.mean() - 2.0  # 50% 处理
                    print(f"\n  对比:")
                    print(f"  随机策略 (50% 处理) 价值: {random_value:.4f}")
                    print(f"  最优策略提升: {stats['total_expected_value'] - random_value:.4f}")
                else:
                    print("  [未完成] 请完成 optimal_treatment_policy 函数")
        except Exception as e:
            print(f"  [错误] {e}")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 60)
