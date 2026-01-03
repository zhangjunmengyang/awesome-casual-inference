"""
练习 1: 智能发券优化

学习目标:
1. 理解营销场景中的用户分群策略
2. 掌握 Uplift 建模在发券场景的应用
3. 学习 ROI 优化决策方法
4. 理解 Persuadables、Sure Things、Lost Causes、Sleeping Dogs 概念

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import GradientBoostingClassifier


# ==================== 练习 1.1: 营销数据生成 ====================

def generate_simple_marketing_data(
    n_samples: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成简化的营销场景数据

    场景: 外卖平台发券实验
    - 特征: 用户年龄、历史订单数、最近活跃度
    - 处理: 是否发券 (T=1 发券, T=0 不发券)
    - 结果: 是否下单 (conversion=1 下单, conversion=0 未下单)

    用户类型 (真实但不可观测):
    - Persuadables: 发券会下单，不发券不下单 (高 Uplift)
    - Sure Things: 发不发券都会下单 (低 Uplift，浪费)
    - Lost Causes: 发不发券都不下单 (低 Uplift)
    - Sleeping Dogs: 发券反而不下单 (负 Uplift)

    TODO: 完成数据生成逻辑

    Returns:
        DataFrame with columns: age, order_freq, days_since_last, T, conversion, user_type
    """
    np.random.seed(seed)

    # TODO: 生成用户特征
    # age: 18-65 岁均匀分布
    # order_freq: 历史订单数，泊松分布 lambda=5
    # days_since_last: 距上次下单天数，指数分布 scale=10
    age = None  # 你的代码
    order_freq = None  # 你的代码
    days_since_last = None  # 你的代码

    # TODO: 随机分配处理 (50% 概率发券)
    T = None  # 你的代码

    # TODO: 确定用户类型 (简化版)
    # Persuadables: 年轻 (age < 30) + 低频 (order_freq < 5)
    # Sure Things: 高频 (order_freq >= 8)
    # Sleeping Dogs: 非常低频 (order_freq < 2) + 不年轻 (age >= 40)
    # Lost Causes: 其他

    user_type = []
    for i in range(n_samples):
        # 你的代码: 根据规则判断用户类型
        pass

    # TODO: 生成转化结果
    # 基线转化率: 0.1 + 0.02 * min(order_freq, 10)
    # Persuadables: 发券 +0.2, 不发券 0
    # Sure Things: 发券 +0.05, 不发券很高
    # Sleeping Dogs: 发券 -0.1
    # Lost Causes: 发券 +0.01

    conversion = []
    for i in range(n_samples):
        # 你的代码: 计算转化概率并生成结果
        pass

    return pd.DataFrame({
        'age': age,
        'order_freq': order_freq,
        'days_since_last': days_since_last,
        'T': T,
        'conversion': conversion,
        'user_type': user_type
    })


# ==================== 练习 1.2: Uplift 模型训练 ====================

class SimpleUpliftModel:
    """
    简单的 Uplift 模型 (Two-Model / T-Learner 方法)

    思想: 分别训练控制组和处理组的模型，差值即为 Uplift
    """

    def __init__(self):
        self.model_control = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self.model_treatment = GradientBoostingClassifier(n_estimators=50, random_state=43)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练 Uplift 模型

        TODO: 实现训练逻辑

        Args:
            X: 特征矩阵 (n_samples, n_features)
            T: 处理指示 (n_samples,)
            Y: 结果变量 (n_samples,)
        """
        # TODO: 分离控制组和处理组数据
        mask_control = None  # 你的代码
        mask_treatment = None  # 你的代码

        # TODO: 训练两个模型
        # self.model_control.fit(...)
        # self.model_treatment.fit(...)

        # 你的代码
        pass

        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """
        预测 Uplift

        TODO: 实现预测逻辑

        Returns:
            uplift: 预测的处理效应 (n_samples,)
        """
        # TODO: 分别预测两组的转化概率
        prob_treatment = None  # 你的代码: self.model_treatment.predict_proba(X)[:, 1]
        prob_control = None  # 你的代码: self.model_control.predict_proba(X)[:, 1]

        # TODO: Uplift = 处理组概率 - 控制组概率
        uplift = None  # 你的代码

        return uplift


# ==================== 练习 1.3: 用户分群 ====================

def segment_users(
    df: pd.DataFrame,
    uplift_scores: np.ndarray,
    high_threshold: float = 0.1,
    low_threshold: float = 0.02
) -> pd.DataFrame:
    """
    根据 Uplift 分数将用户分群

    分群规则:
    - Persuadables: uplift >= high_threshold
    - Sure Things/Lost Causes: 0 < uplift < low_threshold (需结合基线转化率区分)
    - Sleeping Dogs: uplift < 0

    TODO: 实现分群逻辑

    Args:
        df: 原始数据
        uplift_scores: Uplift 预测分数
        high_threshold: 高 Uplift 阈值
        low_threshold: 低 Uplift 阈值

    Returns:
        带 segment 列的 DataFrame
    """
    df = df.copy()
    df['uplift_score'] = uplift_scores

    # TODO: 实现分群规则
    # 提示: 使用 np.where 或条件判断

    segments = []
    for i in range(len(df)):
        uplift = uplift_scores[i]
        # 你的代码: 根据 uplift 判断分群
        pass

    df['segment'] = segments
    return df


# ==================== 练习 1.4: ROI 计算 ====================

def calculate_roi_simple(
    df: pd.DataFrame,
    treatment_mask: np.ndarray,
    revenue_per_conversion: float = 100,
    cost_per_coupon: float = 15
) -> Dict[str, float]:
    """
    计算发券策略的 ROI

    ROI = (收入 - 成本) / 成本

    收入 = 增量转化数 * 每次转化收入
    成本 = 发券数 * 每张券成本

    TODO: 实现 ROI 计算

    Args:
        df: 数据 (必须包含 T, conversion 列)
        treatment_mask: 哪些用户应该发券 (1=发券, 0=不发券)
        revenue_per_conversion: 每次转化的收入
        cost_per_coupon: 每张券的成本

    Returns:
        包含 roi, revenue, cost 等指标的字典
    """
    # TODO: 计算实际效果
    # 在 treatment_mask=1 的用户中，计算实际的 uplift

    # 提示:
    # 1. 筛选出 treatment_mask=1 的用户
    # 2. 在这些用户中，计算 T=1 和 T=0 的转化率差异
    # 3. 这个差异就是实际的 uplift

    # 你的代码
    actual_uplift = 0  # 实际转化率增量

    # TODO: 计算收入和成本
    n_coupons = None  # 发券数量
    expected_conversions = None  # 预期增量转化数
    revenue = None  # 收入
    cost = None  # 成本
    roi = None  # ROI

    return {
        'roi': roi,
        'revenue': revenue,
        'cost': cost,
        'n_coupons': n_coupons,
        'expected_conversions': expected_conversions,
        'actual_uplift': actual_uplift
    }


# ==================== 练习 1.5: 策略对比 ====================

def compare_strategies(
    df: pd.DataFrame,
    uplift_scores: np.ndarray,
    budget_fraction: float = 0.3
) -> pd.DataFrame:
    """
    对比不同发券策略的效果

    策略:
    1. Random: 随机发券
    2. High Frequency: 优先给高频用户发券 (传统方法)
    3. Uplift Model: 根据 Uplift 分数发券 (因果推断方法)

    TODO: 实现策略对比

    Args:
        df: 数据
        uplift_scores: Uplift 预测分数
        budget_fraction: 预算比例 (发券给多少比例的用户)

    Returns:
        对比结果 DataFrame
    """
    n_target = int(budget_fraction * len(df))
    results = []

    # TODO: 策略 1 - Random
    # 随机选择 n_target 个用户
    random_mask = None  # 你的代码
    random_roi = calculate_roi_simple(df, random_mask)
    results.append({
        'strategy': 'Random',
        **random_roi
    })

    # TODO: 策略 2 - High Frequency
    # 选择 order_freq 最高的 n_target 个用户
    high_freq_mask = None  # 你的代码
    high_freq_roi = calculate_roi_simple(df, high_freq_mask)
    results.append({
        'strategy': 'High Frequency',
        **high_freq_roi
    })

    # TODO: 策略 3 - Uplift Model
    # 选择 uplift_scores 最高的 n_target 个用户
    uplift_mask = None  # 你的代码
    uplift_roi = calculate_roi_simple(df, uplift_mask)
    results.append({
        'strategy': 'Uplift Model',
        **uplift_roi
    })

    return pd.DataFrame(results)


# ==================== 练习 1.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 为什么 "Sure Things" 用户会造成补贴浪费?

你的答案:


2. 为什么 "Sleeping Dogs" 用户发券反而转化率下降? 举一个实际例子。

你的答案:


3. 在真实业务中，如何验证 Uplift 模型的准确性?

你的答案:


4. 如果预算有限 (比如只能发券给 20% 的用户)，应该如何调整发券策略?

你的答案:


5. Uplift 建模相比传统的 "响应率建模" 有什么本质区别?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("练习 1: 智能发券优化")
    print("=" * 50)

    # 测试 1.1
    print("\n1.1 生成营销数据")
    df = generate_simple_marketing_data()
    if df is not None and len(df) > 0 and df['age'].iloc[0] is not None:
        print(f"  样本量: {len(df)}")
        print(f"  发券率: {df['T'].mean():.2%}")
        print(f"  整体转化率: {df['conversion'].mean():.2%}")
        print(f"\n  用户类型分布:")
        print(df['user_type'].value_counts())
    else:
        print("  [未完成] 请完成 generate_simple_marketing_data 函数")

    # 测试 1.2
    print("\n1.2 训练 Uplift 模型")
    if df is not None and len(df) > 0 and df['age'].iloc[0] is not None:
        X = df[['age', 'order_freq', 'days_since_last']].values
        T = df['T'].values
        Y = df['conversion'].values

        model = SimpleUpliftModel()
        try:
            model.fit(X, T, Y)
            uplift_pred = model.predict_uplift(X)
            if uplift_pred is not None and len(uplift_pred) > 0:
                print(f"  Uplift 预测范围: [{uplift_pred.min():.3f}, {uplift_pred.max():.3f}]")
                print(f"  平均 Uplift: {uplift_pred.mean():.3f}")
            else:
                print("  [未完成] 请完成 predict_uplift 方法")
        except:
            print("  [未完成] 请完成 fit 方法")

    # 测试 1.3
    print("\n1.3 用户分群")
    if df is not None and 'uplift_pred' in locals() and uplift_pred is not None:
        df_segmented = segment_users(df, uplift_pred)
        if 'segment' in df_segmented.columns:
            print(f"\n  分群分布:")
            print(df_segmented['segment'].value_counts())
            print(f"\n  各群组平均 Uplift:")
            print(df_segmented.groupby('segment')['uplift_score'].mean().sort_values(ascending=False))
        else:
            print("  [未完成] 请完成 segment_users 函数")

    # 测试 1.4
    print("\n1.4 ROI 计算")
    if df is not None and 'uplift_pred' in locals():
        # 测试: 给 Uplift > 0.05 的用户发券
        test_mask = (uplift_pred > 0.05).astype(int)
        roi_result = calculate_roi_simple(df, test_mask)
        if roi_result['roi'] is not None:
            print(f"  发券人数: {roi_result['n_coupons']}")
            print(f"  成本: ¥{roi_result['cost']:.0f}")
            print(f"  收入: ¥{roi_result['revenue']:.0f}")
            print(f"  ROI: {roi_result['roi']:.2f}")
        else:
            print("  [未完成] 请完成 calculate_roi_simple 函数")

    # 测试 1.5
    print("\n1.5 策略对比")
    if df is not None and 'uplift_pred' in locals():
        comparison = compare_strategies(df, uplift_pred, budget_fraction=0.3)
        if comparison is not None and len(comparison) > 0:
            print("\n  ROI 对比:")
            print(comparison[['strategy', 'roi', 'revenue', 'cost']].to_string(index=False))

            best_strategy = comparison.loc[comparison['roi'].idxmax(), 'strategy']
            best_roi = comparison['roi'].max()
            print(f"\n  最佳策略: {best_strategy} (ROI={best_roi:.2f})")
        else:
            print("  [未完成] 请完成 compare_strategies 函数")

    print("\n" + "=" * 50)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 50)
