"""
练习 3: 用户定向干预

学习目标:
1. 理解 CATE (Conditional Average Treatment Effect) 估计
2. 掌握 X-Learner 方法
3. 学习最优干预策略 (Policy Learning)
4. 理解成本-收益权衡

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


# ==================== 练习 3.1: 司机激励数据生成 ====================

def generate_driver_data(
    n_samples: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成网约车司机激励数据

    场景: 网约车平台给司机发放奖励，激励其增加在线时长
    - 特征: 司机评分、历史订单数、是否全职
    - 处理: 是否给予奖励 (T=1 奖励, T=0 无奖励)
    - 结果: 当日在线时长 (小时)

    TODO: 完成数据生成

    Returns:
        DataFrame with columns: rating, order_history, is_fulltime, T, online_hours
    """
    np.random.seed(seed)

    # TODO: 生成司机特征
    # rating: 评分 4.0-5.0，beta 分布 (偏向高分)
    # order_history: 历史完单数，泊松分布 lambda=200
    # is_fulltime: 是否全职，30% 概率为全职
    rating = None  # 你的代码: np.random.beta(8, 1, n_samples) * 1.0 + 4.0
    order_history = None  # 你的代码
    is_fulltime = None  # 你的代码

    # TODO: 随机分配处理 (50% 概率)
    T = None  # 你的代码

    # TODO: 生成在线时长
    # 基线时长: 全职 6 小时，兼职 3 小时 + 一些个体差异
    # 激励效应 (异质性):
    #   - 兼职司机: +2 小时
    #   - 全职司机: +0.5 小时 (已经很活跃，边际效应小)
    #   - 低单量司机 (order_history < 150): +2.5 小时
    #   - 高单量司机: +1 小时

    online_hours = []
    for i in range(n_samples):
        # 你的代码
        pass

    return pd.DataFrame({
        'rating': rating,
        'order_history': order_history,
        'is_fulltime': is_fulltime,
        'T': T,
        'online_hours': online_hours
    })


# ==================== 练习 3.2: CATE 估计 (T-Learner) ====================

class TLearner:
    """
    T-Learner (Two-Model Learner)

    思想: 分别训练处理组和控制组的模型，预测差值即为 CATE
    """

    def __init__(self):
        self.model_control = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.model_treatment = GradientBoostingRegressor(n_estimators=50, random_state=43)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练 T-Learner

        TODO: 实现训练逻辑

        Args:
            X: 特征矩阵
            T: 处理指示
            Y: 结果变量
        """
        # TODO: 分离控制组和处理组
        mask_control = None  # 你的代码
        mask_treatment = None  # 你的代码

        # TODO: 训练两个模型
        # self.model_control.fit(...)
        # self.model_treatment.fit(...)

        # 你的代码
        pass

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        预测 CATE

        TODO: 实现 CATE 预测

        Returns:
            cate: 条件平均处理效应
        """
        # TODO: 分别预测两组的结果
        mu1 = None  # 你的代码: self.model_treatment.predict(X)
        mu0 = None  # 你的代码: self.model_control.predict(X)

        # TODO: CATE = mu1 - mu0
        cate = None  # 你的代码

        return cate


# ==================== 练习 3.3: CATE 估计 (X-Learner) ====================

class XLearner:
    """
    X-Learner (Advanced meta-learner)

    相比 T-Learner，X-Learner 额外学习伪处理效应，通常更准确
    """

    def __init__(self):
        self.model_control = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.model_treatment = GradientBoostingRegressor(n_estimators=50, random_state=43)
        self.tau_control = GradientBoostingRegressor(n_estimators=30, random_state=44)
        self.tau_treatment = GradientBoostingRegressor(n_estimators=30, random_state=45)
        self.propensity_model = LogisticRegression(random_state=46)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练 X-Learner

        TODO: 实现 X-Learner 的三个阶段

        Stage 1: 训练 mu_0 和 mu_1
        Stage 2: 计算伪处理效应并训练 tau_0 和 tau_1
        Stage 3: 估计倾向得分

        Args:
            X: 特征矩阵
            T: 处理指示
            Y: 结果变量
        """
        mask_control = (T == 0)
        mask_treatment = (T == 1)

        # TODO: Stage 1 - 训练 mu_0 和 mu_1
        # self.model_control.fit(...)
        # self.model_treatment.fit(...)

        # 你的代码
        pass

        # TODO: Stage 2 - 计算伪处理效应
        # 对于处理组: D_1 = Y - mu_0(X) (如果接受处理，相比不接受的差异)
        # 对于控制组: D_0 = mu_1(X) - Y (如果接受处理，相比不接受的差异)

        # D_treatment = ...
        # self.tau_treatment.fit(...)

        # D_control = ...
        # self.tau_control.fit(...)

        # 你的代码
        pass

        # TODO: Stage 3 - 估计倾向得分
        # self.propensity_model.fit(...)

        # 你的代码
        pass

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        预测 CATE (使用倾向得分加权)

        TODO: 实现 CATE 预测

        Returns:
            cate: 加权后的 CATE 估计
        """
        # TODO: 预测两个 tau
        tau0 = None  # 你的代码: self.tau_control.predict(X)
        tau1 = None  # 你的代码: self.tau_treatment.predict(X)

        # TODO: 估计倾向得分
        propensity = None  # 你的代码: self.propensity_model.predict_proba(X)[:, 1]

        # TODO: 加权组合
        # cate = g(X) * tau0 + (1 - g(X)) * tau1
        # 其中 g(X) = propensity
        cate = None  # 你的代码

        return cate


# ==================== 练习 3.4: 最优策略学习 ====================

def learn_optimal_policy(
    cate: np.ndarray,
    cost_per_treatment: float = 100,
    value_per_hour: float = 30
) -> Tuple[np.ndarray, Dict]:
    """
    学习最优干预策略

    决策规则: 当 CATE * value > cost 时进行干预

    TODO: 实现最优策略

    Args:
        cate: 估计的 CATE (在线时长增量)
        cost_per_treatment: 每次激励的成本 (元)
        value_per_hour: 每小时的价值 (元，如平台抽成)

    Returns:
        (optimal_policy, metrics)
        - optimal_policy: 0/1 数组，表示是否干预
        - metrics: 策略指标
    """
    # TODO: 计算阈值
    # threshold = cost / value (多少小时增量才划算)
    threshold = None  # 你的代码

    # TODO: 最优策略: CATE > threshold 时干预
    optimal_policy = None  # 你的代码: (cate > threshold).astype(int)

    # TODO: 计算指标
    n_treated = None  # 干预人数
    expected_hours = None  # 预期总时长增量
    total_cost = None  # 总成本
    total_value = None  # 总价值
    net_benefit = None  # 净收益
    roi = None  # ROI

    metrics = {
        'n_treated': n_treated,
        'treatment_rate': n_treated / len(cate) if len(cate) > 0 else 0,
        'expected_hours': expected_hours,
        'total_cost': total_cost,
        'total_value': total_value,
        'net_benefit': net_benefit,
        'roi': roi,
        'threshold': threshold
    }

    return optimal_policy, metrics


# ==================== 练习 3.5: 策略对比 ====================

def compare_targeting_strategies(
    df: pd.DataFrame,
    cate: np.ndarray,
    cost: float = 100,
    value: float = 30
) -> pd.DataFrame:
    """
    对比不同干预策略

    策略:
    1. No Treatment: 不干预
    2. Treat All: 全量干预
    3. Treat Part-time: 只干预兼职司机 (传统规则)
    4. Optimal Policy: 基于 CATE 的最优策略

    TODO: 实现策略对比

    Args:
        df: 司机数据
        cate: CATE 估计
        cost: 激励成本
        value: 每小时价值

    Returns:
        对比结果 DataFrame
    """
    results = []

    # TODO: 策略 1 - No Treatment
    results.append({
        'strategy': 'No Treatment',
        'n_treated': 0,
        'cost': 0,
        'value': 0,
        'net_benefit': 0,
        'roi': 0
    })

    # TODO: 策略 2 - Treat All
    # 计算全量干预的成本和收益
    # 你的代码
    pass

    # TODO: 策略 3 - Treat Part-time Only
    # 只干预兼职司机 (is_fulltime=0)
    # 你的代码
    pass

    # TODO: 策略 4 - Optimal Policy
    optimal_policy, optimal_metrics = learn_optimal_policy(cate, cost, value)
    if optimal_policy is not None:
        results.append({
            'strategy': 'Optimal Policy',
            'n_treated': optimal_metrics['n_treated'],
            'cost': optimal_metrics['total_cost'],
            'value': optimal_metrics['total_value'],
            'net_benefit': optimal_metrics['net_benefit'],
            'roi': optimal_metrics['roi']
        })

    return pd.DataFrame(results)


# ==================== 练习 3.6: 用户分层 ====================

def segment_by_cate(
    df: pd.DataFrame,
    cate: np.ndarray
) -> pd.DataFrame:
    """
    根据 CATE 将用户分层

    分层:
    - High Impact: CATE > p75
    - Medium Impact: p25 < CATE < p75
    - Low Impact: 0 < CATE < p25
    - Negative Impact: CATE < 0

    TODO: 实现分层逻辑

    Args:
        df: 原始数据
        cate: CATE 估计

    Returns:
        带 segment 和 cate 列的 DataFrame
    """
    df = df.copy()
    df['cate'] = cate

    # TODO: 计算分位数
    p75 = None  # 你的代码: np.percentile(cate, 75)
    p25 = None  # 你的代码

    # TODO: 分层
    segments = []
    for c in cate:
        # 你的代码
        pass

    df['segment'] = segments
    return df


# ==================== 练习 3.7: 敏感性分析 ====================

def sensitivity_analysis(
    df: pd.DataFrame,
    cate: np.ndarray,
    cost_range: Tuple[float, float] = (50, 150),
    n_points: int = 10
) -> pd.DataFrame:
    """
    敏感性分析: 不同激励成本下的最优策略

    TODO: 实现敏感性分析

    Args:
        df: 司机数据
        cate: CATE 估计
        cost_range: 成本范围
        n_points: 采样点数

    Returns:
        不同成本下的策略结果
    """
    costs = np.linspace(cost_range[0], cost_range[1], n_points)
    results = []

    for cost in costs:
        # TODO: 计算该成本下的最优策略
        optimal_policy, metrics = learn_optimal_policy(cate, cost_per_treatment=cost)

        if optimal_policy is not None:
            results.append({
                'cost': cost,
                'treatment_rate': metrics['treatment_rate'],
                'net_benefit': metrics['net_benefit'],
                'roi': metrics['roi']
            })

    return pd.DataFrame(results)


# ==================== 练习 3.8: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. T-Learner 和 X-Learner 的主要区别是什么? X-Learner 在什么情况下更优?

你的答案:


2. 最优策略 "CATE * value > cost" 的经济学直觉是什么?

你的答案:


3. 为什么要使用倾向得分 (propensity score) 来加权 X-Learner 的预测?

你的答案:


4. 在真实业务中，如何处理 CATE 估计的不确定性?

你的答案:


5. 如果激励效应会随时间衰减 (激励疲劳)，应该如何调整策略?

你的答案:


6. 用户分层的业务价值是什么? 给出一个具体的应用场景。

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("练习 3: 用户定向干预")
    print("=" * 50)

    # 测试 3.1
    print("\n3.1 生成司机数据")
    df = generate_driver_data(n_samples=2000)
    if df is not None and len(df) > 0 and df['rating'].iloc[0] is not None:
        print(f"  样本量: {len(df)}")
        print(f"  全职司机占比: {df['is_fulltime'].mean():.2%}")
        print(f"  平均在线时长: {df['online_hours'].mean():.1f} 小时")
        print(f"  处理组占比: {df['T'].mean():.2%}")
    else:
        print("  [未完成] 请完成 generate_driver_data 函数")

    # 测试 3.2
    print("\n3.2 T-Learner CATE 估计")
    if df is not None and len(df) > 0 and df['rating'].iloc[0] is not None:
        X = df[['rating', 'order_history', 'is_fulltime']].values
        T = df['T'].values
        Y = df['online_hours'].values

        t_learner = TLearner()
        try:
            t_learner.fit(X, T, Y)
            cate_t = t_learner.predict_cate(X)
            if cate_t is not None and len(cate_t) > 0:
                print(f"  CATE 范围: [{cate_t.min():.2f}, {cate_t.max():.2f}] 小时")
                print(f"  平均 CATE: {cate_t.mean():.2f} 小时")
                print(f"  CATE 标准差: {cate_t.std():.2f} 小时")
            else:
                print("  [未完成] 请完成 TLearner.predict_cate 方法")
        except:
            print("  [未完成] 请完成 TLearner.fit 方法")

    # 测试 3.3
    print("\n3.3 X-Learner CATE 估计")
    if df is not None and len(df) > 0 and df['rating'].iloc[0] is not None:
        x_learner = XLearner()
        try:
            x_learner.fit(X, T, Y)
            cate_x = x_learner.predict_cate(X)
            if cate_x is not None and len(cate_x) > 0:
                print(f"  CATE 范围: [{cate_x.min():.2f}, {cate_x.max():.2f}] 小时")
                print(f"  平均 CATE: {cate_x.mean():.2f} 小时")
                print(f"  与 T-Learner 相关性: {np.corrcoef(cate_t, cate_x)[0,1]:.3f}")
            else:
                print("  [未完成] 请完成 XLearner.predict_cate 方法")
        except:
            print("  [未完成] 请完成 XLearner.fit 方法")

    # 测试 3.4
    print("\n3.4 最优策略学习")
    if 'cate_x' in locals() and cate_x is not None:
        optimal_policy, metrics = learn_optimal_policy(cate_x, cost_per_treatment=100, value_per_hour=30)
        if optimal_policy is not None:
            print(f"  干预人数: {metrics['n_treated']}")
            print(f"  干预比例: {metrics['treatment_rate']:.2%}")
            print(f"  预期时长增量: {metrics['expected_hours']:.0f} 小时")
            print(f"  总成本: ¥{metrics['total_cost']:,.0f}")
            print(f"  总价值: ¥{metrics['total_value']:,.0f}")
            print(f"  净收益: ¥{metrics['net_benefit']:,.0f}")
            print(f"  ROI: {metrics['roi']:.2f}")
            print(f"  阈值: {metrics['threshold']:.2f} 小时")
        else:
            print("  [未完成] 请完成 learn_optimal_policy 函数")

    # 测试 3.5
    print("\n3.5 策略对比")
    if df is not None and 'cate_x' in locals() and cate_x is not None:
        comparison = compare_targeting_strategies(df, cate_x, cost=100, value=30)
        if comparison is not None and len(comparison) > 0:
            print("\n  策略对比:")
            print(comparison.to_string(index=False))

            best_strategy = comparison.loc[comparison['net_benefit'].idxmax(), 'strategy']
            best_benefit = comparison['net_benefit'].max()
            print(f"\n  最佳策略: {best_strategy} (净收益=¥{best_benefit:,.0f})")
        else:
            print("  [未完成] 请完成 compare_targeting_strategies 函数")

    # 测试 3.6
    print("\n3.6 用户分层")
    if df is not None and 'cate_x' in locals() and cate_x is not None:
        df_segmented = segment_by_cate(df, cate_x)
        if 'segment' in df_segmented.columns:
            print(f"\n  分层分布:")
            print(df_segmented['segment'].value_counts())
            print(f"\n  各层平均 CATE:")
            print(df_segmented.groupby('segment')['cate'].mean().sort_values(ascending=False))
        else:
            print("  [未完成] 请完成 segment_by_cate 函数")

    # 测试 3.7
    print("\n3.7 敏感性分析")
    if df is not None and 'cate_x' in locals() and cate_x is not None:
        sensitivity = sensitivity_analysis(df, cate_x, cost_range=(50, 150), n_points=5)
        if sensitivity is not None and len(sensitivity) > 0:
            print("\n  不同成本下的策略:")
            print(sensitivity.to_string(index=False))
        else:
            print("  [未完成] 请完成 sensitivity_analysis 函数")

    print("\n" + "=" * 50)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 50)
