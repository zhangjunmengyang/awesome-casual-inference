"""
练习 3: 双重稳健估计 (Doubly Robust Estimation)

学习目标:
1. 理解双重稳健性的核心概念
2. 实现 AIPW (Augmented IPW) 估计器
3. 验证双重稳健性质
4. 对比不同因果推断方法
5. 理解模型误设定的影响

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ==================== 练习 3.1: 理解 AIPW ====================

def estimate_outcome_models(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    估计结果模型

    mu_1(X) = E[Y|X, T=1]
    mu_0(X) = E[Y|X, T=0]

    TODO: 分别为处理组和控制组训练结果模型

    Args:
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量

    Returns:
        (mu_1预测值, mu_0预测值)
    """
    # TODO: 训练处理组的结果模型
    treated_mask = T == 1
    control_mask = T == 0

    # mu_1(X) - 使用处理组数据训练
    model_1 = Ridge(alpha=1.0)
    # 你的代码

    mu_1 = None  # 对所有样本预测

    # TODO: 训练控制组的结果模型
    # mu_0(X) - 使用控制组数据训练
    model_0 = Ridge(alpha=1.0)
    # 你的代码

    mu_0 = None  # 对所有样本预测

    return mu_1, mu_0


def estimate_ate_aipw(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    propensity: np.ndarray,
    mu_1: np.ndarray,
    mu_0: np.ndarray
) -> Tuple[float, float]:
    """
    使用 AIPW 估计 ATE

    AIPW 估计器:
    ATE = E[(mu_1(X) - mu_0(X)) +
            T*(Y - mu_1(X))/e(X) -
            (1-T)*(Y - mu_0(X))/(1-e(X))]

    TODO: 实现 AIPW 估计

    Args:
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量
        propensity: 倾向得分
        mu_1: E[Y|X, T=1] 的预测
        mu_0: E[Y|X, T=0] 的预测

    Returns:
        (ATE估计, 标准误)
    """
    # TODO: 裁剪倾向得分
    propensity_clipped = np.clip(propensity, 0.01, 0.99)

    # TODO: 计算 AIPW 估计器的三项
    # 第一项: 结果模型预测的差异
    term1 = None  # mu_1 - mu_0

    # 第二项: 处理组的 IPW 修正
    term2 = None  # T * (Y - mu_1) / propensity

    # 第三项: 控制组的 IPW 修正
    term3 = None  # (1 - T) * (Y - mu_0) / (1 - propensity)

    # TODO: AIPW 得分
    aipw_scores = None  # term1 + term2 - term3

    # TODO: ATE 估计
    ate = None  # aipw_scores 的均值

    # TODO: 标准误
    se = None  # aipw_scores 的标准差 / sqrt(n)

    return ate, se


# ==================== 练习 3.2: 实现完整的 AIPW 估计器 ====================

class AIPWEstimator:
    """
    AIPW 估计器类

    TODO: 实现完整的 AIPW 估计流程
    """

    def __init__(
        self,
        propensity_model=None,
        outcome_model=None
    ):
        """
        初始化估计器

        Args:
            propensity_model: 倾向得分模型
            outcome_model: 结果模型
        """
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)
        self.outcome_model = outcome_model or Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练模型

        TODO: 训练倾向得分模型和结果模型
        """
        # TODO: 训练倾向得分模型
        # 你的代码

        # TODO: 训练两个结果模型
        # 你的代码

        return self

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[float, float]:
        """
        估计 ATE

        TODO: 使用训练好的模型估计 ATE
        """
        # TODO: 预测倾向得分
        propensity = None  # 你的代码

        # TODO: 预测结果
        mu_1 = None  # 你的代码
        mu_0 = None  # 你的代码

        # TODO: 计算 AIPW
        ate, se = None, None  # 使用 estimate_ate_aipw

        return ate, se


# ==================== 练习 3.3: 验证双重稳健性 ====================

def simulate_with_misspecification(
    n: int = 2000,
    propensity_correct: bool = True,
    outcome_correct: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    生成数据，可以人为引入模型误设定

    真实 DGP (非线性):
    - X1, X2, X3 ~ N(0, 1)
    - logit(e) = 1.5 * (X1 + 0.5*X2 + 0.3*X1^2)  # 非线性
    - Y = 5 + 2*T + 1.5*X1 + X2 + 0.5*X1^2 + noise  # 非线性

    如果只用线性模型，就会误设定

    TODO: 生成数据，根据参数决定使用的特征

    Args:
        n: 样本量
        propensity_correct: 倾向得分模型是否正确指定
        outcome_correct: 结果模型是否正确指定
        seed: 随机种子

    Returns:
        (X, T, Y, true_ate)
    """
    np.random.seed(seed)

    # TODO: 生成特征
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)

    # TODO: 真实倾向得分 (包含非线性项)
    propensity_logit = 1.5 * (X1 + 0.5*X2 + 0.3*X1**2)
    propensity_true = 1 / (1 + np.exp(-propensity_logit))

    T = np.random.binomial(1, propensity_true)

    # TODO: 真实结果 (包含非线性项)
    true_ate = 2.0
    Y = 5 + true_ate*T + 1.5*X1 + X2 + 0.5*X1**2 + np.random.randn(n)*0.5

    # TODO: 根据正确性返回不同的特征
    if propensity_correct and outcome_correct:
        # 两个都正确: 包含非线性特征
        X = np.column_stack([X1, X2, X3, X1**2, X2**2])
    elif propensity_correct and not outcome_correct:
        # 只有倾向得分正确
        X_prop = np.column_stack([X1, X2, X3, X1**2, X2**2])
        X_outcome = np.column_stack([X1, X2, X3])  # 缺少非线性项
        return (X_prop, X_outcome, T, Y, true_ate)
    elif not propensity_correct and outcome_correct:
        # 只有结果模型正确
        X_prop = np.column_stack([X1, X2, X3])  # 缺少非线性项
        X_outcome = np.column_stack([X1, X2, X3, X1**2, X2**2])
        return (X_prop, X_outcome, T, Y, true_ate)
    else:
        # 两个都错误: 缺少非线性特征
        X = np.column_stack([X1, X2, X3])

    return X, T, Y, true_ate


def test_double_robustness(n: int = 3000) -> pd.DataFrame:
    """
    测试双重稳健性质

    四种情况:
    1. 两个模型都正确
    2. 只有倾向得分正确
    3. 只有结果模型正确
    4. 两个模型都错误

    TODO: 在四种情况下估计 ATE，验证双重稳健性

    Returns:
        DataFrame with results
    """
    scenarios = [
        ('两模型都正确', True, True),
        ('只有倾向得分正确', True, False),
        ('只有结果模型正确', False, True),
        ('两模型都错误', False, False)
    ]

    results = []

    for name, prop_correct, outcome_correct in scenarios:
        # TODO: 生成数据
        data = simulate_with_misspecification(n, prop_correct, outcome_correct)

        # 处理不同的返回格式
        if len(data) == 5:
            X_prop, X_outcome, T, Y, true_ate = data
        else:
            X, T, Y, true_ate = data
            X_prop = X_outcome = X

        # TODO: 估计倾向得分
        lr = LogisticRegression(max_iter=1000)
        # 你的代码
        propensity = None

        # TODO: 估计结果模型
        mu_1, mu_0 = None, None  # 你的代码

        # TODO: AIPW 估计
        aipw_ate, aipw_se = None, None  # 你的代码

        if aipw_ate is None:
            continue

        # TODO: 计算偏差
        bias = None  # 你的代码

        results.append({
            'scenario': name,
            'propensity_correct': prop_correct,
            'outcome_correct': outcome_correct,
            'ate': aipw_ate,
            'se': aipw_se,
            'bias': bias,
            'abs_bias': abs(bias) if bias else None
        })

    return pd.DataFrame(results)


# ==================== 练习 3.4: 对比不同方法 ====================

def compare_causal_methods(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    true_ate: float
) -> pd.DataFrame:
    """
    对比不同的因果推断方法

    TODO: 实现并对比:
    1. 朴素估计
    2. 回归调整
    3. IPW
    4. AIPW

    Args:
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量
        true_ate: 真实 ATE

    Returns:
        DataFrame with comparison
    """
    results = []

    # TODO: 1. 朴素估计
    naive_ate = None  # mean(Y|T=1) - mean(Y|T=0)

    results.append({
        'method': '朴素估计',
        'ate': naive_ate,
        'bias': naive_ate - true_ate if naive_ate else None,
        'se': None
    })

    # TODO: 2. 回归调整
    # Y ~ T + X, 系数就是 ATE
    # 你的代码

    # TODO: 3. IPW
    # 你的代码

    # TODO: 4. AIPW
    # 你的代码

    return pd.DataFrame(results)


# ==================== 练习 3.5: 交叉拟合 ====================

def cross_fitting_aipw(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5
) -> Tuple[float, float]:
    """
    使用交叉拟合的 AIPW

    交叉拟合可以避免过拟合偏差，特别是使用机器学习模型时

    TODO: 实现 K-fold 交叉拟合

    算法:
    1. 将数据分成 K 折
    2. 对每一折:
       - 用其他 K-1 折训练模型
       - 用当前折预测并计算 AIPW 得分
    3. 汇总所有折的得分

    Args:
        X: 特征矩阵
        T: 处理状态
        Y: 结果变量
        n_folds: 折数

    Returns:
        (ATE估计, 标准误)
    """
    from sklearn.model_selection import KFold

    n = len(Y)
    aipw_scores = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        # TODO: 分割数据
        X_train, X_test = None, None  # 你的代码
        T_train, T_test = None, None
        Y_train, Y_test = None, None

        # TODO: 在训练集上训练模型
        # 倾向得分模型
        lr = LogisticRegression(max_iter=1000)
        # 你的代码
        propensity_test = None

        # TODO: 结果模型
        mu_1_test, mu_0_test = None, None  # 你的代码

        # TODO: 计算测试集上的 AIPW 得分
        propensity_clipped = np.clip(propensity_test, 0.01, 0.99)

        term1 = mu_1_test - mu_0_test
        term2 = T_test * (Y_test - mu_1_test) / propensity_clipped
        term3 = (1 - T_test) * (Y_test - mu_0_test) / (1 - propensity_clipped)

        aipw_scores[test_idx] = term1 + term2 - term3

    # TODO: 计算 ATE 和标准误
    ate = None  # 你的代码
    se = None   # 你的代码

    return ate, se


# ==================== 练习 3.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 什么是双重稳健性? 为什么 AIPW 具有这个性质?

你的答案:


2. AIPW 估计器的三项分别代表什么含义?

你的答案:


3. 如果两个模型都错误，AIPW 还会有偏吗? 为什么?

你的答案:


4. AIPW 相比单独的 IPW 或回归调整有什么优势?

你的答案:


5. 什么是交叉拟合? 为什么在使用机器学习模型时需要交叉拟合?

你的答案:


6. 在实践中，如何选择倾向得分模型和结果模型?

你的答案:


7. 双重稳健估计是"银弹"吗? 有什么局限性?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 3: 双重稳健估计 (AIPW)")
    print("=" * 60)

    # 测试 3.1
    print("\n3.1 生成简单数据")
    np.random.seed(42)
    n = 2000
    X = np.random.randn(n, 3)
    propensity_logit = 1.5 * (X[:, 0] + 0.5*X[:, 1])
    propensity_true = 1 / (1 + np.exp(-propensity_logit))
    T = np.random.binomial(1, propensity_true)
    true_ate = 2.0
    Y = 5 + true_ate*T + 1.5*X[:, 0] + X[:, 1] + np.random.randn(n)*0.5

    print(f"  样本量: {n}")
    print(f"  真实 ATE: {true_ate:.4f}")

    # 测试 3.2
    print("\n3.2 估计结果模型")
    mu_1, mu_0 = estimate_outcome_models(X, T, Y)
    if mu_1 is not None and mu_0 is not None:
        print(f"  mu_1 范围: [{mu_1.min():.2f}, {mu_1.max():.2f}]")
        print(f"  mu_0 范围: [{mu_0.min():.2f}, {mu_0.max():.2f}]")
        print(f"  预测的平均处理效应: {(mu_1 - mu_0).mean():.4f}")
    else:
        print("  [未完成] 请完成 estimate_outcome_models 函数")

    # 测试 3.3
    print("\n3.3 AIPW 估计")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, T)
    propensity = lr.predict_proba(X)[:, 1]

    if mu_1 is not None:
        aipw_ate, aipw_se = estimate_ate_aipw(X, T, Y, propensity, mu_1, mu_0)
        if aipw_ate is not None:
            print(f"  AIPW ATE: {aipw_ate:.4f} ± {aipw_se:.4f}")
            print(f"  95% CI: [{aipw_ate - 1.96*aipw_se:.4f}, {aipw_ate + 1.96*aipw_se:.4f}]")
            print(f"  偏差: {aipw_ate - true_ate:.4f}")
        else:
            print("  [未完成] 请完成 estimate_ate_aipw 函数")

    # 测试 3.4
    print("\n3.4 验证双重稳健性")
    dr_results = test_double_robustness(n=3000)
    if dr_results is not None and not dr_results.empty:
        print("\n  双重稳健性测试结果:")
        print(dr_results[['scenario', 'ate', 'bias', 'abs_bias']].to_string(index=False))
        print("\n  关键观察:")
        print("  - 两模型都正确: 偏差应该很小")
        print("  - 只有一个正确: 偏差仍然应该很小 (双重稳健性!)")
        print("  - 两模型都错: 偏差会较大")
    else:
        print("  [未完成] 请完成 test_double_robustness 函数")

    # 测试 3.5
    print("\n3.5 对比不同方法")
    comparison = compare_causal_methods(X, T, Y, true_ate)
    if comparison is not None and not comparison.empty:
        print(comparison.to_string(index=False))
    else:
        print("  [未完成] 请完成 compare_causal_methods 函数")

    # 测试 3.6
    print("\n3.6 交叉拟合 AIPW")
    cf_ate, cf_se = cross_fitting_aipw(X, T, Y, n_folds=5)
    if cf_ate is not None:
        print(f"  交叉拟合 ATE: {cf_ate:.4f} ± {cf_se:.4f}")
        print(f"  偏差: {cf_ate - true_ate:.4f}")

        if aipw_ate is not None:
            print(f"\n  对比:")
            print(f"  标准 AIPW: {aipw_ate:.4f} ± {aipw_se:.4f}")
            print(f"  交叉拟合: {cf_ate:.4f} ± {cf_se:.4f}")
    else:
        print("  [未完成] 请完成 cross_fitting_aipw 函数")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("提示: AIPW 的双重稳健性是因果推断中的重要性质")
    print("=" * 60)
