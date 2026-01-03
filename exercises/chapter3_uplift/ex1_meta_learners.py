"""
练习 1: Meta-Learners

学习目标:
1. 理解 S-Learner 和 T-Learner 的原理
2. 实现基础的 Meta-Learner 算法
3. 理解不同方法的优缺点
4. 掌握 CATE 估计和评估

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from typing import Tuple


# ==================== 练习 1.1: 实现 S-Learner ====================

class SimpleSLearner:
    """
    S-Learner (Single Model Learner)

    核心思想:
    - 将处理 T 作为一个特征
    - 训练单一模型: Y = f(X, T)
    - CATE 估计: tau(x) = f(x, 1) - f(x, 0)

    TODO: 完成 S-Learner 的实现
    """

    def __init__(self):
        # TODO: 初始化基础模型
        # 可以使用 LinearRegression 或 RandomForestRegressor
        self.model = None  # 你的代码

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练 S-Learner

        TODO:
        1. 将 T 和 X 合并为特征矩阵
        2. 训练模型预测 Y
        """
        # 你的代码
        pass

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        预测 CATE (个体处理效应)

        TODO:
        1. 预测 Y(1): 当 T=1 时的结果
        2. 预测 Y(0): 当 T=0 时的结果
        3. 返回差值: Y(1) - Y(0)
        """
        # 你的代码
        pass


# ==================== 练习 1.2: 实现 T-Learner ====================

class SimpleTLearner:
    """
    T-Learner (Two Model Learner)

    核心思想:
    - 分别为处理组和控制组训练两个模型
    - mu_0(x) = E[Y|X=x, T=0]  (控制组模型)
    - mu_1(x) = E[Y|X=x, T=1]  (处理组模型)
    - CATE 估计: tau(x) = mu_1(x) - mu_0(x)

    TODO: 完成 T-Learner 的实现
    """

    def __init__(self):
        # TODO: 初始化两个模型
        self.model_0 = None  # 控制组模型
        self.model_1 = None  # 处理组模型

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练 T-Learner

        TODO:
        1. 将数据按 T 分成两组
        2. 用控制组数据训练 model_0
        3. 用处理组数据训练 model_1
        """
        # 你的代码
        pass

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        预测 CATE

        TODO:
        1. 使用 model_1 预测 Y(1)
        2. 使用 model_0 预测 Y(0)
        3. 返回差值
        """
        # 你的代码
        pass


# ==================== 练习 1.3: 生成 Uplift 数据 ====================

def generate_simple_uplift_data(
    n: int = 1000,
    heterogeneous: bool = True,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成简单的 Uplift 数据

    数据生成过程 (DGP):
    - X1, X2 ~ N(0, 1)
    - T ~ Bernoulli(0.5)  (随机分配)
    - Y(0) = 5 + 2*X1 + X2 + noise
    - Y(1) = Y(0) + tau(X)

    其中:
    - 如果 heterogeneous=True: tau(X) = 2 + X1 - 0.5*X2  (异质性效应)
    - 如果 heterogeneous=False: tau(X) = 2  (常数效应)

    TODO: 完成数据生成代码

    Returns:
        (DataFrame, true_cate)
        DataFrame columns: X1, X2, T, Y
        true_cate: 真实的个体处理效应
    """
    np.random.seed(seed)

    # TODO: 生成特征 X1, X2
    X1 = None  # 你的代码
    X2 = None  # 你的代码

    # TODO: 随机分配处理 T
    T = None  # 你的代码

    # TODO: 计算真实 CATE
    if heterogeneous:
        true_cate = None  # 异质性: 2 + X1 - 0.5*X2
    else:
        true_cate = None  # 常数: 2

    # TODO: 生成潜在结果
    # Y(0) = 5 + 2*X1 + X2 + noise
    # Y(1) = Y(0) + true_cate
    # Y = T * Y(1) + (1-T) * Y(0)
    Y0 = None  # 你的代码
    Y1 = None  # 你的代码
    Y = None   # 你的代码

    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'T': T,
        'Y': Y
    })

    return df, true_cate


# ==================== 练习 1.4: 评估 CATE 估计 ====================

def evaluate_cate_estimation(
    true_cate: np.ndarray,
    predicted_cate: np.ndarray
) -> dict:
    """
    评估 CATE 估计的质量

    TODO: 计算以下指标:
    1. MSE: 均方误差
    2. MAE: 平均绝对误差
    3. Correlation: 与真实 CATE 的相关系数
    4. PEHE: Precision in Estimation of Heterogeneous Effect
           PEHE = sqrt(MSE)

    Returns:
        字典包含所有指标
    """
    metrics = {
        'MSE': None,      # 你的代码
        'MAE': None,      # 你的代码
        'Correlation': None,  # 你的代码
        'PEHE': None      # 你的代码
    }

    return metrics


# ==================== 练习 1.5: 比较 S-Learner 和 T-Learner ====================

def compare_s_and_t_learner(
    n_samples: int = 2000,
    heterogeneous: bool = True
) -> pd.DataFrame:
    """
    比较 S-Learner 和 T-Learner 的性能

    TODO:
    1. 生成数据
    2. 训练 S-Learner
    3. 训练 T-Learner
    4. 评估两者的 CATE 估计
    5. 返回对比结果

    Returns:
        DataFrame with comparison results
    """
    # TODO: 生成数据
    df, true_cate = generate_simple_uplift_data(n_samples, heterogeneous)

    X = df[['X1', 'X2']].values
    T = df['T'].values
    Y = df['Y'].values

    # TODO: 训练 S-Learner
    s_learner = SimpleSLearner()
    # ... 你的代码 ...
    s_pred = None  # s_learner.predict_cate(X)

    # TODO: 训练 T-Learner
    t_learner = SimpleTLearner()
    # ... 你的代码 ...
    t_pred = None  # t_learner.predict_cate(X)

    # TODO: 评估
    results = []

    if s_pred is not None:
        s_metrics = evaluate_cate_estimation(true_cate, s_pred)
        s_metrics['Method'] = 'S-Learner'
        results.append(s_metrics)

    if t_pred is not None:
        t_metrics = evaluate_cate_estimation(true_cate, t_pred)
        t_metrics['Method'] = 'T-Learner'
        results.append(t_metrics)

    return pd.DataFrame(results) if results else None


# ==================== 练习 1.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. S-Learner 的优点和缺点是什么?

你的答案:
优点:
-
-

缺点:
-
-


2. T-Learner 的优点和缺点是什么?

你的答案:
优点:
-
-

缺点:
-
-


3. 在什么情况下 S-Learner 可能表现更好? 在什么情况下 T-Learner 可能表现更好?

你的答案:
S-Learner 更好:
-

T-Learner 更好:
-


4. X-Learner 相比 S-Learner 和 T-Learner 有什么创新之处?

你的答案:
-
-


5. 如果处理组和控制组的样本量严重不平衡 (比如 90% vs 10%)，
   应该选择哪种 Meta-Learner? 为什么?

你的答案:


"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 1: Meta-Learners")
    print("=" * 60)

    # 测试 1.3: 生成数据
    print("\n1.3 生成 Uplift 数据")
    df, true_cate = generate_simple_uplift_data(n=100)
    if df is not None and true_cate is not None:
        print(f"  样本量: {len(df)}")
        print(f"  列: {list(df.columns)}")
        print(f"  真实平均 CATE: {true_cate.mean():.4f}")
        print(df.head())
    else:
        print("  [未完成] 请完成 generate_simple_uplift_data 函数")

    # 测试 1.1: S-Learner
    print("\n1.1 S-Learner")
    if df is not None:
        try:
            s_learner = SimpleSLearner()
            X = df[['X1', 'X2']].values
            T = df['T'].values
            Y = df['Y'].values

            s_learner.fit(X, T, Y)
            s_pred = s_learner.predict_cate(X)

            if s_pred is not None:
                print(f"  预测平均 CATE: {s_pred.mean():.4f}")
                print(f"  真实平均 CATE: {true_cate.mean():.4f}")
            else:
                print("  [未完成] 请完成 S-Learner 的 predict_cate 方法")
        except Exception as e:
            print(f"  [未完成] 请完成 S-Learner 实现: {e}")

    # 测试 1.2: T-Learner
    print("\n1.2 T-Learner")
    if df is not None:
        try:
            t_learner = SimpleTLearner()
            t_learner.fit(X, T, Y)
            t_pred = t_learner.predict_cate(X)

            if t_pred is not None:
                print(f"  预测平均 CATE: {t_pred.mean():.4f}")
                print(f"  真实平均 CATE: {true_cate.mean():.4f}")
            else:
                print("  [未完成] 请完成 T-Learner 的 predict_cate 方法")
        except Exception as e:
            print(f"  [未完成] 请完成 T-Learner 实现: {e}")

    # 测试 1.4: 评估
    print("\n1.4 CATE 评估")
    if df is not None and s_pred is not None:
        metrics = evaluate_cate_estimation(true_cate, s_pred)
        if metrics['MSE'] is not None:
            print("  S-Learner 指标:")
            for key, value in metrics.items():
                print(f"    {key}: {value:.4f}")
        else:
            print("  [未完成] 请完成 evaluate_cate_estimation 函数")

    # 测试 1.5: 对比
    print("\n1.5 S-Learner vs T-Learner 对比")
    print("\n--- 常数效应场景 ---")
    results_constant = compare_s_and_t_learner(n_samples=1000, heterogeneous=False)
    if results_constant is not None:
        print(results_constant.to_string(index=False))
    else:
        print("  [未完成] 请完成所有前序练习")

    print("\n--- 异质性效应场景 ---")
    results_hetero = compare_s_and_t_learner(n_samples=1000, heterogeneous=True)
    if results_hetero is not None:
        print(results_hetero.to_string(index=False))
    else:
        print("  [未完成] 请完成所有前序练习")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("思考题请在代码注释中回答")
    print("=" * 60)
