"""
练习 2: 因果森林 (Causal Forest)

学习目标:
1. 理解因果森林的核心原理
2. 实现简化版因果森林
3. 使用 econml 的 CausalForest
4. 分析特征重要性
5. 对比因果森林与 T-Learner 的性能

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 尝试导入 econml
try:
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("警告: econml 未安装，部分功能将不可用")
    print("安装命令: pip install econml")


# ==================== 练习 2.1: 理解诚实分裂 ====================

def honest_split_data(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    split_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[Tuple, Tuple]:
    """
    诚实分裂: 将数据分为两部分
    - 分裂样本: 用于构建树结构
    - 估计样本: 用于叶节点的 CATE 估计

    这是因果森林的核心创新，防止过拟合

    TODO: 实现数据分裂

    Args:
        X, T, Y: 特征、处理、结果
        split_ratio: 分裂样本的比例
        seed: 随机种子

    Returns:
        ((X_split, T_split, Y_split), (X_est, T_est, Y_est))
    """
    np.random.seed(seed)
    n = len(X)

    # TODO: 随机划分索引
    indices = np.arange(n)
    np.random.shuffle(indices)

    # TODO: 计算分裂点
    split_point = None  # 你的代码

    # TODO: 划分数据
    split_idx = None  # 你的代码
    est_idx = None  # 你的代码

    # 你的代码: 返回两部分数据
    pass


# ==================== 练习 2.2: 简化因果树 ====================

class SimpleCausalTree:
    """
    简化版因果树

    核心思想:
    - 使用分裂样本构建树 (根据特征分裂，最大化叶节点内的效应方差)
    - 使用估计样本计算叶节点的 CATE
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练因果树

        简化实现: 使用标准决策树预测 Y，然后在叶节点计算 CATE

        TODO: 实现训练逻辑
        """
        # TODO: 诚实分裂
        (X_split, T_split, Y_split), (X_est, T_est, Y_est) = honest_split_data(
            X, T, Y, split_ratio=0.5
        )

        # TODO: 使用分裂样本训练决策树
        # 提示: 可以使用 DecisionTreeRegressor
        self.tree = None  # 你的代码

        # 保存估计样本用于预测
        self.X_est = X_est
        self.T_est = T_est
        self.Y_est = Y_est

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测 CATE

        TODO: 实现预测逻辑
        """
        if self.tree is None:
            raise ValueError("模型未训练")

        # TODO: 找到每个样本所在的叶节点
        # TODO: 计算该叶节点的 CATE (使用估计样本)

        cate_pred = np.zeros(len(X))

        # 简化实现: 使用树的预测作为近似
        # 真实实现应该在叶节点内计算 E[Y|T=1] - E[Y|T=0]

        # 你的代码
        pass


# ==================== 练习 2.3: 使用 econml CausalForest ====================

def train_causal_forest(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_estimators: int = 100
) -> Optional[object]:
    """
    使用 econml 的 CausalForest

    TODO: 训练因果森林

    Returns:
        训练好的模型 (如果 econml 可用)
    """
    if not ECONML_AVAILABLE:
        print("econml 不可用，跳过")
        return None

    # TODO: 创建 CausalForest 实例
    # 参数: n_estimators, min_samples_leaf, max_depth 等
    cf = None  # 你的代码

    # TODO: 训练模型
    # 你的代码

    return cf


def get_feature_importances(
    model,
    feature_names: list
) -> pd.DataFrame:
    """
    获取特征重要性

    TODO: 提取特征重要性并排序

    Returns:
        DataFrame with columns: feature, importance
    """
    # TODO: 获取特征重要性
    # 提示: model.feature_importances_

    # 你的代码
    pass


# ==================== 练习 2.4: 对比实验 ====================

def compare_models(
    X_train: np.ndarray,
    T_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    tau_test: np.ndarray
) -> pd.DataFrame:
    """
    对比不同模型的性能

    模型:
    1. T-Learner (Random Forest)
    2. Causal Forest (如果可用)

    TODO: 训练模型并计算评估指标

    Returns:
        性能对比表
    """
    results = []

    # 模型 1: T-Learner
    print("训练 T-Learner...")
    # TODO: 实现 T-Learner
    model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_1 = RandomForestRegressor(n_estimators=100, random_state=43)

    # TODO: 分别训练处理组和对照组模型
    # 你的代码

    # TODO: 预测 CATE
    tau_pred_tlearner = None  # 你的代码

    if tau_pred_tlearner is not None:
        # TODO: 计算 PEHE
        pehe_tlearner = None  # 你的代码: np.sqrt(np.mean((tau_test - tau_pred_tlearner)**2))

        # TODO: 计算 R²
        r2_tlearner = None  # 你的代码

        results.append({
            'Model': 'T-Learner',
            'PEHE': pehe_tlearner,
            'R²': r2_tlearner
        })

    # 模型 2: Causal Forest
    if ECONML_AVAILABLE:
        print("训练 Causal Forest...")
        try:
            # TODO: 训练因果森林
            cf = train_causal_forest(X_train, T_train, Y_train)

            if cf is not None:
                # TODO: 预测 CATE
                tau_pred_cf = None  # 你的代码: cf.predict(X_test)

                # TODO: 计算指标
                pehe_cf = None  # 你的代码
                r2_cf = None  # 你的代码

                results.append({
                    'Model': 'Causal Forest',
                    'PEHE': pehe_cf,
                    'R²': r2_cf
                })
        except Exception as e:
            print(f"Causal Forest 训练失败: {e}")

    return pd.DataFrame(results)


# ==================== 练习 2.5: 特征重要性分析 ====================

def analyze_feature_importance(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    feature_names: list
) -> None:
    """
    分析特征对 CATE 的重要性

    TODO: 训练模型并可视化特征重要性
    """
    if not ECONML_AVAILABLE:
        print("需要 econml 库")
        return

    # TODO: 训练因果森林
    cf = train_causal_forest(X, T, Y)

    if cf is None:
        return

    # TODO: 获取特征重要性
    importance_df = get_feature_importances(cf, feature_names)

    if importance_df is not None:
        print("\n特征重要性:")
        print(importance_df.to_string(index=False))

        # TODO: (可选) 可视化
        # import matplotlib.pyplot as plt
        # plt.barh(importance_df['feature'], importance_df['importance'])
        # plt.xlabel('Importance')
        # plt.title('Feature Importance for CATE')
        # plt.show()


# ==================== 练习 2.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 什么是诚实分裂 (Honest Splitting)? 它如何防止过拟合?

你的答案:


2. 因果森林与标准随机森林有何不同? 为什么不能直接用随机森林估计 CATE?

你的答案:


3. 在什么情况下，因果森林比 T-Learner 表现更好?

你的答案:


4. 特征重要性在因果推断中如何解释? 它与预测模型中的特征重要性有何不同?

你的答案:


5. 因果森林的计算复杂度如何? 在什么情况下应该考虑使用更简单的方法?

你的答案:

"""


# ==================== 辅助函数 ====================

def generate_test_data(
    n: int = 2000,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    生成测试数据
    """
    np.random.seed(seed)

    # 特征
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)

    # 随机处理
    T = np.random.binomial(1, 0.5, n)

    # 异质性效应
    tau = 3.0 + 2.0 * X1 - 1.5 * X2

    # 结果
    Y0 = 10 + 1.5 * X1 + 1.0 * X2 + 0.5 * X3 + np.random.randn(n)
    Y1 = Y0 + tau
    Y = np.where(T == 1, Y1, Y0)

    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'T': T,
        'Y': Y
    })

    return df, tau


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 2: 因果森林")
    print("=" * 60)

    # 生成数据
    print("\n生成测试数据...")
    df, tau_true = generate_test_data(n=2000)

    # 分割训练/测试集
    train_idx = np.random.choice(len(df), int(0.7 * len(df)), replace=False)
    test_idx = np.array([i for i in range(len(df)) if i not in train_idx])

    X_train = df.loc[train_idx, ['X1', 'X2', 'X3']].values
    T_train = df.loc[train_idx, 'T'].values
    Y_train = df.loc[train_idx, 'Y'].values

    X_test = df.loc[test_idx, ['X1', 'X2', 'X3']].values
    T_test = df.loc[test_idx, 'T'].values
    Y_test = df.loc[test_idx, 'Y'].values
    tau_test = tau_true[test_idx]

    # 测试 2.1
    print("\n2.1 诚实分裂")
    try:
        split_data = honest_split_data(X_train, T_train, Y_train)
        if split_data is not None:
            (X_split, T_split, Y_split), (X_est, T_est, Y_est) = split_data
            print(f"  分裂样本: {len(X_split)}")
            print(f"  估计样本: {len(X_est)}")
        else:
            print("  [未完成] 请完成 honest_split_data 函数")
    except Exception as e:
        print(f"  [错误] {e}")

    # 测试 2.2
    print("\n2.2 简化因果树")
    try:
        tree = SimpleCausalTree(max_depth=3, min_samples_leaf=50)
        # tree.fit(X_train, T_train, Y_train)
        # tau_pred = tree.predict(X_test)
        # if tau_pred is not None:
        #     pehe = np.sqrt(np.mean((tau_test - tau_pred)**2))
        #     print(f"  PEHE: {pehe:.4f}")
        print("  [可选练习] SimpleCausalTree 实现较复杂，可跳过")
    except Exception as e:
        print(f"  [错误] {e}")

    # 测试 2.3
    print("\n2.3 使用 econml CausalForest")
    if ECONML_AVAILABLE:
        try:
            cf = train_causal_forest(X_train, T_train, Y_train, n_estimators=100)
            if cf is not None:
                tau_pred_cf = cf.predict(X_test).flatten()
                pehe_cf = np.sqrt(np.mean((tau_test - tau_pred_cf)**2))
                print(f"  PEHE: {pehe_cf:.4f}")
            else:
                print("  [未完成] 请完成 train_causal_forest 函数")
        except Exception as e:
            print(f"  [错误] {e}")
    else:
        print("  [跳过] econml 未安装")

    # 测试 2.4
    print("\n2.4 模型对比")
    try:
        comparison_df = compare_models(
            X_train, T_train, Y_train,
            X_test, tau_test
        )
        if comparison_df is not None and len(comparison_df) > 0:
            print(comparison_df.to_string(index=False))
        else:
            print("  [未完成] 请完成 compare_models 函数")
    except Exception as e:
        print(f"  [错误] {e}")

    # 测试 2.5
    print("\n2.5 特征重要性")
    try:
        analyze_feature_importance(
            X_train, T_train, Y_train,
            feature_names=['X1', 'X2', 'X3']
        )
    except Exception as e:
        print(f"  [错误] {e}")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 60)
