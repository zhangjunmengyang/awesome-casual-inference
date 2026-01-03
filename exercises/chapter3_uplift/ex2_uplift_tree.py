"""
练习 2: Uplift 决策树

学习目标:
1. 理解 Uplift Tree 的分裂准则
2. 实现简单的 Uplift 增益计算
3. 理解与传统决策树的区别
4. 掌握叶节点的 Uplift 估计

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ==================== 练习 2.1: Uplift 增益计算 ====================

def calculate_simple_uplift(
    y: np.ndarray,
    t: np.ndarray
) -> float:
    """
    计算简单的 Uplift (处理组转化率 - 控制组转化率)

    TODO: 实现 Uplift 计算

    Args:
        y: 结果 (0/1)
        t: 处理状态 (0/1)

    Returns:
        uplift value
    """
    # TODO: 计算处理组的平均结果
    # TODO: 计算控制组的平均结果
    # TODO: 返回差值

    # 你的代码
    pass


def calculate_kl_divergence_gain(
    y: np.ndarray,
    t: np.ndarray
) -> float:
    """
    计算 KL 散度增益 (Kullback-Leibler Divergence)

    KL 散度衡量处理组和控制组的分布差异:
    D_KL = p_t * log(p_t / p_c) + (1 - p_t) * log((1 - p_t) / (1 - p_c))

    其中:
    - p_t: 处理组转化率
    - p_c: 控制组转化率

    TODO: 实现 KL 散度计算

    Args:
        y: 结果
        t: 处理状态

    Returns:
        KL divergence value
    """
    # TODO: 计算处理组和控制组转化率
    mask_t = t == 1
    mask_c = t == 0

    n_t = mask_t.sum()
    n_c = mask_c.sum()

    # 如果某组没有样本，返回 0
    if n_t == 0 or n_c == 0:
        return 0.0

    # TODO: 计算转化率，并限制在 [0.001, 0.999] 避免 log(0)
    p_t = None  # 你的代码
    p_c = None  # 你的代码

    # TODO: 计算 KL 散度
    kl = None  # 你的代码

    return kl


def calculate_euclidean_distance_gain(
    y: np.ndarray,
    t: np.ndarray
) -> float:
    """
    计算欧氏距离增益

    ED = (p_t - p_c)^2

    TODO: 实现欧氏距离计算

    Returns:
        Euclidean distance value
    """
    # 你的代码
    pass


# ==================== 练习 2.2: 找最佳分裂点 ====================

def find_best_split_threshold(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    criterion: str = 'KL'
) -> Tuple[Optional[float], float]:
    """
    找到单变量数据的最佳分裂点

    TODO:
    1. 遍历所有可能的分裂阈值
    2. 对每个阈值，计算分裂后的增益
    3. 返回最佳阈值和增益

    Args:
        X: 单个特征的值 (1维数组)
        y: 结果
        t: 处理状态
        criterion: 分裂准则 ('KL', 'ED', 'simple')

    Returns:
        (best_threshold, best_gain)
    """
    # TODO: 获取所有唯一值作为候选分裂点
    unique_values = np.unique(X)

    if len(unique_values) < 2:
        return None, 0.0

    # 候选阈值: 相邻值的中点
    thresholds = None  # 你的代码

    best_threshold = None
    best_gain = -np.inf

    # 选择增益计算函数
    if criterion == 'KL':
        gain_func = calculate_kl_divergence_gain
    elif criterion == 'ED':
        gain_func = calculate_euclidean_distance_gain
    else:
        gain_func = lambda y, t: calculate_simple_uplift(y, t) ** 2

    # TODO: 遍历所有候选阈值
    for threshold in thresholds:
        # TODO: 按阈值分裂数据
        left_mask = None   # 你的代码
        right_mask = None  # 你的代码

        # 确保两边都有足够样本
        if left_mask.sum() < 50 or right_mask.sum() < 50:
            continue

        # TODO: 计算左右子节点的增益
        left_gain = None   # 你的代码
        right_gain = None  # 你的代码

        # TODO: 加权平均增益
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        n_total = len(y)

        weighted_gain = None  # 你的代码

        # 更新最佳分裂
        if weighted_gain > best_gain:
            best_gain = weighted_gain
            best_threshold = threshold

    return best_threshold, best_gain


# ==================== 练习 2.3: 叶节点 Uplift 估计 ====================

def estimate_leaf_uplift(
    y: np.ndarray,
    t: np.ndarray
) -> Tuple[float, dict]:
    """
    估计叶节点的 Uplift 及统计信息

    TODO: 计算以下信息:
    1. uplift: 处理效应估计
    2. n_treated: 处理组样本数
    3. n_control: 控制组样本数
    4. treated_rate: 处理组转化率
    5. control_rate: 控制组转化率

    Returns:
        (uplift, stats_dict)
    """
    mask_t = t == 1
    mask_c = t == 0

    n_treated = mask_t.sum()
    n_control = mask_c.sum()

    if n_treated == 0 or n_control == 0:
        return 0.0, {
            'n_treated': n_treated,
            'n_control': n_control,
            'treated_rate': 0.0,
            'control_rate': 0.0
        }

    # TODO: 计算转化率和 uplift
    treated_rate = None  # 你的代码
    control_rate = None  # 你的代码
    uplift = None        # 你的代码

    stats = {
        'n_treated': n_treated,
        'n_control': n_control,
        'treated_rate': treated_rate,
        'control_rate': control_rate
    }

    return uplift, stats


# ==================== 练习 2.4: 构建简单的 Uplift Tree ====================

class SimpleUpliftTree:
    """
    简单的单层 Uplift Tree

    只实现一次分裂，便于理解核心概念

    TODO: 完成 Uplift Tree 的实现
    """

    def __init__(self, criterion: str = 'KL', min_samples_leaf: int = 100):
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf

        # 树结构
        self.split_feature = None
        self.split_threshold = None
        self.left_uplift = None
        self.right_uplift = None
        self.left_stats = None
        self.right_stats = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        训练 Uplift Tree (单次分裂)

        TODO:
        1. 遍历所有特征，找最佳分裂
        2. 记录最佳分裂特征和阈值
        3. 计算左右叶节点的 uplift
        """
        n_samples, n_features = X.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        # TODO: 遍历所有特征
        for feature_idx in range(n_features):
            # 你的代码
            # 提示: 使用 find_best_split_threshold 函数
            pass

        # TODO: 使用最佳分裂点分裂数据，计算叶节点 uplift
        if best_threshold is not None:
            # 你的代码
            pass

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测 Uplift

        TODO: 根据分裂规则返回对应叶节点的 uplift
        """
        if self.split_threshold is None:
            # 未训练或无有效分裂
            return np.zeros(X.shape[0])

        # TODO: 根据分裂特征和阈值，返回对应的 uplift
        predictions = np.zeros(X.shape[0])
        # 你的代码

        return predictions

    def get_tree_info(self) -> str:
        """
        返回树的结构信息
        """
        if self.split_threshold is None:
            return "Tree not fitted or no valid split found"

        info = f"""
Uplift Tree Structure:
----------------------
Split Rule: X{self.split_feature} <= {self.split_threshold:.4f}

Left Node (X{self.split_feature} <= {self.split_threshold:.4f}):
  Uplift: {self.left_uplift:.4f}
  Treated: {self.left_stats['n_treated']} (rate: {self.left_stats['treated_rate']:.4f})
  Control: {self.left_stats['n_control']} (rate: {self.left_stats['control_rate']:.4f})

Right Node (X{self.split_feature} > {self.split_threshold:.4f}):
  Uplift: {self.right_uplift:.4f}
  Treated: {self.right_stats['n_treated']} (rate: {self.right_stats['treated_rate']:.4f})
  Control: {self.right_stats['n_control']} (rate: {self.right_stats['control_rate']:.4f})
        """
        return info


# ==================== 练习 2.5: 测试 Uplift Tree ====================

def test_uplift_tree():
    """
    测试 Uplift Tree

    生成带有异质性效应的数据，验证树能否正确分裂
    """
    np.random.seed(42)
    n = 2000

    # 生成特征
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X = np.column_stack([X1, X2])

    # 随机分配处理
    T = np.random.binomial(1, 0.5, n)

    # 异质性效应: X1 > 0 时 uplift 高，X1 <= 0 时 uplift 低
    true_uplift = np.where(X1 > 0, 0.3, 0.05)

    # 基线转化率
    baseline_prob = 0.2

    # 生成结果
    prob = baseline_prob + true_uplift * T
    Y = np.random.binomial(1, prob)

    # TODO: 训练 Uplift Tree
    tree = SimpleUpliftTree(criterion='KL')
    # ... 你的代码 ...

    # TODO: 预测
    pred_uplift = None  # tree.predict(X)

    # 打印结果
    if tree.split_threshold is not None:
        print(tree.get_tree_info())
        print(f"\n真实 Uplift (X1>0): {true_uplift[X1 > 0].mean():.4f}")
        print(f"真实 Uplift (X1<=0): {true_uplift[X1 <= 0].mean():.4f}")
    else:
        print("[未完成] 请完成 Uplift Tree 实现")

    return tree, pred_uplift, true_uplift


# ==================== 练习 2.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 为什么 Uplift Tree 需要特殊的分裂准则，而不能使用传统的信息增益或基尼系数?

你的答案:


2. KL 散度、欧氏距离、卡方统计量这三种分裂准则有什么区别? 各自适用于什么场景?

你的答案:
KL 散度:
-

欧氏距离:
-

卡方统计量:
-


3. Uplift Tree 的叶节点估计的是什么? 如何解释叶节点的值?

你的答案:


4. 如果某个叶节点只有处理组样本或只有控制组样本，应该如何处理?

你的答案:


5. Uplift Tree 相比 Meta-Learners 有什么优势和劣势?

你的答案:
优势:
-

劣势:
-


6. 在什么情况下应该使用 Uplift Tree，什么情况下应该使用 Meta-Learners?

你的答案:


"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 2: Uplift 决策树")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 500
    y = np.random.binomial(1, 0.3, n)
    t = np.random.binomial(1, 0.5, n)

    # 测试 2.1: Uplift 增益计算
    print("\n2.1 Uplift 增益计算")
    simple_uplift = calculate_simple_uplift(y, t)
    if simple_uplift is not None:
        print(f"  Simple Uplift: {simple_uplift:.4f}")
    else:
        print("  [未完成] 请完成 calculate_simple_uplift 函数")

    kl_gain = calculate_kl_divergence_gain(y, t)
    if kl_gain is not None:
        print(f"  KL Divergence: {kl_gain:.6f}")
    else:
        print("  [未完成] 请完成 calculate_kl_divergence_gain 函数")

    ed_gain = calculate_euclidean_distance_gain(y, t)
    if ed_gain is not None:
        print(f"  Euclidean Distance: {ed_gain:.6f}")
    else:
        print("  [未完成] 请完成 calculate_euclidean_distance_gain 函数")

    # 测试 2.2: 找最佳分裂点
    print("\n2.2 最佳分裂点搜索")
    X = np.random.randn(n)
    threshold, gain = find_best_split_threshold(X, y, t, criterion='KL')
    if threshold is not None:
        print(f"  最佳阈值: {threshold:.4f}")
        print(f"  分裂增益: {gain:.6f}")
    else:
        print("  [未完成] 请完成 find_best_split_threshold 函数")

    # 测试 2.3: 叶节点估计
    print("\n2.3 叶节点 Uplift 估计")
    uplift, stats = estimate_leaf_uplift(y, t)
    if uplift is not None and stats['treated_rate'] is not None:
        print(f"  Uplift: {uplift:.4f}")
        print(f"  处理组: {stats['n_treated']} 样本, 转化率 {stats['treated_rate']:.4f}")
        print(f"  控制组: {stats['n_control']} 样本, 转化率 {stats['control_rate']:.4f}")
    else:
        print("  [未完成] 请完成 estimate_leaf_uplift 函数")

    # 测试 2.4-2.5: Uplift Tree
    print("\n2.4-2.5 Simple Uplift Tree")
    print("-" * 60)
    tree, pred, true = test_uplift_tree()

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("思考题请在代码注释中回答")
    print("=" * 60)
