"""
练习 2: 因果图与 DAG

学习目标:
1. 理解因果图的基本概念
2. 识别混淆变量、中介变量、碰撞变量
3. 理解后门路径和 d-分离
4. 掌握调整公式

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
from typing import List, Set, Tuple


# ==================== 练习 2.1: 识别因果结构 ====================

def identify_structure(edges: List[Tuple[str, str]], node: str) -> str:
    """
    根据 DAG 边，识别给定节点的因果结构类型

    结构类型:
    - "confounder": 混淆变量 (有指向其他两个节点的边)
    - "mediator": 中介变量 (在因果链上)
    - "collider": 碰撞变量 (有两条边指向它)
    - "exposure": 处理变量
    - "outcome": 结果变量

    Args:
        edges: 边列表，如 [("X", "T"), ("X", "Y"), ("T", "Y")]
        node: 要识别的节点

    TODO: 完成结构识别逻辑

    Returns:
        结构类型字符串
    """
    # 计算入度和出度
    in_degree = sum(1 for edge in edges if edge[1] == node)
    out_degree = sum(1 for edge in edges if edge[0] == node)

    # TODO: 根据入度和出度判断结构类型
    # 提示:
    # - 碰撞变量: 入度 >= 2, 出度 = 0
    # - 混淆变量: 出度 >= 2
    # - 中介变量: 入度 >= 1 且 出度 >= 1

    # 你的代码
    pass


# ==================== 练习 2.2: 后门路径 ====================

def find_all_paths(
    edges: List[Tuple[str, str]],
    start: str,
    end: str,
    ignore_direction: bool = True
) -> List[List[str]]:
    """
    找出 DAG 中从 start 到 end 的所有路径

    Args:
        edges: 边列表
        start: 起始节点
        end: 终止节点
        ignore_direction: 是否忽略边的方向 (后门路径需要忽略)

    TODO: 使用 DFS 找出所有路径

    Returns:
        路径列表，每条路径是节点序列
    """
    # 构建邻接表
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        adj[u].append(v)
        if ignore_direction:
            if v not in adj:
                adj[v] = []
            adj[v].append(u)

    # TODO: 使用 DFS 找出所有路径
    # 提示: 使用递归，记录已访问节点避免循环

    paths = []

    def dfs(current, path, visited):
        # 你的代码
        pass

    dfs(start, [start], {start})
    return paths


def identify_backdoor_paths(
    edges: List[Tuple[str, str]],
    treatment: str,
    outcome: str
) -> List[List[str]]:
    """
    识别从 treatment 到 outcome 的后门路径

    后门路径: 从 T 到 Y 的路径，其中第一条边指向 T

    TODO: 从所有路径中筛选出后门路径

    Returns:
        后门路径列表
    """
    all_paths = find_all_paths(edges, treatment, outcome, ignore_direction=True)

    backdoor_paths = []
    for path in all_paths:
        if len(path) < 2:
            continue

        # TODO: 检查是否是后门路径
        # 提示: 检查第一条边是否指向 treatment
        # 即检查 (path[1], treatment) 是否在 edges 中

        # 你的代码
        pass

    return backdoor_paths


# ==================== 练习 2.3: 调整集 ====================

def is_valid_adjustment_set(
    edges: List[Tuple[str, str]],
    treatment: str,
    outcome: str,
    adjustment_set: Set[str]
) -> bool:
    """
    检查给定的调整集是否有效 (满足后门准则)

    后门准则要求调整集:
    1. 阻断所有后门路径
    2. 不包含 treatment 的后代

    TODO: 实现后门准则检查

    Returns:
        True 如果调整集有效
    """
    # TODO: 检查调整集是否满足后门准则

    # 步骤 1: 找出 treatment 的所有后代
    # 步骤 2: 检查调整集是否包含 treatment 的后代
    # 步骤 3: 检查调整集是否阻断所有后门路径

    # 你的代码
    pass


# ==================== 练习 2.4: 模拟混淆 ====================

def simulate_confounding_dag(
    n: int = 1000,
    seed: int = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    模拟经典混淆 DAG 的数据

    DAG: X -> T, X -> Y, T -> Y

    数据生成过程 (DGP):
    - X ~ N(0, 1)
    - T = 1 if X + noise > 0 else 0
    - Y = 1 + 2*T + 1.5*X + noise

    真实 ATE = 2

    TODO: 完成数据生成

    Returns:
        (DataFrame, dict with true parameters)
    """
    np.random.seed(seed)

    # TODO: 生成混淆变量 X
    X = None  # 你的代码

    # TODO: 生成处理 T (受 X 影响)
    T = None  # 你的代码

    # TODO: 生成结果 Y (受 T 和 X 影响)
    Y = None  # 你的代码

    df = pd.DataFrame({'X': X, 'T': T, 'Y': Y})

    params = {
        'true_ate': 2.0,
        'confounding_effect': 1.5
    }

    return df, params


def estimate_ate_with_adjustment(
    df: pd.DataFrame,
    adjustment_vars: List[str]
) -> float:
    """
    使用线性回归进行调整估计

    TODO: 实现调整估计
    """
    from sklearn.linear_model import LinearRegression

    # TODO: 使用线性回归估计 ATE
    # Y = b0 + b1*T + b2*X1 + b3*X2 + ...
    # b1 就是调整后的 ATE 估计

    # 你的代码
    pass


# ==================== 练习 2.5: 碰撞偏差 ====================

def simulate_collider_bias(
    n: int = 1000,
    seed: int = 42
) -> Tuple[pd.DataFrame, float, float]:
    """
    模拟碰撞偏差

    DAG: T -> C <- Y (T 和 Y 独立，C 是碰撞变量)

    DGP:
    - T ~ Bernoulli(0.5)
    - Y ~ N(0, 1)
    - C = 1 if T + Y > 0.5 else 0

    真实因果效应: T 对 Y 没有因果效应 (0)

    TODO:
    1. 生成数据
    2. 计算总体中 T 和 Y 的相关性
    3. 计算 C=1 子群中 T 和 Y 的相关性

    Returns:
        (DataFrame, overall_correlation, conditional_correlation)
    """
    np.random.seed(seed)

    # TODO: 生成数据
    T = None  # 你的代码
    Y = None  # 你的代码
    C = None  # 你的代码

    df = pd.DataFrame({'T': T, 'Y': Y, 'C': C})

    # TODO: 计算相关性
    overall_corr = None  # 总体相关性
    conditional_corr = None  # C=1 时的相关性

    return df, overall_corr, conditional_corr


# ==================== 练习 2.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 给定 DAG: X -> T -> Y <- U -> X
   - T 和 Y 之间有哪些路径?
   - 哪些是后门路径?
   - 最小调整集是什么?

你的答案:


2. 为什么控制碰撞变量会引入偏差?

你的答案:


3. 中介变量应该控制吗? 在什么情况下应该/不应该?

你的答案:


4. 如果存在未观测混淆 U，有什么方法可以识别因果效应?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("练习 2: 因果图与 DAG")
    print("=" * 50)

    # 测试 DAG
    test_edges = [("X", "T"), ("X", "Y"), ("T", "Y")]

    # 测试 2.1
    print("\n2.1 识别因果结构")
    for node in ["X", "T", "Y"]:
        structure = identify_structure(test_edges, node)
        if structure:
            print(f"  节点 {node}: {structure}")
        else:
            print("  [未完成] 请完成 identify_structure 函数")
            break

    # 测试 2.2
    print("\n2.2 后门路径")
    backdoor = identify_backdoor_paths(test_edges, "T", "Y")
    if backdoor is not None:
        print(f"  后门路径: {backdoor}")
    else:
        print("  [未完成] 请完成 identify_backdoor_paths 函数")

    # 测试 2.3
    print("\n2.3 调整集验证")
    valid = is_valid_adjustment_set(test_edges, "T", "Y", {"X"})
    if valid is not None:
        print(f"  调整集 {{X}} 有效: {valid}")
    else:
        print("  [未完成] 请完成 is_valid_adjustment_set 函数")

    # 测试 2.4
    print("\n2.4 混淆模拟")
    df, params = simulate_confounding_dag()
    if df is not None and df['X'].iloc[0] is not None:
        naive_ate = df[df['T'] == 1]['Y'].mean() - df[df['T'] == 0]['Y'].mean()
        adjusted_ate = estimate_ate_with_adjustment(df, ['X'])
        print(f"  真实 ATE: {params['true_ate']:.4f}")
        print(f"  朴素估计: {naive_ate:.4f}")
        if adjusted_ate:
            print(f"  调整估计: {adjusted_ate:.4f}")
    else:
        print("  [未完成] 请完成 simulate_confounding_dag 函数")

    # 测试 2.5
    print("\n2.5 碰撞偏差")
    df, overall, conditional = simulate_collider_bias()
    if overall is not None:
        print(f"  总体相关性 (应接近 0): {overall:.4f}")
        print(f"  C=1 条件相关性 (应为负): {conditional:.4f}")
    else:
        print("  [未完成] 请完成 simulate_collider_bias 函数")

    print("\n" + "=" * 50)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 50)
