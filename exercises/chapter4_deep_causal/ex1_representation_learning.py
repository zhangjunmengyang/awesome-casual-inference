"""
练习 1: 表示学习基础

学习目标:
1. 理解为什么需要学习表示 (Representation Learning)
2. 掌握简单的神经网络特征提取
3. 理解处理组和对照组表示的差异
4. 为深度因果模型打下基础

完成所有 TODO 部分
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ==================== 练习 1.1: 手工特征 vs 学习表示 ====================

def generate_nonlinear_data(
    n: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成非线性数据: 原始特征无法直接捕获因果关系

    DGP (Data Generating Process):
    - X1, X2 ~ N(0, 1)
    - 真实有用特征: Phi1 = sin(X1), Phi2 = X1 * X2
    - T ~ Bernoulli(sigmoid(Phi1 + Phi2))
    - Y = 1 + 2*T + Phi1 + 0.5*Phi2 + noise

    TODO: 完成数据生成

    Returns:
        (X, T, Y) - 特征矩阵、处理、结果
    """
    np.random.seed(seed)

    # TODO: 生成原始特征 X1, X2
    X1 = None  # 你的代码
    X2 = None  # 你的代码

    # TODO: 计算有用特征
    # Phi1 = sin(X1), Phi2 = X1 * X2
    Phi1 = None  # 你的代码
    Phi2 = None  # 你的代码

    # TODO: 生成处理 T (通过 sigmoid 函数)
    # logit = Phi1 + 0.5*Phi2
    # P(T=1) = sigmoid(logit)
    logit = None  # 你的代码
    propensity = None  # 你的代码
    T = None  # 你的代码

    # TODO: 生成结果 Y
    # Y = 1 + 2*T + Phi1 + 0.5*Phi2 + noise (noise ~ N(0, 0.5))
    Y = None  # 你的代码

    X = np.column_stack([X1, X2])

    return X, T, Y


def naive_linear_estimation(X, T, Y):
    """
    使用线性回归估计 ATE (直接使用原始特征)

    TODO: 实现线性回归估计

    提示: Y = beta0 + beta1*T + beta2*X1 + beta3*X2 + epsilon
          beta1 就是 ATE 估计
    """
    from sklearn.linear_model import LinearRegression

    # TODO: 构造特征矩阵 [T, X1, X2]
    # TODO: 训练线性回归
    # TODO: 返回 T 的系数作为 ATE 估计

    # 你的代码
    pass


# ==================== 练习 1.2: 简单的表示学习 ====================

class SimpleRepresentation(nn.Module):
    """
    简单的表示学习网络

    X -> [Hidden Layer] -> Phi(X) (学习到的表示)
    """

    def __init__(self, input_dim: int, repr_dim: int = 10, hidden_dim: int = 20):
        super().__init__()

        # TODO: 定义网络层
        # 提示: Input -> Hidden (ReLU) -> Representation

        self.network = nn.Sequential(
            # 你的代码
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)


def train_representation(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    repr_dim: int = 10,
    n_epochs: int = 100
) -> nn.Module:
    """
    训练表示学习网络

    目标: 学习能预测 Y 的表示 Phi(X)

    TODO: 完成训练过程

    训练策略:
    1. Phi(X) -> [Linear] -> Y_pred
    2. 最小化 MSE(Y, Y_pred)
    3. 学到的 Phi(X) 可用于后续因果推断
    """

    # 转换为 Tensor
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    # 模型
    repr_model = SimpleRepresentation(input_dim=X.shape[1], repr_dim=repr_dim)

    # TODO: 定义预测头 (Phi(X) -> Y)
    prediction_head = None  # 你的代码

    # TODO: 定义优化器和损失函数
    optimizer = None  # 你的代码
    criterion = None  # 你的代码

    # TODO: 训练循环
    for epoch in range(n_epochs):
        # 你的代码
        pass

    return repr_model


# ==================== 练习 1.3: 可视化表示空间 ====================

def visualize_representation(
    repr_model: nn.Module,
    X: np.ndarray,
    T: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    可视化学到的表示空间

    TODO:
    1. 使用训练好的模型提取表示 Phi(X)
    2. 如果维度 > 2, 使用 PCA 降维到 2D
    3. 返回 2D 表示用于绘图

    Returns:
        (phi_2d, T) - 2D 表示和处理标签
    """

    repr_model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)

        # TODO: 提取表示
        phi = None  # 你的代码
        phi = phi.numpy()

    # TODO: 如果维度 > 2, 使用 PCA 降维
    if phi.shape[1] > 2:
        # 你的代码
        pass

    return phi, T


# ==================== 练习 1.4: 表示平衡性检查 ====================

def check_representation_balance(
    phi: np.ndarray,
    T: np.ndarray
) -> dict:
    """
    检查表示在处理组和对照组之间的平衡性

    平衡性指标:
    1. 标准化均值差 (SMD): |mean(Phi_T) - mean(Phi_C)| / std(Phi)
    2. MMD (Maximum Mean Discrepancy): ||mean(Phi_T) - mean(Phi_C)||^2

    TODO: 计算平衡性指标

    Returns:
        dict with balance metrics
    """

    # 分离处理组和对照组
    phi_treated = phi[T == 1]
    phi_control = phi[T == 0]

    # TODO: 计算 SMD (对每个维度)
    # SMD_j = |mean(Phi_T[:, j]) - mean(Phi_C[:, j])| / std(Phi[:, j])
    smd = None  # 你的代码

    # TODO: 计算 MMD (简化版: 欧氏距离)
    # MMD = ||mean(Phi_T) - mean(Phi_C)||^2
    mmd = None  # 你的代码

    return {
        'smd_mean': np.mean(smd) if smd is not None else None,
        'smd_max': np.max(smd) if smd is not None else None,
        'mmd': mmd
    }


# ==================== 练习 1.5: 为什么需要表示学习? ====================

def compare_linear_vs_learned(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    true_ate: float = 2.0
) -> dict:
    """
    对比线性方法和表示学习方法

    TODO:
    1. 使用线性回归 (原始特征) 估计 ATE
    2. 训练表示学习模型
    3. 使用学到的表示估计 ATE
    4. 比较两种方法的误差

    Returns:
        dict with comparison results
    """

    # TODO: 1. 线性估计
    linear_ate = naive_linear_estimation(X, T, Y)

    # TODO: 2. 表示学习
    repr_model = train_representation(X, T, Y)

    # TODO: 3. 提取表示并估计 ATE
    # 使用线性回归: Y ~ T + Phi(X)
    repr_model.eval()
    with torch.no_grad():
        phi = repr_model(torch.FloatTensor(X)).numpy()

    # 你的代码: 用 Phi 替代 X 进行线性回归
    learned_ate = None

    return {
        'linear_ate': linear_ate,
        'linear_error': abs(linear_ate - true_ate) if linear_ate else None,
        'learned_ate': learned_ate,
        'learned_error': abs(learned_ate - true_ate) if learned_ate else None,
        'true_ate': true_ate
    }


# ==================== 练习 1.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 为什么线性模型在非线性数据上表现不好?

你的答案:


2. 表示学习如何帮助因果推断?

你的答案:


3. 什么是 "表示平衡" (Representation Balance)? 为什么重要?

你的答案:


4. 在深度因果模型中，共享表示层的作用是什么?

你的答案:


5. 如果处理组和对照组的表示分布完全不重叠，会有什么问题?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 1: 表示学习基础")
    print("=" * 60)

    # 测试 1.1
    print("\n1.1 生成非线性数据")
    X, T, Y = generate_nonlinear_data()
    if X is not None and X[0, 0] is not None:
        print(f"  数据形状: X={X.shape}, T={T.shape}, Y={Y.shape}")
        print(f"  处理比例: {T.mean():.2%}")
        print(f"  平均结果: {Y.mean():.4f}")
    else:
        print("  [未完成] 请完成 generate_nonlinear_data 函数")

    # 测试 1.2
    print("\n1.2 线性 vs 表示学习")
    if X is not None:
        results = compare_linear_vs_learned(X, T, Y, true_ate=2.0)
        if results['linear_ate'] is not None:
            print(f"  真实 ATE: {results['true_ate']:.4f}")
            print(f"  线性估计: {results['linear_ate']:.4f} (误差: {results['linear_error']:.4f})")
            if results['learned_ate'] is not None:
                print(f"  学习估计: {results['learned_ate']:.4f} (误差: {results['learned_error']:.4f})")
        else:
            print("  [未完成] 请完成 compare_linear_vs_learned 函数")

    # 测试 1.3
    print("\n1.3 训练表示学习模型")
    if X is not None:
        repr_model = train_representation(X, T, Y, repr_dim=5, n_epochs=50)
        if repr_model is not None:
            print("  模型训练完成")

            # 测试 1.4
            print("\n1.4 表示可视化")
            phi_2d, _ = visualize_representation(repr_model, X, T)
            if phi_2d is not None:
                print(f"  表示维度: {phi_2d.shape}")
            else:
                print("  [未完成] 请完成 visualize_representation 函数")

            # 测试 1.5
            print("\n1.5 表示平衡性检查")
            repr_model.eval()
            with torch.no_grad():
                phi = repr_model(torch.FloatTensor(X)).numpy()
            balance = check_representation_balance(phi, T)
            if balance['mmd'] is not None:
                print(f"  平均 SMD: {balance['smd_mean']:.4f}")
                print(f"  最大 SMD: {balance['smd_max']:.4f}")
                print(f"  MMD: {balance['mmd']:.4f}")
                print("\n  提示: SMD < 0.1 表示良好平衡")
            else:
                print("  [未完成] 请完成 check_representation_balance 函数")
        else:
            print("  [未完成] 请完成 train_representation 函数")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 60)
    print("\n提示: 这是深度因果模型的基础练习")
    print("      理解表示学习对后续 TARNet、DragonNet 至关重要")
