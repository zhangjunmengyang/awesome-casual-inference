"""
练习 2: TARNet (Treatment-Agnostic Representation Network)

学习目标:
1. 理解 TARNet 的架构设计
2. 实现简化版 TARNet
3. 理解 Factual Loss 的含义
4. 训练和评估 TARNet

完成所有 TODO 部分
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


# ==================== 练习 2.1: 理解 TARNet 架构 ====================

class SimpleTARNet(nn.Module):
    """
    简化版 TARNet

    架构:
    X -> [Shared Representation] -> Phi(X)
                                      |
                    +----------------+----------------+
                    |                                 |
                [Head 0]                         [Head 1]
                    |                                 |
                  Y(0)                              Y(1)

    TODO: 完成网络定义
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 50,
        repr_dim: int = 25
    ):
        super().__init__()

        # TODO: 定义共享表示层
        # Input -> Hidden (ReLU) -> Representation
        self.representation = nn.Sequential(
            # 你的代码
        )

        # TODO: 定义控制组输出头 (Y0)
        # Representation -> Hidden (ReLU) -> Output
        self.head0 = nn.Sequential(
            # 你的代码
        )

        # TODO: 定义处理组输出头 (Y1)
        # Representation -> Hidden (ReLU) -> Output
        self.head1 = nn.Sequential(
            # 你的代码
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        TODO: 完成前向传播逻辑

        Returns:
            (y0_pred, y1_pred, representation)
        """
        # TODO: 计算共享表示
        phi = None  # 你的代码

        # TODO: 通过两个头计算 Y(0) 和 Y(1)
        y0 = None  # 你的代码
        y1 = None  # 你的代码

        return y0, y1, phi

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测个体处理效应 ITE = Y(1) - Y(0)

        TODO: 完成 ITE 预测
        """
        # 你的代码
        pass


# ==================== 练习 2.2: Factual Loss ====================

def compute_factual_loss(
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    y0_pred: torch.Tensor,
    y1_pred: torch.Tensor
) -> torch.Tensor:
    """
    计算 Factual Loss

    关键思想: 只在观测到的结果上计算损失
    - 如果 T=1, 损失 = (Y - Y1_pred)^2
    - 如果 T=0, 损失 = (Y - Y0_pred)^2

    TODO: 实现 Factual Loss

    伪代码:
        y_pred = where(T == 1, Y1_pred, Y0_pred)
        loss = MSE(y_true, y_pred)

    Returns:
        scalar loss
    """
    # TODO: 根据处理状态选择预测值
    y_pred = None  # 你的代码

    # TODO: 计算 MSE
    loss = None  # 你的代码

    return loss


# ==================== 练习 2.3: 训练 TARNet ====================

def train_tarnet(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3
) -> Tuple[SimpleTARNet, dict]:
    """
    训练 TARNet

    TODO: 完成训练过程

    Returns:
        (trained_model, training_history)
    """

    # TODO: 转换为 PyTorch Tensor
    X_tensor = None  # 你的代码
    T_tensor = None  # 你的代码
    Y_tensor = None  # 你的代码

    # TODO: 创建 DataLoader
    dataset = None  # 你的代码
    dataloader = None  # 你的代码

    # TODO: 初始化模型
    model = SimpleTARNet(input_dim=X.shape[1])

    # TODO: 定义优化器
    optimizer = None  # 你的代码

    # 训练历史
    history = {'loss': []}

    # TODO: 训练循环
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0

        for batch_x, batch_t, batch_y in dataloader:
            # 你的代码:
            # 1. 清零梯度
            # 2. 前向传播
            # 3. 计算 Factual Loss
            # 4. 反向传播
            # 5. 更新参数
            pass

        # TODO: 记录损失
        history['loss'].append(epoch_loss / n_batches if n_batches > 0 else 0)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {history['loss'][-1]:.4f}")

    return model, history


# ==================== 练习 2.4: 评估 TARNet ====================

def evaluate_tarnet(
    model: SimpleTARNet,
    X: np.ndarray,
    Y0_true: np.ndarray,
    Y1_true: np.ndarray
) -> dict:
    """
    评估 TARNet 性能

    指标:
    1. PEHE (ITE 误差): sqrt(E[(ITE_true - ITE_pred)^2])
    2. ATE 误差: |ATE_true - ATE_pred|

    TODO: 完成评估逻辑

    Returns:
        dict with evaluation metrics
    """

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)

        # TODO: 预测 Y(0) 和 Y(1)
        y0_pred, y1_pred, _ = None  # 你的代码
        y0_pred = y0_pred.numpy()
        y1_pred = y1_pred.numpy()

    # TODO: 计算 PEHE
    # PEHE = sqrt(mean((ITE_true - ITE_pred)^2))
    ite_true = Y1_true - Y0_true
    ite_pred = y1_pred - y0_pred
    pehe = None  # 你的代码

    # TODO: 计算 ATE 误差
    ate_true = np.mean(Y1_true - Y0_true)
    ate_pred = np.mean(y1_pred - y0_pred)
    ate_error = None  # 你的代码

    return {
        'pehe': pehe,
        'ate_true': ate_true,
        'ate_pred': ate_pred,
        'ate_error': ate_error
    }


# ==================== 练习 2.5: 数据生成 ====================

def generate_simple_data(
    n: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成简单的半合成数据

    DGP:
    - X ~ N(0, I)
    - T ~ Bernoulli(sigmoid(0.5*X1 + 0.3*X2))
    - Y(0) = 1 + 0.5*X1 + 0.3*X2 + noise
    - Y(1) = Y(0) + (2 + 0.5*X1)  # 异质性效应

    TODO: 完成数据生成

    Returns:
        (X, T, Y, Y0, Y1)
    """
    np.random.seed(seed)

    # TODO: 生成特征 (5 维)
    X = None  # 你的代码

    # TODO: 生成处理
    propensity = None  # 你的代码
    T = None  # 你的代码

    # TODO: 生成潜在结果
    Y0 = None  # 你的代码
    Y1 = None  # 你的代码

    # TODO: 观测结果
    Y = None  # 你的代码

    return X, T, Y, Y0, Y1


# ==================== 练习 2.6: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 为什么 TARNet 需要两个独立的输出头?

你的答案:


2. Factual Loss 和普通的监督学习损失有什么区别?

你的答案:


3. 如果我们只训练一个头 (比如只训练 Head 1)，会有什么问题?

你的答案:


4. 共享表示层的作用是什么? 为什么不给两个头分别的特征提取器?

你的答案:


5. TARNet 如何预测反事实结果 (比如对于 T=1 的人预测 Y(0))?

你的答案:


6. 在什么情况下 TARNet 会比传统的 Meta-Learner (如 S/T-Learner) 更好?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 2: TARNet")
    print("=" * 60)

    # 测试 2.1
    print("\n2.1 生成数据")
    X, T, Y, Y0, Y1 = generate_simple_data()
    if X is not None and X[0, 0] is not None:
        print(f"  数据形状: X={X.shape}")
        print(f"  处理比例: {T.mean():.2%}")
        print(f"  真实 ATE: {np.mean(Y1 - Y0):.4f}")
    else:
        print("  [未完成] 请完成 generate_simple_data 函数")

    # 测试 2.2
    print("\n2.2 测试 TARNet 架构")
    if X is not None:
        model = SimpleTARNet(input_dim=X.shape[1])
        X_sample = torch.FloatTensor(X[:5])
        try:
            y0, y1, phi = model(X_sample)
            if y0 is not None:
                print(f"  模型输出形状: Y0={y0.shape}, Y1={y1.shape}, Phi={phi.shape}")
            else:
                print("  [未完成] 请完成 SimpleTARNet.forward 函数")
        except:
            print("  [未完成] 请完成 SimpleTARNet 定义")

    # 测试 2.3
    print("\n2.3 测试 Factual Loss")
    if X is not None:
        try:
            y_true = torch.FloatTensor([1.0, 2.0, 3.0])
            t_true = torch.FloatTensor([1.0, 0.0, 1.0])
            y0_pred = torch.FloatTensor([1.5, 2.0, 2.5])
            y1_pred = torch.FloatTensor([2.0, 2.5, 3.0])

            loss = compute_factual_loss(y_true, t_true, y0_pred, y1_pred)
            if loss is not None:
                print(f"  Factual Loss: {loss.item():.4f}")
            else:
                print("  [未完成] 请完成 compute_factual_loss 函数")
        except:
            print("  [未完成] 请完成 compute_factual_loss 函数")

    # 测试 2.4
    print("\n2.4 训练 TARNet")
    if X is not None:
        try:
            print("  开始训练...")
            model, history = train_tarnet(X, T, Y, n_epochs=100, batch_size=64)
            if model is not None and len(history['loss']) > 0:
                print(f"  训练完成! 最终损失: {history['loss'][-1]:.4f}")

                # 测试 2.5
                print("\n2.5 评估 TARNet")
                metrics = evaluate_tarnet(model, X, Y0, Y1)
                if metrics['pehe'] is not None:
                    print(f"  PEHE: {metrics['pehe']:.4f}")
                    print(f"  真实 ATE: {metrics['ate_true']:.4f}")
                    print(f"  预测 ATE: {metrics['ate_pred']:.4f}")
                    print(f"  ATE 误差: {metrics['ate_error']:.4f}")
                else:
                    print("  [未完成] 请完成 evaluate_tarnet 函数")
            else:
                print("  [未完成] 请完成 train_tarnet 函数")
        except Exception as e:
            print(f"  [未完成] 训练出错: {e}")
            print("  请完成 train_tarnet 函数")

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 60)
    print("\n提示: TARNet 是深度因果模型的基础")
    print("      掌握 TARNet 有助于理解 DragonNet 等更复杂模型")
