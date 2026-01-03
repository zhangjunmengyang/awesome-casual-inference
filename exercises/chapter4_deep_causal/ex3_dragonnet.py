"""
练习 3: DragonNet

学习目标:
1. 理解倾向得分头的作用
2. 实现 DragonNet 架构
3. 掌握 DragonNet 的复合损失函数
4. 理解 Targeted Regularization

完成所有 TODO 部分
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict


# ==================== 练习 3.1: DragonNet 架构 ====================

class SimpleDragonNet(nn.Module):
    """
    简化版 DragonNet

    架构:
    X -> [Shared Representation] -> Phi(X)
                                      |
              +----------------------+----------------------+
              |                      |                      |
          [Head 0]              [Head 1]          [Propensity Head]
              |                      |                      |
            Y(0)                   Y(1)                   e(X)

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
        # Input -> Hidden (ELU) -> Representation (ELU)
        # 注意: DragonNet 使用 ELU 激活函数
        self.representation = nn.Sequential(
            # 你的代码
        )

        # TODO: 定义控制组输出头 (Y0)
        self.head0 = nn.Sequential(
            # 你的代码
        )

        # TODO: 定义处理组输出头 (Y1)
        self.head1 = nn.Sequential(
            # 你的代码
        )

        # TODO: 定义倾向得分头
        # Representation -> Hidden (ELU) -> 1 (Sigmoid)
        # 注意: 最后使用 Sigmoid 将输出限制在 [0, 1]
        self.propensity_head = nn.Sequential(
            # 你的代码
        )

        # TODO: 定义 epsilon 参数 (可学习的)
        # 用于 Targeted Regularization
        self.epsilon = None  # 你的代码

    def forward(self, x: torch.Tensor) -> Tuple:
        """
        前向传播

        TODO: 完成前向传播逻辑

        Returns:
            (y0_pred, y1_pred, propensity, epsilon, representation)
        """
        # TODO: 计算共享表示
        phi = None  # 你的代码

        # TODO: 通过三个头计算输出
        y0 = None  # 你的代码
        y1 = None  # 你的代码
        propensity = None  # 你的代码

        return y0, y1, propensity, self.epsilon, phi

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """预测 ITE"""
        y0, y1, _, _, _ = self.forward(x)
        return y1 - y0


# ==================== 练习 3.2: DragonNet 损失函数 ====================

def dragonnet_loss(
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    y0_pred: torch.Tensor,
    y1_pred: torch.Tensor,
    propensity: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    DragonNet 复合损失函数

    L = L_factual + alpha * L_propensity + beta * L_targeted

    其中:
    1. L_factual: 观测结果的预测损失
    2. L_propensity: 倾向得分的二分类交叉熵损失
    3. L_targeted: Targeted Regularization (TMLE 风格)

    TODO: 实现三个损失项

    Parameters:
    -----------
    alpha: 倾向得分损失权重
    beta: targeted regularization 权重

    Returns:
        dict with all loss components
    """

    # TODO: 1. Factual Loss
    # 根据 T 选择对应的预测值，计算 MSE
    y_pred = None  # 你的代码
    factual_loss = None  # 你的代码

    # TODO: 2. Propensity Loss
    # 二分类交叉熵: BCE(propensity, T)
    propensity_loss = None  # 你的代码

    # TODO: 3. Targeted Regularization
    # 这是 DragonNet 的创新!
    #
    # h = T / (e(X) + eps) - (1-T) / (1-e(X) + eps)
    # L_targeted = mean((Y - Y_pred - epsilon * h)^2)
    #
    # 提示:
    # - e(X) 是 propensity
    # - epsilon 是可学习的参数
    # - 添加小常数 1e-8 避免除零

    h = None  # 你的代码
    targeted_reg = None  # 你的代码

    # TODO: 总损失
    total_loss = None  # 你的代码

    return {
        'total': total_loss,
        'factual': factual_loss,
        'propensity': propensity_loss,
        'targeted': targeted_reg
    }


# ==================== 练习 3.3: 训练 DragonNet ====================

def train_dragonnet(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Tuple[SimpleDragonNet, dict]:
    """
    训练 DragonNet

    TODO: 完成训练过程

    Returns:
        (trained_model, training_history)
    """

    # TODO: 数据准备
    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)
    Y_tensor = torch.FloatTensor(Y)

    dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # TODO: 初始化模型
    model = SimpleDragonNet(input_dim=X.shape[1])

    # TODO: 定义优化器
    optimizer = None  # 你的代码

    # 训练历史
    history = {
        'total_loss': [],
        'factual_loss': [],
        'propensity_loss': [],
        'targeted_loss': []
    }

    # TODO: 训练循环
    for epoch in range(n_epochs):
        epoch_losses = {k: 0 for k in history.keys()}
        n_batches = 0

        for batch_x, batch_t, batch_y in dataloader:
            # TODO:
            # 1. 清零梯度
            # 2. 前向传播
            # 3. 计算 DragonNet 损失
            # 4. 反向传播
            # 5. 更新参数
            # 6. 记录各项损失

            # 你的代码
            pass

        # 记录平均损失
        for k in history.keys():
            history[k].append(epoch_losses[k] / n_batches if n_batches > 0 else 0)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Total Loss: {history['total_loss'][-1]:.4f}")

    return model, history


# ==================== 练习 3.4: 倾向得分评估 ====================

def evaluate_propensity_score(
    model: SimpleDragonNet,
    X: np.ndarray,
    T: np.ndarray
) -> dict:
    """
    评估倾向得分估计质量

    TODO: 计算倾向得分指标

    指标:
    1. AUC: 倾向得分预测处理的 ROC AUC
    2. 校准曲线: 将样本分组，比较预测倾向得分和实际处理率

    Returns:
        dict with propensity score metrics
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        _, _, propensity_pred, _, _ = model(X_tensor)
        propensity_pred = propensity_pred.numpy()

    # TODO: 计算 AUC
    auc = None  # 你的代码

    # TODO: 计算校准指标
    # 将样本按预测倾向得分分成 5 组
    # 对比每组的平均预测倾向得分和实际处理率

    n_bins = 5
    bins = np.linspace(0, 1, n_bins + 1)
    calibration = []

    for i in range(n_bins):
        # 你的代码
        pass

    return {
        'auc': auc,
        'calibration': calibration,
        'propensity_mean': propensity_pred.mean(),
        'propensity_std': propensity_pred.std()
    }


# ==================== 练习 3.5: 对比 TARNet vs DragonNet ====================

def compare_tarnet_dragonnet(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    Y0: np.ndarray,
    Y1: np.ndarray
) -> dict:
    """
    对比 TARNet 和 DragonNet 的性能

    TODO:
    1. 训练 TARNet (beta=0 的 DragonNet)
    2. 训练 DragonNet (beta>0)
    3. 比较两者的 PEHE 和 ATE 误差

    Returns:
        dict with comparison results
    """

    # TODO: 训练 TARNet (alpha=0, beta=0)
    print("训练 TARNet...")
    tarnet, _ = None  # 你的代码

    # TODO: 训练 DragonNet (alpha=1.0, beta=1.0)
    print("训练 DragonNet...")
    dragonnet, _ = None  # 你的代码

    # TODO: 评估两个模型
    def evaluate_model(model):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y0_pred, y1_pred, _, _, _ = model(X_tensor)
            y0_pred = y0_pred.numpy()
            y1_pred = y1_pred.numpy()

        ite_true = Y1 - Y0
        ite_pred = y1_pred - y0_pred
        pehe = np.sqrt(np.mean((ite_true - ite_pred) ** 2))

        ate_true = np.mean(Y1 - Y0)
        ate_pred = np.mean(y1_pred - y0_pred)
        ate_error = np.abs(ate_true - ate_pred)

        return {'pehe': pehe, 'ate_error': ate_error}

    tarnet_metrics = evaluate_model(tarnet) if tarnet else None
    dragonnet_metrics = evaluate_model(dragonnet) if dragonnet else None

    return {
        'tarnet': tarnet_metrics,
        'dragonnet': dragonnet_metrics
    }


# ==================== 练习 3.6: 数据生成 ====================

def generate_confounded_data(
    n: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成有混淆的数据

    强混淆场景: 倾向得分头应该能帮助改进估计

    TODO: 生成强混淆数据

    Returns:
        (X, T, Y, Y0, Y1)
    """
    np.random.seed(seed)

    # TODO: 生成特征
    X = np.random.randn(n, 5)

    # TODO: 强混淆的处理分配
    # 让倾向得分高度依赖于 X
    propensity = 1 / (1 + np.exp(-(
        1.0 * X[:, 0] +
        0.8 * X[:, 1] +
        0.6 * X[:, 2]
    )))
    T = None  # 你的代码

    # TODO: 潜在结果 (也依赖于 X)
    Y0 = None  # 你的代码
    Y1 = None  # 你的代码

    Y = np.where(T == 1, Y1, Y0)

    return X, T, Y, Y0, Y1


# ==================== 练习 3.7: 思考题 ====================

"""
思考题 (在代码注释中写下你的答案):

1. 倾向得分头在 DragonNet 中的作用是什么?

你的答案:


2. 为什么 Targeted Regularization 中需要 h = T/e(X) - (1-T)/(1-e(X))?
   这个公式的直觉是什么?

你的答案:


3. epsilon 参数的作用是什么? 为什么它是可学习的?

你的答案:


4. 什么时候 DragonNet 会比 TARNet 效果更好?

你的答案:


5. 如果倾向得分头的预测很差，会对整个模型有什么影响?

你的答案:


6. DragonNet 和传统的倾向得分方法(如 IPW)有什么区别?

你的答案:


7. 在实践中，如何选择 alpha 和 beta 这两个超参数?

你的答案:

"""


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("练习 3: DragonNet")
    print("=" * 60)

    # 测试 3.1
    print("\n3.1 生成混淆数据")
    X, T, Y, Y0, Y1 = generate_confounded_data()
    if X is not None and X[0, 0] is not None:
        print(f"  数据形状: X={X.shape}")
        print(f"  处理比例: {T.mean():.2%}")
        print(f"  真实 ATE: {np.mean(Y1 - Y0):.4f}")
    else:
        print("  [未完成] 请完成 generate_confounded_data 函数")

    # 测试 3.2
    print("\n3.2 测试 DragonNet 架构")
    if X is not None:
        model = SimpleDragonNet(input_dim=X.shape[1])
        X_sample = torch.FloatTensor(X[:5])
        try:
            y0, y1, prop, eps, phi = model(X_sample)
            if y0 is not None:
                print(f"  输出形状: Y0={y0.shape}, Y1={y1.shape}, Prop={prop.shape}")
                print(f"  Epsilon: {eps}")
            else:
                print("  [未完成] 请完成 SimpleDragonNet.forward 函数")
        except Exception as e:
            print(f"  [未完成] 请完成 SimpleDragonNet 定义: {e}")

    # 测试 3.3
    print("\n3.3 测试 DragonNet 损失")
    if X is not None:
        try:
            y_true = torch.FloatTensor([1.0, 2.0, 3.0])
            t_true = torch.FloatTensor([1.0, 0.0, 1.0])
            y0_pred = torch.FloatTensor([1.5, 2.0, 2.5])
            y1_pred = torch.FloatTensor([2.0, 2.5, 3.0])
            propensity = torch.FloatTensor([0.7, 0.3, 0.6])
            epsilon = torch.tensor(0.1)

            losses = dragonnet_loss(y_true, t_true, y0_pred, y1_pred,
                                   propensity, epsilon, alpha=1.0, beta=1.0)
            if losses['total'] is not None:
                print(f"  Total Loss: {losses['total'].item():.4f}")
                print(f"  Factual: {losses['factual'].item():.4f}")
                print(f"  Propensity: {losses['propensity'].item():.4f}")
                print(f"  Targeted: {losses['targeted'].item():.4f}")
            else:
                print("  [未完成] 请完成 dragonnet_loss 函数")
        except Exception as e:
            print(f"  [未完成] 请完成 dragonnet_loss 函数: {e}")

    # 测试 3.4
    print("\n3.4 训练 DragonNet")
    if X is not None:
        try:
            print("  开始训练...")
            model, history = train_dragonnet(
                X, T, Y,
                n_epochs=100,
                batch_size=64,
                alpha=1.0,
                beta=1.0
            )
            if model is not None and len(history['total_loss']) > 0:
                print(f"  训练完成! 最终损失: {history['total_loss'][-1]:.4f}")

                # 测试 3.5
                print("\n3.5 评估倾向得分")
                prop_metrics = evaluate_propensity_score(model, X, T)
                if prop_metrics['auc'] is not None:
                    print(f"  AUC: {prop_metrics['auc']:.4f}")
                    print(f"  倾向得分均值: {prop_metrics['propensity_mean']:.4f}")
                else:
                    print("  [未完成] 请完成 evaluate_propensity_score 函数")

                # 测试 3.6
                print("\n3.6 评估因果效应")
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    y0_pred, y1_pred, _, _, _ = model(X_tensor)
                    y0_pred = y0_pred.numpy()
                    y1_pred = y1_pred.numpy()

                ite_true = Y1 - Y0
                ite_pred = y1_pred - y0_pred
                pehe_val = np.sqrt(np.mean((ite_true - ite_pred) ** 2))

                ate_true = np.mean(Y1 - Y0)
                ate_pred = np.mean(y1_pred - y0_pred)

                print(f"  PEHE: {pehe_val:.4f}")
                print(f"  真实 ATE: {ate_true:.4f}")
                print(f"  预测 ATE: {ate_pred:.4f}")
                print(f"  ATE 误差: {abs(ate_true - ate_pred):.4f}")
            else:
                print("  [未完成] 请完成 train_dragonnet 函数")
        except Exception as e:
            print(f"  [未完成] 训练出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("完成所有 TODO 后，重新运行此脚本验证答案")
    print("=" * 60)
    print("\n提示: DragonNet 是目前最先进的深度因果模型之一")
    print("      掌握 DragonNet 让你能够处理复杂的因果推断问题")
    print("\n进阶挑战: 尝试实现 CEVAE (Causal Effect VAE)")
