"""
TARNet (Treatment-Agnostic Representation Network)

架构:
- 共享表示层: 学习处理无关的特征表示
- 两个输出头: 分别预测 Y(0) 和 Y(1)

论文: Shalit et al., "Estimating individual treatment effect: generalization bounds
      and algorithms" (ICML 2017)
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import generate_ihdp_like_data, pehe, ate_error


class TARNet(nn.Module):
    """
    TARNet 模型

    架构:
    X -> [Shared Representation] -> Phi(X)
                                      |
                    +----------------+----------------+
                    |                                 |
                [Head 0]                         [Head 1]
                    |                                 |
                  Y(0)                              Y(1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        repr_dim: int = 50,
        n_repr_layers: int = 3,
        n_head_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # 共享表示层
        repr_layers = []
        prev_dim = input_dim
        for _ in range(n_repr_layers):
            repr_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        repr_layers.append(nn.Linear(prev_dim, repr_dim))
        self.representation = nn.Sequential(*repr_layers)

        # 控制组输出头 (Y0)
        head0_layers = []
        prev_dim = repr_dim
        for _ in range(n_head_layers):
            head0_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        head0_layers.append(nn.Linear(prev_dim, 1))
        self.head0 = nn.Sequential(*head0_layers)

        # 处理组输出头 (Y1)
        head1_layers = []
        prev_dim = repr_dim
        for _ in range(n_head_layers):
            head1_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        head1_layers.append(nn.Linear(prev_dim, 1))
        self.head1 = nn.Sequential(*head1_layers)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播

        Returns:
            (y0_pred, y1_pred, representation)
        """
        phi = self.representation(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        return y0, y1, phi

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """预测个体处理效应"""
        y0, y1, _ = self.forward(x)
        return y1 - y0


def train_tarnet(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int = 100,
    repr_dim: int = 50,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    alpha: float = 0.0,  # IPM 正则化权重
    device: str = 'cpu'
) -> tuple:
    """
    训练 TARNet

    Parameters:
    -----------
    alpha: IPM 正则化权重 (用于表示平衡)
           0 = TARNet, >0 = CFR (Counterfactual Regression)
    """
    # 数据准备
    X_tensor = torch.FloatTensor(X).to(device)
    T_tensor = torch.FloatTensor(T).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = TARNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        repr_dim=repr_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 训练历史
    history = {'loss': [], 'factual_loss': [], 'ipm_loss': []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_factual = 0
        epoch_ipm = 0

        for batch_x, batch_t, batch_y in dataloader:
            optimizer.zero_grad()

            y0_pred, y1_pred, phi = model(batch_x)

            # Factual loss: 只对观测到的结果计算损失
            y_pred = torch.where(batch_t == 1, y1_pred, y0_pred)
            factual_loss = criterion(y_pred, batch_y)

            # IPM 正则化 (可选): 平衡处理组和控制组的表示
            if alpha > 0:
                phi_t = phi[batch_t == 1]
                phi_c = phi[batch_t == 0]
                if len(phi_t) > 0 and len(phi_c) > 0:
                    # 简化的 MMD (Maximum Mean Discrepancy)
                    ipm_loss = torch.mean(phi_t, dim=0).sub(torch.mean(phi_c, dim=0)).pow(2).sum()
                else:
                    ipm_loss = torch.tensor(0.0)
            else:
                ipm_loss = torch.tensor(0.0)

            loss = factual_loss + alpha * ipm_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_factual += factual_loss.item()
            epoch_ipm += ipm_loss.item()

        n_batches = len(dataloader)
        history['loss'].append(epoch_loss / n_batches)
        history['factual_loss'].append(epoch_factual / n_batches)
        history['ipm_loss'].append(epoch_ipm / n_batches)

    return model, history


def visualize_tarnet(
    n_samples: int,
    hidden_dim: int,
    n_epochs: int,
    alpha: float
) -> tuple:
    """可视化 TARNet 训练和效果"""

    # 生成数据
    X, T, Y, Y0_true, Y1_true = generate_ihdp_like_data(n_samples)

    # 训练模型
    model, history = train_tarnet(
        X, T, Y,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        alpha=alpha
    )

    # 预测
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        Y0_pred, Y1_pred, phi = model(X_tensor)
        Y0_pred = Y0_pred.numpy()
        Y1_pred = Y1_pred.numpy()
        phi = phi.numpy()

    # 计算指标
    pehe_val = pehe(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_err = ate_error(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_true = np.mean(Y1_true - Y0_true)
    ate_pred = np.mean(Y1_pred - Y0_pred)

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Loss',
            'True vs Predicted ITE',
            'Representation (t-SNE would be here)',
            'Y0/Y1 Predictions'
        )
    )

    # 1. 训练损失
    fig.add_trace(go.Scatter(
        y=history['loss'], mode='lines',
        name='Total Loss', line=dict(color='#2D9CDB')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=history['factual_loss'], mode='lines',
        name='Factual Loss', line=dict(color='#27AE60')
    ), row=1, col=1)

    if alpha > 0:
        fig.add_trace(go.Scatter(
            y=history['ipm_loss'], mode='lines',
            name='IPM Loss', line=dict(color='#EB5757')
        ), row=1, col=1)

    # 2. ITE 对比
    ite_true = Y1_true - Y0_true
    ite_pred = Y1_pred - Y0_pred

    sample_idx = np.random.choice(len(ite_true), min(500, len(ite_true)), replace=False)
    fig.add_trace(go.Scatter(
        x=ite_true[sample_idx], y=ite_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=5, opacity=0.5),
        name='ITE'
    ), row=1, col=2)

    # 对角线
    fig.add_trace(go.Scatter(
        x=[ite_true.min(), ite_true.max()],
        y=[ite_true.min(), ite_true.max()],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect'
    ), row=1, col=2)

    # 3. 表示空间 (简化: 用前两个维度)
    if phi.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=phi[T == 0, 0], y=phi[T == 0, 1],
            mode='markers',
            marker=dict(color='blue', size=4, opacity=0.5),
            name='Control'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=phi[T == 1, 0], y=phi[T == 1, 1],
            mode='markers',
            marker=dict(color='red', size=4, opacity=0.5),
            name='Treated'
        ), row=2, col=1)

    # 4. Y0/Y1 预测
    fig.add_trace(go.Scatter(
        x=Y0_true[sample_idx], y=Y0_pred[sample_idx],
        mode='markers',
        marker=dict(color='blue', size=4, opacity=0.5),
        name='Y0'
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=Y1_true[sample_idx], y=Y1_pred[sample_idx],
        mode='markers',
        marker=dict(color='red', size=4, opacity=0.5),
        name='Y1'
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='TARNet Training & Evaluation'
    )

    # 摘要
    summary = f"""
### TARNet 训练结果

| 指标 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 隐藏层维度 | {hidden_dim} |
| 训练轮数 | {n_epochs} |
| IPM 正则化 (alpha) | {alpha} |

### 因果效应估计

| 指标 | 值 |
|------|-----|
| 真实 ATE | {ate_true:.4f} |
| 预测 ATE | {ate_pred:.4f} |
| ATE 误差 | {ate_err:.4f} |
| PEHE (ITE误差) | {pehe_val:.4f} |

### 架构说明

```
Input (X) --> [Shared Repr] --> Phi(X) --> [Head 0] --> Y(0)
                                      |
                                      +--> [Head 1] --> Y(1)
```

**关键思想**:
- 共享表示: 学习对处理不敏感的特征
- 双头输出: 分别估计两个潜在结果
- 反事实推断: ITE = Y(1) - Y(0)
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## TARNet (Treatment-Agnostic Representation Network)

TARNet 是深度因果推断的基础模型，由 Shalit et al. (2017) 提出。

### 核心架构

```
              +---> [Head 0] ---> Y(0) (控制组预测)
              |
X ---> [Shared Representation] ---> Phi(X)
              |
              +---> [Head 1] ---> Y(1) (处理组预测)
```

### 关键思想

1. **共享表示**: 学习处理无关的特征表示 Phi(X)
2. **双头输出**: 分别预测 Y(0) 和 Y(1)
3. **Factual Loss**: 只在观测到的结果上计算损失

### CFR (Counterfactual Regression)

当添加 IPM 正则化 (alpha > 0) 时，TARNet 变成 CFR:
- IPM 约束处理组和控制组的表示分布相似
- 这有助于反事实预测的泛化

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=500, maximum=5000, value=1000, step=500,
                    label="样本量"
                )
                hidden_dim = gr.Slider(
                    minimum=50, maximum=200, value=100, step=25,
                    label="隐藏层维度"
                )
                n_epochs = gr.Slider(
                    minimum=50, maximum=300, value=100, step=50,
                    label="训练轮数"
                )
                alpha = gr.Slider(
                    minimum=0, maximum=1, value=0, step=0.1,
                    label="IPM 正则化权重 (0=TARNet, >0=CFR)"
                )
                run_btn = gr.Button("训练模型", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="TARNet 可视化")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=visualize_tarnet,
            inputs=[n_samples, hidden_dim, n_epochs, alpha],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 损失函数

**Factual Loss** (观测损失):
$$L_{factual} = \\sum_{i: T_i=1} (Y_i - \\hat{Y}_1(X_i))^2 + \\sum_{i: T_i=0} (Y_i - \\hat{Y}_0(X_i))^2$$

**IPM Loss** (分布匹配, 可选):
$$L_{IPM} = MMD(\\Phi(X_{T=1}), \\Phi(X_{T=0}))$$

**总损失**:
$$L = L_{factual} + \\alpha \\cdot L_{IPM}$$

### 练习

完成 `exercises/chapter4_deep_causal/ex1_tarnet.py` 中的练习。
        """)

    return None
