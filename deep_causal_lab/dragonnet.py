"""
DragonNet

架构:
- 共享表示层
- 三个输出头: Y(0), Y(1), 倾向得分 e(X)
- 端到端训练，倾向得分头提供正则化

论文: Shi et al., "Adapting Neural Networks for the Estimation of Treatment Effects" (NeurIPS 2019)
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


class DragonNet(nn.Module):
    """
    DragonNet 模型

    架构:
    X -> [Shared Representation] -> Phi(X)
                                      |
              +----------------------+----------------------+
              |                      |                      |
          [Head 0]              [Head 1]          [Propensity Head]
              |                      |                      |
            Y(0)                   Y(1)                   e(X)

    关键创新:
    - 倾向得分头提供"targeted regularization"
    - 使用 epsilon 层实现端到端优化
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 200,
        repr_dim: int = 100,
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
                nn.ELU(),  # DragonNet 使用 ELU
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        repr_layers.append(nn.Linear(prev_dim, repr_dim))
        repr_layers.append(nn.ELU())
        self.representation = nn.Sequential(*repr_layers)

        # 控制组输出头 (Y0)
        self.head0 = self._build_head(repr_dim, hidden_dim, n_head_layers, dropout)

        # 处理组输出头 (Y1)
        self.head1 = self._build_head(repr_dim, hidden_dim, n_head_layers, dropout)

        # 倾向得分头
        self.propensity_head = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Epsilon 层 (用于 targeted regularization)
        self.epsilon = nn.Parameter(torch.zeros(1))

    def _build_head(self, input_dim, hidden_dim, n_layers, dropout):
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim // 2
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播

        Returns:
            (y0_pred, y1_pred, propensity, epsilon, representation)
        """
        phi = self.representation(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        propensity = self.propensity_head(phi).squeeze(-1)

        return y0, y1, propensity, self.epsilon, phi

    def predict_ite(self, x: torch.Tensor) -> torch.Tensor:
        """预测个体处理效应"""
        y0, y1, _, _, _ = self.forward(x)
        return y1 - y0


def dragonnet_loss(
    y_true: torch.Tensor,
    t_true: torch.Tensor,
    y0_pred: torch.Tensor,
    y1_pred: torch.Tensor,
    propensity: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0
) -> tuple:
    """
    DragonNet 损失函数

    L = L_factual + alpha * L_propensity + beta * L_targeted

    Parameters:
    -----------
    alpha: 倾向得分损失权重
    beta: targeted regularization 权重
    """
    # Factual loss
    y_pred = torch.where(t_true == 1, y1_pred, y0_pred)
    factual_loss = nn.MSELoss()(y_pred, y_true)

    # Propensity loss (交叉熵)
    propensity_loss = nn.BCELoss()(propensity, t_true)

    # Targeted regularization
    # 这是 DragonNet 的创新: 使用 TMLE 风格的正则化
    t_pred = propensity
    h = t_true / (t_pred + 1e-8) - (1 - t_true) / (1 - t_pred + 1e-8)

    # epsilon 是可学习的
    targeted_reg = torch.mean((y_true - y_pred - epsilon * h) ** 2)

    total_loss = factual_loss + alpha * propensity_loss + beta * targeted_reg

    return total_loss, factual_loss, propensity_loss, targeted_reg


def train_dragonnet(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int = 200,
    repr_dim: int = 100,
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 1.0,
    device: str = 'cpu'
) -> tuple:
    """训练 DragonNet"""

    # 数据准备
    X_tensor = torch.FloatTensor(X).to(device)
    T_tensor = torch.FloatTensor(T).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = DragonNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        repr_dim=repr_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练历史
    history = {
        'total_loss': [],
        'factual_loss': [],
        'propensity_loss': [],
        'targeted_loss': []
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {k: 0 for k in history.keys()}

        for batch_x, batch_t, batch_y in dataloader:
            optimizer.zero_grad()

            y0_pred, y1_pred, propensity, epsilon, _ = model(batch_x)

            total, factual, prop, targeted = dragonnet_loss(
                batch_y, batch_t, y0_pred, y1_pred, propensity, epsilon,
                alpha=alpha, beta=beta
            )

            total.backward()
            optimizer.step()

            epoch_losses['total_loss'] += total.item()
            epoch_losses['factual_loss'] += factual.item()
            epoch_losses['propensity_loss'] += prop.item()
            epoch_losses['targeted_loss'] += targeted.item()

        n_batches = len(dataloader)
        for k in history.keys():
            history[k].append(epoch_losses[k] / n_batches)

    return model, history


def visualize_dragonnet(
    n_samples: int,
    hidden_dim: int,
    n_epochs: int,
    alpha: float,
    beta: float
) -> tuple:
    """可视化 DragonNet 训练和效果"""

    # 生成数据
    X, T, Y, Y0_true, Y1_true = generate_ihdp_like_data(n_samples)

    # 训练模型
    model, history = train_dragonnet(
        X, T, Y,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        alpha=alpha,
        beta=beta
    )

    # 预测 - 确保 tensor 在正确的 device 上
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        Y0_pred, Y1_pred, propensity_pred, epsilon, phi = model(X_tensor)
        Y0_pred = Y0_pred.cpu().numpy()
        Y1_pred = Y1_pred.cpu().numpy()
        propensity_pred = propensity_pred.cpu().numpy()

    # 真实倾向得分 (近似)
    propensity_true = T.mean()  # 简化

    # 计算指标
    pehe_val = pehe(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_err = ate_error(Y0_true, Y1_true, Y0_pred, Y1_pred)
    ate_true = np.mean(Y1_true - Y0_true)
    ate_pred = np.mean(Y1_pred - Y0_pred)

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Training Losses',
            'True vs Predicted ITE',
            'Propensity Score Distribution',
            'DragonNet Architecture'
        )
    )

    # 1. 训练损失
    for name, color in [
        ('total_loss', '#2D9CDB'),
        ('factual_loss', '#27AE60'),
        ('propensity_loss', '#EB5757'),
        ('targeted_loss', '#9B59B6')
    ]:
        fig.add_trace(go.Scatter(
            y=history[name], mode='lines',
            name=name.replace('_', ' ').title(),
            line=dict(color=color)
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

    fig.add_trace(go.Scatter(
        x=[ite_true.min(), ite_true.max()],
        y=[ite_true.min(), ite_true.max()],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect'
    ), row=1, col=2)

    # 3. 倾向得分分布
    fig.add_trace(go.Histogram(
        x=propensity_pred[T == 0],
        name='Control',
        marker_color='blue',
        opacity=0.6,
        nbinsx=25
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=propensity_pred[T == 1],
        name='Treated',
        marker_color='red',
        opacity=0.6,
        nbinsx=25
    ), row=2, col=1)

    # 4. 架构图 (简化为文本)
    fig.add_annotation(
        x=0.5, y=0.5,
        text="See architecture diagram in summary",
        showarrow=False,
        font=dict(size=12),
        xref="x4 domain", yref="y4 domain"
    )

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='DragonNet Training & Evaluation',
        barmode='overlay'
    )

    # 摘要
    summary = f"""
### DragonNet 训练结果

| 指标 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 隐藏层维度 | {hidden_dim} |
| 训练轮数 | {n_epochs} |
| Propensity 权重 (alpha) | {alpha} |
| Targeted 权重 (beta) | {beta} |

### 因果效应估计

| 指标 | 值 |
|------|-----|
| 真实 ATE | {ate_true:.4f} |
| 预测 ATE | {ate_pred:.4f} |
| ATE 误差 | {ate_err:.4f} |
| PEHE | {pehe_val:.4f} |

### DragonNet 架构

```
              +---> [Head 0] ---------> Y(0)
              |
X ---> [Shared Repr] ---> Phi(X) ---> [Head 1] ---------> Y(1)
              |
              +---> [Propensity Head] ---> e(X)
                          |
                          v
                     [Epsilon Layer] ---> Targeted Reg
```

### 关键创新

1. **倾向得分头**: 端到端学习倾向得分，提供正则化
2. **Targeted Regularization**: 类似 TMLE 的无偏估计
3. **Epsilon 层**: 可学习的偏差校正

### 损失函数

$$L = L_{factual} + \\alpha \\cdot L_{propensity} + \\beta \\cdot L_{targeted}$$

其中:
- $L_{{factual}}$: 观测结果的预测损失
- $L_{{propensity}}$: 倾向得分的交叉熵损失
- $L_{{targeted}}$: TMLE 风格的目标正则化
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## DragonNet

DragonNet 是 TARNet 的扩展，添加了倾向得分头和 targeted regularization。

### 与 TARNet 的区别

| 特性 | TARNet | DragonNet |
|------|--------|-----------|
| 倾向得分 | 不估计 | 端到端估计 |
| 正则化 | IPM (可选) | Targeted Regularization |
| 理论保证 | 表示平衡 | TMLE 风格无偏性 |

### 为什么需要倾向得分头?

1. **隐式正则化**: 倾向得分头的梯度影响共享表示
2. **表示质量**: 迫使表示包含处理预测信息
3. **Targeted Reg**: 利用倾向得分进行偏差校正

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=500, maximum=5000, value=1000, step=500,
                    label="样本量"
                )
                hidden_dim = gr.Slider(
                    minimum=100, maximum=400, value=200, step=50,
                    label="隐藏层维度"
                )
                n_epochs = gr.Slider(
                    minimum=50, maximum=300, value=100, step=50,
                    label="训练轮数"
                )
                alpha = gr.Slider(
                    minimum=0, maximum=2, value=1.0, step=0.1,
                    label="Propensity Loss 权重 (alpha)"
                )
                beta = gr.Slider(
                    minimum=0, maximum=2, value=1.0, step=0.1,
                    label="Targeted Reg 权重 (beta)"
                )
                run_btn = gr.Button("训练模型", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="DragonNet 可视化")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=visualize_dragonnet,
            inputs=[n_samples, hidden_dim, n_epochs, alpha, beta],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 练习

完成 `exercises/chapter4_deep_causal/ex2_dragonnet.py` 中的练习。
        """)

    return None
