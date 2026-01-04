# Part 5 Deep Learning - Complete Solutions & Interview Guide

## 目录

1. [Part 5.1: Representation Learning - 完整解答](#part-51)
2. [Part 5.2: TARNet & DragonNet - 完整解答](#part-52)
3. [Part 5.3: CEVAE - 完整解答](#part-53)
4. [Part 5.4: GANITE - 完整解答](#part-54)
5. [Part 5.5: VCNet - 完整解答](#part-55)
6. [综合面试题库](#interview-questions)
7. [从零实现示例](#from-scratch)

---

## Part 5.1: Representation Learning

### TODO 完整答案

#### 练习 1.1: 数据生成

```python
def generate_nonlinear_data(n: int = 1000, seed: int = 42):
    np.random.seed(seed)

    # 答案: 生成原始特征
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)

    # 答案: 有用特征
    Phi1 = np.sin(X1)
    Phi2 = X1 * X2

    # 答案: 处理分配
    logit = Phi1 + 0.5 * Phi2
    propensity = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, propensity, n)

    # 答案: 结果生成
    noise = np.random.randn(n) * 0.5
    Y = 1 + 2*T + Phi1 + 0.5*Phi2 + noise

    X = np.column_stack([X1, X2])
    return X, T, Y
```

#### 练习 1.2: 表示学习网络

```python
class SimpleRepresentation(nn.Module):
    def __init__(self, input_dim: int, repr_dim: int = 10, hidden_dim: int = 20):
        super().__init__()

        # 答案: 定义网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, x):
        return self.network(x)


def train_representation(X, T, Y, repr_dim=10, n_epochs=100):
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y).unsqueeze(1)

    repr_model = SimpleRepresentation(input_dim=X.shape[1], repr_dim=repr_dim)

    # 答案: 预测头
    prediction_head = nn.Linear(repr_dim, 1)

    # 答案: 优化器
    params = list(repr_model.parameters()) + list(prediction_head.parameters())
    optimizer = optim.Adam(params, lr=0.01)
    criterion = nn.MSELoss()

    # 答案: 训练循环
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        phi = repr_model(X_tensor)
        y_pred = prediction_head(phi)
        loss = criterion(y_pred, Y_tensor)
        loss.backward()
        optimizer.step()

    return repr_model
```

#### 练习 1.3-1.4: 可视化与平衡检查

```python
def visualize_representation(repr_model, X, T):
    repr_model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        # 答案: 提取表示
        phi = repr_model(X_tensor).numpy()

    # 答案: PCA 降维
    if phi.shape[1] > 2:
        pca = PCA(n_components=2)
        phi = pca.fit_transform(phi)

    return phi, T


def check_representation_balance(phi, T):
    phi_treated = phi[T == 1]
    phi_control = phi[T == 0]

    mean_t = phi_treated.mean(axis=0)
    mean_c = phi_control.mean(axis=0)
    std_all = phi.std(axis=0) + 1e-8

    # 答案: SMD
    smd = np.abs(mean_t - mean_c) / std_all

    # 答案: MMD (简化版)
    mmd = np.sum((mean_t - mean_c)**2)

    return {
        'smd_mean': np.mean(smd),
        'smd_max': np.max(smd),
        'mmd': mmd
    }
```

#### 练习 6.1: MMD 实现

```python
def rbf_kernel(X, Y, gamma=1.0):
    # 答案: 欧氏距离平方
    XX = np.sum(X**2, axis=1).reshape(-1, 1)
    YY = np.sum(Y**2, axis=1).reshape(1, -1)
    XY = X @ Y.T

    dist_sq = XX + YY - 2 * XY

    # 答案: 高斯核
    K = np.exp(-gamma * dist_sq)

    return K


def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    n = X.shape[0]
    m = Y.shape[0]

    # 答案: 核矩阵
    K_XX = rbf_kernel(X, X, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)

    # 答案: MMD^2 (无偏估计)
    term1 = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
    term2 = K_XY.sum() / (n * m)
    term3 = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))

    mmd_sq = term1 - 2 * term2 + term3

    return np.sqrt(max(mmd_sq, 0))
```

### 数学推导

#### 1. IPM 定义与性质

**定义**:
$$\text{IPM}_{\mathcal{F}}(P, Q) = \sup_{f \in \mathcal{F}} \left| \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{x \sim Q}[f(x)] \right|$$

**性质**:
1. **非负性**: $\text{IPM}(P, Q) \geq 0$
2. **对称性**: $\text{IPM}(P, Q) = \text{IPM}(Q, P)$
3. **三角不等式**: $\text{IPM}(P, R) \leq \text{IPM}(P, Q) + \text{IPM}(Q, R)$

#### 2. MMD 核技巧推导

**原始形式**:
$$\text{MMD}^2(P, Q) = \left\| \mathbb{E}_{x \sim P}[\phi(x)] - \mathbb{E}_{y \sim Q}[\phi(y)] \right\|^2$$

**展开**:
$$\begin{align}
\text{MMD}^2(P, Q) &= \left\langle \mathbb{E}_P[\phi(x)], \mathbb{E}_P[\phi(x)] \right\rangle \\
&\quad - 2 \left\langle \mathbb{E}_P[\phi(x)], \mathbb{E}_Q[\phi(y)] \right\rangle \\
&\quad + \left\langle \mathbb{E}_Q[\phi(y)], \mathbb{E}_Q[\phi(y)] \right\rangle
\end{align}$$

**应用核技巧** ($k(x,y) = \langle \phi(x), \phi(y) \rangle$):
$$\begin{align}
\text{MMD}^2(P, Q) &= \mathbb{E}_{x, x' \sim P}[k(x, x')] \\
&\quad - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)] \\
&\quad + \mathbb{E}_{y, y' \sim Q}[k(y, y')]
\end{align}$$

**无偏估计器**:
$$\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i \neq i'} k(x_i, x_{i'}) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j) + \frac{1}{m(m-1)}\sum_{j \neq j'} k(y_j, y_{j'})$$

#### 3. Wasserstein 距离的对偶形式

**Kantorovich-Rubinstein 定理**:
$$W_1(P, Q) = \sup_{\|f\|_L \leq 1} \left| \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{y \sim Q}[f(y)] \right|$$

其中 $\|f\|_L \leq 1$ 表示 $f$ 是 1-Lipschitz 函数。

**证明概要**:
1. 原始定义: $W_1(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$
2. 引入对偶变量 (Lagrange 对偶)
3. 应用 Fenchel-Rockafellar 对偶定理
4. 得到对偶形式

**Sliced Wasserstein**:
$$\text{SWD}(P, Q) = \int_{\mathbb{S}^{d-1}} W_1(P_\theta, Q_\theta) d\theta$$

其中 $P_\theta$ 是 $P$ 在方向 $\theta$ 上的一维投影。

#### 4. 因果推断误差界

**定理 (Shalit et al., 2017)**:

假设表示函数 $\Phi: \mathcal{X} \to \mathcal{R}$ 和假设函数 $h_0, h_1: \mathcal{R} \to \mathbb{R}$。定义:
- 预测误差: $\epsilon_h = \mathbb{E}_{(x,t,y) \sim P} [(y - h_t(\Phi(x)))^2]$
- 表示平衡: $\text{IPM}(P_\Phi^{t=0}, P_\Phi^{t=1})$

则 **ATE 估计误差上界**:
$$\epsilon_{\text{ATE}} \leq \sqrt{\epsilon_h} + \lambda \cdot \text{IPM}(P_\Phi^{t=0}, P_\Phi^{t=1})$$

其中 $\lambda$ 是假设函数的 Lipschitz 常数。

**推论**:
1. 要减少 ATE 估计误差，需要同时:
   - 减少预测误差 $\epsilon_h$
   - 减少表示不平衡 IPM
2. 最优权衡由参数 $\alpha$ 控制

---

## Part 5.2: TARNet & DragonNet

### 数学推导

#### 1. Factual Loss 推导

**问题**: 对于每个样本，我们只观测到一个结果。

**Factual Loss 定义**:
$$\mathcal{L}_{\text{factual}} = \frac{1}{N}\sum_{i=1}^{N} (Y_i - \hat{Y}_i^{\text{factual}})^2$$

其中:
$$\hat{Y}_i^{\text{factual}} = \begin{cases}
\hat{Y}_i(1) & \text{if } T_i = 1 \\
\hat{Y}_i(0) & \text{if } T_i = 0
\end{cases}$$

**简洁形式**:
$$\hat{Y}_i^{\text{factual}} = T_i \cdot \hat{Y}_i(1) + (1 - T_i) \cdot \hat{Y}_i(0)$$

**为什么有效?**
- 使用观测到的结果进行监督学习
- 同时学习 $\mu_0(x)$ 和 $\mu_1(x)$
- 反事实预测通过共享表示泛化

#### 2. Targeted Regularization 理论推导

**半参数效率理论** (Semiparametric Efficiency Theory):

在因果推断中，efficient influence function (EIF) 为:
$$\psi(X, T, Y) = h(X) + \frac{T}{e(X)}(Y - \mu_1(X)) - \frac{1-T}{1-e(X)}(Y - \mu_0(X))$$

其中:
- $h(X) = \mu_1(X) - \mu_0(X)$ 是 CATE
- $e(X) = P(T=1|X)$ 是倾向得分

**Targeted Regularization** 基于 TMLE (Targeted Maximum Likelihood Estimation):

$$\mathcal{L}_{\text{targeted}} = \frac{1}{N}\sum_{i=1}^{N} \left(Y_i - \hat{Y}_i - \epsilon \cdot h_i\right)^2$$

其中:
$$h_i = \frac{T_i}{\hat{e}(X_i)} - \frac{1-T_i}{1-\hat{e}(X_i)}$$

**直觉**:
- $h_i$ 是样本的"权重"
- 倾向得分低的样本权重高（更重要）
- $\epsilon$ 是可学习的调整参数

#### 3. 倾向得分正则化的作用

**DragonNet 损失**:
$$\mathcal{L} = \mathcal{L}_{\text{factual}} + \alpha \cdot \mathcal{L}_{\text{propensity}} + \beta \cdot \mathcal{L}_{\text{targeted}}$$

**倾向得分损失**:
$$\mathcal{L}_{\text{propensity}} = -\frac{1}{N}\sum_{i=1}^{N} [T_i \log \hat{e}_i + (1-T_i) \log (1-\hat{e}_i)]$$

**作用**:
1. **识别混淆因子**: 强迫表示层学习与处理分配相关的特征
2. **正则化效果**: 防止过拟合到特定处理组
3. **双重鲁棒性**: 结合结果回归和倾向得分模型的优势

### 从零实现: TARNet

```python
import torch
import torch.nn as nn

class TARNet(nn.Module):
    """TARNet 从零实现"""

    def __init__(self, input_dim, hidden_dim=64, repr_dim=32):
        super().__init__()

        # 共享表示层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
            nn.ReLU()
        )

        # 对照组头 (T=0)
        self.head_0 = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 处理组头 (T=1)
        self.head_1 = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, t=None):
        phi = self.shared(x)
        y0 = self.head_0(phi).squeeze()
        y1 = self.head_1(phi).squeeze()

        if t is not None:
            y = torch.where(t == 1, y1, y0)
            return y, y0, y1, phi
        else:
            return y0, y1, phi

    def predict_ite(self, x):
        y0, y1, _ = self.forward(x)
        return y1 - y0


def train_tarnet(model, X, T, Y, n_epochs=200, lr=1e-3):
    """训练 TARNet"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)
    Y_tensor = torch.FloatTensor(Y)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Factual loss
        y_pred, y0, y1, phi = model(X_tensor, T_tensor)
        loss = criterion(y_pred, Y_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model
```

### 从零实现: DragonNet

```python
class DragonNet(nn.Module):
    """DragonNet 从零实现"""

    def __init__(self, input_dim, hidden_dim=64, repr_dim=32):
        super().__init__()

        # 共享表示层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, repr_dim),
            nn.ELU()
        )

        # 三个头
        self.head_0 = nn.Linear(repr_dim, 1)  # Y(0)
        self.head_1 = nn.Linear(repr_dim, 1)  # Y(1)
        self.head_prop = nn.Linear(repr_dim, 1)  # 倾向得分

        # Epsilon 参数
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x, t=None):
        phi = self.shared(x)
        y0 = self.head_0(phi).squeeze()
        y1 = self.head_1(phi).squeeze()
        prop = torch.sigmoid(self.head_prop(phi).squeeze())

        if t is not None:
            y = torch.where(t == 1, y1, y0)
            return y, y0, y1, prop, phi
        else:
            return y0, y1, prop, phi


def dragonnet_loss(y_true, t_true, y_pred, y0, y1, prop, epsilon, alpha=1.0, beta=1.0):
    """DragonNet 复合损失"""

    # 1. Factual loss
    factual_loss = torch.mean((y_true - y_pred) ** 2)

    # 2. Propensity loss
    eps = 1e-8
    prop_loss = -torch.mean(
        t_true * torch.log(prop + eps) +
        (1 - t_true) * torch.log(1 - prop + eps)
    )

    # 3. Targeted regularization
    h = t_true / (prop + eps) - (1 - t_true) / (1 - prop + eps)
    targeted_loss = torch.mean((y_true - y_pred - epsilon * h) ** 2)

    # 总损失
    total_loss = factual_loss + alpha * prop_loss + beta * targeted_loss

    return total_loss, factual_loss, prop_loss, targeted_loss


def train_dragonnet(model, X, T, Y, alpha=1.0, beta=1.0, n_epochs=200, lr=1e-3):
    """训练 DragonNet"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)
    Y_tensor = torch.FloatTensor(Y)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        y_pred, y0, y1, prop, phi = model(X_tensor, T_tensor)

        total_loss, factual_loss, prop_loss, targeted_loss = dragonnet_loss(
            Y_tensor, T_tensor, y_pred, y0, y1, prop, model.epsilon, alpha, beta
        )

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Total={total_loss.item():.4f}, "
                  f"Factual={factual_loss.item():.4f}, "
                  f"Prop={prop_loss.item():.4f}, "
                  f"Targeted={targeted_loss.item():.4f}, "
                  f"Epsilon={model.epsilon.item():.4f}")

    return model
```

---

## Part 5.3: CEVAE

### 数学推导

#### 1. ELBO 完整推导

**目标**: 最大化边际对数似然 $\log p_\theta(X, T, Y)$

**引入变分分布** $q_\phi(Z | X, T, Y)$:

$$\begin{align}
\log p_\theta(X, T, Y) &= \log \int p_\theta(X, T, Y, Z) dZ \\
&= \log \int \frac{p_\theta(X, T, Y, Z)}{q_\phi(Z | X, T, Y)} q_\phi(Z | X, T, Y) dZ \\
&= \log \mathbb{E}_{q_\phi} \left[\frac{p_\theta(X, T, Y, Z)}{q_\phi(Z | X, T, Y)}\right]
\end{align}$$

**应用 Jensen 不等式** ($\log \mathbb{E}[·] \geq \mathbb{E}[\log ·]$):

$$\begin{align}
\log p_\theta(X, T, Y) &\geq \mathbb{E}_{q_\phi(Z|X,T,Y)} \left[\log \frac{p_\theta(X, T, Y, Z)}{q_\phi(Z | X, T, Y)}\right] \\
&= \mathbb{E}_{q_\phi} [\log p_\theta(X, T, Y, Z)] - \mathbb{E}_{q_\phi} [\log q_\phi(Z | X, T, Y)] \\
&\equiv \mathcal{L}_{\text{ELBO}}
\end{align}$$

**展开 ELBO**:

$$\begin{align}
\mathcal{L}_{\text{ELBO}} &= \mathbb{E}_{q_\phi(Z|X,T,Y)} [\log p_\theta(X | Z) + \log p_\theta(T | X, Z) + \log p_\theta(Y | T, X, Z) + \log p(Z)] \\
&\quad - \mathbb{E}_{q_\phi(Z|X,T,Y)} [\log q_\phi(Z | X, T, Y)] \\
&= \underbrace{\mathbb{E}_{q_\phi} [\log p_\theta(X | Z)]}_{\text{X 重构}} + \underbrace{\mathbb{E}_{q_\phi} [\log p_\theta(T | X, Z)]}_{\text{T 重构}} + \underbrace{\mathbb{E}_{q_\phi} [\log p_\theta(Y | T, X, Z)]}_{\text{Y 重构}} \\
&\quad - \underbrace{\text{KL}(q_\phi(Z | X, T, Y) \| p(Z))}_{\text{KL 散度}}
\end{align}$$

#### 2. 重参数化技巧 (Reparameterization Trick)

**问题**: 采样操作不可微，无法反向传播。

假设 $Z \sim q_\phi(Z | X) = \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$

**朴素采样** (不可微):
$$Z = \text{sample from } \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$$

**重参数化** (可微):
$$\begin{align}
\epsilon &\sim \mathcal{N}(0, I) \\
Z &= \mu_\phi(X) + \sigma_\phi(X) \odot \epsilon
\end{align}$$

现在 $Z$ 关于 $\mu_\phi$ 和 $\sigma_\phi$ 可微！

**梯度计算**:
$$\begin{align}
\nabla_\phi \mathbb{E}_{q_\phi(Z|X)}[f(Z)] &= \nabla_\phi \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[f(\mu_\phi(X) + \sigma_\phi(X) \odot \epsilon)] \\
&= \mathbb{E}_{\epsilon}[\nabla_\phi f(\mu_\phi(X) + \sigma_\phi(X) \odot \epsilon)]
\end{align}$$

#### 3. KL 散度的解析形式

对于 $q_\phi(Z | X) = \mathcal{N}(\mu, \Sigma)$ 和 $p(Z) = \mathcal{N}(0, I)$:

$$\begin{align}
\text{KL}(q_\phi \| p) &= \mathbb{E}_{Z \sim q_\phi} \left[\log \frac{q_\phi(Z|X)}{p(Z)}\right] \\
&= \mathbb{E}_Z \left[\log q_\phi(Z|X) - \log p(Z)\right] \\
&= -\frac{1}{2} \sum_{j=1}^{d_z} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
\end{align}$$

**简化**:
$$\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{j=1}^{d_z} \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

### 从零实现: CEVAE

```python
class CEVAE(nn.Module):
    """CEVAE 从零实现"""

    def __init__(self, x_dim, latent_dim=20, hidden_dim=200):
        super().__init__()

        self.latent_dim = latent_dim

        # 编码器 q(Z | X, T, Y)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器 p(X | Z)
        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

        # 解码器 p(T | X, Z)
        self.decoder_t = nn.Sequential(
            nn.Linear(x_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 解码器 p(Y | T, X, Z)
        self.decoder_y = nn.Sequential(
            nn.Linear(x_dim + latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, x, t, y):
        """编码: q(Z | X, T, Y)"""
        inputs = torch.cat([x, t.unsqueeze(1), y.unsqueeze(1)], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x, t):
        """解码"""
        x_recon = self.decoder_x(z)
        t_recon = self.decoder_t(torch.cat([x, z], dim=1)).squeeze()
        y_recon = self.decoder_y(torch.cat([x, t.unsqueeze(1), z], dim=1)).squeeze()
        return x_recon, t_recon, y_recon

    def forward(self, x, t, y):
        """完整前向传播"""
        mu, logvar = self.encode(x, t, y)
        z = self.reparameterize(mu, logvar)
        x_recon, t_recon, y_recon = self.decode(z, x, t)

        return {
            'x_recon': x_recon,
            't_recon': t_recon,
            'y_recon': y_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def predict_counterfactual(self, x, t, y, n_samples=100):
        """预测反事实"""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x, t, y)

            y0_samples = []
            y1_samples = []

            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)

                t0 = torch.zeros_like(t)
                t1 = torch.ones_like(t)

                _, _, y0 = self.decode(z, x, t0)
                _, _, y1 = self.decode(z, x, t1)

                y0_samples.append(y0)
                y1_samples.append(y1)

            y0_pred = torch.stack(y0_samples).mean(dim=0)
            y1_pred = torch.stack(y1_samples).mean(dim=0)

        return y0_pred, y1_pred


def cevae_loss(outputs, x, t, y, beta=1.0):
    """CEVAE 损失函数"""

    # X 重构损失
    x_recon_loss = torch.mean((outputs['x_recon'] - x) ** 2)

    # T 重构损失 (BCE)
    eps = 1e-8
    t_recon_loss = -torch.mean(
        t * torch.log(outputs['t_recon'] + eps) +
        (1 - t) * torch.log(1 - outputs['t_recon'] + eps)
    )

    # Y 重构损失
    y_recon_loss = torch.mean((outputs['y_recon'] - y) ** 2)

    # KL 散度
    kl_loss = -0.5 * torch.sum(
        1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
    ) / x.size(0)

    # 总损失
    total_loss = x_recon_loss + t_recon_loss + y_recon_loss + beta * kl_loss

    return total_loss, x_recon_loss, t_recon_loss, y_recon_loss, kl_loss
```

---

## 综合面试题库

### 深度因果推断面试题

#### 理论题

**1. 为什么 TARNet 需要共享表示层？**

**答案**:
- **样本效率**: 两组共享特征提取器，增加有效训练样本
- **泛化能力**: 共享参数防止过拟合到特定组
- **反事实预测**: 通过共享表示泛化到未观测的反事实
- **表示平衡**: 共享层使两组在表示空间中更接近

**2. Factual Loss 与普通监督学习有什么区别？**

**答案**:
- **Factual Loss**: $\mathcal{L} = \sum_i (Y_i - [T_i \hat{Y}_i(1) + (1-T_i) \hat{Y}_i(0)])^2$
  - 每个样本只用观测到的结果
  - 同时训练两个头 ($\mu_0$ 和 $\mu_1$)
  - 损失"选择"对应的头

- **普通监督学习**: $\mathcal{L} = \sum_i (Y_i - f(X_i))^2$
  - 单一预测函数
  - 无反事实概念

**3. DragonNet 的倾向得分头在模型中起什么作用？**

**答案**:
1. **识别混淆**: 强迫表示层学习与处理分配相关的特征（即混淆因子）
2. **正则化**: 防止过拟合，提高泛化
3. **Targeted Regularization**: 配合 epsilon 参数实现双重鲁棒估计
4. **理论保证**: 基于半参数效率理论

**4. CEVAE 如何处理隐混淆？**

**答案**:
- **建模隐变量**: 用 VAE 学习隐变量 $Z$ 的分布
- **代理变量假设**: 观测到的 $X$ 包含关于 $Z$ 的信息
- **条件独立**: 给定 $Z$，处理和结果条件独立
- **边缘化**: 通过积分 $\int p(Y|T,X,Z)p(Z|X) dZ$ 获得因果效应

**5. GANITE 为什么用 GAN 而不是 VAE 生成反事实？**

**答案**:
- **分布质量**: GAN 生成的样本更sharp，VAE 倾向于模糊
- **多模态**: GAN 可以捕获多模态分布
- **对抗训练**: 判别器帮助生成更真实的反事实
- **缺点**: GAN 训练不稳定，VAE 有理论保证（ELBO）

**6. VCNet 如何处理连续处理？**

**答案**:
- **变系数网络**: $W(t) \cdot \phi(X)$，权重随处理强度变化
- **样条基函数**: 保证剂量-响应曲线光滑
- **广义倾向得分**: $e(t|X) = f_{T|X}(t|x)$ 概率密度
- **边际处理效应**: $\frac{\partial \mu(t,x)}{\partial t}$

#### 编程题

**题目 1: 实现 TARNet 的 Factual Loss**

```python
def factual_loss(y_true, t_true, y0_pred, y1_pred):
    """
    实现 Factual Loss

    Args:
        y_true: 真实结果 (N,)
        t_true: 处理标签 (N,)
        y0_pred: Y(0) 预测 (N,)
        y1_pred: Y(1) 预测 (N,)

    Returns:
        loss: Factual Loss
    """
    # 答案:
    y_pred = torch.where(t_true == 1, y1_pred, y0_pred)
    loss = torch.mean((y_true - y_pred) ** 2)
    return loss
```

**题目 2: 实现 VAE 的重参数化**

```python
def reparameterize(mu, logvar):
    """
    实现 VAE 重参数化技巧

    Args:
        mu: 均值 (batch, latent_dim)
        logvar: log 方差 (batch, latent_dim)

    Returns:
        z: 采样的隐变量 (batch, latent_dim)
    """
    # 答案:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z
```

**题目 3: 实现 MMD 的 PyTorch 可微版本**

```python
def mmd_loss_pytorch(phi_t, phi_c, gamma=1.0):
    """
    PyTorch 可微 MMD Loss

    Args:
        phi_t: 处理组表示 (n, d)
        phi_c: 对照组表示 (m, d)
        gamma: RBF 核参数

    Returns:
        mmd: MMD 损失
    """
    def rbf_kernel(X, Y):
        XX = torch.sum(X**2, dim=1, keepdim=True)
        YY = torch.sum(Y**2, dim=1, keepdim=True)
        XY = X @ Y.T
        dist_sq = XX + YY.T - 2 * XY
        return torch.exp(-gamma * dist_sq)

    n = phi_t.shape[0]
    m = phi_c.shape[0]

    K_TT = rbf_kernel(phi_t, phi_t)
    K_TC = rbf_kernel(phi_t, phi_c)
    K_CC = rbf_kernel(phi_c, phi_c)

    term1 = (K_TT.sum() - torch.trace(K_TT)) / (n * (n - 1))
    term2 = K_TC.sum() / (n * m)
    term3 = (K_CC.sum() - torch.trace(K_CC)) / (m * (m - 1))

    mmd_sq = term1 - 2 * term2 + term3

    return mmd_sq  # 返回平方避免 sqrt 梯度问题
```

### 系统设计题

**题目: 设计一个优惠券面额优化系统**

**要求**:
1. 输入: 用户特征 X
2. 输出: 最优优惠券面额
3. 考虑: ROI、预算约束、AB测试

**答案框架**:

```python
class CouponOptimizationSystem:
    """优惠券优化系统"""

    def __init__(self):
        # 模型: VCNet 或 DRNet
        self.dose_response_model = VCNet(input_dim=user_feature_dim)

        # 约束
        self.budget = total_budget
        self.cost_per_yuan = cost_per_yuan

    def train(self, X, T, Y):
        """训练剂量-响应模型"""
        # 使用历史 AB 测试数据
        train_vcnet(self.dose_response_model, X, T, Y)

    def find_optimal_coupon(self, user_features):
        """为单个用户找最优面额"""
        # 搜索空间: [0, 50] 元
        t_values = np.linspace(0, 50, 100)

        # 预测响应
        y_pred = self.dose_response_model.predict_dose_response(
            user_features, t_values
        )

        # 计算 ROI
        costs = t_values * self.cost_per_yuan
        net_profit = y_pred - costs

        # 找最优
        optimal_idx = np.argmax(net_profit)
        optimal_amount = t_values[optimal_idx]

        return optimal_amount

    def batch_optimize(self, user_features_batch):
        """批量优化（考虑预算约束）"""
        n_users = len(user_features_batch)

        # 为每个用户找最优面额
        optimal_amounts = []
        expected_profits = []

        for features in user_features_batch:
            amount = self.find_optimal_coupon(features)
            profit = self.estimate_profit(features, amount)

            optimal_amounts.append(amount)
            expected_profits.append(profit)

        # 预算约束: 选择 ROI 最高的用户发券
        total_cost = sum(optimal_amounts)

        if total_cost > self.budget:
            # 按 ROI 排序
            roi = np.array(expected_profits) / np.array(optimal_amounts)
            sorted_idx = np.argsort(roi)[::-1]

            # 贪心选择
            selected = []
            remaining_budget = self.budget

            for idx in sorted_idx:
                if optimal_amounts[idx] <= remaining_budget:
                    selected.append(idx)
                    remaining_budget -= optimal_amounts[idx]

            return selected, [optimal_amounts[i] for i in selected]
        else:
            return list(range(n_users)), optimal_amounts
```

---

## 学习路径建议

### 初学者 (0-3 个月)

1. **基础理论** (2 周)
   - 潜在结果框架
   - 因果图 (DAG)
   - 混淆、选择偏差

2. **传统方法** (4 周)
   - PSM, IPW, DR
   - Meta-Learners (S/T/X-Learner)
   - Causal Forest

3. **深度学习基础** (2 周)
   - PyTorch 基础
   - 神经网络训练
   - 正则化技巧

4. **Part 5: 深度因果模型** (4 周)
   - Week 1: 表示学习 + IPM
   - Week 2: TARNet + DragonNet
   - Week 3: CEVAE
   - Week 4: GANITE + VCNet

### 进阶者 (3-6 个月)

1. **理论深入**
   - 半参数效率理论
   - 双重鲁棒估计
   - 敏感性分析

2. **高级模型**
   - TEDVAE, Perfect Match
   - SITE, Causal Transformer
   - Continuous Treatment

3. **实战项目**
   - 营销优化
   - 个性化推荐
   - 医疗决策支持

### 面试准备 (1-2 个月)

1. **理论复习**
   - 每个模型的数学推导
   - 假设条件和适用场景
   - 优缺点对比

2. **编程练习**
   - 从零实现所有模型
   - LeetCode 风格编程题
   - 系统设计题

3. **论文阅读**
   - TARNet (ICML 2017)
   - DragonNet (2019)
   - CEVAE (ICLR 2018)
   - GANITE (ICLR 2018)
   - VCNet (ICLR 2021)

---

## 参考文献

1. Shalit, U., Johansson, F. D., & Sontag, D. (2017). Estimating individual treatment effect: generalization bounds and algorithms. ICML.

2. Shi, C., Blei, D., & Veitch, V. (2019). Adapting neural networks for the estimation of treatment effects. NeurIPS.

3. Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., & Welling, M. (2017). Causal effect inference with deep latent-variable models. NeurIPS.

4. Yoon, J., Jordon, J., & van der Schaar, M. (2018). GANITE: Estimation of individualized treatment effects using generative adversarial nets. ICLR.

5. Nie, X., Ye, M., Liu, Q., & Nicolae, D. (2021). VCNet and functional targeted regularization for learning causal effects of continuous treatments. ICLR.

---

**本文档提供了 Part 5 所有 Notebook 的完整解答、数学推导和面试准备材料。**

**建议学习路径**:
1. 先理解理论推导
2. 完成所有编程练习
3. 做面试题巩固
4. 在实际项目中应用

**Good luck! 🚀**
