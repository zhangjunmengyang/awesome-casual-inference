# Part 5: 深度学习因果推断 - 面试速查手册

> 涵盖 Representation Learning, TARNet/DragonNet, CEVAE, GANITE, VCNet 五大核心方法

---

## 目录
1. [2分钟实现题](#2分钟实现题)
2. [高频面试题](#高频面试题)
3. [核心公式速查](#核心公式速查)
4. [方法对比表](#方法对比表)

---

## 2分钟实现题

### 1. RBF核函数 + MMD损失

```python
def rbf_kernel(X, Y, gamma=1.0):
    """RBF核: k(x,y) = exp(-γ||x-y||²)"""
    XX = np.sum(X**2, axis=1).reshape(-1, 1)
    YY = np.sum(Y**2, axis=1).reshape(1, -1)
    dist_sq = XX + YY - 2 * X @ Y.T
    return np.exp(-gamma * dist_sq)

def compute_mmd(X, Y, gamma=1.0):
    """MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]"""
    K_XX, K_XY, K_YY = rbf_kernel(X, X, gamma), rbf_kernel(X, Y, gamma), rbf_kernel(Y, Y, gamma)
    n, m = X.shape[0], Y.shape[0]
    term1 = (K_XX.sum() - np.trace(K_XX)) / (n * (n-1))
    term2 = K_XY.sum() / (n * m)
    term3 = (K_YY.sum() - np.trace(K_YY)) / (m * (m-1))
    return np.sqrt(max(term1 - 2*term2 + term3, 0))
```

### 2. TARNet 双头架构

```python
class SimpleTARNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, repr_dim=25):
        super().__init__()
        self.representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim), nn.ReLU())
        self.head0 = nn.Sequential(nn.Linear(repr_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.head1 = nn.Sequential(nn.Linear(repr_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, x):
        phi = self.representation(x)
        return self.head0(phi), self.head1(phi), phi
```

### 3. Factual Loss

```python
def compute_factual_loss(y_true, t_true, y0_pred, y1_pred):
    """只在观测结果上计算损失: T=1用Y1, T=0用Y0"""
    t = t_true.unsqueeze(1) if len(t_true.shape) == 1 else t_true
    y_pred = torch.where(t == 1, y1_pred, y0_pred)
    return torch.mean((y_true.unsqueeze(1) - y_pred)**2)
```

### 4. DragonNet 复合损失

```python
def dragonnet_loss(y_true, t_true, y0_pred, y1_pred, propensity, epsilon, alpha=1.0, beta=1.0):
    # Factual Loss
    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
    factual_loss = torch.mean((y_true - y_pred)**2)
    # Propensity Loss (BCE)
    prop_loss = -torch.mean(t_true * torch.log(propensity + 1e-8) + (1-t_true) * torch.log(1-propensity + 1e-8))
    # Targeted Regularization
    h = t_true / (propensity + 1e-8) - (1-t_true) / (1-propensity + 1e-8)
    targeted_reg = torch.mean((y_true - y_pred - epsilon * h)**2)
    return factual_loss + alpha * prop_loss + beta * targeted_reg
```

### 5. 重参数化技巧 (VAE/CEVAE)

```python
def reparameterize(mu, log_var):
    """Z = μ + σ·ε, ε~N(0,1) → 使采样可微"""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### 6. KL散度 (高斯)

```python
def kl_divergence(mu, logvar):
    """KL(N(μ,σ²) || N(0,1)) = 0.5·Σ(μ² + σ² - log(σ²) - 1)"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
```

### 7. 反事实生成器 (GANITE)

```python
class CounterfactualGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, noise_dim=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1 + 1 + noise_dim, hidden_dim),  # X + T + Y + Z
            nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1))

    def forward(self, x, t, y):
        z = torch.randn(x.shape[0], self.noise_dim, device=x.device)
        return self.net(torch.cat([x, t.view(-1,1), y.view(-1,1), z], dim=1))
```

### 8. 样条基函数 (VCNet)

```python
class TruncatedBasis(nn.Module):
    """截断幂基: B_k(t) = max(0, t-ξ_k)^p"""
    def __init__(self, num_knots=5, degree=2):
        super().__init__()
        self.register_buffer('knots', torch.linspace(0, 1, num_knots + 2)[1:-1])
        self.degree = degree

    def forward(self, t):
        t = t.unsqueeze(1) if len(t.shape) == 1 else t
        # 多项式项: 1, t, t²
        poly = [torch.ones_like(t)] + [t**d for d in range(1, self.degree+1)]
        # 截断幂项
        trunc = [torch.relu(t - k)**self.degree for k in self.knots]
        return torch.cat(poly + trunc, dim=1)
```

---

## 高频面试题

### 表示学习 (Representation Learning)

**Q1: 为什么需要表示平衡？如何衡量？**
- **原因**: 处理组/对照组在表示空间分布相似才能减少估计偏差
- **指标**: SMD < 0.1 为良好平衡；MMD 越小越好
- **公式**: `SMD_j = |μ_T,j - μ_C,j| / σ_j`

**Q2: MMD vs Wasserstein 如何选择？**

| 特性 | MMD | Wasserstein |
|------|-----|-------------|
| 计算复杂度 | O(n²) | O(n³) |
| 适用维度 | 中低维 | 高维 |
| 几何意义 | 核空间均值差 | 最优传输距离 |
| 推荐场景 | 因果推断 | GAN/图像 |

**Q3: CFR损失函数设计原则？**
$$\mathcal{L} = \mathcal{L}_{prediction} + \alpha \cdot \text{IPM}(P_T, P_C)$$
- α 太小: 分布差异大，估计有偏
- α 太大: 过度追求平衡，损害预测能力
- 典型范围: α ∈ [0.01, 1.0]

---

### TARNet / DragonNet

**Q4: TARNet vs 普通回归的区别？**
1. **架构**: 共享表示 + 双头 vs 直接预测
2. **训练**: Factual Loss (只用观测结果) vs 全样本
3. **输出**: 同时预测 Y(0), Y(1) → ITE
4. **效率**: 共享层允许知识迁移

**Q5: DragonNet 为什么加倾向得分头？**
1. **正则化**: 强迫表示层学习混淆因子
2. **双重鲁棒**: Targeted Reg 实现 DR 估计
3. **捕获混淆**: e(X) 反映哪些因素影响处理分配

**Q6: Targeted Regularization 的 h 是什么？**
$$h = \frac{T}{e(X)} - \frac{1-T}{1-e(X)}$$
- 作用: IPW 调整因子，平衡处理/对照分布
- epsilon: 可学习参数，自适应修正幅度

---

### CEVAE

**Q7: CEVAE 解决什么问题？**
- **问题**: 处理**未观测混淆变量**
- **方法**: VAE 学习隐变量 Z，X 作为 Z 的代理变量
- **假设**: 代理变量假设 - X 包含关于 Z 的足够信息

**Q8: ELBO 推导？**
$$\mathcal{L}_{ELBO} = \underbrace{E_q[\log p(X|Z)]}_{\text{重构}} - \underbrace{KL(q(Z|X) || p(Z))}_{\text{正则化}}$$

**Q9: 为什么需要重参数化技巧？**
- **问题**: 采样 Z ~ q(Z|X) 不可微
- **解决**: Z = μ + σ·ε, ε~N(0,1)
- **效果**: 随机性转移到外部噪声，梯度可传播

---

### GANITE

**Q10: GANITE 两阶段架构？**
- **Block 1 (反事实块)**: G_cf + D_cf → 生成缺失的 Y(0) 或 Y(1)
- **Block 2 (ITE块)**: G_ite + D_ite → 从完整结果估计 ITE

**Q11: GANITE vs TARNet/CEVAE？**
- **TARNet**: 判别式，预测均值
- **CEVAE**: 生成式 VAE，学习后验
- **GANITE**: 生成式 GAN，生成分布 → 不确定性量化

**Q12: 何时用 GANITE？**
- ✅ 需要不确定性量化
- ✅ 结果分布复杂/多峰
- ❌ 数据量小 (GAN 难训练)

---

### VCNet

**Q13: VCNet vs 简单拼接 [X, T]？**
```
简单拼接: Y = NN([X, T])     → 隐式交互
VCNet:    Y = W(T)·φ(X)      → 显式交互 + 样条光滑
```

**Q14: 样条基函数的作用？**
- **光滑性**: 强制 dose-response 连续
- **样本效率**: t=0.5 的观测帮助预测 t=0.51
- **正则化**: 防止数据稀疏区过拟合

**Q15: 连续处理的共同支撑假设难满足吗？**
- **更难**: 需要 ∀t,X 都有 f(t|X) > 0
- **缓解**: 分层随机化实验、限制推断范围、主动学习

---

## 核心公式速查

### IPM & MMD
$$\text{IPM}_{\mathcal{F}}(P, Q) = \sup_{f \in \mathcal{F}} |E_P[f] - E_Q[f]|$$
$$\text{MMD}^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]$$

### TARNet/DragonNet
$$\mathcal{L}_{factual} = \frac{1}{n}\sum (Y_i - \hat{Y}_i^{factual})^2$$
$$\mathcal{L}_{DragonNet} = \mathcal{L}_{factual} + \alpha \cdot \mathcal{L}_{prop} + \beta \cdot \mathcal{L}_{targeted}$$

### CEVAE
$$\mathcal{L}_{ELBO} = E_q[\log p(X,T,Y|Z)] - KL(q(Z|X,T,Y) || p(Z))$$
$$KL = \frac{1}{2}\sum(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$$

### GANITE
$$\mathcal{L}_G = -E[\log D(G(z))] + \lambda \cdot \mathcal{L}_{supervised}$$

### VCNet
$$Y = W(T) \cdot \phi(X) + b(T), \quad W(T) = \sum_k \alpha_k B_k(T)$$

---

## 方法对比表

| 方法 | 处理类型 | 核心思想 | 隐混淆 | 不确定性 | 训练难度 |
|------|---------|---------|--------|---------|---------|
| **CFR/表示学习** | 二元 | IPM 平衡表示 | ❌ | ❌ | 简单 |
| **TARNet** | 二元 | 共享表示+双头 | ❌ | ❌ | 简单 |
| **DragonNet** | 二元 | +倾向得分头+DR | ❌ | ❌ | 中等 |
| **CEVAE** | 二元 | VAE 学隐变量 | ✅ | ✅ | 中等 |
| **GANITE** | 二元 | GAN 生成反事实 | ❌ | ✅ | 困难 |
| **VCNet** | 连续 | 变系数+样条 | ❌ | ❌ | 中等 |

### 选择指南

```
数据量小 (<5k) + 无隐混淆 → TARNet
混淆强 + 需要 DR → DragonNet
怀疑隐混淆 + 代理变量可用 → CEVAE
需要不确定性 + 数据充足 → GANITE
处理连续 (剂量/价格) → VCNet
```

---

## 调参经验

### 通用参数

| 参数 | 推荐范围 |
|------|---------|
| learning_rate | 1e-4 ~ 1e-2 |
| hidden_dim | 50 ~ 200 |
| repr_dim | 25 ~ 100 |
| batch_size | 64 ~ 256 |

### 方法特定

- **DragonNet**: α (prop) = 0.5~2.0, β (targeted) = 0.5~2.0
- **CEVAE**: β (KL weight) = 0.1~2.0, latent_dim = 10~50
- **VCNet**: num_knots = 5~10

---

## 常见陷阱

1. **忘记 Factual Loss**: 不能在反事实上算损失
2. **KL 塌陷**: CEVAE 中 KL→0，用 annealing 解决
3. **GAN 模式崩溃**: GANITE 只学到一个峰，加 diversity loss
4. **共同支撑不满足**: VCNet 数据稀疏区预测不可靠
5. **表示不平衡**: CFR 忘记加 IPM 正则化

---

*Last updated: 2026-01-04*
