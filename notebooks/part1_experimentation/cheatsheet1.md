# Part 1 实验设计面试速查表

> 本文档收集 Part 1 实验设计模块中的核心面试题，包括「2分钟代码实现」和「高频概念题」。

---

## 1. 2分钟代码实现题

### 1.1 样本量计算（连续变量）

**来源**: part1_0_power_analysis.ipynb
**场景**: 面试官让你手写双样本 t 检验的样本量公式

```python
from scipy.stats import norm
import numpy as np

def sample_size_continuous(effect_size, sigma, alpha=0.05, power=0.8):
    """
    计算连续变量的样本量（每组）

    Args:
        effect_size: 预期效应大小 (μ_B - μ_A)
        sigma: 总体标准差
        alpha: 显著性水平（默认0.05）
        power: 统计功效（默认0.8）
    """
    z_alpha = norm.ppf(1 - alpha / 2)  # 双边检验
    z_beta = norm.ppf(power)

    n = 2 * sigma**2 * (z_alpha + z_beta)**2 / effect_size**2
    return int(np.ceil(n))

# 示例：检测5元提升，标准差20元
# sample_size_continuous(5, 20)  # → 每组约252人
```

---

### 1.2 样本量计算（比例/转化率）

**来源**: part1_0_power_analysis.ipynb
**场景**: 计算转化率 A/B 测试需要多少样本

```python
def sample_size_proportion(p_control, p_treatment, alpha=0.05, power=0.8):
    """
    计算比例检验的样本量（每组）

    Args:
        p_control: 对照组转化率
        p_treatment: 实验组预期转化率
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    p_bar = (p_control + p_treatment) / 2
    effect = p_treatment - p_control

    n = 2 * p_bar * (1 - p_bar) * (z_alpha + z_beta)**2 / effect**2
    return int(np.ceil(n))

# 示例：转化率从10%提升到12%
# sample_size_proportion(0.10, 0.12)  # → 每组约1,714人
```

---

### 1.3 统计功效计算

**来源**: part1_0_power_analysis.ipynb
**场景**: 给定样本量，计算能检测到效应的概率

```python
def calculate_power(n_per_group, effect_size, sigma=1.0, alpha=0.05):
    """给定样本量计算统计功效"""
    se = sigma * np.sqrt(2 / n_per_group)
    z_alpha = norm.ppf(1 - alpha / 2)
    ncp = effect_size / se  # 非中心参数

    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return power
```

---

### 1.4 CUPED 方差缩减

**来源**: part1_2_cuped_variance_reduction.ipynb
**场景**: 实现 CUPED 调整，这是大厂必考题

```python
def apply_cuped(Y, X_pre):
    """
    应用 CUPED 方差缩减

    公式: Y_adjusted = Y - θ(X_pre - mean(X_pre))
    其中: θ = Cov(Y, X_pre) / Var(X_pre)

    Args:
        Y: 实验期间的结果变量
        X_pre: 实验前的协变量（如历史数据）

    Returns:
        Y_adjusted: 调整后的结果
        theta: 回归系数
        variance_reduction: 方差缩减率
    """
    # 计算 theta
    cov_yx = np.cov(Y, X_pre)[0, 1]
    var_x = np.var(X_pre)
    theta = cov_yx / var_x

    # 调整 Y
    Y_adjusted = Y - theta * (X_pre - np.mean(X_pre))

    # 方差缩减率 = ρ²
    rho = np.corrcoef(Y, X_pre)[0, 1]
    variance_reduction = rho ** 2

    return Y_adjusted, theta, variance_reduction
```

---

### 1.5 分层 ATE 估计（Neyman 估计量）

**来源**: part1_3_stratified_analysis.ipynb
**场景**: 计算分层随机化实验的 ATE

```python
def stratified_ate(df, outcome_col, treatment_col, strata_col):
    """
    计算分层 ATE（Neyman 估计量）

    公式: ATE = Σ W_h × (ȳ_{h,T} - ȳ_{h,C})
    """
    strata = df[strata_col].unique()
    n_total = len(df)

    ate = 0
    for stratum in strata:
        df_s = df[df[strata_col] == stratum]
        weight = len(df_s) / n_total  # W_h = N_h / N

        y_t = df_s[df_s[treatment_col] == 1][outcome_col].mean()
        y_c = df_s[df_s[treatment_col] == 0][outcome_col].mean()
        tau_h = y_t - y_c

        ate += weight * tau_h

    return ate
```

---

### 1.6 Epsilon-Greedy（MAB 基础算法）

**来源**: part1_7_multi_armed_bandits.ipynb
**场景**: 实现最简单的 MAB 算法

```python
class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)   # 每个臂被选次数
        self.values = np.zeros(n_arms)   # 每个臂的估计价值

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)  # 探索
        return np.argmax(self.values)              # 利用

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # 增量式更新均值: new_avg = old_avg + (reward - old_avg) / n
        self.values[arm] += (reward - self.values[arm]) / n
```

---

### 1.7 Thompson Sampling

**来源**: part1_7_multi_armed_bandits.ipynb
**场景**: 实现贝叶斯 MAB 算法

```python
class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Beta 分布参数（成功次数+1）
        self.beta = np.ones(n_arms)   # Beta 分布参数（失败次数+1）

    def select_arm(self):
        # 从每个臂的后验分布采样
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        # 伯努利奖励：成功+1 或 失败+1
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

---

### 1.8 UCB（置信上界）

**来源**: part1_7_multi_armed_bandits.ipynb
**场景**: 实现 UCB1 算法

```python
class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self):
        self.total_counts += 1
        # 确保每个臂至少被选一次
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB = μ̂ + √(2·log(t) / n_a)
        ucb_values = self.values + np.sqrt(
            2 * np.log(self.total_counts) / self.counts
        )
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
```

---

## 2. 高频概念面试题

### 2.1 假设检验的两类错误

**问题**: 解释 Type I / Type II Error 和 Power

| 错误类型 | 定义 | 控制方法 |
|---------|------|---------|
| Type I (α) | 实际无效应，错误拒绝 H₀（假阳性） | 设置 α=0.05 |
| Type II (β) | 实际有效应，错误接受 H₀（假阴性） | 增加样本量 |
| Power (1-β) | 真实效应存在时，正确检测的概率 | 目标 ≥ 0.8 |

**记忆口诀**: "α 控假阳，β 控假阴，Power 是检测真效应的能力"

---

### 2.2 样本量与效应量的关系

**问题**: 为什么效应量减半，样本量翻4倍？

**答案**:
- 公式: n ∝ 1/δ²，样本量与效应平方成反比
- 直觉: 效应减半 → 信噪比减半 → 需要更多样本"放大信号"
- 数学: SE ∝ 1/√n，要使 SE 减半，n 需翻 4 倍

---

### 2.3 MDE（最小可检测效应）

**问题**: MDE 是什么？如何应用？

**答案**:
- **定义**: 给定样本量、α、Power，能检测出的最小效应
- **公式**: MDE = (z_{α/2} + z_β) × √(2σ²/n)
- **应用**:
  - 如果 MDE > 业务最小可接受提升 → 样本量不够
  - 用于事前规划，确保实验有意义

---

### 2.4 Peeking 问题

**问题**: 为什么不能在实验中途偷看数据并提前停止？

**答案**:
- **问题**: 假阳性率膨胀（5% → 20-30%）
- **原因**: 每次查看 = 一次假设检验，多重比较问题
- **数学**: 查看 k 次，假阳性概率 ≈ 1 - (1-α)^k
- **解决方案**:
  1. 预设固定样本量，只在结束时分析
  2. 使用序贯分析（Sequential Analysis）调整阈值

---

### 2.5 CUPED 原理

**问题**: CUPED 如何减少方差？

**答案**:
- **核心**: 用实验前协变量 X_pre "解释"结果 Y 中的部分变异
- **公式**: Y_adj = Y - θ(X_pre - X̄_pre)，θ = Cov(Y,X)/Var(X)
- **效果**: 方差缩减率 = ρ²（相关系数的平方）
- **类比**: 测减肥药效果时，比较体重变化而非绝对体重

**必要条件**:
1. X_pre ⊥ T（协变量在实验前确定）
2. Cov(Y, X_pre) ≠ 0（与结果相关）

---

### 2.6 分层随机化 vs CUPED

**问题**: 两者都用协变量减少方差，有什么区别？

| 维度 | 分层随机化 | CUPED |
|-----|-----------|-------|
| 时机 | 实验设计阶段 | 分析阶段 |
| 随机化 | 层内独立随机 | 全局随机化 |
| 假设 | 无需线性假设 | 需要线性关系 |
| 优势 | 可解释层内效应 | 更灵活，事后可用 |
| 变量类型 | 离散分层变量 | 连续协变量 |

---

### 2.7 SUTVA 假设

**问题**: SUTVA 是什么？什么时候会违背？

**答案**:
- **全称**: Stable Unit Treatment Value Assumption
- **含义**: Y_i(T_i, T_{-i}) = Y_i(T_i)，个体结果只依赖自己的处理状态
- **违背场景**:
  - 社交网络（朋友用了产品影响你）
  - 双边市场（司机/骑手是共享资源）
  - 共享库存（优惠券池有限）

---

### 2.8 网络效应与溢出

**问题**: 什么是溢出效应？如何估计？

**答案**:
- **定义**: 用户结果受其他用户处理状态影响
- **Ego-Cluster 模型**: Y_i = β₁·T_i + β₂·T̄_{friends} + ε
  - β₁: 直接效应
  - β₂: 溢出效应
  - β₁ + β₂: 总效应（全量推广的真实效应）

---

### 2.9 Switchback 实验

**问题**: 什么是 Switchback？适用场景？

**答案**:
- **核心**: 在时间维度随机化，而非用户维度
- **适用**: Uber/外卖等双边市场、全局策略变更
- **优势**: 避免用户级溢出
- **挑战**: Carryover 效应、时间序列相关性

---

### 2.10 MAB vs A/B Testing

**问题**: 什么时候用 MAB，什么时候用 A/B Testing？

| 场景 | 推荐方法 | 原因 |
|-----|---------|------|
| 需要严格统计推断 | A/B Testing | 置信区间、p值 |
| 快速找最优方案 | MAB | 动态分配流量 |
| 方案数量多(>2) | MAB | 自动淘汰差方案 |
| 决策后果重大 | A/B Testing | 需要因果理解 |
| 流量有限 | A/B Testing | 固定样本量设计 |

---

## 3. 核心公式速查

### 样本量计算

```
连续变量:
n = 2σ²(z_{α/2} + z_β)² / δ²

比例变量:
n = 2p̄(1-p̄)(z_{α/2} + z_β)² / (p₁-p₀)²
其中 p̄ = (p₀ + p₁) / 2
```

### CUPED

```
Y_adjusted = Y - θ(X_pre - X̄_pre)
θ = Cov(Y, X_pre) / Var(X_pre) = ρ × σ_Y / σ_X

方差缩减率 = ρ²
新标准误 = SE × √(1 - ρ²)
```

### 分层估计

```
ATE_strat = Σ W_h × τ_h
其中 W_h = N_h/N, τ_h = ȳ_{h,T} - ȳ_{h,C}

Var(ATE) = Σ W_h² × (s²_{h,T}/n_{h,T} + s²_{h,C}/n_{h,C})
```

### 网络效应

```
Ego-Cluster 回归:
Y_i = β₀ + β₁·T_i + β₂·T̄_{friends,i} + ε_i

直接效应 = β₁
溢出效应 = β₂
总效应 = β₁ + β₂
```

### MAB Regret Bounds

```
Epsilon-Greedy: R_T = O(T^{2/3})
UCB:            R_T = O(√(KT log T))
Thompson:       R_T = O(√(KT))
```

### UCB 公式

```
UCB_a(t) = μ̂_a + √(2 log t / n_a)
选择: a_t = argmax_a UCB_a(t)
```

---

## 4. 面试技巧

### 代码实现题

1. **先写函数签名和注释**，展示思路清晰
2. **说出关键公式**，再转换为代码
3. **处理边界情况**（如除零、空数组）
4. **主动写示例**验证正确性

### 概念题

1. **先给定义**，再解释原理
2. **用类比**帮助理解（如 CUPED 的减肥药类比）
3. **举实际场景**说明应用
4. **对比相似概念**的区别（如分层 vs CUPED）

---

*最后更新: 2026-01-04*
