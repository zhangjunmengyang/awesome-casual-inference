# 因果推断面试 Cheatsheet - Part 3 准实验方法

> 本 Cheatsheet 整理自 Part 3 Quasi-Experiments 模块，收录"2 分钟实现一下 xxx"的编程题和高频核心概念题。
>
> **使用方法**：先看题目思考，再展开答案对照。

---

## 目录

1. [核心公式速查](#核心公式速查)
2. [2 分钟编程题](#2-分钟编程题)
3. [高频概念题](#高频概念题)
4. [方法对比速查](#方法对比速查)
5. [易错点警示](#易错点警示)

---

## 核心公式速查

### DID (Difference-in-Differences)

| 概念 | 公式 | 一句话解释 |
|-----|------|-----------|
| DID 估计量 | $\tau = (Y_{T,post} - Y_{T,pre}) - (Y_{C,post} - Y_{C,pre})$ | 处理组变化 - 对照组变化 |
| 回归形式 | $Y = \beta_0 + \beta_1 Treat + \beta_2 Post + \beta_3 (Treat \times Post) + \epsilon$ | $\beta_3$ 就是 DID 估计量 |
| 核心假设 | 平行趋势假设：$E[Y_{T,t}(0) - Y_{T,t-1}(0)] = E[Y_{C,t}(0) - Y_{C,t-1}(0)]$ | 无处理时两组趋势平行 |

### 合成控制 (Synthetic Control)

| 概念 | 公式 | 一句话解释 |
|-----|------|-----------|
| 合成权重 | $W^* = \arg\min_W \sum_{t=1}^{T_0} (Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt})^2$ | 最小化前处理期的拟合误差 |
| 约束条件 | $w_j \geq 0, \sum w_j = 1$ | 非负权重，和为 1 |
| 处理效应 | $\tau_t = Y_{1t} - \sum_{j=2}^{J+1} w_j^* Y_{jt}$ | 实际值 - 合成值 |

### RDD (Regression Discontinuity Design)

| 概念 | 公式 | 一句话解释 |
|-----|------|-----------|
| Sharp RDD | $\tau = \lim_{x \downarrow c} E[Y\|X=x] - \lim_{x \uparrow c} E[Y\|X=x]$ | 门槛两侧的结果跳跃 |
| Fuzzy RDD | $\tau = \frac{\text{结果跳跃}}{\text{处理跳跃}} = \frac{E[Y\|X=c^+] - E[Y\|X=c^-]}{E[D\|X=c^+] - E[D\|X=c^-]}$ | 用 2SLS 估计 LATE |
| IK 带宽 | $h_{IK} = C_K \cdot \left[\frac{\sigma^2}{n \cdot m_2^2}\right]^{1/5}$ | 最优带宽公式 |

### IV (Instrumental Variables)

| 概念 | 公式 | 一句话解释 |
|-----|------|-----------|
| 2SLS 估计量 | $\hat{\beta}_{2SLS} = \frac{Cov(Z,Y)}{Cov(Z,X)} = \frac{\text{Reduced Form}}{\text{First Stage}}$ | Wald 估计 |
| 第一阶段 | $X = \pi_0 + \pi_1 Z + v$ | 用 IV 预测内生变量 |
| 第二阶段 | $Y = \beta_0 + \beta_1 \hat{X} + \epsilon$ | 用预测值估计因果效应 |

---

## 2 分钟编程题

### 题目 1：DID 估计器

> **题目**：实现双重差分估计量（手动计算 + 回归）

<details>
<summary>💡 参考答案</summary>

```python
def estimate_did(df, outcome_col='Y', treat_col='treat', post_col='post'):
    """
    DID 估计（手动计算）

    公式: τ = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
    """
    # 四个均值
    treat_post = df[(df[treat_col]==1) & (df[post_col]==1)][outcome_col].mean()
    treat_pre = df[(df[treat_col]==1) & (df[post_col]==0)][outcome_col].mean()
    control_post = df[(df[treat_col]==0) & (df[post_col]==1)][outcome_col].mean()
    control_pre = df[(df[treat_col]==0) & (df[post_col]==0)][outcome_col].mean()

    # 第一次差分（时间）
    diff_treat = treat_post - treat_pre
    diff_control = control_post - control_pre

    # 第二次差分（组间）
    did = diff_treat - diff_control

    return did

# 回归方法
import statsmodels.formula.api as smf

def estimate_did_regression(df):
    """DID 回归估计（带聚类标准误）"""
    df['treat_post'] = df['treat'] * df['post']

    # 回归 Y ~ treat + post + treat_post
    model = smf.ols('Y ~ treat + post + treat_post', data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['user_id']}  # 按个体聚类
    )

    return model.params['treat_post'], model.bse['treat_post']
```

**关键点**：
- 手动计算和回归结果一致
- 必须使用聚类标准误（按个体或地区）
- 交互项系数就是 DID 估计量
</details>

---

### 题目 2：合成控制权重优化

> **题目**：实现合成控制的权重估计（二次规划）

<details>
<summary>💡 参考答案</summary>

```python
from scipy.optimize import minimize
import numpy as np

def synthetic_control_weights(treated, donors, T0):
    """
    估计合成控制权重

    参数:
        treated: 处理单位时间序列 (T,)
        donors: 供体池矩阵 (T, J)
        T0: 处理时点索引

    返回:
        weights: 最优权重 (J,)
    """
    # 前处理期数据
    treated_pre = treated[:T0]
    donors_pre = donors[:T0, :]

    # 优化目标：最小化前处理期的 RMSE
    def objective(w):
        synthetic = donors_pre @ w
        return np.sum((treated_pre - synthetic) ** 2)

    # 约束：权重和为 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # 边界：权重非负
    n_donors = donors.shape[1]
    bounds = [(0, 1) for _ in range(n_donors)]

    # 初始值：等权重
    w0 = np.ones(n_donors) / n_donors

    # 求解
    result = minimize(
        objective, w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

# 使用示例
weights = synthetic_control_weights(treated, donors, T0=18)
synthetic = donors @ weights
att = np.mean(treated[T0:] - synthetic[T0:])
```

**关键点**：
- 二次规划问题（QP）
- 约束：$w_j \geq 0, \sum w_j = 1$
- 只用前处理期数据拟合
</details>

---

### 题目 3：RDD 局部线性回归

> **题目**：实现 Sharp RDD 的局部线性回归估计

<details>
<summary>💡 参考答案</summary>

```python
from sklearn.linear_model import LinearRegression

def sharp_rdd_estimate(X, Y, cutoff=0, bandwidth=None):
    """
    Sharp RDD 估计（局部线性回归）

    参数:
        X: 驱动变量 (running variable)
        Y: 结果变量
        cutoff: 门槛值
        bandwidth: 带宽（如为 None 则用 IK 方法）

    返回:
        tau: RDD 估计量
    """
    # 中心化
    X_centered = X - cutoff
    D = (X >= cutoff).astype(int)  # 处理指示变量

    # 自动带宽（简化版 IK）
    if bandwidth is None:
        bandwidth = 1.06 * np.std(X_centered) * len(X)**(-1/5)

    # 带宽内样本
    mask = np.abs(X_centered) <= bandwidth
    X_bw = X_centered[mask]
    Y_bw = Y[mask]
    D_bw = D[mask]

    # 核权重（三角核）
    weights = np.maximum(1 - np.abs(X_bw) / bandwidth, 0)

    # 加权回归: Y ~ D + X_centered + D*X_centered
    X_reg = np.column_stack([
        D_bw,
        X_bw,
        D_bw * X_bw
    ])

    # 手动加权最小二乘
    W = np.diag(weights)
    beta = np.linalg.inv(X_reg.T @ W @ X_reg) @ (X_reg.T @ W @ Y_bw)

    tau = beta[0]  # D 的系数就是 RDD 估计量

    return tau, bandwidth

# 使用示例
tau, h = sharp_rdd_estimate(spending, repurchase_rate, cutoff=200)
print(f"RDD 估计: {tau:.2f} (带宽: {h:.1f})")
```

**关键点**：
- 只用带宽内的数据
- 三角核加权
- 估计门槛处的局部效应
</details>

---

### 题目 4：2SLS 两阶段最小二乘

> **题目**：手动实现 2SLS 估计器

<details>
<summary>💡 参考答案</summary>

```python
from sklearn.linear_model import LinearRegression
from scipy import stats

def two_stage_least_squares(Z, X, Y):
    """
    两阶段最小二乘 (2SLS) 估计

    参数:
        Z: 工具变量 (n,)
        X: 内生处理变量 (n,)
        Y: 结果变量 (n,)

    返回:
        results: 包含估计结果的字典
    """
    # 第一阶段：X ~ Z
    first_stage = LinearRegression()
    first_stage.fit(Z.reshape(-1, 1), X)
    X_hat = first_stage.predict(Z.reshape(-1, 1))

    # 检验工具变量强度（F 统计量）
    residuals_fs = X - X_hat
    rss_fs = np.sum(residuals_fs**2)
    tss_fs = np.sum((X - X.mean())**2)
    r2_fs = 1 - rss_fs / tss_fs

    n = len(X)
    f_stat = (r2_fs / 1) / ((1 - r2_fs) / (n - 2))

    # 第二阶段：Y ~ X_hat
    second_stage = LinearRegression()
    second_stage.fit(X_hat.reshape(-1, 1), Y)
    beta_2sls = second_stage.coef_[0]

    # Wald 估计（验证）
    beta_wald = np.cov(Z, Y)[0, 1] / np.cov(Z, X)[0, 1]

    # 标准误（简化版）
    residuals = Y - second_stage.predict(X_hat.reshape(-1, 1))
    sigma2 = np.sum(residuals**2) / (n - 2)
    var_X_hat = np.var(X_hat)
    se = np.sqrt(sigma2 / (n * var_X_hat))

    return {
        'beta_2sls': beta_2sls,
        'beta_wald': beta_wald,
        'se': se,
        'first_stage_f': f_stat,
        'first_stage_r2': r2_fs
    }

# 使用示例
results = two_stage_least_squares(cost_shock, price, quantity)
print(f"2SLS 估计: {results['beta_2sls']:.4f}")
print(f"第一阶段 F: {results['first_stage_f']:.2f}")
print(f"{'强工具变量' if results['first_stage_f'] > 10 else '弱工具变量'}")
```

**关键点**：
- 第一阶段预测内生变量
- 第二阶段用预测值回归
- F > 10 判断是否为强工具变量
</details>

---

### 题目 5：平行趋势检验

> **题目**：实现 Event Study 可视化平行趋势

<details>
<summary>💡 参考答案</summary>

```python
import statsmodels.formula.api as smf
import pandas as pd

def event_study(df, outcome_col, treat_col, time_col, event_time):
    """
    Event Study 设计检验平行趋势

    参数:
        df: 面板数据
        outcome_col: 结果变量名
        treat_col: 处理组指示变量
        time_col: 时间变量
        event_time: 政策发生时间

    返回:
        event_df: 包含各期系数和置信区间的 DataFrame
    """
    # 创建相对时间变量
    df['rel_time'] = df[time_col] - event_time

    # 创建时间虚拟变量（省略 t=-1 作为基准）
    time_dummies = []
    for t in df['rel_time'].unique():
        if t != -1:  # 基准期
            dummy_name = f'time_{t}'
            df[dummy_name] = ((df['rel_time'] == t) & (df[treat_col] == 1)).astype(int)
            time_dummies.append(dummy_name)

    # Event Study 回归
    formula = f"{outcome_col} ~ {' + '.join(time_dummies)}"
    model = smf.ols(formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['unit_id']}
    )

    # 提取系数和置信区间
    results = []
    for t in sorted(df['rel_time'].unique()):
        if t == -1:
            # 基准期
            results.append({
                'rel_time': t,
                'coef': 0,
                'ci_lower': 0,
                'ci_upper': 0
            })
        else:
            dummy_name = f'time_{t}'
            coef = model.params[dummy_name]
            ci = model.conf_int().loc[dummy_name]
            results.append({
                'rel_time': t,
                'coef': coef,
                'ci_lower': ci[0],
                'ci_upper': ci[1]
            })

    return pd.DataFrame(results)

# 绘图
import plotly.graph_objects as go

event_df = event_study(df, 'spending', 'treat', 'period', event_time=6)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=event_df['rel_time'],
    y=event_df['coef'],
    mode='markers+lines',
    name='DID 估计量'
))

# 置信区间
fig.add_trace(go.Scatter(
    x=event_df['rel_time'],
    y=event_df['ci_upper'],
    fill=None,
    mode='lines',
    line_color='lightblue',
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=event_df['rel_time'],
    y=event_df['ci_lower'],
    fill='tonexty',
    mode='lines',
    line_color='lightblue',
    name='95% CI'
))

fig.add_vline(x=0, line_dash="dash")
fig.add_hline(y=0, line_dash="dash")
fig.show()
```

**关键点**：
- 政策前系数应接近 0（平行趋势）
- 政策后系数显著（效应存在）
- 省略 t=-1 作为基准期
</details>

---

### 题目 6：Placebo Test（合成控制）

> **题目**：实现 Placebo Tests 检验合成控制显著性

<details>
<summary>💡 参考答案</summary>

```python
def placebo_test(treated, donors, T0, donor_names):
    """
    对每个供体单位执行 Placebo Test

    参数:
        treated: 真实处理单位 (T,)
        donors: 供体池 (T, J)
        T0: 处理时点
        donor_names: 供体名称列表

    返回:
        gaps: 所有单位的 gap（实际 - 合成）
        p_value: 排列 p 值
    """
    n_donors = donors.shape[1]

    # 存储每个单位的 gap
    gaps = {}
    effects = {}

    # 真实处理单位
    weights_real = synthetic_control_weights(treated, donors, T0)
    synthetic_real = donors @ weights_real
    gap_real = treated - synthetic_real
    gaps['真实'] = gap_real
    effects['真实'] = np.mean(gap_real[T0:])

    # 对每个供体执行 Placebo
    for i in range(n_donors):
        placebo_treated = donors[:, i]
        placebo_donors = np.delete(donors, i, axis=1)

        weights_placebo = synthetic_control_weights(placebo_treated, placebo_donors, T0)
        synthetic_placebo = placebo_donors @ weights_placebo
        gap_placebo = placebo_treated - synthetic_placebo

        gaps[donor_names[i]] = gap_placebo
        effects[donor_names[i]] = np.mean(gap_placebo[T0:])

    # 计算 p 值
    real_effect = abs(effects['真实'])
    all_effects = [abs(e) for e in effects.values()]
    p_value = np.mean([e >= real_effect for e in all_effects])

    return gaps, p_value

# 使用示例
gaps, p_value = placebo_test(treated, donors, T0=18, donor_names=cities)
print(f"p-value: {p_value:.3f}")
print(f"结论: {'显著' if p_value < 0.05 else '不显著'}")
```

**关键点**：
- 假装每个供体是处理单位
- 真实效应应远离 Placebo 分布
- p 值 < 0.05 说明效应显著
</details>

---

## 高频概念题

### Q1: DID 的平行趋势假设是什么？如何检验？

<details>
<summary>💡 答案</summary>

**定义**：
如果没有政策干预，处理组和对照组的结果变量趋势应该平行。

$$
E[Y_{T,t}(0) - Y_{T,t-1}(0)] = E[Y_{C,t}(0) - E_{C,t-1}(0)]
$$

**含义**：
- 两组的时间趋势相同
- 这是反事实假设，无法直接验证

**检验方法**：

1. **可视化趋势图**
   - 画出政策前两组的趋势
   - 看是否平行

2. **Event Study 设计**
   - 估计政策前各期的 DID 系数
   - 如果接近 0 且不显著 → 平行趋势成立

3. **Placebo Test**
   - 在政策前假设一个虚假的政策时间
   - 估计伪 DID，应不显著

**违反后果**：
- DID 估计有偏
- 可能高估或低估真实效应
</details>

---

### Q2: 合成控制和 DID 的区别？什么时候用哪个？

<details>
<summary>💡 答案</summary>

| 维度 | DID | 合成控制 |
|------|-----|----------|
| 处理单位 | 多个 | 单个 |
| 对照组 | 一个或多个（等权重） | 加权组合（优化权重） |
| 核心假设 | 平行趋势假设 | 结果变量可被供体池线性组合 |
| 前处理期 | 无需完美拟合 | 必须拟合好 |
| 推断方法 | 标准 t 检验 | Placebo Tests |
| 适用场景 | 政策评估（多个地区） | 单一事件（如新城市上线） |

**选择指南**：

```
只有 1 个处理单位？
  ↓ 是
  合成控制

有多个处理单位？
  ↓ 是
  有明确的对照组 + 平行趋势成立？
    ↓ 是
    DID
    ↓ 否
    考虑合成控制（分别估计）或 DID（检验假设）
```

**核心差异**：
- DID 依赖平行趋势假设
- 合成控制通过优化权重减少对假设的依赖
- 合成控制可以看作 DID 的推广（允许非平行趋势）
</details>

---

### Q3: Sharp RDD 和 Fuzzy RDD 的区别？

<details>
<summary>💡 答案</summary>

| 特征 | Sharp RDD | Fuzzy RDD |
|------|-----------|-----------|
| 门槛决定性 | 完全决定处理 | 影响处理概率 |
| 处理概率跳跃 | 0 → 1 | 连续跳跃（如 0.3 → 0.8） |
| 估计方法 | 局部线性回归 | 2SLS (工具变量) |
| 估计量 | ATE（门槛处） | LATE（顺从者效应） |

**Sharp RDD**：
- 门槛完全决定处理状态
- $P(D=1|X=x) = \begin{cases} 1 & x \geq c \\ 0 & x < c \end{cases}$
- 例子：60 分及格，及格必须重修

**Fuzzy RDD**：
- 门槛只影响处理概率
- $P(D=1|X=c^+) \neq P(D=1|X=c^-)$
- 例子：60 分及格有资格申请奖学金，但不是所有人申请

**Fuzzy = IV 设计**：
- 工具变量 $Z = \mathbb{1}[X \geq c]$
- 第一阶段：$D \sim Z$
- 第二阶段：$Y \sim \hat{D}$

**LATE 解释**：
Fuzzy RDD 估计的是 **顺从者 (Compliers)** 的效应：
- 超过门槛就接受处理
- 低于门槛就不接受

无法推断到 Always-takers 和 Never-takers
</details>

---

### Q4: 什么是弱工具变量？如何诊断？

<details>
<summary>💡 答案</summary>

**定义**：
工具变量 $Z$ 与内生变量 $X$ 的相关性很弱：
$$
Cov(Z, X) \approx 0
$$

**问题**：
1. **有限样本偏差**：2SLS 不再无偏
2. **推断失效**：标准误被低估
3. **放大内生性**：即使 $Z$ 与 $\epsilon$ 微弱相关，也导致大偏差

**诊断方法**：

1. **第一阶段 F 统计量**
   - 规则：F > 10 才是强工具变量
   - 公式：$F = \frac{R^2 / k}{(1-R^2)/(n-k-1)}$

2. **第一阶段 $R^2$**
   - 看 IV 对内生变量的解释力
   - 通常希望 $R^2 > 0.1$

**比喻**：
用温度计测体重 - 如果温度对体重影响很弱，即使温度和其他因素（如食欲）有一点点相关，也会导致估计严重有偏。

**经验法则**：
```
第一阶段 F 统计量:
  < 10  → 弱 IV，不可用
  10-20 → 边界，需要稳健推断
  > 20  → 强 IV
```

**解决方案**：
1. 找更强的工具变量
2. 使用多个工具变量组合
3. LIML 或 Fuller-LIML 等稳健估计方法
</details>

---

### Q5: 如何选择 RDD 的带宽？

<details>
<summary>💡 答案</summary>

**带宽的作用**：
- 决定用门槛附近多远的数据
- 小带宽 → 低偏差，高方差
- 大带宽 → 高偏差，低方差

**选择方法**：

1. **IK 方法 (Imbens-Kalyanaraman)**
   - 最优化 MSE
   - 公式：$h_{IK} = C \cdot [\frac{\sigma^2}{n \cdot m_2^2}]^{1/5}$
   - 适用于局部线性回归

2. **CCT 方法 (Calonico-Cattaneo-Titiunik)**
   - 稳健置信区间
   - 考虑偏差修正
   - 更保守（通常更小）

3. **交叉验证**
   - 最小化预测误差
   - 计算成本高

**实践建议**：

1. **报告多个带宽**
   ```python
   for h in [20, 50, 100]:
       tau, se = rdd_estimate(X, Y, cutoff, bandwidth=h)
       print(f"h={h}: τ={tau:.2f} (se={se:.2f})")
   ```

2. **IK 作为基准**
   - 自动选择，理论支撑
   - 报告 0.5×IK, IK, 2×IK 的结果

3. **检查稳健性**
   - 结果在不同带宽下是否稳定
   - 如果差异巨大，说明结果不稳健

**权衡**：
- 太小：标准误太大，精度不足
- 太大：偏离门槛，局部性丧失
</details>

---

### Q6: 工具变量的三个假设是什么？如何验证？

<details>
<summary>💡 答案</summary>

| 假设 | 定义 | 可检验性 | 检验方法 |
|------|------|---------|---------|
| 相关性 | $Cov(Z, X) \neq 0$ | ✅ 可检验 | 第一阶段 F > 10 |
| 排他性 | $Z \rightarrow Y$ 只能通过 $X$ | ❌ 不可检验 | 理论论证 |
| 外生性 | $Cov(Z, \epsilon) = 0$ | ❌ 不可检验 | 背景知识 |

**1. 相关性假设 (Relevance)**

$$Cov(Z, X) \neq 0$$

- 含义：工具变量必须影响内生变量
- 检验：第一阶段回归 F 统计量
  ```python
  # 第一阶段: X ~ Z
  model = LinearRegression().fit(Z, X)
  f_stat = calculate_f_statistic(model)
  # F > 10 → 强 IV
  ```

**2. 排他性假设 (Exclusion Restriction)**

$$Z \text{ 只能通过 } X \text{ 影响 } Y$$

- 含义：没有直接路径 $Z \rightarrow Y$
- 不可检验！需要理论支撑
- 例子：
  - ✅ 成本影响销量只能通过价格
  - ❌ 成本还影响质量（违反排他性）

**3. 外生性假设 (Exogeneity)**

$$Cov(Z, \epsilon) = 0$$

- 含义：工具变量与误差项不相关
- 不可检验！需要制度背景
- 例子：
  - ✅ 自然灾害的供应冲击
  - ❌ 企业主动降价（与需求冲击相关）

**过度识别检验**（多个 IV 时）：
- Hansen J 检验
- 原假设：所有 IV 都有效
- 局限：只能检验"至少一个无效"，无法识别哪个
</details>

---

### Q7: 什么是 LATE？和 ATE 有什么区别？

<details>
<summary>💡 答案</summary>

**LATE (Local Average Treatment Effect)**：
局部平均处理效应，特定子群体的效应。

**出现场景**：
1. **Fuzzy RDD**：门槛附近顺从者的效应
2. **工具变量**：被 IV 诱导改变处理状态的群体
3. **不完全依从的 RCT**：实际接受分配处理的群体

**人群分类**：

| 类型 | 定义 | IV 能估计吗？ |
|------|------|--------------|
| Compliers（顺从者） | $D(Z=1)=1, D(Z=0)=0$ | ✅ 能（LATE） |
| Always-takers | $D(Z=1)=1, D(Z=0)=1$ | ❌ 不能 |
| Never-takers | $D(Z=1)=0, D(Z=0)=0$ | ❌ 不能 |
| Defiers | $D(Z=1)=0, D(Z=0)=1$ | ❌ 假设不存在 |

**ATE vs LATE**：

$$
\text{ATE} = E[Y(1) - Y(0)] \quad \text{(全体)}
$$

$$
\text{LATE} = E[Y(1) - Y(0) | \text{Complier}] \quad \text{(顺从者)}
$$

**例子**：优惠券

- ATE：如果所有人都用券，平均效应是多少
- LATE：因为门槛（满 200）而决定是否用券的人，效应是多少
- 区别：
  - 总是用券的人（Always-taker）：再穷也买
  - 从不用券的人（Never-taker）：再便宜也不买
  - IV 只能估计"可被门槛影响"的人

**外推性问题**：
LATE 只对顺从者有效，无法推断到总体！

**实践建议**：
- 明确报告估计的是 LATE
- 讨论顺从者的特征
- 评估外推到总体的合理性
</details>

---

### Q8: Placebo Tests 在合成控制中的作用？

<details>
<summary>💡 答案</summary>

**目的**：
检验合成控制的效应是否显著（因为只有 1 个处理单位，无法用标准推断）

**原理**：
1. 假装每个供体单位是"处理单位"
2. 对它们也进行合成控制
3. 看真实处理单位的效应是否"异常"

**实施步骤**：

```
对每个供体 j:
  1. 假装它是处理单位
  2. 其他供体作为供体池
  3. 估计合成控制权重
  4. 计算它的 gap（实际 - 合成）

对真实处理单位:
  1. 计算 gap

比较:
  真实 gap 是否远离 Placebo gaps 分布？
```

**p 值计算**：

$$
p = \frac{1}{J+1} \sum_{j=1}^{J+1} \mathbb{1}[|\text{gap}_j| \geq |\text{gap}_{\text{真实}}|]
$$

**解释**：
- p < 0.05：真实效应显著（在 Placebo 分布的尾部）
- p > 0.05：无法区分真实效应和随机波动

**可视化**：
```python
# Gap 图
for each placebo unit:
    plot gap over time (thin gray line)
plot real unit gap (thick red line)

如果红线在灰线区域外 → 效应显著
```

**RMSPE 比值检验**：
- Pre-period RMSPE：前处理期拟合误差
- Post-period RMSPE：后处理期误差
- 比值：Post/Pre
- 如果真实单位的比值远大于 Placebo → 显著

**局限**：
- 供体池太小（< 10）时，检验力不足
- 前处理期拟合很差的 Placebo 应剔除
</details>

---

## 方法对比速查

### 四种方法的核心对比

| 方法 | 核心思想 | 识别假设 | 适用场景 | 估计量 | 推断方法 |
|------|----------|----------|----------|--------|---------|
| **DID** | 两次差分去除偏差 | 平行趋势假设 | 政策评估（多地区） | 交互项系数 | t 检验 |
| **合成控制** | 加权组合构造反事实 | 前处理期可拟合 | 单一事件（1 个单位） | gap 平均 | Placebo Tests |
| **RDD** | 门槛处局部随机化 | 结果连续性 | 有明确门槛 | 门槛处跳跃 | t 检验 |
| **IV** | 外生变异识别因果 | 相关性+排他性+外生性 | 存在内生性 | 2SLS | F 检验 + t 检验 |

### 假设强度对比

从弱到强：
1. **RDD**：只要门槛附近连续
2. **合成控制**：线性组合可近似
3. **DID**：全局平行趋势
4. **IV**：三个假设（排他性最难验证）

### 外部效度对比

从高到低：
1. **DID**：全局效应
2. **IV**：LATE（顺从者）
3. **合成控制**：单一单位
4. **RDD**：门槛处局部效应

### 数据需求对比

| 方法 | 处理单位数 | 对照单位数 | 时间期数 | 特殊要求 |
|------|-----------|-----------|---------|---------|
| DID | 多个 | 多个 | ≥2（前后） | 前处理期数据 |
| 合成控制 | 1 | ≥10 | ≥10 | 长前处理期 |
| RDD | 多个 | 多个 | 1 | 门槛附近密集数据 |
| IV | 多个 | - | 1 | 强工具变量 |

---

## 易错点警示

### 1. DID 的平行趋势是假设不是结论

❌ **错误**："我画图看到政策前两组趋势平行，所以 DID 估计是无偏的"

✅ **正确**：
- 平行趋势是**反事实假设**（关于 $Y(0)$）
- 政策前的平行只是**必要条件**，不是充分条件
- 仍需做 Event Study 和 Placebo Tests

### 2. 合成控制不是简单加权平均

❌ **错误**："随便给几个城市分配权重，然后计算加权平均"

✅ **正确**：
- 权重是**优化**出来的（最小化前处理期拟合误差）
- 必须满足约束：$w_j \geq 0, \sum w_j = 1$
- 必须检验前处理期拟合质量（RMSPE）

### 3. RDD 的带宽不是越大越好

❌ **错误**："用全部数据，带宽越大越好，样本量大"

✅ **正确**：
- 带宽太大 → 偏离门槛，局部性丧失
- 带宽太小 → 样本量小，方差大
- 用 IK 或 CCT 方法优化
- 报告多个带宽的稳健性检验

### 4. Fuzzy RDD ≠ Sharp RDD

❌ **错误**："门槛附近处理概率有跳跃，用 Sharp RDD 估计"

✅ **正确**：
- 处理概率跳跃但不是 0→1 → Fuzzy RDD
- 必须用 2SLS（工具变量方法）
- 估计的是 LATE，不是 ATE

### 5. 弱工具变量比没有 IV 更糟

❌ **错误**："第一阶段 F=5，虽然弱但还能用"

✅ **正确**：
- F < 10 → 弱 IV，2SLS 有偏且推断失效
- 弱 IV 会放大内生性问题
- 宁可不用 IV，也不用弱 IV

### 6. LATE 不能外推到 ATE

❌ **错误**："IV 估计的是 15%，所以全量推广效应就是 15%"

✅ **正确**：
- IV 估计的是顺从者 (Compliers) 的效应
- Always-takers 和 Never-takers 的效应未知
- 外推需要额外假设（如效应同质）

### 7. Placebo Tests 不是万能的

❌ **错误**："Placebo p=0.03，所以效应一定显著"

✅ **正确**：
- 供体池太小（< 10）时检验力不足
- 需要剔除前处理期拟合很差的 Placebo
- 结合 RMSPE 比值检验一起看

### 8. 合成控制不需要平行趋势

❌ **错误**："合成控制和 DID 一样，都要平行趋势"

✅ **正确**：
- 合成控制的优势就是**不需要**平行趋势
- 通过优化权重来拟合非平行趋势
- 只需前处理期拟合好即可

---

## 快速记忆卡片

```
┌─────────────────────────────────────────────────┐
│  DID 核心公式                                    │
│  ─────────────────                              │
│  τ = (Ȳ_T,post - Ȳ_T,pre) - (Ȳ_C,post - Ȳ_C,pre) │
│                                                 │
│  回归形式:                                       │
│  Y = β₀ + β₁·Treat + β₂·Post + β₃·Treat×Post   │
│      └─ DID 估计量                              │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  合成控制核心                                    │
│  ─────────────────                              │
│  找权重 W*: min Σ(Y₁ₜ - ΣwⱼYⱼₜ)²               │
│            前处理期                              │
│  约束: wⱼ ≥ 0, Σwⱼ = 1                         │
│                                                 │
│  推断: Placebo Tests (p 值)                     │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  RDD 核心公式                                    │
│  ─────────────────                              │
│  Sharp: τ = lim E[Y|X=c⁺] - lim E[Y|X=c⁻]      │
│                                                 │
│  Fuzzy: τ = 跳跃(Y) / 跳跃(D)                   │
│          = 用 2SLS 估计                         │
│                                                 │
│  关键: 带宽选择（IK 方法）                       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  IV 三个假设                                     │
│  ─────────────────                              │
│  1. 相关性: Cov(Z,X) ≠ 0  ✅ 可检验 (F>10)     │
│  2. 排他性: Z→Y 只通过 X  ❌ 不可检验           │
│  3. 外生性: Cov(Z,ε) = 0  ❌ 不可检验           │
│                                                 │
│  2SLS: β = Cov(Z,Y) / Cov(Z,X)                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  方法选择决策树                                  │
│  ─────────────────                              │
│  有随机实验？ → RCT                              │
│      ↓ 否                                       │
│  只有1个处理单位？ → 合成控制                     │
│      ↓ 否                                       │
│  有明确门槛？ → RDD                              │
│      ↓ 否                                       │
│  有政策实施时点 + 平行趋势？ → DID                │
│      ↓ 否                                       │
│  有强工具变量？ → IV                             │
│      ↓ 否                                       │
│  条件独立 + 可测混淆？ → PSM/IPW                 │
└─────────────────────────────────────────────────┘
```

---

> **最后提醒**：
> 1. DID 必须检验平行趋势（Event Study + Placebo）
> 2. 合成控制必须检查前处理期拟合（RMSPE）
> 3. RDD 必须报告多个带宽的稳健性
> 4. IV 必须检验工具变量强度（F > 10）
> 5. 所有方法都要可视化数据！

---

*最后更新: 2026-01-04*
