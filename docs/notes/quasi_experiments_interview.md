# Part 3: Quasi-Experiments - 面试指南

## 目录

1. [Difference-in-Differences (DID)](#1-difference-in-differences-did)
2. [Synthetic Control Method (SCM)](#2-synthetic-control-method-scm)
3. [Regression Discontinuity Design (RDD)](#3-regression-discontinuity-design-rdd)
4. [Instrumental Variables (IV)](#4-instrumental-variables-iv)
5. [方法对比与选择](#5-方法对比与选择)

---

## 1. Difference-in-Differences (DID)

### 核心原理

**一句话总结**：通过对照组的变化趋势，推断处理组在没有处理时的反事实趋势。

### 高频面试题

#### Q1: 解释 DID 的核心假设是什么？如何检验？

**答案**：

核心假设是**平行趋势假设（Parallel Trends Assumption）**：

- **定义**：在没有处理的情况下，处理组和对照组的结果变量会有相同的时间趋势
- **数学表达**：E[Y₁ₜ⁽⁰⁾ - Y₁,ₜ₋₁⁽⁰⁾] = E[Y₀ₜ - Y₀,ₜ₋₁]

**检验方法**：

1. **图形化检验**
   - 绘制处理组和对照组在政策前的趋势图
   - 如果两条线平行 → 支持假设

2. **Lead Test（提前期检验）**
   ```python
   # 在政策前的各期添加虚拟处理变量
   for t in pre_periods:
       df[f'lead_{t}'] = treat * (period == t)

   # 回归检验系数是否显著
   # 如果不显著 → 支持平行趋势
   ```

3. **安慰剂检验（Placebo Test）**
   - 假设一个假的政策时间点（在真实政策之前）
   - 估计"假"DID 效应
   - 如果不显著 → 支持平行趋势

**面试加分点**：
- 强调"平行趋势是不可直接检验的"（因为观察不到反事实）
- 所有检验都只能检验政策前的趋势，需要假设政策后也成立
- 如果违反，可以考虑使用合成控制、匹配、或控制趋势差异

---

#### Q2: 平行趋势假设违反了怎么办？

**答案**：

**方法 1：控制时间趋势**
```python
# 允许不同组有不同的线性趋势
model = 'Y ~ treat + post + treat_post + treat*time + post*time'
```

**方法 2：匹配 + DID**
- 先用 PSM 找到趋势相似的对照组
- 再在匹配样本上做 DID

**方法 3：改用合成控制（Synthetic Control）**
- 不假设平行趋势
- 通过优化权重找到最佳对照组合

**方法 4：控制组特定趋势**
```python
# 允许每个组有自己的固定效应和趋势
model = 'Y ~ C(group) + C(time) + treat_post'
```

**方法 5：进行敏感性分析**
- 报告不同规格下的结果
- 检查结果的稳健性

**面试加分点**：
- 没有银弹，需要结合具体场景选择方法
- 透明度很重要，报告所有检验结果
- 如果多种方法都得到相同结论，结果更可信

---

#### Q3: 交错 DID（Staggered DID）有什么问题？

**答案**：

**问题**：当不同单位在不同时间接受处理时，传统的 TWFE（Two-Way Fixed Effects）估计量可能有偏。

**核心原因**：
1. 已接受处理的单位会成为后接受处理单位的"对照组"
2. 如果处理效应随时间变化，会出现"负权重"问题
3. Goodman-Bacon 分解定理表明，TWFE 是多个 2×2 DID 的加权平均，但权重可能为负

**解决方案**：

**方法 1：Callaway & Sant'Anna (2021)**
- 用"Never-treated"或"Not-yet-treated"作为对照组
- 估计每个 cohort × time 的 ATT
- 然后聚合

**方法 2：Sun & Abraham (2021) - 事件研究法**
```python
# 相对于处理时间的事件时间
df['event_time'] = df['time'] - df['treatment_time']

# 估计每个事件时间的效应
for k in event_times:
    df[f'D_{k}'] = (event_time == k) * treated
```

**方法 3：de Chaisemartin & D'Haultfoeuille (2020)**
- 提供 DID_M 估计量
- 检查负权重的比例

**面试加分点**：
- 提到 Goodman-Bacon 分解定理
- 提到"禁忌比较"（Forbidden Comparison）的概念
- 知道 `did` R 包 或 `csdid` Stata 包

---

#### Q4: 如何实现事件研究法（Event Study）？

**答案**：

**代码实现**：
```python
import statsmodels.formula.api as smf

# 1. 创建相对时间变量
df['rel_time'] = df['time'] - df['treatment_time']

# 2. 创建事件时间虚拟变量（排除 -1 作为基准）
for k in range(-5, 6):  # 政策前5期到政策后5期
    if k != -1:  # 排除 -1 作为基准
        df[f'D_{k}'] = ((df['rel_time'] == k) & (df['treated'] == 1)).astype(int)

# 3. 回归
formula = 'Y ~ ' + ' + '.join([f'D_{k}' for k in range(-5, 6) if k != -1])
formula += ' + C(unit) + C(time)'  # 单位和时间固定效应

model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['unit']})

# 4. 提取系数并可视化
coeffs = [model.params[f'D_{k}'] if k != -1 else 0 for k in range(-5, 6)]
ses = [model.bse[f'D_{k}'] if k != -1 else 0 for k in range(-5, 6)]

# 绘制事件研究图
plt.errorbar(range(-5, 6), coeffs, yerr=1.96*np.array(ses))
plt.axhline(0, linestyle='--', color='gray')
plt.axvline(-0.5, linestyle='--', color='red')  # 政策实施时点
```

**解读**：
- **政策前（k < 0）**：系数应接近 0 且不显著（支持平行趋势）
- **政策后（k ≥ 0）**：系数显著偏离 0（政策有效）
- **动态效应**：可以看到效应如何随时间演变

**面试加分点**：
- 提到"event study 是 DID 在时间维度的分解"
- 提到"可以检验预期效应（anticipation）和滞后效应（persistence）"
- 强调聚类标准误的重要性

---

## 2. Synthetic Control Method (SCM)

### 核心原理

**一句话总结**：用多个对照单位的加权组合，合成一个与处理单位最相似的"虚拟"对照组。

### 高频面试题

#### Q5: 合成控制 vs DID，什么时候用哪个？

**答案**：

| 维度 | DID | 合成控制 |
|------|-----|----------|
| **处理单位数** | 多个 | 通常 1 个 |
| **对照组构建** | 简单分组 | 加权组合 |
| **关键假设** | 平行趋势 | 可以线性组合出反事实 |
| **推断方法** | 标准误、t 检验 | Placebo Tests |
| **适用场景** | 政策在多地实施 | 单一事件（某城市、某法案） |
| **灵活性** | 低（等权重） | 高（优化权重） |

**使用 DID**：
- ✅ 多个处理单位
- ✅ 处理时点一致
- ✅ 对照组和处理组趋势相似
- ✅ 需要控制更多协变量

**使用合成控制**：
- ✅ 单个或少数处理单位
- ✅ 找不到完美的对照
- ✅ 平行趋势假设存疑
- ✅ 关注特定事件的因果效应

**面试加分点**：
- DID 是合成控制的特例（等权重）
- 可以组合使用：Synthetic DID
- 提到 Abadie (2021) 的综述文章

---

#### Q6: 合成控制的权重如何估计？

**答案**：

**优化目标**：
$$
W^* = \arg\min_W \sum_{t=1}^{T_0} \left( Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt} \right)^2
$$

**约束条件**：
- $w_j \geq 0$ （非负）
- $\sum_{j} w_j = 1$ （权重和为 1）

**实现**：
```python
from scipy.optimize import minimize

def objective(w, treated_pre, donors_pre):
    synthetic = donors_pre @ w
    return np.sum((treated_pre - synthetic) ** 2)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(n_donors)]
w0 = np.ones(n_donors) / n_donors

result = minimize(objective, w0,
                  args=(treated_pre, donors_pre),
                  method='SLSQP',
                  bounds=bounds,
                  constraints=constraints)

weights = result.x
```

**扩展：协变量匹配**
- 不仅匹配结果变量的历史，还匹配协变量（GDP、人口等）
- 优化目标变为：$\min_W \|X_1 - X_0 W\|_V^2$
- $V$ 是权重矩阵，体现不同特征的重要性

**面试加分点**：
- 这是一个二次规划问题（Quadratic Programming）
- 权重通常很稀疏（很多为 0）
- 稀疏性是好事：避免过拟合，易于解释

---

#### Q7: 合成控制如何做统计推断？

**答案**：

**挑战**：
- 只有 1 个处理单位 → 无法用 t 检验
- 时间序列相关 → 标准误估计困难

**解决方案：Placebo Tests（安慰剂检验）**

**操作步骤**：

1. **假装每个供体都接受了处理**
2. **对每个"假处理"单位估计合成控制**
3. **计算"假效应"**
4. **比较真实效应和假效应的分布**

**代码**：
```python
# 1. 真实处理单位的效应
att_real = np.mean(Y_treated[T0:] - Y_synthetic[T0:])

# 2. 对每个供体做 Placebo
placebo_effects = []
for j in donors:
    # 假装第 j 个供体是处理单位
    Y_placebo_treated = Y_donors[:, j]
    Y_placebo_donors = np.delete(Y_donors, j, axis=1)

    # 估计合成控制
    sc_placebo = fit_synthetic_control(Y_placebo_treated, Y_placebo_donors)
    att_placebo = np.mean(Y_placebo_treated[T0:] - sc_placebo[T0:])

    placebo_effects.append(att_placebo)

# 3. 计算 p-value
p_value = np.mean([abs(e) >= abs(att_real) for e in placebo_effects])
```

**p-value 解释**：
- 有多少比例的单位的效应比真实处理单位还大？
- p < 0.05 → 真实效应显著

**面试加分点**：
- 提到 RMSPE 比值检验（Post/Pre）
- 提到"Pre-treatment fit"过滤（只保留拟合好的 Placebo）
- 知道 Abadie, Diamond & Hainmueller (2010, 2015) 的工作

---

## 3. Regression Discontinuity Design (RDD)

### 核心原理

**一句话总结**：如果处理分配由某个连续变量是否超过阈值决定，那么阈值附近的单位具有可比性。

### 高频面试题

#### Q8: Sharp RDD vs Fuzzy RDD 的区别？

**答案**：

**Sharp RDD**：
- **定义**：跨过阈值 → 100% 接受处理
- **识别**：$\tau = \lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]$
- **估计**：局部线性回归

**Fuzzy RDD**：
- **定义**：跨过阈值 → 接受处理的概率跳跃（但不是 100%）
- **识别**：
  $$\tau = \frac{\lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]}{\lim_{x \downarrow c} E[D|X=x] - \lim_{x \uparrow c} E[D|X=x]}$$
- **估计**：2SLS，用"跨过阈值"作为工具变量
- **解释**：LATE（Local Average Treatment Effect）

**例子**：
- **Sharp**：满 200 元 → 100% 获得优惠券
- **Fuzzy**：满 200 元 → 80% 获得优惠券（有些人拒绝）

**Fuzzy RDD 的实现**：
```python
# 第一阶段：D ~ 1{X >= c}
above_cutoff = (X >= cutoff).astype(int)
first_stage = smf.ols('treatment ~ above_cutoff + X + I(X**2)', data=df).fit()
D_hat = first_stage.fittedvalues

# 第二阶段：Y ~ D_hat
second_stage = smf.ols('outcome ~ D_hat + X + I(X**2)', data=df).fit()
tau_fuzzy = second_stage.params['D_hat']
```

**面试加分点**：
- Fuzzy RDD 本质上是 IV 估计
- 工具变量 Z = 1{X ≥ c}
- 估计的是 Compliers 的 LATE

---

#### Q9: RDD 的带宽（Bandwidth）如何选择？

**答案**：

**权衡（Bias-Variance Tradeoff）**：
- **带宽小 h**：
  - ✅ 偏差小（只用阈值附近的观测）
  - ❌ 方差大（样本量少）
- **带宽大 h**：
  - ✅ 方差小（样本量多）
  - ❌ 偏差大（包含远离阈值的观测）

**选择方法**：

**1. 规则 of Thumb（经验法则）**
```python
h = 1.84 * σ * n^(-1/5)
```

**2. 交叉验证（Cross-Validation）**
- Leave-one-out CV
- K-fold CV

**3. MSE-Optimal 带宽（Imbens & Kalyanaraman, 2012）**
```python
from rdd import rdd
bandwidth = rdd.optimal_bandwidth(X, Y, cutoff)
```

**4. CCT 带宽（Calonico, Cattaneo & Titiunik, 2014）**
```python
# 简化实现
def cct_bandwidth(X, Y, cutoff):
    # 估计条件方差
    sigma2_left = estimate_variance(X[X < cutoff], Y[X < cutoff])
    sigma2_right = estimate_variance(X[X >= cutoff], Y[X >= cutoff])

    # 估计二阶导数
    m2_left = estimate_second_derivative(X[X < cutoff], Y[X < cutoff])
    m2_right = estimate_second_derivative(X[X >= cutoff], Y[X >= cutoff])

    # MSE-optimal 公式
    C_K = 3.44  # Kernel 常数
    n = len(X)
    h_opt = C_K * (sigma2 / (n * m2^2))^(1/5)

    return h_opt
```

**实践建议**：
- 报告多个带宽下的结果（敏感性分析）
- 如果结果对带宽不敏感 → 稳健
- CCT 带宽是理论最优，但实践中可能不稳定

**面试加分点**：
- 提到"undersmoothing"（故意用小带宽减少偏差）
- 知道 `rdrobust` R 包
- 提到 robust bias-corrected 推断

---

#### Q10: RDD 的有效性检查（Validity Checks）有哪些？

**答案**：

**1. 连续性检查（Continuity Checks）**

**a) 协变量的连续性**
- 在阈值处，协变量不应该跳跃
- 如果跳跃 → 可能有其他机制在起作用

```python
for covariate in ['age', 'income', 'education']:
    rdd = RDD(cutoff=200, bandwidth=30)
    rdd.fit(X, df[covariate])
    print(f"{covariate} 在阈值处的跳跃: {rdd.tau_} (p={rdd.pvalue_})")
    # 应该都不显著
```

**b) 密度的连续性（McCrary Test）**
- 如果人们可以操纵 Running Variable → 密度会在阈值处跳跃
- 用 McCrary (2008) 检验

```python
from rdd import mccrary_test
p_value = mccrary_test(X, cutoff=200)
# p > 0.05 → 密度连续，没有操纵
```

**2. Placebo 检验**

**a) Placebo 截断点**
- 在非真实阈值处进行 RDD 估计
- 应该没有效应

```python
placebo_cutoffs = [150, 170, 190, 210, 230, 250]
for c in placebo_cutoffs:
    rdd = RDD(cutoff=c, bandwidth=30)
    rdd.fit(X, Y)
    # 应该都不显著
```

**b) Placebo 结果变量**
- 用不应该受影响的结果变量
- 应该没有效应

**3. 甜甜圈 RDD（Donut RDD）**
- 去掉阈值正上方和正下方的观测（可能被操纵）
- 如果结果稳健 → 更可信

```python
# 去掉 [c-δ, c+δ] 区间的观测
donut_df = df[(df['score'] < cutoff - 5) | (df['score'] > cutoff + 5)]
```

**面试加分点**：
- 知道 Lee (2008) 的边界论文
- 提到"local randomization"的视角
- 知道什么情况下 RDD 可能失效（操纵、反应性）

---

## 4. Instrumental Variables (IV)

### 核心原理

**一句话总结**：找一个只影响结果变量 Y 通过处理变量 X 的外生变量 Z，利用 Z 的变化识别 X 对 Y 的因果效应。

### 高频面试题

#### Q11: 工具变量的三个假设是什么？如何检验？

**答案**：

**三个假设**：

**1. 相关性（Relevance）**
- **定义**：Z 和 X 相关
- **数学**：Cov(Z, X) ≠ 0
- **检验**：
  ```python
  # 第一阶段 F 统计量
  first_stage = smf.ols('X ~ Z', data=df).fit()
  f_stat = first_stage.fvalue

  # 判断标准：F > 10 → 强工具变量
  ```
- **可检验**：✅ 可以直接检验

**2. 排除性（Exclusion Restriction）**
- **定义**：Z 只通过 X 影响 Y（Z 不直接影响 Y）
- **数学**：Z ⊥ Y | X
- **检验**：❌ **不可检验**（需要理论支持）
- **例外**：如果有多个 IV，可以用过度识别检验（Hansen J Test）

**3. 外生性（Exogeneity）**
- **定义**：Z 与未观测混淆因子 U 无关
- **数学**：Cov(Z, U) = 0
- **检验**：❌ **不可检验**（U 不可观测）
- **依赖**：理论、制度知识、自然实验

**检验工具**：

| 假设 | 可检验性 | 检验方法 |
|------|---------|---------|
| Relevance | ✅ | F 统计量 > 10 |
| Exclusion | ⚠️ | 过度识别检验（需多个 IV） |
| Exogeneity | ❌ | 理论论证 |

**面试加分点**：
- 强调"排除性是最难满足的"
- 提到"好的 IV 来自自然实验、随机化、制度特征"
- 知道什么是弱工具变量问题（F < 10）

---

#### Q12: 2SLS 的直觉是什么？手动实现一遍。

**答案**：

**直觉**：
1. **第一阶段**：把 X 分解成"外生部分"和"内生部分"
   - $\hat{X} = f(Z)$ ← 外生部分（只由 Z 决定）
   - $X - \hat{X}$ ← 内生部分（与 U 相关）

2. **第二阶段**：用外生部分 $\hat{X}$ 回归 Y
   - 因为 $\hat{X}$ 与 U 无关，所以估计是无偏的

**手动实现**：

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def two_stage_least_squares(Z, X, Y):
    """
    手动实现 2SLS

    参数:
        Z: 工具变量 (n,)
        X: 内生变量 (n,)
        Y: 结果变量 (n,)
    """
    # 第一阶段：X ~ Z
    first_stage = LinearRegression()
    first_stage.fit(Z.reshape(-1, 1), X)
    X_hat = first_stage.predict(Z.reshape(-1, 1))

    # 检查第一阶段强度
    r2 = first_stage.score(Z.reshape(-1, 1), X)
    f_stat = (r2 / (1 - r2)) * (len(X) - 2)
    print(f"First-stage F = {f_stat:.2f}")

    # 第二阶段：Y ~ X_hat
    second_stage = LinearRegression()
    second_stage.fit(X_hat.reshape(-1, 1), Y)
    beta_2sls = second_stage.coef_[0]

    # Wald 估计量（等价）
    beta_wald = np.cov(Z, Y)[0,1] / np.cov(Z, X)[0,1]

    return {
        'beta_2sls': beta_2sls,
        'beta_wald': beta_wald,
        'first_stage_f': f_stat
    }
```

**为什么有效？**

OLS 估计 X → Y 时：
$$\beta_{OLS} = \frac{Cov(X, Y)}{Var(X)} = \beta + \frac{Cov(X, U)}{Var(X)} \quad \text{(有偏！)}$$

2SLS 估计：
$$\beta_{2SLS} = \frac{Cov(\hat{X}, Y)}{Var(\hat{X})} = \frac{Cov(Z, Y)}{Cov(Z, X)} = \beta \quad \text{(无偏)}$$

因为 $Cov(Z, U) = 0$（外生性假设）。

**面试加分点**：
- 2SLS 的标准误需要调整（不能直接用第二阶段的 SE）
- 正确的 SE 需要考虑第一阶段的不确定性
- 实践中用 `ivreg` 或 `linearmodels.IV2SLS`

---

#### Q13: 什么是 LATE（局部平均处理效应）？

**答案**：

**定义**：
IV 估计的是 **Compliers** 的平均处理效应（ATE），不是所有人的 ATE。

**人群分类（Imbens & Angrist, 1994）**：

根据 $(D_i(Z=0), D_i(Z=1))$，可以分为 4 类人：

| 类型 | $D(Z=0)$ | $D(Z=1)$ | 描述 | 例子（兵役 IV） |
|------|----------|----------|------|----------------|
| **Never-takers** | 0 | 0 | Z 不影响 D | 有健康问题，无法入伍 |
| **Compliers** | 0 | 1 | Z 决定 D | 抽中就去，未抽中不去 |
| **Always-takers** | 1 | 1 | 总是 D=1 | 志愿入伍 |
| **Defiers** | 1 | 0 | Z 反向影响 D | （通常假设不存在） |

**IV 估计的是什么？**

$$\tau_{IV} = E[Y_i(1) - Y_i(0) | \text{Complier}] = \text{LATE}$$

**直觉**：
- Always-takers: Z 的变化不影响他们的 D，所以无法识别效应
- Never-takers: 同上
- Compliers: Z 的变化改变了他们的 D，所以能识别效应
- IV 估计的是 Compliers 的 ATE

**外推性问题**：
- LATE ≠ ATE（除非所有人都是 Compliers）
- 如果 Compliers 很特殊，LATE 可能不能推广到总体

**例子：征兵抽签 & 教育回报率**
- **Z**：是否被抽中征兵
- **X**：教育年限
- **Y**：收入
- **Compliers**：被抽中就去（中断教育），未抽中就继续读书的人
- **LATE**：这部分人多读一年书的收入回报
- **注意**：这可能不是"所有人"的教育回报率

**面试加分点**：
- 提到 Monotonicity Assumption（单调性：没有 Defiers）
- 知道 LATE 的外部有效性局限
- 提到 Fuzzy RDD 估计的也是 LATE

---

#### Q14: 弱工具变量（Weak IV）有什么问题？如何检测？

**答案**：

**问题**：

1. **有限样本偏差**：
   - 2SLS 估计量向 OLS 偏移
   - 即使 n → ∞，偏差也不一定消失（如果 F 太小）

2. **标准误失效**：
   - 渐近标准误严重低估真实标准误
   - 置信区间覆盖率远低于名义水平（如 95%）

3. **检验失效**：
   - t 检验拒绝率远高于名义水平
   - 容易出现假阳性

**检测方法**：

**1. First-Stage F 统计量**
```python
first_stage = smf.ols('X ~ Z + controls', data=df).fit()
f_stat = first_stage.fvalue

# 判断标准（Stock & Yogo, 2005）
if f_stat > 10:
    print("✓ 强工具变量")
elif f_stat > 5:
    print("⚠ 中等强度，需谨慎")
else:
    print("✗ 弱工具变量，结果不可信")
```

**2. Cragg-Donald 统计量（多个内生变量）**
- 推广的 F 统计量
- 临界值表：Stock & Yogo (2005)

**解决方案**：

**1. Anderson-Rubin 检验**
- 在弱 IV 下仍然有效（不依赖渐近理论）
- 但功效较低（更保守）

```python
def anderson_rubin_test(Z, X, Y, beta_0):
    """检验 H0: beta = beta_0"""
    Y_tilde = Y - beta_0 * X
    model = smf.ols('Y_tilde ~ Z', data=df).fit()
    f_stat = model.fvalue
    p_value = model.f_pvalue
    return p_value
```

**2. LIML（Limited Information Maximum Likelihood）**
- 比 2SLS 更稳健（在弱 IV 下偏差更小）
- 但方差更大

**3. 找更强的工具变量**
- 增加工具变量的数量
- 找与 X 相关性更强的 Z

**面试加分点**：
- 提到"Many weak instruments"问题（很多弱 IV 也无济于事）
- 知道 Staiger & Stock (1997) 的临界值 3.84
- 提到有效估计：JIVE, UJIVE, MBTSLS

---

## 5. 方法对比与选择

### Q15: 这四种方法如何选择？

**答案**：

| 方法 | 适用场景 | 核心假设 | 优点 | 缺点 |
|------|---------|---------|------|------|
| **DID** | • 政策在多个单位实施<br>• 有清晰的前/后、处理/对照 | 平行趋势 | • 简单直观<br>• 易于实施<br>• 可控制协变量 | • 假设强<br>• 对趋势敏感 |
| **SCM** | • 单一事件<br>• 找不到完美对照<br>• 有多个潜在对照单位 | 可线性组合 | • 灵活（优化权重）<br>• 直观可视化<br>• 不依赖平行趋势 | • 需要多个对照<br>• 大样本推断困难 |
| **RDD** | • 处理由阈值决定<br>• 阈值附近不可操纵<br>• 运行变量连续 | 阈值附近可比 | • 识别强<br>• 内部效度高<br>• 不需要随机化 | • 外部效度弱（LATE）<br>• 带宽敏感<br>• 需要大样本 |
| **IV** | • 存在混淆<br>• 有合理的外生冲击<br>• 可以论证排除性 | Relevance<br>Exclusion<br>Exogeneity | • 处理内生性<br>• 理论基础强 | • 找好 IV 很难<br>• 弱 IV 问题<br>• 估计 LATE |

**决策树**：

```
START
├─ 有随机实验？
│  ├─ 是 → 不需要准实验，直接比较均值
│  └─ 否 ↓
│
├─ 处理分配由阈值决定？
│  ├─ 是 → 用 RDD
│  └─ 否 ↓
│
├─ 有明确的前/后、处理/对照？
│  ├─ 是 ↓
│  │  ├─ 多个处理单位 + 趋势相似？
│  │  │  ├─ 是 → 用 DID
│  │  │  └─ 否 ↓
│  │  └─ 单个处理单位 + 多个潜在对照？
│  │     └─ 是 → 用 Synthetic Control
│  └─ 否 ↓
│
└─ 有合理的工具变量？
   ├─ 是 → 用 IV
   └─ 否 → 考虑观察性方法（Matching, IPW 等）
```

**面试加分点**：
- 可以组合使用（如 DID + Matching, SCM + Placebo Tests）
- 最好用多种方法验证（Robustness Check）
- 每种方法都有局限性，关键是论证假设的合理性

---

## 6. 从零实现核心算法

### DID 估计器

```python
import numpy as np
import pandas as pd

def did_estimator(df, outcome, treat_col, post_col):
    """
    手动实现 DID 估计

    参数:
        df: DataFrame
        outcome: 结果变量列名
        treat_col: 处理组指示变量
        post_col: 政策后指示变量
    """
    # 计算四个均值
    y_treat_post = df[df[treat_col] & df[post_col]][outcome].mean()
    y_treat_pre = df[df[treat_col] & ~df[post_col]][outcome].mean()
    y_control_post = df[~df[treat_col] & df[post_col]][outcome].mean()
    y_control_pre = df[~df[treat_col] & ~df[post_col]][outcome].mean()

    # DID 估计量
    did = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre)

    # 标准误（假设同方差）
    # 使用 delta method 的简化版本
    n_treat = df[df[treat_col]].shape[0]
    n_control = df[~df[treat_col]].shape[0]

    var_treat = df[df[treat_col]][outcome].var()
    var_control = df[~df[treat_col]][outcome].var()

    se = np.sqrt(var_treat / n_treat + var_control / n_control)

    return {
        'DID估计': did,
        '标准误': se,
        't统计量': did / se,
        'p值': 2 * (1 - stats.norm.cdf(abs(did / se)))
    }
```

### 合成控制估计器

```python
from scipy.optimize import minimize

def synthetic_control(treated, donors, treatment_period):
    """
    手动实现合成控制

    参数:
        treated: 处理单位时间序列 (T,)
        donors: 供体池时间序列矩阵 (T, J)
        treatment_period: 处理开始时间索引
    """
    # 前处理期数据
    treated_pre = treated[:treatment_period]
    donors_pre = donors[:treatment_period, :]

    # 优化目标
    def objective(w):
        synthetic = donors_pre @ w
        return np.sum((treated_pre - synthetic) ** 2)

    # 约束
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(donors.shape[1])]
    w0 = np.ones(donors.shape[1]) / donors.shape[1]

    # 求解
    result = minimize(objective, w0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

    # 生成合成控制
    synthetic = donors @ result.x

    # ATT
    att = np.mean(treated[treatment_period:] - synthetic[treatment_period:])

    return {
        '权重': result.x,
        '合成控制': synthetic,
        'ATT': att
    }
```

### RDD 估计器

```python
def rdd_estimator(X, Y, cutoff, bandwidth, polynomial_order=1):
    """
    手动实现 Sharp RDD

    参数:
        X: Running variable
        Y: Outcome
        cutoff: 阈值
        bandwidth: 带宽
        polynomial_order: 多项式阶数
    """
    # 中心化
    X_centered = X - cutoff

    # 选择带宽内的观测
    mask = abs(X_centered) <= bandwidth
    X_bw = X_centered[mask]
    Y_bw = Y[mask]

    # 左右两侧
    left_mask = X_bw < 0
    right_mask = X_bw >= 0

    # 多项式特征
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=polynomial_order)

    # 左侧回归
    X_left = poly.fit_transform(X_bw[left_mask].reshape(-1, 1))
    model_left = LinearRegression().fit(X_left, Y_bw[left_mask])
    y_left_0 = model_left.predict(poly.transform([[0]]))[0]

    # 右侧回归
    X_right = poly.fit_transform(X_bw[right_mask].reshape(-1, 1))
    model_right = LinearRegression().fit(X_right, Y_bw[right_mask])
    y_right_0 = model_right.predict(poly.transform([[0]]))[0]

    # RDD 估计量
    tau = y_right_0 - y_left_0

    # 标准误（简化版，实际应该用 robust SE）
    resid_left = Y_bw[left_mask] - model_left.predict(X_left)
    resid_right = Y_bw[right_mask] - model_right.predict(X_right)

    sigma2_left = np.var(resid_left)
    sigma2_right = np.var(resid_right)

    n_left = left_mask.sum()
    n_right = right_mask.sum()

    se = np.sqrt(sigma2_left / n_left + sigma2_right / n_right)

    return {
        '处理效应': tau,
        '标准误': se,
        't统计量': tau / se,
        'p值': 2 * (1 - stats.norm.cdf(abs(tau / se)))
    }
```

### 2SLS 估计器

```python
def two_stage_least_squares(Z, X, Y):
    """
    手动实现 2SLS

    参数:
        Z: 工具变量 (n,) 或 (n, k)
        X: 内生变量 (n,)
        Y: 结果变量 (n,)
    """
    # 确保是矩阵形式
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = len(Y)

    # 第一阶段：X ~ Z
    # X = Z * gamma + v
    # gamma = (Z'Z)^{-1} Z'X
    gamma = np.linalg.inv(Z.T @ Z) @ Z.T @ X
    X_hat = Z @ gamma

    # 第一阶段 F 统计量
    ss_res = np.sum((X - X_hat)**2)
    ss_tot = np.sum((X - np.mean(X))**2)
    r2 = 1 - ss_res / ss_tot
    f_stat = (r2 / (1 - r2)) * (n - Z.shape[1] - 1) / Z.shape[1]

    # 第二阶段：Y ~ X_hat
    # Y = X_hat * beta + epsilon
    # beta = (X_hat'X_hat)^{-1} X_hat'Y
    beta_2sls = np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ Y

    # 计算残差
    Y_hat = X_hat @ beta_2sls
    residuals = Y - Y_hat

    # 2SLS 标准误（需要调整）
    # Var(beta) = sigma^2 * (X'P_Z X)^{-1}
    # 其中 P_Z = Z(Z'Z)^{-1}Z'
    sigma2 = np.sum(residuals**2) / (n - X.shape[1])

    P_Z = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    var_beta = sigma2 * np.linalg.inv(X.T @ P_Z @ X)
    se_2sls = np.sqrt(np.diag(var_beta))

    return {
        'beta_2sls': beta_2sls[0, 0],
        'se': se_2sls[0],
        't_stat': beta_2sls[0, 0] / se_2sls[0],
        'first_stage_f': f_stat,
        'weak_iv': f_stat < 10
    }
```

---

## 7. 真实面试题示例

### 案例题 1：美团外卖新功能上线

**背景**：
美团在 2023 年 7 月在北京和上海试点了「无接触配送」功能。你是数据科学家，需要评估这个功能对订单量的影响。

**可用数据**：
- 2023 年 1-12 月，20 个城市的月度订单量
- 北京、上海在 7 月上线功能
- 其他 18 个城市未上线

**问题**：
1. 你会用什么方法？为什么？
2. 核心假设是什么？如何检验？
3. 如果假设不满足怎么办？

**参考答案**：

**1. 方法选择**

我会先尝试 **DID**，如果平行趋势不满足，再考虑 **合成控制**。

理由：
- ✅ 有两个处理单位（北京、上海）
- ✅ 有清晰的前/后时间点
- ✅ 有 18 个潜在对照城市
- ✅ 面板数据结构

**2. 核心假设与检验**

**DID 假设：平行趋势**

检验方法：
```python
# 方法 1：图形化检验
# 绘制 1-6 月的订单量趋势
fig, ax = plt.subplots()
for city in ['北京', '上海']:
    df_city = df[df['city'] == city]
    ax.plot(df_city['month'][:6], df_city['orders'][:6], label=city)

for city in other_cities:
    df_city = df[df['city'] == city]
    ax.plot(df_city['month'][:6], df_city['orders'][:6], alpha=0.3, color='gray')

# 如果趋势平行 → 支持假设

# 方法 2：Event Study
# 估计政策前各月的"假效应"
for month in range(1, 7):
    df[f'lead_{month}'] = df['treat'] * (df['month'] == month)

model = smf.ols('orders ~ treat + C(month) + ' +
                ' + '.join([f'lead_{m}' for m in range(1, 6)]),
                data=df).fit()

# 如果 lead 系数不显著 → 支持平行趋势
```

**3. 假设不满足的应对**

**方案 A：合成控制**
```python
# 用 18 个对照城市的加权组合作为"合成北京/上海"
from synthetic_control import SyntheticControl

sc_beijing = SyntheticControl(treatment_time=7)
sc_beijing.fit(
    treated=df[df['city']=='北京']['orders'],
    donors=df[df['city'].isin(other_cities)].pivot(
        index='month', columns='city', values='orders'
    )
)

# 优点：不依赖平行趋势，权重是数据驱动的
# 缺点：只有 2 个处理单位，推断较弱
```

**方案 B：控制城市特定趋势**
```python
# 允许每个城市有不同的线性趋势
model = smf.ols('orders ~ treat + C(month) + C(city) + ' +
                'treat*month + city*month + treat_post',
                data=df).fit()
```

**方案 C：匹配 + DID**
```python
# 先用 PSM 找到与北京、上海最相似的城市
# 再在匹配样本上做 DID
from sklearn.neighbors import NearestNeighbors

# 匹配协变量：人均 GDP、人口、消费水平等
nn = NearestNeighbors(n_neighbors=5)
nn.fit(df_covariates[df['city'].isin(other_cities)])

matched_cities = nn.kneighbors(df_covariates[df['city'].isin(['北京', '上海'])])
```

---

### 案例题 2：会员等级门槛的效应

**背景**：
某电商平台的会员体系：
- 累计消费 < 1000 元：普通会员
- 累计消费 ≥ 1000 元：金卡会员（享受折扣）

你需要评估金卡会员资格对后续消费的影响。

**问题**：
1. 直接比较金卡和普通会员的消费差异有什么问题？
2. 你会用什么方法？
3. 需要哪些 validity checks？

**参考答案**：

**1. 直接比较的问题**

```python
# 错误做法
avg_gold = df[df['total_spending'] >= 1000]['future_spending'].mean()
avg_regular = df[df['total_spending'] < 1000]['future_spending'].mean()
effect = avg_gold - avg_regular  # ❌ 有偏！
```

**问题**：
- **选择偏差**：消费超过 1000 的人本身就更爱消费（能力、偏好不同）
- **混淆因素**：收入、年龄、地域等
- **反向因果**：可能是因为他们本来就会多消费，所以才达到 1000

**2. 方法选择：RDD**

这是典型的 **Regression Discontinuity Design** 场景：
- 处理分配由阈值决定（1000 元）
- 阈值附近的人应该是相似的（消费 995 vs 1005）

```python
from rdd import RDD

# 1. 准备数据
df['above_1000'] = (df['total_spending'] >= 1000).astype(int)
df['running_var'] = df['total_spending'] - 1000  # 中心化

# 2. RDD 估计
rdd = RDD(cutoff=0, bandwidth=200, polynomial_order=1)
rdd.fit(df['running_var'], df['future_spending'])

print(f"金卡效应: {rdd.tau_:.2f} 元")
print(f"p-value: {rdd.pvalue_:.4f}")

# 3. 可视化
rdd.plot()
```

**直觉**：
- 消费 999 元的人和消费 1001 元的人应该非常相似
- 唯一的区别是后者获得了金卡
- 两者后续消费的差异可以归因于金卡资格

**3. Validity Checks**

**Check 1：密度检验（McCrary Test）**
```python
# 检查是否有人故意操纵到 1000 以上
from rdd import mccrary_test

p_value = mccrary_test(df['running_var'], cutoff=0)

if p_value < 0.05:
    print("⚠️ 密度在阈值处不连续，可能有操纵")
else:
    print("✓ 密度连续，没有操纵证据")
```

**Check 2：协变量连续性**
```python
# 在阈值处，协变量不应该跳跃
for covar in ['age', 'income', 'city_tier']:
    rdd_covar = RDD(cutoff=0, bandwidth=200)
    rdd_covar.fit(df['running_var'], df[covar])

    print(f"{covar} 在阈值处的跳跃: {rdd_covar.tau_:.3f} (p={rdd_covar.pvalue_:.3f})")
    # 应该都不显著
```

**Check 3：Placebo 截断点**
```python
# 在非真实阈值处不应该有跳跃
placebo_cutoffs = [-500, -200, 200, 500]

for c in placebo_cutoffs:
    rdd_placebo = RDD(cutoff=c, bandwidth=200)
    rdd_placebo.fit(df['running_var'], df['future_spending'])

    print(f"Placebo cutoff {c}: tau={rdd_placebo.tau_:.2f}, p={rdd_placebo.pvalue_:.3f}")
    # 应该都不显著
```

**Check 4：带宽敏感性**
```python
# 结果应该对带宽选择不太敏感
bandwidths = [100, 150, 200, 250, 300]

for h in bandwidths:
    rdd_h = RDD(cutoff=0, bandwidth=h)
    rdd_h.fit(df['running_var'], df['future_spending'])

    print(f"Bandwidth {h}: tau={rdd_h.tau_:.2f}")

# 如果变化不大 → 稳健
```

---

## 8. 常见陷阱与误区

### 陷阱 1：混淆 ATT 和 ATE

**错误**：
> "DID 估计的是 ATE"

**正确**：
- DID 估计的是 **ATT**（Average Treatment Effect on the Treated）
- 只有当处理效应同质时，ATT = ATE

**例子**：
```python
# 真实的 DGP
def simulate_heterogeneous():
    # 处理效应取决于基线水平
    baseline = np.random.normal(100, 20, 1000)

    # 高基线的人被选入处理组
    treated = baseline > 110

    # 处理效应：对基线高的人效应更大
    effect = np.where(treated, 0.1 * baseline, 0)

    y = baseline + effect + np.random.normal(0, 5, 1000)

    return {
        'ATT': effect[treated].mean(),  # 只针对处理组
        'ATE': effect.mean(),           # 全体平均
        '差异': effect[treated].mean() - effect.mean()
    }

# ATT ≠ ATE when effect is heterogeneous
```

---

### 陷阱 2：后处理偏差（Post-treatment Bias）

**错误**：
> "在 DID 中控制政策后才出现的变量"

**例子**：
```python
# ❌ 错误：控制了 post-treatment 变量
model = smf.ols(
    'revenue ~ treat + post + treat_post + new_feature_usage',
    data=df
).fit()

# new_feature_usage 是政策实施后才产生的
# 它本身可能是政策效应的一部分！
```

**正确做法**：
- 只控制 **pre-treatment** 变量
- Post-treatment 变量可能是中介变量（mediator）
- 如果要分析机制，用 mediation analysis

---

### 陷阱 3：过拟合的合成控制

**错误**：
> "在整个时间段（包括政策后）上拟合合成控制"

**正确**：
- **只在前处理期拟合权重**
- 政策后的数据用来评估效应，不能用来拟合

```python
# ❌ 错误
sc = SyntheticControl(treatment_period=T0)
sc.fit(treated, donors)  # 默认用全部数据

# ✓ 正确
sc = SyntheticControl(treatment_period=T0)
sc.fit(treated[:T0], donors[:T0, :])  # 只用前处理期
```

---

### 陷阱 4：忽略 RDD 的 LATE 性质

**错误**：
> "RDD 估计的是全体的 ATE"

**正确**：
- RDD 估计的是 **阈值附近**的 LATE
- 不能推广到远离阈值的人群

**例子**：
```python
# 会员门槛 RDD
# 估计的是"接近 1000 元消费"的人获得金卡的效应
# 不是"所有人"获得金卡的效应

# 消费 10000 元的人即使没有金卡，也会继续高消费
# 他们的处理效应可能完全不同
```

---

## 9. 推荐资源

### 必读论文

**DID**:
1. Bertrand, Duflo & Mullainathan (2004) - "How Much Should We Trust DID?"
2. Goodman-Bacon (2021) - "Difference-in-differences with variation in treatment timing"
3. Callaway & Sant'Anna (2021) - "Difference-in-Differences with multiple time periods"

**Synthetic Control**:
1. Abadie & Gardeazabal (2003) - "The Economic Costs of Conflict"
2. Abadie, Diamond & Hainmueller (2010) - "Synthetic Control Methods"
3. Abadie (2021) - "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects"

**RDD**:
1. Lee & Lemieux (2010) - "Regression Discontinuity Designs in Economics"
2. Imbens & Lemieux (2008) - "Regression discontinuity designs: A guide to practice"
3. Cattaneo, Idrobo & Titiunik (2019) - *A Practical Introduction to RDD* (书)

**IV**:
1. Angrist & Krueger (1991) - "Does Compulsory School Attendance Affect Schooling and Earnings?"
2. Angrist, Imbens & Rubin (1996) - "Identification of Causal Effects Using IV"
3. Stock & Yogo (2005) - "Testing for Weak Instruments in Linear IV Regression"

### Python 包

- `linearmodels`: IV 估计
- `pyfixest`: 高性能面板数据回归
- `causalimpact`: 时间序列因果推断（Google 的贝叶斯方法）
- `statsmodels`: DID, 回归, 时间序列

### 在线课程

- Scott Cunningham - *Causal Inference: The Mixtape*
- Matheus Facure - *Causal Inference for The Brave and True*
- Nick Huntington-Klein - *The Effect*

---

**Good luck with your interview! 🎉**
