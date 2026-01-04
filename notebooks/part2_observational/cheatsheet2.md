# Part 2: 观察性研究方法 Cheatsheet

> 面试速查手册：核心公式 + 2分钟代码实现 + 高频面试题

---

## 📝 2分钟代码实现题

### 1. 倾向得分估计 (Propensity Score)

```python
from sklearn.linear_model import LogisticRegression

def estimate_propensity_score(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    估计倾向得分 e(X) = P(T=1|X)
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, T)
    propensity = model.predict_proba(X)[:, 1]
    return propensity
```

### 2. 倾向得分匹配 (PSM)

```python
from sklearn.neighbors import NearestNeighbors

def psm_matching(propensity: np.ndarray, treatment: np.ndarray,
                 n_neighbors: int = 1, caliper: float = None):
    """
    执行倾向得分匹配
    """
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    # 使用 KNN 找最近邻
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(propensity[control_idx].reshape(-1, 1))
    distances, indices = knn.kneighbors(propensity[treated_idx].reshape(-1, 1))

    # 应用卡尺
    matched_treated = []
    matched_control = []

    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if caliper is None or dist[0] <= caliper:
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[idx[0]])

    return np.array(matched_treated), np.array(matched_control)
```

### 3. 标准化均值差 (SMD) 计算

```python
def compute_smd(X_treated: np.ndarray, X_control: np.ndarray) -> np.ndarray:
    """
    计算标准化均值差
    SMD = (mean_treated - mean_control) / pooled_std
    判断标准: |SMD| < 0.1 表示平衡良好
    """
    mean_treated = X_treated.mean(axis=0)
    mean_control = X_control.mean(axis=0)
    mean_diff = mean_treated - mean_control

    var_treated = X_treated.var(axis=0)
    var_control = X_control.var(axis=0)
    pooled_std = np.sqrt((var_treated + var_control) / 2)

    smd = mean_diff / pooled_std
    return smd
```

### 4. IPW 权重计算

```python
def compute_ipw_weights(propensity: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """
    计算 IPW 权重
    处理组: w = 1/e(X)
    控制组: w = 1/(1-e(X))
    """
    # 裁剪倾向得分避免极端权重
    propensity_clipped = np.clip(propensity, 0.01, 0.99)

    # 计算权重
    weights = (treatment / propensity_clipped +
               (1 - treatment) / (1 - propensity_clipped))

    return weights
```

### 5. IPW 估计 ATE (Hajek 估计器)

```python
def estimate_ate_ipw(Y: np.ndarray, treatment: np.ndarray,
                     weights: np.ndarray) -> Tuple[float, float]:
    """
    使用 IPW 估计 ATE
    """
    treated_mask = treatment == 1
    control_mask = treatment == 0

    # Hajek 估计器（归一化权重）
    y1_weighted = (Y[treated_mask] * weights[treated_mask]).sum() / weights[treated_mask].sum()
    y0_weighted = (Y[control_mask] * weights[control_mask]).sum() / weights[control_mask].sum()

    ate = y1_weighted - y0_weighted

    # 标准误（影响函数方法）
    n = len(Y)
    influence_1 = np.zeros(n)
    influence_1[treated_mask] = (Y[treated_mask] - y1_weighted) * weights[treated_mask]

    influence_0 = np.zeros(n)
    influence_0[control_mask] = (Y[control_mask] - y0_weighted) * weights[control_mask]

    influence = influence_1 - influence_0
    se = np.sqrt(np.var(influence) / n)

    return ate, se
```

### 6. 稳定权重计算

```python
def compute_stabilized_weights(propensity: np.ndarray,
                               treatment: np.ndarray) -> np.ndarray:
    """
    计算稳定权重
    w_stab = P(T) / e(X) for treated
    w_stab = (1-P(T)) / (1-e(X)) for control
    """
    marginal_prob = treatment.mean()
    propensity_clipped = np.clip(propensity, 0.01, 0.99)

    weights_stab = np.zeros(len(treatment))
    weights_stab[treatment == 1] = marginal_prob / propensity_clipped[treatment == 1]
    weights_stab[treatment == 0] = (1 - marginal_prob) / (1 - propensity_clipped[treatment == 0])

    return weights_stab
```

### 7. 有效样本量 (ESS) 计算

```python
def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    计算有效样本量
    ESS = (sum(w))^2 / sum(w^2)
    """
    ess = (weights.sum()) ** 2 / (weights ** 2).sum()
    return ess
```

### 8. AIPW 估计器实现

```python
def estimate_ate_aipw(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                      propensity: np.ndarray, mu_1: np.ndarray,
                      mu_0: np.ndarray) -> Tuple[float, float]:
    """
    使用 AIPW 估计 ATE

    AIPW = E[(mu_1 - mu_0) + T*(Y - mu_1)/e - (1-T)*(Y - mu_0)/(1-e)]
    """
    propensity_clipped = np.clip(propensity, 0.01, 0.99)

    # 第一项：结果模型预测的差异
    term1 = mu_1 - mu_0

    # 第二项：处理组的 IPW 修正
    term2 = T * (Y - mu_1) / propensity_clipped

    # 第三项：控制组的 IPW 修正
    term3 = (1 - T) * (Y - mu_0) / (1 - propensity_clipped)

    # AIPW 得分
    aipw_scores = term1 + term2 - term3

    # ATE 估计和标准误
    ate = aipw_scores.mean()
    se = aipw_scores.std() / np.sqrt(len(Y))

    return ate, se
```

### 9. 结果模型估计

```python
from sklearn.linear_model import Ridge

def estimate_outcome_models(X: np.ndarray, T: np.ndarray,
                           Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    估计结果模型
    mu_1(X) = E[Y|X, T=1]
    mu_0(X) = E[Y|X, T=0]
    """
    treated_mask = T == 1
    control_mask = T == 0

    # 训练处理组的结果模型
    model_1 = Ridge(alpha=1.0)
    model_1.fit(X[treated_mask], Y[treated_mask])
    mu_1 = model_1.predict(X)

    # 训练控制组的结果模型
    model_0 = Ridge(alpha=1.0)
    model_0.fit(X[control_mask], Y[control_mask])
    mu_0 = model_0.predict(X)

    return mu_1, mu_0
```

### 10. Double ML (Cross-fitting)

```python
from sklearn.model_selection import KFold

def double_ml_plr(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                  n_folds: int = 5) -> Dict:
    """
    Double Machine Learning for Partially Linear Model
    """
    n = len(Y)
    Y_residuals = np.zeros(n)
    T_residuals = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # 训练结果模型 g(X)
        g_model = Ridge(alpha=1.0)
        g_model.fit(X_train, Y_train)
        g_pred = g_model.predict(X_test)

        # 训练倾向得分模型 m(X)
        m_model = LogisticRegression(max_iter=1000, C=0.1)
        m_model.fit(X_train, T_train)
        m_pred = m_model.predict_proba(X_test)[:, 1]

        # 计算残差
        Y_residuals[test_idx] = Y_test - g_pred
        T_residuals[test_idx] = T_test - m_pred

    # 用残差估计 tau
    tau_hat = (Y_residuals * T_residuals).sum() / (T_residuals ** 2).sum()

    # 标准误
    psi = (Y_residuals - tau_hat * T_residuals) * T_residuals
    J = (T_residuals ** 2).mean()
    var_tau = (psi ** 2).mean() / (n * J ** 2)
    se = np.sqrt(var_tau)

    return {'tau': tau_hat, 'se': se}
```

### 11. E-value 计算

```python
def compute_e_value(observed_rr: float, ci_lower: float = None) -> Dict:
    """
    计算 E-value (敏感性分析)
    E = RR + sqrt(RR * (RR - 1))
    """
    # 确保 RR >= 1
    if observed_rr < 1:
        observed_rr = 1 / observed_rr

    # 计算 E-value
    e_value = observed_rr + np.sqrt(observed_rr * (observed_rr - 1))

    result = {'e_value': e_value}

    # 置信区间下界的 E-value
    if ci_lower is not None:
        if ci_lower < 1:
            ci_lower = 1 / ci_lower
        e_value_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
        result['e_value_ci'] = e_value_ci

    return result
```

---

## 🎯 核心公式速查

### 倾向得分方法

**倾向得分定义**
$$e(X) = P(T=1 | X)$$

**Rosenbaum & Rubin 定理**
$$(Y(0), Y(1)) \perp T | X \Rightarrow (Y(0), Y(1)) \perp T | e(X)$$

**PSM 估计 ATT**
$$\widehat{ATT} = \frac{1}{N_T} \sum_{i: T_i=1} \left( Y_i - \frac{1}{M_i} \sum_{j \in \mathcal{M}(i)} Y_j \right)$$

**标准化均值差 (SMD)**
$$\text{SMD} = \frac{\bar{X}_{\text{treated}} - \bar{X}_{\text{control}}}{\sqrt{(s^2_{\text{treated}} + s^2_{\text{control}})/2}}$$

判断标准: $|\text{SMD}| < 0.1$ 表示平衡良好

### IPW 方法

**IPW 权重**
$$w_i = \frac{T_i}{e(X_i)} + \frac{1-T_i}{1-e(X_i)}$$

**Horvitz-Thompson 估计器**
$$\hat{E}[Y(1)] = \frac{1}{n}\sum_{i=1}^{n} \frac{T_i Y_i}{e(X_i)}$$
$$\hat{E}[Y(0)] = \frac{1}{n}\sum_{i=1}^{n} \frac{(1-T_i) Y_i}{1-e(X_i)}$$
$$\widehat{ATE} = \hat{E}[Y(1)] - \hat{E}[Y(0)]$$

**稳定权重**
$$w_i^{\text{stab}} = \frac{P(T=T_i)}{P(T=T_i|X_i)}$$

**有效样本量 (ESS)**
$$ESS = \frac{(\sum w_i)^2}{\sum w_i^2}$$

### 双重稳健方法

**AIPW 估计器**
$$\hat{\tau}_{AIPW} = \frac{1}{n}\sum_{i=1}^{n}\left[(\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}\right]$$

**双重稳健性质**
- 如果 $\hat{e}(X)$ 正确 OR $\hat{\mu}(X)$ 正确 → 估计一致
- 两个都正确 → 效率最优
- 两个都错 → 一般有偏

### Double ML

**部分线性模型**
$$Y = \tau \cdot T + g(X) + \epsilon$$
$$T = m(X) + \eta$$

**DML 估计器**
$$\hat{\tau}_{DML} = \frac{\sum_{i=1}^{n}(Y_i - \hat{g}_{-k(i)}(X_i))(T_i - \hat{m}_{-k(i)}(X_i))}{\sum_{i=1}^{n}(T_i - \hat{m}_{-k(i)}(X_i))^2}$$

其中 $\hat{g}_{-k(i)}$ 表示在不包含第 $i$ 个样本的折上训练的模型

**Neyman 正交性**
$$\frac{\partial}{\partial \eta} E[\psi(W; \tau_0, \eta)] \Big|_{\eta=\eta_0} = 0$$

### 敏感性分析

**Rosenbaum 敏感性参数 Γ**
$$\frac{1}{\Gamma} \leq \frac{P(T_i=1|X)}{P(T_j=1|X)} \leq \Gamma$$

**E-value 公式**
$$E = RR + \sqrt{RR \times (RR - 1)}$$

E-value 解读:
- E < 1.5: 结论非常脆弱
- 1.5 ≤ E < 2.5: 结论中等稳健
- 2.5 ≤ E < 4.0: 结论较为稳健
- E ≥ 4.0: 结论非常稳健

---

## 💼 高频面试题

### Q1: 什么是倾向得分？为什么可以用它控制混淆？

**答案要点**:
- 倾向得分是给定协变量 X，个体接受处理的概率 $e(X) = P(T=1|X)$
- **Rosenbaum & Rubin 定理**: 如果满足无混淆假设 $(Y(0), Y(1)) \perp T | X$，那么 $(Y(0), Y(1)) \perp T | e(X)$
- **维度缩减**: 把高维的 X 压缩成一维的 e(X)
- **直观理解**: 倾向得分相同的个体，在"接受处理的倾向"上是一样的，处理组和控制组的协变量分布相同

### Q2: PSM 的局限性是什么？

**答案要点**:
1. **只能控制观测到的混淆变量**: 如果存在未观测的混淆，PSM 无能为力
2. **依赖倾向得分模型的正确性**: 如果模型误设定，倾向得分估计有偏
3. **丢弃未匹配样本**: 匹配率可能很低，损失样本量；估计的是 ATT 不是 ATE
4. **共同支撑假设**: 如果某些处理组个体的倾向得分在控制组中找不到对应值，无法匹配
5. **标准误计算复杂**: 需要考虑倾向得分估计的不确定性

### Q3: 如何诊断 PSM 的匹配质量？

**答案要点**:

**1. 标准化均值差 (SMD)**
- 公式: $SMD = \frac{\bar{X}_{treated} - \bar{X}_{control}}{\sqrt{(s^2_{treated} + s^2_{control})/2}}$
- 阈值: |SMD| < 0.1 表示良好平衡
- 需要比较匹配前后的 SMD

**2. 共同支撑检查**
- 可视化倾向得分的分布图
- 检查处理组和控制组的倾向得分重叠区域
- 计算在重叠区域外的样本比例

**3. 方差比**
- 检查匹配后处理组和控制组各协变量的方差比
- 理想值应接近 1

### Q4: PSM 估计的是 ATE 还是 ATT？能估计 ATE 吗？

**答案要点**:

**PSM 默认估计 ATT**

**原因**:
- 我们是为处理组的每个个体找控制组的匹配
- 未被匹配的控制组样本被丢弃
- 最终样本代表的是"接受处理的那群人"

**估计 ATE 的方法**:
1. **双向匹配**: 为处理组找控制组匹配 → 估计 ATT；为控制组找处理组匹配 → 估计 ATC；加权平均: $ATE = P(T=1) \cdot ATT + P(T=0) \cdot ATC$
2. **使用 IPW**: IPW 天然估计 ATE
3. **使用 AIPW**: 结合两种方法的优势

### Q5: IPW 的核心思想是什么？为什么重加权可以去除混淆偏差？

**答案要点**:

IPW 的核心思想是通过**重新加权**，创造一个"伪总体"，在这个伪总体中处理是随机分配的。

**直观解释**:
- 在观测数据中，某些类型的人更可能接受处理
- 这导致处理组和控制组的人群不可比
- IPW 给每个人赋予权重：$w_i = \frac{T_i}{e(X_i)} + \frac{1-T_i}{1-e(X_i)}$
- 这个权重让"不太可能接受处理但接受了"的人贡献更大
- 加权后的数据就像随机实验一样，X 与 T 独立了

**数学本质**: IPW 是 Horvitz-Thompson 估计量，通过逆概率加权来纠正选择偏差

### Q6: IPW 的局限性是什么？如何解决？

**答案要点**:

**主要局限性**:
1. **极端权重问题**: 当 $e(X) \approx 0$ 或 $e(X) \approx 1$ 时，权重会非常大，导致估计不稳定
2. **依赖倾向得分模型**: 如果倾向得分模型误设定，IPW 估计有偏
3. **效率损失**: 相比结果模型方法，方差可能更大

**解决方法**:
1. **权重裁剪**: `weights_clipped = np.clip(weights, None, np.percentile(weights, 99))`
2. **稳定权重**: $w_i^{stab} = \frac{P(T=T_i)}{P(T=T_i|X_i)}$，均值接近 1，方差更小
3. **修剪倾向得分**: 丢弃倾向得分过于极端的样本（如 <0.1 或 >0.9）
4. **使用双重稳健方法 (AIPW)**: 结合 IPW 和结果模型，更稳健

### Q7: 什么是有效样本量(ESS)？它的意义是什么？

**答案要点**:

**定义**: $ESS = \frac{(\sum w_i)^2}{\sum w_i^2}$

**意义**:
- ESS 衡量"有多少样本在真正起作用"
- 当所有权重相等时，ESS = n（所有样本都有效）
- 当权重差异很大时，ESS << n（少数样本主导）

**例子**: 如果 n=1000，但 ESS=100，说明实际上只有 100 个样本的信息量

**经验法则**:
- ESS / n > 0.5: 良好
- ESS / n < 0.3: 警告，可能需要修剪极端权重

### Q8: 什么是双重稳健性？为什么 AIPW 具有这个性质？

**答案要点**:

双重稳健性(Double Robustness)是指：**只要倾向得分模型或结果模型之一正确，估计量就是一致的**

**AIPW 的公式**:
$$\hat{\tau} = \frac{1}{n}\sum_i \left[(\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}\right]$$

**为什么具有双重稳健性**:
1. **如果倾向得分正确**: IPW 修正项会完美抵消结果模型的误差
2. **如果结果模型正确**: 残差 $Y - \hat{\mu}(X)$ 的期望为 0，IPW 修正项不引入偏差

**直观理解**: AIPW 就像买了两份保险，任何一份有效就能得到正确答案

### Q9: AIPW 估计器的三项分别代表什么含义？

**答案要点**:

| 项 | 公式 | 统计含义 | 因果含义 |
|---|------|----------|----------|
| 第一项 | $\hat{\mu}_1(X) - \hat{\mu}_0(X)$ | 两个回归模型预测的差异 | 基于协变量预测的个体效应 |
| 第二项 | $\frac{T(Y - \hat{\mu}_1(X))}{e(X)}$ | 处理组的加权残差 | 修正处理组预测误差 |
| 第三项 | $-\frac{(1-T)(Y - \hat{\mu}_0(X))}{1-e(X)}$ | 控制组的加权残差 | 修正控制组预测误差 |

**工作原理**:
1. 第一项给出"初步估计"（基于结果模型）
2. 第二、三项对"观测到的样本"的预测误差进行加权修正
3. 权重 $\frac{1}{e(X)}$ 确保修正是无偏的

### Q10: DML 和普通 AIPW/DR 有什么区别？为什么需要 Cross-fitting？

**答案要点**:

**核心区别**: Cross-fitting

**问题来源**: 当我们用机器学习模型时，如果用同一份数据训练模型和预测，残差 $Y_i - \hat{g}(X_i)$ 会被低估（overfitting bias），导致标准误失效、置信区间过窄。

**Cross-fitting 的解决方案**:
1. 把数据分成 K 折
2. 对每一折，用其他 K-1 折训练模型
3. 用训练好的模型预测当前折

这样每个 $Y_i$ 的预测都来自「没见过它」的模型，消除过拟合偏差。

### Q11: 什么是 Neyman 正交性？为什么它重要？

**答案要点**:

**定义**: Neyman 正交性是指矩函数对 nuisance 参数的导数为 0:
$$\frac{\partial}{\partial \eta} E[\psi(W; \tau_0, \eta)] \Big|_{\eta=\eta_0} = 0$$

**重要性**:
- 在经典方法中，nuisance 参数估计误差对目标参数估计的影响是**一阶的** $O(||\hat{\eta} - \eta||)$
- 在 Neyman 正交的方法中，影响是**二阶的** $O(||\hat{\eta} - \eta||^2)$

**实际意义**:
- 允许使用正则化 ML 模型（Lasso, Ridge, RF）
- 即使模型有偏差，对因果效应估计影响很小
- 可以达到 $\sqrt{n}$-一致性和渐近正态性

### Q12: E-value 是什么？如何解读？

**答案要点**:

**定义**: E-value 是使观测关联完全被混淆解释所需的最小风险比
$$E = RR + \sqrt{RR \times (RR - 1)}$$

**解读**: E-value = 3.0 意味着需要一个未观测因子 U:
- U 使「接受处理的概率」提高 3 倍
- U 同时使「结果发生概率」提高 3 倍
- 才能完全解释掉观测到的效应

**稳健性评价**:
- E < 1.5: 结论非常脆弱
- 1.5 ≤ E < 2.5: 结论中等稳健
- 2.5 ≤ E < 4.0: 结论较为稳健
- E ≥ 4.0: 结论非常稳健

**重要**: E-value 高不代表无混淆，只表示需要多强的混淆才能推翻结论

---

## 📊 方法选择指南

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **PSM** | 直观、可检查平衡 | 丢弃样本、估计 ATT | 小数据、需要可解释性 |
| **IPW** | 使用全部样本、估计 ATE | 极端权重问题 | 中等混淆、倾向得分模型可靠 |
| **AIPW** | 双重稳健、效率高 | 计算复杂、无 Cross-fitting 有偏 | **中等维度推荐** |
| **DML** | 高维可用、有效推断 | 计算量大、需要大样本 | **高维数据推荐** |

---

## 🎓 理论要点

### 无混淆假设 (Unconfoundedness)

$$(Y(0), Y(1)) \perp T | X$$

给定协变量 X，潜在结果与处理分配独立

### 正值假设 (Positivity)

$$0 < P(T=1|X) < 1$$

每个协变量值下都有一定概率接受或不接受处理

### 共同支撑 (Common Support)

处理组和控制组的倾向得分分布有重叠区域，才能进行比较

### SUTVA (Stable Unit Treatment Value Assumption)

1. **一致性**: $Y = T \cdot Y(1) + (1-T) \cdot Y(0)$
2. **无干扰**: 一个个体的潜在结果不受其他个体处理状态的影响

---

## 💡 实践建议

1. **先可视化**: 匹配/加权前后的协变量分布对比
2. **多种方法**: 尝试 PSM、IPW、AIPW，比较稳健性
3. **敏感性分析**: 使用 E-value 或 Rosenbaum bounds 评估稳健性
4. **保留诊断**: 报告匹配前后的平衡性统计（SMD、方差比）
5. **透明报告**: 说明样本损失、匹配参数选择、模型设定

---

## 📚 延伸阅读

- Rosenbaum & Rubin (1983): "The Central Role of the Propensity Score"
- Stuart (2010): "Matching Methods for Causal Inference: A Review"
- Chernozhukov et al. (2018): "Double/Debiased Machine Learning"
- VanderWeele & Ding (2017): "Sensitivity Analysis in Observational Research"

---

**「因果推断不是魔法，而是在假设下的严谨推理。」**
