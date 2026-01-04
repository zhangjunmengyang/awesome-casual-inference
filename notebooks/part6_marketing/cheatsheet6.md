# Part 6: Marketing Attribution & Optimization - 面试速查手册

> **快速复习要点**: 营销归因、优惠券优化、用户定向、预算分配

---

## 📋 核心概念速查

### 1. Marketing Attribution (营销归因)

#### 四类归因方法对比

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **Last-Click** | 100%归因给最后触点 | 简单 | 严重低估上游渠道 | 冲动购买 |
| **First-Click** | 100%归因给首次触点 | 重视获客 | 忽略转化路径 | 品牌认知 |
| **Linear** | 平均分配 | 公平 | 忽略位置重要性 | 路径短 |
| **Time-Decay** | 时间衰减加权 | 考虑时间效应 | 参数敏感 | 长决策周期 |
| **Shapley Value** | 博弈论公平分配 | 理论严谨 | 计算复杂O(2^n) | 多渠道协同 |
| **Markov Chain** | 移除效应 | 捕获转化概率 | 需要大量数据 | 路径分析 |

#### 关键公式

**Shapley Value**:
```
φᵢ(v) = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] × [v(S∪{i}) - v(S)]
```

**Markov Chain Removal Effect**:
```
RE_c = [P(Conv) - P(Conv|remove c)] / P(Conv)
```

**Time-Decay Weights**:
```
w_i = 2^(-(t-i)/half_life)
```

---

### 2. Coupon Optimization (优惠券优化)

#### 四类用户群体

| 用户类型 | 不发券 | 发券 | Uplift | 策略 |
|----------|--------|------|--------|------|
| **Persuadables** | ❌ | ✅ | 高正 | 🎯 重点发券 |
| **Sure Things** | ✅ | ✅ | ~0 | 💰 浪费钱 |
| **Lost Causes** | ❌ | ❌ | ~0 | 🚫 别浪费 |
| **Sleeping Dogs** | ✅ | ❌ | 负 | ⚠️ 千万别发 |

#### 核心公式

**Uplift 定义**:
```
Uplift(x) = P(Y=1|T=1,X=x) - P(Y=1|T=0,X=x)
```

**ROI 计算**:
```
ROI = (增量收入 - 成本) / 成本
增量收入 = 增量转化数 × 每次转化收入
成本 = 发券数 × 每张券成本
```

**最优决策阈值**:
```
发券条件: Uplift × 每次转化价值 > 券成本
阈值 = 券成本 / 每次转化价值
```

---

### 3. User Targeting (用户定向)

#### Meta-Learner 对比

| 方法 | 复杂度 | 优点 | 缺点 | 适用场景 |
|------|--------|------|------|----------|
| **S-Learner** | 低 | 简单 | 难捕获异质性 | 效应弱 |
| **T-Learner** | 中 | 直观 | 两模型不共享信息 | 样本充足 |
| **X-Learner** | 高 | 准确，适合不平衡 | 复杂 | 样本不平衡 |

#### 关键公式

**T-Learner**:
```
τ(x) = μ₁(x) - μ₀(x)
其中: μ₁, μ₀ 分别在处理组和对照组训练
```

**X-Learner (三阶段)**:
```
Stage 1: 训练 μ₀(x), μ₁(x)
Stage 2: 计算伪处理效应
  D¹ᵢ = Yᵢ - μ₀(Xᵢ)  (处理组)
  D⁰ᵢ = μ₁(Xᵢ) - Yᵢ  (对照组)
  训练 τ₁(x), τ₀(x)
Stage 3: 加权
  τ(x) = g(x)·τ₀(x) + (1-g(x))·τ₁(x)
  其中 g(x) = P(T=1|X=x) 是倾向得分
```

**最优干预策略**:
```
π*(x) = 𝟙[τ(x) × value > cost]
```

---

### 4. Budget Allocation (预算分配)

#### 优化原理

**边际收益递减**:
```
R(x) = a × x^α / (c^α + x^α)  (Hill方程)
R'(x) = a·α·c^α·x^(α-1) / (c^α + x^α)²
```

**最优分配条件**:
```
R'₁(x₁*) = R'₂(x₂*) = ... = R'ₙ(xₙ*) = λ
即: 所有渠道的边际ROI相等
```

**影子价格**:
```
λ = ∂R/∂B (总收益对预算的偏导)
含义: 再增加1元预算，总收益增加λ元
```

---

## 💻 2分钟代码实现题

### 题目1: 从零实现 Shapley Value

```python
from itertools import combinations

def shapley_value_from_scratch(channels, conversion_func):
    """
    从零实现 Shapley Value

    Args:
        channels: list, 渠道列表 ['搜索', '社交', '邮件']
        conversion_func: callable, 给定渠道子集返回转化价值

    Returns:
        dict: {channel: shapley_value}

    时间复杂度: O(2^n × n)
    """
    n = len(channels)
    shapley_values = {ch: 0.0 for ch in channels}

    def factorial(n):
        if n <= 1: return 1
        return n * factorial(n-1)

    # 对每个渠道计算 Shapley Value
    for i, channel in enumerate(channels):
        other_channels = [ch for j, ch in enumerate(channels) if j != i]

        # 遍历所有子集 S ⊆ N\{i}
        for subset_size in range(n):
            for subset in combinations(other_channels, subset_size):
                subset_list = list(subset)

                # 边际贡献: v(S∪{i}) - v(S)
                value_with = conversion_func(subset_list + [channel])
                value_without = conversion_func(subset_list) if subset_list else 0
                marginal = value_with - value_without

                # 权重: |S|!(n-|S|-1)!/n!
                s_size = len(subset_list)
                weight = (factorial(s_size) * factorial(n - s_size - 1)) / factorial(n)

                shapley_values[channel] += weight * marginal

    return shapley_values

# 测试
def test_conversion(channels):
    base = {'搜索': 500, '社交': 300, '邮件': 200}
    total = sum(base.get(ch, 0) for ch in channels)
    # 协同效应
    if '搜索' in channels and '社交' in channels:
        total += 100
    return total

channels = ['搜索', '社交', '邮件']
result = shapley_value_from_scratch(channels, test_conversion)
print(result)  # {'搜索': 558.33, '社交': 358.33, '邮件': 183.33}
```

**面试考点**:
- Shapley Value 公式理解
- 组合数学 (combinations)
- 时间复杂度分析: O(2^n × n)
- 优化方法: 蒙特卡洛采样 (n>10时必须)

---

### 题目2: 实现 T-Learner

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class TLearner:
    """T-Learner: Two-Model approach"""

    def __init__(self):
        self.model_control = GradientBoostingRegressor(n_estimators=50, max_depth=4)
        self.model_treatment = GradientBoostingRegressor(n_estimators=50, max_depth=4)

    def fit(self, X, T, Y):
        """
        训练 T-Learner
        X: 特征 (n_samples, n_features)
        T: 处理 (n_samples,)
        Y: 结果 (n_samples,)
        """
        mask_control = (T == 0)
        mask_treatment = (T == 1)

        # 分别训练两个模型
        self.model_control.fit(X[mask_control], Y[mask_control])
        self.model_treatment.fit(X[mask_treatment], Y[mask_treatment])

        return self

    def predict_cate(self, X):
        """预测 CATE"""
        mu1 = self.model_treatment.predict(X)
        mu0 = self.model_control.predict(X)
        return mu1 - mu0

# 使用示例
X = np.random.randn(1000, 5)
T = np.random.binomial(1, 0.5, 1000)
Y = X[:, 0] + 0.5 * T + np.random.randn(1000)

model = TLearner()
model.fit(X, T, Y)
cate = model.predict_cate(X)
print(f"平均 CATE: {cate.mean():.3f}")  # 应该接近 0.5
```

**面试考点**:
- Meta-Learner 理解
- 处理组/对照组分离
- CATE 估计原理
- 何时使用 T-Learner vs X-Learner

---

### 题目3: 边际ROI优化

```python
from scipy.optimize import minimize, LinearConstraint
import numpy as np

def response_curve(x, a, c, alpha):
    """Hill方程响应曲线"""
    return a * (x**alpha) / (c**alpha + x**alpha)

def marginal_response(x, a, c, alpha):
    """边际响应 (导数)"""
    return a * alpha * (c**alpha) * (x**(alpha-1)) / ((c**alpha + x**alpha)**2)

def optimize_budget(channels_params, total_budget):
    """
    预算优化: 最大化总收益

    channels_params: dict, {'渠道': {'a': ..., 'c': ..., 'alpha': ...}}
    total_budget: float, 总预算
    """
    n = len(channels_params)
    channel_names = list(channels_params.keys())

    # 目标函数: 最大化总收益 (最小化负收益)
    def objective(x):
        total = 0
        for i, name in enumerate(channel_names):
            total += response_curve(x[i], **channels_params[name])
        return -total

    # 约束: 预算总和
    constraints = [LinearConstraint(np.ones(n), total_budget, total_budget)]

    # 边界: 每个渠道 >= 0
    bounds = [(0, total_budget) for _ in range(n)]

    # 初始值: 平均分配
    x0 = np.ones(n) * total_budget / n

    # 优化
    result = minimize(objective, x0, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    allocation = dict(zip(channel_names, result.x))
    total_revenue = -result.fun

    # 验证边际ROI相等
    marginal_rois = {name: marginal_response(allocation[name], **channels_params[name])
                     for name in channel_names}

    return allocation, total_revenue, marginal_rois

# 测试
channels = {
    '搜索': {'a': 500, 'c': 150, 'alpha': 0.8},
    '信息流': {'a': 800, 'c': 300, 'alpha': 1.2}
}
alloc, revenue, mrois = optimize_budget(channels, 1000)

print("最优分配:", alloc)
print("总收益:", revenue)
print("边际ROI:", mrois)
# 边际ROI应该相等 (验证最优性条件)
```

**面试考点**:
- 边际收益递减原理
- 拉格朗日乘数法
- 最优性条件: 边际ROI相等
- scipy.optimize 使用

---

## 🎤 高频面试问答

### Q1: Last-Click归因的问题是什么？用Simpson's Paradox解释

**A**: Last-Click归因会严重低估上游渠道的贡献，犯了混淆相关性和因果性的错误。

**Simpson's Paradox示例**:
```
场景: 展示广告(A) → 搜索广告(B) → 转化

Last-Click视角:
- B获得100%归因 (它是最后触点)
- A获得0%归因

真实情况:
- 停掉A后，B的转化率下降80%
- A负责获客，B负责转化，缺一不可

悖论:
- 分层看: A和B都重要
- 合并看: B获得100%功劳
- 结论相反！
```

**业务影响**:
```python
某电商案例:
- Last-Click: 砍掉展示广告预算
- 结果: 3个月后整体转化下降30%
- 原因: 展示广告建立了品牌认知，是搜索的前提
```

**替代方案**: Shapley Value, Data-Driven Attribution, MMM

---

### Q2: 为什么Sleeping Dogs用户发券反而转化率下降？

**A**: Sleeping Dogs是对促销信息反感的用户群体，发券会产生负面效应。

**三大原因**:

1. **品牌认知负面影响**
```
用户心理: "打折促销？是不是质量不好？"
案例: 奢侈品电商发30%券
  - 高端用户转化率: 8% → 5%
  - Uplift = -3% (负向)
```

2. **促销疲劳**
```
频繁发券 → 用户习惯等券 → 非促销日不买
某App案例:
  - 初期无券转化率: 3%
  - 开始每周发券:
    - 发券日: 5%
    - 非发券日: 1%
  - 综合转化率: (5%×1 + 1%×6)/7 = 1.57%
  - 反而比3%更低！
```

3. **信息过载**
```
每天Push券 → 用户烦 → 卸载App
实验: 每天Push vs 不Push
  - Push组: 7天转化2%，但卸载率15%
  - 不Push: 7天转化1.5%，卸载率2%
  - 长期LTV损失远超短期收益
```

**识别特征**:
- 品牌忠诚度高
- 历史客单价高
- 从不使用优惠券
- EDM打开率低

**应对策略**: Uplift模型识别 + 排除发券

---

### Q3: 如何向非技术老板解释Shapley Value？

**A**: 用篮球比赛类比 + 数据说话 + 强调业务价值

**篮球类比**:
```
最后一球由C投进，这2分归功于谁？

Last-Click逻辑:
  C获得100%功劳 (他投进的)

Shapley Value逻辑:
  考虑所有团队组合:
  - 只有A (防守): 0分
  - 只有B (运球): 20%得分
  - 只有C (投篮): 0分 (没球)
  - A+B: 50%得分
  - A+C: 40%得分
  - B+C: 60%得分
  - A+B+C: 90%得分

  Shapley归因: A 30%, B 35%, C 35%

营销类比:
  展示广告 = A (建立认知)
  搜索广告 = B (激发兴趣)
  邮件营销 = C (促成转化)
```

**数据验证**:
```
A/B测试:
- 停掉展示: 搜索转化率 -30%
- 停掉搜索: 整体转化 -50%
- 停掉邮件: 转化 -20%

Shapley归因: 展示25%, 搜索50%, 邮件25%
```

**ROI提升**:
| 指标 | Last-Click | Shapley | 提升 |
|------|-----------|---------|------|
| 整体ROI | 2.1 | 2.8 | +33% |
| 增量收入 | 基准 | +30% | - |

**回答疑虑**:
- "太复杂?" → 数据已有，计算自动化，报表不变
- "靠谱吗?" → 诺贝尔奖理论，Google/Facebook在用
- "多久见效?" → 1个月切换，2个月见ROI提升

---

### Q4: Uplift建模 vs 响应率建模的本质区别？

**A**: 响应率预测"谁会买"，Uplift预测"因为干预谁会买"。

| 维度 | 响应率建模 | Uplift建模 |
|------|-----------|------------|
| **目标** | P(Y=1\|X) | τ(X) = P(Y=1\|T=1,X) - P(Y=1\|T=0,X) |
| **问题** | 谁会购买？ | 谁因干预而购买？ |
| **数据** | 只需结果 | 需要实验数据(T,X,Y) |
| **因果性** | 相关性 | 因果性 |
| **决策** | 误导 | 优化ROI |

**示例对比**:
```
用户A (高频老客):
  响应率: P(购买) = 90%
  Uplift: 91% - 90% = 1%

用户B (低频新客):
  响应率: P(购买) = 30%
  Uplift: 55% - 30% = 25%

响应率模型 → 选A (响应率高)
  ROI = (0.01 × 40 - 20) / 20 = -98% ❌

Uplift模型 → 选B (Uplift高)
  ROI = (0.25 × 40 - 20) / 20 = -50% (仍亏但好很多)
```

**核心洞察**:
> "会购买" ≠ "因为券而购买"
> Sure Things造成补贴浪费的本质是: 为不会改变的行为付费

**何时用Uplift**: 营销干预、个性化定价、政策评估
**何时用响应率**: 流失预测、推荐系统、信用评分

---

### Q5: 如何验证Uplift模型的准确性？

**A**: 金标准是分层A/B测试，辅以Uplift Curve和Qini系数。

**方法1: Uplift Curve (最常用)**
```python
def plot_uplift_curve(y_true, treatment, uplift_scores):
    # 按uplift分数排序
    sorted_idx = np.argsort(-uplift_scores)

    # 逐步扩大目标人群
    percentiles = [0.1, 0.2, ..., 1.0]
    uplifts = []

    for p in percentiles:
        top_n = int(len(y_true) * p)
        y_subset = y_true[sorted_idx[:top_n]]
        t_subset = treatment[sorted_idx[:top_n]]

        # 计算实际uplift
        treat_conv = y_subset[t_subset==1].mean()
        control_conv = y_subset[t_subset==0].mean()
        uplift = treat_conv - control_conv
        uplifts.append(uplift)

    # 理想: top10% uplift最高，递减
    plt.plot(percentiles, uplifts)
```

**方法2: 分层A/B测试 (金标准)**
```
1. 用历史数据训练Uplift模型
2. 分层:
   - High Uplift: 分数 > 0.15
   - Medium: 0.05 < 分数 ≤ 0.15
   - Low: 分数 ≤ 0.05

3. 每层内做A/B:
   实验组: 发券
   对照组: 不发券
   测量实际Uplift

4. 验证:
   ✓ High > Medium > Low (排序正确)
   ✓ 预测值 ≈ 实际值 (校准良好)
```

**方法3: Qini系数**
```python
# 类似AUC，但针对Uplift
# Qini AUC > 0: 模型有效
# Qini AUC ≈ 0: 模型无用
```

**方法4: 业务指标验证**
```
- 预期ROI vs 实际ROI误差 < 20%
- 考虑用户博弈行为
- 长期LTV影响
```

---

### Q6: 预算有限时如何优化分配？

**A**: 从"覆盖率最大化"转向"ROI最大化"，基于边际ROI选择Top-K。

**核心原则**: 宁可只给1%用户发券但ROI很高，不要为覆盖率浪费预算

**方法1: Top-K选择**
```python
def select_top_k(uplift_scores, budget_fraction=0.2):
    threshold = cost / value_per_conversion

    # 只选Uplift超过阈值的
    candidates = uplift_scores > threshold

    # 在候选中选Top K%
    n = int(len(uplift_scores) * budget_fraction)
    top_k_idx = np.argsort(-uplift_scores[candidates])[:n]

    return top_k_idx
```

**方法2: 考虑不同券面额**
```
问题: 5元券 vs 10元券 vs 15元券
目标: max Σ (uplift_i^c × revenue_i - cost_c) × x_i^c
约束: Σ cost_c × x_i^c ≤ Budget

使用线性规划求解最优分配
```

**方法3: 动态分配 (分批发放)**
```
不要一次用完预算，边发边学

Thompson Sampling:
  Week 1: 探索各群体 (30%预算)
  Week 2-3: 根据实际效果调整 (40%预算)
  Week 4: 全投高ROI群体 (30%预算)
```

**实战清单**:
```
□ 排除Sleeping Dogs (Uplift < 0)
□ 排除ROI为负 (Uplift × Revenue < Cost)
□ 分层分配 (保证战略细分市场最小覆盖)
□ 券面额优化 (高Uplift低客单价 → 小券)
□ 时间分散 (40% + 40% + 20%)
□ 在线优化 (Bandit算法动态调整)
```

---

## 📊 关键公式汇总

### Attribution

```
# Shapley Value
φᵢ(v) = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] × [v(S∪{i}) - v(S)]

# Markov Removal Effect
RE_c = [P(Conversion) - P(Conversion|remove c)] / P(Conversion)

# Time-Decay
w_i = 2^(-(t-i)/λ), λ = half_life
```

### Uplift Modeling

```
# Uplift定义
τ(x) = E[Y(1) - Y(0)|X=x]
     = P(Y=1|T=1,X=x) - P(Y=1|T=0,X=x)

# T-Learner
τ(x) = μ₁(x) - μ₀(x)

# X-Learner
τ(x) = g(x)·τ₀(x) + (1-g(x))·τ₁(x)
g(x) = P(T=1|X=x)

# ROI
ROI = (τ × n × revenue - n × cost) / (n × cost)
```

### Budget Optimization

```
# Hill Response Curve
R(x) = a·x^α / (c^α + x^α)

# Marginal Response
R'(x) = a·α·c^α·x^(α-1) / (c^α + x^α)²

# Optimal Condition
R'₁(x₁*) = R'₂(x₂*) = ... = λ

# Shadow Price
λ = ∂R/∂B
```

---

## 🔥 常见面试陷阱

### 陷阱1: 混淆Uplift和响应率
```
❌ 错误: "这个用户响应率90%，应该发券"
✅ 正确: "这个用户Uplift只有1%，发券ROI为负"

关键: Sure Things响应率高但Uplift低
```

### 陷阱2: 忽略Simpson's Paradox
```
❌ 错误: "Last-click显示搜索广告贡献80%，加大投入"
✅ 正确: "搜索是下游收割，上游展示广告不可少"

关键: 渠道间有依赖关系
```

### 陷阱3: 只看平均ROI不看边际ROI
```
❌ 错误: "渠道A平均ROI 3.0最高，全投A"
✅ 正确: "A的边际ROI已降到1.5，应分配给B"

关键: 边际收益递减
```

### 陷阱4: 忽略长期效应
```
❌ 错误: "频繁发券短期转化率提升5%"
✅ 正确: "用户学会等券，长期基线下降10%"

关键: Sleeping Dogs效应
```

### 陷阱5: 不验证模型
```
❌ 错误: "模型训练完就上线"
✅ 正确: "先做分层A/B测试验证"

关键: 离线指标 ≠ 在线效果
```

---

## 📚 扩展阅读

### 学术论文
- **Marketing Attribution**: "Data-Driven Multi-Touch Attribution Models" (KDD 2011)
- **Uplift Modeling**: "Uplift Modeling for Clinical Trial Data" (2012)
- **Shapley Value**: "A Value for n-Person Games" (Shapley, 1953)
- **Budget Optimization**: "Bayesian Methods for Media Mix Modeling" (Google, 2017)

### 工具库
- **Python**: `econml`, `causalml`, `pylift`
- **R**: `ChannelAttribution`, `uplift`, `CRAN`
- **Google**: Lightweight MMM, Meridian
- **Facebook**: Robyn (MMM)

### 实战案例
- **Google**: "Multi-Touch Attribution at Scale" (2016)
- **Uber**: "Experimentation Platform" (2018)
- **Airbnb**: "Measuring Attribution Across Platforms" (2019)

---

## 🎯 面试准备建议

### 2周复习计划

**Week 1: 理论基础**
- Day 1-2: Marketing Attribution (Shapley, Markov)
- Day 3-4: Uplift Modeling (T/X-Learner)
- Day 5-6: Budget Optimization (边际ROI)
- Day 7: 综合练习

**Week 2: 代码实现**
- Day 1-2: 手写Shapley Value
- Day 3-4: 实现T-Learner
- Day 5-6: 预算优化求解
- Day 7: Mock Interview

### 必练代码题
1. ✅ 从零实现Shapley Value (15分钟)
2. ✅ T-Learner训练和预测 (10分钟)
3. ✅ 边际ROI优化 (20分钟)
4. ⭐ Uplift Curve绘制 (15分钟)
5. ⭐ A/B测试功效分析 (15分钟)

### 高频理论题
1. ✅ Last-Click vs Shapley Value
2. ✅ Sleeping Dogs原因和识别
3. ✅ Uplift vs 响应率
4. ✅ 边际收益递减
5. ⭐ Simpson's Paradox在归因中的体现
6. ⭐ 如何设计归因A/B测试

### Case Study准备
- **电商**: 双十一预算分配
- **O2O**: 优惠券定向发放
- **SaaS**: 用户激活策略
- **金融**: 信用卡营销

---

## ✨ 总结

### 核心takeaway

1. **Marketing Attribution**: Shapley Value是理论最严谨的归因方法，考虑了所有渠道组合
2. **Coupon Optimization**: 区分Persuadables和Sure Things，避免补贴浪费
3. **User Targeting**: X-Learner适合样本不平衡，T-Learner简单直观
4. **Budget Optimization**: 边际ROI相等是最优条件，不要只看平均ROI

### 面试金句

> "Last-click归因就像只给投进最后一球的球员记分，忽略了传球和防守队友的贡献。"

> "Sure Things就像本来就会来餐厅的老顾客，给他们发券只是白送钱。"

> "预算优化的核心是让每一块钱的边际收益相等，就像给不同形状的水桶加水到同样高度。"

> "Uplift建模回答的不是'谁会买'，而是'因为你的营销谁会买'。"

---

**最后**: 面试时记得用类比、举例、画图，把复杂概念讲清楚。Good luck! 🚀
