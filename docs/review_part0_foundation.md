# Part 0 Foundation Notebook 深度审核报告

审核日期: 2026-01-04
审核人: Claude (Data Science Expert)
审核范围: Part 0 Foundation (基础理论) 的 4 个 notebook

---

## 整体评价

Part 0 Foundation 模块是整个因果推断教程的核心基础，目前已完成的 3 个 notebook 在教学设计、内容组织、互动性方面表现优秀，但在几个关键维度上仍有明显缺失，需要补充完善。

**总体评分: 7.5/10**

**优点:**
- 教学叙事流畅，从生活案例引入抽象概念
- 互动式设计（TODO 代码填空）有助于主动学习
- 可视化丰富，帮助理解抽象概念
- 循序渐进，难度曲线设计合理

**核心问题:**
- **缺少"从零实现"的完整示例代码**
- **缺少数学推导和证明**
- **缺少习题的参考答案**
- **缺少面试题模拟**
- **缺少与主流因果库的对比**

---

## 📋 逐个 Notebook 详细审核

---

## 1️⃣ part0_1_potential_outcomes.ipynb

### 整体评分: 7.5/10

### 优点

1. **教学设计优秀**
   - 从"优惠券案例"引入，贴近业务场景
   - 概念解释清晰：Y(0), Y(1), ITE, ATE, 反事实
   - 使用表格、图形辅助理解

2. **互动性强**
   - 5 个 TODO 代码填空，覆盖核心操作
   - 数据生成、ATE 计算、反事实分析都有练习

3. **可视化有效**
   - 潜在结果散点图
   - ITE 分布直方图
   - 随机 vs 混淆分配的对比

4. **延伸思考**
   - 通过"自愿报名"场景引入混淆偏差
   - 为后续 notebook 埋下伏笔

### 问题清单

#### [严重] 缺失完整代码示例 (多处)

**问题位置:**
- Cell 5: `generate_potential_outcomes` 函数
- Cell 10: `calculate_true_ate` 和 `calculate_naive_ate` 函数
- Cell 14: `counterfactual_analysis` 函数
- Cell 17: `compare_random_vs_confounded` 函数

**问题描述:**
所有函数都只有 TODO 标记，没有提供完整的参考实现。学习者填完空后，无法验证自己的实现是否正确或最优。

**建议修复:**
```python
# 在每个 TODO cell 后面，增加一个折叠的 "参考答案" cell
# 示例：

### ===== 参考答案（点击展开） =====
def generate_potential_outcomes_solution(n: int = 100, seed: int = 42):
    """
    参考实现

    关键点：
    1. 使用 np.random.randn() 生成标准正态分布
    2. 注意噪声项的正确添加
    3. ITE = Y1 - Y0（每个人都计算）
    """
    np.random.seed(seed)
    X = np.random.randn(n)
    noise = np.random.randn(n) * 5
    Y0 = 50 + 5 * X + noise
    Y1 = 60 + 5 * X + noise
    T = np.random.binomial(1, 0.5, n)
    Y = T * Y1 + (1 - T) * Y0
    ITE = Y1 - Y0

    return pd.DataFrame({
        'X': X, 'T': T, 'Y0': Y0, 'Y1': Y1, 'Y': Y, 'ITE': ITE
    })

# 对比学习者实现和参考实现
print("你的实现 ATE:", df['ITE'].mean())
print("参考实现 ATE:", generate_potential_outcomes_solution()['ITE'].mean())
```

#### [严重] 缺失数学推导 (Cell 3, 9)

**问题位置:**
- Cell 3: 观测结果公式 $Y_i = T_i \cdot Y_i(1) + (1 - T_i) \cdot Y_i(0)$
- Cell 9: ATE 定义 $\text{ATE} = E[Y(1) - Y(0)]$

**问题描述:**
只给出公式，没有解释为什么这么定义，也没有推导过程。对数学基础薄弱的学习者不友好。

**建议增加:**

```markdown
### 🧮 公式推导

**为什么观测结果是 $Y = T \cdot Y(1) + (1-T) \cdot Y(0)$？**

这是一个"开关函数"：
- 当 T=1 时（吃药）：$Y = 1 \cdot Y(1) + 0 \cdot Y(0) = Y(1)$ ✅
- 当 T=0 时（不吃药）：$Y = 0 \cdot Y(1) + 1 \cdot Y(0) = Y(0)$ ✅

**为什么 ATE = E[Y(1) - Y(0)]？**

证明：在随机实验下，朴素估计是无偏的
\begin{align}
E[Y|T=1] - E[Y|T=0]
&= E[Y(1)|T=1] - E[Y(0)|T=0] \\
&= E[Y(1)] - E[Y(0)] \quad \text{(随机化 → 独立)} \\
&= E[Y(1) - Y(0)] = \text{ATE}
\end{align}

关键：**随机分配保证了 $Y(1), Y(0) \perp T$**
```

#### [中等] 思考题无参考答案 (Cell 22-25)

**问题位置:** Cell 22, 23, 24, 25

**问题描述:**
4 个思考题都是开放问题，但没有提供参考答案或评分标准。

**建议修复:**
在最后增加一个"思考题参考答案"部分：

```markdown
### 思考题参考答案

#### 问题 1: 为什么我们无法同时观测到 Y(0) 和 Y(1)?

**参考答案:**
因为每个个体在某个时刻只能处于一种状态（吃药或不吃药）。一旦做出选择，另一种可能性就成为"反事实"——它是假设的、无法观测的。这被称为**因果推断的根本问题** (Fundamental Problem of Causal Inference, Holland 1986)。

我们只能观测到 Factual（实际发生的），而 Counterfactual（反事实）永远是缺失的。

#### 问题 2: 随机实验为什么能识别 ATE?

**参考答案:**
随机实验通过随机分配处理 T，打破了 T 与潜在结果 $(Y(0), Y(1))$ 之间的关联，使得：

$$Y(0), Y(1) \perp T \quad \text{(独立性)}$$

因此：
- 处理组的平均结果 $E[Y|T=1] = E[Y(1)]$
- 对照组的平均结果 $E[Y|T=0] = E[Y(0)]$
- 两组差异 $= E[Y(1)] - E[Y(0)] = ATE$

**关键:** 随机化使两组在所有特征（包括观测和未观测）上均衡。

#### 问题 3: 混淆分配场景中，朴素估计是偏高还是偏低？

**参考答案:**
偏高。原因：
- 基础好的学生（X 高）更愿意参加补习班（T=1）
- 基础好的学生本来成绩就更高（Y 高）
- 所以参加补习的学生成绩高，**部分是因为补习，部分是因为他们本来就强**
- 朴素估计 = 补习真实效果 + X 的影响 > 真实 ATE

公式表示：
$$E[Y|T=1] - E[Y|T=0] = \underbrace{\text{ATE}}_{\text{真实效应}} + \underbrace{\text{Selection Bias}}_{\text{> 0}}$$

#### 问题 4: ITE 在实践中为什么重要？

**参考答案:**

ITE（个体处理效应）的重要性：

1. **个性化决策** (Personalized Treatment)
   - 医疗: 精准医疗（同一种药对不同人效果不同）
   - 营销: 个性化推荐（同一个优惠券对不同用户价值不同）
   - 教育: 因材施教（同一教学方法对不同学生效果不同）

2. **资源优化**
   - 只给"高 ITE"的个体施加处理
   - 避免浪费在"低 ITE"或"负 ITE"的个体上

3. **异质性理解** (Heterogeneous Treatment Effects)
   - 发现"对谁最有效"
   - 例子: Netflix 的推荐系统需要知道"这个电影对这个用户的吸引力"

**实际案例:**
- DoorDash 优惠券: 不是所有用户都需要优惠券才下单，只给"边缘用户"发券
- 医疗试验: 识别"responder"和"non-responder"
```

#### [中等] 缺少与库的对比 (全文)

**问题描述:**
Notebook 完全是手写实现，没有与主流因果推断库（如 EconML, CausalML, DoWhy）的对比。

**建议增加:**

```python
### 🔧 使用因果推断库

# 使用 EconML 计算 ATE
from econml.metalearners import TLearner
from sklearn.linear_model import LinearRegression

# T-Learner: 分别对处理组和对照组建模
tlearner = TLearner(models=LinearRegression())
tlearner.fit(Y=df['Y'].values, T=df['T'].values, X=df[['X']].values)

# 预测 ITE
ite_econml = tlearner.effect(df[['X']].values)

print(f"手写 ATE: {df['ITE'].mean():.4f}")
print(f"EconML ATE: {ite_econml.mean():.4f}")

# 对比
comparison = pd.DataFrame({
    '手写 ITE': df['ITE'],
    'EconML ITE': ite_econml
})
print(comparison.head())
```

#### [建议] 增加面试题模拟

**建议新增 Cell:**

```markdown
## 💼 面试题模拟

### 问题 1: 快速问答

**面试官:** "什么是 ATE？它和 ITE 有什么区别？"

**参考答案:**
- **ITE (Individual Treatment Effect):** 个体处理效应，表示处理对**单个个体**的因果效应，$\text{ITE}_i = Y_i(1) - Y_i(0)$
- **ATE (Average Treatment Effect):** 平均处理效应，是所有个体 ITE 的期望，$\text{ATE} = E[\text{ITE}] = E[Y(1) - Y(0)]$

**区别:**
- ITE 是个体层面，ATE 是总体层面
- ITE 无法观测（反事实问题），ATE 可以在随机实验下识别
- ITE 用于个性化决策，ATE 用于政策评估

---

### 问题 2: 编码题

**面试官:** "给你一个观测数据集 (X, T, Y)，请估计 ATE。假设 T 是随机分配的。"

```python
def estimate_ate(df):
    """
    估计 ATE

    Args:
        df: DataFrame with columns ['X', 'T', 'Y']

    Returns:
        float: ATE estimate
    """
    # 你的实现
    pass

# 测试数据
test_df = pd.DataFrame({
    'X': np.random.randn(100),
    'T': np.random.binomial(1, 0.5, 100),
    'Y': np.random.randn(100)
})

print(f"ATE estimate: {estimate_ate(test_df):.4f}")
```

**参考答案:**
```python
def estimate_ate(df):
    """最简单的 ATE 估计: 简单差分"""
    return df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()
```

**进阶答案:**
```python
def estimate_ate_robust(df):
    """更稳健的 ATE 估计: 线性回归"""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df[['T', 'X']], df['Y'])
    return model.coef_[0]  # T 的系数就是 ATE
```

---

### 问题 3: 场景分析

**面试官:** "假设你在电商公司，老板问你'上周的促销活动让 GMV 增长了多少？'，你会怎么回答？需要注意什么？"

**参考答案:**

**回答框架:**

1. **澄清问题:**
   - "GMV 增长"指的是与什么对比？（去年同期？上上周？）
   - 促销是如何分配的？（全量？A/B test？）
   - 是否有其他同时发生的事件？（节假日、竞品活动）

2. **分析因果:**
   - 如果是 A/B test（随机分配），可以直接计算 ATE
   - 如果不是随机分配，需要警惕混淆偏差
     - 例如：只给高价值用户发促销 → 高估效果
     - 例如：节假日效应 → 高估效果

3. **量化估计:**
   - 最好情况: $\text{ATE} = E[Y|T=1] - E[Y|T=0]$（随机分配）
   - 次优情况: 使用回归控制已知混淆变量
   - 敏感性分析: 评估未观测混淆的影响

4. **诚实汇报:**
   - 说明估计的可信度（置信区间）
   - 说明潜在的混淆因素
   - 避免过度因果化（Correlation ≠ Causation）

**关键点:** 数据分析师不仅要会算数字，更要懂因果推断，避免给出误导性的结论。
```

---

### 缺失内容清单

- [ ] 从零实现的完整参考代码（5 处函数）
- [ ] 库实现对比（EconML, CausalML）
- [ ] 数学推导和证明（观测公式、ATE 无偏性）
- [ ] 面试题模拟（3-5 道）
- [ ] 思考题参考答案（4 道）

### 具体改进建议

#### 优先级 1（高）: 补充参考实现

在每个 TODO cell 后面，增加可折叠的"参考答案" cell：
```python
# %% [markdown]
# ### ===== 💡 参考答案 =====
#
# <details>
# <summary>点击展开查看参考实现</summary>
#
# ```python
# def xxx_solution(...):
#     """参考实现，关键点：..."""
#     ...
# ```
# </details>
```

#### 优先级 2（高）: 增加数学推导

在 Cell 3 和 Cell 9 后面，增加"公式推导"部分，包含：
1. 为什么 $Y = T \cdot Y(1) + (1-T) \cdot Y(0)$
2. 证明随机实验下 $E[Y|T=1] - E[Y|T=0] = ATE$
3. 说明 $Y(0), Y(1) \perp T$ 的含义

#### 优先级 3（中）: 增加库对比

在最后增加一节"使用因果推断库"，对比：
- 手写实现
- EconML TLearner
- CausalML XLearner

#### 优先级 4（中）: 增加面试题

在思考题部分前，增加"面试题模拟"：
- 快速问答（2-3 题）
- 编码题（1-2 题）
- 场景分析题（1 题）

#### 优先级 5（低）: 思考题答案

在最后增加"思考题参考答案"部分。

---

## 2️⃣ part0_2_causal_dag.ipynb

### 整体评分: 7.5/10

### 优点

1. **可视化效果好**
   - 使用 NetworkX 画 DAG，直观
   - 三种结构（Fork, Chain, Collider）并排对比
   - 好莱坞悖论的散点图很有说服力

2. **核心概念覆盖全面**
   - 三种基本结构
   - 路径与后门路径
   - 后门准则
   - d-分离（隐含）
   - 碰撞偏差

3. **案例生动**
   - 好莱坞悖论（明星的演技和颜值）
   - 运动-收入-健康的关系
   - 成为明星的条件（演技+颜值>12）

4. **代码练习合理**
   - 识别因果结构
   - 找后门路径
   - 验证调整集
   - 模拟碰撞偏差

### 问题清单

#### [严重] 缺失 d-分离的形式化定义 (Cell 10, 14)

**问题位置:** Cell 10 (后门准则), Cell 14 (阻断路径)

**问题描述:**
虽然提到了"阻断路径"，但没有明确解释 d-分离（d-separation）的概念和判断规则。

**建议增加:**

```markdown
### 🔍 d-分离 (d-separation)

**定义:** 给定调整集 Z，路径 P 从 X 到 Y 被 d-分离（阻断），当且仅当：

1. **Chain 结构被阻断:** X → Z → Y（Z ∈ 调整集）
2. **Fork 结构被阻断:** X ← Z → Y（Z ∈ 调整集）
3. **Collider 结构被阻断:** X → Z ← Y（Z ∉ 调整集 且 Z 的后代 ∉ 调整集）

**关键规则:**
- 控制 Chain/Fork 中间节点 → 阻断路径 ✅
- 控制 Collider 节点 → 打开路径 ❌

**d-分离判断算法:**
```python
def is_d_separated(dag, X, Y, Z):
    """
    判断 X 和 Y 在给定 Z 的条件下是否 d-分离

    Args:
        dag: DAG 对象
        X: 起始节点
        Y: 终止节点
        Z: 调整集

    Returns:
        True if d-separated (conditionally independent)
    """
    # 实现 Bayes Ball 算法或直接使用 pgmpy
    from pgmpy.independencies import IndependenceAssertion
    # ...
```

#### [严重] 缺失最小调整集的算法 (Cell 15, 16)

**问题位置:** Cell 15 (`is_valid_adjustment_set`)

**问题描述:**
只验证给定调整集是否有效，没有教学生"如何找到最小调整集"。

**建议增加:**

```python
def find_minimal_adjustment_sets(edges, treatment, outcome):
    """
    找出所有最小调整集（满足后门准则）

    算法:
    1. 找出所有从 T 到 Y 的后门路径
    2. 对每条后门路径，找出能阻断它的节点
    3. 使用集合覆盖算法找最小集合

    Returns:
        List[Set[str]]: 所有最小调整集
    """
    from itertools import combinations

    backdoor_paths = identify_backdoor_paths(edges, treatment, outcome)

    if not backdoor_paths:
        return [set()]  # 无后门路径，空调整集即可

    # 候选节点：所有后门路径上的节点（除了 T 和 Y）
    candidates = set()
    for path in backdoor_paths:
        candidates.update(set(path) - {treatment, outcome})

    # 从小到大枚举调整集
    for size in range(1, len(candidates) + 1):
        for adj_set in combinations(candidates, size):
            adj_set = set(adj_set)
            if is_valid_adjustment_set(edges, treatment, outcome, adj_set):
                yield adj_set

# 示例
edges = [("X", "T"), ("X", "Y"), ("T", "Y")]
minimal_sets = list(find_minimal_adjustment_sets(edges, "T", "Y"))
print(f"最小调整集: {minimal_sets}")  # 输出: [{'X'}]
```

#### [严重] 碰撞偏差案例的数学解释不足 (Cell 21-24)

**问题位置:** Cell 21-24 (好莱坞悖论)

**问题描述:**
好莱坞悖论的模拟很直观，但缺少数学推导：为什么控制 Collider 会产生负相关？

**建议增加:**

```markdown
### 🧮 碰撞偏差的数学原理

**场景:** T → C ← Y，T 和 Y 独立

**关键定理:** 虽然 $T \perp Y$（边际独立），但 $T \not\perp Y | C$（条件相关）

**证明（简化版）:**

假设：
- $T \sim \text{Uniform}(0, 10)$（演技）
- $Y \sim \text{Uniform}(0, 10)$（颜值）
- $C = 1$ 当且仅当 $T + Y > 12$（成为明星）

在明星群体中（$C=1$）：
- 如果某人演技低（$T=5$），那么必须颜值高（$Y>7$）才能成为明星
- 如果某人演技高（$T=9$），那么颜值可以较低（$Y>3$）

这创造了负相关！

**条件概率视角:**
$$P(Y | T, C=1) = \frac{P(C=1 | T, Y) \cdot P(Y)}{P(C=1 | T)}$$

当 $T$ 增大时，$P(C=1 | T)$ 增大，所以对 $Y$ 的要求降低。

**可视化:**
```python
# 在 T-Y 平面上，C=1 的区域是一个三角形（T+Y>12）
# 在这个三角形内，T 和 Y 呈现负相关
```

这就是 **Berkson's Paradox**（贝克森悖论）的本质。
```

#### [中等] DFS 路径查找代码未完成 (Cell 11)

**问题位置:** Cell 11 (`find_all_paths` 中的 DFS)

**问题描述:**
DFS 函数只有注释，没有完整实现，学习者可能不知道如何填空。

**建议修复:**
提供参考实现：

```python
def find_all_paths(edges, start, end, ignore_direction=True):
    """找出从 start 到 end 的所有路径"""
    # 构建邻接表
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        adj[u].append(v)

        if ignore_direction:
            if v not in adj:
                adj[v] = []
            adj[v].append(u)

    paths = []

    def dfs(current, path, visited):
        if current == end:
            paths.append(path.copy())
            return

        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    dfs(start, [start], {start})
    return paths

# 测试
edges = [("X", "T"), ("X", "Y"), ("T", "Y")]
paths = find_all_paths(edges, "T", "Y", ignore_direction=True)
print(f"所有路径: {paths}")
# 输出: [['T', 'Y'], ['T', 'X', 'Y']]
```

#### [中等] 缺少工具变量、前门准则的介绍 (全文)

**问题描述:**
只介绍了后门准则，但未提及其他识别策略（工具变量 IV、前门准则 Front-Door）。

**建议增加:**

```markdown
### 🔧 其他识别策略

#### 1. 前门准则 (Front-Door Criterion)

**场景:** 存在未观测混淆 U，无法使用后门准则

```
    U (未观测)
   ↙ ↘
  T   Y
   ↘ ↗
    M
```

**前门准则:** 如果存在中介变量 M 满足：
1. T → M → Y（M 完全中介 T 对 Y 的效应）
2. T 阻断 M 的所有后门路径
3. M 对 Y 的后门路径都经过 T

则可以通过 M 识别 T 对 Y 的效应。

**公式:**
$$P(Y | \text{do}(T=t)) = \sum_m P(M=m | T=t) \sum_{t'} P(Y | M=m, T=t') P(T=t')$$

#### 2. 工具变量 (Instrumental Variable)

**场景:** 存在未观测混淆 U

```
Z → T → Y
     ↑
     U (未观测)
```

**IV 条件:**
1. **相关性:** $Z \to T$（Z 影响 T）
2. **排他性:** Z 只通过 T 影响 Y
3. **独立性:** $Z \perp U$（Z 与混淆无关）

**识别公式:**
$$\text{ATE} = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, T)}$$

**经典案例:**
- Z = 是否住在学校附近（工具变量）
- T = 是否上大学
- Y = 收入
```

#### [建议] 增加复杂 DAG 的实战案例

**建议新增:**

```python
### 🎯 实战案例：分析复杂 DAG

# 场景：评估在线广告对购买的影响
# DAG:
#   Age → Ad_Exposure → Purchase
#   Age → Income → Purchase
#   Income → Ad_Exposure

edges = [
    ("Age", "Ad_Exposure"),
    ("Age", "Income"),
    ("Income", "Purchase"),
    ("Income", "Ad_Exposure"),
    ("Ad_Exposure", "Purchase")
]

# 问题 1: 找出所有从 Ad_Exposure 到 Purchase 的路径
# 问题 2: 识别后门路径
# 问题 3: 找出最小调整集
# 问题 4: 验证 {Age, Income} 是否是有效调整集

# 你的代码
...
```

---

### 缺失内容清单

- [ ] d-分离的形式化定义和判断算法
- [ ] 最小调整集的查找算法
- [ ] 碰撞偏差的数学推导
- [ ] 工具变量、前门准则的介绍
- [ ] DFS 路径查找的参考实现
- [ ] 复杂 DAG 的实战案例
- [ ] 思考题参考答案
- [ ] 面试题模拟

### 具体改进建议

#### 优先级 1（高）: 补充 d-分离理论

在 Cell 14 后增加：
1. d-分离的定义
2. 三种结构的阻断规则
3. Bayes Ball 算法（可选）

#### 优先级 2（高）: 补充最小调整集算法

在 Cell 16 后增加 `find_minimal_adjustment_sets` 函数，并用示例验证。

#### 优先级 3（中）: 碰撞偏差数学推导

在 Cell 21 后增加数学证明部分，解释为什么控制 Collider 会产生虚假关联。

#### 优先级 4（中）: 增加其他识别策略

在最后增加一节"其他识别策略"，简要介绍：
- 前门准则
- 工具变量
- 断点回归（RDD）
- 双重差分（DID）

#### 优先级 5（低）: 增加实战案例

增加 2-3 个复杂 DAG 的分析案例，覆盖：
- 多个混淆变量
- 混淆+中介+碰撞混合
- 未观测混淆的场景

---

## 3️⃣ part0_3_identification_strategies.ipynb

### 整体评分: N/A（文件损坏）

**问题:** 该文件无法正常解析（JSON 格式错误），需要修复后再审核。

**建议:**
1. 检查文件编码和格式
2. 使用 `nbformat` 库验证：
   ```python
   import nbformat
   with open('part0_3_identification_strategies.ipynb') as f:
       nb = nbformat.read(f, as_version=4)
   ```
3. 如果是手动编辑导致，使用 Jupyter 重新保存

---

## 4️⃣ part0_4_bias_types.ipynb

### 整体评分: 8.0/10

### 优点

1. **理论+实践结合出色**
   - 混淆偏差公式：Bias = γ × δ
   - 通过模拟数据验证公式
   - Simpson's Paradox 的完整案例

2. **可视化丰富**
   - 偏差随混淆强度变化的曲线
   - Simpson's Paradox 的整体 vs 分层对比
   - 敏感性分析热力图

3. **敏感性分析亮点**
   - 引入未观测混淆的概念
   - 热力图展示不同 (γ_u, δ_u) 组合下的效应
   - 实用性强

4. **案例真实**
   - 伯克利性别歧视案例
   - 医院治疗案例
   - 红酒与心脏病案例

### 问题清单

#### [严重] 遗漏变量偏差公式缺少推导 (Cell 2)

**问题位置:** Cell 2

**问题描述:**
直接给出 Bias = γ × δ，没有解释为什么是这个公式。

**建议增加:**

```markdown
### 🧮 遗漏变量偏差公式推导

**场景:** 真实模型包含 X，但我们遗漏了 X

真实模型:
$$Y = \beta_0 + \beta_1 T + \beta_2 X + \epsilon$$

遗漏 X 的模型:
$$Y = \alpha_0 + \alpha_1 T + u$$

**问题:** $\alpha_1$（遗漏模型的 T 系数）和 $\beta_1$（真实效应）的关系？

**推导:**

将真实模型代入遗漏模型：
\begin{align}
Y &= \alpha_0 + \alpha_1 T + u \\
\beta_0 + \beta_1 T + \beta_2 X + \epsilon &= \alpha_0 + \alpha_1 T + u
\end{align}

用 OLS 求解 $\alpha_1$：
$$\alpha_1 = \frac{\text{Cov}(T, Y)}{\text{Var}(T)}$$

展开 $\text{Cov}(T, Y)$：
\begin{align}
\text{Cov}(T, Y) &= \text{Cov}(T, \beta_0 + \beta_1 T + \beta_2 X + \epsilon) \\
&= \beta_1 \text{Var}(T) + \beta_2 \text{Cov}(T, X)
\end{align}

所以：
$$\alpha_1 = \beta_1 + \beta_2 \frac{\text{Cov}(T, X)}{\text{Var}(T)}$$

令：
- $\gamma = \beta_2$（X 对 Y 的效应）
- $\delta = \frac{\text{Cov}(T, X)}{\text{Var}(T)}$（回归 T ~ X 的系数）

得到：
$$\boxed{\text{Bias} = \alpha_1 - \beta_1 = \gamma \times \delta}$$

**关键洞察:**
- $\gamma = 0$ → 无偏差（X 不影响 Y）
- $\delta = 0$ → 无偏差（X 与 T 无关）
- $\gamma \cdot \delta > 0$ → 正向偏差（高估）
- $\gamma \cdot \delta < 0$ → 负向偏差（低估）
```

#### [严重] Simpson's Paradox 代码未完成 (Cell 12, 13)

**问题位置:** Cell 12 (`create_simpson_paradox_data`), Cell 13 (`analyze_simpson_paradox`)

**问题描述:**
大量 TODO 未填写，没有参考实现。

**建议修复:**
提供完整参考实现：

```python
# ===== 参考实现 =====
def create_simpson_paradox_data_solution(n_per_group=500, seed=42):
    """完整实现"""
    np.random.seed(seed)
    data = []

    # 医院 A (重症多，用药多)
    n_A_treated = int(n_per_group * 1.5)
    recovery_A_treated = np.random.binomial(1, 0.50, n_A_treated)
    for i in range(n_A_treated):
        data.append({
            '医院': 'A (重症)',
            '用药': 1,
            '康复': recovery_A_treated[i],
            '病情': '重症'
        })

    n_A_control = int(n_per_group * 0.3)
    recovery_A_control = np.random.binomial(1, 0.30, n_A_control)
    for i in range(n_A_control):
        data.append({
            '医院': 'A (重症)',
            '用药': 0,
            '康复': recovery_A_control[i],
            '病情': '重症'
        })

    # 医院 B (轻症多，用药少)
    n_B_treated = int(n_per_group * 0.3)
    recovery_B_treated = np.random.binomial(1, 0.90, n_B_treated)
    for i in range(n_B_treated):
        data.append({
            '医院': 'B (轻症)',
            '用药': 1,
            '康复': recovery_B_treated[i],
            '病情': '轻症'
        })

    n_B_control = int(n_per_group * 1.5)
    recovery_B_control = np.random.binomial(1, 0.70, n_B_control)
    for i in range(n_B_control):
        data.append({
            '医院': 'B (轻症)',
            '用药': 0,
            '康复': recovery_B_control[i],
            '病情': '轻症'
        })

    return pd.DataFrame(data)

def analyze_simpson_paradox_solution(df):
    """完整实现"""
    results = {}

    # 整体效应
    overall_treated = df[df['用药'] == 1]['康复'].mean()
    overall_control = df[df['用药'] == 0]['康复'].mean()
    results['整体-用药组康复率'] = overall_treated
    results['整体-未用药组康复率'] = overall_control
    results['整体-效应'] = overall_treated - overall_control

    # 医院 A
    df_A = df[df['医院'] == 'A (重症)']
    results['医院A-用药组康复率'] = df_A[df_A['用药'] == 1]['康复'].mean()
    results['医院A-未用药组康复率'] = df_A[df_A['用药'] == 0]['康复'].mean()
    results['医院A-效应'] = results['医院A-用药组康复率'] - results['医院A-未用药组康复率']

    # 医院 B
    df_B = df[df['医院'] == 'B (轻症)']
    results['医院B-用药组康复率'] = df_B[df_B['用药'] == 1]['康复'].mean()
    results['医院B-未用药组康复率'] = df_B[df_B['用药'] == 0]['康复'].mean()
    results['医院B-效应'] = results['医院B-用药组康复率'] - results['医院B-未用药组康复率']

    return results
```

#### [中等] 敏感性分析缺少实用指南 (Cell 18-20)

**问题位置:** Cell 18-20

**问题描述:**
敏感性分析的代码很好，但缺少"如何使用分析结果做决策"的指导。

**建议增加:**

```markdown
### 🎯 敏感性分析的实战应用

#### 如何解读敏感性分析？

1. **评估结论的稳健性**
   - 如果热力图中大部分区域都是正（或负），说明结论稳健
   - 如果零点附近，说明结论对未观测混淆敏感

2. **定量评估**
   - 计算"使效应为 0"的临界值
   - 例如：需要 $\gamma_u \times \delta_u > 2.5$ 才能推翻结论
   - 问自己："这种强度的未观测混淆现实吗？"

3. **与领域知识结合**
   - 咨询领域专家：可能存在哪些未观测混淆？
   - 评估这些混淆的可能强度（γ_u 和 δ_u）
   - 判断是否在敏感区域

#### 实战案例：评估教育对收入的影响

当前估计：大学毕业比高中毕业多赚 $30,000/年

**可能的未观测混淆:** 天赋、家庭背景、社交能力

**敏感性分析:**
- 如果天赋对收入的效应 γ_u = $20,000/年
- 且天赋与上大学的关联 δ_u = 0.5
- 则偏差 = $10,000
- 真实效应可能是 $20,000（仍然显著）

**结论:** 即使存在中等强度的未观测混淆，效应仍为正。

#### 报告模板

```
我们估计处理效应为 X。

敏感性分析表明：
- 如果存在未观测混淆 U 满足 γ_u × δ_u > Y，结论会被推翻
- 这要求 U 对结果的影响至少为 γ_u，且与处理的关联至少为 δ_u
- 根据领域知识，这种强度的混淆 [可能/不太可能] 存在

因此，我们的结论 [稳健/需要谨慎对待]。
```
```

#### [建议] 增加 Rosenbaum 敏感性分析

**建议新增:**

```python
### 📊 Rosenbaum 敏感性分析（Γ 分析）

def rosenbaum_sensitivity(df, gamma_range=None):
    """
    Rosenbaum 敏感性分析

    评估：需要多强的未观测混淆才能改变结论？

    Γ (Gamma): 混淆强度参数
    - Γ = 1: 无未观测混淆（随机实验）
    - Γ = 2: 未观测混淆可能使倾向得分相差 2 倍
    - Γ = 3: 更强的混淆

    Returns:
        DataFrame with columns ['Gamma', 'p_value_lower', 'p_value_upper']
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.5, 2.0, 2.5, 3.0]

    from scipy.stats import binom_test

    results = []
    for gamma in gamma_range:
        # 计算在 gamma 混淆强度下的 p-value 边界
        # （实际实现需要使用 Hodges-Lehmann 估计）
        # 这里简化演示
        p_lower = ...  # 最坏情况的 p-value（下界）
        p_upper = ...  # 最好情况的 p-value（上界）

        results.append({
            'Gamma': gamma,
            'p_value_lower': p_lower,
            'p_value_upper': p_upper,
            'significant': p_upper < 0.05
        })

    return pd.DataFrame(results)

# 示例
print("Rosenbaum 敏感性分析:")
print(rosenbaum_sensitivity(df))

# 解读：如果 Γ=2 时仍然显著，说明结论对中等混淆稳健
```

---

### 缺失内容清单

- [ ] 遗漏变量偏差公式的完整推导
- [ ] Simpson's Paradox 代码的参考实现
- [ ] 敏感性分析的实战应用指南
- [ ] Rosenbaum 敏感性分析
- [ ] 思考题参考答案
- [ ] 面试题模拟

### 具体改进建议

#### 优先级 1（高）: 补充偏差公式推导

在 Cell 2 后增加完整的数学推导，从 OLS 公式出发，一步步推导出 Bias = γ × δ。

#### 优先级 2（高）: 补充 Simpson's Paradox 参考实现

在 Cell 12 和 13 后增加参考答案 cell。

#### 优先级 3（中）: 增加敏感性分析实战指南

在 Cell 20 后增加：
1. 如何解读热力图
2. 如何定量评估临界值
3. 如何与领域知识结合
4. 报告模板

#### 优先级 4（中）: 增加 Rosenbaum 敏感性分析

增加经典的 Γ 分析方法，作为敏感性分析的补充。

#### 优先级 5（低）: 增加实战案例

增加 2-3 个真实的混淆偏差案例分析：
- 教育对收入
- 吸烟对肺癌
- 促销对购买

---

## 🎯 总体建议

### 优先级排序

#### P0 (必须完成)

1. **补充所有 TODO 的参考实现**
   - 每个函数都应该有完整的参考答案
   - 建议使用可折叠的 markdown cell

2. **修复 part0_3_identification_strategies.ipynb**
   - 文件格式错误，无法解析
   - 优先修复后再审核

3. **补充核心数学推导**
   - 观测公式的推导
   - ATE 无偏性证明
   - 遗漏变量偏差公式推导
   - 碰撞偏差的数学原理

#### P1 (强烈建议)

1. **增加面试题模拟**
   - 每个 notebook 增加 3-5 道面试题
   - 包含快速问答、编码题、场景分析

2. **增加库对比**
   - EconML
   - CausalML
   - DoWhy
   - 展示如何用库实现同样的分析

3. **补充思考题答案**
   - 提供详细的参考答案
   - 说明评分标准

#### P2 (建议完成)

1. **增加高级内容**
   - d-分离算法
   - 最小调整集算法
   - 前门准则、工具变量
   - Rosenbaum 敏感性分析

2. **增加实战案例**
   - 复杂 DAG 分析
   - 真实业务场景
   - 端到端分析流程

### 通用模板建议

为了保持 notebook 的一致性，建议使用统一的结构模板：

```markdown
# 📖 第 X 章 练习 Y: [标题]

## 🎯 学习目标
- 目标 1
- 目标 2
- 目标 3

## 📚 Part 1: 理论基础
### 概念解释
### 数学公式
### 公式推导（新增）

## 🔬 Part 2: 从零实现
### 数据生成
### 核心算法实现
### TODO 练习

## 💡 Part 3: 参考实现（新增）
### 完整代码
### 关键点解释
### 对比分析

## 🔧 Part 4: 使用因果推断库（新增）
### EconML 实现
### CausalML 实现
### DoWhy 实现
### 对比总结

## 📊 Part 5: 案例分析
### 真实场景
### 数据分析
### 结果解读

## 💼 Part 6: 面试题模拟（新增）
### 快速问答（2-3 题）
### 编码题（1-2 题）
### 场景分析（1 题）

## 📝 Part 7: 思考题
### 问题 1
### 问题 2
### ...
### 参考答案（新增）

## 🎉 总结
### 核心公式
### 关键洞察
### 下一步
```

### 质量保证 Checklist

对每个 notebook，确保：

- [ ] 所有 TODO 都有参考实现
- [ ] 所有核心公式都有推导
- [ ] 所有思考题都有参考答案
- [ ] 有至少 3 道面试题
- [ ] 有与主流库的对比
- [ ] 有至少 1 个真实案例
- [ ] 代码能正常运行（测试过）
- [ ] 可视化清晰易读
- [ ] 教学叙事流畅
- [ ] 难度曲线合理

---

## 📌 待办事项总结

### Part 0.1 - Potential Outcomes
- [ ] 补充 5 个函数的参考实现
- [ ] 增加观测公式和 ATE 无偏性推导
- [ ] 增加思考题答案
- [ ] 增加面试题模拟（3-5 道）
- [ ] 增加 EconML/CausalML 对比

### Part 0.2 - Causal DAG
- [ ] 补充 d-分离定义和算法
- [ ] 补充最小调整集查找算法
- [ ] 补充碰撞偏差数学推导
- [ ] 补充 DFS 路径查找参考实现
- [ ] 增加前门准则、工具变量介绍
- [ ] 增加复杂 DAG 实战案例
- [ ] 增加思考题答案
- [ ] 增加面试题模拟

### Part 0.3 - Identification Strategies
- [ ] **修复文件格式错误**（最优先）
- [ ] 完成后重新审核

### Part 0.4 - Bias Types
- [ ] 补充遗漏变量偏差公式推导
- [ ] 补充 Simpson's Paradox 参考实现
- [ ] 增加敏感性分析实战指南
- [ ] 增加 Rosenbaum 敏感性分析
- [ ] 增加思考题答案
- [ ] 增加面试题模拟

---

## 🏆 最终评价

Part 0 Foundation 的教学设计和互动性都非常出色，是一个优秀的因果推断入门教程。但作为"教程+面试准备"双重目标的项目，仍然缺少：

1. **完整的参考实现** - 学习者无法验证自己的代码
2. **数学推导** - 只给公式不给证明，理解不够深入
3. **面试准备** - 缺少面试题，无法直接用于面试复习
4. **库对比** - 缺少工业界标准库的使用教学

建议按照 P0 → P1 → P2 的优先级顺序完善，预计需要：
- **P0 内容:** 2-3 天（补充参考实现和核心推导）
- **P1 内容:** 3-4 天（增加面试题、库对比、思考题答案）
- **P2 内容:** 5-7 天（增加高级内容和实战案例）

完成后，Part 0 将成为一个理论扎实、实践完整、面向面试的高质量因果推断教程。

---

**审核完成日期:** 2026-01-04
**下一步:** 开始 Part 1 Experimentation 的审核
