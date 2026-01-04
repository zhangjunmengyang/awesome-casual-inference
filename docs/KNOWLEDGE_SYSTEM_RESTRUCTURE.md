# 因果推断知识体系重构方案

> 站在 Google/Meta/Grab 数据科学家面试官视角，重新设计知识体系
> 目标：学完后具备扎实的理论基础 + 丰富的业务场景经验 + 能应对高难度面试

---

## 一、现有体系的问题诊断

### 1.1 结构性问题

| 问题 | 具体表现 | 影响 |
|------|----------|------|
| **方法论导向而非问题导向** | 按算法分类（PSM→IPW→DR），而非按「什么情况用什么方法」 | 面试时无法快速判断场景该用什么方法 |
| **缺乏识别策略框架** | 直接讲估计方法，跳过「如何识别因果效应」 | 不理解方法背后的假设，容易误用 |
| **准实验方法完全缺失** | 没有 DID、RDD、IV、合成控制 | 这是大厂面试的高频考点 |
| **A/B 测试深度不够** | 只有 CUPED，缺少网络效应、长期效应、Switchback 等 | 无法处理真实业务中的复杂实验设计 |
| **营销归因完全空白** | 作为营销算法岗核心能力，完全没有涉及 | 致命短板 |

### 1.2 深度问题

| 模块 | 现状 | 差距 |
|------|------|------|
| PSM/IPW/DR | 有实现，但缺乏「什么时候用哪个」的决策框架 | 需要加入方法选择的决策树 |
| Meta-Learners | S/T/X-Learner 都有，但缺 R-Learner 和 DR-Learner | R-Learner 是工业界最常用的 |
| 因果森林 | 有 econml 的封装，但原理讲解不够深入 | 需要从 Honest Trees 讲起 |
| 深度学习 | TARNet/DragonNet 有，但缺 CEVAE、表示学习的 IPM Loss | 需要补充完整的表示学习理论 |

### 1.3 广度问题

**完全缺失的关键主题：**

```
准实验方法 (Quasi-Experiments)
├── 双重差分 (DID)
├── 合成控制 (Synthetic Control)
├── 断点回归 (RDD)
└── 工具变量 (IV)

A/B 测试进阶
├── 网络效应 / 溢出效应
├── Switchback 实验
├── 多臂老虎机 (MAB)
├── 长期效应估计 (Surrogate Index)
└── 分层分析 / 分层抽样

营销归因
├── 规则归因 (Last-touch, First-touch, Linear)
├── Shapley 归因
├── Markov Chain 归因
└── 增量归因 vs 触达归因

因果发现
├── PC 算法
├── FCI 算法
└── 因果图学习

高级主题
├── 连续处理效应 (Continuous Treatment)
├── 多值处理效应 (Multi-valued Treatment)
├── 动态处理效应 (Time-varying Treatment)
└── 中介分析 (Mediation Analysis)
```

---

## 二、重构后的知识体系

### 2.1 设计原则

1. **问题驱动**：从「我有什么数据/场景」出发，引导到「该用什么方法」
2. **识别优先**：先讲「在什么假设下能识别因果效应」，再讲「如何估计」
3. **业务闭环**：每个方法都要有真实业务场景的端到端案例
4. **渐进深入**：基础 → 进阶 → 前沿，每层都能独立成体系

### 2.2 新的知识架构

```
                    ┌─────────────────────────────────────────┐
                    │         PART 0: 因果思维基础              │
                    │   (潜在结果、因果图、识别策略框架)          │
                    └─────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   PART 1: 实验方法   │    │  PART 2: 观测数据   │    │  PART 3: 准实验     │
│   (A/B Testing)     │    │  (Observational)    │    │  (Quasi-Experiment) │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
            │                           │                           │
            ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ - 基础实验设计       │    │ - 选择可观测混淆     │    │ - DID               │
│ - CUPED            │    │ - PSM / IPW / DR    │    │ - 合成控制          │
│ - 分层分析          │    │ - 敏感性分析         │    │ - RDD               │
│ - 网络效应          │    │                     │    │ - IV                │
│ - Switchback       │    │                     │    │                     │
│ - 长期效应          │    │                     │    │                     │
│ - MAB/Bandit       │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌─────────────────────┐                ┌─────────────────────┐
        │   PART 4: 异质效应   │                │   PART 5: 深度学习   │
        │   (CATE / Uplift)   │                │   (Representation)  │
        └─────────────────────┘                └─────────────────────┘
                    │                                       │
                    ▼                                       ▼
        ┌─────────────────────┐                ┌─────────────────────┐
        │ - Meta-Learners     │                │ - TARNet / DragonNet│
        │ - 因果森林           │                │ - CEVAE             │
        │ - 评估方法           │                │ - 表示学习理论       │
        └─────────────────────┘                └─────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌─────────────────────┐                ┌─────────────────────┐
        │   PART 6: 营销应用   │                │   PART 7: 高级主题   │
        │   (Marketing)       │                │   (Advanced)        │
        └─────────────────────┘                └─────────────────────┘
                    │                                       │
                    ▼                                       ▼
        ┌─────────────────────┐                ┌─────────────────────┐
        │ - 营销归因           │                │ - 因果发现          │
        │ - 预算分配优化       │                │ - 连续/多值处理      │
        │ - 智能发券           │                │ - 动态处理效应       │
        │ - 用户定向           │                │ - 中介分析          │
        └─────────────────────┘                └─────────────────────┘
```

---

## 三、各 Part 详细设计

### PART 0: 因果思维基础 (Foundation)

**目标**：建立正确的因果推断思维框架，理解「识别」与「估计」的区别

#### 模块 0.1: 潜在结果框架 (Potential Outcomes)
- 反事实的概念
- ATE / ATT / ATC / CATE 的区别
- 因果推断的根本问题

#### 模块 0.2: 因果图与 DAG (Causal DAG)
- 因果图的三种基本结构：链、叉、对撞
- d-分离准则
- 后门准则与前门准则
- 识别因果效应的图方法

#### 模块 0.3: 识别策略框架 (Identification Strategies)
- **这是面试核心！** 需要一个决策树：
  ```
  我想估计 T 对 Y 的因果效应
      │
      ├── 能做随机实验吗？
      │       ├── 能 → A/B Testing (PART 1)
      │       └── 不能 → 继续判断
      │
      ├── 有「自然实验」吗？(外生变异)
      │       ├── 有时间断点 → DID
      │       ├── 有政策门槛 → RDD
      │       ├── 有工具变量 → IV
      │       └── 没有 → 继续判断
      │
      ├── 能控制所有混淆变量吗？
      │       ├── 能 (CIA/Unconfoundedness) → PSM/IPW/DR
      │       └── 不能 → 敏感性分析 + 谨慎解释
      │
      └── 目标是 ATE 还是 CATE？
              ├── ATE → 传统方法
              └── CATE → Meta-Learners / 因果森林
  ```

#### 模块 0.4: 混淆与偏差 (Bias Types)
- 混淆偏差 (Confounding Bias)
- 选择偏差 (Selection Bias)
- 测量误差 (Measurement Error)
- 遗漏变量偏差 (Omitted Variable Bias)

---

### PART 1: 实验方法 (Experimentation)

**目标**：掌握 A/B 测试的完整知识体系，从基础到高级

#### 模块 1.1: 实验设计基础
- 随机化的原理与实现
- 样本量计算与功效分析
- 分层随机化 (Stratified Randomization)
- AA 测试

#### 模块 1.2: CUPED 与方差缩减
- CUPED 原理与实现
- 多协变量 CUPED
- CUPAC (连续结果)
- 最优协变量选择

#### 模块 1.3: 网络效应与溢出
- SUTVA 违背的场景
- 聚类随机化 (Cluster Randomization)
- Ego-cluster 方法
- 溢出效应估计

#### 模块 1.4: Switchback 实验
- 时间序列实验设计
- Uber/Lyft 的 Switchback 实践
- 自相关与方差估计

#### 模块 1.5: 长期效应估计
- Surrogate Index 方法
- 长短期效应的关系建模
- Netflix/Meta 的实践

#### 模块 1.6: 多臂老虎机 (MAB)
- Thompson Sampling
- UCB 算法
- Contextual Bandit
- MAB vs A/B Testing 的权衡

#### 模块 1.7: 异质效应驱动的实验
- 最优停止规则
- 贝叶斯优化
- 个性化实验

---

### PART 2: 观测数据方法 (Observational Methods)

**目标**：掌握基于「选择可观测」假设的因果推断方法

#### 模块 2.1: 倾向得分 (Propensity Score)
- 倾向得分的定义与性质
- Rosenbaum-Rubin 定理
- 估计方法：Logistic / GBM / Neural Network
- 诊断：共同支撑、平衡性检验

#### 模块 2.2: 匹配方法 (Matching)
- 精确匹配 vs 近似匹配
- PSM vs CEM vs 马氏距离匹配
- 匹配后估计量的选择
- 匹配的局限性

#### 模块 2.3: 加权方法 (Weighting)
- IPW 的原理与实现
- 稳定化权重 (Stabilized Weights)
- 重叠权重 (Overlap Weights)
- 熵权重 (Entropy Weights)
- 极端权重的处理

#### 模块 2.4: 双重稳健估计 (Doubly Robust)
- AIPW 的原理
- 为什么「双重稳健」
- 交叉拟合 (Cross-fitting)
- 双重机器学习 (DML)

#### 模块 2.5: 敏感性分析 (Sensitivity Analysis)
- Rosenbaum Bounds
- E-value
- 部分识别 (Partial Identification)
- Omitted Variable Bias 公式

---

### PART 3: 准实验方法 (Quasi-Experimental Methods)

**目标**：掌握利用「自然实验」识别因果效应的方法 —— **面试高频考点**

#### 模块 3.1: 双重差分 (Difference-in-Differences)
- 基本 DID 设计
- 平行趋势假设与检验
- 安慰剂检验
- 两期 DID vs 多期 DID
- 交错 DID (Staggered DID) 与 Callaway-Sant'Anna 估计量
- Event Study 设计

#### 模块 3.2: 合成控制 (Synthetic Control)
- 合成控制的思想
- 权重估计
- 推断方法 (Placebo Tests)
- 与 DID 的关系
- 广义合成控制 (GSC)

#### 模块 3.3: 断点回归 (Regression Discontinuity)
- Sharp RDD vs Fuzzy RDD
- 带宽选择 (Bandwidth Selection)
- 局部多项式回归
- RDD 的检验：McCrary 密度检验、协变量平滑检验
- 业务案例：优惠券门槛、会员等级

#### 模块 3.4: 工具变量 (Instrumental Variables)
- IV 的三个假设
- 2SLS 估计
- 弱工具变量问题与检验
- 过度识别检验
- 局部平均处理效应 (LATE)
- 业务案例：价格弹性估计、广告效果

---

### PART 4: 异质性效应估计 (Heterogeneous Treatment Effects)

**目标**：掌握 CATE 估计的完整方法论，这是营销算法的核心

#### 模块 4.1: Meta-Learners 系列
- S-Learner：原理、优缺点、正则化偏差问题
- T-Learner：原理、优缺点、样本不平衡问题
- X-Learner：设计思想、两阶段估计
- R-Learner：损失函数设计、与 DML 的关系
- DR-Learner：结合双重稳健的 CATE 估计

#### 模块 4.2: 因果森林 (Causal Forest)
- Honest Trees 的设计
- 分裂准则：最大化 CATE 异质性
- 置信区间估计
- 特征重要性解释
- GRF (Generalized Random Forest)

#### 模块 4.3: Uplift Modeling
- Uplift 与 CATE 的关系
- Uplift Tree 的分裂准则
- 评估方法：Qini Curve、AUUC、Uplift Curve
- 与响应模型的区别

#### 模块 4.4: CATE 评估
- PEHE (需要真实 ITE)
- CATE 排序能力评估
- Calibration 检验
- 业务指标：增量收益、ROI

---

### PART 5: 深度学习因果推断 (Deep Causal Learning)

**目标**：理解表示学习在因果推断中的应用

#### 模块 5.1: 表示学习基础
- 为什么需要表示学习
- 平衡表示 (Balanced Representation)
- IPM 距离 (MMD, Wasserstein)

#### 模块 5.2: TARNet 与 DragonNet
- TARNet 架构与 Factual Loss
- DragonNet：加入倾向得分预测
- 目标正则化 (Targeted Regularization)

#### 模块 5.3: CEVAE
- 变分自编码器基础
- CEVAE 的生成模型
- 隐变量建模

#### 模块 5.4: 其他深度因果模型
- CFR (Counterfactual Regression)
- GANITE
- TransTEE (Transformer)

---

### PART 6: 营销应用 (Marketing Applications)

**目标**：将因果推断方法应用于真实营销场景 —— **营销算法岗核心**

#### 模块 6.1: 营销归因 (Marketing Attribution)
- 规则归因：Last-touch, First-touch, Linear, Time-decay
- Shapley 归因：原理与计算
- Markov Chain 归因
- 数据驱动归因 (DDA)
- 增量归因 vs 触达归因的本质区别
- 业务案例：多渠道归因

#### 模块 6.2: 智能发券 (Coupon Optimization)
- 四类用户分群 (Persuadables, Sure Things, Lost Causes, Sleeping Dogs)
- Uplift 建模实战
- ROI 优化策略
- 预算约束下的最优发券

#### 模块 6.3: 用户定向 (User Targeting)
- 增量定向 vs 响应定向
- Uplift-based Targeting
- 预算分配优化
- 多目标优化

#### 模块 6.4: 预算分配 (Budget Allocation)
- 边际 ROI 优化
- 多渠道预算分配
- 约束优化问题建模
- 在线学习与预算控制

---

### PART 7: 高级主题 (Advanced Topics)

**目标**：了解因果推断的前沿方向，体现知识深度

#### 模块 7.1: 因果发现 (Causal Discovery)
- PC 算法
- FCI 算法
- 基于得分的方法
- 因果图学习

#### 模块 7.2: 连续与多值处理
- Generalized Propensity Score
- 剂量-响应曲线 (Dose-Response)
- 多值处理的 CATE

#### 模块 7.3: 时变处理 (Time-varying Treatment)
- 边际结构模型 (MSM)
- G-computation
- 序贯实验

#### 模块 7.4: 中介分析 (Mediation Analysis)
- 直接效应与间接效应
- Baron-Kenny 方法
- 因果中介分析

---

## 四、UI 模块与 Notebook 的对应关系

### 4.1 UI 模块设计原则

UI 用于**可视化展示**和**交互式探索**，应该：
- 突出关键概念的可视化
- 支持参数调节看结果变化
- 不需要太深的数学推导

### 4.2 Notebook 设计原则

Notebook 用于**扎实学习**和**动手实现**，应该：
- 从原理讲起，有完整的数学推导
- 有 TODO 练习，要求自己实现核心算法
- 有思考题，检验理解深度
- 比 UI 更细化，覆盖更多细节

### 4.3 对应关系

| Part | UI 模块 | Notebook 数量 | 说明 |
|------|---------|--------------|------|
| Part 0 | Foundation Lab | 4 个 | UI 展示 DAG 可视化；Notebook 深入推导 |
| Part 1 | Experiment Lab | 6 个 | UI 做样本量计算器；Notebook 实现各方法 |
| Part 2 | Observational Lab | 5 个 | UI 展示匹配可视化；Notebook 实现算法 |
| Part 3 | Quasi-Experiment Lab | 4 个 | UI 展示 DID 图；Notebook 完整实现 |
| Part 4 | CATE Lab | 5 个 | UI 展示 Uplift 曲线；Notebook 实现 Learners |
| Part 5 | Deep Learning Lab | 3 个 | UI 展示网络结构；Notebook 实现 PyTorch |
| Part 6 | Marketing Lab | 4 个 | UI 做归因计算器；Notebook 端到端案例 |
| Part 7 | Advanced Lab | 4 个 | UI 展示因果图学习；Notebook 前沿方法 |

**总计：35 个 Notebook**（现有 18 个，需新增 17 个）

---

## 五、面试准备检查清单

学完本体系后，应该能够回答：

### 5.1 基础概念类
- [ ] 什么是潜在结果框架？ATE/ATT/CATE 的区别？
- [ ] 因果推断的根本问题是什么？
- [ ] DAG 的三种基本结构？什么是 d-分离？
- [ ] 后门准则与前门准则分别是什么？

### 5.2 方法选择类
- [ ] 什么情况下用 PSM vs IPW vs DR？
- [ ] DID 的核心假设是什么？如何检验？
- [ ] RDD 和 DID 的区别？
- [ ] IV 的三个假设分别是什么？

### 5.3 实验设计类
- [ ] CUPED 的原理是什么？
- [ ] 如何处理 A/B 测试中的网络效应？
- [ ] MAB 和传统 A/B 测试的权衡？
- [ ] 如何估计长期效应？

### 5.4 CATE/Uplift 类
- [ ] S/T/X/R-Learner 的区别？
- [ ] 因果森林的 Honest Splitting 是什么？
- [ ] Qini 曲线如何解读？
- [ ] Uplift 建模与响应率建模的本质区别？

### 5.5 营销应用类
- [ ] Shapley 归因的原理？
- [ ] 增量归因与触达归因的区别？
- [ ] 如何设计发券实验？
- [ ] 如何做预算分配优化？

### 5.6 高级问题类
- [ ] 敏感性分析的 E-value 是什么？
- [ ] DML 的核心思想？
- [ ] 如何处理连续处理变量？
- [ ] 中介分析与因果推断的关系？

---

## 六、实施路线图

### Phase 1: 补齐核心缺失 (优先级 P0)
1. 新增 Part 3 准实验方法（DID、RDD、IV、合成控制）
2. 扩展 Part 1 A/B 测试（网络效应、MAB、长期效应）
3. 新增 Part 6 营销归因模块

### Phase 2: 深化现有内容 (优先级 P1)
1. 重构 Part 0，加入识别策略框架
2. 扩展 Part 4，补充 R-Learner 和 DR-Learner
3. 完善 Part 2，加入敏感性分析详细内容

### Phase 3: 补充高级主题 (优先级 P2)
1. 新增 Part 7 高级主题
2. 完善 Part 5 深度学习内容

---

## 七、与现有代码的映射

### 7.1 现有模块重组

| 现有模块 | 新位置 | 改动 |
|----------|--------|------|
| foundation_lab | Part 0 | 拆分，加入识别策略 |
| treatment_effect_lab | Part 2 | 保持，补充敏感性分析 |
| uplift_lab | Part 4 | 保持，补充 R/DR-Learner |
| hetero_effect_lab | Part 4 | 合并到 CATE Lab |
| deep_causal_lab | Part 5 | 保持，补充 CEVAE |
| ab_testing_toolkit | Part 1 | 大幅扩展 |
| case_studies | Part 6 | 重构为营销应用 |
| evaluation_lab | 分散 | 评估方法分散到各 Part |

### 7.2 新增模块

| 新模块 | 位置 | 内容 |
|--------|------|------|
| quasi_experiment_lab | Part 3 | DID, RDD, IV, SC |
| marketing_attribution | Part 6 | 归因模型 |
| causal_discovery | Part 7 | PC, FCI |
| advanced_causal | Part 7 | 连续处理、中介分析 |

---

*本文档为因果推断知识体系重构的总体设计，具体 Notebook 改进方案见 `NOTEBOOK_IMPROVEMENT_PLAN.md`*
