# Notebook 改进方案

> 本文档详细规划 Notebook 的改进方案，包括现有 Notebook 的重组和新增 Notebook 的设计
> 目标：形成 35 个高质量的教学 Notebook，覆盖因果推断完整知识体系

---

## 一、现有 Notebook 清单与重组方案

### 1.1 现有 18 个 Notebook

| 原编号 | 原标题 | 内容评估 |
|--------|--------|----------|
| chapter1_ex1 | potential_outcomes | ✅ 质量高，保留 |
| chapter1_ex2 | causal_dag | ✅ 质量高，保留 |
| chapter1_ex3 | confounding_bias | ✅ 质量高，保留 |
| chapter2_ex1 | propensity_score | ✅ 质量高，保留 |
| chapter2_ex2 | ipw | ✅ 质量高，保留 |
| chapter2_ex3 | doubly_robust | ✅ 质量高，保留 |
| chapter3_ex1 | meta_learners | ✅ 质量高，需补充 R-Learner |
| chapter3_ex2 | uplift_tree | ✅ 质量高，保留 |
| chapter3_ex3 | uplift_evaluation | ✅ 质量高，保留 |
| chapter4_ex1 | representation_learning | ⚠️ 需加深理论 |
| chapter4_ex2 | tarnet | ✅ 质量高，保留 |
| chapter4_ex3 | dragonnet | ✅ 质量高，保留 |
| chapter5_ex1 | cate_basics | ✅ 质量高，保留 |
| chapter5_ex2 | causal_forest | ✅ 质量高，保留 |
| chapter5_ex3 | sensitivity_analysis | ⚠️ 需扩展 E-value |
| chapter6_ex1 | coupon_optimization | ✅ 质量高，保留 |
| chapter6_ex2 | ab_enhancement (CUPED) | ✅ 质量高，保留 |
| chapter6_ex3 | user_targeting | ✅ 质量高，保留 |

### 1.2 重组映射表

| 新编号 | 新标题 | 来源 | 改动说明 |
|--------|--------|------|----------|
| **Part 0: Foundation** |
| 0.1 | potential_outcomes | chapter1_ex1 | 重命名，内容保持 |
| 0.2 | causal_dag | chapter1_ex2 | 重命名，内容保持 |
| 0.3 | identification_strategies | **新增** | 核心新内容 |
| 0.4 | bias_types | chapter1_ex3 | 重命名，扩展选择偏差 |
| **Part 1: Experimentation** |
| 1.1 | ab_testing_basics | **新增** | 基础实验设计 |
| 1.2 | cuped_variance_reduction | chapter6_ex2 | 移动，扩展多协变量 |
| 1.3 | stratified_analysis | **新增** | 分层分析 |
| 1.4 | network_effects | **新增** | 网络效应处理 |
| 1.5 | switchback_experiments | **新增** | Switchback 设计 |
| 1.6 | long_term_effects | **新增** | Surrogate Index |
| 1.7 | multi_armed_bandits | **新增** | MAB/Contextual Bandit |
| **Part 2: Observational Methods** |
| 2.1 | propensity_score | chapter2_ex1 | 重命名，内容保持 |
| 2.2 | matching_methods | **新增** | 各种匹配方法对比 |
| 2.3 | ipw_weighting | chapter2_ex2 | 重命名，扩展各种权重 |
| 2.4 | doubly_robust | chapter2_ex3 | 重命名，加入 DML |
| 2.5 | sensitivity_analysis | chapter5_ex3 | 移动，扩展 E-value |
| **Part 3: Quasi-Experiments** |
| 3.1 | difference_in_differences | **新增** | DID 完整内容 |
| 3.2 | synthetic_control | **新增** | 合成控制法 |
| 3.3 | regression_discontinuity | **新增** | RDD |
| 3.4 | instrumental_variables | **新增** | IV/2SLS |
| **Part 4: CATE / Uplift** |
| 4.1 | cate_basics | chapter5_ex1 | 移动，内容保持 |
| 4.2 | meta_learners | chapter3_ex1 | 移动，补充 R/DR-Learner |
| 4.3 | causal_forest | chapter5_ex2 | 移动，加深 GRF |
| 4.4 | uplift_tree | chapter3_ex2 | 移动，内容保持 |
| 4.5 | uplift_evaluation | chapter3_ex3 | 移动，内容保持 |
| **Part 5: Deep Learning** |
| 5.1 | representation_learning | chapter4_ex1 | 重命名，加深 IPM |
| 5.2 | tarnet_dragonnet | chapter4_ex2 + ex3 | 合并优化 |
| 5.3 | cevae_advanced | **新增** | CEVAE 等 |
| **Part 6: Marketing Applications** |
| 6.1 | marketing_attribution | **新增** | 归因模型 |
| 6.2 | coupon_optimization | chapter6_ex1 | 移动，内容保持 |
| 6.3 | user_targeting | chapter6_ex3 | 移动，内容保持 |
| 6.4 | budget_allocation | **新增** | 预算分配优化 |
| **Part 7: Advanced Topics** |
| 7.1 | causal_discovery | **新增** | PC/FCI 算法 |
| 7.2 | continuous_treatment | **新增** | 连续处理效应 |
| 7.3 | time_varying_treatment | **新增** | 时变处理/MSM |
| 7.4 | mediation_analysis | **新增** | 中介分析 |

---

## 二、新增 Notebook 详细设计

### Part 0.3: 识别策略框架 (identification_strategies)

**学习目标**：
1. 理解「识别」与「估计」的区别
2. 掌握因果效应识别的决策框架
3. 能够根据数据/场景选择正确的方法

**内容大纲**：

```markdown
# 识别策略框架 - 从数据到方法的决策指南

## Part 1: 识别 vs 估计
- 什么是「识别」(Identification)
- 什么是「估计」(Estimation)
- 为什么识别比估计更重要

## Part 2: 识别假设的层级
- Level 1: 随机化 (Randomization)
- Level 2: 条件独立 (CIA / Unconfoundedness)
- Level 3: 自然实验 (Natural Experiments)
- Level 4: 工具变量 (IV)

## Part 3: 决策树实战
- 场景 1: 有随机实验 → A/B Testing
- 场景 2: 有时间断点 → DID
- 场景 3: 有政策门槛 → RDD
- 场景 4: 有工具变量 → IV
- 场景 5: 只有观测数据 → PSM/IPW/DR

## Part 4: 案例分析
- 案例 1: 优惠券效果评估 (有实验)
- 案例 2: 政策效果评估 (DID)
- 案例 3: 价格弹性估计 (IV)
- 案例 4: 用户行为分析 (观测数据)

## 思考题
1. 为什么随机化是「黄金标准」？
2. CIA 假设在什么情况下可能不成立？
3. 如何判断一个变量是否是有效的工具变量？
```

---

### Part 1.1: A/B 测试基础 (ab_testing_basics)

**学习目标**：
1. 理解随机化实验的原理
2. 掌握样本量计算和功效分析
3. 学习分层随机化设计

**内容大纲**：

```markdown
# A/B 测试基础 - 从设计到分析

## Part 1: 随机化的原理
- 为什么随机化能识别因果效应
- 随机化的实现方式
- 伪随机与真随机

## Part 2: 实验设计
- 实验单元的选择
- 指标体系设计 (北极星指标、护栏指标)
- 分流比例的选择

## Part 3: 样本量计算
- MDE (最小可检测效应)
- 功效分析 (Power Analysis)
- 实战：样本量计算器

## Part 4: 分层随机化
- 为什么需要分层
- 分层变量的选择
- 分层 vs 事后分层

## Part 5: AA 测试
- AA 测试的作用
- 如何进行 AA 测试
- 常见问题诊断

## 练习
- 实现简单的 A/B 测试分析框架
- 设计一个完整的实验方案

## 思考题
1. 如果 p-value = 0.06，应该怎么决策？
2. 为什么需要预先确定样本量？
3. 多重比较问题如何处理？
```

---

### Part 1.4: 网络效应 (network_effects)

**学习目标**：
1. 理解 SUTVA 假设及其违背情况
2. 掌握处理网络效应的方法
3. 学习溢出效应的估计

**内容大纲**：

```markdown
# 网络效应 - 当用户之间存在相互影响

## Part 1: SUTVA 假设
- SUTVA 是什么
- SUTVA 违背的场景
  - 社交网络产品
  - 双边市场 (Uber, 外卖)
  - 共享资源 (优惠券池)

## Part 2: 聚类随机化
- 原理：以群组为单位随机化
- 聚类的选择
- 方差估计的调整

## Part 3: Ego-cluster 方法
- Ego-network 的定义
- 实验设计
- 效应估计

## Part 4: 溢出效应估计
- 直接效应 vs 间接效应
- 溢出效应的识别
- 估计方法

## Part 5: 业务案例
- 案例：社交平台新功能上线
- 案例：共享单车定价实验

## 练习
- 模拟网络数据
- 实现聚类随机化
- 估计溢出效应

## 思考题
1. 如果忽略网络效应会有什么后果？
2. 聚类大小如何影响功效？
3. 如何判断是否存在显著的溢出效应？
```

---

### Part 1.6: 长期效应估计 (long_term_effects)

**学习目标**：
1. 理解短期实验估计长期效应的挑战
2. 掌握 Surrogate Index 方法
3. 学习 Netflix/Meta 的实践

**内容大纲**：

```markdown
# 长期效应估计 - 用短期实验预测长期影响

## Part 1: 为什么需要估计长期效应
- 短期指标 vs 长期指标
- 新奇效应 (Novelty Effect)
- 学习效应 (Learning Effect)

## Part 2: Surrogate Index 方法
- 代理变量的概念
- Surrogate Index 的定义
- 识别假设

## Part 3: 估计方法
- 两阶段估计
- 长短期关系建模
- 置信区间

## Part 4: 业务实践
- Netflix 的做法
- Meta 的做法
- 常见陷阱

## Part 5: 案例实战
- 推荐算法的长期效应
- 补贴策略的长期效应

## 练习
- 实现 Surrogate Index 估计
- 分析长短期效应的差异

## 思考题
1. Surrogate Index 的关键假设是什么？
2. 如何选择好的代理变量？
3. 如果长短期效应方向相反怎么办？
```

---

### Part 1.7: 多臂老虎机 (multi_armed_bandits)

**学习目标**：
1. 理解 Explore-Exploit 权衡
2. 掌握 Thompson Sampling 和 UCB
3. 学习 Contextual Bandit

**内容大纲**：

```markdown
# 多臂老虎机 - 边学边优化

## Part 1: MAB vs A/B Testing
- A/B Testing 的「遗憾」
- MAB 的 Explore-Exploit 权衡
- 什么时候用 MAB

## Part 2: 基础算法
- Epsilon-Greedy
- UCB (Upper Confidence Bound)
- Thompson Sampling

## Part 3: Contextual Bandit
- 从 MAB 到 Contextual Bandit
- LinUCB
- 神经网络方法

## Part 4: 遗憾分析
- Regret 的定义
- 各算法的 Regret Bound
- 实践中的选择

## Part 5: 业务应用
- 推荐系统中的应用
- 广告投放
- 动态定价

## 练习
- 实现 Thompson Sampling
- 实现 LinUCB
- 对比实验

## 思考题
1. MAB 和 A/B Testing 如何选择？
2. Thompson Sampling 为什么有效？
3. Contextual Bandit 的「Context」如何选择？
```

---

### Part 3.1: 双重差分 (difference_in_differences)

**学习目标**：
1. 理解 DID 的核心思想和假设
2. 掌握平行趋势检验
3. 学习交错 DID 和 Event Study

**内容大纲**：

```markdown
# 双重差分 (DID) - 政策评估的利器

## Part 1: DID 的直觉
- 「差分的差分」是什么意思
- 为什么需要两次差分
- DID 与随机实验的关系

## Part 2: 基本 DID 设计
- 两期两组 DID
- 回归框架
- 标准误的选择 (聚类标准误)

## Part 3: 平行趋势假设
- 假设的含义
- 图形化检验
- 统计检验
- 安慰剂检验

## Part 4: 多期 DID
- 从两期到多期
- Event Study 设计
- 动态效应估计

## Part 5: 交错 DID (Staggered DID)
- 异质性处理时间问题
- TWFE 估计量的问题
- Callaway-Sant'Anna 估计量
- Sun-Abraham 估计量

## Part 6: 业务案例
- 案例 1: 最低工资政策效果
- 案例 2: 平台政策变更评估
- 案例 3: 补贴政策效果

## 练习
- 实现基本 DID
- 平行趋势检验
- Event Study 图

## 思考题
1. 如果平行趋势检验不通过怎么办？
2. DID 与 PSM 可以结合吗？
3. 交错 DID 的「负权重」问题是什么？
```

---

### Part 3.2: 合成控制 (synthetic_control)

**学习目标**：
1. 理解合成控制的思想
2. 掌握权重估计方法
3. 学习推断方法

**内容大纲**：

```markdown
# 合成控制法 - 构建「虚拟」对照组

## Part 1: 为什么需要合成控制
- 当只有一个处理单位时
- DID 的局限性
- 合成控制的思想

## Part 2: 方法原理
- 合成对照的构建
- 权重的约束
- 优化问题

## Part 3: 估计方法
- 协变量匹配
- 预测因子的选择
- 迭代算法

## Part 4: 推断方法
- Placebo Tests (安慰剂检验)
- 排列检验
- 置信区间

## Part 5: 与 DID 的关系
- 合成控制是 DID 的推广
- 什么时候用哪个

## Part 6: 业务案例
- 案例：新城市上线效果评估
- 案例：大客户流失影响

## 练习
- 实现基本合成控制
- Placebo Tests
- 可视化

## 思考题
1. 合成控制的权重为什么要非负？
2. 如果找不到好的「供体」怎么办？
3. 合成控制和 PSM 有什么关系？
```

---

### Part 3.3: 断点回归 (regression_discontinuity)

**学习目标**：
1. 理解 RDD 的核心思想
2. 掌握 Sharp RDD 和 Fuzzy RDD
3. 学习带宽选择和检验方法

**内容大纲**：

```markdown
# 断点回归 (RDD) - 门槛处的自然实验

## Part 1: RDD 的直觉
- 「门槛」创造的局部随机化
- 为什么断点处可比
- RDD 与随机实验的关系

## Part 2: Sharp RDD
- 设计与识别
- 局部多项式回归
- 估计与推断

## Part 3: Fuzzy RDD
- 从 Sharp 到 Fuzzy
- 与 IV 的关系
- LATE 解释

## Part 4: 带宽选择
- 偏差-方差权衡
- MSE-optimal 带宽
- 稳健推断

## Part 5: 检验方法
- McCrary 密度检验 (操纵检验)
- 协变量连续性检验
- Placebo 检验

## Part 6: 业务案例
- 案例 1: 优惠券门槛效应 (满减)
- 案例 2: 会员等级效应
- 案例 3: 信用评分门槛

## 练习
- 实现 Sharp RDD
- 带宽选择
- 各种检验

## 思考题
1. RDD 估计的是什么效应？(LATE)
2. 如果有人「操纵」自己的分数怎么办？
3. RDD 和 DID 可以结合吗？
```

---

### Part 3.4: 工具变量 (instrumental_variables)

**学习目标**：
1. 理解 IV 的三个假设
2. 掌握 2SLS 估计
3. 学习弱工具变量检验

**内容大纲**：

```markdown
# 工具变量 (IV) - 处理内生性问题

## Part 1: 什么是内生性
- 内生性的来源
- OLS 的偏差
- IV 的思想

## Part 2: IV 的三个假设
- 相关性假设 (Relevance)
- 排他性假设 (Exclusion)
- 外生性假设 (Exogeneity)

## Part 3: 2SLS 估计
- 两阶段最小二乘
- 手动实现
- 解释与推断

## Part 4: 弱工具变量
- 弱 IV 的问题
- F 统计量检验
- 弱 IV 稳健推断

## Part 5: 过度识别检验
- Hansen J 检验
- 多个 IV 的情况

## Part 6: LATE 解释
- Compliers, Always-takers, Never-takers
- LATE vs ATE
- 外部效度

## Part 7: 业务案例
- 案例 1: 价格弹性估计 (成本作为 IV)
- 案例 2: 广告效果 (天气作为 IV)
- 案例 3: 教育回报 (距离作为 IV)

## 练习
- 实现 2SLS
- 弱 IV 检验
- LATE 解释

## 思考题
1. 如何判断一个变量是否是有效的 IV？
2. 弱 IV 会带来什么问题？
3. LATE 和 ATE 什么时候相等？
```

---

### Part 4.2: Meta-Learners 扩展 (meta_learners)

**改进说明**：在现有基础上补充 R-Learner 和 DR-Learner

**新增内容**：

```markdown
## Part 5: R-Learner (新增)
- 设计动机：直接优化 CATE
- 损失函数推导
- 与 DML 的关系
- 实现与对比

## Part 6: DR-Learner (新增)
- 设计动机：结合 DR 的优势
- 伪结果构造
- 实现与对比

## Part 7: Meta-Learner 选择指南 (新增)
- 什么时候用 S-Learner
- 什么时候用 T-Learner
- 什么时候用 X-Learner
- 什么时候用 R-Learner
- 什么时候用 DR-Learner
```

---

### Part 6.1: 营销归因 (marketing_attribution)

**学习目标**：
1. 理解不同归因模型的原理
2. 掌握 Shapley 归因的计算
3. 区分增量归因与触达归因

**内容大纲**：

```markdown
# 营销归因 - 谁「真正」带来了转化？

## Part 1: 归因问题的本质
- 多触点转化路径
- 归因的目的
- 归因 vs 因果推断

## Part 2: 规则归因
- Last-touch Attribution
- First-touch Attribution
- Linear Attribution
- Time-decay Attribution
- Position-based Attribution
- 各方法的优缺点

## Part 3: Shapley 归因
- 合作博弈论基础
- Shapley Value 的定义
- 归因的公理性
- 计算方法 (精确 vs 近似)

## Part 4: Markov Chain 归因
- 转移概率矩阵
- 移除效应
- 计算与实现

## Part 5: 增量归因 vs 触达归因
- 触达归因：谁「参与了」转化
- 增量归因：谁「导致了」转化
- 本质区别与业务含义
- 如何做增量归因 (因果推断方法)

## Part 6: 业务实践
- 案例：多渠道营销归因
- 预算重分配决策
- 常见陷阱

## 练习
- 实现各种规则归因
- 实现 Shapley 归因
- 实现 Markov Chain 归因

## 思考题
1. Last-touch 归因有什么问题？
2. Shapley 归因为什么「公平」？
3. 增量归因需要什么数据？
```

---

### Part 6.4: 预算分配优化 (budget_allocation)

**学习目标**：
1. 理解边际 ROI 优化
2. 掌握约束优化建模
3. 学习多渠道预算分配

**内容大纲**：

```markdown
# 预算分配优化 - 把钱花在刀刃上

## Part 1: 预算分配问题
- 为什么需要优化
- 边际收益递减
- 多目标权衡

## Part 2: 边际 ROI 优化
- 边际 ROI 的定义
- 响应曲线建模
- 最优分配条件

## Part 3: 约束优化建模
- 目标函数设计
- 约束条件
- 拉格朗日方法
- 数值优化

## Part 4: 多渠道分配
- 渠道间的替代与互补
- 联合优化
- 实际操作流程

## Part 5: 不确定性与稳健优化
- 参数估计的不确定性
- 稳健优化方法
- 敏感性分析

## Part 6: 业务案例
- 案例：广告渠道预算分配
- 案例：优惠券类型分配

## 练习
- 响应曲线建模
- 单渠道优化
- 多渠道联合优化

## 思考题
1. 为什么不能只看 ROI 最高的渠道？
2. 如何处理新渠道（没有历史数据）？
3. 实时优化 vs 离线优化的权衡？
```

---

### Part 7.1: 因果发现 (causal_discovery)

**学习目标**：
1. 理解因果发现的问题
2. 掌握 PC 算法原理
3. 了解因果图学习

**内容大纲**：

```markdown
# 因果发现 - 从数据中学习因果结构

## Part 1: 因果发现的问题
- 从相关到因果
- 与因果推断的区别
- 可识别性问题

## Part 2: 基于约束的方法
- PC 算法
- 条件独立性检验
- FCI 算法 (处理隐变量)

## Part 3: 基于得分的方法
- 贝叶斯评分
- BIC/MDL 准则
- 搜索算法

## Part 4: 函数因果模型
- LiNGAM (非高斯)
- ANM (非线性)

## Part 5: 实践注意事项
- 样本量要求
- 假设检验
- 结果解释

## 练习
- 实现 PC 算法
- 因果图可视化
- 案例分析

## 思考题
1. 为什么因果发现很难？
2. PC 算法的假设是什么？
3. 发现的因果图如何验证？
```

---

## 三、Notebook 文件重命名方案

### 3.1 重命名命令

```bash
# Part 0: Foundation
mv chapter1_ex1_potential_outcomes.ipynb part0_1_potential_outcomes.ipynb
mv chapter1_ex2_causal_dag.ipynb part0_2_causal_dag.ipynb
mv chapter1_ex3_confounding_bias.ipynb part0_4_bias_types.ipynb

# Part 1: Experimentation
mv chapter6_ex2_ab_enhancement.ipynb part1_2_cuped_variance_reduction.ipynb

# Part 2: Observational Methods
mv chapter2_ex1_propensity_score.ipynb part2_1_propensity_score.ipynb
mv chapter2_ex2_ipw.ipynb part2_3_ipw_weighting.ipynb
mv chapter2_ex3_doubly_robust.ipynb part2_4_doubly_robust.ipynb
mv chapter5_ex3_sensitivity_analysis.ipynb part2_5_sensitivity_analysis.ipynb

# Part 4: CATE / Uplift
mv chapter5_ex1_cate_basics.ipynb part4_1_cate_basics.ipynb
mv chapter3_ex1_meta_learners.ipynb part4_2_meta_learners.ipynb
mv chapter5_ex2_causal_forest.ipynb part4_3_causal_forest.ipynb
mv chapter3_ex2_uplift_tree.ipynb part4_4_uplift_tree.ipynb
mv chapter3_ex3_uplift_evaluation.ipynb part4_5_uplift_evaluation.ipynb

# Part 5: Deep Learning
mv chapter4_ex1_representation_learning.ipynb part5_1_representation_learning.ipynb
mv chapter4_ex2_tarnet.ipynb part5_2_tarnet_dragonnet.ipynb
# chapter4_ex3_dragonnet.ipynb 合并到上面

# Part 6: Marketing Applications
mv chapter6_ex1_coupon_optimization.ipynb part6_2_coupon_optimization.ipynb
mv chapter6_ex3_user_targeting.ipynb part6_3_user_targeting.ipynb
```

### 3.2 最终文件结构

```
notebooks/
├── part0_foundation/
│   ├── part0_1_potential_outcomes.ipynb
│   ├── part0_2_causal_dag.ipynb
│   ├── part0_3_identification_strategies.ipynb  [新增]
│   └── part0_4_bias_types.ipynb
│
├── part1_experimentation/
│   ├── part1_1_ab_testing_basics.ipynb  [新增]
│   ├── part1_2_cuped_variance_reduction.ipynb
│   ├── part1_3_stratified_analysis.ipynb  [新增]
│   ├── part1_4_network_effects.ipynb  [新增]
│   ├── part1_5_switchback_experiments.ipynb  [新增]
│   ├── part1_6_long_term_effects.ipynb  [新增]
│   └── part1_7_multi_armed_bandits.ipynb  [新增]
│
├── part2_observational/
│   ├── part2_1_propensity_score.ipynb
│   ├── part2_2_matching_methods.ipynb  [新增]
│   ├── part2_3_ipw_weighting.ipynb
│   ├── part2_4_doubly_robust.ipynb
│   └── part2_5_sensitivity_analysis.ipynb
│
├── part3_quasi_experiments/
│   ├── part3_1_difference_in_differences.ipynb  [新增]
│   ├── part3_2_synthetic_control.ipynb  [新增]
│   ├── part3_3_regression_discontinuity.ipynb  [新增]
│   └── part3_4_instrumental_variables.ipynb  [新增]
│
├── part4_cate_uplift/
│   ├── part4_1_cate_basics.ipynb
│   ├── part4_2_meta_learners.ipynb  [扩展]
│   ├── part4_3_causal_forest.ipynb
│   ├── part4_4_uplift_tree.ipynb
│   └── part4_5_uplift_evaluation.ipynb
│
├── part5_deep_learning/
│   ├── part5_1_representation_learning.ipynb  [扩展]
│   ├── part5_2_tarnet_dragonnet.ipynb  [合并]
│   └── part5_3_cevae_advanced.ipynb  [新增]
│
├── part6_marketing/
│   ├── part6_1_marketing_attribution.ipynb  [新增]
│   ├── part6_2_coupon_optimization.ipynb
│   ├── part6_3_user_targeting.ipynb
│   └── part6_4_budget_allocation.ipynb  [新增]
│
└── part7_advanced/
    ├── part7_1_causal_discovery.ipynb  [新增]
    ├── part7_2_continuous_treatment.ipynb  [新增]
    ├── part7_3_time_varying_treatment.ipynb  [新增]
    └── part7_4_mediation_analysis.ipynb  [新增]
```

---

## 四、实施计划

### Phase 1: 重组现有内容 (1-2 天)
- [ ] 创建新的目录结构
- [ ] 重命名现有 Notebook
- [ ] 更新内部引用

### Phase 2: P0 新增内容 (1-2 周)
- [ ] part0_3_identification_strategies.ipynb
- [ ] part1_1_ab_testing_basics.ipynb
- [ ] part3_1_difference_in_differences.ipynb
- [ ] part3_4_instrumental_variables.ipynb
- [ ] part6_1_marketing_attribution.ipynb

### Phase 3: P1 新增内容 (2-3 周)
- [ ] part1_4_network_effects.ipynb
- [ ] part1_6_long_term_effects.ipynb
- [ ] part1_7_multi_armed_bandits.ipynb
- [ ] part3_2_synthetic_control.ipynb
- [ ] part3_3_regression_discontinuity.ipynb
- [ ] part4_2_meta_learners.ipynb (扩展 R/DR-Learner)
- [ ] part6_4_budget_allocation.ipynb

### Phase 4: P2 新增内容 (2-3 周)
- [ ] part1_3_stratified_analysis.ipynb
- [ ] part1_5_switchback_experiments.ipynb
- [ ] part2_2_matching_methods.ipynb
- [ ] part5_3_cevae_advanced.ipynb
- [ ] part7_1_causal_discovery.ipynb
- [ ] part7_2_continuous_treatment.ipynb
- [ ] part7_3_time_varying_treatment.ipynb
- [ ] part7_4_mediation_analysis.ipynb

---

## 五、Notebook 质量标准

每个 Notebook 应包含：

### 5.1 结构标准
- [ ] 学习目标 (3-5 个明确目标)
- [ ] 直觉解释 (生活化类比)
- [ ] 数学推导 (核心公式)
- [ ] 代码实现 (TODO 练习)
- [ ] 可视化 (关键概念)
- [ ] 思考题 (5 个以上)
- [ ] 总结表格

### 5.2 内容标准
- [ ] 理论深度适中 (面试够用)
- [ ] 有业务场景案例
- [ ] TODO 练习可独立完成
- [ ] 思考题有区分度

### 5.3 代码标准
- [ ] 函数有完整 docstring
- [ ] 有类型注解
- [ ] 有输入验证
- [ ] 可视化美观

---

*本文档为 Notebook 改进的详细方案，配合 `KNOWLEDGE_SYSTEM_RESTRUCTURE.md` 使用*
