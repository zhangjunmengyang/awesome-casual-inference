# Pitfalls 面试速查手册 ⚡

这是 5 个 pitfall notebooks 的精华提炼，面试前 30 分钟快速复习用。

---

## Pitfall 01: PSM Failure Modes

### 核心知识点

**4 大失败模式**:
1. ❌ 未检查 Balance → SMD > 0.1
2. ❌ 共同支撑违背 → 倾向得分分布不重叠
3. ❌ 样本丢失过多 → 匹配率 < 70%
4. ❌ 隐变量遗漏 → 未观测混淆

### 面试必答题

**Q: PSM 的核心假设是什么？**

A: **Unconfoundedness** (条件独立性)
- 给定观测协变量 X，处理分配与潜在结果独立
- 数学表达: (Y₀, Y₁) ⊥ T | X
- 白话: 控制住 X 后，处理就像是随机分配的

**Q: 如何检查 Balance？**

A: 3 步法:
1. **SMD** (Standardized Mean Difference): |SMD| < 0.1
2. **Love Plot**: 可视化匹配前后的 SMD
3. **分布图**: 比较两组的协变量分布

**Q: Balance 检查通过就够了吗？**

A: **不够**！Balance 只是必要条件，不是充分条件。
- 还需要：共同支撑检查、敏感性分析
- 无法检验未观测混淆（Unobserved confounding）
- 建议：Rosenbaum bounds, E-value

### 快速诊断清单

```python
# PSM 质量检查（5 秒判断）
□ Max SMD < 0.1?
□ 匹配率 > 80%?
□ 倾向得分重叠?
□ 第一阶段 AUC = 0.5-0.8?（不要太高也不要太低）
□ 做了敏感性分析?
```

### 30 秒电梯演讲

"PSM 通过匹配倾向得分构造反事实，核心是模拟随机化。但它只能控制观测到的混淆，关键是检查 Balance（SMD < 0.1）、共同支撑、样本丢失。实际使用时要做敏感性分析，评估隐藏偏差的影响。"

---

## Pitfall 02: CUPED Misuse

### 核心知识点

**4 大失败模式**:
1. ❌ 相关性太低 → ρ < 0.3，方差缩减 < 10%
2. ❌ 新用户无历史数据 → 缺失值处理不当
3. ❌ 协变量受处理影响 → 用了实验期间的数据
4. ❌ 样本量过小 → n < 200/组，θ 估计不稳定

### 面试必答题

**Q: CUPED 的原理是什么？**

A: 利用实验前协变量减少方差
- 公式: Y_adj = Y - θ(X - X̄)
- θ = Cov(Y,X) / Var(X)
- 方差缩减: Var(Y_adj) = Var(Y) × (1 - ρ²)
- 关键: ρ 是 Y 和 X 的相关系数

**Q: CUPED 的前提条件？**

A: 3 个关键条件:
1. **协变量是实验前的** (Pre-experiment)
2. **与结果相关** (ρ > 0.3)
3. **不受处理影响** (Unaffected by treatment)

**Q: 新用户怎么办？**

A: **分层 CUPED**
- 老用户: 用 CUPED
- 新用户: 用原始值
- 合并两层结果

### 快速诊断清单

```python
# CUPED 前置检查（5 秒判断）
□ 相关性 ρ > 0.3?
□ 样本量 > 200/组?
□ 缺失率 < 30%?
□ 协变量在实验前?
□ 两组协变量平衡? (p > 0.05)
```

### 30 秒电梯演讲

"CUPED 通过减去可预测的部分来降低噪声，核心是 Y_adj = Y - θ(X - X̄)。关键是协变量必须是实验前的、相关性要高（ρ > 0.3）。新用户用分层 CUPED，样本量至少 200/组。可以减少 30-50% 方差，缩短实验周期。"

---

## Pitfall 03: DID Violations

### 核心知识点

**3 大违背情形**:
1. ❌ 平行趋势不满足 → 趋势差异
2. ❌ Anticipation Effect → 提前响应
3. ❌ Spillover → SUTVA 违背

### 面试必答题

**Q: 平行趋势假设是什么？**

A: **Parallel Trends Assumption**
- 数学: E[Y₁ₜ(0) - Y₁ₛ(0)] = E[Y₀ₜ(0) - Y₀ₛ(0)]
- 白话: 在没有处理的情况下，两组的结果会以相同的趋势变化
- 关键: 这是反事实假设，无法直接检验

**Q: 如何检验平行趋势？**

A: 3 种方法:
1. **可视化**: 画处理前两组的时间趋势
2. **Event Study**: 估计每期的"处理效应"，处理前应接近 0
3. **统计检验**: 处理前交互项 (treated × time) 的系数检验

**Q: 平行趋势不满足怎么办？**

A: 4 种解决方案:
1. **组特定趋势 DID** (Group-specific trends)
2. **Synthetic Control** (合成控制)
3. **Change-in-Changes** (非参数方法)
4. **换对照组** 或 **换方法** (RDD, IV)

### 快速诊断清单

```python
# DID 诊断清单（5 秒判断）
□ 处理前至少 3 期?
□ Event study 处理前系数接近 0?
□ 平行趋势检验 p > 0.05?
□ 无 Anticipation Effect?
□ Placebo test 不显著?
```

### 30 秒电梯演讲

"DID 通过双重差分消除组间固定差异和时间趋势，核心假设是平行趋势。用 Event Study 检验：处理前各期系数应接近 0。如果有 Anticipation，调整处理时点；如果平行趋势不满足，用组特定趋势或 Synthetic Control。"

---

## Pitfall 04: Weak IV

### 核心知识点

**弱工具变量的危害**:
1. ❌ 估计偏差大 → 向 OLS 偏移
2. ❌ 方差爆炸 → 置信区间巨宽
3. ❌ 假阳性率高 → 即使大样本也有偏

### 面试必答题

**Q: 什么是弱工具变量？**

A: **第一阶段 F 统计量 < 10**
- F 统计量衡量工具变量对处理的预测力
- Stock-Yogo 临界值:
  * F > 16.38 → 偏差 < 10%
  * F > 8.96 → 偏差 < 15%
  * F < 10 → 弱工具变量

**Q: 弱工具变量怎么办？**

A: 3 种方法:
1. **Anderson-Rubin CI** → 在弱 IV 下有正确覆盖率
2. **LIML** → 比 2SLS 偏差更小
3. **找更强的工具变量** → 根本解决方案

**Q: IV 估计的是什么效应？**

A: **LATE** (Local Average Treatment Effect)
- 不是 ATE！
- 只对 "Compliers" 有效（被工具变量说服的人）
- 外推性有限

### 快速诊断清单

```python
# IV 强度检查（5 秒判断）
□ 第一阶段 F > 10?
□ t 统计量 > 3.16?
□ R² > 0.1?
□ 如有多 IV，Sargan test p > 0.05?
□ 排斥性假设合理?（领域知识）
```

### 30 秒电梯演讲

"IV 用于处理内生性，需要满足相关性和排斥性。弱工具变量（F < 10）会导致估计偏差大、方差爆炸。检查第一阶段 F 统计量，如果弱就用 Anderson-Rubin CI 或 LIML。IV 估计的是 LATE，不是 ATE，解释时要说明适用人群。"

---

## Pitfall 05: A/B Test Common Mistakes

### 核心知识点

**4 大天坑**:
1. ❌ SRM (Sample Ratio Mismatch) → 分流异常
2. ❌ Peeking Problem → 反复查看显著性
3. ❌ Multiple Testing → 多重比较问题
4. ❌ Network Effects → SUTVA 违背

### 面试必答题

**Q: 什么是 SRM？如何检测？**

A: **Sample Ratio Mismatch** (样本比例失配)
- 检测: 卡方检验，p < 0.001 则有 SRM
- 常见原因: 分流 bug、数据丢失、bot 过滤
- 后果: 分流有偏，实验结论无效
- **红线**: 有 SRM 绝对不能用实验结果！

**Q: Peeking Problem 是什么？**

A: **反复检查显著性导致假阳性率膨胀**
- 例: α=0.05，检查 14 次 → 实际假阳性率 ≈ 51%
- 解决: Sequential Testing (序贯检验)
  * Alpha Spending Function
  * Group Sequential Design
  * Always Valid Inference

**Q: 多重检验如何校正？**

A: 2 种方法:
1. **Bonferroni**: α/m (控制 FWER)
   - 保守，适合关键决策
2. **Benjamini-Hochberg**: 控制 FDR
   - 更有 power，适合探索性分析

**Q: Network Effects 如何处理？**

A: 3 种设计:
1. **Cluster Randomization**: 按集群（城市、社区）随机化
2. **Switchback Experiments**: 时间维度切换
3. **Ego-network Randomization**: 按社交网络分组

### 快速诊断清单

```python
# A/B 测试质量检查（5 秒判断）
□ SRM 检验 p > 0.001?
□ 没有提前 peeking?
□ 多指标用了校正?
□ 样本量达到了吗?
□ AA 测试通过?
```

### 30 秒电梯演讲

"A/B 测试四大天坑：SRM（分流异常，卡方检验 p > 0.001）、Peeking（反复看显著性，用 Alpha Spending）、Multiple Testing（多指标校正，Bonferroni 或 BH）、Network Effects（用 cluster randomization）。务必先做 SRM 检测和 AA 测试。"

---

## 综合面试策略

### 面试官问："你做过因果推断吗？"

**推荐回答结构** (STAR 法):

1. **Situation** (背景):
   "在 XX 公司，我们需要评估 XX 功能对 YY 指标的影响"

2. **Task** (任务):
   "由于无法做 A/B 测试（或已有数据），我选择了 PSM/DID/IV 方法"

3. **Action** (行动):
   - 说明具体方法和诊断流程
   - 强调遇到的问题和解决方案
   - 展示对 pitfalls 的认识

4. **Result** (结果):
   - 定量结果 + 置信区间
   - 敏感性分析
   - 业务影响

### 万能回答模板

当被问到任何因果方法的局限性:

```
这个方法的核心假设是 [XXX]。主要局限性有 3 个:

1. **假设可能违背**: [具体说明]
   - 诊断方法: [XXX]
   - 解决方案: [XXX]

2. **外推性**: [说明适用范围]

3. **与 RCT 的差距**: [说明还有哪些东西做不到]

实际使用中，我会:
- 做诊断检查
- 敏感性分析
- 与其他方法比较
- 诚实汇报局限性
```

### 面试加分技巧

1. **主动提 Pitfalls**:
   "用 PSM 时，我特别注意了 Balance 检查和共同支撑问题..."

2. **展示 Trade-offs 意识**:
   "Bonferroni 更保守但 power 低，BH 有更好的 power 但可能有假阳性..."

3. **结合实际业务**:
   "考虑到我们的业务场景（双边市场/社交网络），SUTVA 可能违背，所以我用了..."

4. **诚实谦逊**:
   "这个方法无法完全解决 XX 问题，所以我做了 YY 敏感性分析，给出了不同假设下的估计范围"

---

## 速查表 - 方法选择决策树

```
能做 RCT?
├─ 是 → A/B 测试（注意 SRM, Peeking, Multiple Testing）
└─ 否 → 有观测数据
    ├─ 有历史对照? → DID
    │   ├─ 平行趋势 OK? → 标准 DID
    │   └─ 平行趋势不 OK? → 组特定趋势 / Synthetic Control
    │
    ├─ 有工具变量? → IV
    │   ├─ F > 10? → 2SLS
    │   └─ F < 10? → Anderson-Rubin / LIML
    │
    └─ 只有横截面? → PSM / IPW
        ├─ Balance OK? → PSM
        ├─ Balance 不 OK 但 overlap OK? → IPW
        └─ 都不 OK? → 改用其他方法或诚实说做不了
```

---

## 最后的忠告

### 面试时要说的话:

✅ "我检查了 Balance/平行趋势/工具变量强度"
✅ "我做了敏感性分析"
✅ "我了解这个方法的局限性，所以..."
✅ "我比较了多种方法，选择 XX 是因为..."

### 面试时千万别说:

❌ "我直接用了 XX 方法"（没有诊断）
❌ "结果是显著的"（没有置信区间）
❌ "这就是因果效应"（没有说明假设）
❌ "这个方法完美解决了问题"（没有局限性）

---

**Good Luck! 🍀**

记住：面试官喜欢的不是"全知全能"，而是"知道自己在做什么，知道局限性在哪里"的候选人。
