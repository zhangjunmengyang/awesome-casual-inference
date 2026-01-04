# 因果推断知识体系评审报告

> 评审时间: 2026-01-04
> 评审目标: 评估作为面试展示、知识体系、对外教程的项目质量

---

## 一、总体评价

### 1.1 优势亮点

| 维度 | 评分 | 说明 |
|------|------|------|
| **体系完整性** | ⭐⭐⭐⭐⭐ | 从基础理论到工程实践，覆盖 7 大模块 35+ notebooks |
| **面试导向** | ⭐⭐⭐⭐⭐ | 准实验方法(DID/RDD/IV)、CUPED、Uplift 等高频考点完整覆盖 |
| **教学质量** | ⭐⭐⭐⭐☆ | 业务场景引入、直觉解释、代码实现、TODO 练习，结构完整 |
| **知识深度** | ⭐⭐⭐⭐☆ | 有 Deep Dive 系列，但仍有提升空间 |
| **工程实践** | ⭐⭐⭐☆☆ | 有 Case Study，但缺乏端到端项目 |

### 1.2 核心竞争力

1. **问题驱动的知识组织**
   - 从「我有什么场景」出发，而非方法论堆砌
   - `part0_3_identification_strategies.ipynb` 的决策树设计非常实用

2. **准实验方法的深度覆盖**
   - DID 覆盖了交错设计、平行趋势检验、Event Study
   - 这是国内因果推断教程普遍缺失的部分

3. **营销归因的本质区分**
   - 明确区分「触达归因」vs「增量归因」
   - 这是营销算法岗面试的核心考点

4. **面试友好的结构**
   - 学习目标明确
   - 思考题设计合理
   - 总结表格便于复习

---

## 二、知识体系分析

### 2.1 知识覆盖度评估

```
┌─────────────────────────────────────────────────────────────────┐
│                     因果推断知识体系                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Part 0: Foundation] ████████████ 100% ✅                     │
│    潜在结果、DAG、识别策略、偏差类型                               │
│                                                                 │
│  [Part 1: A/B Testing] ██████████ 100% ✅                      │
│    CUPED、网络效应、Switchback、MAB                              │
│                                                                 │
│  [Part 2: Observational] ████████████ 100% ✅                  │
│    PSM、IPW、DR、DML、敏感性分析                                 │
│                                                                 │
│  [Part 3: Quasi-Experiments] ████████████ 100% ✅              │
│    DID、合成控制、RDD、IV                                        │
│                                                                 │
│  [Part 4: CATE/Uplift] ████████████ 100% ✅                    │
│    Meta-Learners、Causal Forest、Uplift Tree                    │
│                                                                 │
│  [Part 5: Deep Learning] ████████████ 100% ✅                  │
│    TARNet、DragonNet、CEVAE、GANITE、VCNet                      │
│                                                                 │
│  [Part 6: Marketing] ████████████ 100% ✅                      │
│    归因、发券优化、用户定向、预算分配                              │
│                                                                 │
│  [Part 7: Advanced] ████████████ 100% ✅                       │
│    因果发现、连续处理、时变处理、中介分析                          │
│    多重处理、Bunching 估计、合规者分析 (CACE/LATE)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 面试考点覆盖分析

| 考点类别 | 覆盖率 | 状态 | 建议 |
|----------|--------|------|------|
| **基础概念** (ATE/ATT/CATE, DAG) | 100% | ✅ | - |
| **实验设计** (CUPED, 网络效应) | 100% | ✅ | - |
| **准实验方法** (DID, RDD, IV) | 100% | ✅ | 补充 DID 的 Bacon 分解 |
| **观测数据方法** (PSM, IPW, DR) | 100% | ✅ | - |
| **异质效应** (Meta-Learners, CF) | 100% | ✅ | - |
| **营销归因** (Shapley, Markov) | 100% | ✅ | - |
| **工程实践** (脏数据, Debug) | 60% | ⚠️ | 需加强 |
| **系统设计** (实验平台架构) | 20% | ❌ | 建议新增 |

---

## 三、Notebook 质量评估

### 3.1 质量评分细则

以 `part3_1_difference_in_differences.ipynb` 为例（评分 A）：

| 评估维度 | 满分 | 得分 | 说明 |
|----------|------|------|------|
| 学习目标 | 10 | 10 | 5 个明确目标 |
| 业务场景引入 | 15 | 15 | 电商会员政策案例生动 |
| 直觉解释 | 15 | 14 | 减肥药比喻很好，可加更多 |
| 数学推导 | 15 | 15 | 公式完整，符号清晰 |
| 代码实现 | 20 | 18 | 有 TODO 练习，可加更多边界情况 |
| 可视化 | 15 | 15 | Plotly 交互图表丰富 |
| 思考题 | 10 | 10 | 5 个有深度的问题 |
| **总分** | **100** | **97** | **A** |

以 `part6_1_marketing_attribution.ipynb` 为例（评分 A）：

| 评估维度 | 满分 | 得分 | 说明 |
|----------|------|------|------|
| 学习目标 | 10 | 10 | 6 个明确目标 |
| 业务场景引入 | 15 | 15 | 营销总监视角，非常实战 |
| 直觉解释 | 15 | 15 | 三人开店 Shapley 例子绝佳 |
| 数学推导 | 15 | 14 | Shapley 公式完整，Markov 可更详细 |
| 代码实现 | 20 | 19 | 实现完整，有 TODO |
| 可视化 | 15 | 14 | 图表丰富，可加桑基图 |
| 思考题 | 10 | 10 | 问题有深度 |
| **总分** | **100** | **97** | **A** |

### 3.2 各 Part 质量总评

| Part | 质量 | 说明 |
|------|------|------|
| Part 0 | A | 识别策略决策树是亮点 |
| Part 1 | A- | CUPED 推导可更详细 |
| Part 2 | A | DML 深潜很好 |
| Part 3 | A+ | DID 深度和广度都很好 |
| Part 4 | A | Meta-Learners 对比表格实用 |
| Part 5 | A | TARNet 到 VCNet 完整覆盖，含 GAN 方法 |
| Part 6 | A | 增量 vs 触达的区分是核心亮点 |
| Part 7 | A | 多重处理、Bunching、LATE 补充完整 |

---

## 四、改进建议

### 4.1 高优先级 (P0) - 面试致命短板

#### 4.1.1 补充「实验平台系统设计」

面试中常被问到：「如何设计一个 A/B 测试平台？」

**建议新增**：`notebooks/part1_experimentation/part1_8_experiment_platform_design.ipynb`

内容大纲：
```markdown
1. 实验平台架构设计
   - 分流层设计（用户 ID hash、流量分层）
   - 指标计算层（实时 vs 离线）
   - 结果分析层（统计引擎）

2. 关键技术挑战
   - 如何保证随机性？
   - 如何处理新用户？
   - 如何支持多实验并行？

3. 工业实践
   - Google Vizier
   - Netflix XP 平台
   - 字节跳动 DataTester
```

#### 4.1.2 补充「Pitfall 系列」

目前只有 1 个 Pitfall notebook，面试中经常被问「踩过什么坑」。

**建议扩展**：
```
pitfalls/
├── pitfall_01_psm_failure_modes.ipynb     ✅ 已有
├── pitfall_02_did_parallel_trends.ipynb   ❌ 新增
├── pitfall_03_iv_weak_instrument.ipynb    ❌ 新增
├── pitfall_04_uplift_negative_effect.ipynb ❌ 新增
└── pitfall_05_cuped_covariate_choice.ipynb ❌ 新增
```

每个 Pitfall 结构：
1. **错误现象**：代码跑出来了，但结果有问题
2. **诊断方法**：如何发现问题
3. **根因分析**：为什么会出错
4. **修复方案**：正确的做法
5. **预防措施**：如何避免

#### 4.1.3 补充「端到端项目」

当前 Case Study 偏理想化，面试官会追问「脏数据怎么处理」。

**建议新增**：`notebooks/projects/` 目录

```
projects/
├── project_01_coupon_uplift_modeling/
│   ├── 01_data_exploration.ipynb      # 脏数据探索
│   ├── 02_feature_engineering.ipynb   # 特征工程
│   ├── 03_model_training.ipynb        # Uplift 建模
│   ├── 04_offline_evaluation.ipynb    # 离线评估
│   └── 05_online_ab_testing.ipynb     # 线上实验设计
│
└── project_02_marketing_mix_modeling/
    ├── 01_data_preparation.ipynb
    ├── 02_bayesian_mmm.ipynb
    └── 03_budget_optimization.ipynb
```

### 4.2 中优先级 (P1) - 提升亮点深度

#### 4.2.1 Deep Dive 系列扩展

当前 Deep Dive 只有 2 个，建议扩展到 5-6 个：

| 主题 | 场景 | 从调包到改架构 |
|------|------|---------------|
| 不平衡处理 | 发券数据 5% 正样本 | BCE → Focal → Propensity-weighted |
| 多处理效应 | 5 种优惠券类型 | 修改 DragonNet 输出层 |
| **极端倾向得分** | PS 接近 0 或 1 | Trimming → Overlap Weights → Entropy Balancing |
| **协变量选择** | 上百个特征 | LASSO → Double Selection → DeepIV |
| **长尾分布** | 收入数据严重右偏 | Log transform → Quantile Regression → DML |

#### 4.2.2 补充 Interview Cheatsheet 的代码片段

`docs/INTERVIEW_CHEATSHEET.md` 目前只有文字，建议加「一分钟代码片段」：

```python
# 面试快速实现：CUPED
def cuped_estimator(Y_treatment, Y_control, X_treatment, X_control):
    """30 行实现 CUPED"""
    theta = np.cov(Y_control, X_control)[0,1] / np.var(X_control)
    Y_adj_t = Y_treatment - theta * (X_treatment - X_treatment.mean())
    Y_adj_c = Y_control - theta * (X_control - X_control.mean())
    return Y_adj_t.mean() - Y_adj_c.mean()
```

#### 4.2.3 补充「方法对比总结表」

在每个 Part 的最后，增加方法对比表：

**Part 3 准实验方法对比**：

| 方法 | 核心假设 | 识别的是 | 数据要求 | 常见应用 |
|------|---------|---------|---------|---------|
| DID | 平行趋势 | ATT | 面板数据 | 政策评估 |
| RDD | 局部随机化 | LATE | 有断点 | 优惠券门槛 |
| IV | 排他性约束 | LATE | 有好工具 | 供需分析 |
| 合成控制 | 权重可加 | ATT | 长时间序列 | 单案例 |

### 4.3 低优先级 (P2) - 锦上添花

#### 4.3.1 补充深度学习前沿方法

`Part 5` 已补充：
- ✅ GANITE (ICLR 2018)：生成对抗网络估计 ITE
- ✅ VCNet (ICLR 2021)：连续处理效应

未来可继续扩展：
- TransTEE (NeurIPS 2022)：Transformer 因果推断

#### 4.3.2 补充交互式 Demo

利用 frontend 的可视化能力，做几个交互式 Demo：
- DAG 的 d-分离判断器
- CUPED 方差缩减模拟器
- Uplift 四象限可视化

#### 4.3.3 补充论文阅读指南

在 `docs/LEARNING_RESOURCES.md` 中，为每篇核心论文加「3 分钟速读」：

```markdown
### TARNet/CFR (ICML 2017)

**核心贡献**: 提出平衡表示学习，用 IPM 距离约束
**关键公式**: L = L_factual + α * IPM(Φ(X|T=0), Φ(X|T=1))
**局限性**: 只能处理二元处理
**面试考点**: 为什么需要平衡表示？IPM 距离有哪些选择？
```

---

## 五、知识体系扩展建议

### 5.1 可纳入的新兴方向

| 方向 | 重要性 | 建议优先级 |
|------|--------|-----------|
| **因果强化学习** | ⭐⭐⭐ | P2 |
| **因果公平性** | ⭐⭐⭐ | P2 |
| **LLM + 因果** | ⭐⭐⭐⭐ | P1 |
| **图神经网络因果** | ⭐⭐ | P3 |

### 5.2 建议的学习路径图

```
                        ┌─────────────────────────────────────┐
                        │         Part 0: Foundation          │
                        │   潜在结果、DAG、识别策略、偏差       │
                        └─────────────────┬───────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
    ┌───────────────────────────┐             ┌───────────────────────────┐
    │   Part 1: Experimentation │             │  Part 2: Observational    │
    │   A/B Testing, CUPED      │             │  PSM, IPW, DR, DML        │
    └─────────────┬─────────────┘             └─────────────┬─────────────┘
                  │                                         │
                  │         ┌───────────────────────────────┘
                  │         │
                  ▼         ▼
    ┌───────────────────────────┐
    │  Part 3: Quasi-Experiments │
    │  DID, RDD, IV, SC          │
    └─────────────┬─────────────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │    Part 4: CATE/Uplift    │
    │  Meta-Learners, CF, Tree  │
    └─────────────┬─────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│Part 5: Deep   │   │Part 6: Market │
│TARNet, Dragon │   │归因、发券、定向│
└───────────────┘   └───────────────┘
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │    Part 7: Advanced       │
    │  因果发现、连续处理、中介   │
    └───────────────────────────┘
```

---

## 六、面试准备建议

### 6.1 知识体系 → 面试叙事

将知识体系转化为面试故事线：

**故事 1: 「从 A/B 到因果推断」**
> 在做 A/B 测试时，我发现很多场景无法随机化（如定价、政策）。
> 这驱动我系统学习因果推断，从 DID 开始，逐步掌握 RDD、IV。
> 我用 DID 分析了 XX 政策效果，发现 XX 效应...

**故事 2: 「从 Uplift 到营销决策」**
> 在做发券优化时，我意识到传统 CTR 模型的问题：高响应率 ≠ 高增量。
> 我引入 Uplift Modeling，用 X-Learner 识别「可说服人群」。
> 最终在同等成本下，提升了 XX% 的增量转化...

**故事 3: 「从触达归因到增量归因」**
> 在做营销归因时，我发现 Last-touch 严重高估 Retargeting 价值。
> 我设计了增量实验（Geo-experiment），发现真实增量只有 2%。
> 基于此调整预算分配，整体 ROI 提升了 XX%...

### 6.2 每个 Part 的「面试金句」

| Part | 金句 |
|------|------|
| Part 0 | 「因果推断的核心是反事实：如果没有处理，结果会怎样？」|
| Part 1 | 「CUPED 的本质是用历史数据预测 Y(0)，减小方差」|
| Part 2 | 「DR 估计的魅力在于：PS 或 OR 任一正确，估计就无偏」|
| Part 3 | 「DID 的假设是平行趋势，检验方法是 Event Study」|
| Part 4 | 「Uplift 不是预测响应率，而是预测处理效应」|
| Part 5 | 「深度因果的核心挑战是平衡表示：让处理组和对照组在特征空间中可比」|
| Part 6 | 「触达归因告诉你谁参与了，增量归因告诉你谁导致了」|

---

## 七、执行计划

### 7.1 短期 (2 周)

- [ ] 补充 Pitfall 系列 (2-3 个)
- [ ] 补充实验平台设计 notebook
- [ ] 在 INTERVIEW_CHEATSHEET 中加代码片段

### 7.2 中期 (1 个月)

- [ ] 完成 1 个端到端项目 (Coupon Uplift Modeling)
- [ ] 扩展 Deep Dive 到 5 个
- [ ] 补充方法对比总结表

### 7.3 长期 (3 个月)

- [ ] 补充深度学习前沿方法
- [ ] 做交互式 Demo
- [ ] 写论文阅读指南

---

## 八、结论

### 8.1 总体评价

这是一个**高质量、体系完整**的因果推断知识库，在以下方面表现出色：

1. **面试导向**：覆盖了数据科学/营销算法岗的核心考点
2. **教学质量**：业务场景引入、直觉解释、代码实现、思考题，结构完整
3. **知识深度**：准实验方法、营销归因的深度超过市面大部分教程

### 8.2 核心改进方向

1. **补短板**：Pitfall 系列、实验平台设计、端到端项目
2. **强亮点**：Deep Dive 扩展、面试叙事准备
3. **扩前沿**：深度学习新方法、因果 + LLM

### 8.3 作为「三位一体」项目的评价

| 定位 | 评分 | 说明 |
|------|------|------|
| **知识体系** | A | 结构完整，可持续扩展 |
| **面试展示** | A- | 补 Pitfall 和项目后可达 A+ |
| **对外教程** | A | 教学质量高，可作为开源教程 |

**最终评价**：这是一个值得长期维护的「因果推断工作台」，建议持续迭代，逐步打造成该领域的标杆项目。

---

## 附录：原有改进建议（保留参考）

### 1. 从调包到改架构的 Case Study

当前问题：虽然有代码，但主要是"调库 + 可视化"，缺乏从原理出发的改进。

建议新增「深水区」系列 Notebook：

```
notebooks/deep_dive/
├── deep_dive_01_custom_loss_for_imbalanced_treatment.ipynb
│   # 场景：发券数据正样本只有5%，如何修改 IPW loss？
│   # 内容：从 BCE → Focal Loss → 自定义 Propensity-weighted Loss
│
├── deep_dive_02_modify_dragonnet_for_multi_treatment.ipynb
│   # 场景：多种优惠券（5折、7折、满减），如何扩展 DragonNet？
│   # 内容：修改输出层 → 处理类别不平衡 → 多头输出设计
│
├── deep_dive_03_causal_forest_confidence_interval.ipynb
│   # 场景：业务问"有多大把握"，如何输出置信区间？
│   # 内容：Honest Splitting → Bootstrap CI → Variance Estimation
│
├── deep_dive_04_online_uplift_with_cold_start.ipynb
│   # 场景：新用户没有历史数据，怎么做 Uplift？
│   # 内容：Thompson Sampling + Prior → 逐步更新策略
│
└── deep_dive_05_handling_propensity_extremes.ipynb
    # 场景：倾向得分有些是 0.99，有些是 0.01
    # 内容：Trimming → Overlap Weights → CRUMP bounds
```

### 2. End-to-End 真实项目 Case

当前问题：有 case study 但偏理想化，缺乏脏数据、工程约束、迭代过程。

建议新增「实战项目」系列：

```
notebooks/real_world_projects/
├── project_01_voucher_optimization/
│   ├── 01_data_exploration.ipynb      # 真实数据 EDA
│   ├── 02_baseline_and_pitfalls.ipynb  # 踩坑记录
│   ├── 03_model_iteration.ipynb        # 模型迭代过程
│   ├── 04_online_deployment.ipynb      # 上线部署考量
│   └── 05_monitoring_and_debug.ipynb   # 线上监控与问题排查
│
├── project_02_ab_testing_platform/
│   ├── 01_experiment_design.ipynb
│   ├── 02_data_pipeline.ipynb
│   ├── 03_statistical_engine.ipynb
│   └── 04_dashboard_design.ipynb
│
└── project_03_marketing_attribution/
    ├── 01_data_challenges.ipynb        # 触点数据不全、归因窗口
    ├── 02_model_comparison.ipynb
    └── 03_budget_reallocation.ipynb
```

### 3. Debug & Pitfall 系列

面试常问："你遇到过什么问题？怎么解决的？"

建议新增：

```
notebooks/pitfalls/
├── pitfall_01_psm_failure_modes.ipynb
│   # 常见错误：未检查 balance、caliper 太松、样本丢失过多
│
├── pitfall_02_did_violations.ipynb
│   # 平行趋势不满足时怎么办？anticipation effect？
│
├── pitfall_03_iv_weak_instrument.ipynb
│   # 弱工具变量的诊断与处理
│
├── pitfall_04_uplift_negative_effects.ipynb
│   # 发券后转化反而下降？Sleeping Dogs 的处理
│
└── pitfall_05_ab_test_common_mistakes.ipynb
    # SRM、Peeking、Multiple Testing、Network Effects
```

### 10 周学习计划

确保根据教程能学会：

**Week 1-2: 夯实基础**
- Part 0: 因果思维（必须能画 DAG、识别 confounder）
- Part 1.1-1.2: A/B 测试基础 + CUPED
- 练习：设计一个电商促销实验

**Week 3-4: 观测数据方法**
- Part 2: PSM → IPW → DR
- Deep Dive: 处理极端倾向得分
- 练习：用 Lalonde 数据估计 ATT

**Week 5-6: 准实验方法**
- Part 3: DID + RDD
- Pitfall: 平行趋势检验实战
- 练习：用合成控制法分析政策效果

**Week 7-8: 异质性效应**
- Part 4: Meta-Learners + Causal Forest
- Deep Dive: 修改 DragonNet 处理多处理
- 练习：设计一个智能发券策略

**Week 9-10: 项目实战**
- 完成一个 End-to-End 项目
- 准备面试八股文
- 模拟面试

### 核心能力检查清单

- [ ] 能手画 DAG，识别 confounder/collider/mediator
- [ ] 能解释 ATE/ATT/CATE 的区别和估计方法
- [ ] 能推导 IPW 估计量的无偏性
- [ ] 能实现 T-Learner（不用库，用 sklearn 基模型）
- [ ] 能设计 A/B 测试并计算样本量
- [ ] 能用 CUPED 减少 30%+ 方差（并解释原理）
- [ ] 能诊断 PSM 的 balance 问题并修复
- [ ] 能解释 DID 的平行趋势假设并检验
- [ ] 能修改 DragonNet loss 处理不平衡数据
- [ ] 能设计 Uplift 模型的评估方案（Qini/AUUC）

---

*评审完成于 2026-01-04*
