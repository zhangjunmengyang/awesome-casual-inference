# Chapter 5: 异质性处理效应 (Heterogeneous Treatment Effects)

本章节练习聚焦于理解和估计异质性处理效应，这是因果推断中的高级主题。

## 练习文件

### ex1_cate_basics.py - CATE 基础

**学习目标:**
- 理解条件平均处理效应 (CATE) 的概念
- 识别和分析子群体效应差异
- 可视化效应异质性
- 理解 CATE 与 ATE 的关系

**核心概念:**
- CATE: τ(x) = E[Y(1) - Y(0) | X = x]
- T-Learner: 分别为处理组和对照组训练模型
- PEHE: 评估 CATE 估计精度的指标
- 子群体分析: 根据预测 CATE 将样本分组
- 基于 CATE 的最优处理策略

**练习内容:**
1. 生成具有异质性效应的数据
2. 实现 T-Learner 估计 CATE
3. 计算 ATE 和 PEHE
4. 识别和分析子群体
5. 可视化 CATE 分布和与特征的关系
6. 设计基于 CATE 的最优干预策略

---

### ex2_causal_forest.py - 因果森林

**学习目标:**
- 理解因果森林的核心原理
- 实现诚实分裂 (Honest Splitting)
- 使用 econml 的 CausalForest
- 分析特征重要性
- 对比因果森林与 T-Learner 的性能

**核心概念:**
- 诚实分裂: 分裂样本 vs 估计样本
- 自适应邻域: 使用树结构定义相似样本
- 渐近正态性: 理论保证的置信区间
- 特征重要性: 识别驱动异质性的关键特征

**练习内容:**
1. 实现诚实分裂
2. (可选) 实现简化版因果树
3. 使用 econml.grf.CausalForest
4. 对比因果森林与 T-Learner
5. 分析特征对 CATE 的重要性

**依赖:**
```bash
pip install econml  # 可选，但推荐安装
```

---

### ex3_sensitivity_analysis.py - 敏感性分析

**学习目标:**
- 理解未观测混淆的影响
- 实现 Rosenbaum 敏感性边界
- 计算 E-value
- 进行稳健性检验
- 理解敏感性分析在因果推断中的重要性

**核心概念:**
- 无混淆假设: 通常无法验证
- Rosenbaum Γ: 量化允许的未观测混淆强度
- E-value: 使观测关联完全被混淆解释所需的最小风险比
- Placebo 测试: 使用不应受影响的结果进行检验
- 稳健性: 结论对假设违背的敏感性

**练习内容:**
1. 模拟包含未观测混淆的数据
2. 计算 Rosenbaum 敏感性边界
3. 计算 E-value
4. 实现 Placebo 测试
5. 可视化敏感性分析结果

---

## 运行练习

每个练习文件都是独立的，可以直接运行:

```bash
# 运行练习 1
python exercises/chapter5_hetero_effect/ex1_cate_basics.py

# 运行练习 2
python exercises/chapter5_hetero_effect/ex2_causal_forest.py

# 运行练习 3
python exercises/chapter5_hetero_effect/ex3_sensitivity_analysis.py
```

## 学习路径

建议按照以下顺序学习:

1. **ex1_cate_basics.py** - 先理解 CATE 的基本概念
2. **ex2_causal_forest.py** - 学习更高级的 CATE 估计方法
3. **ex3_sensitivity_analysis.py** - 评估结论的稳健性

## 配套资源

### 相关代码
- `hetero_effect_lab/cate_visualization.py` - CATE 可视化
- `hetero_effect_lab/causal_forest.py` - 因果森林实现
- `hetero_effect_lab/sensitivity.py` - 敏感性分析
- `hetero_effect_lab/utils.py` - 工具函数

### 参考文献

**CATE 基础:**
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning"

**因果森林:**
- Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"
- Athey, Tibshirani & Wager (2019). "Generalized Random Forests"

**敏感性分析:**
- Rosenbaum (2002). "Observational Studies"
- VanderWeele & Ding (2017). "Sensitivity Analysis in Observational Research"

## 评估标准

完成练习后，你应该能够:

- ✓ 解释 CATE 与 ATE 的区别和联系
- ✓ 实现 T-Learner 估计 CATE
- ✓ 理解 PEHE 指标的含义
- ✓ 识别和解释子群体效应差异
- ✓ 理解诚实分裂的原理和作用
- ✓ 使用因果森林估计 CATE
- ✓ 分析特征对异质性的贡献
- ✓ 理解未观测混淆的影响
- ✓ 计算和解释 Rosenbaum 边界
- ✓ 进行敏感性分析和稳健性检验

## 常见问题

**Q: 为什么 CATE 比 ATE 更有用?**

A: CATE 揭示了处理效应的异质性，可以:
- 识别最受益的子群体 (精准营销、个性化医疗)
- 避免对可能受损的群体进行干预
- 优化资源分配，提高 ROI

**Q: 因果森林相比 T-Learner 有什么优势?**

A: 因果森林的优势:
- 诚实分裂防止过拟合
- 提供渐近正态性的理论保证
- 可以构造有效的置信区间
- 对混淆的鲁棒性更好

**Q: 什么时候需要敏感性分析?**

A: 总是需要！特别是:
- 观测性研究 (非随机化)
- 高风险决策 (医疗、政策)
- 可能存在重要未观测混淆

**Q: 如何解释 Rosenbaum Γ = 2?**

A: Γ = 2 意味着:
- 即使两个个体的观测协变量完全相同
- 他们接受处理的概率仍可能相差 2 倍 (由于未观测混淆)
- 这属于中等强度的未观测混淆

## 进阶挑战

完成基础练习后，可以尝试:

1. **实现 X-Learner 和 S-Learner**
   - 对比三种 Meta-Learner 的性能
   - 分析它们在不同场景下的优劣

2. **使用真实数据集**
   - 在 IHDP、Jobs 等基准数据集上测试
   - 与发表的结果进行对比

3. **实现 CATE 的置信区间**
   - 使用 Bootstrap 方法
   - 或使用因果森林的渐近方差

4. **探索更多敏感性分析方法**
   - Partial Identification Bounds
   - Regression-based Sensitivity
   - 工具变量方法

---

祝学习愉快！如有问题，请参考 `hetero_effect_lab/` 下的源码实现。
