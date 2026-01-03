# HeteroEffectLab - 异质性处理效应实验室

## 概述

HeteroEffectLab 是因果推断工作台的核心模块之一，专注于异质性处理效应 (Heterogeneous Treatment Effect, HTE) 的估计、可视化和应用。

## 模块结构

```
hetero_effect_lab/
├── __init__.py              # 模块导出
├── utils.py                 # 工具函数
├── causal_forest.py         # 因果森林实现
├── sensitivity.py           # 敏感性分析
├── cate_visualization.py    # CATE 可视化
└── README.md               # 本文档
```

## 核心功能

### 1. utils.py - 工具函数

提供异质性效应分析的基础工具:

- **generate_heterogeneous_data()**: 生成具有异质性处理效应的模拟数据
  - 支持三种异质性强度: weak, moderate, strong
  - 可控的混淆强度和噪声水平
  - 返回真实的潜在结果用于评估

- **compute_pehe()**: 计算 PEHE (Precision in Estimation of HTE)
  - 黄金标准的 ITE 估计精度指标
  - sqrt(E[(ITE_true - ITE_pred)²])

- **compute_ate_bias()**: 计算 ATE 估计偏差
  - |E[ITE_true] - E[ITE_pred]|
  - 评估总体效应估计的准确性

- **identify_subgroups()**: 根据 CATE 分组识别子群体
  - 基于分位数的自动分组
  - 用于目标人群识别

- **compute_policy_value()**: 计算基于 CATE 的策略价值
  - 评估干预策略的效果

- **compute_r_squared()**: 计算决定系数
  - 评估 CATE 预测的拟合优度

### 2. causal_forest.py - 因果森林

实现和对比因果森林方法:

**核心特性**:
- 使用 econml 的 CausalForest (如果可用)
- 备选: SimpleCausalForest (基于 T-Learner)
- 与 T-Learner 的性能对比
- 特征重要性分析
- 子群体 CATE 估计

**可视化**:
- CATE 分布对比
- 真实 vs 预测 CATE 散点图
- 特征重要性柱状图
- 子群体平均 CATE 对比

**教学内容**:
- 因果森林的原理和优势
- 诚实分裂 (Honest Splitting) 概念
- 渐近正态性和置信区间
- 与其他方法的对比

**函数接口**:
```python
compare_causal_forest_vs_tlearner(
    n_samples: int,
    effect_heterogeneity: str,
    confounding_strength: float,
    n_trees: int
) -> (Figure, str)
```

### 3. sensitivity.py - 敏感性分析

评估因果推断结果对未观测混淆的敏感性:

**核心特性**:
- Rosenbaum Bounds 计算
- 模拟未观测混淆的影响
- 敏感性参数 Γ (Gamma) 的解释
- 置信区间的敏感性曲线

**可视化**:
- Rosenbaum 敏感性边界曲线
- 未观测混淆 U 的分布
- 处理分配随 U 的变化
- 结果 Y 按 T 和 U 分组

**教学内容**:
- 无混淆假设的局限性
- Rosenbaum Bounds 方法
- Γ 参数的含义和解释
- E-value 等其他敏感性方法
- 如何报告敏感性分析

**函数接口**:
```python
visualize_sensitivity_analysis(
    n_samples: int,
    confounder_strength: float,
    correlation_with_x: float,
    max_gamma: float
) -> (Figure, str)
```

### 4. cate_visualization.py - CATE 可视化

多维度可视化和解释 CATE:

**核心特性**:
- 带置信区间的 CATE 估计 (TLearnerWithCI)
- 按特征分组的 CATE 分析
- 子群体识别和对比
- 个体处理效应分布

**可视化** (6 panel 综合图):
1. CATE 分布 + 95% 置信区间
2. 真实 vs 预测 CATE 散点图
3. CATE 随特征 X1 变化
4. CATE 随特征 X2 变化
5. 子群体平均 CATE 对比
6. 个体处理效应分布

**教学内容**:
- CATE 可视化的重要性
- 置信区间的计算和解释
- 子群体识别方法
- 精准干预决策应用
- 实践建议和常见陷阱

**函数接口**:
```python
visualize_cate_analysis(
    n_samples: int,
    effect_heterogeneity: str,
    n_bootstrap: int,
    n_subgroups: int
) -> (Figure, str)
```

## 技术栈

- **UI**: Gradio
- **可视化**: Plotly (统一配色: #2D9CDB, #27AE60, #EB5757)
- **因果推断**: econml (可选)
- **机器学习**: scikit-learn
- **数据处理**: numpy, pandas

## 设计原则

1. **教学优先**: 每个模块都包含详细的原理说明和应用指导
2. **交互式学习**: 通过参数调整观察效应变化
3. **真实数据**: 使用模拟数据展示真实场景
4. **对比分析**: 多种方法对比，理解优劣
5. **实用导向**: 提供实际应用案例和建议

## 使用示例

### 基础用法

```python
from hetero_effect_lab import causal_forest, sensitivity, cate_visualization

# 渲染因果森林界面
causal_forest.render()

# 渲染敏感性分析界面
sensitivity.render()

# 渲染 CATE 可视化界面
cate_visualization.render()
```

### 编程使用

```python
from hetero_effect_lab.utils import generate_heterogeneous_data, compute_pehe

# 生成数据
df, true_cate, Y0, Y1 = generate_heterogeneous_data(
    n_samples=1000,
    effect_heterogeneity='moderate'
)

# 训练模型并评估
# ... 你的模型训练代码 ...

# 计算 PEHE
pehe = compute_pehe(Y0, Y1, Y0_pred, Y1_pred)
print(f"PEHE: {pehe:.4f}")
```

## 应用场景

### 1. 精准营销

识别对促销最敏感的客户群体:

```python
# 估计每个客户的 CATE
cate_pred = model.predict(customer_features)

# 识别高响应客户
high_responders = cate_pred > np.percentile(cate_pred, 75)

# 只对这些客户发送优惠券
# 预期 ROI 提升 = avg(CATE[high_responders]) * cost_saving
```

### 2. 医疗决策

个性化治疗方案:

```python
# 估计患者的 CATE 及置信区间
cate, lower_ci, upper_ci = model.predict(patient_data, return_ci=True)

# 保守决策: 只对置信区间下界 > 0 的患者推荐治疗
should_treat = lower_ci > 0
```

### 3. 政策评估

识别政策受益/受损人群:

```python
# 估计政策的异质性效应
cate_pred = model.predict(population_features)

# 识别受益和受损群体
benefited = cate_pred > 0
harmed = cate_pred < 0

# 制定差异化补偿政策
```

## 评估指标

- **PEHE**: 异质性处理效应估计精度 (越小越好)
- **ATE Bias**: 平均处理效应偏差 (越小越好)
- **R²**: CATE 预测的拟合优度 (越接近 1 越好)
- **AUUC**: Uplift 曲线下面积 (在 evaluation 模块)

## 与其他模块的关系

- **uplift_lab**: 提供基础的 Meta-Learners (S/T/X-Learner)
- **deep_causal_lab**: 提供深度学习方法 (TARNet, DragonNet)
- **evaluation_lab**: 提供评估和诊断工具
- **treatment_effect_lab**: 提供传统因果推断方法 (PSM, IPW)

HeteroEffectLab 专注于异质性效应的高级方法和应用。

## 未来扩展

- [ ] BART (Bayesian Additive Regression Trees)
- [ ] Generalized Random Forest (GRF)
- [ ] Neural Networks for HTE
- [ ] 更多敏感性分析方法 (E-value, Partial Identification)
- [ ] 时间序列异质性效应

## 参考文献

### 因果森林
- Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests". JASA.
- Athey, Tibshirani & Wager (2019). "Generalized Random Forests". Annals of Statistics.

### 敏感性分析
- Rosenbaum (2002). "Observational Studies". Springer.
- VanderWeele & Ding (2017). "Sensitivity Analysis in Observational Research". Annals of Internal Medicine.

### CATE 估计
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning". PNAS.
- Foster et al. (2011). "Subgroup identification from randomized clinical trial data". Statistics in Medicine.

## 贡献指南

欢迎贡献! 请确保:
1. 遵循项目的代码风格 (参考 CLAUDE.md)
2. 每个功能模块包含 `render()` 函数
3. 使用统一的可视化配色方案
4. 提供详细的教学说明和应用案例
5. 编写完整的 docstring

## 许可证

本项目遵循与主项目相同的许可证。

---

**HeteroEffectLab** - 让因果推断真正服务于个性化决策
