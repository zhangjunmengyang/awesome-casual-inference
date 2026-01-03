# TreatmentEffectLab - 处理效应估计实验室

## 概述

TreatmentEffectLab 提供了从观测数据中估计因果效应的核心方法，包括倾向得分匹配、逆概率加权和双重稳健估计。

## 模块结构

```
treatment_effect_lab/
├── __init__.py              # 模块导出
├── utils.py                 # 工具函数
├── propensity_score.py      # 倾向得分匹配 (PSM)
├── ipw.py                   # 逆概率加权 (IPW/AIPW)
└── doubly_robust.py         # 双重稳健估计
```

## 核心概念

### 1. 倾向得分 (Propensity Score)

**定义**: e(X) = P(T=1|X) - 个体接受处理的概率

**核心定理** (Rosenbaum & Rubin, 1983):
- 如果 (Y(0), Y(1)) ⊥ T | X (强可忽略性)
- 则 (Y(0), Y(1)) ⊥ T | e(X)

**应用**: 通过在倾向得分上条件化，可以平衡协变量分布

### 2. 倾向得分匹配 (PSM)

**方法**:
1. 估计倾向得分 e(X)
2. 为每个处理组个体找到倾向得分相近的控制组个体
3. 计算匹配样本中的平均处理效应

**匹配策略**:
- 最近邻匹配 (Nearest Neighbor)
- 卡尺匹配 (Caliper Matching)
- 核匹配 (Kernel Matching)

### 3. 逆概率加权 (IPW)

**核心思想**: 使用倾向得分的逆作为权重，创造"伪总体"

**权重**:
```
w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))
```

**ATE 估计**:
```
ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
```

### 4. 增强 IPW (AIPW)

**双重稳健性质**: 只要倾向得分模型或结果模型有一个正确，估计就是一致的

**公式**:
```
ATE = E[(mu_1(X) - mu_0(X)) +
        T*(Y - mu_1(X))/e(X) -
        (1-T)*(Y - mu_0(X))/(1-e(X))]
```

其中:
- mu_0(X) = E[Y|X, T=0]
- mu_1(X) = E[Y|X, T=1]
- e(X) = P(T=1|X)

## 使用示例

### 倾向得分匹配

```python
from treatment_effect_lab.propensity_score import (
    PropensityScoreEstimator,
    PropensityScoreMatching
)

# 估计倾向得分
ps_model = PropensityScoreEstimator()
propensity = ps_model.fit_predict(X, T)

# 匹配
psm = PropensityScoreMatching(n_neighbors=1, caliper=0.1)
matched_t, matched_c = psm.match(propensity, T)

# 估计 ATE
ate, se = psm.estimate_ate(Y)
print(f"PSM ATE: {ate:.4f} ± {se:.4f}")
```

### 逆概率加权

```python
from treatment_effect_lab.ipw import IPWEstimator

# IPW 估计
ipw = IPWEstimator(clip_weights=True)
ipw.fit(X, T)
ate, se, weights = ipw.estimate_ate(X, T, Y)
print(f"IPW ATE: {ate:.4f} ± {se:.4f}")
```

### 增强 IPW (推荐)

```python
from treatment_effect_lab.ipw import AIPWEstimator

# AIPW 估计 (双重稳健)
aipw = AIPWEstimator()
ate, se = aipw.estimate_ate(X, T, Y)
print(f"AIPW ATE: {ate:.4f} ± {se:.4f}")
```

### 双重稳健估计

```python
from treatment_effect_lab.doubly_robust import DoublyRobustEstimator

# 双重稳健估计
dr = DoublyRobustEstimator(model_type='linear')
ate, se = dr.estimate_ate(X, T, Y,
                           propensity_correct=True,
                           outcome_correct=True)
print(f"DR ATE: {ate:.4f} ± {se:.4f}")
```

## 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **PSM** | 直观、非参数 | 可能丢失样本、效率低 | 样本量大、需检查共同支撑 |
| **IPW** | 使用所有样本 | 对极端权重敏感 | 倾向得分模型准确 |
| **AIPW** | 双重稳健、高效 | 计算复杂度较高 | **推荐首选** |

## 关键假设

### 1. 强可忽略性 (Strong Ignorability)

```
(Y(0), Y(1)) ⊥ T | X
```

给定协变量 X，处理分配与潜在结果独立

### 2. 共同支撑 (Common Support)

```
0 < P(T=1|X) < 1  对所有 X
```

处理组和控制组的倾向得分分布有重叠

### 3. SUTVA

- 稳定单元处理值假设
- 无干扰、无隐藏版本的处理

## 诊断工具

### 平衡性检查

**标准化均值差 (SMD)**:
```
SMD = (mean_treated - mean_control) / pooled_std
```

一般认为 |SMD| < 0.1 表示良好平衡

### 重叠检查

- 倾向得分分布图
- 重叠区间计算
- 极端权重检查

### 有效样本量

```
ESS = (sum(w))^2 / sum(w^2)
```

衡量权重分散程度，越接近 n 越好

## 实践建议

1. **优先使用 AIPW**: 双重稳健性提供额外保护
2. **检查假设**: 验证共同支撑和平衡性
3. **裁剪权重**: IPW 方法中裁剪极端权重
4. **敏感性分析**: 测试对模型选择的敏感性
5. **交叉验证**: 使用交叉拟合避免过拟合偏差

## 进阶主题

### 交叉拟合 (Cross-fitting)

避免过拟合偏差，特别是在使用机器学习模型时

### 目标化学习 (Targeted Learning)

进一步优化效率，减小渐进方差

### 去偏机器学习 (Debiased ML)

结合高维机器学习和双重稳健性

## 参考文献

1. Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects.
2. Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed.
3. Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and causal inference models.
4. Lunceford, J. K., & Davidian, M. (2004). Stratification and weighting via the propensity score in estimation of causal treatment effects.

## 练习

完成以下练习以加深理解:

1. `exercises/chapter2_treatment_effect/ex1_psm.py` - PSM 实践
2. `exercises/chapter2_treatment_effect/ex2_ipw.py` - IPW 实践
3. `exercises/chapter2_treatment_effect/ex3_doubly_robust.py` - 双重稳健性验证

## 可视化示例

在 Gradio 界面中，你可以:

1. **PSM 模块**: 观察匹配前后的协变量平衡
2. **IPW 模块**: 查看权重分布和有效样本量
3. **双重稳健模块**: 验证双重稳健性质

运行 `python app.py` 并访问 TreatmentEffectLab 标签页。
