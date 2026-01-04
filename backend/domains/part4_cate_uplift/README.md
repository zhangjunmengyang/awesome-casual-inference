# Part 4: CATE & Uplift Modeling

异质性处理效应估计与 Uplift 建模模块

## 模块概述

该模块合并了原有的 `uplift_lab` 和 `hetero_effect_lab` 的功能，提供完整的异质性效应估计和 Uplift 建模能力。

## 文件结构

```
part4_cate_uplift/
├── __init__.py              # 模块导出
├── meta_learners.py         # Meta-Learners (S/T/X/R/DR-Learner)
├── causal_forest.py         # 因果森林
├── uplift_tree.py           # Uplift Tree
├── uplift_evaluation.py     # Uplift 评估方法（Qini, AUUC）
├── cate_visualization.py    # CATE 可视化
├── utils.py                 # 工具函数
├── api.py                   # API 适配层
└── README.md                # 本文档
```

## 核心组件

### 1. Meta-Learners (meta_learners.py)

实现了 5 种主流的 Meta-Learner:

- **SLearner**: 单一模型，将处理作为特征
- **TLearner**: 两个独立模型，分别建模处理组和控制组
- **XLearner**: 两阶段方法，利用反事实估计
- **RLearner**: 基于残差的双重稳健方法
- **DRLearner**: 双重稳健 Meta-Learner

```python
from domains.part4_cate_uplift import SLearner, TLearner, XLearner

# 训练 T-Learner
model = TLearner()
model.fit(X, T, Y)
cate = model.predict(X)
```

### 2. Causal Forest (causal_forest.py)

因果森林实现，支持:
- EconML 的 CausalForest (如果可用)
- SimpleTLearner 作为备选方案

```python
from domains.part4_cate_uplift.causal_forest import get_causal_forest_model

model = get_causal_forest_model(n_trees=100)
model.fit(X, T, Y)
cate = model.predict(X)
```

### 3. Uplift Tree (uplift_tree.py)

Uplift 决策树实现，包括:
- 多种分裂准则: KL, ED, Chi, DDP
- 最佳分裂点搜索
- SimpleUpliftTree 类

```python
from domains.part4_cate_uplift import SimpleUpliftTree

tree = SimpleUpliftTree(criterion='KL', max_depth=3)
tree.fit(X, T, Y)
uplift = tree.predict(X)
```

### 4. Uplift Evaluation (uplift_evaluation.py)

提供完整的 Uplift 评估指标:
- Qini 曲线
- Uplift 曲线
- AUUC (Area Under Uplift Curve)
- Qini 系数
- 累积增益

```python
from domains.part4_cate_uplift import calculate_qini_curve, calculate_auuc

fraction, qini = calculate_qini_curve(y_true, treatment, uplift_score)
auuc = calculate_auuc(y_true, treatment, uplift_score)
```

### 5. CATE Visualization (cate_visualization.py)

CATE 可视化与分析工具:
- 带置信区间的 T-Learner
- 按特征分析 CATE
- 子群体识别
- 分布统计

```python
from domains.part4_cate_uplift.cate_visualization import (
    identify_subgroups,
    compute_subgroup_statistics
)

subgroups = identify_subgroups(X, cate, n_groups=4)
stats = compute_subgroup_statistics(Y, T, cate, subgroups)
```

### 6. Utils (utils.py)

数据生成和工具函数:
- `generate_heterogeneous_data()`: 生成异质性效应数据
- `generate_uplift_data()`: 生成 Uplift 数据
- `generate_marketing_uplift_data()`: 生成营销场景数据
- `compute_pehe()`: 计算 PEHE
- `compute_policy_value()`: 计算策略价值
- 可视化函数

## API 接口

### API 函数列表

所有 API 函数返回统一格式:
```python
{
    "charts": [...],      # Plotly 图表数据
    "tables": [...],      # 表格数据 (dict records)
    "summary": "...",     # Markdown 格式的总结
    "metrics": {...}      # 关键指标
}
```

#### 1. analyze_meta_learners

比较不同 Meta-Learners 的性能

```python
from domains.part4_cate_uplift.api import analyze_meta_learners

result = analyze_meta_learners(
    n_samples=5000,
    effect_type='moderate',  # weak/moderate/strong
    noise_level=0.5,
    confounding_strength=0.3
)
```

#### 2. analyze_causal_forest

分析因果森林

```python
from domains.part4_cate_uplift.api import analyze_causal_forest

result = analyze_causal_forest(
    n_samples=5000,
    effect_heterogeneity='moderate',
    confounding_strength=0.3,
    n_trees=100
)
```

#### 3. analyze_uplift_tree

分析 Uplift Tree

```python
from domains.part4_cate_uplift.api import analyze_uplift_tree

result = analyze_uplift_tree(
    n_samples=5000,
    feature_effect='heterogeneous',
    criterion='KL'  # KL/ED/Chi/DDP
)
```

#### 4. analyze_uplift_evaluation

分析 Uplift 评估方法

```python
from domains.part4_cate_uplift.api import analyze_uplift_evaluation

result = analyze_uplift_evaluation(
    n_samples=5000,
    model_quality='good'  # perfect/good/random
)
```

#### 5. visualize_cate

CATE 可视化

```python
from domains.part4_cate_uplift.api import visualize_cate

result = visualize_cate(
    n_samples=5000,
    effect_heterogeneity='moderate',
    n_bootstrap=50,
    n_subgroups=4
)
```

## 评估指标

### CATE 估计指标

- **PEHE** (Precision in Estimation of Heterogeneous Effect): 异质性效应估计精度
- **R²**: 决定系数
- **Correlation**: 与真实 CATE 的相关性
- **ATE Bias**: 平均处理效应偏差

### Uplift 评估指标

- **AUUC** (Area Under Uplift Curve): Uplift 曲线下面积
- **Qini Coefficient**: Qini 系数
- **Uplift@K**: Top-K% 的平均 Uplift
- **Cumulative Gain**: 累积增益

## 使用示例

### 完整的工作流程

```python
import numpy as np
from domains.part4_cate_uplift import TLearner
from domains.part4_cate_uplift.utils import generate_heterogeneous_data
from domains.part4_cate_uplift.uplift_evaluation import calculate_auuc
from domains.part4_cate_uplift.cate_visualization import identify_subgroups

# 1. 生成数据
df, true_cate, Y0, Y1 = generate_heterogeneous_data(
    n_samples=5000,
    effect_heterogeneity='moderate'
)

X = df[['X1', 'X2', 'X3', 'X4', 'X5']].values
T = df['T'].values
Y = df['Y'].values

# 2. 训练模型
model = TLearner()
model.fit(X, T, Y)

# 3. 预测 CATE
cate_pred = model.predict(X)

# 4. 评估
from domains.part4_cate_uplift.utils import compute_pehe
pehe = compute_pehe(Y0, Y1, Y0, Y0 + cate_pred)
print(f"PEHE: {pehe:.4f}")

# 5. 子群体分析
subgroups = identify_subgroups(X, cate_pred, n_groups=4)
print(f"Subgroup sizes: {np.bincount(subgroups)}")

# 6. Uplift 评估
auuc = calculate_auuc(Y, T, cate_pred)
print(f"AUUC: {auuc:.4f}")
```

## 对应的 Notebooks

该模块对应以下 notebooks:

- `notebooks/part4_cate_uplift/part4_1_cate_basics.ipynb`: CATE 基础
- `notebooks/part4_cate_uplift/part4_2_meta_learners.ipynb`: Meta-Learners
- `notebooks/part4_cate_uplift/part4_3_causal_forest.ipynb`: 因果森林
- `notebooks/part4_cate_uplift/part4_4_uplift_tree.ipynb`: Uplift Tree
- `notebooks/part4_cate_uplift/part4_5_uplift_evaluation.ipynb`: Uplift 评估

## 技术细节

### Meta-Learners 比较

| 模型 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| S-Learner | 简单，数据利用率高 | 难以捕捉异质性 | 效应异质性弱 |
| T-Learner | 简单，易理解 | 需要足够样本 | 各组样本充足 |
| X-Learner | 利用反事实，样本效率高 | 复杂度高 | 样本不平衡 |
| R-Learner | 双重稳健 | 计算复杂 | 混淆严重 |
| DR-Learner | 最优双重稳健 | 最复杂 | 高质量估计 |

### Uplift Tree 分裂准则

- **KL 散度**: 衡量分布差异，理论基础强
- **欧氏距离 (ED)**: 简单直观
- **卡方统计量 (Chi)**: 统计显著性
- **DDP**: 直接优化 Uplift

## 依赖

- NumPy
- Pandas
- Scikit-learn
- Plotly
- EconML (可选，用于 CausalForest)

## 迁移说明

该模块整合了以下原有模块:

- `uplift_lab/meta_learners.py` → `meta_learners.py`
- `uplift_lab/uplift_tree.py` → `uplift_tree.py`
- `uplift_lab/evaluation.py` → `uplift_evaluation.py`
- `uplift_lab/utils.py` → `utils.py` (部分)
- `hetero_effect_lab/causal_forest.py` → `causal_forest.py`
- `hetero_effect_lab/cate_visualization.py` → `cate_visualization.py`
- `hetero_effect_lab/utils.py` → `utils.py` (部分)

所有功能已经过测试和验证。

## 参考文献

1. Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning"
2. Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"
3. Radcliffe & Surry (2011). "Real-World Uplift Modelling with Significance-Based Uplift Trees"
4. Rzepakowski & Jaroszewicz (2012). "Decision trees for uplift modeling with single and multiple treatments"
