# Datasets Module - 因果推断数据集

专为因果推断学习和评估设计的数据集模块。

## 概述

本模块提供:
1. **经典因果推断数据集**: LaLonde, IHDP
2. **合成数据生成器**: 多种因果模型的数据生成
3. **实用工具**: 数据划分、描述统计、平衡性检查

## 快速开始

```python
# 导入数据集
from datasets import (
    load_lalonde,
    generate_ihdp_semi_synthetic,
    generate_heterogeneous_dgp,
    train_test_split_causal,
    describe_dataset
)

# 1. 加载 LaLonde 数据
df = load_lalonde(version='nsw')
print(df.head())

# 2. 生成 IHDP 数据 (用于 CATE 评估)
X, T, Y, true_ite = generate_ihdp_semi_synthetic(n_samples=747, setting='A')

# 3. 生成异质性效应数据
X, T, Y, true_ite = generate_heterogeneous_dgp(
    n_samples=1000,
    heterogeneity_type='linear'
)

# 4. 数据划分 (保持处理组比例)
X_train, X_test, T_train, T_test, Y_train, Y_test, ite_train, ite_test = \
    train_test_split_causal(X, T, Y, true_ite, test_size=0.3)

# 5. 描述统计
stats = describe_dataset(X, T, Y, true_ite)
print(stats)
```

## 数据集详解

### 1. LaLonde 数据集

经典的就业培训项目数据，用于评估观测数据方法。

```python
from datasets import load_lalonde

# NSW 实验数据 (RCT, n=722)
nsw_df = load_lalonde('nsw')

# PSID 观测数据 (n=2490, 存在选择偏差)
psid_df = load_lalonde('psid')

# CPS 观测数据 (n=15992, 更大选择偏差)
cps_df = load_lalonde('cps')
```

**变量说明:**
- `age`: 年龄
- `education`: 受教育年限
- `black`: 是否黑人 (1=是)
- `hispanic`: 是否西班牙裔 (1=是)
- `married`: 是否已婚 (1=是)
- `nodegree`: 是否没有学位 (1=是)
- `re74`: 1974年收入 (美元)
- `re75`: 1975年收入 (美元)
- `treat`: 是否接受培训 (1=是)
- `re78`: 1978年收入 (结果变量, 美元)

**应用场景:**
- 倾向得分匹配 (PSM)
- 协变量平衡检验
- 观测数据 vs RCT 对比

### 2. IHDP 数据集

婴儿健康发展计划数据，CATE 评估的黄金标准。

```python
from datasets import generate_ihdp_semi_synthetic

# 设置 A: 中等非线性
X, T, Y, true_ite = generate_ihdp_semi_synthetic(
    n_samples=747,
    setting='A',
    seed=42
)

# 设置 B: 高度非线性 (更具挑战)
X, T, Y, true_ite = generate_ihdp_semi_synthetic(
    n_samples=747,
    setting='B',
    seed=42
)
```

**数据特征:**
- 样本量: 747 (139 处理, 608 对照)
- 协变量: 25 个 (连续 + 离散)
- 真实 ITE: 已知，便于评估

**应用场景:**
- CATE 方法评估 (S/T/X-Learner, Causal Forest)
- 深度因果模型 (TARNet, DragonNet)
- 异质性效应分析

### 3. 合成数据生成器

#### 3.1 线性 DGP

```python
from datasets import generate_linear_dgp

X, T, Y, true_ite = generate_linear_dgp(
    n_samples=1000,
    n_features=5,
    treatment_effect=2.0,    # ATE
    confounding=True,        # 是否有混淆
    noise_std=1.0
)
```

**特点:** 常数处理效应，适合入门学习

#### 3.2 非线性 DGP

```python
from datasets import generate_nonlinear_dgp

X, T, Y, true_ite = generate_nonlinear_dgp(
    n_samples=1000,
    complexity='high',  # 'low', 'medium', 'high'
    noise_std=1.0
)
```

**特点:** 非线性响应函数，测试模型表达能力

#### 3.3 异质性 DGP

```python
from datasets import generate_heterogeneous_dgp

X, T, Y, true_ite = generate_heterogeneous_dgp(
    n_samples=1000,
    heterogeneity_type='threshold',  # 'linear', 'interaction', 'threshold', 'complex'
    noise_std=1.0
)
```

**异质性类型:**
- `linear`: τ(X) = α + β'X
- `interaction`: τ(X) 包含交互项
- `threshold`: 阈值效应 (分段常数)
- `complex`: 复杂非线性

#### 3.4 营销场景数据

```python
from datasets.synthetic import generate_marketing_dgp

df, true_uplift = generate_marketing_dgp(
    n_samples=5000,
    scenario='coupon'  # 'coupon', 'email', 'recommendation'
)
```

**场景:**
- `coupon`: 优惠券发放 (转化率)
- `email`: 邮件营销 (点击率)
- `recommendation`: 推荐系统 (购买率)

## 工具函数

### 1. 数据划分

```python
from datasets import train_test_split_causal

# 分层抽样 (保持处理组比例)
X_train, X_test, T_train, T_test, Y_train, Y_test, ite_train, ite_test = \
    train_test_split_causal(
        X, T, Y, true_ite,
        test_size=0.3,
        stratify_treatment=True,  # 关键参数
        seed=42
    )

print(f"Train treatment rate: {T_train.mean():.2%}")
print(f"Test treatment rate: {T_test.mean():.2%}")
```

### 2. 描述统计

```python
from datasets import describe_dataset

stats = describe_dataset(X, T, Y, true_ite, feature_names=['age', 'income'])
print(stats)
```

输出示例:
```
           Metric   Overall Treatment  Control
      Sample Size      1000       502      498
   Treatment Rate     50.2%         -        -
     Outcome Mean     5.123     6.234    4.012
      Outcome Std     2.345     2.456    2.234
         Naive ATE     2.222         -        -
          True ATE     2.000         -        -
           ATE Bias     0.222         -        -
```

### 3. 协变量平衡检查

```python
from datasets.utils import check_covariate_balance

balance = check_covariate_balance(X, T, threshold=0.1)
print(balance)

# 查看不平衡特征
imbalanced = balance[balance['SMD'] > 0.1]
print(f"\nImbalanced features: {len(imbalanced)}")
```

### 4. 倾向得分计算

```python
from datasets.utils import compute_propensity_score

ps = compute_propensity_score(X, T, method='logistic')
print(f"PS range: [{ps.min():.3f}, {ps.max():.3f}]")
```

### 5. 可视化

```python
from datasets.utils import plot_dataset_overview, plot_propensity_overlap

# 数据集概览
fig1 = plot_dataset_overview(X, T, Y, true_ite)
fig1.show()

# 倾向得分重叠
fig2 = plot_propensity_overlap(X, T)
fig2.show()
```

## 完整示例

### 示例 1: 评估 CATE 方法

```python
from datasets import generate_ihdp_semi_synthetic, train_test_split_causal
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 生成数据
X, T, Y, true_ite = generate_ihdp_semi_synthetic(setting='A')

# 划分数据
X_train, X_test, T_train, T_test, Y_train, Y_test, ite_train, ite_test = \
    train_test_split_causal(X, T, Y, true_ite, test_size=0.3)

# T-Learner
model_0 = RandomForestRegressor().fit(X_train[T_train==0], Y_train[T_train==0])
model_1 = RandomForestRegressor().fit(X_train[T_train==1], Y_train[T_train==1])

pred_ite = model_1.predict(X_test) - model_0.predict(X_test)

# 评估
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ite_test, pred_ite)
print(f"CATE MSE: {mse:.3f}")
```

### 示例 2: 对比不同混淆程度

```python
from datasets import generate_linear_dgp

for confounding in [False, True]:
    X, T, Y, true_ite = generate_linear_dgp(
        n_samples=1000,
        confounding=confounding,
        treatment_effect=2.0
    )

    naive_ate = Y[T==1].mean() - Y[T==0].mean()
    bias = abs(naive_ate - true_ite.mean())

    print(f"Confounding={confounding}: Naive ATE={naive_ate:.3f}, Bias={bias:.3f}")
```

输出:
```
Confounding=False: Naive ATE=2.012, Bias=0.012
Confounding=True: Naive ATE=2.987, Bias=0.987
```

### 示例 3: LaLonde 实验

```python
from datasets import load_lalonde
from datasets.utils import check_covariate_balance
import pandas as pd

# 加载 NSW (RCT) 和 PSID (观测)
nsw_df = load_lalonde('nsw')
psid_df = load_lalonde('psid')

# 提取特征
feature_cols = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']

# 检查平衡性
print("NSW (RCT) Covariate Balance:")
balance_nsw = check_covariate_balance(
    nsw_df[feature_cols].values,
    nsw_df['treat'].values
)
print(balance_nsw)

print("\nPSID (Observational) Covariate Balance:")
balance_psid = check_covariate_balance(
    psid_df[feature_cols].values,
    psid_df['treat'].values
)
print(balance_psid)
```

## 最佳实践

### 1. 选择合适的数据集

| 任务 | 推荐数据集 |
|------|-----------|
| 学习因果推断基础 | `generate_linear_dgp` |
| 评估 CATE 方法 | `generate_ihdp_semi_synthetic` |
| 测试非线性模型 | `generate_nonlinear_dgp` |
| 研究异质性效应 | `generate_heterogeneous_dgp` |
| 营销场景应用 | `generate_marketing_dgp` |
| 观测数据方法 | `load_lalonde('psid')` |

### 2. 数据划分注意事项

```python
# ✓ 正确: 分层抽样
X_train, X_test, ... = train_test_split_causal(
    X, T, Y, true_ite,
    stratify_treatment=True  # 保持处理组比例
)

# ✗ 错误: 简单随机划分可能导致比例失衡
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.3)  # 不推荐
```

### 3. 评估 CATE 方法

```python
# 使用真实 ITE 评估
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(true_ite, predicted_ite)
r2 = r2_score(true_ite, predicted_ite)

print(f"CATE MSE: {mse:.3f}")
print(f"CATE R²: {r2:.3f}")
```

### 4. 检查假设

```python
from datasets.utils import check_covariate_balance, plot_propensity_overlap

# 1. 协变量平衡
balance = check_covariate_balance(X, T)
if (balance['SMD'] > 0.1).any():
    print("⚠ Warning: Covariate imbalance detected")

# 2. 共同支撑 (Overlap)
fig = plot_propensity_overlap(X, T)
fig.show()
```

## API 参考

完整 API 文档请参考各模块的 docstring:

```python
from datasets import load_lalonde
help(load_lalonde)
```

## 贡献指南

添加新数据集时，请确保:
1. 提供完整的 docstring
2. 包含 `__main__` 测试代码
3. 返回格式一致: `(X, T, Y, true_ite)`
4. 添加到 `__init__.py`

## 许可证

本模块遵循项目主许可证。
