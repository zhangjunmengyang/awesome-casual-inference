# Datasets 快速上手指南

5 分钟快速掌握 datasets 模块的核心功能。

## 1. 基础导入

```python
from datasets import (
    load_lalonde,                    # LaLonde 数据
    generate_ihdp_semi_synthetic,   # IHDP 数据
    generate_linear_dgp,             # 线性数据
    generate_heterogeneous_dgp,      # 异质性数据
    train_test_split_causal,         # 数据划分
    describe_dataset                 # 描述统计
)
```

## 2. 三个最常用的数据生成函数

### 2.1 简单入门 - 线性数据

```python
# 生成有混淆的线性数据
X, T, Y, true_ite = generate_linear_dgp(
    n_samples=1000,
    confounding=True,
    treatment_effect=2.0
)

# 查看统计
print(f"样本量: {len(T)}")
print(f"真实 ATE: {true_ite.mean():.3f}")
print(f"朴素 ATE: {Y[T==1].mean() - Y[T==0].mean():.3f}")
```

### 2.2 评估 CATE - IHDP 数据

```python
# 生成 IHDP 半合成数据 (已知真实 ITE)
X, T, Y, true_ite = generate_ihdp_semi_synthetic(
    n_samples=747,
    setting='A',  # 或 'B' (更难)
    seed=42
)

# 划分训练/测试集
X_train, X_test, T_train, T_test, Y_train, Y_test, ite_train, ite_test = \
    train_test_split_causal(X, T, Y, true_ite, test_size=0.3)

# 训练你的 CATE 模型...
# 使用 ite_test 评估性能
```

### 2.3 研究异质性 - 异质性数据

```python
# 生成异质性处理效应数据
X, T, Y, true_ite = generate_heterogeneous_dgp(
    n_samples=1000,
    heterogeneity_type='threshold'  # 'linear', 'interaction', 'complex'
)

# 异质性统计
print(f"ATE: {true_ite.mean():.3f}")
print(f"ITE 标准差: {true_ite.std():.3f}")
print(f"异质性系数: {true_ite.std() / true_ite.mean():.2f}")
```

## 3. 完整评估流程 (T-Learner 示例)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 生成数据
X, T, Y, true_ite = generate_ihdp_semi_synthetic(setting='A')

# 2. 划分数据
X_tr, X_te, T_tr, T_te, Y_tr, Y_te, ite_tr, ite_te = \
    train_test_split_causal(X, T, Y, true_ite, test_size=0.3)

# 3. 训练 T-Learner
model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
model_1 = RandomForestRegressor(n_estimators=100, random_state=42)

model_0.fit(X_tr[T_tr==0], Y_tr[T_tr==0])
model_1.fit(X_tr[T_tr==1], Y_tr[T_tr==1])

# 4. 预测 CATE
pred_ite = model_1.predict(X_te) - model_0.predict(X_te)

# 5. 评估
rmse = np.sqrt(mean_squared_error(ite_te, pred_ite))
print(f"CATE RMSE: {rmse:.3f}")
```

## 4. 数据集描述和诊断

```python
# 生成数据
X, T, Y, true_ite = generate_heterogeneous_dgp(n_samples=1000)

# 描述统计
stats = describe_dataset(X, T, Y, true_ite)
print(stats)

# 协变量平衡检查
from datasets.utils import check_covariate_balance
balance = check_covariate_balance(X, T)
print(balance)

# 倾向得分
from datasets.utils import compute_propensity_score
ps = compute_propensity_score(X, T)
print(f"PS 范围: [{ps.min():.3f}, {ps.max():.3f}]")
```

## 5. LaLonde 实验 (观测数据偏差)

```python
# 对比 RCT vs 观测数据
nsw_df = load_lalonde('nsw')   # RCT
psid_df = load_lalonde('psid') # 观测数据

# 朴素 ATE
ate_nsw = nsw_df[nsw_df['treat']==1]['re78'].mean() - \
          nsw_df[nsw_df['treat']==0]['re78'].mean()

ate_psid = psid_df[psid_df['treat']==1]['re78'].mean() - \
           psid_df[psid_df['treat']==0]['re78'].mean()

print(f"NSW (RCT) ATE: ${ate_nsw:,.2f}")      # ~$1,900
print(f"PSID (Obs) ATE: ${ate_psid:,.2f}")    # 负值 (偏差!)
```

## 6. 可视化

```python
from datasets.utils import plot_dataset_overview, plot_propensity_overlap

# 生成数据
X, T, Y, true_ite = generate_heterogeneous_dgp()

# 数据集概览
fig1 = plot_dataset_overview(X, T, Y, true_ite)
fig1.show()  # 在 Jupyter 中

# 倾向得分重叠
fig2 = plot_propensity_overlap(X, T)
fig2.show()
```

## 7. 营销场景

```python
from datasets.synthetic import generate_marketing_dgp

# 优惠券场景
df, true_uplift = generate_marketing_dgp(
    n_samples=5000,
    scenario='coupon'  # 'email', 'recommendation'
)

print(df.head())
print(f"平均 Uplift: {true_uplift.mean():.4f}")
```

## 常见问题

### Q1: 如何选择合适的数据集?

| 目标 | 推荐 |
|------|------|
| 学习因果基础 | `generate_linear_dgp` |
| 评估 CATE 方法 | `generate_ihdp_semi_synthetic` |
| 测试非线性模型 | `generate_nonlinear_dgp` |
| 研究异质性 | `generate_heterogeneous_dgp` |
| 观测数据方法 | `load_lalonde` |

### Q2: 如何确保数据划分合理?

```python
# ✓ 使用 train_test_split_causal (保持处理组比例)
X_tr, X_te, ... = train_test_split_causal(
    X, T, Y, true_ite,
    stratify_treatment=True  # 关键!
)

# ✗ 不要使用普通的 train_test_split
# from sklearn.model_selection import train_test_split  # 不推荐
```

### Q3: 如何评估 CATE 模型?

```python
from sklearn.metrics import mean_squared_error, r2_score

# 使用真实 ITE
mse = mean_squared_error(true_ite, predicted_ite)
r2 = r2_score(true_ite, predicted_ite)

print(f"RMSE: {np.sqrt(mse):.3f}")
print(f"R²: {r2:.3f}")
```

### Q4: 如何检查混淆?

```python
from datasets.utils import check_covariate_balance

# 检查协变量平衡
balance = check_covariate_balance(X, T, threshold=0.1)

# 查看不平衡特征
imbalanced = balance[balance['SMD'] > 0.1]
if len(imbalanced) > 0:
    print("⚠ 存在混淆!")
    print(imbalanced)
```

## 下一步

1. 运行完整演示: `python -m datasets.demo`
2. 查看详细文档: `datasets/README.md`
3. 在实际项目中使用

## 速查表

```python
# 数据生成
generate_linear_dgp(n_samples=1000, confounding=True)
generate_ihdp_semi_synthetic(n_samples=747, setting='A')
generate_heterogeneous_dgp(heterogeneity_type='threshold')
load_lalonde(version='nsw')  # 'psid', 'cps'

# 数据处理
train_test_split_causal(X, T, Y, true_ite, stratify_treatment=True)
describe_dataset(X, T, Y, true_ite)

# 诊断
check_covariate_balance(X, T)
compute_propensity_score(X, T)

# 可视化
plot_dataset_overview(X, T, Y, true_ite)
plot_propensity_overlap(X, T)
```

---

更多信息: `datasets/README.md` | 演示代码: `datasets/demo.py`
