# TreatmentEffectLab 快速上手指南

## 5 分钟快速开始

### 1. 启动应用

```bash
cd /Users/zhangjunmengyang/PycharmProjects/awesome-casual-inference
python app.py
```

访问: http://localhost:7860

### 2. 导航到 TreatmentEffectLab

点击顶部 "TreatmentEffectLab" 标签页

### 3. 探索三个子模块

#### 📊 Propensity Score Matching (PSM)
**学习目标**: 理解倾向得分和匹配方法

**操作步骤**:
1. 调整样本量 (推荐 2000)
2. 设置混淆强度 (推荐 1.5)
3. 尝试开启/关闭卡尺匹配
4. 点击 "运行 PSM"
5. 观察:
   - 匹配前后倾向得分分布
   - SMD (标准化均值差) 变化
   - ATE 估计改进

**关键洞察**:
- 匹配后 SMD 应该 < 0.1
- PSM 减少但不完全消除偏差
- 卡尺可以提高匹配质量

#### ⚖️ Inverse Probability Weighting (IPW)
**学习目标**: 理解 IPW 和双重稳健性

**操作步骤**:
1. 调整样本量
2. 设置混淆强度
3. 选择是否裁剪权重
4. 点击 "运行 IPW/AIPW"
5. 对比:
   - IPW vs AIPW 估计
   - 权重分布
   - 有效样本量

**关键洞察**:
- AIPW 标准误通常小于 IPW
- 极端权重会影响估计稳定性
- 权重裁剪是常用技巧

#### 🛡️ Doubly Robust
**学习目标**: 验证双重稳健性质

**子标签 1: 双重稳健性演示**
1. 设置样本量
2. 点击 "演示双重稳健性"
3. 观察四种情况:
   - ✓✓ 两模型都对 → 最优
   - ✓✗ 只有倾向得分对 → **仍然一致**
   - ✗✓ 只有结果模型对 → **仍然一致**
   - ✗✗ 都错 → 有偏

**子标签 2: 方法对比**
1. 对比所有方法:
   - 朴素估计
   - 回归调整
   - IPW
   - AIPW
2. 观察 AIPW 通常表现最好

## 代码使用示例

### 基础使用

```python
from treatment_effect_lab.utils import generate_confounded_data
from treatment_effect_lab.ipw import AIPWEstimator

# 1. 生成数据
df, params = generate_confounded_data(
    n_samples=2000,
    confounding_strength=1.5
)

# 2. 准备数据
feature_names = [f'X{i+1}' for i in range(5)]
X = df[feature_names].values
T = df['T'].values
Y = df['Y'].values

# 3. AIPW 估计 (推荐)
aipw = AIPWEstimator()
ate, se = aipw.estimate_ate(X, T, Y)

print(f"真实 ATE: {params['true_ate']:.4f}")
print(f"AIPW 估计: {ate:.4f} ± {se:.4f}")
print(f"95% CI: [{ate - 1.96*se:.4f}, {ate + 1.96*se:.4f}]")
```

### 方法对比

```python
from treatment_effect_lab.propensity_score import (
    PropensityScoreEstimator, PropensityScoreMatching
)
from treatment_effect_lab.ipw import IPWEstimator, AIPWEstimator
from treatment_effect_lab.utils import compute_naive_ate

# 朴素估计
naive_ate = compute_naive_ate(df)

# PSM
ps_model = PropensityScoreEstimator()
propensity = ps_model.fit_predict(X, T)
psm = PropensityScoreMatching(n_neighbors=1)
matched_t, matched_c = psm.match(propensity, T)
psm_ate, _ = psm.estimate_ate(Y)

# IPW
ipw = IPWEstimator()
ipw.fit(X, T)
ipw_ate, _, _ = ipw.estimate_ate(X, T, Y)

# AIPW
aipw = AIPWEstimator()
aipw_ate, _ = aipw.estimate_ate(X, T, Y)

# 对比
print(f"朴素:  {naive_ate:.4f}")
print(f"PSM:   {psm_ate:.4f}")
print(f"IPW:   {ipw_ate:.4f}")
print(f"AIPW:  {aipw_ate:.4f}")
print(f"真实:  {params['true_ate']:.4f}")
```

## 理解核心概念

### 倾向得分
```
e(X) = P(T=1|X)
```
个体接受处理的概率，是协变量 X 的函数

### IPW 权重
```
w_i = T_i/e(X_i) + (1-T_i)/(1-e(X_i))
```
- 处理组: 低倾向得分 → 高权重
- 控制组: 高倾向得分 → 高权重
- 目的: 创造"伪总体"，平衡协变量

### AIPW (双重稳健)
```
ATE = E[(mu_1(X) - mu_0(X)) +
        T*(Y - mu_1(X))/e(X) -
        (1-T)*(Y - mu_0(X))/(1-e(X))]
```

**为什么双重稳健?**
- 结果模型对 → IPW 修正项期望为 0
- 倾向得分对 → IPW 创造伪总体

## 实践建议

### 1. 数据准备
```python
# 确保特征标准化
from treatment_effect_lab.utils import standardize_features
X_scaled, scaler = standardize_features(X)
```

### 2. 诊断检查
```python
from treatment_effect_lab.utils import (
    compute_smd,
    compute_propensity_overlap
)

# 平衡性
smd = compute_smd(X[T==1], X[T==0])
print(f"最大 SMD: {abs(smd).max():.4f}")  # 应该 < 0.1

# 重叠
overlap = compute_propensity_overlap(propensity, T)
print(f"重叠区间: [{overlap['overlap_min']:.4f}, "
      f"{overlap['overlap_max']:.4f}]")
```

### 3. 方法选择
- **有混淆** → AIPW (首选)
- **样本量小** → PSM (丢失样本少)
- **倾向得分准确** → IPW
- **不确定** → AIPW (双重保护)

## 常见问题

### Q1: 为什么朴素估计有偏差?
A: 因为有混淆变量 X 同时影响处理 T 和结果 Y，导致选择偏差。

### Q2: PSM 为什么丢失样本?
A: 匹配时，部分处理组个体可能找不到足够相似的控制组个体。

### Q3: IPW 权重为什么会很大?
A: 当倾向得分接近 0 或 1 时，w = 1/e 或 w = 1/(1-e) 会很大。
   解决: 使用权重裁剪或 AIPW。

### Q4: AIPW 总是最好吗?
A: 不一定。如果两个模型都错误，AIPW 也会有偏。
   但在实践中，AIPW 通常最稳健。

### Q5: 如何验证假设?
A:
- 共同支撑: 检查倾向得分重叠
- 平衡性: 计算 SMD
- 模型拟合: 使用交叉验证

## 进阶学习

### 1. 敏感性分析
测试结果对模型选择的敏感性

### 2. 交叉拟合
```python
from sklearn.model_selection import KFold

# 使用交叉拟合避免过拟合偏差
kf = KFold(n_splits=5, shuffle=True)
# 在不同折上拟合模型和估计效应
```

### 3. 机器学习模型
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 使用随机森林代替逻辑回归
aipw = AIPWEstimator(
    propensity_model=RandomForestClassifier(),
    outcome_model=RandomForestRegressor()
)
```

## 相关资源

- **理论**: `treatment_effect_lab/README.md`
- **代码**: `treatment_effect_lab/*.py`
- **练习**: `exercises/chapter2_treatment_effect/`
- **可视化**: Gradio 界面

## 下一步

1. ✅ 完成 PSM 模块练习
2. ✅ 理解 IPW 权重原理
3. ✅ 验证双重稳健性质
4. → 尝试自己的数据
5. → 探索 HeteroEffectLab (CATE 估计)

---

**提示**: 在 Gradio 界面中，每个模块都有详细的理论说明和使用指导。
建议先在界面中探索，再使用代码进行深入分析。
