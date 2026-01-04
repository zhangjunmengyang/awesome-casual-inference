#!/usr/bin/env python3
"""
创建 Part 7.4: 中介分析 Notebook
"""
import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    })

# ========== Header ==========
add_markdown("""# Part 7.4: 中介分析 (Mediation Analysis)

## 学习目标

1. 理解中介效应的定义和意义
2. 掌握直接效应和间接效应的分解
3. 学习因果中介分析框架
4. 实现从零中介分析算法
5. 应用于真实业务场景

---

## 业务场景：优惠券如何提升转化？

想象你是某电商平台的数据科学家。A/B测试显示，发送优惠券可以提升15%的购买转化率。

**老板的追问**：
- 优惠券是怎么起效的？
- 是因为增加了用户访问次数？
- 还是提高了单次访问的购买意愿？
- 如果不发券，能否通过其他方式（如推送）达到同样效果？

**核心问题**：不仅要知道 \"有没有效\"，还要知道 \"怎么起效\"！

这就是 **中介分析 (Mediation Analysis)** 要解决的问题。

---""")

# ========== Setup ==========
add_code("""# 环境准备
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# 颜色配置
COLORS = {
    'primary': '#2D9CDB',
    'success': '#27AE60',
    'danger': '#EB5757',
    'warning': '#F2994A',
    'info': '#9B51E0'
}

print("✅ 环境准备完成！")""")

# ========== Concepts ==========
add_markdown("""## Part 1: 核心概念

### 因果图

中介分析的典型因果结构：

```
      T (处理: 优惠券)
     / \\
    /   \\
   v     v
  M      Y
(中介)  (结果)
   \\    /
    \\  /
     v
     Y
```

- **T**: 处理变量 (Treatment) - 是否发券
- **M**: 中介变量 (Mediator) - 访问次数
- **Y**: 结果变量 (Outcome) - 是否购买
- **直接路径**: T → Y (不经过M)
- **间接路径**: T → M → Y (经过M)

### 效应分解

**总效应 (Total Effect, TE)**：
$$TE = E[Y(T=1) - Y(T=0)]$$

**直接效应 (Direct Effect, DE)**：
处理对结果的直接影响，不经过中介
$$NDE = E[Y(T=1, M(T=0)) - Y(T=0, M(T=0))]$$

**间接效应 (Indirect Effect, IE)**：
处理通过中介对结果的影响
$$NIE = E[Y(T=0, M(T=1)) - Y(T=0, M(T=0))]$$

**分解关系**：
$$TE = NDE + NIE$$

### 识别假设

1. **顺序忽略性**：
   - $Y(t,m) \\perp T | X$
   - $M(t) \\perp T | X$
   - $Y(t,m) \\perp M | T, X$

2. **无混淆**：没有未观测的混淆因子

---""")

# ========== Data Generation ==========
add_markdown("""## Part 2: 数据生成""")

add_code("""def generate_mediation_data(n=2000, seed=42):
    \"\"\"
    生成中介分析数据

    场景：优惠券(T) → 访问次数(M) → 购买(Y)
    \"\"\"
    np.random.seed(seed)

    # 协变量
    X1 = np.random.normal(0, 1, n)  # 用户活跃度
    X2 = np.random.normal(0, 1, n)  # 价格敏感度
    X = np.column_stack([X1, X2])

    # 处理分配（有混淆）
    propensity = 1 / (1 + np.exp(-(0.5 + 0.3*X1 + 0.2*X2)))
    T = np.random.binomial(1, propensity)

    # 中介变量：访问次数
    # M = f(T, X) + noise
    M = (
        2 +                    # 基线
        0.5 * T +             # 发券增加访问
        0.3 * X1 +            # 活跃度影响
        np.random.normal(0, 0.5, n)
    )
    M = np.maximum(0, M)

    # 结果变量：购买概率
    # Y = f(T, M, X)
    logit_y = (
        -2 +                   # 基线
        0.3 * T +             # 券的直接效应
        0.5 * M +             # 访问次数效应
        0.2 * X2 +            # 价格敏感度
        np.random.normal(0, 0.3, n)
    )
    prob_y = 1 / (1 + np.exp(-logit_y))
    Y = np.random.binomial(1, prob_y)

    # 真实效应（基于DGP）
    # 间接效应: T → M → Y = 0.5 * 0.5 = 0.25 (logit scale)
    # 直接效应: T → Y = 0.3
    # 总效应: 0.3 + 0.25 = 0.55

    return {
        'X': X,
        'T': T,
        'M': M,
        'Y': Y,
        'true_effects': {
            'direct': 0.3,
            'indirect': 0.25,
            'total': 0.55
        }
    }

# 生成数据
data = generate_mediation_data()
X, T, M, Y = data['X'], data['T'], data['M'], data['Y']

print(f"数据维度: n={len(T)}")
print(f"处理组比例: {T.mean():.2%}")
print(f"平均访问次数: {M.mean():.2f}")
print(f"购买率: {Y.mean():.2%}")
print(f"\\n真实效应:")
for k, v in data['true_effects'].items():
    print(f"  {k}: {v:.3f}")""")

# ========== Visualization ==========
add_markdown("""## Part 3: 探索性分析""")

add_code("""# 可视化因果路径
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('T对M的影响', 'M对Y的影响', 'T对Y的总效应')
)

# T → M
m_t1 = M[T==1]
m_t0 = M[T==0]

fig.add_trace(go.Box(y=m_t1, name='发券', marker_color=COLORS['success']), row=1, col=1)
fig.add_trace(go.Box(y=m_t0, name='不发券', marker_color=COLORS['danger']), row=1, col=1)

# M → Y (分层)
m_bins = pd.qcut(M, q=5, duplicates='drop')
y_by_m = pd.DataFrame({'M_bin': m_bins, 'Y': Y}).groupby('M_bin')['Y'].mean()

fig.add_trace(go.Bar(
    x=[str(b) for b in y_by_m.index],
    y=y_by_m.values,
    marker_color=COLORS['primary'],
    showlegend=False
), row=1, col=2)

# T → Y
y_rate_t1 = Y[T==1].mean()
y_rate_t0 = Y[T==0].mean()

fig.add_trace(go.Bar(
    x=['发券', '不发券'],
    y=[y_rate_t1, y_rate_t0],
    marker_color=[COLORS['success'], COLORS['danger']],
    text=[f'{y_rate_t1:.1%}', f'{y_rate_t0:.1%}'],
    textposition='outside',
    showlegend=False
), row=1, col=3)

fig.update_layout(height=400, template='plotly_white', showlegend=False)
fig.show()

print(f"\\n📊 观察:")
print(f"1. 发券组访问次数更多: {m_t1.mean():.2f} vs {m_t0.mean():.2f}")
print(f"2. 访问越多，购买率越高")
print(f"3. 发券组购买率更高: {y_rate_t1:.1%} vs {y_rate_t0:.1%}")
print(f"\\n❓ 问题: 发券的效应中，有多少是通过增加访问次数实现的？")""")

# ========== Baron-Kenny Method ==========
add_markdown("""## Part 4: Baron-Kenny 方法（传统方法）

### 方法步骤

**Step 1**: 回归 Y ~ T (总效应)
$$Y = \\alpha_1 + \\tau \\cdot T + \\epsilon_1$$

**Step 2**: 回归 M ~ T (T对M的效应)
$$M = \\alpha_2 + a \\cdot T + \\epsilon_2$$

**Step 3**: 回归 Y ~ T + M (直接效应和M的效应)
$$Y = \\alpha_3 + \\tau' \\cdot T + b \\cdot M + \\epsilon_3$$

**效应分解**：
- 间接效应: $IE = a \\times b$
- 直接效应: $DE = \\tau'$
- 总效应: $TE = \\tau = \\tau' + a \\times b$

### 局限性
- 假设线性关系
- 假设无交互作用
- 假设无混淆""")

add_code("""class BaronKennyMediation:
    \"\"\"Baron-Kenny 中介分析\"\"\"

    def __init__(self):
        self.model_total = None
        self.model_mediator = None
        self.model_direct = None

    def fit(self, T, M, Y, X=None):
        \"\"\"拟合三个回归模型\"\"\"
        # 准备特征
        if X is not None:
            T_X = np.column_stack([T, X])
            T_M_X = np.column_stack([T, M, X])
        else:
            T_X = T.reshape(-1, 1)
            T_M_X = np.column_stack([T, M])

        # Step 1: Y ~ T (+ X)
        self.model_total = LinearRegression()
        self.model_total.fit(T_X, Y)

        # Step 2: M ~ T (+ X)
        self.model_mediator = LinearRegression()
        self.model_mediator.fit(T_X, M)

        # Step 3: Y ~ T + M (+ X)
        self.model_direct = LinearRegression()
        self.model_direct.fit(T_M_X, Y)

        return self

    def get_effects(self):
        \"\"\"计算效应\"\"\"
        # 系数
        tau = self.model_total.coef_[0]  # 总效应
        a = self.model_mediator.coef_[0]  # T → M
        tau_prime = self.model_direct.coef_[0]  # 直接效应
        b = self.model_direct.coef_[1]  # M → Y

        # 间接效应
        indirect = a * b

        return {
            'total': tau,
            'direct': tau_prime,
            'indirect': indirect,
            'proportion_mediated': indirect / tau if tau != 0 else 0
        }

# 应用 Baron-Kenny
bk = BaronKennyMediation()
bk.fit(T, M, Y, X)
bk_effects = bk.get_effects()

print("Baron-Kenny 中介分析结果")
print("="*60)
for key, val in bk_effects.items():
    if key == 'proportion_mediated':
        print(f"{key}: {val:.1%}")
    else:
        print(f"{key}: {val:.4f}")

print(f"\\n与真实值对比:")
print(f"  直接效应: {bk_effects['direct']:.3f} vs {data['true_effects']['direct']:.3f}")
print(f"  间接效应: {bk_effects['indirect']:.3f} vs {data['true_effects']['indirect']:.3f}")""")

# ========== Mathematical Derivation ==========
add_markdown("""## Part 5: 数学推导

### 直接效应和间接效应的定义

#### 自然直接效应 (Natural Direct Effect, NDE)

**定义**：固定中介变量在控制组的水平，处理对结果的效应

$$NDE = E[Y(T=1, M(T=0)) - Y(T=0, M(T=0))]$$

**直觉**：如果给处理组，但保持中介在控制组水平，结果会如何变化？

#### 自然间接效应 (Natural Indirect Effect, NIE)

**定义**：固定处理在控制组，中介变化对结果的效应

$$NIE = E[Y(T=0, M(T=1)) - Y(T=0, M(T=0))]$$

**直觉**：如果不给处理，但中介变成处理组水平，结果会如何变化？

### 识别公式（Pearl推导）

在顺序忽略性假设下：

**NDE**:
$$NDE = \\sum_m E[Y|T=1, M=m] \\cdot P(M=m|T=0) - E[Y|T=0]$$

**NIE**:
$$NIE = \\sum_m E[Y|T=0, M=m] \\cdot [P(M=m|T=1) - P(M=m|T=0)]$$

### 线性情况下的简化

如果：
- $M = \\alpha_M + a \\cdot T + \\epsilon_M$
- $Y = \\alpha_Y + \\tau' \\cdot T + b \\cdot M + \\epsilon_Y$

则：
- $DE = \\tau'$
- $IE = a \\times b$
- $TE = \\tau' + a \\times b$

---""")

# ========== Causal Mediation Analysis ==========
add_markdown("""## Part 6: 因果中介分析（从零实现）

实现完整的因果中介框架，处理非线性关系和交互效应。""")

add_code("""class CausalMediationAnalysis:
    \"\"\"
    因果中介分析

    实现 Imai, Keele, Tingley (2010) 的框架
    \"\"\"

    def __init__(self, mediator_model=None, outcome_model=None):
        self.mediator_model = mediator_model or LinearRegression()
        self.outcome_model = outcome_model or LinearRegression()

    def fit(self, T, M, Y, X=None):
        \"\"\"
        拟合中介模型和结果模型

        M ~ T + X
        Y ~ T + M + T*M + X
        \"\"\"
        n = len(T)

        # 准备特征
        if X is not None:
            X_with_T = np.column_stack([T, X])
            X_with_T_M = np.column_stack([T, M, T*M, X])  # 包含交互项
        else:
            X_with_T = T.reshape(-1, 1)
            X_with_T_M = np.column_stack([T, M, T*M])

        # 拟合 M ~ T + X
        self.mediator_model.fit(X_with_T, M)

        # 拟合 Y ~ T + M + T*M + X
        self.outcome_model.fit(X_with_T_M, Y)

        self.T = T
        self.M = M
        self.Y = Y
        self.X = X

        return self

    def predict_mediator(self, T, X=None):
        \"\"\"预测中介变量\"\"\"
        if X is not None:
            X_pred = np.column_stack([T, X])
        else:
            X_pred = T.reshape(-1, 1)
        return self.mediator_model.predict(X_pred)

    def predict_outcome(self, T, M, X=None):
        \"\"\"预测结果变量\"\"\"
        if X is not None:
            X_pred = np.column_stack([T, M, T*M, X])
        else:
            X_pred = np.column_stack([T, M, T*M])
        return self.outcome_model.predict(X_pred)

    def estimate_effects(self, n_samples=None):
        \"\"\"
        估计因果中介效应

        使用模拟方法（参数化g-formula）
        \"\"\"
        if n_samples is None:
            n_samples = len(self.T)

        # 使用观测数据的协变量
        if self.X is not None:
            X_sim = self.X
        else:
            X_sim = None

        n = len(X_sim) if X_sim is not None else n_samples

        # Y(1, M(1))
        T1 = np.ones(n)
        M1 = self.predict_mediator(T1, X_sim)
        Y_1_M1 = self.predict_outcome(T1, M1, X_sim)

        # Y(0, M(0))
        T0 = np.zeros(n)
        M0 = self.predict_mediator(T0, X_sim)
        Y_0_M0 = self.predict_outcome(T0, M0, X_sim)

        # Y(1, M(0)) - NDE
        Y_1_M0 = self.predict_outcome(T1, M0, X_sim)

        # Y(0, M(1)) - NIE
        Y_0_M1 = self.predict_outcome(T0, M1, X_sim)

        # 计算效应
        total_effect = np.mean(Y_1_M1 - Y_0_M0)
        nde = np.mean(Y_1_M0 - Y_0_M0)
        nie = np.mean(Y_0_M1 - Y_0_M0)

        return {
            'total': total_effect,
            'direct': nde,
            'indirect': nie,
            'proportion_mediated': nie / total_effect if total_effect != 0 else 0
        }

# 应用因果中介分析
cma = CausalMediationAnalysis()
cma.fit(T, M, Y, X)
cma_effects = cma.estimate_effects()

print("因果中介分析结果")
print("="*60)
for key, val in cma_effects.items():
    if key == 'proportion_mediated':
        print(f"{key}: {val:.1%}")
    else:
        print(f"{key}: {val:.4f}")

print(f"\\n与真实值对比:")
print(f"  直接效应: {cma_effects['direct']:.3f} vs {data['true_effects']['direct']:.3f}")
print(f"  间接效应: {cma_effects['indirect']:.3f} vs {data['true_effects']['indirect']:.3f}")""")

# ========== TODO and Interview Questions ==========
add_markdown("""## 思考题与练习

### 基础理解

1. **直接效应和间接效应的区别是什么？用生活例子解释。**

2. **为什么说Baron-Kenny方法有局限性？什么情况下会失效？**

3. **因果中介分析需要哪些识别假设？哪个最难验证？**

### 深入分析

4. **如果中介变量M和结果变量Y都是二元的，应该如何修改模型？**

5. **如果存在多个中介变量，如何分析？**

6. **中介效应的置信区间如何计算？（提示：Bootstrap）**

### 面试题

**题目1**：某公司测试新UI设计对用户留存的影响。发现新UI提升了15%留存率。

追问：
- 如何判断这个效应是否通过「用户满意度」中介？
- 需要收集什么数据？
- 如何设计分析流程？

**题目2**：编码题 - 实现Bootstrap置信区间

```python
def bootstrap_mediation_ci(T, M, Y, X=None, n_bootstrap=1000, alpha=0.05):
    \"\"\"
    计算中介效应的Bootstrap置信区间

    参数:
        T, M, Y, X: 数据
        n_bootstrap: Bootstrap次数
        alpha: 显著性水平

    返回:
        各效应的置信区间
    \"\"\"
    # TODO: 实现这个函数
    pass
```

---""")

# ========== Summary ==========
add_markdown("""## 总结

### 核心要点

| 概念 | 定义 | 公式 |
|------|------|------|
| **总效应** | 处理对结果的总影响 | $TE = E[Y(1) - Y(0)]$ |
| **直接效应** | 不经过中介的效应 | $NDE = E[Y(1,M(0)) - Y(0,M(0))]$ |
| **间接效应** | 通过中介的效应 | $NIE = E[Y(0,M(1)) - Y(0,M(0))]$ |
| **中介比例** | 间接效应占总效应的比例 | $PM = NIE / TE$ |

### 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Baron-Kenny** | 简单直观 | 线性假设强 | 连续线性关系 |
| **因果中介分析** | 灵活，允许非线性和交互 | 需要更多假设 | 一般场景 |
| **敏感性分析** | 评估未观测混淆影响 | 计算复杂 | 高风险决策 |

### 实践建议

1. **画因果图**：明确T→M→Y的路径
2. **检查假设**：顺序忽略性最关键
3. **敏感性分析**：测试未观测混淆的影响
4. **业务解释**：将统计结果转化为可操作建议

### 延伸阅读

- **经典论文**：
  - Baron & Kenny (1986): "The Moderator-Mediator Variable Distinction"
  - Imai, Keele & Tingley (2010): "A General Approach to Causal Mediation Analysis"
  - Pearl (2001): "Direct and Indirect Effects"

- **Python工具**：
  - `mediation` package
  - `causalml` 中的中介分析模块

---

**恭喜完成中介分析的学习！** 🎉

你现在可以：
- ✅ 分解因果效应为直接和间接部分
- ✅ 理解\"怎么起效\"而不仅是\"有没有效\"
- ✅ 应用于真实业务场景做机制分析
""")

# 保存notebook
output_path = "/Users/zhangjunmengyang/PycharmProjects/awesome-casual-inference/notebooks/part7_advanced/part7_4_mediation_analysis.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"✅ Notebook created: {output_path}")
