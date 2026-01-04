#!/usr/bin/env python3
"""
创建 Part 7.6: Bunching 估计 Notebook
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
add_markdown("""# Part 7.6: Bunching 估计 (Bunching Estimation)

## 学习目标

1. 理解 Bunching 现象及其经济学含义
2. 掌握 Bunching 估计的基本原理
3. 学习反事实分布的构造方法
4. 应用于税收、补贴等政策评估
5. 实现从零 Bunching 估计算法

---

## 什么是 Bunching？

### 生活中的例子

想象你是某打车平台的司机，平台有如下补贴政策：

**政策**：
- 每天完成 < 10 单：无补贴
- 每天完成 ≥ 10 单：每单奖励 5 元

**你会怎么做？**
- 如果已经完成 9 单，很可能会多接 1 单达到 10 单门槛
- 如果已经完成 10 单，可能不会刻意多接单

**结果**：在 10 单这个位置会出现 **聚集 (Bunching)**！

### 分布形状

```
频数
 |
 |            * *
 |          * * * *    ← Bunching!
 |        * * * * * *
 |      * * * * * * * *
 |    * * * * * * * * * * *
 |  * * * * * * * * * * * * *
 |_____________________________
   0  2  4  6  8  10 12 14 16
                ↑
              门槛点
```

---

## 业务场景

### 场景1：税收门槛

某国规定年收入低于10万免税，超过10万部分征税30%。

**问题**：有多少人为了避税，刻意将收入控制在10万以下？

### 场景2：平台补贴

外卖平台：月单量≥200单的骑手，次月获得1000元奖励。

**问题**：有多少骑手为了拿奖励，在月末冲单？

### 场景3：考试及格线

某课程60分及格，59分需要重修。

**问题**：老师是否在59-60分之间给分更宽松（手下留情）？

---""")

# ========== Setup ==========
add_code("""# 环境准备
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
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

### Bunching 的形成机制

**关键要素**：
1. **Notch（门槛）**：某个政策在特定点发生跳跃
2. **最优化行为**：个体会调整行为以最大化收益
3. **调整成本**：调整行为有成本，不是所有人都会调整

### 因果推断视角

**反事实问题**：如果没有这个门槛政策，分布会是什么样？

- **观测分布 $f(x)$**：在政策下的实际分布
- **反事实分布 $f_0(x)$**：无政策时的分布
- **Bunching量**：$B = \\int_{\\underline{x}}^{\\bar{x}} [f(x) - f_0(x)] dx$

### 识别策略

**核心思想**：
1. 门槛附近的 bunching 是政策导致的
2. 远离门槛的分布反映无政策情况
3. 用远离门槛的分布外推，估计门槛附近的反事实分布

---""")

# ========== Data Generation ==========
add_markdown("""## Part 2: 数据生成""")

add_code("""def generate_bunching_data(n=10000, threshold=10, subsidy=5, seed=42):
    \"\"\"
    生成 bunching 数据

    场景：打车平台订单量
    - threshold: 补贴门槛（10单）
    - subsidy: 超过门槛的每单奖励（5元）
    \"\"\"
    np.random.seed(seed)

    # 无政策时的 \"真实\" 订单量分布（反事实）
    # 假设服从 Poisson 分布（均值=8）
    orders_counterfactual = np.random.poisson(lam=8, size=n)

    # 个体调整成本（随机）
    # 成本越低，越容易调整到门槛
    adjustment_cost = np.random.gamma(shape=2, scale=2, size=n)

    # 决策：是否调整到门槛
    # 如果当前订单量在 threshold-3 到 threshold-1 之间
    # 且调整收益 > 调整成本，则调整
    orders_observed = orders_counterfactual.copy()

    for i in range(n):
        current_orders = orders_counterfactual[i]

        # 如果接近门槛但未达到
        if threshold - 3 <= current_orders < threshold:
            # 调整到门槛的收益
            extra_orders_needed = threshold - current_orders
            benefit = subsidy * threshold  # 达到门槛后的总奖励
            cost = adjustment_cost[i] * extra_orders_needed

            # 如果收益 > 成本，则调整
            if benefit > cost:
                orders_observed[i] = threshold

    return {
        'orders_observed': orders_observed,
        'orders_counterfactual': orders_counterfactual,
        'threshold': threshold,
        'subsidy': subsidy
    }

# 生成数据
data = generate_bunching_data()
orders_obs = data['orders_observed']
orders_cf = data['orders_counterfactual']
threshold = data['threshold']

print(f"数据维度: n={len(orders_obs)}")
print(f"门槛值: {threshold} 单")
print(f"观测分布均值: {orders_obs.mean():.2f}")
print(f"反事实分布均值: {orders_cf.mean():.2f}")
print(f"\\n门槛处的 bunching:")
print(f"  观测频数: {(orders_obs == threshold).sum()}")
print(f"  反事实频数: {(orders_cf == threshold).sum()}")
print(f"  差异: {(orders_obs == threshold).sum() - (orders_cf == threshold).sum()}")""")

# ========== Visualization ==========
add_markdown("""## Part 3: 可视化 Bunching 现象""")

add_code("""# 绘制分布对比
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('观测分布（有政策）', '反事实分布（无政策）')
)

# 计算直方图
bins = np.arange(0, 25, 1)
hist_obs, _ = np.histogram(orders_obs, bins=bins)
hist_cf, _ = np.histogram(orders_cf, bins=bins)

bin_centers = (bins[:-1] + bins[1:]) / 2

# 观测分布
fig.add_trace(go.Bar(
    x=bin_centers,
    y=hist_obs,
    marker_color=COLORS['primary'],
    name='观测分布',
    showlegend=False
), row=1, col=1)

# 标注门槛
fig.add_vline(x=threshold, line_dash='dash', line_color='red',
              annotation_text=f'门槛={threshold}', row=1, col=1)

# 反事实分布
fig.add_trace(go.Bar(
    x=bin_centers,
    y=hist_cf,
    marker_color=COLORS['success'],
    name='反事实分布',
    showlegend=False
), row=1, col=2)

fig.add_vline(x=threshold, line_dash='dash', line_color='red',
              annotation_text=f'门槛={threshold}', row=1, col=2)

fig.update_xaxes(title_text='订单量', row=1, col=1)
fig.update_xaxes(title_text='订单量', row=1, col=2)
fig.update_yaxes(title_text='频数', row=1, col=1)
fig.update_yaxes(title_text='频数', row=1, col=2)

fig.update_layout(height=400, template='plotly_white')
fig.show()

print("📊 观察: 观测分布在门槛处有明显的峰值（bunching）！")""")

# ========== Bunching Estimation ==========
add_markdown("""## Part 4: Bunching 估计算法

### 算法步骤

**Step 1**: 定义排除窗口
- 排除门槛附近的区域 $[\\underline{x}, \\bar{x}]$
- 这部分受政策影响最大

**Step 2**: 估计反事实分布
- 用排除窗口外的数据拟合平滑函数
- 常用：多项式回归、样条回归

**Step 3**: 外推到排除窗口
- 预测排除窗口内的反事实频数

**Step 4**: 计算 Bunching 量
$$B = \\sum_{x \\in [\\underline{x}, \\bar{x}]} (f(x) - \\hat{f}_0(x))$$

**Step 5**: 计算弹性（如适用）
$$\\epsilon = \\frac{B}{\\Delta \\tau} \\cdot \\frac{1}{x^*}$$

---""")

add_code("""class BunchingEstimator:
    \"\"\"
    Bunching 估计器

    实现 Chetty et al. (2011) 的方法
    \"\"\"

    def __init__(self, threshold, exclusion_width=2, poly_degree=5):
        \"\"\"
        参数:
            threshold: 门槛值
            exclusion_width: 排除窗口宽度（门槛左右各多少）
            poly_degree: 多项式阶数
        \"\"\"
        self.threshold = threshold
        self.exclusion_width = exclusion_width
        self.poly_degree = poly_degree
        self.counterfactual_model = None

    def fit(self, data, bins=None):
        \"\"\"
        拟合反事实分布

        参数:
            data: 观测数据（1维数组）
            bins: 用于直方图的bins
        \"\"\"
        if bins is None:
            bins = np.arange(data.min(), data.max()+1, 1)

        # 计算直方图
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        self.hist = hist
        self.bin_centers = bin_centers

        # 定义排除窗口
        lower_bound = self.threshold - self.exclusion_width
        upper_bound = self.threshold + self.exclusion_width

        # 排除窗口外的数据
        mask = (bin_centers < lower_bound) | (bin_centers > upper_bound)
        X_train = bin_centers[mask].reshape(-1, 1)
        y_train = hist[mask]

        # 拟合多项式
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        poly = PolynomialFeatures(degree=self.poly_degree)
        X_poly = poly.fit_transform(X_train)

        self.poly = poly
        self.counterfactual_model = LinearRegression()
        self.counterfactual_model.fit(X_poly, y_train)

        # 预测整个范围的反事实分布
        X_all = bin_centers.reshape(-1, 1)
        X_all_poly = poly.transform(X_all)
        self.counterfactual_dist = self.counterfactual_model.predict(X_all_poly)

        # 确保非负
        self.counterfactual_dist = np.maximum(self.counterfactual_dist, 0)

        return self

    def estimate_bunching(self):
        \"\"\"计算 bunching 量\"\"\"
        # 排除窗口
        lower_bound = self.threshold - self.exclusion_width
        upper_bound = self.threshold + self.exclusion_width
        mask = (self.bin_centers >= lower_bound) & (self.bin_centers <= upper_bound)

        # Bunching = 观测 - 反事实
        bunching = self.hist[mask].sum() - self.counterfactual_dist[mask].sum()

        return {
            'bunching': bunching,
            'fraction_bunching': bunching / len(self.hist) if len(self.hist) > 0 else 0,
            'threshold': self.threshold
        }

    def plot(self):
        \"\"\"可视化结果\"\"\"
        fig = go.Figure()

        # 观测分布
        fig.add_trace(go.Bar(
            x=self.bin_centers,
            y=self.hist,
            marker_color=COLORS['primary'],
            name='观测分布',
            opacity=0.7
        ))

        # 反事实分布
        fig.add_trace(go.Scatter(
            x=self.bin_centers,
            y=self.counterfactual_dist,
            mode='lines',
            line=dict(color=COLORS['danger'], width=3, dash='dash'),
            name='反事实分布（估计）'
        ))

        # 门槛线
        fig.add_vline(x=self.threshold, line_dash='dot', line_color='gray',
                      annotation_text=f'门槛={self.threshold}')

        # 排除窗口
        lower = self.threshold - self.exclusion_width
        upper = self.threshold + self.exclusion_width
        fig.add_vrect(x0=lower, x1=upper, fillcolor='yellow', opacity=0.2,
                      annotation_text='排除窗口', annotation_position='top left')

        fig.update_layout(
            title='Bunching 估计',
            xaxis_title='数值',
            yaxis_title='频数',
            template='plotly_white',
            height=500
        )

        return fig

# 应用 Bunching 估计
estimator = BunchingEstimator(threshold=10, exclusion_width=2, poly_degree=5)
estimator.fit(orders_obs)

bunching_results = estimator.estimate_bunching()

print("Bunching 估计结果")
print("="*60)
for key, val in bunching_results.items():
    if key == 'fraction_bunching':
        print(f"{key}: {val:.2%}")
    else:
        print(f"{key}: {val:.2f}")

# 可视化
fig = estimator.plot()
fig.show()

# 真实 bunching（已知反事实）
true_bunching = (orders_obs == threshold).sum() - (orders_cf == threshold).sum()
print(f"\\n真实 bunching: {true_bunching}")
print(f"估计 bunching: {bunching_results['bunching']:.0f}")
print(f"误差: {abs(bunching_results['bunching'] - true_bunching):.0f}")""")

# ========== Mathematical Derivation ==========
add_markdown("""## Part 5: 数学推导

### 反事实分布的识别

**假设**：

1. **平滑性**：无政策时，分布在门槛处连续且平滑
2. **局部性**：政策只影响门槛附近的小区域 $[\\underline{x}, \\bar{x}]$
3. **单调性**：个体只会向门槛单方向调整（不会越过门槛）

### Bunching 量的定义

$$B = \\int_{\\underline{x}}^{\\bar{x}} [f(x) - f_0(x)] dx$$

其中：
- $f(x)$: 观测密度
- $f_0(x)$: 反事实密度

### 弹性估计

对于税收问题，可以估计 **应税收入弹性 (Elasticity of Taxable Income, ETI)**：

$$\\epsilon = \\frac{dz/z}{d(1-\\tau)/(1-\\tau)}$$

近似：
$$\\epsilon \\approx \\frac{B}{h_0(z^*)} \\cdot \\frac{1}{\\Delta \\tau} \\cdot \\frac{1}{z^*}$$

其中：
- $h_0(z^*)$: 门槛处的反事实密度
- $\\Delta \\tau$: 税率跳跃
- $z^*$: 门槛值

---""")

# ========== Interview Questions ==========
add_markdown("""## Part 6: 思考题与面试题

### 基础理解

1. **Bunching 方法的核心假设是什么？哪个最关键？**

2. **为什么需要排除窗口？窗口大小如何选择？**

3. **Bunching 估计与 RDD 有什么区别？**

### 深入分析

4. **如果观测到门槛处的 bunching，能否推断政策一定有效？**
   - 提示：考虑其他可能的原因（如报告误差）

5. **如果 bunching 是负的（凹陷），说明什么？**
   - 提示：政策可能是惩罚性的

6. **如何检验反事实分布的拟合质量？**
   - 提示：Placebo test

### 面试编程题

**题目**：实现 Bootstrap 标准误

```python
def bootstrap_bunching_se(data, threshold, n_bootstrap=100):
    \"\"\"
    计算 bunching 估计的 Bootstrap 标准误

    参数:
        data: 观测数据
        threshold: 门槛值
        n_bootstrap: Bootstrap 次数

    返回:
        标准误和置信区间
    \"\"\"
    # TODO: 实现这个函数
    pass
```

---""")

# ========== Case Study ==========
add_markdown("""## Part 7: 案例分析

### 案例：美国 EITC 税收抵免

**背景**：
- EITC (Earned Income Tax Credit) 是美国的税收抵免政策
- 收入在特定范围内，可获得退税
- 在某些收入点存在 \"notch\"（门槛）

**研究问题**：
- 纳税人是否会调整收入以最大化 EITC？
- 调整行为有多普遍？

**Bunching 分析**：
1. 观察收入分布，在 EITC 门槛处是否有 bunching
2. 估计调整人数
3. 计算应税收入弹性

**发现**（Chetty et al. 2013）：
- 在第一个 kink point（约$8000）有显著 bunching
- 约 1-2% 的纳税人精确调整到该点
- 弹性约为 0.2-0.3

---""")

# ========== Summary ==========
add_markdown("""## 总结

### 核心要点

| 概念 | 定义 | 重要性 |
|------|------|--------|
| **Bunching** | 分布在门槛处的聚集 | 政策影响的证据 |
| **反事实分布** | 无政策时的分布 | 估计的基准 |
| **排除窗口** | 受政策影响的区域 | 需要排除以外推 |
| **弹性估计** | 行为对政策的敏感度 | 政策设计的关键 |

### 方法优势

✅ **不需要处理组和对照组**
✅ **利用分布的不连续性**
✅ **可以估计行为参数（如弹性）**
✅ **适用于普遍性政策**

### 方法局限

❌ **需要明确的门槛点**
❌ **假设分布平滑**
❌ **难以处理多重门槛**
❌ **对反事实拟合敏感**

### 应用场景

1. **税收政策评估**：收入税、消费税的 bunching
2. **社会福利项目**：福利门槛的行为响应
3. **平台补贴设计**：订单量、销售额门槛
4. **考试评分**：及格线附近的分数分布

### 扩展阅读

**经典论文**：
- Saez (2010): \"Do Taxpayers Bunch at Kink Points?\"
- Chetty et al. (2011): \"Adjustment Costs, Firm Responses, and Micro vs. Macro Labor Supply Elasticities\"
- Kleven & Waseem (2013): \"Using Notches to Uncover Optimization Frictions\"

**Python 实现**：
- `bunching` package

---

**恭喜完成 Bunching 估计的学习！** 🎉

你现在可以：
- ✅ 识别和可视化 bunching 现象
- ✅ 估计反事实分布
- ✅ 量化政策的行为效应
- ✅ 应用于真实政策评估

这是 Part 7 高级主题的最后一章，你已经掌握了因果推断的核心方法！""")

# 保存notebook
output_path = "/Users/zhangjunmengyang/PycharmProjects/awesome-casual-inference/notebooks/part7_advanced/part7_6_bunching.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"✅ Notebook created: {output_path}")
