"""
Causal Forest - 因果森林

基于 Wager & Athey (2018) 的因果森林方法，用于估计异质性处理效应。

核心思想:
- 修改的随机森林，专门用于估计 CATE
- 诚实分裂 (Honest Splitting): 训练和估计使用不同数据
- 自适应 Neighborhood: 使用树结构定义相似样本
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

try:
    from econml.dml import CausalForestDML
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

from .utils import (
    generate_heterogeneous_data,
    compute_pehe,
    compute_ate_bias,
    identify_subgroups,
    compute_r_squared
)


class SimpleCausalForest:
    """
    简化版因果森林 (基于 T-Learner)

    当 econml 不可用时的备选方案。
    注意: 这不是真正的因果森林，只是一个简单的近似。
    """

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model_0 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.model_1 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + 1
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        mask_0 = T == 0
        mask_1 = T == 1

        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        Y0 = self.model_0.predict(X)
        Y1 = self.model_1.predict(X)
        return Y1 - Y0

    def feature_importances_(self) -> np.ndarray:
        """特征重要性 (平均两个模型)"""
        return (self.model_0.feature_importances_ + self.model_1.feature_importances_) / 2


def compare_causal_forest_vs_tlearner(
    n_samples: int,
    effect_heterogeneity: str,
    confounding_strength: float,
    n_trees: int
) -> tuple:
    """比较因果森林与 T-Learner 的性能"""

    # 生成数据
    df, true_cate, Y0_true, Y1_true = generate_heterogeneous_data(
        n_samples=n_samples,
        n_features=8,
        effect_heterogeneity=effect_heterogeneity,
        confounding_strength=confounding_strength,
        noise_level=0.5
    )

    X = df[[f'X{i+1}' for i in range(8)]].values
    T = df['T'].values
    Y = df['Y'].values

    # 分割数据
    train_idx = np.random.choice(len(X), int(0.7 * len(X)), replace=False)
    test_idx = np.array([i for i in range(len(X)) if i not in train_idx])

    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_test, T_test, Y_test = X[test_idx], T[test_idx], Y[test_idx]
    true_cate_test = true_cate[test_idx]
    Y0_test, Y1_test = Y0_true[test_idx], Y1_true[test_idx]

    # 训练模型
    models = {}

    # T-Learner (作为基线)
    t_learner = SimpleCausalForest(n_estimators=n_trees)
    t_learner.fit(X_train, T_train, Y_train)
    cate_tlearner = t_learner.predict(X_test)
    models['T-Learner'] = cate_tlearner

    # Causal Forest
    if ECONML_AVAILABLE:
        try:
            # 使用 CausalForest
            cf = CausalForest(
                n_estimators=n_trees,
                min_samples_leaf=10,
                max_depth=None,
                random_state=42
            )
            cf.fit(X_train, T_train, Y_train)
            cate_cf = cf.predict(X_test).flatten()
            models['Causal Forest'] = cate_cf
        except Exception as e:
            print(f"CausalForest failed: {e}, using T-Learner only")
    else:
        # 如果没有 econml，使用改进的 T-Learner
        print("econml not available, using improved T-Learner")

    # 计算指标
    metrics_data = []
    for name, cate_pred in models.items():
        # 需要构造 Y0_pred, Y1_pred 用于 PEHE
        # 简化: 假设 CATE_pred = Y1_pred - Y0_pred, 且 Y0_pred 接近真实 Y0
        Y0_pred = Y0_test  # 简化假设
        Y1_pred = Y0_pred + cate_pred

        pehe = compute_pehe(Y0_test, Y1_test, Y0_pred, Y1_pred)
        ate_bias = compute_ate_bias(Y0_test, Y1_test, Y0_pred, Y1_pred)
        r2 = compute_r_squared(true_cate_test, cate_pred)

        metrics_data.append({
            'Model': name,
            'PEHE': f'{pehe:.4f}',
            'ATE Bias': f'{ate_bias:.4f}',
            'R²': f'{r2:.4f}'
        })

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'CATE Distribution',
            'True vs Predicted CATE',
            'Feature Importance',
            'CATE by Subgroups'
        )
    )

    colors = {'T-Learner': '#2D9CDB', 'Causal Forest': '#27AE60'}

    # 1. CATE 分布
    fig.add_trace(go.Histogram(
        x=true_cate_test, name='True CATE',
        marker_color='gray', opacity=0.7, nbinsx=30
    ), row=1, col=1)

    for name, cate_pred in models.items():
        fig.add_trace(go.Histogram(
            x=cate_pred, name=name,
            marker_color=colors.get(name, '#EB5757'),
            opacity=0.5, nbinsx=30
        ), row=1, col=1)

    # 2. 真实 vs 预测
    sample_idx = np.random.choice(len(true_cate_test), min(500, len(true_cate_test)), replace=False)

    for name, cate_pred in models.items():
        fig.add_trace(go.Scatter(
            x=true_cate_test[sample_idx], y=cate_pred[sample_idx],
            mode='markers',
            marker=dict(color=colors.get(name, '#EB5757'), size=5, opacity=0.6),
            name=f'{name} (R²={compute_r_squared(true_cate_test, cate_pred):.3f})'
        ), row=1, col=2)

    # 对角线
    min_val, max_val = true_cate_test.min(), true_cate_test.max()
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=1, col=2)

    # 3. 特征重要性
    feature_names = [f'X{i+1}' for i in range(8)]

    if 'Causal Forest' in models and ECONML_AVAILABLE:
        # Causal Forest 的特征重要性
        try:
            importances = cf.feature_importances_
            fig.add_trace(go.Bar(
                x=feature_names, y=importances,
                name='CF Importance',
                marker_color='#27AE60'
            ), row=2, col=1)
        except:
            # Fallback
            importances = t_learner.feature_importances_()
            fig.add_trace(go.Bar(
                x=feature_names, y=importances,
                name='T-Learner Importance',
                marker_color='#2D9CDB'
            ), row=2, col=1)
    else:
        importances = t_learner.feature_importances_()
        fig.add_trace(go.Bar(
            x=feature_names, y=importances,
            name='T-Learner Importance',
            marker_color='#2D9CDB'
        ), row=2, col=1)

    # 4. 子群体 CATE
    # 使用最佳模型
    best_model_name = list(models.keys())[-1]  # 优先使用 Causal Forest
    best_cate = models[best_model_name]

    groups = identify_subgroups(X_test, true_cate_test, n_groups=4)
    group_names = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']

    true_by_group = [true_cate_test[groups == i].mean() for i in range(4)]
    pred_by_group = [best_cate[groups == i].mean() for i in range(4)]

    fig.add_trace(go.Bar(
        x=group_names, y=true_by_group,
        name='True CATE', marker_color='gray', opacity=0.7
    ), row=2, col=2)

    fig.add_trace(go.Bar(
        x=group_names, y=pred_by_group,
        name=f'{best_model_name}',
        marker_color=colors.get(best_model_name, '#27AE60'),
        opacity=0.7
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Causal Forest vs T-Learner Comparison',
        barmode='overlay'
    )

    # 指标表格
    metrics_df = pd.DataFrame(metrics_data)
    metrics_md = f"""
### 模型性能对比

{metrics_df.to_markdown(index=False)}

### 关键指标说明

- **PEHE**: Precision in Estimation of HTE (异质性处理效应估计精度)
  - sqrt(E[(ITE_true - ITE_pred)²])
  - 越小越好，衡量个体处理效应估计的准确性

- **ATE Bias**: 平均处理效应偏差
  - |E[ITE_true] - E[ITE_pred]|
  - 越小越好，衡量总体效应估计的偏差

- **R²**: 决定系数
  - 1 - SSE/SST
  - 越接近 1 越好，衡量预测 CATE 的拟合优度

### Causal Forest 优势

1. **诚实分裂**: 训练和估计使用不同数据，减少过拟合
2. **自适应邻域**: 使用树结构定义相似样本
3. **渐近正态性**: 提供理论保证的置信区间
4. **处理混淆**: 对混淆的鲁棒性更好

### 实验设置

- 样本量: {n_samples}
- 效应异质性: {effect_heterogeneity}
- 混淆强度: {confounding_strength}
- 树的数量: {n_trees}
    """

    return fig, metrics_md


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## Causal Forest - 因果森林

因果森林是由 Wager & Athey (2018) 提出的强大的异质性处理效应估计方法。

### 核心思想

**标准随机森林的问题**:
- 用于预测 Y，不是专门用于估计处理效应
- 可能对处理变量 T 过拟合
- 没有理论保证

**因果森林的改进**:

1. **诚实分裂 (Honest Splitting)**:
   ```
   样本 → [分裂样本] 用于构建树结构
         ↓
       [估计样本] 用于叶节点的 CATE 估计
   ```
   这防止了过拟合，提供了更可靠的估计。

2. **自适应邻域**:
   - 树结构定义了样本的"邻域"
   - 落在同一叶节点的样本被认为相似
   - CATE 在叶节点内局部估计

3. **渐近正态性**:
   - 在一定条件下，CATE 估计渐近正态
   - 可以构造有效的置信区间

### 算法流程

```
对于每棵树:
  1. 随机抽样 (bootstrap)
  2. 分裂样本: 用于决定分裂位置
  3. 估计样本: 用于计算叶节点 CATE
  4. 在叶节点 L 中:
     CATE(x) = avg(Y_i | T_i=1, i∈L) - avg(Y_i | T_i=0, i∈L)

最终预测: 所有树的平均
```

---
        """)

        # 检查 econml 是否可用
        if not ECONML_AVAILABLE:
            gr.Markdown("""
### ⚠️ 注意

`econml` 库未安装。本模块将使用简化版 T-Learner 作为近似。

安装 econml 以使用真正的因果森林:
```bash
pip install econml
```
            """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=1000, maximum=10000, value=3000, step=500,
                    label="样本量"
                )
                effect_heterogeneity = gr.Radio(
                    choices=['weak', 'moderate', 'strong'],
                    value='moderate',
                    label="效应异质性强度"
                )
                confounding_strength = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.1,
                    label="混淆强度"
                )
                n_trees = gr.Slider(
                    minimum=50, maximum=500, value=100, step=50,
                    label="树的数量"
                )
                run_btn = gr.Button("运行对比", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="Causal Forest 对比")

        with gr.Row():
            metrics_output = gr.Markdown()

        run_btn.click(
            fn=compare_causal_forest_vs_tlearner,
            inputs=[n_samples, effect_heterogeneity, confounding_strength, n_trees],
            outputs=[plot_output, metrics_output]
        )

        gr.Markdown("""
---

### 数学细节

**叶节点的 CATE 估计**:

对于叶节点 L(x) (包含样本 x 的叶节点):

$$\\hat{\\tau}(x) = \\frac{1}{|\\{i: X_i \\in L(x), T_i=1\\}|} \\sum_{i: X_i \\in L(x), T_i=1} Y_i - \\frac{1}{|\\{i: X_i \\in L(x), T_i=0\\}|} \\sum_{i: X_i \\in L(x), T_i=0} Y_i$$

**渐近性质**:

在正则性条件下:
$$\\sqrt{n}(\\hat{\\tau}(x) - \\tau(x)) \\xrightarrow{d} N(0, \\sigma^2(x))$$

这允许我们构造置信区间:
$$\\hat{\\tau}(x) \\pm z_{\\alpha/2} \\cdot \\hat{\\sigma}(x) / \\sqrt{n}$$

### 与其他方法的对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| **T-Learner** | 简单，灵活 | 样本效率低，无理论保证 |
| **X-Learner** | 小样本下好 | 依赖倾向得分估计 |
| **Causal Forest** | 理论保证，自适应 | 计算较慢，需要大样本 |

### 应用场景

- 精准营销: 识别对促销敏感的客户
- 医疗决策: 个性化治疗方案
- 政策评估: 识别政策受益人群

### 参考文献

- Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests". JASA.
- Athey, Tibshirani & Wager (2019). "Generalized Random Forests". Annals of Statistics.

### 练习

完成 `exercises/chapter5_hetero_effect/ex1_causal_forest.py` 中的练习。
        """)

    return None
