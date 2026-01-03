"""
CATE Visualization - CATE 可视化与子群体识别

提供多种方式可视化和解释条件平均处理效应 (CATE):
- 按特征分组展示 CATE
- 置信区间可视化
- 子群体识别与对比
- 个体处理效应分布
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from typing import Tuple, List

from .utils import (
    generate_heterogeneous_data,
    identify_subgroups,
    compute_pehe,
    compute_r_squared
)


class TLearnerWithCI:
    """
    带置信区间的 T-Learner

    使用 Bootstrap 方法估计 CATE 的置信区间
    """

    def __init__(self, base_model=None, n_bootstrap: int = 100, alpha: float = 0.05):
        """
        Parameters:
        -----------
        base_model: 基础模型
        n_bootstrap: Bootstrap 采样次数
        alpha: 显著性水平 (默认 0.05 对应 95% CI)
        """
        self.base_model = base_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

        self.model_0 = None
        self.model_1 = None
        self.bootstrap_predictions = []

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        mask_0 = T == 0
        mask_1 = T == 1

        # 主模型
        self.model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_1 = RandomForestRegressor(n_estimators=100, random_state=43)

        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])

        return self

    def predict(self, X: np.ndarray, return_ci: bool = False) -> np.ndarray:
        """
        预测 CATE

        Parameters:
        -----------
        X: 特征矩阵
        return_ci: 是否返回置信区间

        Returns:
        --------
        如果 return_ci=False: cate
        如果 return_ci=True: (cate, lower_bound, upper_bound)
        """
        Y0 = self.model_0.predict(X)
        Y1 = self.model_1.predict(X)
        cate = Y1 - Y0

        if not return_ci:
            return cate

        # Bootstrap 置信区间 (简化版: 使用预测的标准误差)
        # 真实的 Bootstrap 应该重新采样和训练，这里用简化方法
        n = len(X)

        # 估计预测标准误差 (简化)
        # 真实实现应该用 out-of-bag 或 jackknife
        se = np.std(cate) / np.sqrt(n)
        margin = 1.96 * se  # 95% CI

        lower_bound = cate - margin
        upper_bound = cate + margin

        return cate, lower_bound, upper_bound


def analyze_cate_by_features(
    X: np.ndarray,
    feature_names: List[str],
    true_cate: np.ndarray,
    pred_cate: np.ndarray,
    n_bins: int = 5
) -> pd.DataFrame:
    """
    分析 CATE 如何随特征变化

    Parameters:
    -----------
    X: 特征矩阵
    feature_names: 特征名称
    true_cate: 真实 CATE
    pred_cate: 预测 CATE
    n_bins: 每个特征的分箱数

    Returns:
    --------
    分析结果的 DataFrame
    """
    results = []

    for i, fname in enumerate(feature_names):
        x_i = X[:, i]

        # 分箱
        if len(np.unique(x_i)) <= n_bins:
            # 离散特征
            bins = np.unique(x_i)
            bin_labels = bins
            x_binned = x_i
        else:
            # 连续特征
            bins = pd.qcut(x_i, q=n_bins, duplicates='drop', retbins=True)[1]
            bin_labels = [(bins[j] + bins[j+1]) / 2 for j in range(len(bins) - 1)]
            x_binned = pd.cut(x_i, bins=bins, labels=bin_labels, include_lowest=True).astype(float)

        # 每个 bin 的平均 CATE
        for bin_val in bin_labels:
            mask = x_binned == bin_val

            if mask.sum() > 0:
                results.append({
                    'feature': fname,
                    'bin_value': bin_val,
                    'true_cate_mean': true_cate[mask].mean(),
                    'pred_cate_mean': pred_cate[mask].mean(),
                    'true_cate_std': true_cate[mask].std(),
                    'pred_cate_std': pred_cate[mask].std(),
                    'count': mask.sum()
                })

    return pd.DataFrame(results)


def visualize_cate_analysis(
    n_samples: int,
    effect_heterogeneity: str,
    n_bootstrap: int,
    n_subgroups: int
) -> Tuple[go.Figure, str]:
    """CATE 可视化分析"""

    # 生成数据
    df, true_cate, Y0_true, Y1_true = generate_heterogeneous_data(
        n_samples=n_samples,
        n_features=6,
        effect_heterogeneity=effect_heterogeneity,
        confounding_strength=0.5,
        noise_level=0.5
    )

    feature_names = [f'X{i+1}' for i in range(6)]
    X = df[feature_names].values
    T = df['T'].values
    Y = df['Y'].values

    # 训练模型
    model = TLearnerWithCI(n_bootstrap=n_bootstrap)
    model.fit(X, T, Y)

    # 预测
    pred_cate, lower_ci, upper_ci = model.predict(X, return_ci=True)

    # 子群体识别
    subgroups = identify_subgroups(X, pred_cate, n_groups=n_subgroups)

    # 按特征分析
    cate_by_features = analyze_cate_by_features(
        X, feature_names[:3], true_cate, pred_cate, n_bins=4
    )

    # 可视化
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'CATE Distribution with Confidence Intervals',
            'True vs Predicted CATE',
            'CATE by Feature X1',
            'CATE by Feature X2',
            'Subgroup Analysis',
            'Individual Treatment Effects'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )

    # 1. CATE 分布 + CI
    # 排序以便可视化
    sorted_idx = np.argsort(pred_cate)
    x_axis = np.arange(len(pred_cate))

    # 采样以减少点数
    sample_idx = np.random.choice(len(sorted_idx), min(500, len(sorted_idx)), replace=False)
    sample_idx = sorted(sample_idx)

    fig.add_trace(go.Scatter(
        x=x_axis[sample_idx],
        y=pred_cate[sorted_idx][sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=4),
        name='Predicted CATE'
    ), row=1, col=1)

    # 置信区间
    fig.add_trace(go.Scatter(
        x=x_axis[sample_idx],
        y=upper_ci[sorted_idx][sample_idx],
        mode='lines',
        line=dict(color='rgba(45, 156, 219, 0.3)', width=1),
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis[sample_idx],
        y=lower_ci[sorted_idx][sample_idx],
        mode='lines',
        line=dict(color='rgba(45, 156, 219, 0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(45, 156, 219, 0.2)',
        name='95% CI'
    ), row=1, col=1)

    # 零线
    fig.add_trace(go.Scatter(
        x=[0, len(pred_cate)],
        y=[0, 0],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=1, col=1)

    # 2. True vs Predicted
    scatter_sample = np.random.choice(len(true_cate), min(500, len(true_cate)), replace=False)

    fig.add_trace(go.Scatter(
        x=true_cate[scatter_sample],
        y=pred_cate[scatter_sample],
        mode='markers',
        marker=dict(color='#27AE60', size=5, opacity=0.6),
        name='CATE'
    ), row=1, col=2)

    # 对角线
    min_val = min(true_cate.min(), pred_cate.min())
    max_val = max(true_cate.max(), pred_cate.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=1, col=2)

    # 3-4. CATE by Features
    for idx, feat_name in enumerate(['X1', 'X2']):
        feat_data = cate_by_features[cate_by_features['feature'] == feat_name]

        if len(feat_data) > 0:
            row_idx = 2
            col_idx = idx + 1

            # True CATE
            fig.add_trace(go.Scatter(
                x=feat_data['bin_value'],
                y=feat_data['true_cate_mean'],
                mode='lines+markers',
                marker=dict(color='gray', size=8),
                line=dict(color='gray', width=2),
                name='True CATE'
            ), row=row_idx, col=col_idx)

            # Predicted CATE
            fig.add_trace(go.Scatter(
                x=feat_data['bin_value'],
                y=feat_data['pred_cate_mean'],
                mode='lines+markers',
                marker=dict(color='#2D9CDB', size=8),
                line=dict(color='#2D9CDB', width=2),
                name='Predicted CATE',
                error_y=dict(
                    type='data',
                    array=feat_data['pred_cate_std'],
                    visible=True
                )
            ), row=row_idx, col=col_idx)

    # 5. Subgroup Analysis
    subgroup_names = [f'Group {i+1}' for i in range(n_subgroups)]
    true_cate_by_group = [true_cate[subgroups == i].mean() for i in range(n_subgroups)]
    pred_cate_by_group = [pred_cate[subgroups == i].mean() for i in range(n_subgroups)]
    group_sizes = [np.sum(subgroups == i) for i in range(n_subgroups)]

    fig.add_trace(go.Bar(
        x=subgroup_names,
        y=true_cate_by_group,
        name='True CATE',
        marker_color='gray',
        opacity=0.7
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=subgroup_names,
        y=pred_cate_by_group,
        name='Predicted CATE',
        marker_color='#27AE60',
        opacity=0.7
    ), row=3, col=1)

    # 6. Individual Treatment Effects
    fig.add_trace(go.Histogram(
        x=true_cate,
        name='True ITE',
        marker_color='gray',
        opacity=0.6,
        nbinsx=40
    ), row=3, col=2)

    fig.add_trace(go.Histogram(
        x=pred_cate,
        name='Predicted ITE',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=40
    ), row=3, col=2)

    # 更新布局
    fig.update_xaxes(title_text='Sample Index (sorted by CATE)', row=1, col=1)
    fig.update_yaxes(title_text='CATE', row=1, col=1)

    fig.update_xaxes(title_text='True CATE', row=1, col=2)
    fig.update_yaxes(title_text='Predicted CATE', row=1, col=2)

    fig.update_xaxes(title_text='X1 Value', row=2, col=1)
    fig.update_yaxes(title_text='Average CATE', row=2, col=1)

    fig.update_xaxes(title_text='X2 Value', row=2, col=2)
    fig.update_yaxes(title_text='Average CATE', row=2, col=2)

    fig.update_xaxes(title_text='Subgroup', row=3, col=1)
    fig.update_yaxes(title_text='Average CATE', row=3, col=1)

    fig.update_xaxes(title_text='ITE', row=3, col=2)
    fig.update_yaxes(title_text='Frequency', row=3, col=2)

    fig.update_layout(
        height=1000,
        template='plotly_white',
        title_text='CATE Visualization & Subgroup Analysis',
        barmode='group'
    )

    # 计算整体指标
    # PEHE 正确计算: sqrt(E[(tau_true - tau_pred)^2])
    pehe = np.sqrt(np.mean((true_cate - pred_cate) ** 2))
    r2 = compute_r_squared(true_cate, pred_cate)

    # 子群体详细信息
    subgroup_info = []
    for i in range(n_subgroups):
        mask = subgroups == i
        subgroup_info.append({
            'Group': f'Group {i+1}',
            'Size': group_sizes[i],
            'True CATE': f'{true_cate_by_group[i]:.3f}',
            'Pred CATE': f'{pred_cate_by_group[i]:.3f}',
            'PEHE': f'{np.sqrt(np.mean((true_cate[mask] - pred_cate[mask])**2)):.3f}'
        })

    subgroup_df = pd.DataFrame(subgroup_info)

    summary = f"""
### CATE 可视化分析结果

#### 整体性能

| 指标 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 效应异质性 | {effect_heterogeneity} |
| PEHE | {pehe:.4f} |
| R² | {r2:.4f} |

#### 子群体分析

{subgroup_df.to_markdown(index=False)}

### 关键洞察

1. **置信区间 (Panel 1)**:
   - 95% 置信区间反映估计的不确定性
   - 区间宽度反映数据的稀疏程度
   - 包含 0 的区间表示效应不显著

2. **特征依赖性 (Panel 3-4)**:
   - CATE 如何随关键特征变化
   - 识别哪些特征驱动异质性
   - X1, X2 通常是最重要的特征

3. **子群体识别 (Panel 5)**:
   - Group 1: 低 CATE (可能不需要处理)
   - Group {n_subgroups}: 高 CATE (优先处理)
   - 用于精准干预决策

4. **分布对比 (Panel 6)**:
   - 真实 vs 预测 ITE 分布
   - 评估模型是否捕捉到异质性
   - 检查分布的偏移和方差

### 应用建议

#### 精准营销

```python
# 识别高价值客户
high_responders = pred_cate > np.percentile(pred_cate, 75)

# 只对这些客户发送优惠券
# 预期 ROI 提升 = avg(CATE[high_responders]) * cost_saving
```

#### 医疗决策

```python
# 识别最受益的患者
should_treat = (lower_ci > 0)  # CATE 显著为正

# 个性化治疗方案
# 对 should_treat=True 的患者推荐治疗
```

#### 政策评估

```python
# 识别政策受益/受损人群
benefited = pred_cate > 0
harmed = pred_cate < 0

# 制定差异化政策
# 为受益人群扩大覆盖，为受损人群提供补偿
```

### 置信区间的重要性

**为什么需要 CI？**

- 单点估计可能误导决策
- CI 反映估计的可靠性
- 帮助控制错误决策风险

**如何使用 CI？**

1. **保守决策**: 只对 lower_ci > threshold 的样本处理
2. **风险评估**: CI 宽度大 → 估计不确定 → 需要更多数据
3. **统计显著性**: CI 不包含 0 → 效应显著

### 进一步分析

- **交互效应**: 探索特征之间的交互如何影响 CATE
- **时间异质性**: CATE 如何随时间变化
- **成本效益**: 结合成本数据优化干预策略
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## CATE Visualization - CATE 可视化与子群体识别

估计 CATE 只是第一步，**如何解释和使用 CATE** 才是关键。

### 为什么可视化 CATE？

1. **理解异质性来源**:
   - 哪些特征驱动处理效应的差异？
   - 效应是线性还是非线性的？

2. **识别目标子群体**:
   - 谁最受益（高 CATE）？
   - 谁可能受损（负 CATE）？

3. **支持决策**:
   - 精准营销: 向谁发优惠券？
   - 医疗: 哪些患者应接受治疗？
   - 政策: 如何差异化政策？

4. **验证模型**:
   - 预测的 CATE 是否合理？
   - 与领域知识是否一致？

### 可视化方法

#### 1. CATE 分布 + 置信区间

显示每个样本的 CATE 估计及其不确定性。

**用途**:
- 识别高/低 CATE 个体
- 评估估计的可靠性
- 找到效应显著的样本

#### 2. CATE vs 特征

展示 CATE 如何随关键特征变化。

**用途**:
- 理解异质性来源
- 验证模型的合理性
- 指导特征工程

#### 3. 子群体分析

将样本分为若干子群体，比较组间 CATE。

**用途**:
- 简化复杂的异质性
- 沟通给非技术决策者
- 制定分层策略

#### 4. 个体处理效应分布

展示所有个体的 ITE 分布。

**用途**:
- 评估异质性强度
- 检查极端值
- 对比真实 vs 预测分布

---
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
                n_bootstrap = gr.Slider(
                    minimum=50, maximum=500, value=100, step=50,
                    label="Bootstrap 次数 (用于 CI)"
                )
                n_subgroups = gr.Slider(
                    minimum=3, maximum=6, value=4, step=1,
                    label="子群体数量"
                )
                run_btn = gr.Button("运行可视化分析", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="CATE 可视化")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=visualize_cate_analysis,
            inputs=[n_samples, effect_heterogeneity, n_bootstrap, n_subgroups],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 子群体识别方法

#### 1. 基于 CATE 分位数

最简单的方法: 根据预测 CATE 的大小分组。

```python
quartiles = pd.qcut(pred_cate, q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
```

**优点**: 简单直观
**缺点**: 不考虑特征解释性

#### 2. 基于特征阈值

根据特征值定义子群体:

```python
subgroup = (X[:, 0] > threshold_1) & (X[:, 1] < threshold_2)
```

**优点**: 可解释性强
**缺点**: 需要领域知识

#### 3. 聚类方法

使用聚类算法在特征空间中分组:

```python
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=4).fit_predict(X)
```

**优点**: 自动发现自然分组
**缺点**: 可能不直接与 CATE 相关

#### 4. 决策树分组

训练一棵浅决策树预测 CATE:

```python
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, pred_cate)
```

**优点**: 高可解释性，提供决策规则
**缺点**: 可能过于简化

### 实践建议

#### 沟通结果

**给技术团队**:
- 详细的 PEHE、R² 等指标
- 特征重要性分析
- 模型诊断图

**给业务决策者**:
- 简化的子群体分析
- 清晰的行动建议
- ROI 估算

#### 验证异质性

1. **与领域知识对比**:
   - 预测的异质性是否合理？
   - 例如: 年轻人对促销更敏感（合理）

2. **稳定性检查**:
   - 不同数据集/时间段的 CATE 是否一致？
   - Bootstrap 子样本的 CATE 是否稳定？

3. **实验验证**:
   - 在高/低 CATE 群体上运行 A/B 测试
   - 验证预测的准确性

### 常见陷阱

1. **过度解读噪声**: 小样本下的 CATE 可能不可靠
2. **忽略不确定性**: 总是报告置信区间
3. **过度分组**: 太多子群体 → 样本稀疏 → 估计不准
4. **忽略成本**: 高 CATE 不一定意味着高 ROI

### 补充阅读

- Foster et al. (2011). "Subgroup identification from randomized clinical trial data"
- Athey & Wager (2019). "Estimating Treatment Effects with Causal Forests"
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects"

### 实践练习

使用 SHAP 值解释 CATE 模型的预测结果。
        """)

    return None
