"""
Part 4 CATE & Uplift API 适配层

将核心业务逻辑转换为返回标准格式的 API 函数。

返回格式: {"charts": [...], "tables": [], "summary": "...", "metrics": {...}}
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any

from .meta_learners import SLearner, TLearner, XLearner, RLearner, DRLearner
from .causal_forest import get_causal_forest_model
from .uplift_tree import SimpleUpliftTree, find_best_split, calculate_uplift_gain
from .uplift_evaluation import calculate_qini_curve, calculate_uplift_curve, calculate_auuc, calculate_cumulative_gain
from .cate_visualization import identify_subgroups, compute_subgroup_statistics, compute_cate_distribution_stats
from .utils import (
    generate_heterogeneous_data,
    generate_uplift_data,
    compute_pehe,
    compute_r_squared,
)


def _fig_to_chart_data(fig: go.Figure) -> dict:
    """将 Plotly Figure 转换为前端可用的图表数据"""
    return fig.to_dict()


def analyze_meta_learners(
    n_samples: int = 5000,
    effect_type: str = 'moderate',
    noise_level: float = 0.5,
    confounding_strength: float = 0.3
) -> dict:
    """
    分析比较不同的 Meta-Learners

    Parameters:
    -----------
    n_samples: 样本量
    effect_type: 效应异质性类型 (weak/moderate/strong)
    noise_level: 噪声水平
    confounding_strength: 混淆强度

    Returns:
    --------
    标准 API 响应格式
    """
    # 生成数据
    df, true_cate, Y0_true, Y1_true = generate_heterogeneous_data(
        n_samples=n_samples,
        n_features=5,
        effect_heterogeneity=effect_type,
        confounding_strength=confounding_strength,
        noise_level=noise_level,
        seed=42
    )

    X = df[[f'X{i+1}' for i in range(5)]].values
    T = df['T'].values
    Y = df['Y'].values

    # 训练各个 Meta-Learner
    models = {
        'S-Learner': SLearner(),
        'T-Learner': TLearner(),
        'X-Learner': XLearner(),
        'R-Learner': RLearner(),
        'DR-Learner': DRLearner(),
    }

    predictions = {}
    metrics = {}

    for name, model in models.items():
        try:
            model.fit(X, T, Y)
            cate_pred = model.predict(X)
            predictions[name] = cate_pred

            # 计算指标
            pehe = compute_pehe(Y0_true, Y1_true, Y0_true, Y0_true + cate_pred)
            r2 = compute_r_squared(true_cate, cate_pred)
            correlation = np.corrcoef(true_cate, cate_pred)[0, 1]

            metrics[name] = {
                'pehe': float(pehe),
                'r_squared': float(r2),
                'correlation': float(correlation),
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
            predictions[name] = np.zeros(n_samples)
            metrics[name] = {'pehe': np.nan, 'r_squared': np.nan, 'correlation': np.nan}

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'CATE Distribution Comparison',
            'True vs Predicted CATE',
            'Model Performance (PEHE)',
            'Correlation with True CATE'
        ),
        specs=[
            [{'type': 'histogram'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ]
    )

    colors = ['#2D9CDB', '#27AE60', '#EB5757', '#9B59B6', '#F2994A']

    # 1. CATE 分布
    fig.add_trace(go.Histogram(
        x=true_cate,
        name='True CATE',
        marker_color='gray',
        opacity=0.5,
        nbinsx=30
    ), row=1, col=1)

    for i, (name, pred) in enumerate(predictions.items()):
        fig.add_trace(go.Histogram(
            x=pred,
            name=name,
            marker_color=colors[i % len(colors)],
            opacity=0.5,
            nbinsx=30
        ), row=1, col=1)

    # 2. True vs Predicted (使用第一个模型作为示例)
    for i, (name, pred) in enumerate(predictions.items()):
        sample_idx = np.random.choice(n_samples, min(500, n_samples), replace=False)
        fig.add_trace(go.Scatter(
            x=true_cate[sample_idx],
            y=pred[sample_idx],
            mode='markers',
            name=name,
            marker=dict(color=colors[i % len(colors)], size=3, opacity=0.5),
            showlegend=False
        ), row=1, col=2)

    # 对角线
    min_val = true_cate.min()
    max_val = true_cate.max()
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect',
        showlegend=False
    ), row=1, col=2)

    # 3. PEHE 对比
    pehe_values = [metrics[name]['pehe'] for name in models.keys()]
    fig.add_trace(go.Bar(
        x=list(models.keys()),
        y=pehe_values,
        marker_color=colors[:len(models)],
        name='PEHE',
        showlegend=False
    ), row=2, col=1)

    # 4. Correlation 对比
    corr_values = [metrics[name]['correlation'] for name in models.keys()]
    fig.add_trace(go.Bar(
        x=list(models.keys()),
        y=corr_values,
        marker_color=colors[:len(models)],
        name='Correlation',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Meta-Learners Comparison',
        barmode='overlay'
    )

    fig.update_xaxes(title_text='CATE', row=1, col=1)
    fig.update_xaxes(title_text='True CATE', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Predicted CATE', row=1, col=2)
    fig.update_yaxes(title_text='PEHE (lower is better)', row=2, col=1)
    fig.update_yaxes(title_text='Correlation', row=2, col=2)

    # 构建性能表格
    performance_df = pd.DataFrame(metrics).T
    performance_df = performance_df.reset_index()
    performance_df.columns = ['Model', 'PEHE', 'R²', 'Correlation']

    summary = f"""
## Meta-Learners 比较分析

### 数据设置
- 样本量: {n_samples}
- 效应异质性: {effect_type}
- 噪声水平: {noise_level}
- 混淆强度: {confounding_strength}

### 模型概述

1. **S-Learner**: 单一模型，将处理作为特征
2. **T-Learner**: 两个独立模型，分别建模处理组和控制组
3. **X-Learner**: 两阶段方法，利用反事实估计
4. **R-Learner**: 基于残差的双重稳健方法
5. **DR-Learner**: 双重稳健 Meta-Learner

### 性能指标

- **PEHE**: 异质性效应估计精度 (越小越好)
- **R²**: 决定系数 (越接近 1 越好)
- **Correlation**: 与真实 CATE 的相关性 (越接近 1 越好)

### 最佳模型

{performance_df.loc[performance_df['PEHE'].idxmin(), 'Model']} 在 PEHE 指标上表现最优。
"""

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [performance_df.to_dict('records')],
        'summary': summary,
        'metrics': {
            'true_ate': float(true_cate.mean()),
            'models': metrics
        }
    }


def analyze_causal_forest(
    n_samples: int = 5000,
    effect_heterogeneity: str = 'moderate',
    confounding_strength: float = 0.3,
    n_trees: int = 100
) -> dict:
    """
    分析因果森林

    Parameters:
    -----------
    n_samples: 样本量
    effect_heterogeneity: 效应异质性 (weak/moderate/strong)
    confounding_strength: 混淆强度
    n_trees: 树的数量

    Returns:
    --------
    标准 API 响应格式
    """
    # 生成数据
    df, true_cate, Y0_true, Y1_true = generate_heterogeneous_data(
        n_samples=n_samples,
        n_features=5,
        effect_heterogeneity=effect_heterogeneity,
        confounding_strength=confounding_strength,
        noise_level=0.5,
        seed=42
    )

    X = df[[f'X{i+1}' for i in range(5)]].values
    T = df['T'].values
    Y = df['Y'].values

    # 训练因果森林
    model = get_causal_forest_model(n_trees=n_trees, min_samples_leaf=10, random_state=42)
    model.fit(X, T, Y)
    cate_pred = model.predict(X)

    # 计算指标
    pehe = compute_pehe(Y0_true, Y1_true, Y0_true, Y0_true + cate_pred)
    r2 = compute_r_squared(true_cate, cate_pred)
    correlation = np.corrcoef(true_cate, cate_pred)[0, 1]

    # 识别子群体
    subgroups = identify_subgroups(X, cate_pred, n_groups=4, method='quantile')
    subgroup_stats = compute_subgroup_statistics(Y, T, cate_pred, subgroups)

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'True vs Predicted CATE',
            'CATE by Subgroup',
            'Feature Importance',
            'CATE Distribution'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'box'}],
            [{'type': 'bar'}, {'type': 'histogram'}]
        ]
    )

    # 1. True vs Predicted
    sample_idx = np.random.choice(n_samples, min(500, n_samples), replace=False)
    fig.add_trace(go.Scatter(
        x=true_cate[sample_idx],
        y=cate_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=4, opacity=0.5),
        name='Predictions',
        showlegend=False
    ), row=1, col=1)

    # 对角线
    min_val = min(true_cate.min(), cate_pred.min())
    max_val = max(true_cate.max(), cate_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect',
        showlegend=False
    ), row=1, col=1)

    # 2. CATE by Subgroup
    for group in range(4):
        mask = subgroups == group
        fig.add_trace(go.Box(
            y=cate_pred[mask],
            name=f'Group {group+1}',
            marker_color=['#2D9CDB', '#27AE60', '#EB5757', '#9B59B6'][group]
        ), row=1, col=2)

    # 3. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_()
    else:
        importances = np.ones(5) / 5  # 均匀分布

    fig.add_trace(go.Bar(
        x=[f'X{i+1}' for i in range(5)],
        y=importances,
        marker_color='#27AE60',
        name='Importance',
        showlegend=False
    ), row=2, col=1)

    # 4. CATE Distribution
    fig.add_trace(go.Histogram(
        x=true_cate,
        name='True CATE',
        marker_color='gray',
        opacity=0.5,
        nbinsx=30
    ), row=2, col=2)

    fig.add_trace(go.Histogram(
        x=cate_pred,
        name='Predicted CATE',
        marker_color='#2D9CDB',
        opacity=0.5,
        nbinsx=30
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Causal Forest Analysis',
        barmode='overlay'
    )

    fig.update_xaxes(title_text='True CATE', row=1, col=1)
    fig.update_yaxes(title_text='Predicted CATE', row=1, col=1)
    fig.update_yaxes(title_text='CATE', row=1, col=2)
    fig.update_yaxes(title_text='Importance', row=2, col=1)
    fig.update_xaxes(title_text='CATE', row=2, col=2)

    summary = f"""
## 因果森林分析

### 模型设置
- 样本量: {n_samples}
- 树的数量: {n_trees}
- 效应异质性: {effect_heterogeneity}

### 性能指标
- PEHE: {pehe:.4f} (越小越好)
- R²: {r2:.4f}
- Correlation: {correlation:.4f}

### 子群体分析

将样本按预测 CATE 分为 4 个子群体:
- Group 1: 最低 CATE (可能是 sleeping dogs)
- Group 2-3: 中等 CATE
- Group 4: 最高 CATE (persuadables)

### 关键洞察

因果森林通过诚实分裂和自适应邻域等技术，能够较好地估计异质性处理效应。
"""

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [subgroup_stats.to_dict('records')],
        'summary': summary,
        'metrics': {
            'pehe': float(pehe),
            'r_squared': float(r2),
            'correlation': float(correlation),
            'true_ate': float(true_cate.mean()),
            'pred_ate': float(cate_pred.mean()),
        }
    }


def analyze_uplift_tree(
    n_samples: int = 5000,
    feature_effect: str = 'heterogeneous',
    criterion: str = 'KL'
) -> dict:
    """
    分析 Uplift Tree

    Parameters:
    -----------
    n_samples: 样本量
    feature_effect: 特征效应类型
    criterion: 分裂准则 (KL/ED/Chi/DDP)

    Returns:
    --------
    标准 API 响应格式
    """
    # 生成数据
    df, true_uplift = generate_uplift_data(
        n_samples=n_samples,
        n_features=5,
        treatment_effect_type=feature_effect,
        noise_level=0.5,
        seed=42
    )

    X = df[[f'X{i+1}' for i in range(5)]].values
    T = df['T'].values
    Y = df['Y'].values

    # 训练 Uplift Tree
    model = SimpleUpliftTree(
        criterion=criterion,
        max_depth=3,
        min_samples_leaf=100,
        min_samples_split=200
    )
    model.fit(X, T, Y)
    uplift_pred = model.predict(X)

    # 计算指标
    correlation = np.corrcoef(true_uplift, uplift_pred)[0, 1]
    rmse = np.sqrt(np.mean((true_uplift - uplift_pred) ** 2))

    # 分析最佳分裂
    split_analysis = []
    for i in range(5):
        threshold, gain, left_uplift, right_uplift = find_best_split(
            X, Y, T, i, criterion, min_samples_leaf=100
        )
        split_analysis.append({
            'feature': f'X{i+1}',
            'threshold': threshold if threshold is not None else np.nan,
            'gain': gain,
            'left_uplift': left_uplift,
            'right_uplift': right_uplift,
            'uplift_diff': abs(left_uplift - right_uplift)
        })

    split_df = pd.DataFrame(split_analysis)

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'True vs Predicted Uplift',
            'Uplift Distribution',
            'Split Gain by Feature',
            'Uplift Difference by Split'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'histogram'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ]
    )

    # 1. True vs Predicted
    sample_idx = np.random.choice(n_samples, min(500, n_samples), replace=False)
    fig.add_trace(go.Scatter(
        x=true_uplift[sample_idx],
        y=uplift_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=4, opacity=0.5),
        name='Predictions',
        showlegend=False
    ), row=1, col=1)

    # 对角线
    min_val = min(true_uplift.min(), uplift_pred.min())
    max_val = max(true_uplift.max(), uplift_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=1, col=1)

    # 2. Uplift Distribution
    fig.add_trace(go.Histogram(
        x=true_uplift,
        name='True Uplift',
        marker_color='gray',
        opacity=0.5,
        nbinsx=30
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=uplift_pred,
        name='Predicted Uplift',
        marker_color='#2D9CDB',
        opacity=0.5,
        nbinsx=30
    ), row=1, col=2)

    # 3. Split Gain
    fig.add_trace(go.Bar(
        x=split_df['feature'],
        y=split_df['gain'],
        marker_color='#27AE60',
        name='Gain',
        showlegend=False
    ), row=2, col=1)

    # 4. Uplift Difference
    fig.add_trace(go.Bar(
        x=split_df['feature'],
        y=split_df['uplift_diff'],
        marker_color='#EB5757',
        name='Uplift Diff',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text=f'Uplift Tree Analysis (Criterion: {criterion})',
        barmode='overlay'
    )

    fig.update_xaxes(title_text='True Uplift', row=1, col=1)
    fig.update_yaxes(title_text='Predicted Uplift', row=1, col=1)
    fig.update_xaxes(title_text='Uplift', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)

    summary = f"""
## Uplift Tree 分析

### 模型设置
- 样本量: {n_samples}
- 分裂准则: {criterion}
- 最大深度: 3
- 特征效应: {feature_effect}

### 分裂准则说明
- **KL**: KL 散度，衡量处理组和控制组分布的差异
- **ED**: 欧氏距离
- **Chi**: 卡方统计量
- **DDP**: Delta-Delta P (差异的差异)

### 性能指标
- Correlation: {correlation:.4f}
- RMSE: {rmse:.4f}

### 最佳分裂特征

{split_df.loc[split_df['gain'].idxmax(), 'feature']} 具有最高的分裂增益。
"""

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [split_df.to_dict('records')],
        'summary': summary,
        'metrics': {
            'correlation': float(correlation),
            'rmse': float(rmse),
            'true_ate': float(true_uplift.mean()),
            'pred_ate': float(uplift_pred.mean()),
        }
    }


def analyze_uplift_evaluation(
    n_samples: int = 5000,
    model_quality: str = 'good'
) -> dict:
    """
    分析 Uplift 评估方法

    Parameters:
    -----------
    n_samples: 样本量
    model_quality: 模型质量 (perfect/good/random)

    Returns:
    --------
    标准 API 响应格式
    """
    # 生成数据
    df, true_uplift = generate_uplift_data(
        n_samples=n_samples,
        n_features=5,
        treatment_effect_type='heterogeneous',
        noise_level=0.5,
        seed=42
    )

    X = df[[f'X{i+1}' for i in range(5)]].values
    T = df['T'].values
    Y = df['Y'].values

    # 生成不同质量的预测
    if model_quality == 'perfect':
        uplift_score = true_uplift
    elif model_quality == 'good':
        # 添加一些噪声
        uplift_score = true_uplift + np.random.randn(n_samples) * 0.5
    else:  # random
        uplift_score = np.random.randn(n_samples)

    # 计算 Qini 和 Uplift 曲线
    fraction_qini, qini = calculate_qini_curve(Y, T, uplift_score)
    fraction_uplift, cumulative_uplift = calculate_uplift_curve(Y, T, uplift_score)
    auuc = calculate_auuc(Y, T, uplift_score, normalize=False)

    # 随机基线
    random_score = np.random.randn(n_samples)
    fraction_random, qini_random = calculate_qini_curve(Y, T, random_score)
    auuc_random = calculate_auuc(Y, T, random_score, normalize=False)

    # 累积增益表
    gain_df = calculate_cumulative_gain(Y, T, uplift_score, n_bins=10)

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Qini Curve',
            'Uplift Curve',
            'Cumulative Gain by Decile',
            'AUUC Comparison'
        )
    )

    # 1. Qini Curve
    fig.add_trace(go.Scatter(
        x=fraction_qini, y=qini,
        mode='lines',
        name=f'Model (AUUC={auuc:.2f})',
        line=dict(color='#2D9CDB', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fraction_random, y=qini_random,
        mode='lines',
        name=f'Random (AUUC={auuc_random:.2f})',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)

    # 2. Uplift Curve
    fig.add_trace(go.Scatter(
        x=fraction_uplift, y=cumulative_uplift,
        mode='lines',
        name='Model',
        line=dict(color='#27AE60', width=2),
        showlegend=False
    ), row=1, col=2)

    # 3. Cumulative Gain
    fig.add_trace(go.Bar(
        x=gain_df['bin'],
        y=gain_df['cumulative_uplift'],
        marker_color='#EB5757',
        name='Cumulative Uplift',
        showlegend=False
    ), row=2, col=1)

    # 4. AUUC Comparison
    fig.add_trace(go.Bar(
        x=['Model', 'Random'],
        y=[auuc, auuc_random],
        marker_color=['#2D9CDB', 'gray'],
        name='AUUC',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Uplift Evaluation Methods'
    )

    fig.update_xaxes(title_text='Fraction Targeted', row=1, col=1)
    fig.update_yaxes(title_text='Qini Value', row=1, col=1)
    fig.update_xaxes(title_text='Fraction Targeted', row=1, col=2)
    fig.update_yaxes(title_text='Cumulative Uplift', row=1, col=2)
    fig.update_xaxes(title_text='Decile', row=2, col=1)
    fig.update_yaxes(title_text='Cumulative Uplift', row=2, col=1)
    fig.update_yaxes(title_text='AUUC', row=2, col=2)

    summary = f"""
## Uplift 评估方法分析

### 数据设置
- 样本量: {n_samples}
- 模型质量: {model_quality}

### 评估指标

1. **Qini 曲线**: 衡量按 uplift 得分排序后的累积增量收益
   - 曲线下面积 (AUUC): {auuc:.4f}
   - 随机基线 AUUC: {auuc_random:.4f}

2. **Uplift 曲线**: 展示每个分位数的平均 uplift

3. **累积增益**: 按 decile 分组的累积 uplift

### 关键洞察

- AUUC 越大，模型的 uplift 排序能力越好
- Qini 曲线越凸向左上角，模型越优
- 累积增益图展示了逐步扩大目标人群的边际收益
"""

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [gain_df.to_dict('records')],
        'summary': summary,
        'metrics': {
            'auuc': float(auuc),
            'auuc_random': float(auuc_random),
            'auuc_lift': float(auuc - auuc_random),
        }
    }


def visualize_cate(
    n_samples: int = 5000,
    effect_heterogeneity: str = 'moderate',
    n_bootstrap: int = 50,
    n_subgroups: int = 4
) -> dict:
    """
    CATE 可视化

    Parameters:
    -----------
    n_samples: 样本量
    effect_heterogeneity: 效应异质性
    n_bootstrap: Bootstrap 次数
    n_subgroups: 子群体数量

    Returns:
    --------
    标准 API 响应格式
    """
    # 生成数据
    df, true_cate, Y0_true, Y1_true = generate_heterogeneous_data(
        n_samples=n_samples,
        n_features=5,
        effect_heterogeneity=effect_heterogeneity,
        confounding_strength=0.3,
        noise_level=0.5,
        seed=42
    )

    X = df[[f'X{i+1}' for i in range(5)]].values
    T = df['T'].values
    Y = df['Y'].values

    # 训练 T-Learner
    from .meta_learners import TLearner
    model = TLearner()
    model.fit(X, T, Y)
    cate_pred = model.predict(X)

    # 识别子群体
    subgroups = identify_subgroups(X, cate_pred, n_groups=n_subgroups, method='quantile')
    subgroup_stats = compute_subgroup_statistics(Y, T, cate_pred, subgroups)

    # CATE 分布统计
    dist_stats = compute_cate_distribution_stats(true_cate, cate_pred)

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'CATE by Subgroup',
            'True vs Predicted CATE',
            'CATE Distribution',
            'Observed Uplift by Subgroup'
        ),
        specs=[
            [{'type': 'box'}, {'type': 'scatter'}],
            [{'type': 'histogram'}, {'type': 'bar'}]
        ]
    )

    colors = ['#2D9CDB', '#27AE60', '#EB5757', '#9B59B6']

    # 1. CATE by Subgroup
    for group in range(n_subgroups):
        mask = subgroups == group
        fig.add_trace(go.Box(
            y=cate_pred[mask],
            name=f'Group {group+1}',
            marker_color=colors[group % len(colors)]
        ), row=1, col=1)

    # 2. True vs Predicted
    sample_idx = np.random.choice(n_samples, min(500, n_samples), replace=False)
    fig.add_trace(go.Scatter(
        x=true_cate[sample_idx],
        y=cate_pred[sample_idx],
        mode='markers',
        marker=dict(color='#2D9CDB', size=4, opacity=0.5),
        name='Predictions',
        showlegend=False
    ), row=1, col=2)

    # 对角线
    min_val = min(true_cate.min(), cate_pred.min())
    max_val = max(true_cate.max(), cate_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=1, col=2)

    # 3. CATE Distribution
    fig.add_trace(go.Histogram(
        x=true_cate,
        name='True CATE',
        marker_color='gray',
        opacity=0.5,
        nbinsx=30
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=cate_pred,
        name='Predicted CATE',
        marker_color='#2D9CDB',
        opacity=0.5,
        nbinsx=30
    ), row=2, col=1)

    # 4. Observed Uplift by Subgroup
    fig.add_trace(go.Bar(
        x=subgroup_stats['group'] + 1,
        y=subgroup_stats['observed_uplift'],
        marker_color=colors[:n_subgroups],
        name='Observed Uplift',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='CATE Visualization',
        barmode='overlay'
    )

    fig.update_yaxes(title_text='Predicted CATE', row=1, col=1)
    fig.update_xaxes(title_text='True CATE', row=1, col=2)
    fig.update_yaxes(title_text='Predicted CATE', row=1, col=2)
    fig.update_xaxes(title_text='CATE', row=2, col=1)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    fig.update_xaxes(title_text='Subgroup', row=2, col=2)
    fig.update_yaxes(title_text='Observed Uplift', row=2, col=2)

    summary = f"""
## CATE 可视化分析

### 数据设置
- 样本量: {n_samples}
- 效应异质性: {effect_heterogeneity}
- 子群体数量: {n_subgroups}

### 性能指标
- Correlation: {dist_stats['correlation']:.4f}
- R²: {dist_stats['r_squared']:.4f}
- RMSE: {dist_stats['rmse']:.4f}

### 子群体分析

将样本按预测 CATE 分为 {n_subgroups} 个子群体:
- Group 1: 最低 CATE
- Group {n_subgroups}: 最高 CATE

每个子群体的观测 uplift 与预测 CATE 应该一致。

### 关键洞察

通过子群体分析，可以识别出对处理响应不同的人群，
从而实现精准化的处理策略。
"""

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [subgroup_stats.to_dict('records')],
        'summary': summary,
        'metrics': dist_stats
    }
