"""Part 7 高级主题 API 适配层

将各个模块转换为统一的 API 响应格式
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from .utils import (
    generate_causal_discovery_data,
    generate_continuous_treatment_data,
    generate_time_varying_data,
    generate_mediation_data,
)
from .causal_discovery import (
    discover_causal_structure,
    evaluate_discovery_performance,
    compute_structural_hamming_distance,
)
from .continuous_treatment import (
    GeneralizedPropensityScore,
    DoseResponseEstimator,
    estimate_drf_spline,
    compute_marginal_effect,
    find_optimal_treatment,
)
from .time_varying_treatment import (
    estimate_time_varying_weights,
    estimate_msm,
    g_computation,
    compute_cumulative_effect,
)
from .mediation_analysis import (
    MediationAnalyzer,
    baron_kenny_test,
    sensitivity_analysis_mediation,
)


def _fig_to_chart_data(fig: go.Figure) -> dict:
    """将 Plotly Figure 转换为前端可用的图表数据"""
    return fig.to_dict()


def analyze_causal_discovery(
    n_samples: int = 1000,
    n_variables: int = 6,
    graph_type: str = "chain"
) -> dict:
    """
    因果发现分析

    Args:
        n_samples: 样本量
        n_variables: 变量数量
        graph_type: 图类型 ('chain', 'fork', 'collider', 'complex')

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成数据
    data, true_graph = generate_causal_discovery_data(
        n_samples=n_samples,
        n_variables=n_variables,
        graph_type=graph_type
    )

    # 运行因果发现算法
    discovery_result = discover_causal_structure(data, alpha=0.05, max_cond_size=2)

    # 提取真实边（转换格式）
    true_edges = []
    for edge in true_graph['edges']:
        if isinstance(edge, tuple):
            true_edges.append(edge)
        elif isinstance(edge, str) and '->' in edge:
            src, dst = edge.split(' -> ')
            true_edges.append((src.strip(), dst.strip()))

    # 评估性能
    performance = evaluate_discovery_performance(
        discovery_result['edges'],
        true_edges
    )

    shd = compute_structural_hamming_distance(
        discovery_result['edges'],
        true_edges
    )

    # 创建可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['真实因果图', '发现的因果图'],
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )

    # 绘制真实图
    _add_graph_to_subplot(fig, true_edges, data.columns, row=1, col=1, color='#27AE60')

    # 绘制发现的图
    _add_graph_to_subplot(fig, discovery_result['edges'], data.columns, row=1, col=2, color='#2D9CDB')

    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white',
        title_text='因果发现：PC 算法结果'
    )

    # 相关性矩阵
    corr_matrix = data.corr()
    corr_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    corr_fig.update_layout(
        title='变量相关性矩阵',
        height=400,
        template='plotly_white'
    )

    # 构建摘要
    summary = f"""
## 因果发现分析结果

### 数据信息
- 样本量: {n_samples}
- 变量数: {n_variables}
- 图类型: {graph_type}

### 发现性能
- **Precision**: {performance['precision']:.3f}
- **Recall**: {performance['recall']:.3f}
- **F1 Score**: {performance['f1']:.3f}
- **SHD**: {shd}

### 边的比较
- 真实边数: {len(true_edges)}
- 发现边数: {len(discovery_result['edges'])}
- 正确发现: {performance['true_positive']}
- 错误发现: {performance['false_positive']}
- 遗漏: {performance['false_negative']}

### 关键洞察
- PC 算法基于条件独立性检验识别因果结构
- F1 Score 反映了整体发现质量
- 需要足够样本量才能获得可靠结果
"""

    # 构建表格数据
    edge_table = pd.DataFrame({
        '真实边': [f"{e[0]} -> {e[1]}" for e in true_edges],
    })

    discovered_edge_strs = [f"{e[0]} -> {e[1]}" for e in discovery_result['edges']]
    if len(discovered_edge_strs) < len(true_edges):
        discovered_edge_strs.extend([''] * (len(true_edges) - len(discovered_edge_strs)))
    edge_table['发现的边'] = discovered_edge_strs[:len(true_edges)]

    return {
        'charts': [_fig_to_chart_data(fig), _fig_to_chart_data(corr_fig)],
        'tables': [edge_table.to_dict('records')],
        'summary': summary,
        'metrics': {
            'precision': float(performance['precision']),
            'recall': float(performance['recall']),
            'f1_score': float(performance['f1']),
            'shd': int(shd),
            'n_true_edges': len(true_edges),
            'n_discovered_edges': len(discovery_result['edges'])
        }
    }


def analyze_continuous_treatment(
    n_samples: int = 1000,
    treatment_distribution: str = "normal"
) -> dict:
    """
    连续处理效应分析

    Args:
        n_samples: 样本量
        treatment_distribution: 处理分布类型

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成数据
    data = generate_continuous_treatment_data(
        n_samples=n_samples,
        treatment_distribution=treatment_distribution
    )

    T = data['T'].values
    X = data[['X1', 'X2']].values
    Y = data['Y'].values

    # 估计广义倾向得分
    gps_estimator = GeneralizedPropensityScore()
    gps_estimator.fit(T, X)
    gps_values = gps_estimator.predict(T, X)

    # 估计剂量响应函数
    drf_estimator = DoseResponseEstimator(treatment_degree=2, gps_degree=2)
    drf_estimator.fit(T, X, Y)

    # 预测网格
    t_grid = np.linspace(T.min(), T.max(), 100)
    drf_estimates = drf_estimator.predict_drf(t_grid, X)

    # 真实 DRF: 100 + 3*t - 0.1*t^2
    true_drf = 100 + 3 * t_grid - 0.1 * t_grid**2

    # 样条估计
    drf_spline = estimate_drf_spline(T, Y, gps_values, t_grid, smoothing=1000)

    # 计算边际效应
    marginal_effect = compute_marginal_effect(t_grid, drf_estimates)

    # 找到最优处理
    optimal_t, max_outcome = find_optimal_treatment(t_grid, drf_estimates)

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '剂量响应函数估计',
            '边际效应曲线',
            'GPS 分布',
            '处理-结果散点图'
        ]
    )

    # 1. DRF 估计
    fig.add_trace(go.Scatter(
        x=t_grid, y=true_drf,
        mode='lines', name='真实 DRF',
        line=dict(color='black', width=3, dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_grid, y=drf_estimates,
        mode='lines', name='GPS 估计',
        line=dict(color='#2D9CDB', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_grid, y=drf_spline,
        mode='lines', name='样条估计',
        line=dict(color='#27AE60', width=2)
    ), row=1, col=1)

    # 标注最优点
    fig.add_trace(go.Scatter(
        x=[optimal_t], y=[max_outcome],
        mode='markers', name=f'最优点 (t={optimal_t:.1f})',
        marker=dict(size=12, color='#EB5757', symbol='star')
    ), row=1, col=1)

    # 2. 边际效应
    fig.add_trace(go.Scatter(
        x=t_grid, y=marginal_effect,
        mode='lines', name='边际效应',
        line=dict(color='#F2994A', width=2),
        showlegend=False
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dot", line_color="red", row=1, col=2)

    # 3. GPS 分布
    fig.add_trace(go.Histogram(
        x=gps_values, nbinsx=50,
        marker_color='#9B51E0',
        name='GPS',
        showlegend=False
    ), row=2, col=1)

    # 4. 散点图
    sample_idx = np.random.choice(len(T), min(500, len(T)), replace=False)
    fig.add_trace(go.Scatter(
        x=T[sample_idx], y=Y[sample_idx],
        mode='markers',
        marker=dict(size=3, color='#2D9CDB', opacity=0.5),
        name='观测数据',
        showlegend=False
    ), row=2, col=2)

    fig.update_xaxes(title_text="处理剂量 (T)", row=1, col=1)
    fig.update_xaxes(title_text="处理剂量 (T)", row=1, col=2)
    fig.update_xaxes(title_text="GPS 值", row=2, col=1)
    fig.update_xaxes(title_text="处理剂量 (T)", row=2, col=2)

    fig.update_yaxes(title_text="期望结果 E[Y(t)]", row=1, col=1)
    fig.update_yaxes(title_text="dE[Y]/dt", row=1, col=2)
    fig.update_yaxes(title_text="频数", row=2, col=1)
    fig.update_yaxes(title_text="结果 (Y)", row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='连续处理效应分析',
        showlegend=True
    )

    # 计算 MSE
    mse_gps = np.mean((drf_estimates - true_drf) ** 2)
    mse_spline = np.mean((drf_spline - true_drf) ** 2)

    summary = f"""
## 连续处理效应分析结果

### 数据信息
- 样本量: {n_samples}
- 处理分布: {treatment_distribution}
- 处理范围: [{T.min():.2f}, {T.max():.2f}]

### 剂量响应函数
- **最优处理水平**: {optimal_t:.2f}
- **最大期望结果**: {max_outcome:.2f}
- GPS 方法 MSE: {mse_gps:.4f}
- 样条方法 MSE: {mse_spline:.4f}

### 关键发现
1. **剂量响应关系**: 二次函数形式，存在最优点
2. **边际效应**: 随处理剂量递减，在 t≈15 处变为负
3. **最优策略**: 处理剂量约为 {optimal_t:.1f} 时结果最优

### 方法说明
- 使用广义倾向得分（GPS）调整混淆
- 同时提供参数和非参数估计
- 边际效应帮助理解剂量变化的影响
"""

    # 构建表格
    summary_table = pd.DataFrame({
        '处理水平': t_grid[::10],
        '估计结果': drf_estimates[::10],
        '真实结果': true_drf[::10],
        '边际效应': marginal_effect[::10]
    })

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [summary_table.to_dict('records')],
        'summary': summary,
        'metrics': {
            'optimal_treatment': float(optimal_t),
            'max_outcome': float(max_outcome),
            'mse_gps': float(mse_gps),
            'mse_spline': float(mse_spline),
            'treatment_range_min': float(T.min()),
            'treatment_range_max': float(T.max())
        }
    }


def analyze_time_varying_treatment(
    n_periods: int = 5,
    treatment_pattern: str = "random"
) -> dict:
    """
    时变处理效应分析

    Args:
        n_periods: 时间周期数
        treatment_pattern: 处理模式

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成数据
    data = generate_time_varying_data(
        n_subjects=500,
        n_periods=n_periods,
        treatment_pattern=treatment_pattern
    )

    # 估计 IPW 权重
    weights = estimate_time_varying_weights(
        data,
        treatment_col='T',
        covariate_cols=['X'],
        stabilized=True
    )

    # 估计 MSM
    msm_result = estimate_msm(
        data,
        outcome_col='Y',
        treatment_col='T',
        weights=weights
    )

    # G-computation
    g_always_treat = g_computation(
        data,
        outcome_col='Y',
        treatment_col='T',
        covariate_cols=['X'],
        intervention='always_treat'
    )

    g_never_treat = g_computation(
        data,
        outcome_col='Y',
        treatment_col='T',
        covariate_cols=['X'],
        intervention='never_treat'
    )

    # 累积效应
    cumulative_summary = compute_cumulative_effect(data, 'T', 'Y')

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '边际结构模型（MSM）',
            '累积处理效应',
            'G-computation 对比',
            '权重分布'
        ]
    )

    # 1. MSM
    fig.add_trace(go.Scatter(
        x=msm_result['cumulative_range'],
        y=msm_result['predicted_outcomes'],
        mode='lines+markers',
        name='MSM 预测',
        line=dict(color='#2D9CDB', width=3)
    ), row=1, col=1)

    # 2. 累积效应
    fig.add_trace(go.Scatter(
        x=cumulative_summary['cumulative_treatment'],
        y=cumulative_summary['mean'],
        mode='markers',
        marker=dict(size=10, color='#27AE60'),
        error_y=dict(
            type='data',
            array=cumulative_summary['std'],
            visible=True
        ),
        name='观测均值',
        showlegend=False
    ), row=1, col=2)

    # 3. G-computation
    interventions = ['Always Treat', 'Never Treat', 'Natural']
    outcomes = [
        g_always_treat['mean_outcome'],
        g_never_treat['mean_outcome'],
        g_always_treat['natural_mean']
    ]

    fig.add_trace(go.Bar(
        x=interventions,
        y=outcomes,
        marker_color=['#27AE60', '#EB5757', '#2D9CDB'],
        text=[f'{y:.1f}' for y in outcomes],
        textposition='outside',
        showlegend=False
    ), row=2, col=1)

    # 4. 权重分布
    fig.add_trace(go.Histogram(
        x=weights,
        nbinsx=50,
        marker_color='#F2994A',
        name='权重',
        showlegend=False
    ), row=2, col=2)

    fig.update_xaxes(title_text="累积处理次数", row=1, col=1)
    fig.update_xaxes(title_text="累积处理次数", row=1, col=2)
    fig.update_xaxes(title_text="干预策略", row=2, col=1)
    fig.update_xaxes(title_text="IPW 权重", row=2, col=2)

    fig.update_yaxes(title_text="预期结果", row=1, col=1)
    fig.update_yaxes(title_text="观测结果", row=1, col=2)
    fig.update_yaxes(title_text="平均结果", row=2, col=1)
    fig.update_yaxes(title_text="频数", row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='时变处理效应分析',
        showlegend=False
    )

    # 计算平均处理效应
    ate = g_always_treat['mean_outcome'] - g_never_treat['mean_outcome']

    summary = f"""
## 时变处理效应分析结果

### 数据信息
- 个体数: 500
- 时间周期数: {n_periods}
- 处理模式: {treatment_pattern}

### 边际结构模型（MSM）
- **累积处理效应**: {msm_result['treatment_effect']:.3f} 每次处理
- 基线结果: {msm_result['baseline']:.2f}
- 累积 {n_periods} 次处理的预期效应: {msm_result['treatment_effect'] * n_periods:.2f}

### G-computation 估计
- 始终处理: {g_always_treat['mean_outcome']:.2f}
- 从不处理: {g_never_treat['mean_outcome']:.2f}
- **平均处理效应 (ATE)**: {ate:.2f}

### 关键洞察
1. **累积效应**: 处理效应随时间累积
2. **时间依赖**: 历史处理影响当前结果
3. **权重调整**: IPW 权重平均为 {weights.mean():.2f}，范围 [{weights.min():.2f}, {weights.max():.2f}]

### 方法说明
- MSM: 边际结构模型，使用 IPW 调整时变混淆
- G-computation: 标准化方法，模拟不同干预策略
"""

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [cumulative_summary.to_dict('records')],
        'summary': summary,
        'metrics': {
            'msm_treatment_effect': float(msm_result['treatment_effect']),
            'msm_baseline': float(msm_result['baseline']),
            'ate': float(ate),
            'always_treat_outcome': float(g_always_treat['mean_outcome']),
            'never_treat_outcome': float(g_never_treat['mean_outcome']),
            'avg_weight': float(weights.mean())
        }
    }


def analyze_mediation(
    n_samples: int = 1000,
    direct_effect: float = 2.0,
    indirect_effect: float = 1.5
) -> dict:
    """
    中介分析

    Args:
        n_samples: 样本量
        direct_effect: 直接效应大小
        indirect_effect: 间接效应大小

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成数据
    data = generate_mediation_data(
        n_samples=n_samples,
        direct_effect=direct_effect,
        indirect_effect=indirect_effect
    )

    T = data['T'].values
    M = data['M'].values
    Y = data['Y'].values
    X = data['X'].values

    # 中介分析
    analyzer = MediationAnalyzer()
    analyzer.fit(T, M, Y, X.reshape(-1, 1))
    effects = analyzer.decompose_effects(T, X.reshape(-1, 1))

    # Baron-Kenny 检验
    bk_result = baron_kenny_test(T, M, Y, X.reshape(-1, 1))

    # 敏感性分析
    sensitivity = sensitivity_analysis_mediation(
        effects['natural_indirect_effect'],
        rho_range=np.linspace(0, 0.5, 20)
    )

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '效应分解',
            '中介路径系数',
            '敏感性分析',
            'T-M-Y 关系'
        ],
        specs=[
            [{'type': 'bar'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}]
        ]
    )

    # 1. 效应分解
    effect_names = ['总效应', '直接效应', '间接效应']
    effect_values = [
        effects['total_effect'],
        effects['natural_direct_effect'],
        effects['natural_indirect_effect']
    ]
    colors_bar = ['#2D9CDB', '#27AE60', '#F2994A']

    fig.add_trace(go.Bar(
        x=effect_names,
        y=effect_values,
        marker_color=colors_bar,
        text=[f'{v:.3f}' for v in effect_values],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)

    # 2. 中介路径
    path_diagram_x = [0, 0.5, 1, 0.5]
    path_diagram_y = [0, 0.5, 0, -0.5]
    path_labels = ['T', 'M', 'Y', '']

    fig.add_trace(go.Scatter(
        x=path_diagram_x[:3],
        y=path_diagram_y[:3],
        mode='markers+text',
        marker=dict(size=40, color='#2D9CDB'),
        text=path_labels[:3],
        textposition='middle center',
        textfont=dict(size=14, color='white'),
        showlegend=False
    ), row=1, col=2)

    # 添加箭头注释
    annotations = [
        dict(x=0.5, y=0.5, ax=0, ay=0, text=f"a={bk_result['step2_t_to_m']:.2f}",
             xref='x2', yref='y2', axref='x2', ayref='y2', showarrow=True, arrowhead=2),
        dict(x=1, y=0, ax=0.5, ay=0.5, text=f"b={bk_result['step3_m_to_y']:.2f}",
             xref='x2', yref='y2', axref='x2', ayref='y2', showarrow=True, arrowhead=2),
        dict(x=1, y=0, ax=0, ay=0, text=f"c'={bk_result['step3_direct_effect']:.2f}",
             xref='x2', yref='y2', axref='x2', ayref='y2', showarrow=True, arrowhead=2)
    ]

    # 3. 敏感性分析
    fig.add_trace(go.Scatter(
        x=sensitivity['rho_range'],
        y=sensitivity['adjusted_effects'],
        mode='lines',
        line=dict(color='#9B51E0', width=3),
        name='调整后间接效应',
        showlegend=False
    ), row=2, col=1)

    fig.add_hline(
        y=sensitivity['observed_effect'],
        line_dash="dash",
        line_color="red",
        row=2, col=1
    )

    fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=1)

    # 4. T-M-Y 关系
    treat_group = data[data['T'] == 1]
    control_group = data[data['T'] == 0]

    fig.add_trace(go.Scatter(
        x=control_group['M'],
        y=control_group['Y'],
        mode='markers',
        marker=dict(size=4, color='#2D9CDB', opacity=0.5),
        name='对照组 (T=0)',
        showlegend=True
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=treat_group['M'],
        y=treat_group['Y'],
        mode='markers',
        marker=dict(size=4, color='#EB5757', opacity=0.5),
        name='处理组 (T=1)',
        showlegend=True
    ), row=2, col=2)

    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2, showticklabels=False)
    fig.update_xaxes(title_text="混淆强度 (ρ)", row=2, col=1)
    fig.update_xaxes(title_text="中介变量 (M)", row=2, col=2)

    fig.update_yaxes(title_text="效应大小", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2, showticklabels=False)
    fig.update_yaxes(title_text="间接效应", row=2, col=1)
    fig.update_yaxes(title_text="结果 (Y)", row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='中介分析',
        annotations=annotations
    )

    # 构建摘要
    proportion_pct = effects['proportion_mediated'] * 100

    summary = f"""
## 中介分析结果

### 数据信息
- 样本量: {n_samples}
- 真实直接效应: {direct_effect}
- 真实间接效应: {indirect_effect}

### 效应分解
- **总效应 (TE)**: {effects['total_effect']:.3f}
- **自然直接效应 (NDE)**: {effects['natural_direct_effect']:.3f}
- **自然间接效应 (NIE)**: {effects['natural_indirect_effect']:.3f}
- **中介比例**: {proportion_pct:.1f}%

### Baron-Kenny 检验
- 步骤 1 (T → Y): {bk_result['step1_total_effect']:.3f}
- 步骤 2 (T → M): {bk_result['step2_t_to_m']:.3f}
- 步骤 3 (T → Y | M): {bk_result['step3_direct_effect']:.3f}
- 中介类型: {bk_result['mediation_type']}

### 关键洞察
1. **中介机制**: 处理通过中介变量 M 产生 {proportion_pct:.1f}% 的效应
2. **直接路径**: 仍有 {(1-effects['proportion_mediated'])*100:.1f}% 的效应通过直接路径
3. **稳健性**: 敏感性分析显示在中等混淆下结论稳健

### 方法说明
- 使用潜在结果框架分解效应
- NDE: 自然直接效应（阻断中介路径）
- NIE: 自然间接效应（仅通过中介）
"""

    # 构建表格
    effects_table = pd.DataFrame({
        '效应类型': ['总效应', '直接效应', '间接效应'],
        '估计值': [
            effects['total_effect'],
            effects['natural_direct_effect'],
            effects['natural_indirect_effect']
        ],
        '占比': [
            '100%',
            f"{(effects['natural_direct_effect']/effects['total_effect']*100):.1f}%",
            f"{(effects['natural_indirect_effect']/effects['total_effect']*100):.1f}%"
        ]
    })

    return {
        'charts': [_fig_to_chart_data(fig)],
        'tables': [effects_table.to_dict('records')],
        'summary': summary,
        'metrics': {
            'total_effect': float(effects['total_effect']),
            'direct_effect': float(effects['natural_direct_effect']),
            'indirect_effect': float(effects['natural_indirect_effect']),
            'proportion_mediated': float(effects['proportion_mediated']),
            'mediation_type': bk_result['mediation_type']
        }
    }


def _add_graph_to_subplot(
    fig: go.Figure,
    edges: list,
    node_names: list,
    row: int,
    col: int,
    color: str = '#2D9CDB'
):
    """添加图到子图"""
    import networkx as nx

    # 创建图
    G = nx.DiGraph()
    for edge in edges:
        if isinstance(edge, tuple) and len(edge) == 2:
            G.add_edge(edge[0], edge[1])

    # 布局
    try:
        pos = nx.spring_layout(G, seed=42, k=2)
    except:
        # 如果没有边，手动布局
        pos = {node: (i, 0) for i, node in enumerate(node_names)}

    # 绘制边
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=2, color=color),
            hoverinfo='none',
            showlegend=False
        ), row=row, col=col)

        # 箭头
        angle = np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2],
            y=[(y0 + y1) / 2],
            mode='markers',
            marker=dict(size=10, color=color, symbol='arrow', angle=angle),
            hoverinfo='none',
            showlegend=False
        ), row=row, col=col)

    # 绘制节点
    if pos:
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(size=20, color=color, line=dict(width=2, color='white')),
            hoverinfo='text',
            showlegend=False
        ), row=row, col=col)

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
