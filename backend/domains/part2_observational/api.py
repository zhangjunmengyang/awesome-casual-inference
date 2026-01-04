"""
Part 2: 观测数据方法 API 适配层

提供统一的 API 接口，返回格式为:
{
    "charts": [...],
    "tables": [],
    "summary": "...",
    "metrics": {...}
}
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional

from .utils import (
    generate_confounded_data,
    compute_naive_ate,
    compute_smd,
    compute_propensity_overlap,
    compute_covariate_balance,
    compute_effective_sample_size
)
from .propensity_score import PropensityScoreEstimator, PropensityScoreMatching
from .matching import (
    NearestNeighborMatching,
    MahalanobisMatching,
    CovariateExactMatching
)
from .weighting import (
    IPWEstimator,
    StabilizedIPW,
    OverlapWeighting,
    TrimmedIPW
)
from .doubly_robust import DoublyRobustEstimator, AIPWEstimator, TMLEEstimator
from .sensitivity_analysis import RosenbaumBounds, EValueAnalysis, ConfoundingBiasAnalysis


def _fig_to_chart_data(fig: go.Figure) -> dict:
    """将 Plotly Figure 转换为前端可用的图表数据"""
    return fig.to_dict()


def analyze_psm(
    n_samples: int = 1000,
    confounding_strength: float = 1.0,
    caliper: float = 0.2,
    n_neighbors: int = 1
) -> Dict[str, Any]:
    """
    倾向得分匹配分析

    Parameters:
    -----------
    n_samples: 样本数
    confounding_strength: 混淆强度
    caliper: 卡尺宽度
    n_neighbors: 匹配邻居数

    Returns:
    --------
    API response dict
    """
    # 生成数据
    df, params = generate_confounded_data(
        n_samples=n_samples,
        confounding_strength=confounding_strength,
        seed=42
    )

    feature_names = [col for col in df.columns if col.startswith('X')]
    X = df[feature_names].values
    T = df['T'].values
    Y = df['Y'].values

    true_ate = params['true_ate']
    naive_ate = compute_naive_ate(df)

    # 估计倾向得分
    ps_model = PropensityScoreEstimator()
    propensity = ps_model.fit_predict(X, T)

    # PSM 匹配
    psm = PropensityScoreMatching(
        n_neighbors=n_neighbors,
        caliper=caliper,
        replace=False
    )
    matched_treated, matched_control = psm.match(propensity, T)

    # 估计 ATE
    psm_ate, psm_se = psm.estimate_ate(Y)

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '倾向得分分布 (匹配前)',
            '倾向得分分布 (匹配后)',
            '协变量平衡: SMD (匹配前)',
            '协变量平衡: SMD (匹配后)'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    treated_mask = T == 1
    control_mask = T == 0

    # 匹配前倾向得分分布
    fig.add_trace(go.Histogram(
        x=propensity[control_mask].tolist(),
        name='控制组',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=30
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=propensity[treated_mask].tolist(),
        name='处理组',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=30
    ), row=1, col=1)

    # 匹配后倾向得分分布
    fig.add_trace(go.Histogram(
        x=propensity[matched_control].tolist(),
        name='控制组 (匹配)',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=30,
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=propensity[matched_treated].tolist(),
        name='处理组 (匹配)',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=30,
        showlegend=False
    ), row=1, col=2)

    # SMD 匹配前
    X_t_before = X[treated_mask]
    X_c_before = X[control_mask]
    smd_before = compute_smd(X_t_before, X_c_before)

    fig.add_trace(go.Bar(
        x=feature_names,
        y=np.abs(smd_before).tolist(),
        name='匹配前',
        marker_color='#F2994A'
    ), row=2, col=1)

    fig.add_hline(y=0.1, row=2, col=1, line_dash="dash", line_color="green")

    # SMD 匹配后
    X_t_after = X[matched_treated]
    X_c_after = X[matched_control]
    smd_after = compute_smd(X_t_after, X_c_after)

    fig.add_trace(go.Bar(
        x=feature_names,
        y=np.abs(smd_after).tolist(),
        name='匹配后',
        marker_color='#27AE60',
        showlegend=False
    ), row=2, col=2)

    fig.add_hline(y=0.1, row=2, col=2, line_dash="dash", line_color="green")

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='倾向得分匹配 (PSM) 分析',
        barmode='overlay'
    )

    n_matched = len(matched_treated)
    n_treated = int(T.sum())
    match_rate = n_matched / n_treated * 100

    summary = f"""
## 倾向得分匹配 (PSM) 结果

### 匹配统计

| 指标 | 值 |
|------|-----|
| 总样本数 | {n_samples} |
| 处理组样本 | {n_treated} |
| 成功匹配 | {n_matched} ({match_rate:.1f}%) |
| 匹配邻居数 | {n_neighbors} |
| 卡尺宽度 | {caliper} |

### ATE 估计

| 方法 | 估计值 | 标准误 | 偏差 |
|------|--------|--------|------|
| 真实 ATE | {true_ate:.4f} | - | - |
| 朴素估计 | {naive_ate:.4f} | - | {naive_ate - true_ate:.4f} |
| PSM | {psm_ate:.4f} | {psm_se:.4f} | {psm_ate - true_ate:.4f} |

### 关键洞察

- PSM 通过平衡协变量来减少混淆偏差
- 匹配后的 SMD 应该接近 0 (< 0.1 表示良好平衡)
- 匹配率: {match_rate:.1f}% - 表示有 {100-match_rate:.1f}% 的处理组样本未找到匹配
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "true_ate": float(true_ate),
            "naive_ate": float(naive_ate),
            "psm_ate": float(psm_ate),
            "psm_se": float(psm_se),
            "bias": float(psm_ate - true_ate),
            "match_rate": float(match_rate),
            "n_matched": int(n_matched),
        },
    }


def analyze_matching_methods(
    n_samples: int = 1000,
    confounding_strength: float = 1.0
) -> Dict[str, Any]:
    """
    对比不同匹配方法

    Returns:
    --------
    API response dict
    """
    # 生成数据
    df, params = generate_confounded_data(
        n_samples=n_samples,
        confounding_strength=confounding_strength,
        seed=42
    )

    feature_names = [col for col in df.columns if col.startswith('X')]
    X = df[feature_names].values
    T = df['T'].values
    Y = df['Y'].values

    true_ate = params['true_ate']
    naive_ate = compute_naive_ate(df)

    # 倾向得分 (用于 PSM)
    ps_model = PropensityScoreEstimator()
    propensity = ps_model.fit_predict(X, T)

    results = {}

    # 1. PSM
    psm = PropensityScoreMatching(n_neighbors=1, caliper=0.2)
    psm.match(propensity, T)
    psm_ate, psm_se = psm.estimate_ate(Y)
    results['PSM'] = {'ate': psm_ate, 'se': psm_se}

    # 2. 最近邻匹配 (协变量空间)
    nnm = NearestNeighborMatching(n_neighbors=1, metric='euclidean')
    nnm.match(X, T)
    nnm_ate, nnm_se = nnm.estimate_ate(Y)
    results['NN (Euclidean)'] = {'ate': nnm_ate, 'se': nnm_se}

    # 3. 马氏距离匹配
    mdm = MahalanobisMatching(n_neighbors=1)
    mdm.match(X, T)
    mdm_ate, mdm_se = mdm.estimate_ate(Y)
    results['Mahalanobis'] = {'ate': mdm_ate, 'se': mdm_se}

    # 4. CEM
    cem = CovariateExactMatching(n_bins=5)
    cem.match(X, T)
    cem_ate, cem_se = cem.estimate_ate(Y)
    results['CEM'] = {'ate': cem_ate, 'se': cem_se}

    # 可视化
    fig = go.Figure()

    methods = list(results.keys())
    estimates = [results[m]['ate'] for m in methods]
    ses = [results[m]['se'] for m in methods]
    colors = ['#2D9CDB', '#27AE60', '#9B59B6', '#F2994A']

    for i, (method, est, se, color) in enumerate(zip(methods, estimates, ses, colors)):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[est],
            error_y=dict(type='data', array=[1.96 * se]),
            mode='markers',
            marker=dict(size=12, color=color),
            name=method
        ))

    fig.add_hline(
        y=true_ate,
        line_dash="dash",
        line_color="green",
        annotation_text=f"真实 ATE = {true_ate:.4f}"
    )

    fig.update_layout(
        title='不同匹配方法对比',
        xaxis=dict(
            ticktext=methods,
            tickvals=list(range(len(methods)))
        ),
        yaxis_title='ATE 估计',
        template='plotly_white',
        height=500,
        showlegend=False
    )

    summary = f"""
## 匹配方法对比

### ATE 估计

| 方法 | 估计值 | 标准误 | 偏差 |
|------|--------|--------|------|
| 真实 ATE | {true_ate:.4f} | - | - |
| 朴素估计 | {naive_ate:.4f} | - | {naive_ate - true_ate:.4f} |
"""

    for method in methods:
        ate = results[method]['ate']
        se = results[method]['se']
        bias = ate - true_ate
        summary += f"| {method} | {ate:.4f} | {se:.4f} | {bias:.4f} |\n"

    summary += """

### 方法说明

- **PSM**: 倾向得分匹配 - 在倾向得分空间匹配
- **NN (Euclidean)**: 最近邻匹配 - 在协变量空间使用欧氏距离
- **Mahalanobis**: 马氏距离匹配 - 考虑协变量相关性
- **CEM**: 粗糙精确匹配 - 分层后精确匹配
"""

    metrics = {
        "true_ate": float(true_ate),
        "naive_ate": float(naive_ate),
    }
    for method in methods:
        metrics[f"{method}_ate"] = float(results[method]['ate'])
        metrics[f"{method}_se"] = float(results[method]['se'])
        metrics[f"{method}_bias"] = float(results[method]['ate'] - true_ate)

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": metrics,
    }


def analyze_ipw(
    n_samples: int = 1000,
    confounding_strength: float = 1.0,
    stabilized: bool = True,
    trimming: float = 0.01
) -> Dict[str, Any]:
    """
    逆概率加权分析

    Returns:
    --------
    API response dict
    """
    # 生成数据
    df, params = generate_confounded_data(
        n_samples=n_samples,
        confounding_strength=confounding_strength,
        seed=42
    )

    feature_names = [col for col in df.columns if col.startswith('X')]
    X = df[feature_names].values
    T = df['T'].values
    Y = df['Y'].values

    true_ate = params['true_ate']
    naive_ate = compute_naive_ate(df)

    # IPW 估计
    if stabilized:
        ipw = StabilizedIPW(clip_propensity=(trimming, 1 - trimming))
    else:
        ipw = IPWEstimator(clip_propensity=(trimming, 1 - trimming))

    ipw.fit(X, T)
    ipw_ate, ipw_se, weights = ipw.estimate_ate(X, T, Y)

    propensity = ipw.propensity

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '倾向得分分布',
            'IPW 权重分布',
            '加权前后的协变量平衡',
            '估计效应对比'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    treated_mask = T == 1
    control_mask = T == 0

    # 倾向得分分布
    fig.add_trace(go.Histogram(
        x=propensity[control_mask].tolist(),
        name='控制组',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=30
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=propensity[treated_mask].tolist(),
        name='处理组',
        marker_color='#EB5757',
        opacity=0.6,
        nbinsx=30
    ), row=1, col=1)

    # 权重分布
    fig.add_trace(go.Histogram(
        x=weights[control_mask].tolist(),
        name='控制组权重',
        marker_color='#2D9CDB',
        opacity=0.6,
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=weights[treated_mask].tolist(),
        name='处理组权重',
        marker_color='#EB5757',
        opacity=0.6,
        showlegend=False
    ), row=1, col=2)

    # 协变量平衡
    balance_before = compute_covariate_balance(X, T)
    balance_after = compute_covariate_balance(X, T, weights)

    fig.add_trace(go.Bar(
        x=feature_names,
        y=[abs(s) for s in balance_before['smd']],
        name='加权前',
        marker_color='#F2994A'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=feature_names,
        y=[abs(s) for s in balance_after['smd']],
        name='加权后',
        marker_color='#27AE60'
    ), row=2, col=1)

    fig.add_hline(y=0.1, row=2, col=1, line_dash="dash", line_color="red")

    # 估计效应对比
    methods = ['真实 ATE', '朴素估计', 'IPW']
    values = [true_ate, naive_ate, ipw_ate]
    colors_bar = ['#27AE60', '#F2994A', '#2D9CDB']

    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        marker_color=colors_bar,
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template='plotly_white',
        title_text='逆概率加权 (IPW) 分析',
        barmode='group'
    )

    # 有效样本量
    ess = compute_effective_sample_size(weights)
    ess_fraction = ess / n_samples

    # 重叠统计
    overlap_stats = compute_propensity_overlap(propensity, T)

    summary = f"""
## 逆概率加权 (IPW) 结果

### 参数设置

| 参数 | 值 |
|------|-----|
| 样本量 | {n_samples} |
| 混淆强度 | {confounding_strength} |
| 稳定权重 | {'是' if stabilized else '否'} |
| 截断阈值 | {trimming} |

### ATE 估计

| 方法 | 估计值 | 标准误 | 偏差 |
|------|--------|--------|------|
| 真实 ATE | {true_ate:.4f} | - | - |
| 朴素估计 | {naive_ate:.4f} | - | {naive_ate - true_ate:.4f} |
| IPW | {ipw_ate:.4f} | {ipw_se:.4f} | {ipw_ate - true_ate:.4f} |

### 权重统计

| 指标 | 值 |
|------|-----|
| 有效样本量 (ESS) | {ess:.1f} ({ess_fraction*100:.1f}%) |
| 最大权重 | {weights.max():.2f} |
| 最小权重 | {weights.min():.2f} |

### 关键洞察

- IPW 通过给每个样本赋予权重来模拟随机实验
- 稳定权重可以减少极端权重的影响
- 有效样本量: {ess:.1f} ({ess_fraction*100:.1f}%) - 衡量权重分散程度
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "true_ate": float(true_ate),
            "naive_ate": float(naive_ate),
            "ipw_ate": float(ipw_ate),
            "ipw_se": float(ipw_se),
            "bias": float(ipw_ate - true_ate),
            "ess": float(ess),
            "ess_fraction": float(ess_fraction),
            "max_weight": float(weights.max()),
            "min_weight": float(weights.min()),
        },
    }


def analyze_doubly_robust(
    n_samples: int = 1000,
    confounding_strength: float = 1.0,
    method: str = "aipw"
) -> Dict[str, Any]:
    """
    双重稳健估计分析

    Parameters:
    -----------
    method: 方法选择 ('aipw', 'tmle')

    Returns:
    --------
    API response dict
    """
    # 生成数据
    df, params = generate_confounded_data(
        n_samples=n_samples,
        confounding_strength=confounding_strength,
        seed=42
    )

    feature_names = [col for col in df.columns if col.startswith('X')]
    X = df[feature_names].values
    T = df['T'].values
    Y = df['Y'].values

    true_ate = params['true_ate']
    naive_ate = compute_naive_ate(df)

    # 双重稳健估计
    if method == "aipw":
        dr = AIPWEstimator()
        dr_ate, dr_se = dr.estimate_ate(X, T, Y)
        method_name = "AIPW"
    elif method == "tmle":
        dr = TMLEEstimator()
        dr_ate, dr_se = dr.estimate_ate(X, T, Y)
        method_name = "TMLE"
    else:
        dr = DoublyRobustEstimator()
        dr_ate, dr_se = dr.estimate_ate(X, T, Y)
        method_name = "DR"

    # 单独的估计量 (用于对比)
    # IPW
    ipw = IPWEstimator()
    ipw.fit(X, T)
    ipw_ate, _, _ = ipw.estimate_ate(X, T, Y)

    # 结果回归
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    X_with_t = np.column_stack([X, T])
    lr.fit(X_with_t, Y)
    reg_ate = lr.coef_[-1]

    # 可视化
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '各方法估计对比',
            '估计偏差对比'
        )
    )

    methods_list = ['真实 ATE', '朴素估计', '回归调整', 'IPW', method_name]
    values = [true_ate, naive_ate, reg_ate, ipw_ate, dr_ate]
    colors = ['#27AE60', '#EB5757', '#F2994A', '#2D9CDB', '#9B59B6']

    # 估计对比
    fig.add_trace(go.Bar(
        x=methods_list,
        y=values,
        marker_color=colors,
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(y=true_ate, row=1, col=1, line_dash="dash", line_color="green")

    # 偏差对比
    biases = [0, abs(naive_ate - true_ate), abs(reg_ate - true_ate),
              abs(ipw_ate - true_ate), abs(dr_ate - true_ate)]

    fig.add_trace(go.Bar(
        x=methods_list,
        y=biases,
        marker_color=colors,
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=500,
        template='plotly_white',
        title_text=f'双重稳健估计 ({method_name}) 分析'
    )

    summary = f"""
## 双重稳健估计 ({method_name}) 结果

### ATE 估计

| 方法 | 估计值 | 标准误 | 偏差 |
|------|--------|--------|------|
| 真实 ATE | {true_ate:.4f} | - | - |
| 朴素估计 | {naive_ate:.4f} | - | {naive_ate - true_ate:.4f} |
| 回归调整 | {reg_ate:.4f} | - | {reg_ate - true_ate:.4f} |
| IPW | {ipw_ate:.4f} | - | {ipw_ate - true_ate:.4f} |
| **{method_name}** | **{dr_ate:.4f}** | {dr_se:.4f} | {dr_ate - true_ate:.4f} |

### 关键洞察

- {method_name} 估计量具有**双重稳健性**: 只需倾向得分模型或结果模型之一正确即可一致
- 当两个模型都正确时，{method_name} 具有更小的方差
- 相比单独的 IPW 或回归，{method_name} 提供了双重保护
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "true_ate": float(true_ate),
            "naive_ate": float(naive_ate),
            "reg_ate": float(reg_ate),
            "ipw_ate": float(ipw_ate),
            "dr_ate": float(dr_ate),
            "dr_se": float(dr_se),
            "bias": float(dr_ate - true_ate),
        },
    }


def analyze_sensitivity(
    estimated_ate: float,
    ci_lower: Optional[float] = None,
    effect_type: str = "mean_difference"
) -> Dict[str, Any]:
    """
    敏感性分析 (E-value)

    Parameters:
    -----------
    estimated_ate: 估计的 ATE
    ci_lower: 置信区间下界
    effect_type: 效应类型

    Returns:
    --------
    API response dict
    """
    # E-value 分析
    evalue_analyzer = EValueAnalysis()
    evalue_result = evalue_analyzer.compute_evalue(
        estimated_ate,
        effect_type=effect_type,
        ci_lower=ci_lower
    )

    # 混淆偏差分析
    bias_analyzer = ConfoundingBiasAnalysis()
    contour_data = bias_analyzer.sensitivity_contour(
        observed_ate=estimated_ate,
        treatment_corr_range=(-0.5, 0.5),
        outcome_corr_range=(-2.0, 2.0),
        n_points=20
    )

    # 可视化: E-value 解释
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'E-value 解释',
            '敏感性等高线图'
        ),
        specs=[[{"type": "indicator"}, {"type": "contour"}]]
    )

    # E-value 指示器
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=evalue_result['e_value'],
        title={'text': "E-value"},
        delta={'reference': 2.0, 'position': "bottom"}
    ), row=1, col=1)

    # 敏感性等高线图
    fig.add_trace(go.Contour(
        x=contour_data['treatment_corrs'],
        y=contour_data['outcome_corrs'],
        z=contour_data['adjusted_ates'],
        colorscale='RdBu',
        contours=dict(
            start=-abs(estimated_ate) * 2,
            end=abs(estimated_ate) * 2,
            size=abs(estimated_ate) * 0.2
        ),
        colorbar=dict(title="调整后 ATE")
    ), row=1, col=2)

    # 零效应曲线
    if contour_data['zero_effect_curve']:
        zero_curve_tc = [p['treatment_corr'] for p in contour_data['zero_effect_curve']]
        zero_curve_oc = [p['outcome_corr'] for p in contour_data['zero_effect_curve']]
        fig.add_trace(go.Scatter(
            x=zero_curve_tc,
            y=zero_curve_oc,
            mode='lines',
            line=dict(color='black', width=3, dash='dash'),
            name='零效应线',
            showlegend=True
        ), row=1, col=2)

    fig.update_xaxes(title_text='处理-混淆相关性', row=1, col=2)
    fig.update_yaxes(title_text='结果-混淆相关性', row=1, col=2)

    fig.update_layout(
        height=500,
        template='plotly_white',
        title_text='敏感性分析'
    )

    summary = f"""
## 敏感性分析结果

### E-value

| 指标 | 值 |
|------|-----|
| E-value | {evalue_result['e_value']:.2f} |
| E-value (CI) | {evalue_result['e_value_ci']:.2f if evalue_result['e_value_ci'] else 'N/A'} |

**解释**: {evalue_result['interpretation']}

**所需混淆强度**: {evalue_result['required_confounder_strength']}

### 敏感性等高线图

- 黑色虚线: 使调整后 ATE = 0 的混淆参数组合
- 横轴: 处理与未测量混淆的相关性
- 纵轴: 结果与未测量混淆的相关性

### 关键洞察

- E-value 越大，结果对未测量混淆越鲁棒
- E-value < 2: 结果对未测量混淆敏感
- E-value > 3: 结果对未测量混淆有较强鲁棒性
"""

    return {
        "charts": [_fig_to_chart_data(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {
            "e_value": float(evalue_result['e_value']),
            "e_value_ci": float(evalue_result['e_value_ci']) if evalue_result['e_value_ci'] else None,
            "interpretation": evalue_result['interpretation']
        },
    }
