"""
因果图与 DAG (Directed Acyclic Graph)

核心概念:
- 因果图: 用有向无环图表示因果关系
- 混淆变量 (Confounder): 同时影响处理和结果的变量
- 中介变量 (Mediator): 传递因果效应的变量
- 碰撞变量 (Collider): 被两个变量同时影响的变量
- 后门路径 (Backdoor Path): 从 T 到 Y 的非因果路径
- d-分离 (d-separation): 判断条件独立性的图准则
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Dict
from sklearn.linear_model import LinearRegression


def create_dag_visualization(
    dag_type: str = "confounding"
) -> Tuple[go.Figure, str]:
    """
    创建不同类型的 DAG 可视化

    Parameters:
    -----------
    dag_type: 'confounding', 'mediation', 'collider', 'complex'

    Returns:
    --------
    figure: Plotly 图表
    explanation: 文字说明
    """
    dag_configs = {
        "confounding": {
            "nodes": {"U": (0.5, 1), "T": (0, 0), "Y": (1, 0)},
            "edges": [("U", "T"), ("U", "Y"), ("T", "Y")],
            "title": "混淆结构: U 是混淆变量",
            "node_colors": {"U": "#FFE66D", "T": "#FF6B6B", "Y": "#4ECDC4"},
            "explanation": """
### 混淆 (Confounding)

**DAG 结构**: U → T, U → Y, T → Y

**因果路径**: T → Y

**后门路径**: T ← U → Y (非因果路径)

**识别条件**: 控制 U 可以阻断后门路径

**调整公式**:
$$P(Y|do(T)) = \\sum_u P(Y|T,U=u) \\cdot P(U=u)$$

**实践建议**: 收集并控制混淆变量 U
            """
        },
        "mediation": {
            "nodes": {"T": (0, 0.5), "M": (0.5, 1), "Y": (1, 0.5)},
            "edges": [("T", "M"), ("M", "Y"), ("T", "Y")],
            "title": "中介结构: M 是中介变量",
            "node_colors": {"T": "#FF6B6B", "M": "#9B59B6", "Y": "#4ECDC4"},
            "explanation": """
### 中介 (Mediation)

**DAG 结构**: T → M → Y, T → Y

**直接效应**: T → Y

**间接效应**: T → M → Y

**总效应**: 直接效应 + 间接效应

**分解**: ATE = NDE + NIE
- NDE: 自然直接效应
- NIE: 自然间接效应

**注意**: 控制 M 会阻断间接效应!
            """
        },
        "collider": {
            "nodes": {"T": (0, 1), "Y": (1, 1), "C": (0.5, 0)},
            "edges": [("T", "C"), ("Y", "C"), ("T", "Y")],
            "title": "碰撞结构: C 是碰撞变量 (控制会引入偏差!)",
            "node_colors": {"T": "#FF6B6B", "Y": "#4ECDC4", "C": "#E74C3C"},
            "explanation": """
### 碰撞 (Collider)

**DAG 结构**: T → C ← Y, T → Y

**碰撞变量**: C 被 T 和 Y 同时影响

**危险**: 控制 C 会打开 T 和 Y 之间的虚假关联!

这就是 **Berkson's Paradox** 的来源。

**实践建议**: 永远不要控制碰撞变量

**经典例子**:
- 医院数据分析 (住院状态是碰撞变量)
- 大学录取研究 (录取状态是碰撞变量)
            """
        },
        "complex": {
            "nodes": {
                "X1": (0, 1), "X2": (1, 1), "U": (0.5, 1.2),
                "T": (0.3, 0), "Y": (0.7, 0)
            },
            "edges": [
                ("X1", "T"), ("X1", "Y"),
                ("X2", "T"), ("X2", "Y"),
                ("U", "T"), ("U", "Y"),
                ("T", "Y")
            ],
            "title": "复杂结构: U 是未观测混淆变量",
            "node_colors": {
                "X1": "#FFE66D", "X2": "#FFE66D",
                "U": "#95E1D3", "T": "#FF6B6B", "Y": "#4ECDC4"
            },
            "explanation": """
### 复杂结构 (包含未观测混淆)

**后门路径**:
1. T ← X1 → Y (可控制)
2. T ← X2 → Y (可控制)
3. T ← U → Y (不可控制!)

**识别条件**: 控制 {X1, X2} 可阻断可观测的后门路径

**问题**: U 是未观测混淆变量，无法完全识别因果效应

**解决方案**:
- 工具变量 (Instrumental Variables)
- 敏感性分析 (Sensitivity Analysis)
- 前门调整 (Front-door Adjustment)
- 设计更好的研究 (随机实验)
            """
        }
    }

    config = dag_configs.get(dag_type, dag_configs["confounding"])

    # 创建图表
    fig = go.Figure()

    # 绘制边 (箭头)
    for src, dst in config["edges"]:
        x0, y0 = config["nodes"][src]
        x1, y1 = config["nodes"][dst]

        # 添加线条
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(color='#666', width=2),
            hoverinfo='none',
            showlegend=False
        ))

        # 添加箭头注释
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#666'
        )

    # 绘制节点
    for node, (x, y) in config["nodes"].items():
        color = config["node_colors"].get(node, "#B8B8B8")

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=50, color=color, line=dict(color='#333', width=2)),
            text=[node],
            textposition='middle center',
            textfont=dict(size=16, color='#333', family='Arial Black'),
            hovertext=f'变量: {node}',
            showlegend=False
        ))

    fig.update_layout(
        title=config["title"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 1.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 1.5]),
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig, config["explanation"]


def analyze_dag_structure(scenario: str) -> Dict:
    """
    分析 DAG 结构并返回识别策略

    Returns:
    --------
    分析结果字典，包含: paths, backdoor_set, identified
    """
    structures = {
        "confounding": {
            "causal_paths": ["T → Y"],
            "backdoor_paths": ["T ← U → Y"],
            "backdoor_set": ["U"],
            "identified": True,
            "strategy": "控制混淆变量 U"
        },
        "mediation": {
            "causal_paths": ["T → Y (直接)", "T → M → Y (间接)"],
            "backdoor_paths": [],
            "backdoor_set": [],
            "identified": True,
            "strategy": "无需控制 (无后门路径)"
        },
        "collider": {
            "causal_paths": ["T → Y"],
            "backdoor_paths": [],
            "backdoor_set": [],
            "identified": True,
            "strategy": "无需控制, 但不要控制 C!"
        },
        "complex": {
            "causal_paths": ["T → Y"],
            "backdoor_paths": ["T ← X1 → Y", "T ← X2 → Y", "T ← U → Y"],
            "backdoor_set": ["X1", "X2", "U"],
            "identified": False,
            "strategy": "控制 X1, X2; U 未观测导致无法完全识别"
        }
    }

    return structures.get(scenario, structures["confounding"])


def identify_backdoor_paths(dag_type: str) -> str:
    """识别后门路径的详细说明"""
    analysis = analyze_dag_structure(dag_type)

    explanation = f"""
### 路径分析

**因果路径**:
{chr(10).join(['- ' + p for p in analysis['causal_paths']])}

**后门路径**:
{chr(10).join(['- ' + p for p in analysis['backdoor_paths']]) if analysis['backdoor_paths'] else '- 无'}

**后门调整集**:
{chr(10).join(['- ' + v for v in analysis['backdoor_set']]) if analysis['backdoor_set'] else '- 空集 (无需控制)'}

**可识别性**: {'✓ 可识别' if analysis['identified'] else '✗ 不可识别'}

**识别策略**: {analysis['strategy']}
    """

    return explanation


def simulate_confounding_effect(
    confounding_strength: float = 1.0,
    n_samples: int = 1000
) -> Tuple[go.Figure, dict]:
    """
    模拟混淆强度对因果效应估计的影响

    Returns:
    --------
    figure: 展示不同混淆强度下的估计偏差
    stats: 统计信息
    """
    np.random.seed(42)
    true_ate = 2.0

    # 测试不同混淆强度
    confound_values = np.linspace(0, 3, 20)
    naive_estimates = []
    adjusted_estimates = []

    for c in confound_values:
        # 生成数据
        X = np.random.randn(n_samples)
        T = (np.random.randn(n_samples) + c * X > 0).astype(int)
        Y = 1 + true_ate * T + c * X + np.random.randn(n_samples) * 0.5

        # 朴素估计
        naive_ate = Y[T == 1].mean() - Y[T == 0].mean()
        naive_estimates.append(naive_ate)

        # 调整估计 (控制 X)
        model = LinearRegression()
        model.fit(np.column_stack([T, X]), Y)
        adjusted_estimates.append(model.coef_[0])

    # 可视化
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=confound_values, y=naive_estimates,
        mode='lines+markers',
        name='朴素估计 (有偏)',
        line=dict(color='#EB5757', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=confound_values, y=adjusted_estimates,
        mode='lines+markers',
        name='调整估计 (控制X)',
        line=dict(color='#27AE60', width=2)
    ))

    fig.add_hline(
        y=true_ate,
        line_dash="dash",
        line_color="#2D9CDB",
        annotation_text=f"真实 ATE = {true_ate}"
    )

    fig.update_layout(
        title='混淆强度对 ATE 估计的影响',
        xaxis_title='混淆强度',
        yaxis_title='估计的 ATE',
        template='plotly_white',
        height=400
    )

    # 计算当前参数下的统计
    X = np.random.randn(n_samples)
    T = (np.random.randn(n_samples) + confounding_strength * X > 0).astype(int)
    Y = 1 + true_ate * T + confounding_strength * X + np.random.randn(n_samples) * 0.5

    naive_ate = Y[T == 1].mean() - Y[T == 0].mean()
    model = LinearRegression()
    model.fit(np.column_stack([T, X]), Y)
    adjusted_ate = model.coef_[0]

    stats = {
        'confounding_strength': confounding_strength,
        'true_ate': true_ate,
        'naive_ate': float(naive_ate),
        'adjusted_ate': float(adjusted_ate),
        'naive_bias': float(naive_ate - true_ate),
        'adjusted_bias': float(adjusted_ate - true_ate)
    }

    return fig, stats
