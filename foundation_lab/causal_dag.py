"""
因果图与 DAG (Directed Acyclic Graph) 可视化模块

核心概念:
- 因果图 (Causal Graph): 用有向无环图表示因果关系
- 混淆变量 (Confounder): 同时影响处理和结果的变量
- 后门路径 (Backdoor Path): 从 T 到 Y 的非因果路径
- d-分离 (d-separation): 判断条件独立性的图准则
- 调整公式 (Adjustment Formula): 控制混淆后的因果效应识别
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from typing import List, Tuple, Dict


def create_dag_figure(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    positions: Dict[str, Tuple[float, float]],
    node_colors: Dict[str, str] = None,
    title: str = "因果图 (DAG)"
) -> go.Figure:
    """创建 DAG 可视化图"""

    if node_colors is None:
        node_colors = {}

    # 默认颜色
    default_colors = {
        'T': '#FF6B6B',  # 处理 - 红色
        'Y': '#4ECDC4',  # 结果 - 青色
        'X': '#FFE66D',  # 混淆 - 黄色
        'U': '#95E1D3',  # 未观测 - 浅绿
    }

    fig = go.Figure()

    # 绘制边
    for edge in edges:
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]

        # 计算箭头方向
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)

        # 缩短线段以适应节点大小
        node_radius = 0.08
        x0_adj = x0 + node_radius * dx / length
        y0_adj = y0 + node_radius * dy / length
        x1_adj = x1 - node_radius * dx / length
        y1_adj = y1 - node_radius * dy / length

        fig.add_trace(go.Scatter(
            x=[x0_adj, x1_adj],
            y=[y0_adj, y1_adj],
            mode='lines',
            line=dict(color='#666', width=2),
            hoverinfo='none',
            showlegend=False
        ))

        # 添加箭头
        fig.add_annotation(
            x=x1_adj, y=y1_adj,
            ax=x0_adj, ay=y0_adj,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#666'
        )

    # 绘制节点
    for node in nodes:
        x, y = positions[node]
        color = node_colors.get(node, default_colors.get(node[0], '#B8B8B8'))

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=50, color=color, line=dict(color='#333', width=2)),
            text=[node],
            textposition='middle center',
            textfont=dict(size=14, color='#333', family='Arial Black'),
            hoverinfo='text',
            hovertext=f'节点: {node}',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1.5]),
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_confounding_dag() -> go.Figure:
    """创建经典混淆 DAG: X -> T, X -> Y, T -> Y"""
    nodes = ['X', 'T', 'Y']
    edges = [('X', 'T'), ('X', 'Y'), ('T', 'Y')]
    positions = {'X': (0.5, 1), 'T': (0, 0), 'Y': (1, 0)}

    return create_dag_figure(
        nodes, edges, positions,
        title="经典混淆结构: X 是混淆变量"
    )


def create_mediator_dag() -> go.Figure:
    """创建中介变量 DAG: T -> M -> Y, T -> Y"""
    nodes = ['T', 'M', 'Y']
    edges = [('T', 'M'), ('M', 'Y'), ('T', 'Y')]
    positions = {'T': (0, 0.5), 'M': (0.5, 1), 'Y': (1, 0.5)}

    return create_dag_figure(
        nodes, edges, positions,
        node_colors={'M': '#9B59B6'},
        title="中介结构: M 是中介变量"
    )


def create_collider_dag() -> go.Figure:
    """创建碰撞变量 DAG: T -> C <- Y"""
    nodes = ['T', 'C', 'Y']
    edges = [('T', 'C'), ('Y', 'C')]
    positions = {'T': (0, 1), 'C': (0.5, 0), 'Y': (1, 1)}

    return create_dag_figure(
        nodes, edges, positions,
        node_colors={'C': '#E74C3C'},
        title="碰撞结构: C 是碰撞变量 (控制会引入偏差!)"
    )


def create_complex_dag() -> go.Figure:
    """创建复杂 DAG 示例"""
    nodes = ['X1', 'X2', 'U', 'T', 'Y']
    edges = [
        ('X1', 'T'), ('X1', 'Y'),
        ('X2', 'T'), ('X2', 'Y'),
        ('U', 'T'), ('U', 'Y'),
        ('T', 'Y')
    ]
    positions = {
        'X1': (0, 1), 'X2': (1, 1), 'U': (0.5, 1),
        'T': (0.25, 0), 'Y': (0.75, 0)
    }

    return create_dag_figure(
        nodes, edges, positions,
        node_colors={'U': '#95E1D3'},
        title="复杂结构: U 是未观测混淆变量"
    )


def identify_backdoor_paths(dag_type: str) -> str:
    """识别后门路径"""

    explanations = {
        "confounding": """
### 混淆结构分析

**DAG**: X → T, X → Y, T → Y

**因果路径**: T → Y

**后门路径**: T ← X → Y

**识别条件**: 控制 X 可以阻断后门路径

**调整公式**:
$$P(Y|do(T)) = \\sum_x P(Y|T,X=x) \\cdot P(X=x)$$

**实践建议**: 收集并控制混淆变量 X
        """,

        "mediator": """
### 中介结构分析

**DAG**: T → M → Y, T → Y

**因果路径**:
1. 直接效应: T → Y
2. 间接效应: T → M → Y

**后门路径**: 无 (T 没有指向它的边)

**注意**: 控制 M 会阻断部分因果路径!

**总效应**: 不应控制 M
**直接效应**: 需要控制 M
        """,

        "collider": """
### 碰撞结构分析

**DAG**: T → C ← Y

**因果路径**: 无 (T 和 Y 没有因果关系)

**后门路径**: 无

**危险**: 控制 C 会打开 T 和 Y 之间的关联!

这就是 **Berkson's Paradox** 的来源。

**实践建议**: 永远不要控制碰撞变量
        """,

        "complex": """
### 复杂结构分析

**DAG**: X1,X2,U → T,Y, T → Y

**后门路径**:
1. T ← X1 → Y
2. T ← X2 → Y
3. T ← U → Y (不可阻断!)

**识别条件**: 控制 {X1, X2} 可以阻断可观测的后门路径

**问题**: U 是未观测混淆变量，无法识别因果效应!

**解决方案**:
- 工具变量 (IV)
- 敏感性分析
- 设计更好的研究
        """
    }

    return explanations.get(dag_type, "请选择一个 DAG 类型")


def simulate_confounding_effect(confounding_strength: float) -> Tuple[go.Figure, str]:
    """模拟混淆效应"""
    np.random.seed(42)
    n = 1000
    true_ate = 2.0

    # 生成数据
    X = np.random.randn(n)
    T = (np.random.randn(n) + confounding_strength * X > 0).astype(int)
    Y = 1 + true_ate * T + confounding_strength * X + np.random.randn(n) * 0.5

    # 计算估计
    naive_ate = Y[T == 1].mean() - Y[T == 0].mean()

    # 调整估计 (简单线性回归)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(np.column_stack([T, X]), Y)
    adjusted_ate = model.coef_[0]

    # 可视化
    fig = go.Figure()

    confound_values = np.linspace(0, 3, 20)
    naive_estimates = []
    adjusted_estimates = []

    for c in confound_values:
        X_temp = np.random.randn(n)
        T_temp = (np.random.randn(n) + c * X_temp > 0).astype(int)
        Y_temp = 1 + true_ate * T_temp + c * X_temp + np.random.randn(n) * 0.5

        naive_estimates.append(Y_temp[T_temp == 1].mean() - Y_temp[T_temp == 0].mean())

        model = LinearRegression()
        model.fit(np.column_stack([T_temp, X_temp]), Y_temp)
        adjusted_estimates.append(model.coef_[0])

    fig.add_trace(go.Scatter(
        x=confound_values, y=naive_estimates,
        mode='lines+markers',
        name='朴素估计',
        line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=confound_values, y=adjusted_estimates,
        mode='lines+markers',
        name='调整估计',
        line=dict(color='green', width=2)
    ))

    fig.add_hline(y=true_ate, line_dash="dash", line_color="blue",
                  annotation_text=f"真实 ATE = {true_ate}")

    fig.update_layout(
        title='混淆强度对估计的影响',
        xaxis_title='混淆强度',
        yaxis_title='ATE 估计',
        template='plotly_white',
        height=400
    )

    summary = f"""
### 当前参数: 混淆强度 = {confounding_strength}

| 估计方法 | 估计值 | 偏差 |
|---------|--------|------|
| 真实 ATE | {true_ate:.4f} | - |
| 朴素估计 | {naive_ate:.4f} | {naive_ate - true_ate:+.4f} |
| 调整估计 | {adjusted_ate:.4f} | {adjusted_ate - true_ate:+.4f} |

**结论**: 混淆越强，朴素估计偏差越大；控制混淆变量后，估计接近真实值。
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 因果图与 DAG (Directed Acyclic Graph)

因果图是表示变量之间因果关系的有向无环图，是因果推断的重要工具。

### 核心概念

- **节点**: 代表变量
- **边**: 代表直接因果关系
- **路径**: 节点之间的连接序列
- **后门路径**: 从 T 到 Y 的非因果路径 (需要阻断)

---
        """)

        gr.Markdown("### 经典因果结构")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**混淆结构 (Confounding)**")
                confound_plot = gr.Plot(value=create_confounding_dag())

            with gr.Column():
                gr.Markdown("**中介结构 (Mediation)**")
                mediator_plot = gr.Plot(value=create_mediator_dag())

        with gr.Row():
            with gr.Column():
                gr.Markdown("**碰撞结构 (Collider)**")
                collider_plot = gr.Plot(value=create_collider_dag())

            with gr.Column():
                gr.Markdown("**复杂结构 (未观测混淆)**")
                complex_plot = gr.Plot(value=create_complex_dag())

        gr.Markdown("---")
        gr.Markdown("### 后门路径分析")

        with gr.Row():
            dag_selector = gr.Radio(
                choices=["confounding", "mediator", "collider", "complex"],
                value="confounding",
                label="选择 DAG 类型"
            )

        with gr.Row():
            path_analysis = gr.Markdown(value=identify_backdoor_paths("confounding"))

        dag_selector.change(
            fn=identify_backdoor_paths,
            inputs=[dag_selector],
            outputs=[path_analysis]
        )

        gr.Markdown("---")
        gr.Markdown("### 混淆效应模拟")

        with gr.Row():
            confound_slider = gr.Slider(
                minimum=0, maximum=3, value=1.0, step=0.1,
                label="混淆强度"
            )
            simulate_btn = gr.Button("模拟", variant="primary")

        with gr.Row():
            sim_plot = gr.Plot()

        with gr.Row():
            sim_summary = gr.Markdown()

        simulate_btn.click(
            fn=simulate_confounding_effect,
            inputs=[confound_slider],
            outputs=[sim_plot, sim_summary]
        )

        gr.Markdown("""
### 思考题

1. 为什么控制碰撞变量会引入偏差？
2. 如何判断一个变量是混淆变量还是中介变量？
3. 当存在未观测混淆时，有哪些识别策略？

### 练习

完成 `exercises/chapter1_foundation/ex2_causal_dag.py` 中的练习。
        """)

    return None
