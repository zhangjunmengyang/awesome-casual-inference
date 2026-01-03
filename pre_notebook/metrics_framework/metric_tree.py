"""
指标树构建模块

业务背景：
---------
指标树是将业务目标层层拆解的分析框架：
1. 从顶层目标开始
2. 逐层分解为驱动因素
3. 找到可行动的叶子节点

示例：GMV = 用户数 × 转化率 × 客单价
     用户数 = 新客 + 老客
     转化率 = 加购率 × 付款率
     ...

面试考点：
---------
- 如何构建指标树？
- 乘法分解 vs 加法分解？
- 如何用指标树做归因分析？
- 杜邦分析法？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DecompositionType(Enum):
    """分解类型"""
    MULTIPLICATIVE = "multiplicative"  # 乘法分解
    ADDITIVE = "additive"  # 加法分解


@dataclass
class MetricNode:
    """指标节点"""
    name: str
    value: float
    previous_value: Optional[float] = None
    children: List['MetricNode'] = field(default_factory=list)
    decomposition_type: DecompositionType = DecompositionType.MULTIPLICATIVE

    @property
    def change(self) -> Optional[float]:
        """计算变化率"""
        if self.previous_value and self.previous_value != 0:
            return (self.value - self.previous_value) / self.previous_value
        return None

    @property
    def contribution(self) -> Optional[float]:
        """计算对父节点变化的贡献"""
        # 简化计算，实际需要更复杂的归因逻辑
        return self.change


class MetricTree:
    """指标树"""

    def __init__(self, root: MetricNode):
        self.root = root

    def calculate_contribution(self) -> Dict[str, float]:
        """
        计算各节点对根节点变化的贡献

        使用对数分解法：
        对于乘法分解 Y = A × B × C：
            Δln(Y) = Δln(A) + Δln(B) + Δln(C)
            各因素贡献 = Δln(factor) / Δln(Y) * 100%

        对于加法分解 Y = A + B + C：
            各因素贡献 = ΔA / ΔY * 100%

        返回各叶子节点对根节点变化的贡献百分比
        """
        contributions = {}

        # 计算根节点的对数变化
        root_log_change = 0.0
        if (self.root.value is not None and self.root.previous_value is not None
                and self.root.value > 0 and self.root.previous_value > 0):
            root_log_change = np.log(self.root.value / self.root.previous_value)

        def _calculate_log_contribution(node: MetricNode, parent_type: DecompositionType) -> float:
            """递归计算节点的对数贡献"""
            if node.value is None or node.previous_value is None:
                return 0.0

            if node.value <= 0 or node.previous_value <= 0:
                # 对于非正值，使用绝对变化
                return node.value - node.previous_value

            log_change = np.log(node.value / node.previous_value)

            if len(node.children) == 0:
                # 叶子节点：返回对数变化
                return log_change
            else:
                # 中间节点：递归处理子节点
                if node.decomposition_type == DecompositionType.MULTIPLICATIVE:
                    # 乘法分解：子节点对数变化之和等于父节点对数变化
                    for child in node.children:
                        child_contrib = _calculate_log_contribution(child, node.decomposition_type)
                        if len(child.children) == 0:  # 只记录叶子节点
                            contributions[child.name] = child_contrib
                else:
                    # 加法分解：子节点绝对变化之和等于父节点绝对变化
                    parent_abs_change = node.value - node.previous_value
                    for child in node.children:
                        if child.value is not None and child.previous_value is not None:
                            child_abs_change = child.value - child.previous_value
                            # 转换为对根节点的贡献（相对于父节点的对数变化）
                            if parent_abs_change != 0:
                                child_contrib = (child_abs_change / parent_abs_change) * log_change
                            else:
                                child_contrib = 0.0

                            if len(child.children) == 0:
                                contributions[child.name] = child_contrib
                            else:
                                _calculate_log_contribution(child, node.decomposition_type)

                return log_change

        _calculate_log_contribution(self.root, self.root.decomposition_type)

        # 将对数贡献转换为百分比（相对于根节点的对数变化）
        if root_log_change != 0:
            for name in contributions:
                contributions[name] = contributions[name] / root_log_change
        else:
            # 如果根节点没有变化，所有贡献为0
            for name in contributions:
                contributions[name] = 0.0

        return contributions

    def to_dict(self) -> Dict:
        """转换为字典"""
        def _node_to_dict(node: MetricNode) -> Dict:
            return {
                'name': node.name,
                'value': node.value,
                'previous_value': node.previous_value,
                'change': node.change,
                'children': [_node_to_dict(c) for c in node.children]
            }
        return _node_to_dict(self.root)


def create_ecommerce_metric_tree(
    current_data: Dict[str, float],
    previous_data: Dict[str, float]
) -> MetricTree:
    """
    创建电商指标树

    GMV = 访问用户 × 转化率 × 客单价
        = (新客 + 老客) × (浏览→加购率 × 加购→支付率) × (件单价 × 件数)
    """
    # 构建叶子节点
    new_users = MetricNode(
        name='新客数',
        value=current_data.get('new_users', 10000),
        previous_value=previous_data.get('new_users', 9500)
    )

    old_users = MetricNode(
        name='老客数',
        value=current_data.get('old_users', 5000),
        previous_value=previous_data.get('old_users', 4800)
    )

    view_to_cart = MetricNode(
        name='浏览→加购率',
        value=current_data.get('view_to_cart', 0.15),
        previous_value=previous_data.get('view_to_cart', 0.14)
    )

    cart_to_pay = MetricNode(
        name='加购→支付率',
        value=current_data.get('cart_to_pay', 0.30),
        previous_value=previous_data.get('cart_to_pay', 0.28)
    )

    unit_price = MetricNode(
        name='件单价',
        value=current_data.get('unit_price', 50),
        previous_value=previous_data.get('unit_price', 48)
    )

    units_per_order = MetricNode(
        name='每单件数',
        value=current_data.get('units_per_order', 2.5),
        previous_value=previous_data.get('units_per_order', 2.4)
    )

    # 构建中间节点
    total_users = MetricNode(
        name='访问用户数',
        value=current_data.get('new_users', 10000) + current_data.get('old_users', 5000),
        previous_value=previous_data.get('new_users', 9500) + previous_data.get('old_users', 4800),
        children=[new_users, old_users],
        decomposition_type=DecompositionType.ADDITIVE
    )

    conversion_rate = MetricNode(
        name='转化率',
        value=current_data.get('view_to_cart', 0.15) * current_data.get('cart_to_pay', 0.30),
        previous_value=previous_data.get('view_to_cart', 0.14) * previous_data.get('cart_to_pay', 0.28),
        children=[view_to_cart, cart_to_pay],
        decomposition_type=DecompositionType.MULTIPLICATIVE
    )

    avg_order_value = MetricNode(
        name='客单价',
        value=current_data.get('unit_price', 50) * current_data.get('units_per_order', 2.5),
        previous_value=previous_data.get('unit_price', 48) * previous_data.get('units_per_order', 2.4),
        children=[unit_price, units_per_order],
        decomposition_type=DecompositionType.MULTIPLICATIVE
    )

    # 根节点 GMV
    gmv = MetricNode(
        name='GMV',
        value=total_users.value * conversion_rate.value * avg_order_value.value,
        previous_value=total_users.previous_value * conversion_rate.previous_value * avg_order_value.previous_value,
        children=[total_users, conversion_rate, avg_order_value],
        decomposition_type=DecompositionType.MULTIPLICATIVE
    )

    return MetricTree(gmv)


def waterfall_decomposition(tree: MetricTree) -> go.Figure:
    """
    瀑布图分解：展示各因素对变化的贡献
    """
    root = tree.root

    # 收集所有叶子节点的贡献
    contributions = []

    def collect_contributions(node: MetricNode, level: int = 0):
        if not node.children:  # 叶子节点
            if node.change is not None:
                contributions.append({
                    'name': node.name,
                    'change': node.change,
                    'value': node.value,
                    'previous': node.previous_value
                })
        else:
            for child in node.children:
                collect_contributions(child, level + 1)

    collect_contributions(root)

    # 计算贡献（简化为对数分解）
    total_change = root.change or 0

    # 构建瀑布图
    names = ['上期'] + [c['name'] for c in contributions] + ['本期']
    values = [root.previous_value]

    running_total = root.previous_value
    total_abs_change = root.value - root.previous_value if root.value and root.previous_value else 0

    for c in contributions:
        # 基于绝对变化值计算贡献
        # 每个子指标的贡献 = 子指标的绝对变化值
        if c.get('previous_value') is not None and c.get('value') is not None:
            contrib_value = c['value'] - c['previous_value']
        else:
            # 如果没有具体值，按变化率比例分配
            contrib_value = total_abs_change * (c['change'] if c['change'] else 0)
        values.append(contrib_value)
        running_total += contrib_value

    values.append(root.value)

    # 计算增减
    measures = ['absolute'] + ['relative'] * len(contributions) + ['total']

    fig = go.Figure(go.Waterfall(
        name="指标分解",
        orientation="v",
        measure=measures,
        x=names,
        textposition="outside",
        text=[f"{v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#27AE60"}},
        decreasing={"marker": {"color": "#EB5757"}},
        totals={"marker": {"color": "#2D9CDB"}}
    ))

    fig.update_layout(
        title=f"{root.name} 变化分解",
        showlegend=False,
        template='plotly_white',
        height=400
    )

    return fig


def plot_metric_tree(tree: MetricTree) -> go.Figure:
    """
    可视化指标树结构
    """
    # 简化的树形可视化
    fig = go.Figure()

    def add_node(node: MetricNode, x: float, y: float, level: int = 0):
        # 添加节点
        color = '#2D9CDB' if level == 0 else '#27AE60' if node.change and node.change > 0 else '#EB5757'
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=30 + (3 - level) * 10, color=color),
            text=[f"{node.name}<br>{node.value:,.0f}<br>({node.change*100:+.1f}%)" if node.change else f"{node.name}<br>{node.value:,.2f}"],
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        ))

        # 添加子节点
        n_children = len(node.children)
        if n_children > 0:
            child_spacing = 2 / (n_children + 1)
            for i, child in enumerate(node.children):
                child_x = x - 1 + (i + 1) * child_spacing
                child_y = y - 1

                # 添加连线
                fig.add_trace(go.Scatter(
                    x=[x, child_x], y=[y, child_y],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='none'
                ))

                add_node(child, child_x, child_y, level + 1)

    add_node(tree.root, 0, 3, 0)

    fig.update_layout(
        title='指标树结构',
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        template='plotly_white',
        height=500
    )

    return fig


def run_metric_tree_analysis(
    new_users: int,
    old_users: int,
    view_to_cart: float,
    cart_to_pay: float,
    unit_price: float,
    units_per_order: float
) -> Tuple[go.Figure, go.Figure, str]:
    """运行指标树分析"""

    # 上期数据（假设）
    previous_data = {
        'new_users': int(new_users * 0.95),
        'old_users': int(old_users * 0.95),
        'view_to_cart': view_to_cart * 0.95,
        'cart_to_pay': cart_to_pay * 0.95,
        'unit_price': unit_price * 0.98,
        'units_per_order': units_per_order * 0.98
    }

    current_data = {
        'new_users': new_users,
        'old_users': old_users,
        'view_to_cart': view_to_cart,
        'cart_to_pay': cart_to_pay,
        'unit_price': unit_price,
        'units_per_order': units_per_order
    }

    # 构建指标树
    tree = create_ecommerce_metric_tree(current_data, previous_data)

    # 可视化
    tree_fig = plot_metric_tree(tree)
    waterfall_fig = waterfall_decomposition(tree)

    # 生成报告
    root = tree.root
    contributions = tree.calculate_contribution()

    report = f"""
### 指标树分析报告

#### GMV 概况

| 指标 | 上期 | 本期 | 变化 |
|-----|-----|-----|------|
| GMV | ¥{root.previous_value:,.0f} | ¥{root.value:,.0f} | {root.change*100:+.1f}% |

#### 一级分解

| 因素 | 变化率 | 影响方向 |
|-----|-------|---------|
"""
    for child in root.children:
        direction = '↑' if child.change and child.change > 0 else '↓'
        report += f"| {child.name} | {child.change*100 if child.change else 0:+.1f}% | {direction} |\n"

    # 找到最大贡献因素（贡献度现在是百分比形式，范围 0-1）
    max_contrib = max(contributions.items(), key=lambda x: x[1])
    min_contrib = min(contributions.items(), key=lambda x: x[1])

    report += f"""
#### 关键发现

1. **最大正向贡献**: {max_contrib[0]} (贡献 {max_contrib[1]*100:.1f}% 的增长)
2. **需要关注**: {min_contrib[0]} (贡献 {min_contrib[1]*100:.1f}% 的增长)

#### 行动建议

1. 继续强化 {max_contrib[0]} 相关的运营策略
2. 深入分析 {min_contrib[0]} 下降的原因
3. 关注各环节的转化瓶颈
"""

    return tree_fig, waterfall_fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 指标树分析

### 什么是指标树？

指标树是将业务目标层层分解的分析框架：

```
GMV = 访问用户 × 转化率 × 客单价

         GMV
        / | \\
   用户数 转化率 客单价
   / \\    / \\    / \\
 新客 老客 ...  ...  ...
```

### 分解方法

| 方法 | 公式 | 适用场景 |
|-----|------|---------|
| **乘法分解** | Y = A × B × C | 转化漏斗、杜邦分析 |
| **加法分解** | Y = A + B + C | 用户分群、渠道拆分 |

---
        """)

        with gr.Row():
            with gr.Column():
                new_users = gr.Number(value=10000, label="新客数", precision=0)
                old_users = gr.Number(value=5000, label="老客数", precision=0)
                view_to_cart = gr.Slider(0.05, 0.30, 0.15, step=0.01, label="浏览→加购率")

            with gr.Column():
                cart_to_pay = gr.Slider(0.10, 0.50, 0.30, step=0.01, label="加购→支付率")
                unit_price = gr.Number(value=50, label="件单价")
                units_per_order = gr.Slider(1.0, 5.0, 2.5, step=0.1, label="每单件数")

        run_btn = gr.Button("运行分析", variant="primary")

        with gr.Row():
            tree_output = gr.Plot(label="指标树")

        with gr.Row():
            waterfall_output = gr.Plot(label="贡献分解")

        with gr.Row():
            report_output = gr.Markdown()

        run_btn.click(
            fn=run_metric_tree_analysis,
            inputs=[new_users, old_users, view_to_cart, cart_to_pay, unit_price, units_per_order],
            outputs=[tree_output, waterfall_output, report_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: GMV 下降，怎么分析？**
> 使用指标树逐层拆解：
> 1. GMV = 用户 × 转化 × 客单价
> 2. 找到下降最多的因素
> 3. 继续向下拆解
> 4. 找到可行动的叶子节点

**Q2: 杜邦分析法是什么？**
> 杜邦分析是财务指标的乘法分解：
> ROE = 净利润/股东权益
>     = (净利润/收入) × (收入/资产) × (资产/权益)
>     = 利润率 × 资产周转率 × 杠杆率

**Q3: 指标分解的陷阱？**
> 1. **Simpson's Paradox**: 整体下降但各部分上升
> 2. **基数效应**: 小基数的高增长贡献小
> 3. **相关性混淆**: A和B同时变化时归因困难

**Q4: 如何找到杠杆点？**
> 1. 找到"低挂的果实" (提升空间大、难度低)
> 2. 对比行业基准
> 3. 评估每个节点的可控性
        """)

    return None


if __name__ == "__main__":
    tree_fig, waterfall_fig, report = run_metric_tree_analysis(
        10000, 5000, 0.15, 0.30, 50, 2.5
    )
    print(report)
