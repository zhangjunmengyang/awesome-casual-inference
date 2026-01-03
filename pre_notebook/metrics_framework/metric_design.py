"""
指标设计方法论

业务背景：
---------
数据科学家的重要职责是帮助业务定义正确的指标：
1. 什么是好的指标？
2. 如何避免虚荣指标？
3. 北极星指标怎么选？
4. 实验指标怎么设计？

核心概念：
---------
1. 北极星指标 (North Star Metric)
   - 公司级别的核心指标
   - 反映核心价值交付
   - 例：Airbnb 的"预订夜数"，Facebook 的"DAU"

2. 过程指标 (Process Metrics)
   - 驱动北极星指标的中间过程
   - 例：转化率、留存率、活跃度

3. 护栏指标 (Guardrail Metrics)
   - 防止优化过程中损害其他方面
   - 例：页面加载时间、退款率

4. 反指标 (Counter Metrics)
   - 避免局部优化损害整体
   - 例：推送过多→卸载率

面试考点：
---------
- 如何选择北极星指标？
- 虚荣指标的特征？
- 指标之间的权衡？
- 如何设计实验指标？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """指标类型"""
    NORTH_STAR = "north_star"  # 北极星指标
    INPUT = "input"  # 输入指标
    OUTPUT = "output"  # 输出指标
    GUARDRAIL = "guardrail"  # 护栏指标
    COUNTER = "counter"  # 反指标


class MetricGranularity(Enum):
    """指标粒度"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    REAL_TIME = "real_time"


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    description: str
    metric_type: MetricType
    formula: str
    numerator: str
    denominator: Optional[str] = None
    data_source: str = ""
    owner: str = ""
    granularity: MetricGranularity = MetricGranularity.DAILY
    direction: str = "higher_is_better"  # or "lower_is_better"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'type': self.metric_type.value,
            'formula': self.formula,
            'direction': self.direction,
            'granularity': self.granularity.value,
        }


class MetricEvaluator:
    """指标质量评估器"""

    GOOD_METRIC_CRITERIA = {
        'measurable': '可测量：能够准确计算',
        'understandable': '可理解：业务人员能理解含义',
        'actionable': '可行动：能指导具体行动',
        'timely': '及时性：能够快速获取',
        'comparable': '可比较：可以跨时间/群体对比',
        'sensitive': '敏感性：能检测到变化',
        'robust': '稳健性：不易被操纵',
    }

    def evaluate(self, metric: MetricDefinition) -> Dict[str, float]:
        """
        评估指标质量

        返回各维度得分 (0-1)
        """
        scores = {}

        # 简化评估（实际应该有更复杂的逻辑）
        scores['measurable'] = 1.0 if metric.formula else 0.5
        scores['understandable'] = 1.0 if len(metric.description) > 20 else 0.5
        scores['actionable'] = 0.8  # 需要人工判断
        scores['timely'] = 1.0 if metric.granularity == MetricGranularity.DAILY else 0.7
        scores['comparable'] = 1.0 if metric.denominator else 0.6
        scores['sensitive'] = 0.7  # 需要数据验证
        scores['robust'] = 0.8 if metric.metric_type != MetricType.INPUT else 0.6

        return scores

    def get_recommendation(self, scores: Dict[str, float]) -> str:
        """基于评估给出建议"""
        weak_areas = [k for k, v in scores.items() if v < 0.7]

        if not weak_areas:
            return "✅ 这是一个高质量的指标"

        recommendations = []
        for area in weak_areas:
            if area == 'measurable':
                recommendations.append("• 确保指标公式清晰可计算")
            elif area == 'understandable':
                recommendations.append("• 添加更详细的业务描述")
            elif area == 'actionable':
                recommendations.append("• 明确指标变化时的行动方案")
            elif area == 'timely':
                recommendations.append("• 考虑提高数据更新频率")
            elif area == 'comparable':
                recommendations.append("• 使用比率指标便于对比")
            elif area == 'sensitive':
                recommendations.append("• 验证指标对变化的敏感性")
            elif area == 'robust':
                recommendations.append("• 警惕指标被人为操纵的风险")

        return "⚠️ 改进建议：\n" + "\n".join(recommendations)


class VanityMetricDetector:
    """虚荣指标检测器"""

    VANITY_SIGNALS = [
        ('累计值', '累计指标只会增长，无法反映当前状态'),
        ('无分母', '绝对数字无法反映效率'),
        ('易操纵', '容易通过刷量等方式提升'),
        ('无行动', '变化后不知道该做什么'),
        ('无对比', '无法与历史或竞品对比'),
    ]

    def detect(self, metric: MetricDefinition) -> List[Dict]:
        """检测虚荣指标信号"""
        warnings = []

        # 检查累计值
        if '累计' in metric.name or 'total' in metric.name.lower():
            warnings.append({
                'signal': '累计值',
                'reason': '累计指标只会增长，建议改用增量或比率',
                'severity': 'high'
            })

        # 检查是否有分母
        if not metric.denominator:
            warnings.append({
                'signal': '无分母',
                'reason': '绝对数字难以对比，建议使用比率指标',
                'severity': 'medium'
            })

        # 检查输入指标风险
        if metric.metric_type == MetricType.INPUT:
            warnings.append({
                'signal': '输入指标',
                'reason': '输入指标容易被操纵，建议关注输出指标',
                'severity': 'medium'
            })

        return warnings


# 预定义的指标模板库
METRIC_TEMPLATES = {
    'e_commerce': [
        MetricDefinition(
            name='GMV',
            description='总成交金额，反映平台交易规模',
            metric_type=MetricType.NORTH_STAR,
            formula='SUM(order_amount)',
            numerator='订单金额总和',
            denominator=None,
            direction='higher_is_better',
            tags=['电商', '核心']
        ),
        MetricDefinition(
            name='转化率',
            description='访问用户中完成购买的比例',
            metric_type=MetricType.OUTPUT,
            formula='购买用户数 / 访问用户数',
            numerator='购买用户数',
            denominator='访问用户数',
            direction='higher_is_better',
            tags=['电商', '转化']
        ),
        MetricDefinition(
            name='客单价',
            description='平均每个订单的金额',
            metric_type=MetricType.OUTPUT,
            formula='GMV / 订单数',
            numerator='GMV',
            denominator='订单数',
            direction='higher_is_better',
            tags=['电商', '客单']
        ),
        MetricDefinition(
            name='复购率',
            description='购买用户中再次购买的比例',
            metric_type=MetricType.OUTPUT,
            formula='复购用户数 / 购买用户数',
            numerator='复购用户数',
            denominator='购买用户数',
            direction='higher_is_better',
            tags=['电商', '留存']
        ),
        MetricDefinition(
            name='退款率',
            description='订单中退款的比例（护栏指标）',
            metric_type=MetricType.GUARDRAIL,
            formula='退款订单数 / 总订单数',
            numerator='退款订单数',
            denominator='总订单数',
            direction='lower_is_better',
            tags=['电商', '护栏']
        ),
    ],
    'saas': [
        MetricDefinition(
            name='MRR',
            description='月度经常性收入，SaaS 核心指标',
            metric_type=MetricType.NORTH_STAR,
            formula='SUM(monthly_subscription)',
            numerator='月订阅收入总和',
            denominator=None,
            direction='higher_is_better',
            tags=['SaaS', '核心']
        ),
        MetricDefinition(
            name='Churn Rate',
            description='客户流失率',
            metric_type=MetricType.GUARDRAIL,
            formula='流失客户数 / 期初客户数',
            numerator='流失客户数',
            denominator='期初客户数',
            direction='lower_is_better',
            tags=['SaaS', '留存']
        ),
        MetricDefinition(
            name='NPS',
            description='净推荐值，用户满意度指标',
            metric_type=MetricType.OUTPUT,
            formula='推荐者比例 - 贬损者比例',
            numerator='推荐者比例 - 贬损者比例',
            denominator=None,
            direction='higher_is_better',
            tags=['SaaS', '满意度']
        ),
    ],
    'content': [
        MetricDefinition(
            name='DAU',
            description='日活跃用户数',
            metric_type=MetricType.NORTH_STAR,
            formula='COUNT(DISTINCT active_users)',
            numerator='活跃用户数',
            denominator=None,
            direction='higher_is_better',
            tags=['内容', '核心']
        ),
        MetricDefinition(
            name='用户时长',
            description='用户平均使用时长',
            metric_type=MetricType.OUTPUT,
            formula='总时长 / DAU',
            numerator='总使用时长',
            denominator='DAU',
            direction='higher_is_better',
            tags=['内容', '参与度']
        ),
        MetricDefinition(
            name='次日留存',
            description='新用户次日返回的比例',
            metric_type=MetricType.OUTPUT,
            formula='次日返回用户 / 新用户',
            numerator='次日返回用户数',
            denominator='新用户数',
            direction='higher_is_better',
            tags=['内容', '留存']
        ),
    ]
}


def create_metric_dashboard(industry: str) -> Tuple[go.Figure, str]:
    """创建指标仪表盘"""
    metrics = METRIC_TEMPLATES.get(industry, [])

    if not metrics:
        return go.Figure(), "未找到该行业的指标模板"

    evaluator = MetricEvaluator()
    detector = VanityMetricDetector()

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '指标类型分布',
            '指标质量评分',
            '指标关系图',
            '虚荣指标检测'
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )

    # 指标类型分布
    type_counts = {}
    for m in metrics:
        t = m.metric_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    fig.add_trace(
        go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.4
        ),
        row=1, col=1
    )

    # 指标质量评分
    metric_names = [m.name for m in metrics]
    avg_scores = []
    for m in metrics:
        scores = evaluator.evaluate(m)
        avg_scores.append(np.mean(list(scores.values())) * 100)

    fig.add_trace(
        go.Bar(
            x=metric_names,
            y=avg_scores,
            marker_color=['#27AE60' if s > 70 else '#F2994A' if s > 50 else '#EB5757'
                          for s in avg_scores]
        ),
        row=1, col=2
    )

    # 简化的指标关系图
    fig.add_trace(
        go.Scatter(
            x=[0, 1, 2, 0.5, 1.5],
            y=[1, 1, 1, 0, 0],
            mode='markers+text',
            text=metric_names[:5],
            textposition='top center',
            marker=dict(size=20, color='#2D9CDB')
        ),
        row=2, col=1
    )

    # 虚荣指标检测
    warning_counts = []
    for m in metrics:
        warnings = detector.detect(m)
        warning_counts.append(len(warnings))

    fig.add_trace(
        go.Bar(
            x=metric_names,
            y=warning_counts,
            marker_color=['#EB5757' if w > 0 else '#27AE60' for w in warning_counts]
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )

    # 生成报告
    report = f"""
### {industry.upper()} 行业指标体系

#### 指标清单

| 指标 | 类型 | 公式 | 方向 | 质量分 |
|-----|------|------|-----|-------|
"""
    for m, score in zip(metrics, avg_scores):
        report += f"| {m.name} | {m.metric_type.value} | {m.formula} | {'↑' if 'higher' in m.direction else '↓'} | {score:.0f} |\n"

    report += """
#### 建议

1. **北极星指标**: 作为公司级目标，需要全员对齐
2. **过程指标**: 分解北极星指标，指导日常工作
3. **护栏指标**: 设置阈值，防止过度优化
4. **定期审视**: 每季度评估指标体系的有效性
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 指标设计方法论

### 好指标的特征 (SMART+)

| 特征 | 说明 | 反例 |
|-----|------|------|
| **S**pecific | 定义清晰，无歧义 | "用户活跃度" |
| **M**easurable | 可准确计算 | 主观评分 |
| **A**ctionable | 能指导行动 | 无法改变的指标 |
| **R**elevant | 与业务目标相关 | 与核心目标无关 |
| **T**imely | 能及时获取 | 月度才能算的指标 |
| **+Robust** | 不易被操纵 | 点击量（刷量） |

---

### 虚荣指标 vs 行动指标

| 虚荣指标 | 行动指标 |
|---------|---------|
| 注册用户总数 | 月活跃用户数 |
| 页面浏览量 | 转化率 |
| 下载量 | 激活率 |
| 点赞数 | 留存率 |

---
        """)

        with gr.Row():
            industry = gr.Radio(
                choices=['e_commerce', 'saas', 'content'],
                value='e_commerce',
                label="选择行业"
            )
            run_btn = gr.Button("生成指标体系", variant="primary")

        with gr.Row():
            plot_output = gr.Plot()

        with gr.Row():
            report_output = gr.Markdown()

        run_btn.click(
            fn=create_metric_dashboard,
            inputs=[industry],
            outputs=[plot_output, report_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: 如何选择北极星指标？**
> 北极星指标应该：
> 1. 反映核心价值交付
> 2. 可分解为过程指标
> 3. 全公司可对齐
> 例：Airbnb → 预订夜数，Facebook → DAU

**Q2: GMV 是好指标吗？**
> GMV 作为结果指标可以，但需注意：
> - 是否包含退款？
> - 是否区分自营和平台？
> - 需要搭配利润率等指标

**Q3: 如何避免指标造假？**
> 1. 使用输出指标而非输入指标
> 2. 设置护栏指标
> 3. 交叉验证
> 4. 关注长期指标（如留存）

**Q4: 实验应该看哪些指标？**
> - **主指标**: 实验假设相关的核心指标
> - **次要指标**: 辅助理解效果的指标
> - **护栏指标**: 确保不损害其他方面
> - **诊断指标**: 帮助 debug 的指标
        """)

    return None


if __name__ == "__main__":
    fig, report = create_metric_dashboard('e_commerce')
    print(report)
