"""
漏斗分析模块

业务背景：
---------
漏斗分析是用户转化路径的可视化：
1. 定义关键步骤（曝光→点击→加购→支付）
2. 计算每步转化率
3. 识别流失瓶颈
4. 指导优化方向

面试考点：
---------
- 如何用 SQL 计算漏斗转化率？
- 如何处理漏斗中的时间窗口？
- 如何分析漏斗的异质性？
- 漏斗分析的陷阱？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple


# SQL 模板
SQL_TEMPLATES = {
    'basic_funnel': """
-- 基础漏斗分析
-- 假设表结构: user_events(user_id, event_type, event_time)

WITH funnel_events AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS step1_view,
        MAX(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS step2_click,
        MAX(CASE WHEN event_type = 'add_cart' THEN 1 ELSE 0 END) AS step3_cart,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS step4_purchase
    FROM user_events
    WHERE event_time >= DATE_SUB(CURRENT_DATE, INTERVAL 7 DAY)
    GROUP BY user_id
)

SELECT
    'Page View' AS step,
    1 AS step_order,
    COUNT(*) AS users,
    100.0 AS conversion_rate
FROM funnel_events WHERE step1_view = 1

UNION ALL

SELECT
    'Click' AS step,
    2 AS step_order,
    SUM(step2_click) AS users,
    ROUND(SUM(step2_click) * 100.0 / COUNT(*), 2) AS conversion_rate
FROM funnel_events WHERE step1_view = 1

UNION ALL

SELECT
    'Add to Cart' AS step,
    3 AS step_order,
    SUM(step3_cart) AS users,
    ROUND(SUM(step3_cart) * 100.0 / COUNT(*), 2) AS conversion_rate
FROM funnel_events WHERE step1_view = 1

UNION ALL

SELECT
    'Purchase' AS step,
    4 AS step_order,
    SUM(step4_purchase) AS users,
    ROUND(SUM(step4_purchase) * 100.0 / COUNT(*), 2) AS conversion_rate
FROM funnel_events WHERE step1_view = 1

ORDER BY step_order;
""",

    'ordered_funnel': """
-- 严格顺序漏斗（用户必须按顺序完成）

WITH ordered_events AS (
    SELECT
        user_id,
        event_type,
        event_time,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) AS event_order
    FROM user_events
    WHERE event_type IN ('page_view', 'click', 'add_cart', 'purchase')
),

step1_cte AS (
    SELECT user_id, MIN(event_time) AS step1_time
    FROM ordered_events
    WHERE event_type = 'page_view'
    GROUP BY user_id
),
step2_cte AS (
    SELECT o.user_id, MIN(o.event_time) AS step2_time
    FROM ordered_events o
    JOIN step1_cte s1 ON o.user_id = s1.user_id
    WHERE o.event_type = 'click' AND o.event_time > s1.step1_time
    GROUP BY o.user_id
),
step3_cte AS (
    SELECT o.user_id, MIN(o.event_time) AS step3_time
    FROM ordered_events o
    JOIN step2_cte s2 ON o.user_id = s2.user_id
    WHERE o.event_type = 'add_cart' AND o.event_time > s2.step2_time
    GROUP BY o.user_id
),
step4_cte AS (
    SELECT o.user_id, MIN(o.event_time) AS step4_time
    FROM ordered_events o
    JOIN step3_cte s3 ON o.user_id = s3.user_id
    WHERE o.event_type = 'purchase' AND o.event_time > s3.step3_time
    GROUP BY o.user_id
),
funnel_steps AS (
    SELECT
        s1.user_id,
        s1.step1_time,
        s2.step2_time,
        s3.step3_time,
        s4.step4_time
    FROM step1_cte s1
    LEFT JOIN step2_cte s2 ON s1.user_id = s2.user_id
    LEFT JOIN step3_cte s3 ON s1.user_id = s3.user_id
    LEFT JOIN step4_cte s4 ON s1.user_id = s4.user_id
)

SELECT
    COUNT(*) AS total_users,
    SUM(CASE WHEN step1_time IS NOT NULL THEN 1 ELSE 0 END) AS step1_users,
    SUM(CASE WHEN step2_time IS NOT NULL THEN 1 ELSE 0 END) AS step2_users,
    SUM(CASE WHEN step3_time IS NOT NULL THEN 1 ELSE 0 END) AS step3_users,
    SUM(CASE WHEN step4_time IS NOT NULL THEN 1 ELSE 0 END) AS step4_users;
""",

    'time_window_funnel': """
-- 带时间窗口的漏斗（限制转化时间）
-- 例：从浏览到支付必须在 24 小时内

WITH funnel_with_time AS (
    SELECT
        user_id,
        MIN(CASE WHEN event_type = 'page_view' THEN event_time END) AS view_time,
        MIN(CASE WHEN event_type = 'purchase' THEN event_time END) AS purchase_time
    FROM user_events
    GROUP BY user_id
)

SELECT
    COUNT(*) AS total_viewers,
    SUM(CASE WHEN purchase_time IS NOT NULL
             AND purchase_time <= DATE_ADD(view_time, INTERVAL 24 HOUR)
        THEN 1 ELSE 0 END) AS converted_24h,
    ROUND(
        SUM(CASE WHEN purchase_time IS NOT NULL
                 AND purchase_time <= DATE_ADD(view_time, INTERVAL 24 HOUR)
            THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) AS conversion_rate_24h
FROM funnel_with_time
WHERE view_time IS NOT NULL;
""",

    'segment_funnel': """
-- 分群漏斗分析

WITH funnel_by_segment AS (
    SELECT
        u.user_segment,  -- 用户分群字段
        COUNT(DISTINCT e.user_id) AS total_users,
        COUNT(DISTINCT CASE WHEN e.event_type = 'page_view' THEN e.user_id END) AS viewers,
        COUNT(DISTINCT CASE WHEN e.event_type = 'click' THEN e.user_id END) AS clickers,
        COUNT(DISTINCT CASE WHEN e.event_type = 'add_cart' THEN e.user_id END) AS carters,
        COUNT(DISTINCT CASE WHEN e.event_type = 'purchase' THEN e.user_id END) AS purchasers
    FROM user_events e
    JOIN user_profiles u ON e.user_id = u.user_id
    GROUP BY u.user_segment
)

SELECT
    user_segment,
    viewers,
    clickers,
    carters,
    purchasers,
    ROUND(clickers * 100.0 / viewers, 2) AS view_to_click,
    ROUND(carters * 100.0 / clickers, 2) AS click_to_cart,
    ROUND(purchasers * 100.0 / carters, 2) AS cart_to_purchase,
    ROUND(purchasers * 100.0 / viewers, 2) AS overall_conversion
FROM funnel_by_segment
ORDER BY overall_conversion DESC;
"""
}


def generate_funnel_data(
    n_users: int = 10000,
    step_conversion: List[float] = [1.0, 0.4, 0.3, 0.5],
    seed: int = 42
) -> pd.DataFrame:
    """
    生成漏斗数据

    Parameters:
    -----------
    n_users: 用户数
    step_conversion: 每步转化率
    seed: 随机种子
    """
    np.random.seed(seed)

    steps = ['page_view', 'click', 'add_cart', 'purchase']

    records = []
    for user_id in range(n_users):
        current_step = 0
        user_segment = np.random.choice(['new', 'returning', 'vip'], p=[0.5, 0.35, 0.15])
        is_mobile = np.random.choice([0, 1], p=[0.4, 0.6])

        # 分群调整转化率
        segment_multiplier = {'new': 0.8, 'returning': 1.0, 'vip': 1.3}
        device_multiplier = 0.9 if is_mobile else 1.0

        for step_idx, (step, base_conv) in enumerate(zip(steps, step_conversion)):
            if step_idx == 0 or np.random.rand() < base_conv * segment_multiplier[user_segment] * device_multiplier:
                records.append({
                    'user_id': user_id,
                    'event_type': step,
                    'step_order': step_idx + 1,
                    'user_segment': user_segment,
                    'is_mobile': is_mobile
                })
                current_step = step_idx + 1
            else:
                break

    return pd.DataFrame(records)


def calculate_funnel(df: pd.DataFrame, segment: Optional[str] = None) -> pd.DataFrame:
    """计算漏斗转化率"""
    if segment:
        df = df[df['user_segment'] == segment]

    steps = ['page_view', 'click', 'add_cart', 'purchase']

    funnel_data = []
    total_users = df[df['event_type'] == 'page_view']['user_id'].nunique()

    for step_idx, step in enumerate(steps):
        step_users = df[df['event_type'] == step]['user_id'].nunique()

        funnel_data.append({
            'step': step,
            'step_order': step_idx + 1,
            'users': step_users,
            'conversion_from_start': step_users / total_users if total_users > 0 else 0,
            'conversion_from_prev': step_users / funnel_data[-1]['users'] if step_idx > 0 and funnel_data[-1]['users'] > 0 else 1.0
        })

    return pd.DataFrame(funnel_data)


def plot_funnel(funnel: pd.DataFrame, title: str = "转化漏斗") -> go.Figure:
    """绘制漏斗图"""
    fig = go.Figure(go.Funnel(
        y=funnel['step'],
        x=funnel['users'],
        textposition="inside",
        textinfo="value+percent previous",
        opacity=0.85,
        marker=dict(color=['#2D9CDB', '#27AE60', '#F2994A', '#9B59B6']),
        connector=dict(line=dict(color="royalblue", dash="dot", width=3))
    ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400
    )

    return fig


def plot_funnel_comparison(df: pd.DataFrame) -> go.Figure:
    """绘制分群漏斗对比"""
    segments = df['user_segment'].unique()

    fig = make_subplots(
        rows=1, cols=len(segments),
        subplot_titles=[f'{s} 用户' for s in segments]
    )

    colors = {'new': '#2D9CDB', 'returning': '#27AE60', 'vip': '#9B59B6'}

    for i, segment in enumerate(segments):
        funnel = calculate_funnel(df, segment)

        fig.add_trace(
            go.Bar(
                x=funnel['step'],
                y=funnel['conversion_from_start'] * 100,
                name=segment,
                marker_color=colors.get(segment, '#6B7280'),
                text=[f"{v:.1f}%" for v in funnel['conversion_from_start'] * 100],
                textposition='outside'
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        title='分群漏斗对比',
        showlegend=False,
        template='plotly_white',
        height=400
    )

    for i in range(len(segments)):
        fig.update_yaxes(title_text='转化率 (%)', row=1, col=i+1)

    return fig


def identify_bottleneck(funnel: pd.DataFrame) -> str:
    """识别漏斗瓶颈"""
    # 找到转化率最低的步骤
    min_idx = funnel['conversion_from_prev'].idxmin()
    bottleneck_step = funnel.loc[min_idx, 'step']
    bottleneck_rate = funnel.loc[min_idx, 'conversion_from_prev']

    # 与行业基准对比（假设）
    benchmarks = {
        'click': 0.5,
        'add_cart': 0.4,
        'purchase': 0.6
    }

    gap = benchmarks.get(bottleneck_step, 0.5) - bottleneck_rate

    return f"**瓶颈步骤**: {bottleneck_step} (转化率 {bottleneck_rate*100:.1f}%)\n\n与基准差距: {gap*100:.1f}%"


def run_funnel_analysis(
    n_users: int,
    view_to_click: float,
    click_to_cart: float,
    cart_to_purchase: float,
    seed: int
) -> Tuple[go.Figure, go.Figure, str, str]:
    """运行漏斗分析"""
    # 生成数据
    df = generate_funnel_data(
        n_users=n_users,
        step_conversion=[1.0, view_to_click/100, click_to_cart/100, cart_to_purchase/100],
        seed=seed
    )

    # 计算漏斗
    funnel = calculate_funnel(df)

    # 可视化
    funnel_fig = plot_funnel(funnel)
    comparison_fig = plot_funnel_comparison(df)

    # 分析报告
    bottleneck = identify_bottleneck(funnel)

    # 分群统计
    segment_stats = df.groupby('user_segment').apply(
        lambda x: x[x['event_type'] == 'purchase']['user_id'].nunique() / x[x['event_type'] == 'page_view']['user_id'].nunique()
    ).reset_index()
    segment_stats.columns = ['segment', 'conversion']

    report = f"""
### 漏斗分析报告

#### 整体转化

| 步骤 | 用户数 | 从起点转化 | 从上一步转化 |
|-----|-------|----------|------------|
| 浏览 | {funnel.iloc[0]['users']:,} | 100% | - |
| 点击 | {funnel.iloc[1]['users']:,} | {funnel.iloc[1]['conversion_from_start']*100:.1f}% | {funnel.iloc[1]['conversion_from_prev']*100:.1f}% |
| 加购 | {funnel.iloc[2]['users']:,} | {funnel.iloc[2]['conversion_from_start']*100:.1f}% | {funnel.iloc[2]['conversion_from_prev']*100:.1f}% |
| 支付 | {funnel.iloc[3]['users']:,} | {funnel.iloc[3]['conversion_from_start']*100:.1f}% | {funnel.iloc[3]['conversion_from_prev']*100:.1f}% |

#### 瓶颈分析

{bottleneck}

#### 分群对比

| 用户分群 | 整体转化率 |
|---------|----------|
"""
    for _, row in segment_stats.iterrows():
        report += f"| {row['segment']} | {row['conversion']*100:.1f}% |\n"

    report += """
#### 优化建议

1. **针对瓶颈步骤**: 深入分析用户流失原因
2. **分群策略**: 对不同用户群采取差异化策略
3. **A/B 测试**: 对关键页面进行优化实验
"""

    return funnel_fig, comparison_fig, report, SQL_TEMPLATES['segment_funnel']


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 漏斗分析

### 什么是漏斗分析？

漏斗分析将用户行为路径可视化，帮助识别转化瓶颈：

```
浏览 (10000) → 点击 (4000) → 加购 (1200) → 支付 (600)
   100%          40%          12%          6%
```

---

### 漏斗类型

| 类型 | 说明 | 适用场景 |
|-----|------|---------|
| **宽松漏斗** | 只要完成过某步骤就算 | 多入口场景 |
| **严格漏斗** | 必须按顺序完成 | 单一路径 |
| **时间窗口漏斗** | 限制转化时间 | 即时转化分析 |

---
        """)

        with gr.Row():
            with gr.Column():
                n_users = gr.Number(value=10000, label="用户数", precision=0)
                view_to_click = gr.Slider(10, 80, 40, step=5, label="浏览→点击 (%)")
                click_to_cart = gr.Slider(10, 80, 30, step=5, label="点击→加购 (%)")
                cart_to_purchase = gr.Slider(20, 80, 50, step=5, label="加购→支付 (%)")
                seed = gr.Number(value=42, label="随机种子", precision=0)
                run_btn = gr.Button("运行分析", variant="primary")

        with gr.Row():
            with gr.Column():
                funnel_output = gr.Plot(label="转化漏斗")
            with gr.Column():
                comparison_output = gr.Plot(label="分群对比")

        with gr.Row():
            report_output = gr.Markdown()

        with gr.Row():
            sql_output = gr.Code(label="SQL 模板", language="sql")

        run_btn.click(
            fn=run_funnel_analysis,
            inputs=[n_users, view_to_click, click_to_cart, cart_to_purchase, seed],
            outputs=[funnel_output, comparison_output, report_output, sql_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: 漏斗分析的 SQL 怎么写？**
> 关键是使用 CASE WHEN 和 COUNT DISTINCT：
```sql
SELECT
    COUNT(DISTINCT user_id) AS step1,
    COUNT(DISTINCT CASE WHEN clicked THEN user_id END) AS step2,
    COUNT(DISTINCT CASE WHEN purchased THEN user_id END) AS step3
FROM user_events;
```

**Q2: 如何处理跳步行为？**
> 1. **宽松漏斗**: 允许跳步，只看是否完成
> 2. **严格漏斗**: 必须按顺序，用窗口函数判断
> 3. **实际分析**: 区分"正常跳步"和"异常跳步"

**Q3: 漏斗分析的陷阱？**
> 1. **样本选择偏差**: 只看完成的用户
> 2. **时间窗口**: 不同转化时长混在一起
> 3. **Simpson's 悖论**: 分群后结论相反
> 4. **幸存者偏差**: 只分析活跃用户

**Q4: 如何优化转化率？**
> 1. 找到最大的流失步骤
> 2. 分析流失用户的特征
> 3. 对比高转化用户的行为
> 4. 设计 A/B 测试验证假设
        """)

    return None


if __name__ == "__main__":
    funnel, comparison, report, sql = run_funnel_analysis(10000, 40, 30, 50, 42)
    print(report)
