"""
留存分析模块

业务背景：
---------
留存是衡量产品粘性的核心指标：
1. N日留存：新用户在第N天返回的比例
2. 同期群留存：按注册时间分组的留存趋势
3. 滚动留存：在第N天及之后返回的比例

SQL 模板：
---------
提供标准的留存计算 SQL，适配各种数据仓库

面试考点：
---------
- 写出计算次日留存的 SQL
- 如何处理同期群分析？
- 留存和活跃的区别？
- 如何诊断留存下降？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


# SQL 模板
SQL_TEMPLATES = {
    'day1_retention': """
-- 次日留存率
-- 假设表结构: user_events(user_id, event_date, event_type)

WITH new_users AS (
    -- 新用户：首次活跃日期
    SELECT
        user_id,
        MIN(event_date) AS first_date
    FROM user_events
    GROUP BY user_id
),

day1_active AS (
    -- 次日活跃用户
    SELECT DISTINCT
        n.user_id,
        n.first_date
    FROM new_users n
    JOIN user_events e
        ON n.user_id = e.user_id
        AND e.event_date = DATE_ADD(n.first_date, INTERVAL 1 DAY)
)

SELECT
    n.first_date AS cohort_date,
    COUNT(DISTINCT n.user_id) AS new_users,
    COUNT(DISTINCT d.user_id) AS retained_users,
    ROUND(COUNT(DISTINCT d.user_id) * 100.0 / COUNT(DISTINCT n.user_id), 2) AS retention_rate
FROM new_users n
LEFT JOIN day1_active d ON n.user_id = d.user_id AND n.first_date = d.first_date
GROUP BY n.first_date
ORDER BY n.first_date;
""",

    'n_day_retention': """
-- N日留存率（参数化）
-- 参数: @n = 留存天数

WITH new_users AS (
    SELECT
        user_id,
        MIN(event_date) AS first_date
    FROM user_events
    GROUP BY user_id
),

day_n_active AS (
    SELECT DISTINCT
        n.user_id,
        n.first_date
    FROM new_users n
    JOIN user_events e
        ON n.user_id = e.user_id
        AND e.event_date = DATE_ADD(n.first_date, INTERVAL @n DAY)
)

SELECT
    n.first_date AS cohort_date,
    COUNT(DISTINCT n.user_id) AS new_users,
    COUNT(DISTINCT d.user_id) AS retained_users,
    ROUND(COUNT(DISTINCT d.user_id) * 100.0 / COUNT(DISTINCT n.user_id), 2) AS day_n_retention
FROM new_users n
LEFT JOIN day_n_active d ON n.user_id = d.user_id AND n.first_date = d.first_date
GROUP BY n.first_date
ORDER BY n.first_date;
""",

    'cohort_retention': """
-- 同期群留存分析
-- 输出: 留存矩阵

WITH new_users AS (
    SELECT
        user_id,
        MIN(event_date) AS first_date,
        DATE_TRUNC('week', MIN(event_date)) AS cohort_week  -- 按周分组
    FROM user_events
    GROUP BY user_id
),

user_activity AS (
    SELECT
        n.user_id,
        n.cohort_week,
        n.first_date,
        e.event_date,
        DATEDIFF(e.event_date, n.first_date) AS days_since_signup,
        FLOOR(DATEDIFF(e.event_date, n.first_date) / 7) AS weeks_since_signup
    FROM new_users n
    JOIN user_events e ON n.user_id = e.user_id
)

SELECT
    cohort_week,
    weeks_since_signup,
    COUNT(DISTINCT user_id) AS active_users,
    FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
        PARTITION BY cohort_week
        ORDER BY weeks_since_signup
    ) AS cohort_size,
    ROUND(
        COUNT(DISTINCT user_id) * 100.0 /
        FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
            PARTITION BY cohort_week
            ORDER BY weeks_since_signup
        ),
        2
    ) AS retention_rate
FROM user_activity
GROUP BY cohort_week, weeks_since_signup
ORDER BY cohort_week, weeks_since_signup;
""",

    'rolling_retention': """
-- 滚动留存（第N天及之后返回）
-- 比经典留存更稳健

WITH new_users AS (
    SELECT
        user_id,
        MIN(event_date) AS first_date
    FROM user_events
    GROUP BY user_id
),

return_activity AS (
    SELECT DISTINCT
        n.user_id,
        n.first_date,
        MIN(e.event_date) AS first_return_date
    FROM new_users n
    JOIN user_events e
        ON n.user_id = e.user_id
        AND e.event_date >= DATE_ADD(n.first_date, INTERVAL @n DAY)
    GROUP BY n.user_id, n.first_date
)

SELECT
    n.first_date AS cohort_date,
    COUNT(DISTINCT n.user_id) AS new_users,
    COUNT(DISTINCT r.user_id) AS returned_users,
    ROUND(COUNT(DISTINCT r.user_id) * 100.0 / COUNT(DISTINCT n.user_id), 2) AS rolling_retention
FROM new_users n
LEFT JOIN return_activity r ON n.user_id = r.user_id
WHERE n.first_date <= DATE_SUB(CURRENT_DATE, INTERVAL @n DAY)  -- 足够观察期
GROUP BY n.first_date
ORDER BY n.first_date;
"""
}


def generate_retention_data(
    n_users: int = 10000,
    n_days: int = 30,
    base_retention: float = 0.4,
    decay_rate: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成留存数据

    Parameters:
    -----------
    n_users: 用户数
    n_days: 观察天数
    base_retention: 基础留存率
    decay_rate: 衰减率
    seed: 随机种子
    """
    np.random.seed(seed)

    records = []

    for cohort_day in range(n_days):
        # 该天的新用户数
        n_new = int(n_users / n_days * (1 + 0.2 * np.random.randn()))

        # 用户留存（逐日衰减）
        for user_idx in range(n_new):
            user_id = f"U{cohort_day:02d}_{user_idx:05d}"

            # 首日活跃
            records.append({
                'user_id': user_id,
                'cohort_date': f'2024-01-{cohort_day+1:02d}',
                'event_date': f'2024-01-{cohort_day+1:02d}',
                'days_since_signup': 0
            })

            # 后续是否留存
            for day_offset in range(1, n_days - cohort_day):
                # 留存概率随天数衰减
                retention_prob = base_retention * np.exp(-decay_rate * day_offset)
                # 添加一些随机性
                retention_prob *= (1 + 0.1 * np.random.randn())
                retention_prob = np.clip(retention_prob, 0, 1)

                if np.random.rand() < retention_prob:
                    event_day = cohort_day + day_offset + 1
                    if event_day <= n_days:
                        records.append({
                            'user_id': user_id,
                            'cohort_date': f'2024-01-{cohort_day+1:02d}',
                            'event_date': f'2024-01-{event_day:02d}',
                            'days_since_signup': day_offset
                        })

    return pd.DataFrame(records)


def calculate_retention(df: pd.DataFrame) -> pd.DataFrame:
    """计算留存率"""
    # 每个同期群的用户数
    cohort_sizes = df[df['days_since_signup'] == 0].groupby('cohort_date')['user_id'].nunique()

    # 每个同期群每天的活跃用户
    retention = df.groupby(['cohort_date', 'days_since_signup'])['user_id'].nunique().reset_index()
    retention.columns = ['cohort_date', 'day', 'active_users']

    # 计算留存率
    retention['cohort_size'] = retention['cohort_date'].map(cohort_sizes)
    retention['retention_rate'] = retention['active_users'] / retention['cohort_size']

    return retention


def plot_retention_curve(retention: pd.DataFrame) -> go.Figure:
    """绘制留存曲线"""
    # 平均留存曲线
    avg_retention = retention.groupby('day')['retention_rate'].mean()

    fig = go.Figure()

    # 平均留存曲线
    fig.add_trace(go.Scatter(
        x=avg_retention.index,
        y=avg_retention.values * 100,
        mode='lines+markers',
        name='平均留存率',
        line=dict(color='#2D9CDB', width=3),
        marker=dict(size=8)
    ))

    # 添加关键节点标注
    key_days = [1, 7, 14, 30]
    for day in key_days:
        if day in avg_retention.index:
            fig.add_annotation(
                x=day,
                y=avg_retention[day] * 100,
                text=f"D{day}: {avg_retention[day]*100:.1f}%",
                showarrow=True,
                arrowhead=2
            )

    fig.update_layout(
        title='用户留存曲线',
        xaxis_title='注册后天数',
        yaxis_title='留存率 (%)',
        template='plotly_white',
        height=400
    )

    return fig


def plot_retention_heatmap(retention: pd.DataFrame) -> go.Figure:
    """绘制留存热力图"""
    # 转为宽格式
    pivot = retention.pivot(index='cohort_date', columns='day', values='retention_rate')

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=[f'D{d}' for d in pivot.columns],
        y=pivot.index,
        colorscale='RdYlGn',
        zmin=0,
        zmax=100,
        text=np.round(pivot.values * 100, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title='留存率 (%)')
    ))

    fig.update_layout(
        title='同期群留存热力图',
        xaxis_title='注册后天数',
        yaxis_title='同期群',
        template='plotly_white',
        height=500
    )

    return fig


def run_retention_analysis(
    n_users: int,
    n_days: int,
    base_retention: float,
    seed: int
) -> Tuple[go.Figure, go.Figure, str, str]:
    """运行留存分析"""
    # 生成数据
    df = generate_retention_data(
        n_users=n_users,
        n_days=n_days,
        base_retention=base_retention / 100,
        seed=seed
    )

    # 计算留存
    retention = calculate_retention(df)

    # 可视化
    curve_fig = plot_retention_curve(retention)
    heatmap_fig = plot_retention_heatmap(retention)

    # 关键指标
    avg_retention = retention.groupby('day')['retention_rate'].mean()
    d1 = avg_retention.get(1, 0) * 100
    d7 = avg_retention.get(7, 0) * 100
    d30 = avg_retention.get(min(29, n_days-1), 0) * 100

    report = f"""
### 留存分析报告

#### 关键指标

| 指标 | 值 |
|-----|-----|
| 总用户数 | {df['user_id'].nunique():,} |
| 观察天数 | {n_days} |
| **次日留存 (D1)** | **{d1:.1f}%** |
| **7日留存 (D7)** | **{d7:.1f}%** |
| **30日留存 (D30)** | **{d30:.1f}%** |

#### 解读

- 次日留存 {d1:.1f}% {'✅ 良好' if d1 > 30 else '⚠️ 偏低，需要优化新用户体验'}
- 7日留存 {d7:.1f}% {'✅ 良好' if d7 > 15 else '⚠️ 偏低，需要增强用户粘性'}
- 留存曲线在 D3-D7 快速下降，之后趋于稳定

#### 建议

1. **新用户引导**: 优化首日体验，提升 D1 留存
2. **价值传递**: 在前 7 天内让用户体验核心功能
3. **召回机制**: 对流失用户进行定向触达
"""

    return curve_fig, heatmap_fig, report, SQL_TEMPLATES['cohort_retention']


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 留存分析

### 为什么留存重要？

> "留存是最重要的增长杠杆" — Casey Winters (Pinterest)

- **获客成本**: 留住老用户比获取新用户便宜 5-25 倍
- **复利效应**: 留存率提高 5%，利润可提高 25-95%
- **产品健康**: 留存反映产品真正的价值

---

### 留存指标类型

| 类型 | 定义 | 适用场景 |
|-----|------|---------|
| **经典留存** | 第 N 天恰好返回的用户比例 | 高频产品（日活） |
| **滚动留存** | 第 N 天及之后返回的用户比例 | 低频产品（周活） |
| **区间留存** | 第 N-M 天内返回的用户比例 | 灵活定义 |

---
        """)

        with gr.Row():
            with gr.Column():
                n_users = gr.Number(value=10000, label="用户数", precision=0)
                n_days = gr.Number(value=30, label="观察天数", precision=0)
                base_retention = gr.Slider(20, 60, 40, step=5, label="基础留存率 (%)")
                seed = gr.Number(value=42, label="随机种子", precision=0)
                run_btn = gr.Button("运行分析", variant="primary")

        with gr.Row():
            with gr.Column():
                curve_output = gr.Plot(label="留存曲线")
            with gr.Column():
                heatmap_output = gr.Plot(label="同期群热力图")

        with gr.Row():
            report_output = gr.Markdown()

        with gr.Row():
            sql_output = gr.Code(label="SQL 模板", language="sql")

        run_btn.click(
            fn=run_retention_analysis,
            inputs=[n_users, n_days, base_retention, seed],
            outputs=[curve_output, heatmap_output, report_output, sql_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: 写一个计算次日留存的 SQL**
```sql
SELECT
    DATE(first_date) AS cohort_date,
    COUNT(DISTINCT CASE WHEN day1_active THEN user_id END) /
    COUNT(DISTINCT user_id) AS d1_retention
FROM (
    SELECT
        user_id,
        MIN(event_date) AS first_date,
        MAX(CASE WHEN event_date = DATE_ADD(first_date, 1) THEN 1 END) AS day1_active
    FROM user_events
    GROUP BY user_id
) t
GROUP BY DATE(first_date);
```

**Q2: 留存下降怎么分析？**
> 1. 按渠道拆分：是否某个渠道的留存特别差？
> 2. 按版本拆分：是否某个版本更新导致？
> 3. 按用户特征拆分：是否某类用户留存差？
> 4. 查看功能使用：核心功能的使用率是否下降？

**Q3: 如何提升留存？**
> 1. **Aha Moment**: 找到让用户"顿悟"的关键行为
> 2. **引导流程**: 优化新用户首日体验
> 3. **召回策略**: 对流失用户进行触达
> 4. **习惯养成**: 设计每日任务、打卡机制
        """)

    return None


if __name__ == "__main__":
    curve, heatmap, report, sql = run_retention_analysis(10000, 30, 40, 42)
    print(report)
