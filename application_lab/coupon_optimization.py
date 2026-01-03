"""
智能发券优化模块

场景: 外卖/电商平台优惠券分配优化
目标: 最大化 GMV 增量，最小化补贴浪费
方法: Uplift 建模识别敏感用户

参考: DoorDash, Meituan, Uber Eats
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

from .utils import generate_marketing_data, compute_roi


class UpliftCouponOptimizer:
    """
    优惠券 Uplift 优化器

    使用 Two-Model (T-Learner) 方法估计 CATE
    """

    def __init__(self):
        self.model_treatment = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model_control = GradientBoostingClassifier(n_estimators=100, random_state=43)
        self.feature_cols = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练 Uplift 模型"""
        # 控制组模型
        mask_control = T == 0
        self.model_control.fit(X[mask_control], Y[mask_control])

        # 处理组模型
        mask_treatment = T == 1
        self.model_treatment.fit(X[mask_treatment], Y[mask_treatment])

        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """预测 Uplift"""
        prob_treatment = self.model_treatment.predict_proba(X)[:, 1]
        prob_control = self.model_control.predict_proba(X)[:, 1]

        uplift = prob_treatment - prob_control
        return uplift


def segment_users_by_uplift(
    df: pd.DataFrame,
    uplift_scores: np.ndarray,
    percentile_high: float = 75,
    percentile_low: float = 25
) -> pd.DataFrame:
    """
    根据 Uplift 分数将用户分成 4 组

    - Persuadables: 高 Uplift，发券有效
    - Sure Things: 低 Uplift + 高基线转化，发券浪费
    - Lost Causes: 低 Uplift + 低基线转化，发券无效
    - Sleeping Dogs: 负 Uplift，发券反而有害

    Parameters:
    -----------
    df: 原始数据
    uplift_scores: Uplift 预测分数
    percentile_high: 高 Uplift 阈值百分位
    percentile_low: 低 Uplift 阈值百分位

    Returns:
    --------
    带有 segment 列的 DataFrame
    """
    df = df.copy()
    df['uplift_score'] = uplift_scores

    # 计算基线转化率 (控制组)
    mask_control = df['T'] == 0
    if mask_control.sum() > 0:
        # 基线转化率应该使用控制组 (未发券用户) 的转化率
        df['baseline_conversion'] = df.loc[mask_control, 'conversion'].mean()
    else:
        df['baseline_conversion'] = 0.15

    # 分组规则
    uplift_high = np.percentile(uplift_scores, percentile_high)
    uplift_low = np.percentile(uplift_scores, percentile_low)
    baseline_median = df['baseline_conversion'].median()

    conditions = [
        (uplift_scores >= uplift_high),  # Persuadables
        (uplift_scores < 0),  # Sleeping Dogs
        (uplift_scores < uplift_low) & (df['baseline_conversion'] >= baseline_median),  # Sure Things
        (uplift_scores < uplift_low) & (df['baseline_conversion'] < baseline_median),  # Lost Causes
    ]

    choices = ['Persuadables', 'Sleeping Dogs', 'Sure Things', 'Lost Causes']
    df['segment'] = np.select(conditions, choices, default='Others')

    return df


def optimize_coupon_allocation(
    n_samples: int,
    budget_fraction: float,
    model_quality: str
) -> Tuple[go.Figure, str]:
    """
    优惠券分配优化主函数

    Parameters:
    -----------
    n_samples: 样本数量
    budget_fraction: 预算比例 (发券比例)
    model_quality: 模型质量 ('perfect', 'good', 'poor')

    Returns:
    --------
    (figure, summary_markdown)
    """
    # 生成数据
    df, true_uplift, gmv_lift = generate_marketing_data(n_samples)

    # 准备特征
    feature_cols = ['age', 'avg_order_value', 'order_frequency', 'days_since_last_order',
                    'is_member', 'app_sessions', 'city_tier']
    X = df[feature_cols].values
    T = df['T'].values
    Y = df['conversion'].values

    # 训练模型
    optimizer = UpliftCouponOptimizer()
    optimizer.fit(X, T, Y)

    # 预测 Uplift
    predicted_uplift = optimizer.predict_uplift(X)

    # 添加噪声模拟不同模型质量
    noise_levels = {'perfect': 0.0, 'good': 0.5, 'poor': 1.5}
    noise = noise_levels.get(model_quality, 0.5)
    predicted_uplift = predicted_uplift + np.random.randn(len(predicted_uplift)) * noise * predicted_uplift.std()

    # 用户分组
    df_segmented = segment_users_by_uplift(df, predicted_uplift)

    # === 计算不同策略的 ROI ===
    strategies = {
        'Random': np.random.randn(n_samples),
        'Uplift Model': predicted_uplift,
        'High Frequency': df['order_frequency'].values,  # 传统策略: 针对高频用户
    }

    strategy_results = []
    for strategy_name, scores in strategies.items():
        roi_metrics = compute_roi(
            df,
            scores,
            target_fraction=budget_fraction,
            revenue_per_conversion=100,
            cost_per_treatment=15
        )
        roi_metrics['strategy'] = strategy_name
        strategy_results.append(roi_metrics)

    results_df = pd.DataFrame(strategy_results)

    # === 可视化 ===
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '用户分群分布',
            'ROI 对比',
            'Uplift Score 分布',
            '预算分配优化曲线'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'histogram'}, {'type': 'scatter'}]
        ]
    )

    # 1. 用户分群分布
    segment_counts = df_segmented['segment'].value_counts()
    segment_colors = {
        'Persuadables': '#27AE60',
        'Sure Things': '#2D9CDB',
        'Lost Causes': '#95A5A6',
        'Sleeping Dogs': '#EB5757',
        'Others': '#BDC3C7'
    }

    colors = [segment_colors.get(seg, '#BDC3C7') for seg in segment_counts.index]

    fig.add_trace(go.Bar(
        x=segment_counts.index,
        y=segment_counts.values,
        marker_color=colors,
        text=[f'{v/len(df)*100:.1f}%' for v in segment_counts.values],
        textposition='outside',
        name='User Count'
    ), row=1, col=1)

    # 2. ROI 对比
    fig.add_trace(go.Bar(
        x=results_df['strategy'],
        y=results_df['roi'],
        marker_color=['#95A5A6', '#27AE60', '#2D9CDB'],
        text=[f'{v:.2f}' for v in results_df['roi']],
        textposition='outside',
        name='ROI'
    ), row=1, col=2)

    # 3. Uplift Score 分布
    for segment in ['Persuadables', 'Sleeping Dogs', 'Others']:
        mask = df_segmented['segment'] == segment
        if mask.sum() > 0:
            fig.add_trace(go.Histogram(
                x=df_segmented.loc[mask, 'uplift_score'],
                name=segment,
                marker_color=segment_colors.get(segment, '#BDC3C7'),
                opacity=0.6,
                nbinsx=30
            ), row=2, col=1)

    # 4. 预算分配优化曲线
    fractions = np.linspace(0.05, 1.0, 20)
    roi_curve = []

    for frac in fractions:
        roi_metrics = compute_roi(
            df,
            predicted_uplift,
            target_fraction=frac,
            revenue_per_conversion=100,
            cost_per_treatment=15
        )
        roi_curve.append(roi_metrics.get('roi', 0))

    fig.add_trace(go.Scatter(
        x=fractions * 100,
        y=roi_curve,
        mode='lines+markers',
        name='ROI Curve',
        line=dict(color='#9B59B6', width=3),
        marker=dict(size=6)
    ), row=2, col=2)

    # 标记最优点
    best_idx = np.argmax(roi_curve)
    best_fraction = fractions[best_idx]
    best_roi = roi_curve[best_idx]

    fig.add_trace(go.Scatter(
        x=[best_fraction * 100],
        y=[best_roi],
        mode='markers+text',
        marker=dict(color='red', size=15, symbol='star'),
        text=[f'Optimal: {best_fraction*100:.0f}%'],
        textposition='top center',
        name='Optimal Point',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='智能发券优化分析',
        showlegend=True,
        barmode='overlay'
    )

    fig.update_xaxes(title_text='User Segment', row=1, col=1)
    fig.update_xaxes(title_text='Strategy', row=1, col=2)
    fig.update_xaxes(title_text='Uplift Score', row=2, col=1)
    fig.update_xaxes(title_text='Budget Allocation (%)', row=2, col=2)

    fig.update_yaxes(title_text='User Count', row=1, col=1)
    fig.update_yaxes(title_text='ROI', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    fig.update_yaxes(title_text='ROI', row=2, col=2)

    # === 生成摘要 ===
    uplift_model_roi = results_df[results_df['strategy'] == 'Uplift Model']['roi'].values[0]
    random_roi = results_df[results_df['strategy'] == 'Random']['roi'].values[0]
    high_freq_roi = results_df[results_df['strategy'] == 'High Frequency']['roi'].values[0]

    persuadables_pct = (df_segmented['segment'] == 'Persuadables').sum() / len(df) * 100
    sleeping_dogs_pct = (df_segmented['segment'] == 'Sleeping Dogs').sum() / len(df) * 100

    uplift_model_revenue = results_df[results_df['strategy'] == 'Uplift Model']['revenue'].values[0]
    uplift_model_cost = results_df[results_df['strategy'] == 'Uplift Model']['cost'].values[0]

    summary = f"""
### 优化结果摘要

#### 核心指标

| 指标 | 值 |
|------|-----|
| 总样本数 | {n_samples:,} |
| 预算比例 | {budget_fraction*100:.0f}% |
| Persuadables 占比 | {persuadables_pct:.1f}% |
| Sleeping Dogs 占比 | {sleeping_dogs_pct:.1f}% |
| 最优发券比例 | {best_fraction*100:.0f}% |

#### ROI 对比

| 策略 | ROI | 提升 |
|------|-----|------|
| Uplift Model | {uplift_model_roi:.2f} | - |
| High Frequency | {high_freq_roi:.2f} | {(uplift_model_roi - high_freq_roi) / abs(high_freq_roi) * 100:.1f}% |
| Random | {random_roi:.2f} | {(uplift_model_roi - random_roi) / abs(random_roi) * 100:.1f}% |

#### 财务影响

| 项目 | 金额 (元) |
|------|----------|
| 预期收入 | {uplift_model_revenue:,.0f} |
| 发券成本 | {uplift_model_cost:,.0f} |
| 净利润 | {uplift_model_revenue - uplift_model_cost:,.0f} |

### 关键洞察

1. **Uplift 模型优势**: 相比随机发券，ROI 提升 {(uplift_model_roi - random_roi) / abs(random_roi) * 100:.1f}%
2. **精准识别**: {persuadables_pct:.1f}% 用户是 Persuadables，重点发券
3. **避免浪费**: {sleeping_dogs_pct:.1f}% 用户是 Sleeping Dogs，不应发券
4. **最优预算**: 建议发券比例为 {best_fraction*100:.0f}%，而非全量发放

### 业务建议

- 优先发券给 Persuadables 用户 (低频、年轻、非会员)
- 避免发券给 Sleeping Dogs 和 Sure Things
- 定期重新训练模型，捕捉用户行为变化
- 结合业务规则 (如用户生命周期阶段) 进一步优化
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 智能发券优化 (Coupon Optimization)

基于 Uplift Modeling 的优惠券分配优化，最大化 ROI。

### 业务场景

外卖/电商平台每天面临的核心问题:
- 给谁发券效果最好?
- 如何避免补贴浪费?
- 最优预算分配是多少?

### 用户分群策略

| 用户类型 | 特征 | 策略 |
|---------|------|------|
| **Persuadables** | 高 Uplift | 重点发券 |
| **Sure Things** | 本来就会买 | 不发券 (浪费) |
| **Lost Causes** | 无论如何不会买 | 不发券 |
| **Sleeping Dogs** | 负 Uplift | 千万别发券! |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=5000, maximum=20000, value=10000, step=1000,
                    label="样本量"
                )
                budget_fraction = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                    label="预算比例 (发券比例)"
                )
                model_quality = gr.Radio(
                    choices=['perfect', 'good', 'poor'],
                    value='good',
                    label="模型质量"
                )
                run_btn = gr.Button("运行优化", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("""
### 实际案例参考

**DoorDash**: 使用 Uplift Modeling 优化促销活动，ROI 提升 30%+

**Meituan**: 智能发券系统每年节省数亿补贴成本

**关键指标**:
- ROI = (Revenue - Cost) / Cost
- Uplift = P(conversion|coupon) - P(conversion|no coupon)
                """)

        with gr.Row():
            plot_output = gr.Plot(label="优化分析")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=optimize_coupon_allocation,
            inputs=[n_samples, budget_fraction, model_quality],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 技术细节

**Uplift 估计方法**: Two-Model (T-Learner)
```
model_control.fit(X[T==0], Y[T==0])
model_treatment.fit(X[T==1], Y[T==1])
uplift = model_treatment.predict(X) - model_control.predict(X)
```

**分群规则**:
- Persuadables: uplift > p75
- Sleeping Dogs: uplift < 0
- Sure Things: uplift < p25 AND baseline_conversion > median
- Lost Causes: uplift < p25 AND baseline_conversion < median

### 扩展阅读

- [Uplift Modeling at Uber](https://eng.uber.com/uplift-modeling/)
- [Netflix Experimentation Platform](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15)
        """)

    return None
