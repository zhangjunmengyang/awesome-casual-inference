"""
用户增长归因案例

业务背景：
---------
互联网公司增长团队需要回答：
1. 各获客渠道的真实 ROI 是多少？（渠道归因）
2. 不同渠道获取用户的 LTV 差异？
3. 如何优化获客预算分配？

核心挑战：
---------
- 归因难题：用户可能接触多个渠道，Last-click 归因是否合理？
- 选择偏差：优质渠道可能吸引本身就高价值的用户
- 长期价值：短期转化 vs 长期 LTV

方法论：
-------
1. 增量归因：通过实验或因果推断估计渠道真实贡献
2. LTV 预测：基于用户特征预测长期价值
3. CAC/LTV 优化：找到最优获客策略

面试考点：
---------
- 什么是 MTA (Multi-Touch Attribution)？
- Shapley Value 归因怎么算？
- 如何预测用户 LTV？
- CAC 和 LTV 的关系是什么？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ChannelMetrics:
    """渠道指标"""
    channel: str
    users: int
    spend: float
    cac: float  # Cost per Acquisition
    day7_retention: float
    day30_retention: float
    avg_ltv_30: float
    avg_ltv_90: float
    roi_30: float
    roi_90: float


def generate_growth_data(
    n_users: int = 30000,
    observation_days: int = 90,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成用户增长数据

    Returns:
    --------
    (user_df, event_df): 用户表和事件表
    """
    np.random.seed(seed)

    # 获客渠道
    channels = ['organic', 'paid_search', 'paid_social', 'referral', 'content']
    channel_probs = [0.25, 0.30, 0.25, 0.10, 0.10]
    channel_cpc = {'organic': 0, 'paid_search': 2.5, 'paid_social': 1.8, 'referral': 0.5, 'content': 0.3}

    # 用户获客渠道
    user_channel = np.random.choice(channels, n_users, p=channel_probs)

    # 用户内在质量（不可观测）- 影响渠道选择和后续行为
    user_quality = np.random.beta(2, 5, n_users)

    # 渠道对用户质量有选择效应
    # organic 和 referral 用户质量更高
    quality_boost = {
        'organic': 0.15,
        'paid_search': 0.05,
        'paid_social': -0.05,
        'referral': 0.20,
        'content': 0.10
    }

    for i, ch in enumerate(user_channel):
        user_quality[i] += quality_boost[ch]
    user_quality = np.clip(user_quality, 0, 1)

    # 用户特征
    age = np.random.normal(30, 8, n_users).clip(18, 60)
    is_mobile = np.random.binomial(1, 0.7, n_users)
    signup_day = np.random.randint(0, 60, n_users)  # 前60天内注册

    # 留存概率（依赖于用户质量和渠道）
    base_retention_7 = 0.3 + 0.4 * user_quality
    base_retention_30 = 0.15 + 0.3 * user_quality

    # 渠道对留存的因果效应
    channel_retention_effect = {
        'organic': 0.05,
        'paid_search': 0.00,
        'paid_social': -0.03,
        'referral': 0.08,
        'content': 0.03
    }

    retention_7_prob = base_retention_7.copy()
    retention_30_prob = base_retention_30.copy()

    for i, ch in enumerate(user_channel):
        retention_7_prob[i] += channel_retention_effect[ch]
        retention_30_prob[i] += channel_retention_effect[ch] * 0.8

    retention_7_prob = np.clip(retention_7_prob, 0, 1)
    retention_30_prob = np.clip(retention_30_prob, 0, 1)

    retained_7 = np.random.binomial(1, retention_7_prob)
    retained_30 = np.random.binomial(1, retention_30_prob) * retained_7  # 30天留存前提是7天留存

    # LTV 生成
    # 基础 LTV 依赖于用户质量
    base_ltv = np.random.exponential(50, n_users) * (1 + 2 * user_quality)

    # 渠道对 LTV 的因果效应
    channel_ltv_multiplier = {
        'organic': 1.3,
        'paid_search': 1.1,
        'paid_social': 0.9,
        'referral': 1.4,
        'content': 1.2
    }

    ltv_30 = np.zeros(n_users)
    ltv_90 = np.zeros(n_users)

    for i in range(n_users):
        ch = user_channel[i]
        days_observed = observation_days - signup_day[i]

        if days_observed >= 30 and retained_30[i]:
            ltv_30[i] = base_ltv[i] * 0.4 * channel_ltv_multiplier[ch]
        if days_observed >= 90:
            ltv_90[i] = base_ltv[i] * channel_ltv_multiplier[ch]

        # 添加噪声
        ltv_30[i] *= np.random.lognormal(0, 0.3)
        ltv_90[i] *= np.random.lognormal(0, 0.3)

    # 触点数据（多触点）
    touchpoints = []
    for i in range(n_users):
        n_touches = np.random.poisson(2) + 1  # 至少1个触点
        touches = np.random.choice(channels, min(n_touches, 5), replace=True)
        # 确保最后一个触点是获客渠道
        touches = list(touches[:-1]) + [user_channel[i]]
        touchpoints.append(touches)

    # 构建用户表
    user_df = pd.DataFrame({
        'user_id': range(n_users),
        'acquisition_channel': user_channel,
        'age': age,
        'is_mobile': is_mobile,
        'signup_day': signup_day,
        'retained_7': retained_7,
        'retained_30': retained_30,
        'ltv_30': ltv_30,
        'ltv_90': ltv_90,
        'touchpoints': touchpoints,
        '_user_quality': user_quality,  # 隐藏变量
    })

    # 计算渠道花费
    spend_by_channel = {}
    for ch in channels:
        n_users_ch = (user_channel == ch).sum()
        # 假设 5:1 的点击转化率
        clicks = n_users_ch * 5
        spend_by_channel[ch] = clicks * channel_cpc[ch]

    user_df['_channel_spend'] = [spend_by_channel[ch] for ch in user_channel]

    # 构建事件表（简化）
    events = []
    for i in range(n_users):
        if retained_7[i]:
            for day in range(1, 8):
                if np.random.rand() < 0.3:
                    events.append({
                        'user_id': i,
                        'event_type': 'active',
                        'event_day': signup_day[i] + day
                    })
        if ltv_30[i] > 0:
            n_purchases = max(1, int(ltv_30[i] / 30))
            for _ in range(n_purchases):
                events.append({
                    'user_id': i,
                    'event_type': 'purchase',
                    'event_day': signup_day[i] + np.random.randint(1, 31),
                    'revenue': ltv_30[i] / n_purchases
                })

    event_df = pd.DataFrame(events)

    return user_df, event_df


class MultiTouchAttributor:
    """多触点归因模型"""

    def __init__(self, method: str = 'shapley'):
        """
        Parameters:
        -----------
        method: 归因方法
            - 'last_click': 最后触点归因
            - 'first_click': 首次触点归因
            - 'linear': 线性平均归因
            - 'shapley': Shapley Value 归因
            - 'position': 位置加权归因
        """
        self.method = method

    def attribute(self, touchpoints: List[List[str]], conversions: np.ndarray,
                  values: np.ndarray) -> Dict[str, float]:
        """
        计算各渠道归因价值

        Parameters:
        -----------
        touchpoints: 每个用户的触点序列
        conversions: 是否转化 (0/1)
        values: 转化价值

        Returns:
        --------
        各渠道的归因价值
        """
        all_channels = set()
        for tp in touchpoints:
            all_channels.update(tp)

        attribution = {ch: 0.0 for ch in all_channels}

        for i, (tp, conv, val) in enumerate(zip(touchpoints, conversions, values)):
            if not conv or val <= 0:
                continue

            if self.method == 'last_click':
                attribution[tp[-1]] += val

            elif self.method == 'first_click':
                attribution[tp[0]] += val

            elif self.method == 'linear':
                share = val / len(tp)
                for ch in tp:
                    attribution[ch] += share

            elif self.method == 'position':
                # U 型：首尾 40%，中间平分 20%
                n = len(tp)
                if n == 1:
                    attribution[tp[0]] += val
                elif n == 2:
                    attribution[tp[0]] += val * 0.5
                    attribution[tp[1]] += val * 0.5
                else:
                    attribution[tp[0]] += val * 0.4
                    attribution[tp[-1]] += val * 0.4
                    middle_share = val * 0.2 / (n - 2)
                    for ch in tp[1:-1]:
                        attribution[ch] += middle_share

            elif self.method == 'shapley':
                attribution = self._shapley_attribute(tp, val, attribution)

        return attribution

    def _shapley_attribute(self, touchpoints: List[str], value: float,
                           current_attribution: Dict) -> Dict:
        """Shapley Value 归因"""
        unique_channels = list(set(touchpoints))
        n = len(unique_channels)

        if n == 0:
            return current_attribution

        # 简化：假设所有渠道贡献相等的边际价值
        # 真实场景需要基于转化率建模
        shapley_values = {ch: value / n for ch in unique_channels}

        for ch, sv in shapley_values.items():
            current_attribution[ch] += sv

        return current_attribution


class LTVPredictor:
    """LTV 预测模型"""

    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        self.feature_importance = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str = 'ltv_90'):
        """训练 LTV 预测模型"""
        X = df[feature_cols].values
        y = df[target_col].values

        self.model.fit(X, y)
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))

        return self

    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """预测 LTV"""
        X = df[feature_cols].values
        return self.model.predict(X)

    def segment_users(self, df: pd.DataFrame, predicted_ltv: np.ndarray,
                      n_segments: int = 4) -> np.ndarray:
        """基于预测 LTV 分群"""
        percentiles = np.percentile(predicted_ltv, np.linspace(0, 100, n_segments + 1))
        segments = np.digitize(predicted_ltv, percentiles[1:-1])
        return segments


def calculate_channel_metrics(user_df: pd.DataFrame) -> List[ChannelMetrics]:
    """计算各渠道指标"""
    channels = user_df['acquisition_channel'].unique()
    metrics = []

    for ch in channels:
        ch_users = user_df[user_df['acquisition_channel'] == ch]
        n_users = len(ch_users)

        if n_users == 0:
            continue

        total_spend = ch_users['_channel_spend'].iloc[0] if '_channel_spend' in ch_users.columns else 0

        metrics.append(ChannelMetrics(
            channel=ch,
            users=n_users,
            spend=total_spend,
            cac=total_spend / n_users if n_users > 0 else 0,
            day7_retention=ch_users['retained_7'].mean(),
            day30_retention=ch_users['retained_30'].mean(),
            avg_ltv_30=ch_users['ltv_30'].mean(),
            avg_ltv_90=ch_users['ltv_90'].mean(),
            roi_30=(ch_users['ltv_30'].sum() - total_spend) / total_spend if total_spend > 0 else 0,
            roi_90=(ch_users['ltv_90'].sum() - total_spend) / total_spend if total_spend > 0 else 0,
        ))

    return sorted(metrics, key=lambda x: x.roi_90, reverse=True)


def run_growth_analysis(
    n_users: int = 30000,
    attribution_method: str = 'shapley',
    seed: int = 42
) -> Tuple[go.Figure, str]:
    """运行增长归因分析"""

    # 1. 生成数据
    user_df, event_df = generate_growth_data(n_users=n_users, seed=seed)

    # 2. 计算渠道指标
    channel_metrics = calculate_channel_metrics(user_df)

    # 3. 多触点归因
    attributor = MultiTouchAttributor(method=attribution_method)
    conversions = (user_df['ltv_30'] > 0).astype(int).values
    values = user_df['ltv_30'].values
    touchpoints = user_df['touchpoints'].tolist()

    attribution = attributor.attribute(touchpoints, conversions, values)

    # Last-click 对比
    lc_attributor = MultiTouchAttributor(method='last_click')
    lc_attribution = lc_attributor.attribute(touchpoints, conversions, values)

    # 4. LTV 预测
    feature_cols = ['age', 'is_mobile', 'retained_7']
    # 添加渠道 dummy 变量
    for ch in user_df['acquisition_channel'].unique():
        user_df[f'ch_{ch}'] = (user_df['acquisition_channel'] == ch).astype(int)
        feature_cols.append(f'ch_{ch}')

    ltv_predictor = LTVPredictor()
    ltv_predictor.fit(user_df, feature_cols, 'ltv_90')
    predicted_ltv = ltv_predictor.predict(user_df, feature_cols)
    user_df['predicted_ltv'] = predicted_ltv
    user_df['ltv_segment'] = ltv_predictor.segment_users(user_df, predicted_ltv)

    # 5. 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '各渠道 ROI 对比',
            '归因方法对比 (Last-click vs Shapley)',
            'LTV 分布 by 渠道',
            'CAC vs LTV (气泡图)'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    colors = ['#27AE60', '#2D9CDB', '#9B59B6', '#F2994A', '#EB5757']

    # 图1：ROI 对比
    channels = [m.channel for m in channel_metrics]
    roi_30 = [m.roi_30 * 100 for m in channel_metrics]
    roi_90 = [m.roi_90 * 100 for m in channel_metrics]

    fig.add_trace(
        go.Bar(x=channels, y=roi_30, name='30天 ROI', marker_color='#2D9CDB'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=channels, y=roi_90, name='90天 ROI', marker_color='#27AE60'),
        row=1, col=1
    )

    # 图2：归因对比
    attr_channels = list(attribution.keys())
    shapley_values = [attribution.get(ch, 0) for ch in attr_channels]
    lc_values = [lc_attribution.get(ch, 0) for ch in attr_channels]

    fig.add_trace(
        go.Bar(x=attr_channels, y=lc_values, name='Last-click', marker_color='#EB5757'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=attr_channels, y=shapley_values, name='Shapley', marker_color='#27AE60'),
        row=1, col=2
    )

    # 图3：LTV 分布
    for i, ch in enumerate(channels[:5]):
        ch_data = user_df[user_df['acquisition_channel'] == ch]['ltv_90']
        fig.add_trace(
            go.Box(y=ch_data, name=ch, marker_color=colors[i % len(colors)]),
            row=2, col=1
        )

    # 图4：CAC vs LTV 气泡图
    cac_values = [m.cac for m in channel_metrics]
    ltv_values = [m.avg_ltv_90 for m in channel_metrics]
    user_counts = [m.users for m in channel_metrics]

    fig.add_trace(
        go.Scatter(
            x=cac_values,
            y=ltv_values,
            mode='markers+text',
            text=channels,
            textposition='top center',
            marker=dict(
                size=[u / 500 for u in user_counts],
                color=colors[:len(channels)],
                opacity=0.7
            ),
            name='渠道'
        ),
        row=2, col=2
    )

    # 添加 LTV = CAC 参考线
    max_val = max(max(cac_values), max(ltv_values))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='LTV = CAC'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        template='plotly_white',
        title_text='用户增长归因分析',
        title_x=0.5,
        barmode='group'
    )

    fig.update_xaxes(title_text='渠道', row=1, col=1)
    fig.update_yaxes(title_text='ROI (%)', row=1, col=1)
    fig.update_xaxes(title_text='渠道', row=1, col=2)
    fig.update_yaxes(title_text='归因价值 (¥)', row=1, col=2)
    fig.update_yaxes(title_text='LTV (¥)', row=2, col=1)
    fig.update_xaxes(title_text='CAC (¥)', row=2, col=2)
    fig.update_yaxes(title_text='LTV (¥)', row=2, col=2)

    # 6. 生成报告
    report = f"""
### 用户增长归因报告

#### 1. 数据概况
- 分析用户数: {n_users:,}
- 观察周期: 90 天
- 获客渠道: {len(channels)} 个

#### 2. 渠道效果排名 (按90天ROI)

| 渠道 | 用户数 | CAC | 7日留存 | 30日留存 | 90日LTV | ROI |
|-----|-------|-----|--------|---------|--------|-----|
"""
    for m in channel_metrics:
        report += f"| {m.channel} | {m.users:,} | ¥{m.cac:.1f} | {m.day7_retention*100:.1f}% | {m.day30_retention*100:.1f}% | ¥{m.avg_ltv_90:.0f} | {m.roi_90*100:.0f}% |\n"

    report += f"""
#### 3. 归因分析

**Last-click vs {attribution_method.title()} 归因对比:**

| 渠道 | Last-click | {attribution_method.title()} | 差异 |
|-----|-----------|---------|------|
"""
    for ch in attr_channels:
        lc = lc_attribution.get(ch, 0)
        sh = attribution.get(ch, 0)
        diff = (sh - lc) / lc * 100 if lc > 0 else 0
        report += f"| {ch} | ¥{lc:,.0f} | ¥{sh:,.0f} | {diff:+.1f}% |\n"

    report += f"""
#### 4. 关键发现

1. **最高 ROI 渠道**: {channel_metrics[0].channel} ({channel_metrics[0].roi_90*100:.0f}% ROI)
2. **最高 LTV 渠道**: {max(channel_metrics, key=lambda x: x.avg_ltv_90).channel}
3. **最低 CAC 渠道**: {min(channel_metrics, key=lambda x: x.cac if x.cac > 0 else float('inf')).channel}

#### 5. 业务建议

1. **增加投放**: {channel_metrics[0].channel} 和 {channel_metrics[1].channel} 的 ROI 最高
2. **优化或减少**: {channel_metrics[-1].channel} 的 ROI 最低，需分析原因
3. **关注长期价值**: 部分渠道短期 ROI 低但长期价值高
4. **归因调整**: Last-click 可能低估了 {max(attr_channels, key=lambda x: attribution[x] - lc_attribution.get(x, 0))} 的贡献

---
*基于因果推断的增长归因分析*
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 用户增长归因

### 业务背景

增长团队的核心问题：
- 哪个渠道的获客效率最高？
- 如何在渠道间分配预算？
- 用户的长期价值 (LTV) 如何预测？

**常见挑战**：
- 选择偏差：优质渠道可能吸引本身就高价值的用户
- 多触点：用户可能接触多个渠道后才转化
- 短视偏见：只看即时转化，忽略长期价值

---

### 核心概念

| 概念 | 定义 | 应用 |
|-----|------|------|
| **CAC** | Customer Acquisition Cost，获客成本 | 渠道效率评估 |
| **LTV** | Lifetime Value，用户生命周期价值 | 用户价值评估 |
| **MTA** | Multi-Touch Attribution，多触点归因 | 渠道贡献分配 |
| **Shapley Value** | 博弈论公平分配方法 | 多渠道归因 |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_users = gr.Slider(10000, 50000, 30000, step=5000, label="用户数量")
                attribution_method = gr.Radio(
                    choices=['shapley', 'last_click', 'first_click', 'linear', 'position'],
                    value='shapley',
                    label="归因方法"
                )
                seed = gr.Number(value=42, label="随机种子", precision=0)
                run_btn = gr.Button("运行分析", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="分析可视化")

        with gr.Row():
            report_output = gr.Markdown()

        run_btn.click(
            fn=run_growth_analysis,
            inputs=[n_users, attribution_method, seed],
            outputs=[plot_output, report_output]
        )

        gr.Markdown("""
---

### 归因方法对比

| 方法 | 原理 | 优点 | 缺点 |
|-----|------|------|------|
| **Last-click** | 全部归因给最后触点 | 简单 | 低估上游渠道 |
| **First-click** | 全部归因给首次触点 | 强调获客 | 忽略转化渠道 |
| **Linear** | 均匀分配 | 公平 | 忽略位置差异 |
| **Position** | U型分配 (首尾40%+中间20%) | 平衡 | 假设固定 |
| **Shapley** | 博弈论边际贡献 | 理论完备 | 计算复杂 |

---

### 面试常见问题

**Q1: LTV 怎么计算？**
> LTV = ARPU × 用户生命周期
> 或: LTV = Σ(月收入 × 留存率^月份)

**Q2: CAC/LTV 比例多少合适？**
> 通常目标是 LTV/CAC > 3
> SaaS 行业可接受 LTV/CAC > 3，回收期 < 12 个月

**Q3: 如何处理多触点？**
> 1. 规则方法：Last-click, Linear 等
> 2. 数据驱动：Markov Chain, Shapley Value
> 3. 因果方法：增量实验

**Q4: Organic 渠道怎么归因？**
> Organic 没有直接成本，但有机会成本
> 可通过品牌实验测量品牌贡献
        """)

    return None


if __name__ == "__main__":
    fig, report = run_growth_analysis()
    print(report)
