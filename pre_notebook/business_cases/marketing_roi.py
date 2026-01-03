"""
智能营销 ROI 优化案例

业务背景：
---------
电商平台每月有 1000 万营销预算，需要决定：
1. 预算如何在不同渠道/活动间分配？
2. 哪些用户应该被触达？（人群圈选）
3. 每个渠道的真实增量贡献是多少？（增量归因）

核心挑战：
---------
- 选择偏差：活跃用户更可能被营销触达，也更可能转化
- 多触点归因：用户可能被多个渠道触达
- 预算约束：资源有限，需要最大化 ROI

方法论：
-------
1. 增量测量：使用因果推断估计真实增量效果
2. 人群优化：基于 CATE 识别高敏感人群
3. 预算分配：基于边际 ROI 优化分配

面试考点：
---------
- 如何设计增量实验？
- PSA (Public Service Announcement) 实验是什么？
- 如何处理多触点归因？
- 预算分配的优化目标是什么？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class CampaignResult:
    """营销活动结果"""
    channel: str
    spend: float
    impressions: int
    conversions: int
    revenue: float
    naive_roas: float  # 简单 ROAS
    incremental_conversions: float  # 增量转化
    incremental_revenue: float  # 增量收入
    true_roas: float  # 真实增量 ROAS


def generate_marketing_data(
    n_users: int = 50000,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成营销数据

    模拟真实场景：
    - 用户有不同特征（活跃度、历史消费、人口属性）
    - 营销触达有选择偏差（活跃用户更可能被触达）
    - 不同用户对营销的敏感度不同（CATE 异质性）
    """
    np.random.seed(seed)

    # 用户特征
    user_activity = np.random.beta(2, 5, n_users)  # 活跃度 [0,1]
    historical_spend = np.random.exponential(200, n_users)  # 历史消费
    days_since_last = np.random.exponential(30, n_users)  # 距上次消费天数
    age = np.random.normal(35, 10, n_users).clip(18, 65)
    is_member = np.random.binomial(1, 0.3, n_users)  # 会员状态

    # 渠道触达（有选择偏差）
    # Push 通知：活跃用户更可能收到
    push_propensity = 1 / (1 + np.exp(-(
        -1 + 2 * user_activity + 0.5 * is_member
    )))
    push_treated = np.random.binomial(1, push_propensity)

    # Email：历史消费高的用户更可能收到
    email_propensity = 1 / (1 + np.exp(-(
        -1.5 + 0.003 * historical_spend + 0.3 * is_member
    )))
    email_treated = np.random.binomial(1, email_propensity)

    # SMS：近期不活跃用户更可能收到（召回）
    sms_propensity = 1 / (1 + np.exp(-(
        -2 + 0.02 * days_since_last - user_activity
    )))
    sms_treated = np.random.binomial(1, sms_propensity)

    # 基线转化概率（无营销）
    baseline_prob = 1 / (1 + np.exp(-(
        -3 + 2 * user_activity + 0.002 * historical_spend - 0.01 * days_since_last + 0.5 * is_member
    )))

    # 各渠道的增量效果（CATE - 因人而异）
    # Push 对不活跃用户效果更好
    push_lift = 0.03 + 0.05 * (1 - user_activity) + 0.02 * is_member

    # Email 对历史高消费用户效果更好
    email_lift = 0.02 + 0.00005 * historical_spend + 0.01 * is_member

    # SMS 对沉默用户效果更好
    sms_lift = 0.01 + 0.001 * np.minimum(days_since_last, 60)

    # 实际转化概率
    conversion_prob = baseline_prob.copy()
    conversion_prob += push_treated * push_lift
    conversion_prob += email_treated * email_lift
    conversion_prob += sms_treated * sms_lift
    conversion_prob = np.clip(conversion_prob, 0, 1)

    # 实际转化
    converted = np.random.binomial(1, conversion_prob)

    # 转化金额
    order_value = np.where(
        converted == 1,
        np.random.lognormal(4, 0.8, n_users),  # 平均约 55 元
        0
    )

    # 构建 DataFrame
    df = pd.DataFrame({
        'user_id': range(n_users),
        'user_activity': user_activity,
        'historical_spend': historical_spend,
        'days_since_last': days_since_last,
        'age': age,
        'is_member': is_member,
        'push_treated': push_treated,
        'email_treated': email_treated,
        'sms_treated': sms_treated,
        'converted': converted,
        'order_value': order_value,
        # 真实值（用于评估，实际业务中不可见）
        '_baseline_prob': baseline_prob,
        '_push_lift': push_lift,
        '_email_lift': email_lift,
        '_sms_lift': sms_lift,
        '_push_propensity': push_propensity,
        '_email_propensity': email_propensity,
        '_sms_propensity': sms_propensity,
    })

    return df


class IncrementalAttributor:
    """增量归因器 - 估计各渠道的真实增量效果"""

    def __init__(self):
        self.ps_models = {}
        self.outcome_models = {}

    def fit(self, df: pd.DataFrame, treatment_col: str,
            feature_cols: List[str], outcome_col: str = 'converted'):
        """拟合倾向得分和结果模型"""
        X = df[feature_cols].values
        T = df[treatment_col].values
        Y = df[outcome_col].values

        # 倾向得分模型
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        self.ps_models[treatment_col] = ps_model

        # 结果模型（分处理组和对照组）
        outcome_model_t1 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        outcome_model_t0 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=43)

        mask_t1 = T == 1
        mask_t0 = T == 0

        if mask_t1.sum() > 10:
            outcome_model_t1.fit(X[mask_t1], Y[mask_t1])
        if mask_t0.sum() > 10:
            outcome_model_t0.fit(X[mask_t0], Y[mask_t0])

        self.outcome_models[treatment_col] = (outcome_model_t1, outcome_model_t0)

        return self

    def estimate_ate(self, df: pd.DataFrame, treatment_col: str,
                     feature_cols: List[str], outcome_col: str = 'converted') -> Dict:
        """
        使用双重稳健估计 ATE

        Returns:
        --------
        Dict with ATE estimate and diagnostics
        """
        X = df[feature_cols].values
        T = df[treatment_col].values
        Y = df[outcome_col].values
        n = len(Y)

        # 倾向得分
        ps = self.ps_models[treatment_col].predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)

        # 结果预测
        model_t1, model_t0 = self.outcome_models[treatment_col]
        mu1 = model_t1.predict_proba(X)[:, 1] if hasattr(model_t1, 'predict_proba') else model_t1.predict(X)
        mu0 = model_t0.predict_proba(X)[:, 1] if hasattr(model_t0, 'predict_proba') else model_t0.predict(X)

        # AIPW 估计
        aipw_scores = (
            (mu1 - mu0) +
            T * (Y - mu1) / ps -
            (1 - T) * (Y - mu0) / (1 - ps)
        )

        ate = aipw_scores.mean()
        se = aipw_scores.std() / np.sqrt(n)

        # 朴素估计（对比）
        naive_ate = Y[T == 1].mean() - Y[T == 0].mean()

        return {
            'ate': ate,
            'se': se,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'naive_ate': naive_ate,
            'bias': naive_ate - ate,
            'n_treated': T.sum(),
            'n_control': (1 - T).sum(),
        }

    def estimate_cate(self, df: pd.DataFrame, treatment_col: str,
                      feature_cols: List[str]) -> np.ndarray:
        """估计个体处理效应 (CATE)"""
        X = df[feature_cols].values
        model_t1, model_t0 = self.outcome_models[treatment_col]

        mu1 = model_t1.predict_proba(X)[:, 1]
        mu0 = model_t0.predict_proba(X)[:, 1]

        return mu1 - mu0


class BudgetOptimizer:
    """预算优化器 - 基于增量效果优化预算分配"""

    def __init__(self, channel_costs: Dict[str, float]):
        """
        Parameters:
        -----------
        channel_costs: 各渠道单次触达成本
            e.g., {'push': 0.01, 'email': 0.05, 'sms': 0.1}
        """
        self.channel_costs = channel_costs

    def optimize_allocation(
        self,
        df: pd.DataFrame,
        cate_by_channel: Dict[str, np.ndarray],
        total_budget: float,
        avg_order_value: float = 55.0
    ) -> Dict:
        """
        优化预算分配

        策略：按边际 ROI 排序，贪心分配

        Parameters:
        -----------
        df: 用户数据
        cate_by_channel: 各渠道的 CATE 估计
        total_budget: 总预算
        avg_order_value: 平均订单金额

        Returns:
        --------
        优化后的分配方案
        """
        n_users = len(df)

        # 计算每个用户在每个渠道的边际价值
        user_channel_values = []

        for channel, cate in cate_by_channel.items():
            cost = self.channel_costs.get(channel, 0.05)
            # 边际价值 = 增量转化概率 * 平均订单金额 - 触达成本
            marginal_value = cate * avg_order_value - cost

            for i in range(n_users):
                user_channel_values.append({
                    'user_id': i,
                    'channel': channel,
                    'cate': cate[i],
                    'marginal_value': marginal_value[i],
                    'cost': cost
                })

        # 按边际价值排序
        user_channel_values.sort(key=lambda x: x['marginal_value'], reverse=True)

        # 贪心分配
        allocated = []
        spent = 0
        user_allocated = set()  # 每个用户只分配一个渠道

        for item in user_channel_values:
            if spent + item['cost'] > total_budget:
                continue
            if item['user_id'] in user_allocated:
                continue
            if item['marginal_value'] <= 0:
                continue

            allocated.append(item)
            spent += item['cost']
            user_allocated.add(item['user_id'])

        # 汇总结果
        allocation_by_channel = {}
        for item in allocated:
            ch = item['channel']
            if ch not in allocation_by_channel:
                allocation_by_channel[ch] = {
                    'users': 0,
                    'spend': 0,
                    'expected_incremental_conversions': 0,
                    'expected_incremental_revenue': 0
                }
            allocation_by_channel[ch]['users'] += 1
            allocation_by_channel[ch]['spend'] += item['cost']
            allocation_by_channel[ch]['expected_incremental_conversions'] += item['cate']
            allocation_by_channel[ch]['expected_incremental_revenue'] += item['cate'] * avg_order_value

        return {
            'total_budget': total_budget,
            'total_spent': spent,
            'total_users_targeted': len(user_allocated),
            'allocation_by_channel': allocation_by_channel,
            'expected_total_incremental_conversions': sum(
                v['expected_incremental_conversions'] for v in allocation_by_channel.values()
            ),
            'expected_total_incremental_revenue': sum(
                v['expected_incremental_revenue'] for v in allocation_by_channel.values()
            ),
            'expected_roi': (sum(v['expected_incremental_revenue'] for v in allocation_by_channel.values()) - spent) / spent if spent > 0 else 0
        }


def run_marketing_analysis(
    n_users: int = 50000,
    total_budget: float = 5000,
    seed: int = 42
) -> Tuple[go.Figure, str]:
    """
    运行完整的营销分析

    Returns:
    --------
    (figure, markdown_report)
    """
    # 1. 生成数据
    df = generate_marketing_data(n_users=n_users, seed=seed)

    feature_cols = ['user_activity', 'historical_spend', 'days_since_last', 'age', 'is_member']
    channels = ['push_treated', 'email_treated', 'sms_treated']
    channel_names = {'push_treated': 'Push', 'email_treated': 'Email', 'sms_treated': 'SMS'}
    channel_costs = {'push_treated': 0.01, 'email_treated': 0.05, 'sms_treated': 0.10}

    # 2. 增量归因
    attributor = IncrementalAttributor()
    results = {}
    cate_by_channel = {}

    for channel in channels:
        attributor.fit(df, channel, feature_cols, 'converted')
        results[channel] = attributor.estimate_ate(df, channel, feature_cols, 'converted')
        cate_by_channel[channel] = attributor.estimate_cate(df, channel, feature_cols)

    # 3. 预算优化
    optimizer = BudgetOptimizer(channel_costs)

    # 对比：均匀分配 vs 优化分配
    uniform_allocation = {
        ch: total_budget / 3 / channel_costs[ch]
        for ch in channels
    }

    optimized = optimizer.optimize_allocation(
        df, cate_by_channel, total_budget, avg_order_value=55.0
    )

    # 4. 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '各渠道增量效果对比',
            '朴素估计 vs 因果估计',
            '优化预算分配',
            'CATE 分布（识别敏感人群）'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "histogram"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    colors = ['#2D9CDB', '#27AE60', '#9B59B6']

    # 图1：各渠道 ATE
    channel_labels = [channel_names[ch] for ch in channels]
    ates = [results[ch]['ate'] * 100 for ch in channels]  # 转为百分比
    errors = [results[ch]['se'] * 100 * 1.96 for ch in channels]

    fig.add_trace(
        go.Bar(
            x=channel_labels,
            y=ates,
            error_y=dict(type='data', array=errors),
            marker_color=colors,
            name='增量转化率',
            text=[f'{a:.2f}%' for a in ates],
            textposition='outside'
        ),
        row=1, col=1
    )

    # 图2：朴素 vs 因果
    naive_ates = [results[ch]['naive_ate'] * 100 for ch in channels]

    fig.add_trace(
        go.Bar(
            x=channel_labels,
            y=naive_ates,
            name='朴素估计',
            marker_color='#EB5757',
            opacity=0.7
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            x=channel_labels,
            y=ates,
            name='因果估计',
            marker_color='#27AE60',
            opacity=0.7
        ),
        row=1, col=2
    )

    # 图3：预算分配饼图
    if optimized['allocation_by_channel']:
        pie_labels = [channel_names[ch] for ch in optimized['allocation_by_channel'].keys()]
        pie_values = [v['spend'] for v in optimized['allocation_by_channel'].values()]

        fig.add_trace(
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker_colors=colors[:len(pie_labels)],
                textinfo='label+percent',
                hole=0.4
            ),
            row=2, col=1
        )

    # 图4：CATE 分布
    all_cate = np.concatenate([cate_by_channel[ch] for ch in channels])
    fig.add_trace(
        go.Histogram(
            x=all_cate * 100,
            nbinsx=50,
            marker_color='#2D9CDB',
            opacity=0.7,
            name='CATE 分布'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        template='plotly_white',
        title_text='智能营销 ROI 优化分析',
        title_x=0.5
    )

    fig.update_xaxes(title_text='渠道', row=1, col=1)
    fig.update_yaxes(title_text='增量转化率 (%)', row=1, col=1)
    fig.update_xaxes(title_text='渠道', row=1, col=2)
    fig.update_yaxes(title_text='转化率提升 (%)', row=1, col=2)
    fig.update_xaxes(title_text='CATE (%)', row=2, col=2)
    fig.update_yaxes(title_text='用户数', row=2, col=2)

    # 5. 生成报告
    report = f"""
### 营销效果分析报告

#### 1. 数据概况
- 用户总数: {n_users:,}
- 分析渠道: Push、Email、SMS
- 总预算: ¥{total_budget:,.0f}

#### 2. 各渠道增量效果

| 渠道 | 触达人数 | 朴素转化率提升 | **真实增量** | 偏差 |
|------|---------|---------------|-------------|------|
| Push | {results['push_treated']['n_treated']:,} | {results['push_treated']['naive_ate']*100:.2f}% | **{results['push_treated']['ate']*100:.2f}%** | {results['push_treated']['bias']*100:+.2f}% |
| Email | {results['email_treated']['n_treated']:,} | {results['email_treated']['naive_ate']*100:.2f}% | **{results['email_treated']['ate']*100:.2f}%** | {results['email_treated']['bias']*100:+.2f}% |
| SMS | {results['sms_treated']['n_treated']:,} | {results['sms_treated']['naive_ate']*100:.2f}% | **{results['sms_treated']['ate']*100:.2f}%** | {results['sms_treated']['bias']*100:+.2f}% |

**关键发现**: 朴素估计普遍高估了营销效果，因为活跃用户更可能被触达也更可能转化（选择偏差）。

#### 3. 优化预算分配

| 指标 | 值 |
|-----|-----|
| 目标用户数 | {optimized['total_users_targeted']:,} |
| 实际花费 | ¥{optimized['total_spent']:,.2f} |
| 预期增量转化 | {optimized['expected_total_incremental_conversions']:.0f} |
| 预期增量收入 | ¥{optimized['expected_total_incremental_revenue']:,.0f} |
| **预期 ROI** | **{optimized['expected_roi']*100:.1f}%** |

#### 4. 业务建议

1. **停止盲目群发**: 朴素分析高估效果 2-3 倍，实际 ROI 远低于预期
2. **聚焦高敏感人群**: CATE > 5% 的用户仅占 {(np.mean(all_cate > 0.05) * 100):.1f}%，但贡献大部分增量
3. **渠道差异化**:
   - Push 适合唤醒不活跃用户
   - Email 适合高价值会员
   - SMS 成本高，需精准投放
4. **建立增量实验机制**: 每次大促前预留 5-10% 用户作为对照组

---
*报告生成时间: 基于因果推断的增量分析*
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 智能营销 ROI 优化

### 业务背景

电商/互联网公司每年投入大量营销预算，但如何衡量真实效果？

**常见误区**：
- 看触达用户的转化率 → 高估（选择偏差）
- Last-click 归因 → 低估品牌渠道
- 不做对照组 → 无法区分自然转化和营销增量

**本案例展示**：
1. 如何用因果推断测量真实增量效果
2. 如何识别高敏感人群（人群圈选）
3. 如何优化预算分配最大化 ROI

---

### 核心概念

| 概念 | 定义 | 重要性 |
|-----|------|--------|
| **增量效果** | 营销带来的额外转化，排除本来就会转化的用户 | 衡量真实 ROI |
| **选择偏差** | 被营销触达的用户本身就更可能转化 | 导致效果高估 |
| **CATE** | 条件平均处理效应，不同用户的敏感度不同 | 人群圈选依据 |
| **边际 ROI** | 多投入 1 元带来的增量收益 | 预算优化依据 |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_users = gr.Slider(10000, 100000, 50000, step=10000, label="用户数量")
                total_budget = gr.Slider(1000, 20000, 5000, step=1000, label="营销预算 (¥)")
                seed = gr.Number(value=42, label="随机种子", precision=0)
                run_btn = gr.Button("运行分析", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="分析可视化")

        with gr.Row():
            report_output = gr.Markdown()

        run_btn.click(
            fn=run_marketing_analysis,
            inputs=[n_users, total_budget, seed],
            outputs=[plot_output, report_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: 如何设计增量实验？**
> 随机留出 5-10% 用户作为对照组（不触达），对比实验组和对照组的转化率差异。
> 注意：需要足够样本量保证统计显著性。

**Q2: 什么是 PSA 实验？**
> Public Service Announcement 实验。对照组展示公益广告而非商业广告，
> 用于测量品牌广告的增量效果，同时保持用户体验一致。

**Q3: 多触点归因怎么做？**
> - 简单：Last-click、First-click、线性分配
> - 进阶：Shapley Value、Markov Chain
> - 因果：增量实验 + 因果推断

**Q4: 预算有限时如何优化？**
> 按边际 ROI 排序，优先投放到高敏感人群。
> 关键：CATE 异质性分析 + 成本约束优化。

---

### 实践练习

1. 实现增量归因器，比较与传统归因模型的差异
2. 计算各渠道的真实因果 ROI
3. 基于 CATE 异质性设计预算优化策略
        """)

    return None


if __name__ == "__main__":
    # 测试
    fig, report = run_marketing_analysis()
    print(report)
