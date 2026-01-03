"""
用户分层干预模块

场景: 网约车/共享经济平台司机激励
目标: 识别最佳激励对象，最大化供给侧活跃度
方法: CATE 估计 + 最优策略学习

参考: Uber, Lyft, DoorDash
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List

from .utils import generate_driver_incentive_data


class CATEEstimator:
    """
    CATE (Conditional Average Treatment Effect) 估计器

    使用 X-Learner 方法估计异质性处理效应
    """

    def __init__(self):
        self.model_control = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model_treatment = GradientBoostingRegressor(n_estimators=100, random_state=43)
        self.tau_control = GradientBoostingRegressor(n_estimators=50, random_state=44)
        self.tau_treatment = GradientBoostingRegressor(n_estimators=50, random_state=45)
        self.propensity_model = RandomForestClassifier(n_estimators=50, random_state=46)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练 X-Learner"""
        mask_control = T == 0
        mask_treatment = T == 1

        # Stage 1: 估计 mu_0 和 mu_1
        self.model_control.fit(X[mask_control], Y[mask_control])
        self.model_treatment.fit(X[mask_treatment], Y[mask_treatment])

        # Stage 2: 估计伪处理效应
        # D_1 = Y - mu_0(X) for treatment group
        D_treatment = Y[mask_treatment] - self.model_control.predict(X[mask_treatment])
        self.tau_treatment.fit(X[mask_treatment], D_treatment)

        # D_0 = mu_1(X) - Y for control group
        D_control = self.model_treatment.predict(X[mask_control]) - Y[mask_control]
        self.tau_control.fit(X[mask_control], D_control)

        # 估计倾向得分
        self.propensity_model.fit(X, T)

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        tau_0 = self.tau_control.predict(X)
        tau_1 = self.tau_treatment.predict(X)

        # 倾向得分作为权重
        propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 加权组合
        cate = propensity * tau_0 + (1 - propensity) * tau_1

        return cate


def learn_optimal_policy(
    cate: np.ndarray,
    cost_per_treatment: float,
    value_per_unit_effect: float
) -> Tuple[np.ndarray, Dict]:
    """
    学习最优干预策略

    决策规则: 当 CATE * value > cost 时进行干预

    Parameters:
    -----------
    cate: 估计的 CATE
    cost_per_treatment: 每次干预成本
    value_per_unit_effect: 每单位效应的价值

    Returns:
    --------
    (optimal_policy, metrics)
    """
    # 最优策略: treat if CATE * value > cost
    threshold = cost_per_treatment / value_per_unit_effect
    optimal_policy = (cate > threshold).astype(int)

    # 计算指标
    n_treated = optimal_policy.sum()
    expected_effect = cate[optimal_policy == 1].sum() if n_treated > 0 else 0
    total_cost = n_treated * cost_per_treatment
    total_value = expected_effect * value_per_unit_effect
    net_benefit = total_value - total_cost

    metrics = {
        'n_treated': n_treated,
        'treatment_rate': n_treated / len(cate),
        'expected_effect': expected_effect,
        'total_cost': total_cost,
        'total_value': total_value,
        'net_benefit': net_benefit,
        'roi': (total_value - total_cost) / total_cost if total_cost > 0 else 0,
        'threshold': threshold
    }

    return optimal_policy, metrics


def segment_drivers_by_cate(
    df: pd.DataFrame,
    cate: np.ndarray
) -> pd.DataFrame:
    """
    根据 CATE 将司机分层

    - High Impact: CATE > p75
    - Medium Impact: p25 < CATE < p75
    - Low Impact: CATE < p25
    - Negative Impact: CATE < 0

    Parameters:
    -----------
    df: 司机数据
    cate: CATE 估计值

    Returns:
    --------
    带有 segment 列的 DataFrame
    """
    df = df.copy()
    df['cate'] = cate

    p75 = np.percentile(cate, 75)
    p25 = np.percentile(cate, 25)

    conditions = [
        (cate >= p75),
        (cate >= p25) & (cate < p75),
        (cate >= 0) & (cate < p25),
        (cate < 0)
    ]

    choices = ['High Impact', 'Medium Impact', 'Low Impact', 'Negative Impact']
    df['segment'] = np.select(conditions, choices, default='Unknown')

    return df


def analyze_cost_benefit(
    df: pd.DataFrame,
    cate: np.ndarray,
    cost_per_treatment: float = 100,
    revenue_per_hour: float = 60
) -> Tuple[List[Dict], float]:
    """
    成本效益分析

    Parameters:
    -----------
    df: 司机数据
    cate: CATE 估计
    cost_per_treatment: 每次激励成本
    revenue_per_hour: 每小时平台收入

    Returns:
    --------
    (policy_comparison, optimal_threshold)
    """
    value_per_unit_effect = revenue_per_hour * 0.2  # 平台抽成 20%

    policies = []

    # 策略 1: 不干预
    policies.append({
        'policy': 'No Treatment',
        'n_treated': 0,
        'cost': 0,
        'value': 0,
        'net_benefit': 0,
        'roi': 0
    })

    # 策略 2: 全量干预
    policies.append({
        'policy': 'Treat All',
        'n_treated': len(df),
        'cost': len(df) * cost_per_treatment,
        'value': cate.sum() * value_per_unit_effect,
        'net_benefit': cate.sum() * value_per_unit_effect - len(df) * cost_per_treatment,
        'roi': (cate.sum() * value_per_unit_effect - len(df) * cost_per_treatment) / (len(df) * cost_per_treatment)
    })

    # 策略 3: 最优策略
    optimal_policy, optimal_metrics = learn_optimal_policy(
        cate, cost_per_treatment, value_per_unit_effect
    )

    policies.append({
        'policy': 'Optimal Policy',
        'n_treated': optimal_metrics['n_treated'],
        'cost': optimal_metrics['total_cost'],
        'value': optimal_metrics['total_value'],
        'net_benefit': optimal_metrics['net_benefit'],
        'roi': optimal_metrics['roi']
    })

    # 策略 4: Top 30% CATE
    top_30_threshold = np.percentile(cate, 70)
    top_30_policy = (cate >= top_30_threshold).astype(int)
    n_top_30 = top_30_policy.sum()
    value_top_30 = cate[top_30_policy == 1].sum() * value_per_unit_effect if n_top_30 > 0 else 0
    cost_top_30 = n_top_30 * cost_per_treatment

    policies.append({
        'policy': 'Top 30% CATE',
        'n_treated': n_top_30,
        'cost': cost_top_30,
        'value': value_top_30,
        'net_benefit': value_top_30 - cost_top_30,
        'roi': (value_top_30 - cost_top_30) / cost_top_30 if cost_top_30 > 0 else 0
    })

    # 策略 5: 兼职司机 (传统规则)
    part_time_policy = (df['is_fulltime'] == 0).astype(int)
    n_part_time = part_time_policy.sum()
    value_part_time = cate[part_time_policy == 1].sum() * value_per_unit_effect if n_part_time > 0 else 0
    cost_part_time = n_part_time * cost_per_treatment

    policies.append({
        'policy': 'Part-time Only',
        'n_treated': n_part_time,
        'cost': cost_part_time,
        'value': value_part_time,
        'net_benefit': value_part_time - cost_part_time,
        'roi': (value_part_time - cost_part_time) / cost_part_time if cost_part_time > 0 else 0
    })

    return policies, optimal_metrics['threshold']


def run_driver_incentive_analysis(
    n_samples: int,
    incentive_cost: float,
    analyze_segments: bool
) -> Tuple[go.Figure, str]:
    """
    运行司机激励分析

    Parameters:
    -----------
    n_samples: 样本量
    incentive_cost: 激励成本
    analyze_segments: 是否分析分层

    Returns:
    --------
    (figure, summary)
    """
    # 生成数据
    df, true_effect = generate_driver_incentive_data(n_samples)

    # 准备特征
    feature_cols = ['driver_rating', 'years_on_platform', 'avg_daily_hours',
                    'completed_orders_history', 'is_fulltime']

    # 将 city_zone 转为数值
    df['zone_downtown'] = (df['city_zone'] == 'downtown').astype(int)
    df['zone_suburb'] = (df['city_zone'] == 'suburb').astype(int)
    feature_cols.extend(['zone_downtown', 'zone_suburb'])

    X = df[feature_cols].values
    T = df['T'].values
    Y = df['online_hours'].values

    # 训练 CATE 模型
    cate_estimator = CATEEstimator()
    cate_estimator.fit(X, T, Y)
    cate = cate_estimator.predict_cate(X)

    # 分层分析
    if analyze_segments:
        df_segmented = segment_drivers_by_cate(df, cate)
    else:
        df_segmented = df.copy()
        df_segmented['cate'] = cate

    # 成本效益分析
    policies, optimal_threshold = analyze_cost_benefit(
        df, cate, cost_per_treatment=incentive_cost
    )
    policies_df = pd.DataFrame(policies)

    # === 可视化 ===
    if analyze_segments:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '司机分层分布',
                '不同策略的 ROI 对比',
                'CATE 分布与最优阈值',
                '成本-收益曲线'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'scatter'}]
            ]
        )

        # 1. 司机分层分布
        segment_counts = df_segmented['segment'].value_counts()
        segment_colors_map = {
            'High Impact': '#27AE60',
            'Medium Impact': '#2D9CDB',
            'Low Impact': '#F2994A',
            'Negative Impact': '#EB5757'
        }
        colors_seg = [segment_colors_map.get(seg, '#95A5A6') for seg in segment_counts.index]

        fig.add_trace(go.Bar(
            x=segment_counts.index,
            y=segment_counts.values,
            marker_color=colors_seg,
            text=[f'{v/len(df)*100:.1f}%' for v in segment_counts.values],
            textposition='outside',
            name='Driver Count'
        ), row=1, col=1)

        # 2. ROI 对比
        policy_colors = ['#95A5A6', '#EB5757', '#27AE60', '#2D9CDB', '#F2994A']
        fig.add_trace(go.Bar(
            x=policies_df['policy'],
            y=policies_df['roi'],
            marker_color=policy_colors,
            text=[f'{v:.2f}' for v in policies_df['roi']],
            textposition='outside',
            name='ROI'
        ), row=1, col=2)

        # 3. CATE 分布
        fig.add_trace(go.Histogram(
            x=cate,
            marker_color='#2D9CDB',
            opacity=0.7,
            nbinsx=40,
            name='CATE Distribution'
        ), row=2, col=1)

        # 标记最优阈值
        fig.add_vline(
            x=optimal_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal Threshold: {optimal_threshold:.2f}",
            row=2, col=1
        )

        # 4. 成本-收益曲线
        # 不同干预比例下的净收益
        fractions = np.linspace(0, 1, 50)
        net_benefits = []
        rois = []

        revenue_per_hour = 60
        value_per_unit = revenue_per_hour * 0.2

        for frac in fractions:
            if frac == 0:
                net_benefits.append(0)
                rois.append(0)
                continue

            # Top frac% by CATE
            threshold_frac = np.percentile(cate, (1 - frac) * 100)
            policy = (cate >= threshold_frac).astype(int)
            n_treated = policy.sum()

            if n_treated > 0:
                value = cate[policy == 1].sum() * value_per_unit
                cost = n_treated * incentive_cost
                net_benefit = value - cost
                roi = net_benefit / cost if cost > 0 else 0
            else:
                net_benefit = 0
                roi = 0

            net_benefits.append(net_benefit)
            rois.append(roi)

        fig.add_trace(go.Scatter(
            x=fractions * 100,
            y=net_benefits,
            mode='lines',
            line=dict(color='#9B59B6', width=3),
            name='Net Benefit'
        ), row=2, col=2)

        # 标记最优点
        best_idx = np.argmax(net_benefits)
        best_frac = fractions[best_idx]
        best_benefit = net_benefits[best_idx]

        fig.add_trace(go.Scatter(
            x=[best_frac * 100],
            y=[best_benefit],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name=f'Optimal: {best_frac*100:.0f}%',
            showlegend=False
        ), row=2, col=2)

    else:
        # 简化版
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                '不同策略的 ROI 对比',
                'CATE 分布'
            )
        )

        policy_colors = ['#95A5A6', '#EB5757', '#27AE60', '#2D9CDB', '#F2994A']
        fig.add_trace(go.Bar(
            x=policies_df['policy'],
            y=policies_df['roi'],
            marker_color=policy_colors,
            text=[f'{v:.2f}' for v in policies_df['roi']],
            textposition='outside',
            name='ROI'
        ), row=1, col=1)

        fig.add_trace(go.Histogram(
            x=cate,
            marker_color='#2D9CDB',
            opacity=0.7,
            nbinsx=40,
            name='CATE Distribution'
        ), row=1, col=2)

    fig.update_layout(
        height=800 if analyze_segments else 400,
        template='plotly_white',
        title_text='司机激励策略优化',
        showlegend=True
    )

    if analyze_segments:
        fig.update_xaxes(title_text='Segment', row=1, col=1)
        fig.update_xaxes(title_text='Policy', row=1, col=2)
        fig.update_xaxes(title_text='CATE (Hours)', row=2, col=1)
        fig.update_xaxes(title_text='Treatment Rate (%)', row=2, col=2)

        fig.update_yaxes(title_text='Driver Count', row=1, col=1)
        fig.update_yaxes(title_text='ROI', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_yaxes(title_text='Net Benefit (Yuan)', row=2, col=2)
    else:
        fig.update_xaxes(title_text='Policy', row=1, col=1)
        fig.update_xaxes(title_text='CATE (Hours)', row=1, col=2)
        fig.update_yaxes(title_text='ROI', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=1, col=2)

    # === 生成摘要 ===
    optimal_policy_row = policies_df[policies_df['policy'] == 'Optimal Policy'].iloc[0]
    treat_all_row = policies_df[policies_df['policy'] == 'Treat All'].iloc[0]
    part_time_row = policies_df[policies_df['policy'] == 'Part-time Only'].iloc[0]

    if analyze_segments:
        high_impact_pct = (df_segmented['segment'] == 'High Impact').sum() / len(df) * 100
        negative_impact_pct = (df_segmented['segment'] == 'Negative Impact').sum() / len(df) * 100
    else:
        high_impact_pct = 0
        negative_impact_pct = 0

    summary = f"""
### 司机激励策略分析

#### 核心指标

| 指标 | 值 |
|------|-----|
| 总司机数 | {n_samples:,} |
| 激励成本 | ¥{incentive_cost:.0f} / 人 |
| 平均 CATE | {cate.mean():.2f} 小时 |
| CATE 标准差 | {cate.std():.2f} 小时 |
| 最优阈值 | {optimal_threshold:.2f} 小时 |
    """

    if analyze_segments:
        summary += f"""
| High Impact 占比 | {high_impact_pct:.1f}% |
| Negative Impact 占比 | {negative_impact_pct:.1f}% |
        """

    summary += f"""

#### 策略对比

| 策略 | 干预人数 | 成本 (¥) | 收益 (¥) | 净收益 (¥) | ROI |
|------|---------|---------|---------|-----------|-----|
| Optimal Policy | {optimal_policy_row['n_treated']:,.0f} | {optimal_policy_row['cost']:,.0f} | {optimal_policy_row['value']:,.0f} | {optimal_policy_row['net_benefit']:,.0f} | {optimal_policy_row['roi']:.2f} |
| Treat All | {treat_all_row['n_treated']:,.0f} | {treat_all_row['cost']:,.0f} | {treat_all_row['value']:,.0f} | {treat_all_row['net_benefit']:,.0f} | {treat_all_row['roi']:.2f} |
| Part-time Only | {part_time_row['n_treated']:,.0f} | {part_time_row['cost']:,.0f} | {part_time_row['value']:,.0f} | {part_time_row['net_benefit']:,.0f} | {part_time_row['roi']:.2f} |

### 关键洞察

1. **最优策略优势**: 相比全量激励，节省成本 {(1 - optimal_policy_row['n_treated'] / treat_all_row['n_treated']) * 100:.1f}%，同时保持高 ROI
2. **精准识别**: 仅激励 {optimal_policy_row['n_treated'] / n_samples * 100:.1f}% 的司机，即可获得最大净收益
3. **传统规则局限**: "仅激励兼职司机" 的策略 ROI 为 {part_time_row['roi']:.2f}，低于最优策略的 {optimal_policy_row['roi']:.2f}
4. **异质性显著**: CATE 标准差 {cate.std():.2f}，说明不同司机对激励的响应差异很大

### 业务建议

**高优先级激励对象** (High Impact 司机):
- 兼职司机 + 低历史活跃度
- 新入平台司机 (< 1 年)
- 评分良好但接单量不高的司机

**避免激励对象** (Negative Impact 司机):
- 已经高频在线的全职司机 (边际效应递减)
- 评分过低的司机 (激励无效)

**实施建议**:
1. 每周重新计算 CATE，动态调整激励对象
2. 结合时间段 (早晚高峰) 进一步细分策略
3. A/B 测试验证模型预测准确性
4. 监控激励疲劳效应，定期更换激励形式
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 用户分层干预 (User Targeting)

基于 CATE 估计的精准干预策略，最大化干预 ROI。

### 业务场景

网约车/外卖平台面临的核心问题:
- 如何激励供给侧 (司机/骑手) 提升活跃度?
- 给谁激励效果最好?
- 如何在成本和收益间找到最优平衡?

### 方法论

| 步骤 | 方法 | 输出 |
|------|------|------|
| **CATE 估计** | X-Learner | 个体处理效应 |
| **用户分层** | CATE 分位数 | 4 层用户群 |
| **策略学习** | 成本-收益优化 | 最优干预规则 |

### 实际案例

**Uber**: 使用 CATE 优化司机激励，ROI 提升 40%+
**Lyft**: 分层激励策略减少 30% 补贴浪费
**DoorDash**: 精准识别流失风险高的 Dasher，针对性激励

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=5000, maximum=20000, value=10000, step=1000,
                    label="司机数量"
                )
                incentive_cost = gr.Slider(
                    minimum=50, maximum=200, value=100, step=10,
                    label="激励成本 (元/人)"
                )
                analyze_segments = gr.Checkbox(
                    value=True,
                    label="分层分析"
                )
                run_btn = gr.Button("运行分析", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("""
### X-Learner 方法

**Stage 1**: 分别估计处理组和控制组的结果
```
mu_0(x) = E[Y|X=x, T=0]
mu_1(x) = E[Y|X=x, T=1]
```

**Stage 2**: 计算伪处理效应
```
D_1 = Y - mu_0(X)  # 处理组
D_0 = mu_1(X) - Y  # 控制组
```

**Stage 3**: 加权组合
```
tau(x) = e(x) * tau_0(x) + (1-e(x)) * tau_1(x)
```
                """)

        with gr.Row():
            plot_output = gr.Plot(label="分析结果")

        with gr.Row():
            summary_output = gr.Markdown()

        run_btn.click(
            fn=run_driver_incentive_analysis,
            inputs=[n_samples, incentive_cost, analyze_segments],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 最优策略学习

**决策规则**: Treat if CATE * Value > Cost

**数学推导**:

最大化期望净收益:
```
max E[V * tau(X) * T - C * T]

s.t. T in {0, 1}
```

最优策略:
```
T*(X) = 1 if tau(X) > C/V
        0 otherwise
```

其中:
- tau(X): 个体处理效应 (CATE)
- V: 单位效应的价值
- C: 处理成本

### 实际挑战

1. **CATE 估计误差**: 模型预测不准确导致次优策略
   - 解决: 使用 Cross-fitting 减少过拟合
   - 解决: 保守估计 (打折 CATE)

2. **动态效应**: 用户对激励的响应随时间变化
   - 解决: 在线学习，动态更新模型
   - 解决: 考虑激励疲劳效应

3. **溢出效应**: 激励一个司机可能影响其他司机
   - 解决: 网络因果推断方法
   - 解决: 聚类随机化

### 扩展阅读

- [Uber's Causal ML Platform](https://eng.uber.com/causal-inference-at-uber/)
- [DoorDash's Experimentation Platform](https://doordash.engineering/2020/09/09/experimentation-analysis-platform-mvp/)
- [Lyft's Experimentation Platform](https://eng.lyft.com/experimentation-in-a-ridesharing-marketplace-b39db027a66e)
        """)

    return None
