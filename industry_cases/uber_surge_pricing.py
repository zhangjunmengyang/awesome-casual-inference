"""
Uber 动态定价案例

场景: Surge Pricing 对供需平衡的影响
混淆: 需求高峰时段更可能启动 Surge，同时也影响匹配率
方法: 回归断点设计 (RDD) + 逆概率加权 (IPW)
业务价值: 优化定价策略，平衡司机供给和乘客需求

真实背景:
Uber 的动态定价 (Surge Pricing) 是经典的双边市场问题:
- 价格高吸引司机，但抑制乘客需求
- 需求/供给比 (D/S Ratio) 是定价的关键阈值
- 使用回归断点设计评估不同 Surge 倍数的因果效应
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
from typing import Tuple, Dict

from .utils import (
    generate_uber_surge_pricing_data,
    plot_causal_dag,
    compute_ate_with_ci
)


class RegressionDiscontinuityUber:
    """
    回归断点设计 (RDD) 用于 Uber Surge Pricing 分析

    利用需求/供给比的断点识别 Surge 的因果效应
    """

    def __init__(self, cutoff: float = 1.2, bandwidth: float = 0.3):
        """
        Parameters:
        -----------
        cutoff: 断点值 (D/S Ratio 阈值)
        bandwidth: 带宽 (只使用断点附近的样本)
        """
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.model_below = LinearRegression()
        self.model_above = LinearRegression()

    def fit(self, running_var: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        拟合 RDD 模型

        Parameters:
        -----------
        running_var: 运行变量 (D/S Ratio)
        T: 处理变量 (Surge Multiplier)
        Y: 结果变量 (Match Rate / Wait Time)
        """
        # 只使用带宽内的样本
        mask_bandwidth = np.abs(running_var - self.cutoff) <= self.bandwidth

        running_centered = running_var[mask_bandwidth] - self.cutoff
        T_band = T[mask_bandwidth]
        Y_band = Y[mask_bandwidth]

        # 分别拟合断点两侧
        mask_below = running_centered < 0
        mask_above = running_centered >= 0

        if mask_below.sum() > 0:
            self.model_below.fit(running_centered[mask_below].reshape(-1, 1), Y_band[mask_below])

        if mask_above.sum() > 0:
            self.model_above.fit(running_centered[mask_above].reshape(-1, 1), Y_band[mask_above])

        return self

    def estimate_rd_effect(self) -> float:
        """
        估计断点处的跳跃 (Treatment Effect)

        Returns:
        --------
        rd_effect: 断点处的因果效应
        """
        # 断点处的预测值
        y_above = self.model_above.predict([[0]])[0]
        y_below = self.model_below.predict([[0]])[0]

        rd_effect = y_above - y_below
        return rd_effect

    def plot_rd(self, running_var: np.ndarray, Y: np.ndarray) -> go.Figure:
        """绘制 RDD 图"""
        # 只使用带宽内的样本
        mask_bandwidth = np.abs(running_var - self.cutoff) <= self.bandwidth
        running_centered = running_var[mask_bandwidth] - self.cutoff
        Y_band = Y[mask_bandwidth]

        # 分组
        mask_below = running_centered < 0
        mask_above = running_centered >= 0

        fig = go.Figure()

        # 散点图
        fig.add_trace(go.Scatter(
            x=running_centered[mask_below] + self.cutoff,
            y=Y_band[mask_below],
            mode='markers',
            marker=dict(size=4, color='#3498DB', opacity=0.5),
            name='Below Cutoff'
        ))

        fig.add_trace(go.Scatter(
            x=running_centered[mask_above] + self.cutoff,
            y=Y_band[mask_above],
            mode='markers',
            marker=dict(size=4, color='#E74C3C', opacity=0.5),
            name='Above Cutoff'
        ))

        # 拟合线
        x_below = np.linspace(-self.bandwidth, 0, 50)
        x_above = np.linspace(0, self.bandwidth, 50)

        y_below_pred = self.model_below.predict(x_below.reshape(-1, 1))
        y_above_pred = self.model_above.predict(x_above.reshape(-1, 1))

        fig.add_trace(go.Scatter(
            x=x_below + self.cutoff,
            y=y_below_pred,
            mode='lines',
            line=dict(color='#3498DB', width=3),
            name='Fit (Below)',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=x_above + self.cutoff,
            y=y_above_pred,
            mode='lines',
            line=dict(color='#E74C3C', width=3),
            name='Fit (Above)',
            showlegend=False
        ))

        # 断点线
        fig.add_vline(
            x=self.cutoff,
            line_dash='dash',
            line_color='black',
            annotation_text=f'Cutoff: {self.cutoff}',
            annotation_position='top'
        )

        fig.update_layout(
            title='回归断点设计 (RDD) - Surge Pricing 效应',
            xaxis_title='Demand/Supply Ratio',
            yaxis_title='Outcome',
            template='plotly_white',
            height=500
        )

        return fig


def inverse_probability_weighting(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    逆概率加权 (IPW) 估计

    Parameters:
    -----------
    X: 特征矩阵
    T: 处理变量 (多值: 1.0, 1.5, 2.0, 2.5)
    Y: 结果变量

    Returns:
    --------
    (ate, weights): ATE 和 IPW 权重
    """
    # 为简化，将 T 二值化: >= 1.5 vs < 1.5
    T_binary = (T >= 1.5).astype(int)

    # 估计倾向得分
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T_binary)
    propensity = ps_model.predict_proba(X)[:, 1]

    # 防止极端权重
    propensity = np.clip(propensity, 0.05, 0.95)

    # IPW 权重
    weights = np.where(T_binary == 1, 1 / propensity, 1 / (1 - propensity))

    # 加权平均
    y1_weighted = (Y * T_binary * weights).sum() / (T_binary * weights).sum()
    y0_weighted = (Y * (1 - T_binary) * weights).sum() / ((1 - T_binary) * weights).sum()

    ate = y1_weighted - y0_weighted

    return ate, weights


def analyze_surge_elasticity(
    df: pd.DataFrame
) -> Tuple[Dict[str, float], go.Figure]:
    """
    分析 Surge 定价弹性

    Parameters:
    -----------
    df: Uber 数据

    Returns:
    --------
    (elasticity_metrics, figure)
    """
    surge_levels = sorted(df['surge_multiplier'].unique())

    metrics = {
        'avg_supply': [],
        'avg_demand': [],
        'avg_match_rate': [],
        'avg_wait_time': [],
        'avg_revenue': []
    }

    for surge in surge_levels:
        mask = df['surge_multiplier'] == surge

        metrics['avg_supply'].append(df.loc[mask, 'actual_supply'].mean())
        metrics['avg_demand'].append(df.loc[mask, 'actual_demand'].mean())
        metrics['avg_match_rate'].append(df.loc[mask, 'match_rate'].mean())
        metrics['avg_wait_time'].append(df.loc[mask, 'wait_time'].mean())
        metrics['avg_revenue'].append(df.loc[mask, 'platform_revenue'].mean())

    # 计算弹性
    supply_elasticity = []
    demand_elasticity = []

    for i in range(1, len(surge_levels)):
        # 供给弹性: % change in supply / % change in price
        price_change = (surge_levels[i] - surge_levels[i-1]) / surge_levels[i-1]
        supply_change = (metrics['avg_supply'][i] - metrics['avg_supply'][i-1]) / metrics['avg_supply'][i-1]
        demand_change = (metrics['avg_demand'][i] - metrics['avg_demand'][i-1]) / metrics['avg_demand'][i-1]

        if price_change != 0:
            supply_elasticity.append(supply_change / price_change)
            demand_elasticity.append(demand_change / price_change)

    elasticity_metrics = {
        'avg_supply_elasticity': np.mean(supply_elasticity) if supply_elasticity else 0,
        'avg_demand_elasticity': np.mean(demand_elasticity) if demand_elasticity else 0,
    }

    # 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Surge vs 供给/需求',
            'Surge vs 匹配率',
            'Surge vs 等待时间',
            'Surge vs 平台收入'
        )
    )

    # 1. 供给/需求
    fig.add_trace(go.Scatter(
        x=surge_levels,
        y=metrics['avg_supply'],
        mode='lines+markers',
        name='Supply',
        line=dict(color='#27AE60', width=3),
        marker=dict(size=8)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=surge_levels,
        y=metrics['avg_demand'],
        mode='lines+markers',
        name='Demand',
        line=dict(color='#E74C3C', width=3),
        marker=dict(size=8)
    ), row=1, col=1)

    # 2. 匹配率
    fig.add_trace(go.Scatter(
        x=surge_levels,
        y=metrics['avg_match_rate'],
        mode='lines+markers',
        name='Match Rate',
        line=dict(color='#9B59B6', width=3),
        marker=dict(size=8),
        showlegend=False
    ), row=1, col=2)

    # 3. 等待时间
    fig.add_trace(go.Scatter(
        x=surge_levels,
        y=metrics['avg_wait_time'],
        mode='lines+markers',
        name='Wait Time',
        line=dict(color='#F39C12', width=3),
        marker=dict(size=8),
        showlegend=False
    ), row=2, col=1)

    # 4. 平台收入
    fig.add_trace(go.Scatter(
        x=surge_levels,
        y=metrics['avg_revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#2D9CDB', width=3),
        marker=dict(size=8),
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text='Surge Pricing 弹性分析',
        showlegend=True
    )

    fig.update_xaxes(title_text='Surge Multiplier', row=1, col=1)
    fig.update_xaxes(title_text='Surge Multiplier', row=1, col=2)
    fig.update_xaxes(title_text='Surge Multiplier', row=2, col=1)
    fig.update_xaxes(title_text='Surge Multiplier', row=2, col=2)

    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='Match Rate', row=1, col=2)
    fig.update_yaxes(title_text='Wait Time (min)', row=2, col=1)
    fig.update_yaxes(title_text='Revenue ($)', row=2, col=2)

    return elasticity_metrics, fig


def analyze_uber_surge_pricing(
    n_samples: int,
    analysis_type: str,
    cutoff_value: float
) -> Tuple[go.Figure, str]:
    """
    Uber Surge Pricing 因果分析

    Parameters:
    -----------
    n_samples: 订单数量
    analysis_type: 分析类型 ('rdd', 'ipw', 'elasticity')
    cutoff_value: RDD 断点值

    Returns:
    --------
    (figure, summary_text)
    """
    # 生成数据
    df, true_effect = generate_uber_surge_pricing_data(n_samples)

    # 准备数据
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'zone_type', 'weather', 'has_event']
    X = df[feature_cols].values
    T = df['surge_multiplier'].values
    running_var = df['ds_ratio'].values
    Y_match = df['match_rate'].values
    Y_wait = df['wait_time'].values

    summary = f"""
### Uber Surge Pricing 因果分析结果

#### 数据概览

| 指标 | 值 |
|------|-----|
| 订单请求数 | {n_samples:,} |
| Surge 1.0x | {(T == 1.0).sum():,} ({(T == 1.0).sum()/len(T)*100:.1f}%) |
| Surge 1.5x | {(T == 1.5).sum():,} ({(T == 1.5).sum()/len(T)*100:.1f}%) |
| Surge 2.0x | {(T == 2.0).sum():,} ({(T == 2.0).sum()/len(T)*100:.1f}%) |
| Surge 2.5x | {(T == 2.5).sum():,} ({(T == 2.5).sum()/len(T)*100:.1f}%) |

"""

    if analysis_type == 'rdd':
        # === 回归断点设计 ===
        rdd = RegressionDiscontinuityUber(cutoff=cutoff_value, bandwidth=0.3)
        rdd.fit(running_var, T, Y_match)

        rd_effect = rdd.estimate_rd_effect()

        fig = rdd.plot_rd(running_var, Y_match)

        summary += f"""
#### 回归断点设计 (RDD) 结果

**断点**: D/S Ratio = {cutoff_value}
**带宽**: ±0.3

| 指标 | 值 |
|------|-----|
| 断点处效应 (Match Rate) | {rd_effect:+.4f} |
| 断点下方均值 | {rdd.model_below.predict([[0]])[0]:.4f} |
| 断点上方均值 | {rdd.model_above.predict([[0]])[0]:.4f} |

#### 关键洞察

1. **Surge 启动效应**: 在 D/S Ratio = {cutoff_value} 断点处，匹配率 {('提升' if rd_effect > 0 else '下降')} {abs(rd_effect)*100:.2f}%
2. **因果识别**: 利用断点附近的"准实验"，控制混淆因素
3. **局部效应**: RDD 估计的是断点附近的局部效应，外推需谨慎

#### 方法解释

**为什么用 RDD?**

Surge 定价基于 D/S Ratio 的阈值规则:
- D/S < 1.0: 不启动 Surge
- D/S ≥ 1.0: 启动 Surge

断点附近的样本"几乎随机":
- D/S = 0.99 vs 1.01 只是随机波动
- 但一个启动 Surge，一个不启动
- 对比两者差异 = Surge 的因果效应

**RDD 假设**:
1. 断点附近协变量连续 (无操纵)
2. 结果变量在断点处平滑 (除了跳跃)
3. 带宽内样本可比
        """

    elif analysis_type == 'ipw':
        # === 逆概率加权 ===
        ate_ipw, weights = inverse_probability_weighting(X, T, Y_match)

        # Naive 估计
        T_binary = (T >= 1.5).astype(int)
        naive_ate = Y_match[T_binary == 1].mean() - Y_match[T_binary == 0].mean()

        # 可视化
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('IPW 权重分布', 'ATE 估计对比')
        )

        fig.add_trace(go.Histogram(
            x=weights[T_binary == 0],
            name='Control Weights',
            marker_color='#3498DB',
            opacity=0.6,
            nbinsx=30
        ), row=1, col=1)

        fig.add_trace(go.Histogram(
            x=weights[T_binary == 1],
            name='Treatment Weights',
            marker_color='#E74C3C',
            opacity=0.6,
            nbinsx=30
        ), row=1, col=1)

        methods = ['Naive', 'IPW']
        ates = [naive_ate, ate_ipw]

        fig.add_trace(go.Bar(
            x=methods,
            y=ates,
            marker_color=['#95A5A6', '#27AE60'],
            text=[f'{v:.4f}' for v in ates],
            textposition='outside',
            showlegend=False
        ), row=1, col=2)

        fig.update_layout(
            height=400,
            template='plotly_white',
            title_text='逆概率加权 (IPW) 分析'
        )

        fig.update_xaxes(title_text='IPW Weight', row=1, col=1)
        fig.update_xaxes(title_text='Method', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        fig.update_yaxes(title_text='ATE (Match Rate)', row=1, col=2)

        summary += f"""
#### 逆概率加权 (IPW) 结果

**处理定义**: Surge ≥ 1.5x vs < 1.5x

| 方法 | ATE (Match Rate) |
|------|-----------------|
| Naive 估计 | {naive_ate:.4f} |
| IPW 估计 | {ate_ipw:.4f} |
| 偏差修正 | {(ate_ipw - naive_ate):.4f} |

#### 关键洞察

1. **混淆调整**: IPW 通过重新加权，平衡处理组和对照组的协变量分布
2. **高 Surge 效应**: Surge ≥ 1.5x 时，匹配率 {('提升' if ate_ipw > 0 else '下降')} {abs(ate_ipw)*100:.2f}%
3. **权重分布**: 极端权重 (>10) 占比 {(weights > 10).sum() / len(weights) * 100:.1f}%，需要修剪

#### 方法解释

**IPW 原理**:

倾向得分 e(X) = P(T=1|X) 反映了处理分配的偏差:
- 高活跃用户更可能分到高 Surge
- IPW 给低 e(X) 的处理样本更高权重
- 相当于"合成"一个随机化实验

**权重公式**:
- 处理组: w = 1 / e(X)
- 对照组: w = 1 / (1 - e(X))

**优势**: 无需结果模型，直接调整混淆
**劣势**: 对极端权重敏感，需要修剪
        """

    else:  # elasticity
        # === 弹性分析 ===
        elasticity_metrics, fig = analyze_surge_elasticity(df)

        supply_elasticity = elasticity_metrics['avg_supply_elasticity']
        demand_elasticity = elasticity_metrics['avg_demand_elasticity']

        summary += f"""
#### Surge Pricing 弹性分析

**定价弹性**:

| 指标 | 弹性系数 | 解释 |
|------|---------|------|
| 供给弹性 | {supply_elasticity:.2f} | Surge 每提高 10%，司机供给增加 {supply_elasticity*10:.1f}% |
| 需求弹性 | {demand_elasticity:.2f} | Surge 每提高 10%，乘客需求减少 {abs(demand_elasticity)*10:.1f}% |

#### 关键洞察

1. **供给响应**: 司机对价格敏感，Surge 能有效吸引司机上线
2. **需求抑制**: 高 Surge 会流失部分乘客，需要平衡
3. **最优定价**: 在 Surge 1.5-2.0x 之间，匹配率和收入达到最优
4. **异质性**: 不同时段/地区的弹性差异大

#### 业务建议

**动态定价策略优化**:

1. **高峰时段** (早晚高峰):
   - Surge 2.0x 左右最优
   - 供给弹性高，司机响应快
   - 乘客接受度较高 (刚需)

2. **恶劣天气**:
   - 适度提高 Surge (1.5-2.0x)
   - 司机成本高，需要补偿
   - 乘客需求也增加

3. **低峰时段**:
   - 保持 Surge 1.0x
   - 供给充足，无需激励
   - 避免流失价格敏感用户

#### 商业价值

假设 Uber 每天:
- 完成 1000 万单
- 优化定价提升匹配率 5%
- 每单平台收入 5 美元

**年化收益**:
- 新增订单: 1000万 × 5% × 365 = 1.825 亿单/年
- 收入增加: 1.825亿 × 5 = 9.125 亿美元/年
        """

    summary += """

#### 真实案例参考

Uber 工程博客提到:
- Surge Pricing 使用复杂的供需预测模型
- 结合因果推断评估不同定价策略效果
- 使用 RDD 和 IPW 等方法控制混淆
- 最终实现供需平衡，司机收入和乘客体验双赢

#### 扩展阅读

- [Uber Surge Pricing Explained](https://www.uber.com/blog/surge-pricing/)
- [Dynamic Pricing at Uber](https://eng.uber.com/research/)
- Angrist & Pischke (2009): Mostly Harmless Econometrics (RDD 章节)
    """

    return fig, summary


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## Uber Surge Pricing 案例

基于回归断点设计和逆概率加权的动态定价分析。

### 业务背景

Uber 的动态定价 (Surge Pricing) 是双边市场的经典问题:
- 需求高峰时，司机供给不足
- 提高价格吸引司机，但可能流失乘客
- 需要找到最优定价策略，平衡供需

### 因果挑战

**核心问题**: Surge Pricing 真的改善了匹配率吗?

| 混淆因素 | 如何影响 |
|---------|---------|
| **需求高峰** | 高峰时段启动 Surge；同时高峰本身匹配率就低 |
| **天气** | 恶劣天气启动 Surge；同时天气本身影响供需 |
| **地区** | 繁忙地区更频繁 Surge；同时这些地区供需本身就不平衡 |

**解决方案**:
- RDD 利用定价阈值的断点
- IPW 调整混淆因素

---
        """)

        with gr.Row():
            # 左侧参数
            with gr.Column(scale=1):
                n_samples = gr.Slider(
                    minimum=5000, maximum=20000, value=12000, step=1000,
                    label="订单请求数"
                )
                analysis_type = gr.Radio(
                    choices=['rdd', 'ipw', 'elasticity'],
                    value='elasticity',
                    label="分析类型"
                )
                cutoff_value = gr.Slider(
                    minimum=0.8, maximum=2.0, value=1.2, step=0.1,
                    label="RDD 断点值 (D/S Ratio)"
                )
                run_btn = gr.Button("运行分析", variant="primary")

            # 右侧说明
            with gr.Column(scale=1):
                gr.Markdown("""
### 分析方法说明

**RDD (回归断点设计)**:
- 利用 D/S Ratio 阈值的"准实验"
- 断点附近样本"几乎随机"
- 估计局部因果效应

**IPW (逆概率加权)**:
- 调整处理分配的选择偏差
- 重新加权平衡协变量
- 估计全局平均效应

**Elasticity (弹性分析)**:
- 供给/需求对价格的响应
- 识别最优定价策略
- 异质性效应分析

### Uber 实践

- Surge 倍数基于实时供需比
- 使用机器学习预测未来需求
- 因果推断评估定价效果
- 动态调整算法参数
                """)

        # 因果图
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 因果图 (DAG)")
                dag_plot = gr.Plot(value=plot_causal_dag('uber'))

        # 分析结果
        with gr.Row():
            plot_output = gr.Plot(label="Surge Pricing 因果分析")

        with gr.Row():
            summary_output = gr.Markdown()

        # 事件绑定
        run_btn.click(
            fn=analyze_uber_surge_pricing,
            inputs=[n_samples, analysis_type, cutoff_value],
            outputs=[plot_output, summary_output]
        )

        gr.Markdown("""
---

### 技术细节

#### 回归断点设计 (RDD)

**核心思想**: 利用处理分配规则的断点

```
T = 1  if  X ≥ cutoff
T = 0  if  X < cutoff
```

在断点附近，X 的微小差异"几乎随机"，因此可以对比:
```
ATE_RDD = lim[E[Y|X=c+] - E[Y|X=c-]]
```

**估计步骤**:
1. 选择带宽 h (断点附近的窗口)
2. 分别拟合断点两侧的回归
3. 计算断点处的跳跃

**关键假设**:
- 断点附近无操纵 (McCrary 检验)
- 协变量连续 (平衡性检验)
- 函数形式正确 (敏感性分析)

#### 逆概率加权 (IPW)

**核心思想**: 通过加权"合成"随机化实验

倾向得分: e(X) = P(T=1|X)

IPW 估计量:
```
ATE_IPW = E[Y·T/e(X)] - E[Y·(1-T)/(1-e(X))]
```

**优势**:
- 无需结果模型
- 直观易懂

**挑战**:
- 极端权重问题
- 需要修剪 (truncation)
- 对倾向得分模型敏感

#### Surge Pricing 优化

**双边市场平衡**:

目标: 最大化 总福利 = 司机收入 + 乘客剩余 + 平台利润

约束:
- 供给响应: S(p) = S₀ + β_s · (p - 1)
- 需求响应: D(p) = D₀ - β_d · (p - 1)
- 市场出清: S(p) = D(p)

最优价格:
```
p* = 1 + (D₀ - S₀) / (β_s + β_d)
```

### 扩展阅读

- [Uber Dynamic Pricing](https://eng.uber.com/tag/pricing/)
- Lee & Lemieux (2010): Regression Discontinuity Designs in Economics
- [Two-Sided Marketplace Economics](https://www.nber.org/papers/w14860)
        """)

    return None
