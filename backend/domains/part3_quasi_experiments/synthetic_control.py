"""合成控制法 (Synthetic Control Method) 分析"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

from .utils import generate_synthetic_control_data, fig_to_chart_data


class SyntheticControl:
    """合成控制估计器"""

    def __init__(self, treatment_period: int):
        """初始化

        Args:
            treatment_period: 处理开始的时间索引
        """
        self.treatment_period = treatment_period
        self.weights = None
        self.synthetic_control = None
        self.treatment_effect = None

    def fit(self, treated: np.ndarray, donors: np.ndarray):
        """估计合成控制权重

        Args:
            treated: 处理单位的时间序列 (T,)
            donors: 供体池的时间序列矩阵 (T, J)
        """
        treated = np.array(treated)
        donors = np.array(donors)

        # 提取前处理期数据
        treated_pre = treated[:self.treatment_period]
        donors_pre = donors[:self.treatment_period, :]

        # 优化目标：最小化前处理期的RMSE
        def objective(w):
            synthetic = donors_pre @ w
            return np.sum((treated_pre - synthetic) ** 2)

        # 约束条件
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
        )

        # 边界：权重非负
        bounds = [(0, 1) for _ in range(donors.shape[1])]

        # 初始值：等权重
        w0 = np.ones(donors.shape[1]) / donors.shape[1]

        # 求解
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        self.weights = result.x

        # 生成合成控制
        self.synthetic_control = donors @ self.weights

        # 计算处理效应
        self.treatment_effect = treated[self.treatment_period:] - self.synthetic_control[self.treatment_period:]

        return self

    def get_weights(self):
        """返回权重"""
        return self.weights

    def get_effect(self):
        """返回平均处理效应"""
        return np.mean(self.treatment_effect)

    def predict(self):
        """返回合成控制的预测值"""
        return self.synthetic_control


def analyze_synthetic_control(
    n_control_units: int = 6,
    n_pre_periods: int = 18,
    effect_size: float = -15.0
) -> dict:
    """合成控制法分析

    Args:
        n_control_units: 对照单位数量
        n_pre_periods: 处理前时期数
        effect_size: 处理效应大小

    Returns:
        包含图表、表格和摘要的字典
    """
    # 生成数据
    df, treatment_time = generate_synthetic_control_data(
        n_control_units=n_control_units,
        n_periods=31,
        treatment_period=n_pre_periods,
        treatment_effect=effect_size
    )

    # 准备数据
    treated = df['treated'].values
    donor_cols = [col for col in df.columns if col not in ['year', 'treated']]
    donors = df[donor_cols].values

    # 拟合模型
    sc = SyntheticControl(treatment_period=treatment_time)
    sc.fit(treated, donors)

    # 获取权重
    weights = sc.get_weights()
    weight_df = pd.DataFrame({
        '供体单位': donor_cols,
        '权重': weights
    }).sort_values('权重', ascending=False)

    # 创建可视化 - 主图
    fig_main = go.Figure()

    # 处理单位
    fig_main.add_trace(go.Scatter(
        x=df['year'],
        y=df['treated'],
        name='实际（处理单位）',
        line=dict(color='#EB5757', width=3),
        mode='lines+markers'
    ))

    # 合成控制
    synthetic = sc.predict()
    fig_main.add_trace(go.Scatter(
        x=df['year'],
        y=synthetic,
        name='合成控制（反事实）',
        line=dict(color='#2D9CDB', width=3, dash='dash'),
        mode='lines+markers'
    ))

    # 处理时点
    treatment_year = df['year'].iloc[treatment_time]
    fig_main.add_vline(
        x=treatment_year,
        line_dash="dash",
        line_color="gray",
        annotation_text="处理开始",
        annotation_position="top"
    )

    # 添加处理后的阴影区域
    fig_main.add_vrect(
        x0=treatment_year, x1=df['year'].max(),
        fillcolor="lightgray", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="处理后", annotation_position="top left"
    )

    fig_main.update_layout(
        title='合成控制法：实际 vs 反事实',
        xaxis_title='年份',
        yaxis_title='结果变量',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    # 创建权重图
    fig_weights = go.Figure()

    fig_weights.add_trace(go.Bar(
        x=weight_df['供体单位'],
        y=weight_df['权重'],
        marker_color='#2D9CDB',
        text=weight_df['权重'].round(3),
        textposition='auto',
    ))

    fig_weights.update_layout(
        title='合成控制权重分布',
        xaxis_title='供体单位',
        yaxis_title='权重',
        template='plotly_white',
        height=400
    )

    # 计算统计量
    avg_effect = sc.get_effect()

    # 前处理期拟合度
    pre_rmse = np.sqrt(np.mean((treated[:treatment_time] - synthetic[:treatment_time]) ** 2))

    # 构建权重表格
    weight_table = weight_df.head(10).to_dict('records')

    summary = f"""
## 合成控制法分析结果

### 核心思想

合成控制法通过对照单位的**加权组合**构建反事实:
- 不是找一个完美的对照单位
- 而是用多个不完美的对照单位"合成"一个完美的

### 关键指标

| 指标 | 值 |
|------|-----|
| 处理前拟合RMSE | {pre_rmse:.4f} |
| 平均处理效应 | {avg_effect:.4f} |
| 真实效应 | {effect_size:.4f} |
| 非零权重单位数 | {(weights > 0.01).sum()} / {len(weights)} |

### 权重解读

- **权重最大的单位**: {weight_df.iloc[0]['供体单位']} ({weight_df.iloc[0]['权重']:.1%})
- **权重稀疏性**: {(weights < 0.01).sum()} 个单位权重接近0

### 方法优势

1. **灵活性**: 不需要找到完美的单配对
2. **透明性**: 权重明确显示每个对照单位的贡献
3. **外推最小化**: 在凸组合内进行插值

### 关键假设

1. **可合成性**: 处理单位的反事实可由对照单位线性组合近似
2. **无干扰**: 对照单位未受处理影响
3. **前期拟合**: 处理前拟合越好，估计越可靠
"""

    return {
        "charts": [
            fig_to_chart_data(fig_main),
            fig_to_chart_data(fig_weights)
        ],
        "tables": [
            {
                "title": "权重分配（前10）",
                "headers": ["供体单位", "权重"],
                "rows": [[row['供体单位'], f"{row['权重']:.4f}"] for row in weight_table]
            }
        ],
        "summary": summary,
        "metrics": {
            "avg_effect": float(avg_effect),
            "pre_rmse": float(pre_rmse),
            "true_effect": float(effect_size),
            "n_nonzero_weights": int((weights > 0.01).sum()),
            "max_weight_unit": weight_df.iloc[0]['供体单位'],
            "max_weight_value": float(weight_df.iloc[0]['权重'])
        }
    }
