"""
定价策略优化案例

业务背景：
---------
电商/零售公司需要制定最优定价策略：
1. 价格弹性：价格变化对需求的影响有多大？
2. 促销效果：折扣的真实增量销售是多少？
3. 动态定价：如何根据需求实时调整价格？

核心挑战：
---------
- 内生性问题：价格往往在需求高时上调，导致价格-需求关系被混淆
- 促销选择：参与促销的商品/用户本身就不同
- 长期效应：频繁促销可能损害品牌价值

方法论：
-------
1. 价格弹性估计：使用工具变量或自然实验
2. 促销增量测量：因果推断分离自然销售和增量销售
3. 最优定价：基于弹性的利润最大化

面试考点：
---------
- 什么是价格弹性？如何估计？
- 促销的真实 ROI 怎么算？
- 什么是"促销疲劳"？
- 动态定价的伦理问题？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProductPricing:
    """产品定价结果"""
    product_id: str
    current_price: float
    optimal_price: float
    price_elasticity: float
    expected_demand_change: float
    expected_revenue_change: float
    expected_profit_change: float


def generate_pricing_data(
    n_products: int = 100,
    n_days: int = 365,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成定价数据

    模拟真实场景：
    - 价格随成本、竞争、需求波动
    - 需求受价格、季节、促销影响
    - 存在内生性：需求高时可能提价
    """
    np.random.seed(seed)

    records = []

    for product_id in range(n_products):
        # 产品基础属性
        base_price = np.random.uniform(20, 200)
        base_cost = base_price * np.random.uniform(0.4, 0.7)
        base_demand = np.random.uniform(50, 500)
        price_sensitivity = np.random.uniform(1.0, 3.0)  # 价格弹性

        # 品类（影响弹性）
        category = np.random.choice(['electronics', 'fashion', 'food', 'home'])
        category_elasticity = {
            'electronics': 1.5,
            'fashion': 2.0,
            'food': 0.8,
            'home': 1.2
        }

        for day in range(n_days):
            # 季节性
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * day / 365)

            # 星期效应
            weekday = day % 7
            weekend_boost = 1.2 if weekday >= 5 else 1.0

            # 需求冲击（不可观测）
            demand_shock = np.random.normal(0, 0.2)

            # 成本冲击
            cost_shock = np.random.normal(0, 0.05)
            current_cost = base_cost * (1 + cost_shock)

            # 竞争对手价格（工具变量）
            competitor_price = base_price * np.random.uniform(0.9, 1.1)

            # 定价决策（内生性：需求高时可能提价）
            # 价格 = 基础价格 * (1 + 需求因素 + 成本因素 + 随机)
            price_adjustment = 0.1 * demand_shock + 0.5 * cost_shock + np.random.normal(0, 0.05)
            current_price = base_price * (1 + price_adjustment)

            # 是否促销
            is_promotion = np.random.rand() < 0.1  # 10% 天数有促销
            promotion_depth = np.random.uniform(0.1, 0.3) if is_promotion else 0
            final_price = current_price * (1 - promotion_depth)

            # 需求函数（真实因果关系）
            # log(Q) = a - b * log(P) + 季节 + 周末 + 冲击
            log_demand = (
                np.log(base_demand) -
                price_sensitivity * category_elasticity[category] * np.log(final_price / base_price) +
                0.2 * (seasonality - 1) +
                0.1 * (weekend_boost - 1) +
                demand_shock +
                0.3 * is_promotion  # 促销额外吸引
            )

            quantity = np.exp(log_demand) * np.random.lognormal(0, 0.1)
            quantity = max(1, quantity)

            revenue = final_price * quantity
            cost = current_cost * quantity
            profit = revenue - cost

            records.append({
                'product_id': f'P{product_id:03d}',
                'category': category,
                'day': day,
                'weekday': weekday,
                'base_price': base_price,
                'base_cost': base_cost,
                'current_cost': current_cost,
                'list_price': current_price,
                'final_price': final_price,
                'competitor_price': competitor_price,
                'is_promotion': int(is_promotion),
                'promotion_depth': promotion_depth,
                'quantity': quantity,
                'revenue': revenue,
                'profit': profit,
                'seasonality': seasonality,
                '_true_elasticity': price_sensitivity * category_elasticity[category],
                '_demand_shock': demand_shock,
            })

    return pd.DataFrame(records)


class PriceElasticityEstimator:
    """价格弹性估计器"""

    def __init__(self, method: str = 'ols'):
        """
        Parameters:
        -----------
        method: 估计方法
            - 'ols': 普通最小二乘（有偏）
            - 'iv': 工具变量法
            - 'fe': 固定效应
        """
        self.method = method
        self.elasticity = None
        self.se = None

    def fit(self, df: pd.DataFrame, product_id: Optional[str] = None) -> Dict:
        """
        估计价格弹性

        使用对数-对数模型：log(Q) = α - β * log(P) + ε
        β 就是价格弹性
        """
        if product_id:
            data = df[df['product_id'] == product_id].copy()
        else:
            data = df.copy()

        # 对数变换
        data['log_price'] = np.log(data['final_price'])
        data['log_quantity'] = np.log(data['quantity'])
        data['log_competitor'] = np.log(data['competitor_price'])

        if self.method == 'ols':
            return self._fit_ols(data)
        elif self.method == 'iv':
            return self._fit_iv(data)
        elif self.method == 'fe':
            return self._fit_fe(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_ols(self, data: pd.DataFrame) -> Dict:
        """OLS 估计（可能有偏）"""
        X = data[['log_price', 'seasonality', 'is_promotion']].values
        y = data['log_quantity'].values

        model = LinearRegression()
        model.fit(X, y)

        self.elasticity = -model.coef_[0]  # 负号因为是需求函数

        # 计算标准误
        y_pred = model.predict(X)
        residuals = y - y_pred
        n = len(y)
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k - 1)
        var_coef = mse * np.linalg.inv(X.T @ X)
        self.se = np.sqrt(var_coef[0, 0])

        return {
            'method': 'OLS',
            'elasticity': self.elasticity,
            'se': self.se,
            'ci_lower': self.elasticity - 1.96 * self.se,
            'ci_upper': self.elasticity + 1.96 * self.se,
            'r_squared': model.score(X, y),
            'n_obs': n,
            'bias_warning': '可能因内生性而有偏'
        }

    def _fit_iv(self, data: pd.DataFrame) -> Dict:
        """
        工具变量估计

        使用竞争对手价格作为工具变量：
        - 与自身价格相关（竞争）
        - 与需求冲击不相关（外生）
        """
        # 第一阶段：用 IV 预测价格
        Z = data[['log_competitor', 'seasonality']].values
        X_endo = data['log_price'].values

        stage1 = LinearRegression()
        stage1.fit(Z, X_endo)
        X_hat = stage1.predict(Z)

        # 第二阶段：用预测价格估计需求
        X_stage2 = np.column_stack([X_hat, data['seasonality'].values, data['is_promotion'].values])
        y = data['log_quantity'].values

        stage2 = LinearRegression()
        stage2.fit(X_stage2, y)

        self.elasticity = -stage2.coef_[0]

        # 标准误（简化计算）
        y_pred = stage2.predict(X_stage2)
        residuals = y - y_pred
        n = len(y)
        mse = np.sum(residuals**2) / (n - 4)
        self.se = np.sqrt(mse / np.sum((X_hat - X_hat.mean())**2))

        # 第一阶段 F 统计量
        ss_total = np.sum((X_endo - X_endo.mean())**2)
        ss_resid = np.sum((X_endo - X_hat)**2)
        r2_stage1 = 1 - ss_resid / ss_total
        f_stat = (r2_stage1 / 1) / ((1 - r2_stage1) / (n - 2))

        return {
            'method': 'IV (2SLS)',
            'elasticity': self.elasticity,
            'se': self.se,
            'ci_lower': self.elasticity - 1.96 * self.se,
            'ci_upper': self.elasticity + 1.96 * self.se,
            'first_stage_f': f_stat,
            'n_obs': n,
            'instrument': 'competitor_price',
            'note': 'F > 10 表示工具变量有效'
        }

    def _fit_fe(self, data: pd.DataFrame) -> Dict:
        """固定效应估计（去除产品固定效应）"""
        # 对每个产品去均值
        data = data.copy()
        for col in ['log_price', 'log_quantity', 'seasonality']:
            data[f'{col}_dm'] = data.groupby('product_id')[col].transform(
                lambda x: x - x.mean()
            )

        X = data[['log_price_dm', 'seasonality_dm']].values
        y = data['log_quantity_dm'].values

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        self.elasticity = -model.coef_[0]

        # 标准误
        y_pred = model.predict(X)
        residuals = y - y_pred
        n = len(y)
        n_products = data['product_id'].nunique()
        mse = np.sum(residuals**2) / (n - n_products - 2)
        var_coef = mse * np.linalg.inv(X.T @ X)
        self.se = np.sqrt(var_coef[0, 0])

        return {
            'method': 'Fixed Effects',
            'elasticity': self.elasticity,
            'se': self.se,
            'ci_lower': self.elasticity - 1.96 * self.se,
            'ci_upper': self.elasticity + 1.96 * self.se,
            'n_obs': n,
            'n_products': n_products,
            'note': '控制了产品固定效应'
        }


class PromotionAnalyzer:
    """促销效果分析器"""

    def __init__(self):
        self.results = {}

    def analyze(self, df: pd.DataFrame) -> Dict:
        """分析促销的增量效果"""
        # 朴素对比
        promo_sales = df[df['is_promotion'] == 1]['quantity'].mean()
        non_promo_sales = df[df['is_promotion'] == 0]['quantity'].mean()
        naive_lift = (promo_sales - non_promo_sales) / non_promo_sales

        # 控制混淆因素
        df = df.copy()
        X = df[['seasonality', 'weekday', 'final_price']].values
        T = df['is_promotion'].values
        Y = df['quantity'].values

        # 倾向得分
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)

        # IPW 估计
        weights = T / ps + (1 - T) / (1 - ps)
        ate_ipw = np.average(Y * T / ps, weights=None) - np.average(Y * (1 - T) / (1 - ps), weights=None)
        ate_ipw_lift = ate_ipw / non_promo_sales

        # 计算促销 ROI
        avg_margin = df['profit'].sum() / df['revenue'].sum()
        promo_cost = df[df['is_promotion'] == 1]['promotion_depth'].mean() * \
                     df[df['is_promotion'] == 1]['list_price'].mean() * \
                     df[df['is_promotion'] == 1]['quantity'].mean()

        incremental_revenue = ate_ipw * df[df['is_promotion'] == 1]['final_price'].mean()
        incremental_profit = incremental_revenue * avg_margin
        promo_roi = (incremental_profit - promo_cost) / promo_cost if promo_cost > 0 else 0

        self.results = {
            'promo_days': int(T.sum()),
            'non_promo_days': int((1 - T).sum()),
            'promo_avg_sales': promo_sales,
            'non_promo_avg_sales': non_promo_sales,
            'naive_lift': naive_lift,
            'causal_lift': ate_ipw_lift,
            'bias': naive_lift - ate_ipw_lift,
            'incremental_sales': ate_ipw,
            'promo_roi': promo_roi,
        }

        return self.results


class DynamicPricingOptimizer:
    """动态定价优化器"""

    def __init__(self, elasticity: float, base_price: float, base_cost: float):
        self.elasticity = elasticity
        self.base_price = base_price
        self.base_cost = base_cost

    def optimal_price(self, base_demand: float = 100) -> Tuple[float, float, float]:
        """
        计算利润最大化价格

        利润 = (P - C) * Q(P)
        Q(P) = Q0 * (P/P0)^(-ε)

        最优价格: P* = ε * C / (ε - 1)  (当 ε > 1)
        """
        if self.elasticity <= 1:
            # 弹性小于1时，提价总能增加收入
            return self.base_price * 1.5, base_demand * 0.8, None

        # 最优价格公式
        optimal_p = self.elasticity * self.base_cost / (self.elasticity - 1)

        # 限制在合理范围
        optimal_p = np.clip(optimal_p, self.base_cost * 1.1, self.base_price * 2)

        # 预期需求变化
        demand_change = (optimal_p / self.base_price) ** (-self.elasticity)

        # 利润比较
        current_profit = (self.base_price - self.base_cost) * base_demand
        optimal_profit = (optimal_p - self.base_cost) * base_demand * demand_change

        return optimal_p, demand_change, (optimal_profit - current_profit) / current_profit


def run_pricing_analysis(
    n_products: int = 50,
    n_days: int = 365,
    estimation_method: str = 'iv',
    seed: int = 42
) -> Tuple[go.Figure, str]:
    """运行定价分析"""

    # 1. 生成数据
    df = generate_pricing_data(n_products=n_products, n_days=n_days, seed=seed)

    # 2. 估计价格弹性（多种方法对比）
    estimators = {
        'OLS': PriceElasticityEstimator(method='ols'),
        'IV': PriceElasticityEstimator(method='iv'),
        'FE': PriceElasticityEstimator(method='fe')
    }

    elasticity_results = {}
    for name, est in estimators.items():
        elasticity_results[name] = est.fit(df)

    # 真实弹性
    true_elasticity = df['_true_elasticity'].mean()

    # 3. 促销分析
    promo_analyzer = PromotionAnalyzer()
    promo_results = promo_analyzer.analyze(df)

    # 4. 最优定价计算（使用IV估计的弹性）
    iv_elasticity = elasticity_results['IV']['elasticity']
    avg_price = df['base_price'].mean()
    avg_cost = df['base_cost'].mean()

    optimizer = DynamicPricingOptimizer(iv_elasticity, avg_price, avg_cost)
    optimal_price, demand_change, profit_change = optimizer.optimal_price(100)

    # 5. 可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '价格弹性估计对比',
            '价格-需求关系',
            '促销效果分析',
            '最优定价模拟'
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    colors = ['#EB5757', '#2D9CDB', '#27AE60']

    # 图1：弹性估计对比
    methods = list(elasticity_results.keys()) + ['True']
    elasticities = [elasticity_results[m]['elasticity'] for m in elasticity_results.keys()] + [true_elasticity]
    errors = [elasticity_results[m]['se'] * 1.96 for m in elasticity_results.keys()] + [0]

    fig.add_trace(
        go.Bar(
            x=methods,
            y=elasticities,
            error_y=dict(type='data', array=errors),
            marker_color=colors + ['#9B59B6'],
            text=[f'{e:.2f}' for e in elasticities],
            textposition='outside'
        ),
        row=1, col=1
    )

    # 图2：价格-需求散点图
    sample = df.sample(min(1000, len(df)), random_state=seed)
    fig.add_trace(
        go.Scatter(
            x=np.log(sample['final_price']),
            y=np.log(sample['quantity']),
            mode='markers',
            marker=dict(
                color=sample['is_promotion'],
                colorscale=[[0, '#2D9CDB'], [1, '#EB5757']],
                size=5,
                opacity=0.5
            ),
            name='观测数据'
        ),
        row=1, col=2
    )

    # 添加拟合线
    x_range = np.linspace(sample['final_price'].min(), sample['final_price'].max(), 100)
    y_pred = np.log(sample['quantity'].mean()) - iv_elasticity * (np.log(x_range) - np.log(sample['final_price'].mean()))
    fig.add_trace(
        go.Scatter(
            x=np.log(x_range),
            y=y_pred,
            mode='lines',
            line=dict(color='#27AE60', width=2),
            name=f'弹性={iv_elasticity:.2f}'
        ),
        row=1, col=2
    )

    # 图3：促销效果
    lift_labels = ['朴素估计', '因果估计', '偏差']
    lift_values = [
        promo_results['naive_lift'] * 100,
        promo_results['causal_lift'] * 100,
        promo_results['bias'] * 100
    ]

    fig.add_trace(
        go.Bar(
            x=lift_labels,
            y=lift_values,
            marker_color=['#EB5757', '#27AE60', '#F2994A'],
            text=[f'{v:.1f}%' for v in lift_values],
            textposition='outside'
        ),
        row=2, col=1
    )

    # 图4：最优定价曲线
    price_range = np.linspace(avg_cost * 1.1, avg_price * 1.5, 100)
    base_demand = 100

    revenues = []
    profits = []
    for p in price_range:
        q = base_demand * (p / avg_price) ** (-iv_elasticity)
        revenues.append(p * q)
        profits.append((p - avg_cost) * q)

    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=revenues,
            mode='lines',
            name='收入',
            line=dict(color='#2D9CDB', width=2)
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=profits,
            mode='lines',
            name='利润',
            line=dict(color='#27AE60', width=2)
        ),
        row=2, col=2
    )

    # 标记最优点
    fig.add_trace(
        go.Scatter(
            x=[optimal_price],
            y=[max(profits)],
            mode='markers',
            marker=dict(color='#EB5757', size=12, symbol='star'),
            name='最优价格'
        ),
        row=2, col=2
    )

    fig.add_vline(x=avg_price, line_dash="dash", line_color="gray", row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=True,
        template='plotly_white',
        title_text='定价策略优化分析',
        title_x=0.5
    )

    fig.update_xaxes(title_text='方法', row=1, col=1)
    fig.update_yaxes(title_text='价格弹性', row=1, col=1)
    fig.update_xaxes(title_text='log(价格)', row=1, col=2)
    fig.update_yaxes(title_text='log(销量)', row=1, col=2)
    fig.update_xaxes(title_text='估计类型', row=2, col=1)
    fig.update_yaxes(title_text='销量提升 (%)', row=2, col=1)
    fig.update_xaxes(title_text='价格', row=2, col=2)
    fig.update_yaxes(title_text='收入/利润', row=2, col=2)

    # 6. 生成报告
    report = f"""
### 定价策略分析报告

#### 1. 数据概况
- 产品数量: {n_products}
- 观察天数: {n_days}
- 总交易数: {len(df):,}

#### 2. 价格弹性估计

| 方法 | 弹性估计 | 标准误 | 95% CI | 说明 |
|-----|---------|-------|--------|------|
| OLS | {elasticity_results['OLS']['elasticity']:.2f} | {elasticity_results['OLS']['se']:.2f} | [{elasticity_results['OLS']['ci_lower']:.2f}, {elasticity_results['OLS']['ci_upper']:.2f}] | 有内生性偏差 |
| IV (2SLS) | {elasticity_results['IV']['elasticity']:.2f} | {elasticity_results['IV']['se']:.2f} | [{elasticity_results['IV']['ci_lower']:.2f}, {elasticity_results['IV']['ci_upper']:.2f}] | 使用竞品价格作为工具变量 |
| 固定效应 | {elasticity_results['FE']['elasticity']:.2f} | {elasticity_results['FE']['se']:.2f} | [{elasticity_results['FE']['ci_lower']:.2f}, {elasticity_results['FE']['ci_upper']:.2f}] | 控制产品固定效应 |
| **真实值** | **{true_elasticity:.2f}** | - | - | 数据生成参数 |

**关键发现**: OLS 低估了价格弹性（偏差 {(elasticity_results['OLS']['elasticity'] - true_elasticity):.2f}），因为需求高时商家倾向于提价。

#### 3. 促销效果分析

| 指标 | 值 |
|-----|-----|
| 促销天数 | {promo_results['promo_days']:,} |
| 促销日均销量 | {promo_results['promo_avg_sales']:.0f} |
| 非促销日均销量 | {promo_results['non_promo_avg_sales']:.0f} |
| **朴素提升** | {promo_results['naive_lift']*100:.1f}% |
| **因果提升** | {promo_results['causal_lift']*100:.1f}% |
| 偏差 | {promo_results['bias']*100:.1f}% |
| 促销 ROI | {promo_results['promo_roi']*100:.1f}% |

**关键发现**: 朴素估计高估促销效果 {promo_results['bias']*100:.1f}%，因为促销常在旺季进行。

#### 4. 最优定价建议

| 指标 | 当前 | 最优 | 变化 |
|-----|-----|-----|------|
| 价格 | ¥{avg_price:.0f} | ¥{optimal_price:.0f} | {(optimal_price/avg_price-1)*100:+.1f}% |
| 预期需求变化 | - | - | {(demand_change-1)*100:+.1f}% |
| 预期利润变化 | - | - | {profit_change*100 if profit_change else 'N/A':+.1f}% |

#### 5. 业务建议

1. **定价策略**
   - 弹性 > 1 的商品：适度降价可增加总收入
   - 弹性 < 1 的商品：可考虑提价
   - 建议按品类制定差异化策略

2. **促销策略**
   - 控制促销频率，避免"促销疲劳"
   - 真实增量仅为朴素估计的 {promo_results['causal_lift']/promo_results['naive_lift']*100:.0f}%
   - 建议在淡季而非旺季促销，增量更大

3. **数据收集**
   - 收集竞品价格，用于因果估计
   - 考虑进行价格实验（随机调价）
   - 建立价格弹性监控系统

---
*基于因果推断的定价策略分析*
"""

    return fig, report


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 定价策略优化

### 业务背景

定价是最直接影响利润的杠杆：
- 价格提高 1%，利润平均提高 11%（麦肯锡研究）
- 但定价太高会损失销量，太低会损失利润

**核心问题**：
- 价格变化对销量的影响是多少？（价格弹性）
- 促销的真实增量效果？
- 最优价格应该是多少？

**常见陷阱**：
- 内生性：需求高时提价，导致价格-需求关系被高估
- 促销选择：旺季促销，效果被高估
- 短期思维：频繁促销损害品牌

---

### 核心概念

| 概念 | 公式 | 解释 |
|-----|------|------|
| **价格弹性** | ε = -(∂Q/∂P) × (P/Q) | 价格变化1%，需求变化ε% |
| **最优价格** | P* = εC/(ε-1) | 利润最大化价格 |
| **工具变量** | Cov(Z,X)≠0, Cov(Z,ε)=0 | 解决内生性的方法 |

---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                n_products = gr.Slider(20, 100, 50, step=10, label="产品数量")
                n_days = gr.Slider(90, 365, 365, step=30, label="观察天数")
                estimation_method = gr.Radio(
                    choices=['ols', 'iv', 'fe'],
                    value='iv',
                    label="主要估计方法"
                )
                seed = gr.Number(value=42, label="随机种子", precision=0)
                run_btn = gr.Button("运行分析", variant="primary")

        with gr.Row():
            plot_output = gr.Plot(label="分析可视化")

        with gr.Row():
            report_output = gr.Markdown()

        run_btn.click(
            fn=run_pricing_analysis,
            inputs=[n_products, n_days, estimation_method, seed],
            outputs=[plot_output, report_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: 什么是价格弹性？**
> 价格弹性 = 需求变化百分比 / 价格变化百分比
> - ε > 1: 弹性需求，降价增加收入
> - ε < 1: 刚性需求，提价增加收入
> - ε = 1: 单位弹性，收入不变

**Q2: 为什么 OLS 会有偏？**
> 内生性问题：价格不是随机的，而是受需求影响
> - 需求高 → 商家提价 → 观测到"高价高销量"
> - 导致弹性被低估

**Q3: 什么是工具变量？**
> 满足两个条件的变量：
> 1. 相关性：与内生变量（价格）相关
> 2. 外生性：与误差项（需求冲击）无关
> 例：竞品价格、成本冲击、汇率

**Q4: 促销的长期效应？**
> - 促销疲劳：用户学会"等促销"
> - 品牌稀释：频繁促销降低品牌价值
> - 参考价格效应：促销价成为"锚点"

**Q5: 动态定价的伦理问题？**
> - 价格歧视：对不同人群收不同价格
> - 算法共谋：算法可能隐性串通
> - 信任损害：用户感到被"宰"
        """)

    return None


if __name__ == "__main__":
    fig, report = run_pricing_analysis()
    print(report)
