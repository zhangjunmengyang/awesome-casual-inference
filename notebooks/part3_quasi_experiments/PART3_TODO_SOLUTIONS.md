# Part 3: Quasi-Experiments - 完整TODO答案与从零实现

## 目录
1. [DID - 双重差分法](#1-did---双重差分法)
2. [Synthetic Control - 合成控制法](#2-synthetic-control---合成控制法)
3. [RDD - 断点回归](#3-rdd---断点回归)
4. [IV - 工具变量](#4-iv---工具变量)

---

# 1. DID - 双重差分法

## TODO 练习 1: 安慰剂检验（参考答案）

```python
# TODO练习1：安慰剂检验
# 使用政策前的数据，假设一个假的政策时间点

# 步骤1：筛选政策前的数据（2024-01-01 到 2024-02-29）
df_pre_only = df_city[df_city['date'] < pd.Timestamp('2024-03-01')].copy()

# 步骤2：假设假的政策时间点（比如2024-02-01）
fake_policy_date = pd.Timestamp('2024-02-01')
df_pre_only['post_fake'] = (df_pre_only['date'] >= fake_policy_date).astype(int)

# 步骤3：运行DID估计
# 使用相同的处理组定义
X_placebo = df_pre_only[['treated', 'post_fake', 'treated_x_post']].values
y_placebo = df_pre_only['orders'].values

model_placebo = LinearRegression()
model_placebo.fit(X_placebo, y_placebo)

placebo_did = model_placebo.coef_[2]  # treated_x_post系数

# 步骤4：检验显著性
from scipy import stats

# 计算标准误（简化版）
residuals = y_placebo - model_placebo.predict(X_placebo)
n = len(y_placebo)
k = X_placebo.shape[1]
mse = np.sum(residuals**2) / (n - k - 1)
var_covar = mse * np.linalg.inv(X_placebo.T @ X_placebo)
se_placebo = np.sqrt(var_covar[2, 2])

t_stat = placebo_did / se_placebo
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))

print("="*60)
print("安慰剂检验结果")
print("="*60)
print(f"假政策时间点: {fake_policy_date.date()}")
print(f"DID估计: {placebo_did:.2f}")
print(f"标准误: {se_placebo:.2f}")
print(f"t统计量: {t_stat:.2f}")
print(f"p值: {p_value:.4f}")
print(f"显著性: {'显著❌ (有问题!)' if p_value < 0.05 else '不显著✅ (通过检验)'}")
print("="*60)

# 解读
if p_value >= 0.05:
    print("\n✅ 安慰剂检验通过！")
    print("   在政策前假设的处理时点没有发现显著效应，")
    print("   说明真实政策效应不是由趋势差异导致的。")
else:
    print("\n❌ 安慰剂检验未通过！")
    print("   在政策前就检测到了'效应'，可能违反平行趋势假设。")
```

## TODO 练习 2: 平台政策变更案例（完整解答）

```python
# 生成数据
np.random.seed(42)

# 城市和时间
cities = ['北京', '上海', '广州', '深圳', '成都', '杭州']
dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')

# 处理组：北京、上海（2024-03-01推出无接触配送）
treated_cities = ['北京', '上海']
policy_date = pd.Timestamp('2024-03-01')

data = []
for city in cities:
    is_treated = city in treated_cities

    # 基线订单量（城市规模差异）
    base_orders = {'北京': 5000, '上海': 4500, '广州': 3000,
                   '深圳': 3200, '成都': 2500, '杭州': 2000}[city]

    # 时间趋势（所有城市都在增长）
    trend = np.random.normal(10, 2)

    for date in dates:
        # 天数（从1月1日开始）
        days = (date - dates[0]).days

        # 基础订单（带趋势）
        orders = base_orders + trend * days

        # 星期效应（周末订单多）
        if date.dayofweek >= 5:  # 周末
            orders += 500

        # 随机波动
        orders += np.random.normal(0, 200)

        # 政策效应（只对处理组，且在政策后）
        if is_treated and date >= policy_date:
            # 真实政策效应：+15%
            orders *= 1.15

        data.append({
            'city': city,
            'date': date,
            'orders': orders,
            'treated': int(is_treated),
            'post': int(date >= policy_date)
        })

df_case = pd.DataFrame(data)
df_case['treated_x_post'] = df_case['treated'] * df_case['post']

# 1. 可视化趋势
fig = go.Figure()

for city in cities:
    city_data = df_case[df_case['city'] == city]
    color = '#EB5757' if city in treated_cities else '#95A5A6'

    fig.add_trace(go.Scatter(
        x=city_data['date'],
        y=city_data['orders'],
        name=city,
        line=dict(color=color, width=2 if city in treated_cities else 1),
        mode='lines'
    ))

fig.add_vline(x=policy_date, line_dash="dash", line_color="black",
              annotation_text="无接触配送上线")

fig.update_layout(
    title='各城市订单量趋势（案例2）',
    xaxis_title='日期',
    yaxis_title='订单量',
    template='plotly_white',
    height=500
)
fig.show()

# 2. DID估计
X = df_case[['treated', 'post', 'treated_x_post']].values
y = df_case['orders'].values

model_case = LinearRegression()
model_case.fit(X, y)

did_effect = model_case.coef_[2]

print("\n" + "="*60)
print("案例2：无接触配送功能对订单量的影响")
print("="*60)
print(f"DID估计: {did_effect:.2f}")
print(f"解释: 无接触配送使订单量平均增加 {did_effect:.0f} 单")

# 3. 计算百分比效应
treated_pre_mean = df_case[(df_case['treated']==1) & (df_case['post']==0)]['orders'].mean()
pct_effect = (did_effect / treated_pre_mean) * 100
print(f"百分比效应: {pct_effect:.2f}%")
print("="*60)

# 4. 平行趋势检验
# 按周汇总
df_case['week'] = df_case['date'].dt.to_period('W').dt.to_timestamp()
df_weekly = df_case.groupby(['city', 'week', 'treated', 'post']).agg({'orders': 'mean'}).reset_index()

# 只看政策前的数据
df_pre = df_weekly[df_weekly['post'] == 0]

# 对每个城市拟合趋势
treated_trend = []
control_trend = []

for city in treated_cities:
    city_data = df_pre[df_pre['city'] == city]
    X_trend = np.arange(len(city_data)).reshape(-1, 1)
    y_trend = city_data['orders'].values
    model_trend = LinearRegression().fit(X_trend, y_trend)
    treated_trend.append(model_trend.coef_[0])

for city in cities:
    if city not in treated_cities:
        city_data = df_pre[df_pre['city'] == city]
        X_trend = np.arange(len(city_data)).reshape(-1, 1)
        y_trend = city_data['orders'].values
        model_trend = LinearRegression().fit(X_trend, y_trend)
        control_trend.append(model_trend.coef_[0])

print("\n平行趋势检验（政策前）：")
print(f"处理组平均趋势: {np.mean(treated_trend):.2f}")
print(f"对照组平均趋势: {np.mean(control_trend):.2f}")
print(f"趋势差异: {np.mean(treated_trend) - np.mean(control_trend):.2f}")

if abs(np.mean(treated_trend) - np.mean(control_trend)) < 5:
    print("✅ 趋势差异较小，平行趋势假设较为合理")
else:
    print("⚠️ 趋势差异较大，需要谨慎解读DID结果")
```

---

# 2. Synthetic Control - 合成控制法

## 从零实现：完整的合成控制估计器

```python
class MySyntheticControlFromScratch:
    """
    从零实现合成控制法

    核心思想：
    1. 找到权重 W，使得合成控制在前处理期尽可能接近处理单位
    2. 权重约束：非负 + 和为1
    3. 优化方法：二次规划
    """

    def __init__(self, treatment_period):
        self.treatment_period = treatment_period
        self.weights = None
        self.synthetic_outcome = None

    def optimize_weights(self, Y_treated_pre, Y_donors_pre):
        """
        手动实现权重优化（二次规划）

        目标：min ||Y_treated_pre - Y_donors_pre @ W||^2
        约束：W >= 0, sum(W) = 1
        """
        from scipy.optimize import minimize

        n_donors = Y_donors_pre.shape[1]

        # 目标函数
        def objective(w):
            synthetic = Y_donors_pre @ w
            return np.sum((Y_treated_pre - synthetic) ** 2)

        # 约束
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 和为1
        ]

        # 边界
        bounds = [(0, 1) for _ in range(n_donors)]

        # 初始值（等权重）
        w0 = np.ones(n_donors) / n_donors

        # 优化
        result = minimize(
            objective, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )

        return result.x

    def fit(self, Y_treated, Y_donors):
        """
        拟合合成控制
        """
        # 分割前后期
        Y_treated_pre = Y_treated[:self.treatment_period]
        Y_donors_pre = Y_donors[:self.treatment_period, :]

        # 优化权重
        self.weights = self.optimize_weights(Y_treated_pre, Y_donors_pre)

        # 生成合成控制（全时期）
        self.synthetic_outcome = Y_donors @ self.weights

        # 计算处理效应
        self.treatment_effect = Y_treated[self.treatment_period:] - self.synthetic_outcome[self.treatment_period:]

        return self

    def get_ate(self):
        """平均处理效应"""
        return np.mean(self.treatment_effect)

    def plot_comparison(self, years, treated_name='处理单位'):
        """可视化"""
        fig = go.Figure()

        # 处理单位
        fig.add_trace(go.Scatter(
            x=years,
            y=Y_treated,
            name=f'{treated_name}（实际）',
            line=dict(color='red', width=3)
        ))

        # 合成控制
        fig.add_trace(go.Scatter(
            x=years,
            y=self.synthetic_outcome,
            name='合成控制（反事实）',
            line=dict(color='blue', width=3, dash='dash')
        ))

        # 处理时点
        fig.add_vline(x=years[self.treatment_period], line_dash="dash")

        fig.update_layout(
            title='合成控制效果对比',
            xaxis_title='时间',
            yaxis_title='结果变量',
            template='plotly_white'
        )

        return fig


# 手动推导和验证
print("="*60)
print("从零实现合成控制法：数学推导")
print("="*60)

print("\n1. 优化问题：")
print("   min_W ||Y_1 - Y_0 * W||^2")
print("   s.t. W >= 0, sum(W) = 1")

print("\n2. 拉格朗日函数：")
print("   L(W, λ, μ) = ||Y_1 - Y_0*W||^2 + λ(sum(W)-1) - μ^T*W")

print("\n3. KKT条件：")
print("   ∂L/∂W = -2*Y_0^T*(Y_1 - Y_0*W) + λ*1 - μ = 0")
print("   sum(W) = 1")
print("   W >= 0, μ >= 0, μ^T*W = 0")

print("\n4. 求解方法：")
print("   使用序列二次规划（SLSQP）数值求解")
print("="*60)
```

## TODO 1: 实现带协变量的合成控制（完整答案）

```python
class SyntheticControlWithCovariates:
    """
    带协变量匹配的合成控制
    """

    def __init__(self, treatment_period):
        self.treatment_period = treatment_period
        self.weights = None

    def fit(self, treated_outcome, donors_outcome,
            treated_covariates=None, donors_covariates=None,
            alpha=0.5):
        """
        alpha: 协变量权重（0=只匹配结果, 1=只匹配协变量）
        """
        from scipy.optimize import minimize

        # 前处理期数据
        Y_treated_pre = treated_outcome[:self.treatment_period]
        Y_donors_pre = donors_outcome[:self.treatment_period, :]

        # 标准化（使结果变量和协变量在同一尺度）
        def standardize(x):
            return (x - np.mean(x)) / np.std(x)

        def objective(w):
            loss = 0

            # 1. 结果变量匹配（前处理期）
            if alpha < 1:
                synthetic_outcome = Y_donors_pre @ w
                outcome_loss = np.sum((standardize(Y_treated_pre) - standardize(synthetic_outcome)) ** 2)
                loss += (1 - alpha) * outcome_loss

            # 2. 协变量匹配
            if alpha > 0 and treated_covariates is not None:
                for i, (cov_treated, cov_donors) in enumerate(zip(treated_covariates, donors_covariates.T)):
                    synthetic_cov = cov_donors @ w
                    cov_loss = (standardize(cov_treated) - standardize(synthetic_cov)) ** 2
                    loss += alpha * cov_loss / len(treated_covariates)

            return loss

        # 约束和边界
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(donors_outcome.shape[1])]
        w0 = np.ones(donors_outcome.shape[1]) / donors_outcome.shape[1]

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        self.weights = result.x
        self.synthetic_outcome = donors_outcome @ self.weights

        return self

    def get_effect(self, treated_outcome):
        effect = treated_outcome[self.treatment_period:] - self.synthetic_outcome[self.treatment_period:]
        return np.mean(effect)


# 示例使用
print("\n使用协变量匹配的合成控制：")

# 生成带协变量的数据
n_time = 30
n_donors = 5

# 城市协变量（人口、GDP、互联网渗透率）
city_covariates_treated = np.array([2500, 4300, 85])  # 上海
city_covariates_donors = np.array([
    [2200, 4200, 82],  # 北京
    [1900, 2900, 78],  # 广州
    [1800, 3200, 80],  # 深圳
    [2100, 2100, 75],  # 成都
    [1200, 1900, 83]   # 杭州
])

# 结果变量（GMV）
treated_gmv = np.random.normal(2000, 100, n_time)
donors_gmv = np.random.normal(1800, 100, (n_time, n_donors))

# 拟合（alpha=0.5：结果和协变量各占50%）
sc_cov = SyntheticControlWithCovariates(treatment_period=20)
sc_cov.fit(
    treated_gmv, donors_gmv,
    city_covariates_treated, city_covariates_donors,
    alpha=0.5
)

print(f"权重: {sc_cov.weights}")
print(f"权重和: {np.sum(sc_cov.weights):.4f}")
print(f"最大权重城市: {np.argmax(sc_cov.weights)}")
```

## TODO 2: 对比 DID 和合成控制（完整答案）

```python
# 对比 DID 和合成控制的估计结果

# 使用案例1的数据（上海上线）
df_shanghai = gmv_df.copy()

# 1. Simple DID估计
# 处理组：上海
# 对照组：其他城市的简单平均

treated_city = '上海'
control_cities = [c for c in cities if c != treated_city]

# 前后期分割
pre_period = df_shanghai['month'] < months[treatment_month]
post_period = ~pre_period

# DID四个均值
treated_pre = df_shanghai.loc[pre_period, treated_city].mean()
treated_post = df_shanghai.loc[post_period, treated_city].mean()
control_pre = df_shanghai.loc[pre_period, control_cities].mean(axis=1).mean()
control_post = df_shanghai.loc[post_period, control_cities].mean(axis=1).mean()

# DID估计量
did_estimate = (treated_post - treated_pre) - (control_post - control_pre)

# 2. 合成控制估计
treated_gmv = df_shanghai[treated_city].values
donors_gmv = df_shanghai[control_cities].values

sc_model = SyntheticControl(treatment_period=treatment_month)
sc_model.fit(treated_gmv, donors_gmv)
sc_estimate = sc_model.get_effect()

# 3. 结果对比
print("="*60)
print("DID vs 合成控制：方法对比")
print("="*60)

comparison_df = pd.DataFrame({
    '方法': ['简单DID', '合成控制'],
    '估计值': [did_estimate, sc_estimate],
    '真实值': [200, 200],
    '误差': [abs(did_estimate - 200), abs(sc_estimate - 200)]
})

print(comparison_df.to_string(index=False))

print("\n" + "="*60)
print("为什么结果不同？")
print("="*60)
print("1. DID使用等权重对照组")
print("   - 所有对照城市权重相同（1/N）")
print("   - 可能不能很好匹配处理城市的趋势")

print("\n2. 合成控制使用优化权重")
print("   - 权重根据前处理期拟合度优化")
print("   - 更好地模拟了处理城市的反事实")

print(f"\n合成控制的权重分布：")
for i, city in enumerate(control_cities):
    print(f"   {city}: {sc_model.weights[i]:.3f}")

print("\n3. 何时用哪个方法？")
print("   - 多个处理单位 → DID")
print("   - 单个处理单位 + 找不到完美对照 → 合成控制")
print("   - 平行趋势成立 → DID更简单")
print("   - 平行趋势存疑 → 合成控制更灵活")
print("="*60)

# 可视化对比
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['DID: 等权重对照组', '合成控制: 优化权重对照组']
)

# 左图：DID
simple_avg = df_shanghai[control_cities].mean(axis=1)

fig.add_trace(go.Scatter(
    x=df_shanghai['month'], y=df_shanghai[treated_city],
    name='上海（实际）', line=dict(color='red', width=2)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_shanghai['month'], y=simple_avg,
    name='简单平均（对照）', line=dict(color='gray', width=2, dash='dash')
), row=1, col=1)

# 右图：合成控制
fig.add_trace(go.Scatter(
    x=df_shanghai['month'], y=df_shanghai[treated_city],
    name='上海（实际）', line=dict(color='red', width=2),
    showlegend=False
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=df_shanghai['month'], y=sc_model.synthetic_control,
    name='合成控制（对照）', line=dict(color='blue', width=2, dash='dash')
), row=1, col=2)

# 处理时点
for col in [1, 2]:
    fig.add_vline(x=months[treatment_month], line_dash="dash",
                  line_color="black", row=1, col=col)

fig.update_layout(height=500, template='plotly_white')
fig.show()
```

## TODO 3: 评估上海上线的因果效应（完整解答）

```python
# 完整的上海上线效果评估流程

print("="*60)
print("上海上线效果评估（完整流程）")
print("="*60)

# 步骤1: 估计合成控制
sc_shanghai = SyntheticControl(treatment_month)
sc_shanghai.fit(
    gmv_df['上海'].values,
    gmv_df[cities[1:]].values
)

print("\n步骤1: 合成控制估计")
print(f"平均处理效应: {sc_shanghai.get_effect():.2f} 万元/月")

# 步骤2: 权重分析
weights_df = pd.DataFrame({
    '城市': cities[1:],
    '权重': sc_shanghai.weights
}).sort_values('权重', ascending=False)

print("\n步骤2: 权重分布")
print(weights_df.to_string(index=False))

# 步骤3: 可视化
synthetic_gmv = sc_shanghai.predict()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=gmv_df['month'],
    y=gmv_df['上海'],
    name='上海（实际）',
    line=dict(color='red', width=3)
))

fig.add_trace(go.Scatter(
    x=gmv_df['month'],
    y=synthetic_gmv,
    name='合成上海（反事实）',
    line=dict(color='blue', width=3, dash='dash')
))

fig.add_vline(x=months[treatment_month], line_dash="dash", line_color="black",
              annotation_text="上海上线")

fig.update_layout(
    title='上海上线效果评估',
    xaxis_title='月份',
    yaxis_title='GMV（万元）',
    template='plotly_white',
    height=500
)
fig.show()

# 步骤4: Placebo Tests
print("\n步骤3: Placebo Tests")

placebo_results_sh = placebo_test(
    gmv_df['上海'].values,
    gmv_df[cities[1:]].values,
    treatment_month,
    cities[1:]
)

# 计算p值
real_effect = abs(placebo_results_sh['effects']['上海（真实）'])
all_effects = [abs(v) for v in placebo_results_sh['effects'].values()]
p_value = np.mean([e >= real_effect for e in all_effects])

print(f"真实效应: {placebo_results_sh['effects']['上海（真实）']:.2f}")
print(f"p值: {p_value:.3f}")
print(f"显著性: {'显著✅' if p_value < 0.05 else '不显著❌'}")

# 步骤5: 业务建议
print("\n" + "="*60)
print("业务建议")
print("="*60)

effect_pct = (sc_shanghai.get_effect() / gmv_df.loc[gmv_df['month'] < months[treatment_month], '上海'].mean()) * 100

print(f"1. 效应大小: 上海上线使GMV增加 {sc_shanghai.get_effect():.0f} 万元/月")
print(f"   相当于提升 {effect_pct:.1f}%")

print(f"\n2. 显著性: p={p_value:.3f}", end="")
if p_value < 0.05:
    print(" → 效应显著，建议继续推广")
else:
    print(" → 效应不显著，需要更多数据或优化策略")

print(f"\n3. ROI分析:")
假设上线成本 = 500万元
预计12个月收益 = sc_shanghai.get_effect() * 12
roi = ((预计12个月收益 - 假设上线成本) / 假设上线成本) * 100
print(f"   上线成本: {假设上线成本} 万元")
print(f"   12个月增量GMV: {预计12个月收益:.0f} 万元")
print(f"   ROI: {roi:.1f}%")

print(f"\n4. 推广建议:")
if roi > 100:
    print("   ✅ ROI良好，建议快速推广到其他城市")
    print("   ✅ 优先推广到与上海相似的城市（高权重城市）")
else:
    print("   ⚠️ ROI较低，建议优化功能后再推广")

print("="*60)
```

---

# 3. RDD - 断点回归

## 从零实现：完整的RDD估计器

```python
class MyRDDFromScratch:
    """
    从零实现断点回归设计

    核心思想：
    1. 在门槛附近，个体是可比的（局部随机化）
    2. 估计门槛两侧的条件期望差异
    3. 用局部多项式回归拟合
    """

    def __init__(self, cutoff, bandwidth=None, order=1):
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.order = order
        self.tau = None
        self.se = None

    def _triangular_kernel(self, x):
        """三角核函数"""
        u = x / self.bandwidth
        return np.maximum(0, 1 - np.abs(u))

    def _select_bandwidth(self, X, Y):
        """
        简化的带宽选择（IK方法的简化版）
        """
        n = len(X)
        # 经验公式：h ∝ n^(-1/5)
        range_x = np.max(X) - np.min(X)
        h = 1.84 * np.std(Y) * (n ** (-1/5)) * range_x / 10
        return h

    def fit(self, X, Y):
        """
        拟合RDD模型

        步骤：
        1. 选择带宽（如果未指定）
        2. 筛选带宽内的样本
        3. 构建多项式特征
        4. 加权最小二乘
        """
        X = np.array(X)
        Y = np.array(Y)

        # 处理状态
        D = (X >= self.cutoff).astype(int)

        # 自动选择带宽
        if self.bandwidth is None:
            self.bandwidth = self._select_bandwidth(X, Y)
            print(f"自动选择带宽: {self.bandwidth:.2f}")

        # 筛选带宽内样本
        mask = np.abs(X - self.cutoff) <= self.bandwidth
        X_bw = X[mask]
        Y_bw = Y[mask]
        D_bw = D[mask]

        # 核权重
        weights = self._triangular_kernel(X_bw - self.cutoff)

        # 中心化
        X_centered = X_bw - self.cutoff

        # 构建特征矩阵: [1, D, X-c, D*(X-c), (X-c)^2, D*(X-c)^2, ...]
        features = [np.ones(len(X_bw)), D_bw]

        for p in range(1, self.order + 1):
            features.append(X_centered ** p)
            features.append(D_bw * (X_centered ** p))

        Z = np.column_stack(features)

        # 加权最小二乘
        W = np.diag(weights)

        # beta = (Z'WZ)^(-1) Z'WY
        ZWZ = Z.T @ W @ Z
        ZWY = Z.T @ W @ Y_bw

        beta = np.linalg.solve(ZWZ, ZWY)

        # 提取处理效应（beta[1]）
        self.tau = beta[1]

        # 标准误（异方差稳健）
        residuals = Y_bw - Z @ beta
        meat = Z.T @ W @ np.diag(residuals**2) @ W @ Z
        bread = np.linalg.inv(ZWZ)
        vcov = bread @ meat @ bread
        self.se = np.sqrt(vcov[1, 1])

        return self

    def summary(self):
        """输出结果"""
        from scipy import stats

        t_stat = self.tau / self.se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        ci = [self.tau - 1.96*self.se, self.tau + 1.96*self.se]

        print("\n" + "="*60)
        print("RDD估计结果（从零实现）")
        print("="*60)
        print(f"带宽: {self.bandwidth:.2f}")
        print(f"多项式阶数: {self.order}")
        print(f"处理效应 (τ): {self.tau:.4f}")
        print(f"标准误 (SE): {self.se:.4f}")
        print(f"t统计量: {t_stat:.4f}")
        print(f"p值: {p_value:.4f}")
        print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print("="*60)

        return {
            'tau': self.tau,
            'se': self.se,
            'p_value': p_value,
            'ci': ci
        }


# 数学推导展示
print("="*60)
print("RDD的数学推导")
print("="*60)

print("\n1. 因果效应定义:")
print("   τ = lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]")

print("\n2. 局部多项式近似:")
print("   E[Y|X=x, X<c] = α₀ + β₁(x-c) + β₂(x-c)² + ...")
print("   E[Y|X=x, X≥c] = α₀ + τ + β₃(x-c) + β₄(x-c)² + ...")

print("\n3. 回归模型:")
print("   Y = α + τ·D + Σ[βₚ·(X-c)^p + γₚ·D·(X-c)^p] + ε")
print("   其中 D = 1{X≥c}")

print("\n4. 加权最小二乘:")
print("   最小化 Σ Kₕ(Xᵢ-c)·[Yᵢ - Z'β]²")
print("   其中 Kₕ(u) = K(u/h)·1{|u|≤h}")

print("\n5. 估计量:")
print("   β̂ = (Z'WZ)⁻¹ Z'WY")
print("   τ̂ = β̂[1] (第二个系数)")

print("\n6. 标准误:")
print("   SE(τ̂) = √[Var(τ̂)]")
print("   使用异方差稳健的方差估计（HC0/HC1）")
print("="*60)
```

## TODO 1: 带宽敏感性分析（完整答案）

```python
# TODO 1: 带宽敏感性分析 - 完整实现

bandwidths = np.linspace(10, 100, 20)
tau_estimates = []
se_estimates = []
ci_lower_list = []
ci_upper_list = []

for h in bandwidths:
    # 拟合RDD模型
    rdd = SharpRDD(cutoff=200, bandwidth=h, polynomial_order=1)
    rdd.fit(df['spending'], df['repurchase_rate'])

    # 保存结果
    tau_estimates.append(rdd.tau_)
    se_estimates.append(rdd.se_)
    ci_lower_list.append(rdd.tau_ - 1.96 * rdd.se_)
    ci_upper_list.append(rdd.tau_ + 1.96 * rdd.se_)

# 可视化
fig = go.Figure()

# 点估计
fig.add_trace(go.Scatter(
    x=bandwidths,
    y=tau_estimates,
    mode='lines+markers',
    name='点估计',
    line=dict(color='#2D9CDB', width=2),
    marker=dict(size=8)
))

# 置信区间（阴影）
fig.add_trace(go.Scatter(
    x=bandwidths,
    y=ci_upper_list,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=bandwidths,
    y=ci_lower_list,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(45, 156, 219, 0.2)',
    name='95% CI',
    hoverinfo='skip'
))

# 真实值参考线
fig.add_hline(y=15, line_dash="dash", line_color='#EB5757',
              annotation_text="真实效应 = 15%", annotation_position="right")

# 最优带宽参考线（假设是50）
optimal_h = 50
fig.add_vline(x=optimal_h, line_dash="dot", line_color='#27AE60',
              annotation_text=f"建议带宽 ≈ {optimal_h}", annotation_position="top")

fig.update_layout(
    title='带宽敏感性分析：不同带宽下的RDD估计',
    xaxis_title='带宽 (h)',
    yaxis_title='估计的处理效应 (%)',
    template='plotly_white',
    height=500,
    hovermode='x unified'
)

fig.show()

# 统计分析
print("\n" + "="*60)
print("带宽敏感性分析结果")
print("="*60)

sensitivity_df = pd.DataFrame({
    '带宽': bandwidths,
    '估计值': tau_estimates,
    '标准误': se_estimates,
    'CI下限': ci_lower_list,
    'CI上限': ci_upper_list
})

# 筛选关键带宽
key_bandwidths = [20, 40, 60, 80, 100]
print(sensitivity_df[sensitivity_df['带宽'].isin(key_bandwidths)].to_string(index=False))

print("\n关键观察：")
print(f"1. 小带宽（10-30）: 估计不稳定，置信区间宽")
print(f"   → 样本少，方差大")

print(f"\n2. 中等带宽（40-60）: 估计稳定，接近真实值")
print(f"   → 偏差-方差平衡良好")

print(f"\n3. 大带宽（70-100）: 估计可能有偏")
print(f"   → 包含了远离门槛的样本，线性近似不准")

print(f"\n4. 建议：使用数据驱动的带宽选择方法（IK或CCT）")
print("="*60)
```

## TODO 2: CCT带宽选择（完整答案）

```python
# TODO 2: CCT 带宽选择（简化版实现）

def cct_bandwidth(X, Y, cutoff, kernel='triangular'):
    """
    Calonico-Cattaneo-Titiunik (2014) 带宽选择

    简化实现：
    1. 估计左右两侧的条件方差
    2. 估计二阶导数（用三阶多项式）
    3. 计算MSE-optimal带宽

    公式: h_opt = C * (σ² / m²)^(1/5) * n^(-1/5)
    其中:
    - σ²: 残差方差
    - m: 二阶导数
    - C: 核函数常数（triangular ≈ 3.56）
    """
    X = np.array(X)
    Y = np.array(Y)

    # 分左右两侧
    left_mask = X < cutoff
    right_mask = X >= cutoff

    X_left = X[left_mask]
    Y_left = Y[left_mask]
    X_right = X[right_mask]
    Y_right = Y[right_mask]

    # 步骤1: 估计条件方差（用三阶多项式拟合）
    from sklearn.preprocessing import PolynomialFeatures

    def fit_polynomial(X_side, Y_side, degree=3):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_side.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, Y_side)
        Y_pred = model.predict(X_poly)
        residuals = Y_side - Y_pred
        variance = np.var(residuals)
        return model, variance

    model_left, var_left = fit_polynomial(X_left - cutoff, Y_left)
    model_right, var_right = fit_polynomial(X_right - cutoff, Y_right)

    # 平均方差
    sigma_sq = (var_left + var_right) / 2

    # 步骤2: 估计二阶导数
    # 用三阶多项式的二阶系数
    # f(x) = β₀ + β₁x + β₂x² + β₃x³
    # f''(x) = 2β₂ + 6β₃x
    # 在x=0处: f''(0) = 2β₂

    coef_left = model_left.coef_
    coef_right = model_right.coef_

    # 多项式系数顺序: [1, x, x², x³]
    # 所以 β₂ = coef[2]
    second_deriv_left = 2 * coef_left[2] if len(coef_left) > 2 else 0
    second_deriv_right = 2 * coef_right[2] if len(coef_right) > 2 else 0

    # 平均二阶导数
    m = (abs(second_deriv_left) + abs(second_deriv_right)) / 2

    # 避免除以0
    if m < 1e-6:
        m = 1e-6

    # 步骤3: 计算最优带宽
    n = len(X)

    # 核函数常数（triangular kernel）
    C_kernel = 3.56

    # CCT公式
    h_opt = C_kernel * ((sigma_sq / m**2) ** (1/5)) * (n ** (-1/5))

    # 调整到合理范围
    x_range = np.max(X) - np.min(X)
    h_opt = np.clip(h_opt, x_range * 0.05, x_range * 0.3)

    return h_opt

# 测试CCT带宽选择
h_cct = cct_bandwidth(df['spending'].values, df['repurchase_rate'].values, cutoff=200)

print("="*60)
print("CCT 带宽选择结果")
print("="*60)
print(f"数据驱动的最优带宽: h = {h_cct:.2f}")

# 与不同经验值对比
print("\n带宽对比：")
print(f"  IK方法（简化）: {h_cct * 0.8:.2f}")
print(f"  CCT方法: {h_cct:.2f}")
print(f"  经验法则（0.5倍range）: {(df['spending'].max() - df['spending'].min()) * 0.5 / 2:.2f}")

# 用CCT带宽重新估计
rdd_cct = SharpRDD(cutoff=200, bandwidth=h_cct, polynomial_order=1)
rdd_cct.fit(df['spending'], df['repurchase_rate'])
results_cct = rdd_cct.summary()

print(f"\n使用CCT带宽的RDD估计: {results_cct['tau']:.2f}%")
print("="*60)
```

## TODO 3: Placebo检验 - 伪门槛（完整答案）

```python
# TODO 3: Placebo 检验 - 伪门槛

# 真实门槛: 200
# 伪门槛: 150, 170, 230, 250

placebo_cutoffs = [150, 170, 230, 250]
placebo_results = []

for cutoff_placebo in placebo_cutoffs:
    # 对每个伪门槛进行RDD估计
    rdd_placebo = SharpRDD(cutoff=cutoff_placebo, bandwidth=40, polynomial_order=1)
    rdd_placebo.fit(df['spending'], df['repurchase_rate'])

    # 计算置信区间
    ci_lower = rdd_placebo.tau_ - 1.96 * rdd_placebo.se_
    ci_upper = rdd_placebo.tau_ + 1.96 * rdd_placebo.se_

    # t统计量和p值
    from scipy import stats
    t_stat = rdd_placebo.tau_ / rdd_placebo.se_
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    # 是否显著
    significant = "是❌" if p_value < 0.05 else "否✅"

    placebo_results.append({
        'Cutoff': cutoff_placebo,
        'Estimate': rdd_placebo.tau_,
        'SE': rdd_placebo.se_,
        'p_value': p_value,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'Significant': significant
    })

# 转为DataFrame
placebo_df = pd.DataFrame(placebo_results)

# 可视化Placebo结果
fig = go.Figure()

# 伪门槛的点估计和误差棒
fig.add_trace(go.Scatter(
    x=placebo_df['Cutoff'],
    y=placebo_df['Estimate'],
    mode='markers',
    marker=dict(size=12, color='#F2994A'),
    name='伪效应',
    error_y=dict(
        type='data',
        symmetric=False,
        array=placebo_df['CI_upper'] - placebo_df['Estimate'],
        arrayminus=placebo_df['Estimate'] - placebo_df['CI_lower'],
        thickness=2,
        width=5
    )
))

# 真实门槛
fig.add_vline(x=200, line_dash="dot", line_color='#EB5757',
              annotation_text="真实门槛 (200)", annotation_position="top")

# 零线
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

# 标注显著性
for _, row in placebo_df.iterrows():
    if row['Significant'].startswith('是'):
        fig.add_annotation(
            x=row['Cutoff'],
            y=row['Estimate'],
            text="⚠️",
            showarrow=False,
            yshift=20
        )

fig.update_layout(
    title='Placebo 检验: 伪门槛处不应有显著效应',
    xaxis_title='门槛位置',
    yaxis_title='估计效应 (%)',
    template='plotly_white',
    height=500
)

fig.show()

# 打印结果表格
print("\n" + "="*60)
print("Placebo 检验结果（伪门槛）")
print("="*60)
print(placebo_df[['Cutoff', 'Estimate', 'SE', 'p_value', 'Significant']].to_string(index=False))

print("\n" + "="*60)
print("解读：")
print("="*60)
print("✅ 合格的Placebo检验：伪门槛处估计效应不显著（p > 0.05）")
print("❌ 不合格：伪门槛处也检测到显著效应 → 可能违反RDD假设")

# 统计显著性个数
n_significant = placebo_df['Significant'].str.startswith('是').sum()
print(f"\n显著的伪门槛数量: {n_significant}/{len(placebo_cutoffs)}")

if n_significant == 0:
    print("✅ 所有伪门槛都不显著，Placebo检验通过！")
    print("   这增强了我们对真实门槛效应的信心。")
else:
    print("⚠️ 部分伪门槛显著，需要谨慎解读RDD结果。")
    print("   可能的原因：")
    print("   1. 非线性关系在伪门槛处也有跳跃")
    print("   2. 样本量不足导致的假阳性")
    print("   3. RDD假设（连续性）可能违反")

print("="*60)
```

---

# 4. IV - 工具变量

## 从零实现：完整的2SLS估计器

```python
class My2SLSFromScratch:
    """
    从零实现两阶段最小二乘（2SLS）

    数学推导：
    1. 第一阶段: X = π₀ + π₁Z + ν
       得到: X̂ = π̂₀ + π̂₁Z

    2. 第二阶段: Y = β₀ + β₁X̂ + ε

    3. Wald估计量: β₁ = Cov(Z,Y) / Cov(Z,X)
    """

    def __init__(self):
        self.first_stage_model = None
        self.second_stage_model = None
        self.beta_2sls = None
        self.se = None

    def fit(self, Z, X, Y):
        """
        拟合2SLS模型
        """
        Z = np.array(Z).reshape(-1, 1)
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y)

        n = len(Y)

        # 第一阶段：X ~ Z
        # X = Z @ π + ν
        Z_design = np.column_stack([np.ones(n), Z])

        # OLS: π̂ = (Z'Z)^(-1) Z'X
        pi_hat = np.linalg.solve(Z_design.T @ Z_design, Z_design.T @ X)

        # 预测值
        X_hat = Z_design @ pi_hat

        # 第一阶段F统计量
        residuals_1st = X - X_hat
        SSR_1st = np.sum(residuals_1st ** 2)
        SST_1st = np.sum((X - X.mean()) ** 2)
        R2_1st = 1 - SSR_1st / SST_1st

        k = 1  # 工具变量个数
        F_stat = (R2_1st / k) / ((1 - R2_1st) / (n - k - 1))

        # 第二阶段：Y ~ X̂
        X_hat_design = np.column_stack([np.ones(n), X_hat])

        # OLS: β̂ = (X̂'X̂)^(-1) X̂'Y
        beta_hat = np.linalg.solve(X_hat_design.T @ X_hat_design, X_hat_design.T @ Y)

        self.beta_2sls = beta_hat[1]

        # Wald估计（等价）
        cov_matrix = np.cov(Z.flatten(), Y)
        cov_zy = cov_matrix[0, 1]

        cov_matrix_zx = np.cov(Z.flatten(), X.flatten())
        cov_zx = cov_matrix_zx[0, 1]

        beta_wald = cov_zy / cov_zx

        # 标准误（异方差稳健）
        residuals_2nd = Y - X_hat_design @ beta_hat

        # 2SLS的方差估计比较复杂，这里用简化版
        # Var(β̂) ≈ σ²/(n·Cov(Z,X)²)
        sigma_sq = np.var(residuals_2nd)
        self.se = np.sqrt(sigma_sq / (n * cov_zx ** 2))

        # 保存结果
        self.first_stage_f = F_stat
        self.first_stage_coef = pi_hat[1]
        self.X_hat = X_hat

        return self

    def summary(self):
        """输出结果"""
        from scipy import stats

        t_stat = self.beta_2sls / self.se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        ci = [self.beta_2sls - 1.96*self.se, self.beta_2sls + 1.96*self.se]

        print("\n" + "="*60)
        print("2SLS估计结果（从零实现）")
        print("="*60)
        print("第一阶段诊断：")
        print(f"  F统计量: {self.first_stage_f:.2f}")
        if self.first_stage_f > 10:
            print("  ✅ 工具变量强度充足（F > 10）")
        else:
            print("  ❌ 弱工具变量问题（F < 10）")

        print(f"  第一阶段系数: {self.first_stage_coef:.4f}")

        print("\n第二阶段估计：")
        print(f"  处理效应 (β): {self.beta_2sls:.4f}")
        print(f"  标准误 (SE): {self.se:.4f}")
        print(f"  t统计量: {t_stat:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print("="*60)

        return {
            'beta': self.beta_2sls,
            'se': self.se,
            'first_stage_f': self.first_stage_f,
            'p_value': p_value
        }


# 数学推导展示
print("="*60)
print("2SLS的数学推导")
print("="*60)

print("\n1. 结构方程:")
print("   Y = β₀ + β₁X + ε")
print("   其中 Cov(X, ε) ≠ 0 (内生性)")

print("\n2. 工具变量Z的条件:")
print("   (1) 相关性: Cov(Z, X) ≠ 0")
print("   (2) 排他性: Z只通过X影响Y")
print("   (3) 外生性: Cov(Z, ε) = 0")

print("\n3. 第一阶段:")
print("   X = π₀ + π₁Z + ν")
print("   X̂ = π̂₀ + π̂₁Z")

print("\n4. 第二阶段:")
print("   Y = β₀ + β₁X̂ + u")

print("\n5. Wald估计量:")
print("   β̂₁ = Cov(Z,Y) / Cov(Z,X)")
print("   = [E(ZY) - E(Z)E(Y)] / [E(ZX) - E(Z)E(X)]")

print("\n6. 渐近分布:")
print("   √n(β̂₁ - β₁) →ᵈ N(0, V)")
print("   其中 V = σ²/[n·Cov(Z,X)²]")

print("\n7. 弱IV问题:")
print("   如果 Cov(Z,X) → 0，则 V → ∞")
print("   导致估计量方差爆炸！")
print("="*60)
```

## TODO 1: 模拟好的工具变量场景（完整答案）

```python
def simulate_good_iv(n=1000):
    """
    模拟一个满足IV假设的场景

    因果图:
    Z (成本冲击) → X (价格) → Y (销量)
                      ↑          ↑
                      |          |
                      +--- U ----+
                    (需求冲击)
    """
    # 不可观测的需求冲击
    demand_shock = np.random.normal(0, 10, n)

    # 创建一个外生的成本冲击Z（与需求冲击无关）
    cost_shock = np.random.normal(0, 5, n)

    # 价格受成本冲击和需求冲击影响
    # 公式: P = 10 + 0.5*Z + 0.2*U + noise
    price = 10 + 0.5 * cost_shock + 0.2 * demand_shock + np.random.normal(0, 1, n)

    # 销量只受价格和需求冲击影响（成本不直接影响销量）
    # 公式: Q = 100 - 2*P + U + noise
    # 真实价格弹性 = -2
    quantity = 100 - 2 * price + demand_shock + np.random.normal(0, 2, n)

    return pd.DataFrame({
        'cost_shock': cost_shock,        # Z: 工具变量
        'price': price,                  # X: 内生处理变量
        'quantity': quantity,            # Y: 结果变量
        'demand_shock': demand_shock     # U: 不可观测混淆
    })

# 生成数据
df_iv = simulate_good_iv()

# 验证IV假设
print("="*60)
print("检验工具变量的三个假设")
print("="*60)

# 假设1: 相关性
corr_zx = df_iv['cost_shock'].corr(df_iv['price'])
print(f"\n1. 相关性: Corr(Z, X) = {corr_zx:.3f}")
if abs(corr_zx) > 0.3:
    print("   ✅ 满足！工具变量与处理变量强相关")
    print("   解释：成本冲击确实影响价格")
else:
    print("   ❌ 不满足！弱工具变量")

# 假设2: 排他性（间接检验）
print(f"\n2. 排他性: Z只通过X影响Y")
print("   ⚠️ 无法直接检验，需要理论支撑")
print("   本例中：成本只影响价格，不直接影响需求")
print("   逻辑检验：✅ 合理")

# 假设3: 外生性
corr_zu = df_iv['cost_shock'].corr(df_iv['demand_shock'])
print(f"\n3. 外生性: Corr(Z, U) = {corr_zu:.3f}")
if abs(corr_zu) < 0.1:
    print("   ✅ 满足！工具变量与混淆因素基本无关")
    print("   解释：成本冲击独立于需求冲击")
else:
    print("   ❌ 不满足！工具变量可能不外生")

print("="*60)

# OLS vs 2SLS对比
ols = LinearRegression().fit(df_iv[['price']], df_iv['quantity'])
beta_ols = ols.coef_[0]

tsls = My2SLSFromScratch()
tsls.fit(df_iv['cost_shock'], df_iv['price'], df_iv['quantity'])
results_tsls = tsls.summary()

print("\n估计结果对比：")
print(f"真实弹性: -2.00")
print(f"OLS估计: {beta_ols:.2f} (有偏)")
print(f"2SLS估计: {results_tsls['beta']:.2f} (无偏)")
```

## TODO 2: 手动实现2SLS估计（完整答案）

```python
def two_stage_least_squares(Z, X, Y):
    """
    手动实现2SLS（步骤展示版）

    参数:
        Z: 工具变量 (n,)
        X: 内生处理变量 (n,)
        Y: 结果变量 (n,)

    返回:
        完整的估计结果
    """
    # 转换为numpy数组
    Z = np.array(Z).reshape(-1, 1)
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    print("="*60)
    print("2SLS估计过程（步骤展示）")
    print("="*60)

    # 第一阶段：回归X ~ Z
    print("\n步骤1: 第一阶段回归 X ~ Z")

    first_stage = LinearRegression()
    first_stage.fit(Z, X)

    print(f"  模型: X = {first_stage.intercept_[0]:.4f} + {first_stage.coef_[0][0]:.4f}*Z")

    # 获取预测值X_hat
    X_hat = first_stage.predict(Z)

    print(f"  预测的X范围: [{X_hat.min():.2f}, {X_hat.max():.2f}]")

    # 检验第一阶段强度
    from sklearn.metrics import r2_score
    r2_first = r2_score(X, X_hat)

    n = len(Y)
    k = 1
    F_stat = (r2_first / k) / ((1 - r2_first) / (n - k - 1))

    print(f"  R²: {r2_first:.4f}")
    print(f"  F统计量: {F_stat:.2f}")

    if F_stat > 10:
        print("  ✅ 工具变量强度充足")
    else:
        print("  ⚠️ 可能存在弱工具变量问题")

    # 第二阶段：回归Y ~ X_hat
    print("\n步骤2: 第二阶段回归 Y ~ X̂")

    second_stage = LinearRegression()
    second_stage.fit(X_hat.reshape(-1, 1), Y)

    beta_2sls = second_stage.coef_[0]

    print(f"  模型: Y = {second_stage.intercept_:.4f} + {beta_2sls:.4f}*X̂")

    # Wald估计（验证）
    print("\n步骤3: Wald估计（验证）")

    cov_zy = np.cov(Z.flatten(), Y)[0, 1]
    cov_zx = np.cov(Z.flatten(), X.flatten())[0, 1]
    beta_wald = cov_zy / cov_zx

    print(f"  Cov(Z, Y) = {cov_zy:.4f}")
    print(f"  Cov(Z, X) = {cov_zx:.4f}")
    print(f"  β_Wald = Cov(Z,Y) / Cov(Z,X) = {beta_wald:.4f}")

    print(f"\n验证: 2SLS = Wald? {np.isclose(beta_2sls, beta_wald)}")
    print(f"  2SLS: {beta_2sls:.6f}")
    print(f"  Wald: {beta_wald:.6f}")
    print(f"  差异: {abs(beta_2sls - beta_wald):.6f}")

    print("="*60)

    return {
        'beta_2sls': beta_2sls,
        'beta_wald': beta_wald,
        'first_stage': first_stage,
        'second_stage': second_stage,
        'X_hat': X_hat,
        'F_stat': F_stat
    }

# 应用2SLS
results = two_stage_least_squares(
    df_iv['cost_shock'].values,
    df_iv['price'].values,
    df_iv['quantity'].values
)

print(f"\n最终结果:")
print(f"真实因果效应: -2.0")
print(f"2SLS估计: {results['beta_2sls']:.3f}")
print(f"估计误差: {abs(results['beta_2sls'] + 2):.3f}")
```

## TODO 3: Hansen J检验（完整答案）

```python
def hansen_j_test(Z_list, X, Y):
    """
    Hansen J过度识别检验

    检验多个工具变量是否都有效

    H₀: 所有工具变量都有效
    H₁: 至少有一个工具变量无效

    统计量: J = n·R² ~ χ²(m-k)
    其中 m=工具变量个数, k=内生变量个数
    """
    n = len(Y)
    m = len(Z_list)  # 工具变量个数
    k = 1  # 内生变量个数

    print("="*60)
    print("Hansen J 过度识别检验")
    print("="*60)

    # 构造工具变量矩阵Z
    Z = np.column_stack(Z_list)

    print(f"\n数据信息:")
    print(f"  样本量: n = {n}")
    print(f"  工具变量个数: m = {m}")
    print(f"  内生变量个数: k = {k}")
    print(f"  过度识别: m - k = {m - k}")

    # 第一阶段：回归X ~ Z
    print(f"\n步骤1: 第一阶段回归 X ~ Z₁ + Z₂ + ... + Z{m}")

    first_stage = LinearRegression()
    first_stage.fit(Z, X)
    X_hat = first_stage.predict(Z)

    # 第二阶段：回归Y ~ X_hat
    print(f"\n步骤2: 第二阶段回归 Y ~ X̂")

    second_stage = LinearRegression()
    second_stage.fit(X_hat.reshape(-1, 1), Y)

    # 计算2SLS残差
    residuals = Y - second_stage.predict(X_hat.reshape(-1, 1))

    print(f"\n步骤3: 回归残差对所有工具变量")
    print(f"  模型: ε̂ ~ Z₁ + Z₂ + ... + Z{m}")

    # 回归残差对所有IV
    residual_model = LinearRegression()
    residual_model.fit(Z, residuals)

    # 计算R²
    y_pred = residual_model.predict(Z)
    ss_res = np.sum((residuals - y_pred) ** 2)
    ss_tot = np.sum((residuals - residuals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"  R² = {r2:.6f}")

    # J统计量
    j_stat = n * r2

    # 自由度
    df = m - k

    # p值
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(j_stat, df)

    print(f"\n步骤4: 计算J统计量")
    print(f"  J = n × R² = {n} × {r2:.6f} = {j_stat:.4f}")
    print(f"  自由度 = m - k = {df}")
    print(f"  p值 = {p_value:.4f}")

    print("\n" + "="*60)
    print("检验结果:")
    print("="*60)

    if p_value > 0.05:
        conclusion = "无法拒绝原假设 → 所有工具变量可能都有效✅"
    else:
        conclusion = "拒绝原假设 → 至少有一个工具变量无效❌"

    print(f"  {conclusion}")

    print("\n注意事项:")
    print("  1. J检验只能检测'至少有一个IV无效'")
    print("  2. 无法告诉你'哪个IV有问题'")
    print("  3. 需要至少一个IV是有效的（无法检验全部无效）")
    print("  4. 检验力取决于样本量和IV质量")
    print("="*60)

    return {
        'J统计量': j_stat,
        '自由度': df,
        'p值': p_value,
        'R²': r2,
        '结论': conclusion
    }

# 模拟多个工具变量的场景
def simulate_multiple_iv(n=1000):
    demand_shock = np.random.normal(0, 10, n)

    # 两个有效的工具变量（都外生）
    cost_shock_1 = np.random.normal(0, 5, n)  # 原材料成本
    cost_shock_2 = np.random.normal(0, 5, n)  # 运输成本

    # 价格受两个成本冲击影响
    price = 10 + 0.5 * cost_shock_1 + 0.3 * cost_shock_2 + 0.2 * demand_shock + np.random.normal(0, 1, n)

    # 销量
    quantity = 100 - 2 * price + demand_shock + np.random.normal(0, 2, n)

    return cost_shock_1, cost_shock_2, price, quantity

Z1, Z2, X_multi, Y_multi = simulate_multiple_iv()

# 执行J检验
j_results = hansen_j_test([Z1, Z2], X_multi, Y_multi)

# 对比：如果加入一个坏的IV会怎样？
print("\n\n" + "="*60)
print("对比实验：加入一个无效的工具变量")
print("="*60)

# 创建一个坏的IV（与需求冲击相关）
demand_shock_multi = np.random.normal(0, 10, len(X_multi))
Z3_bad = 0.8 * demand_shock_multi + np.random.normal(0, 2, len(X_multi))  # 与混淆因素相关！

print("\n坏IV的特征：")
print(f"  与X的相关性: {np.corrcoef(Z3_bad, X_multi)[0,1]:.3f}")
print(f"  与混淆的相关性: {np.corrcoef(Z3_bad, demand_shock_multi)[0,1]:.3f} ❌")

j_results_bad = hansen_j_test([Z1, Z2, Z3_bad], X_multi, Y_multi)
```

---

## 面试模拟题（所有方法）

### DID

**Q1: DID的平行趋势假设是什么？如何检验？**

A: 平行趋势假设指在没有处理的情况下，处理组和对照组的结果变量应该保持平行的趋势。

检验方法：
1. **事件研究法（Event Study）**：估计每个时期的处理效应，检验政策前是否显著
2. **可视化检验**：绘制处理组和对照组的趋势图
3. **安慰剂检验**：在政策前假设一个虚假的政策时点

**Q2: 从零推导DID估计量**

```
设：
- Y_{it}(0): 没有处理时的潜在结果
- Y_{it}(1): 接受处理时的潜在结果
- D_i: 是否为处理组（1=是，0=否）
- T_t: 是否为政策后（1=是，0=否）

平行趋势假设：
E[Y_{i1}(0) - Y_{i0}(0) | D_i=1] = E[Y_{i1}(0) - Y_{i0}(0) | D_i=0]

DID估计量：
τ_DID = [E(Y|D=1,T=1) - E(Y|D=1,T=0)] - [E(Y|D=0,T=1) - E(Y|D=0,T=0)]
     = (处理组的前后差异) - (对照组的前后差异)
```

### Synthetic Control

**Q1: 合成控制什么时候比DID更好？**

A:
- **单个处理单位**：DID需要多个单位，合成控制可以处理N=1的情况
- **平行趋势不满足**：合成控制通过优化权重来拟合前处理期趋势
- **没有完美对照组**：合成控制可以用多个不完美的对照组合成一个

**Q2: 合成控制的权重优化问题是什么？**

```
优化问题：
min_W ||Y_1 - Y_0 W||²

约束：
- W >= 0 (非负)
- sum(W) = 1 (和为1)

求解方法：
- 序列二次规划（SLSQP）
- 内点法
- 凸优化
```

### RDD

**Q1: RDD的识别假设是什么？**

A: **连续性假设**：在门槛处，如果没有处理，结果变量应该是连续的。

数学表达：
```
lim_{x↓c} E[Y(0)|X=x] = lim_{x↑c} E[Y(0)|X=x]
```

含义：除了处理状态，其他所有因素在门槛处都是连续的。

**Q2: Sharp RDD vs Fuzzy RDD的区别？**

| 维度 | Sharp RDD | Fuzzy RDD |
|------|-----------|-----------|
| 处理分配 | 完全由门槛决定 | 门槛影响处理概率 |
| 跳跃 | 0→1 | 跳跃幅度<1 |
| 估计方法 | 局部多项式 | 2SLS/IV |
| 解释 | 门槛处的ATE | LATE（顺从者效应） |

### IV

**Q1: 工具变量的三个假设，哪个最难满足？**

A: **排他性假设**最难满足且无法检验。

- 相关性：可以用F统计量检验✅
- 外生性：不可直接检验，但可以用制度背景论证
- 排他性：完全不可检验❌，需要强有力的经济学逻辑

**Q2: 弱工具变量为什么比没有工具变量更糟？**

A: 当F统计量<10时：
1. **有限样本偏差**：2SLS估计量向OLS方向偏
2. **推断失效**：标准误被严重低估
3. **放大内生性**：即使Z与ε只有微小相关，也会导致大偏差

数学直觉：
```
β̂_2SLS = Cov(Z,Y) / Cov(Z,X)

如果Cov(Z,X)→0（弱IV），分母接近0，估计量方差爆炸！
```

---

## 总结：方法选择流程图

```
开始：我想估计处理效应
    |
    v
有随机分配吗？
    |
    |--是--> RCT（最优）
    |
    |--否--> 有门槛/分数线吗？
              |
              |--是--> 门槛完全决定处理？
              |         |
              |         |--是--> Sharp RDD
              |         |--否--> Fuzzy RDD (用IV)
              |
              |--否--> 有时间维度吗？
                        |
                        |--是--> 多个处理单位？
                        |         |
                        |         |--是--> DID
                        |         |--否--> 单个处理单位？
                        |                   |
                        |                   |--是--> Synthetic Control
                        |
                        |--否--> 有外生冲击吗？
                                  |
                                  |--是--> IV/2SLS
                                  |--否--> Matching/PSM
```
