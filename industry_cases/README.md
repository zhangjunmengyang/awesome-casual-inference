# IndustryCases - 行业真实案例模块

展示 DoorDash、Netflix、Uber 等科技公司如何在真实业务场景中应用因果推断。

## 模块结构

```
industry_cases/
├── __init__.py                    # 模块导出
├── utils.py                       # 工具函数和数据生成器
├── doordash_delivery.py           # DoorDash 配送优化案例
├── netflix_recommendation.py      # Netflix 推荐系统案例
└── uber_surge_pricing.py          # Uber 动态定价案例
```

## 案例说明

### 1. DoorDash 配送优化 (doordash_delivery.py)

**业务场景**: 配送时间预估与智能调度算法评估

**因果挑战**:
- 新算法非随机部署，优先在系统负载低时使用
- 天气、时段同时影响算法使用和配送时间
- 简单对比会高估算法效果

**方法**:
- 倾向得分匹配 (PSM): 匹配相似特征的样本
- 双重稳健 (DR): 结合倾向得分和结果模型
- 重叠检查: 验证共同支撑假设

**关键指标**:
- 配送时间 (主要结果)
- 客户满意度
- 准时率

**数据生成器**: `generate_doordash_delivery_data()`
- 8000 订单样本
- 混淆因素: 天气、时段、距离、餐厅类型
- 真实效应: 新算法平均减少 3-5 分钟配送时间

### 2. Netflix 推荐系统 (netflix_recommendation.py)

**业务场景**: 新推荐算法对用户留存和观看时长的影响

**因果挑战**:
- 高活跃用户优先分到新算法 (渐进式发布)
- 不同用户群效果差异大 (异质性)
- 需要识别哪些用户真正受益

**方法**:
- 因果森林 (Causal Forest): 估计个体效应 (CATE)
- 用户分群: 数据驱动识别高/中/低效应用户
- 异质性分析: 分析特征与效应的关系

**关键指标**:
- 30 天留存率 (主要结果)
- 观看时长
- 用户生命周期价值 (LTV)

**数据生成器**: `generate_netflix_recommendation_data()`
- 10000 用户样本
- 混淆因素: 用户活跃度、观看历史、内容偏好
- 异质性效应: 低活跃用户受益更多

**关键洞察**:
- 月观看 < 20 小时的用户获益最大
- 内容偏好单一的用户，新推荐能拓展兴趣
- 年轻用户 (< 25 岁) 更喜欢新算法

### 3. Uber 动态定价 (uber_surge_pricing.py)

**业务场景**: Surge Pricing 对供需平衡的影响

**因果挑战**:
- 需求高峰时启动 Surge，同时高峰本身影响匹配率
- 定价是基于需求/供给比 (D/S Ratio) 的阈值规则
- 需要识别不同 Surge 倍数的效果

**方法**:
- 回归断点设计 (RDD): 利用 D/S Ratio 阈值的"准实验"
- 逆概率加权 (IPW): 调整混淆因素
- 弹性分析: 供给/需求对价格的响应

**关键指标**:
- 匹配率 (主要结果)
- 等待时间
- 司机供给 / 乘客需求
- 平台收入

**数据生成器**: `generate_uber_surge_pricing_data()`
- 12000 订单请求样本
- 混淆因素: 时段、天气、地区、事件
- Surge 倍数: 1.0, 1.5, 2.0, 2.5

**关键洞察**:
- 供给弹性 > 0: Surge 提高，司机供给增加
- 需求弹性 < 0: Surge 提高，乘客需求下降
- 最优 Surge: 1.5-2.0x 之间平衡供需

## 工具函数 (utils.py)

### 数据生成器

1. **generate_doordash_delivery_data(n_samples, seed)**
   - 生成 DoorDash 配送数据
   - 返回: (df, true_effect)

2. **generate_netflix_recommendation_data(n_samples, seed)**
   - 生成 Netflix 推荐数据
   - 返回: (df, true_effect)

3. **generate_uber_surge_pricing_data(n_samples, seed)**
   - 生成 Uber 定价数据
   - 返回: (df, true_effect)

### 可视化工具

4. **plot_causal_dag(case_name)**
   - 绘制因果图 (DAG)
   - case_name: 'doordash', 'netflix', 'uber'

5. **compute_ate_with_ci(Y, T, alpha)**
   - 计算 ATE 及置信区间
   - 返回: (ate, ci_lower, ci_upper)

## 使用示例

```python
from industry_cases import doordash_delivery, netflix_recommendation, uber_surge_pricing
from industry_cases.utils import generate_doordash_delivery_data

# 生成 DoorDash 数据
df, true_effect = generate_doordash_delivery_data(n_samples=5000)

# 运行分析
fig, summary = doordash_delivery.analyze_doordash_delivery(
    n_samples=5000,
    method='all',
    show_confounding=True
)

# 渲染 Gradio 界面
doordash_delivery.render()
```

## 集成到主应用

已集成到 `app.py` 的 IndustryCases Tab:

```python
with gr.Tab("IndustryCases", id="industry_cases"):
    from industry_cases import (
        doordash_delivery,
        netflix_recommendation,
        uber_surge_pricing
    )

    with gr.Tabs() as industry_tabs:
        with gr.Tab("DoorDash Delivery", id="doordash"):
            doordash_delivery.render()

        with gr.Tab("Netflix Recommendation", id="netflix"):
            netflix_recommendation.render()

        with gr.Tab("Uber Surge Pricing", id="uber"):
            uber_surge_pricing.render()
```

## 技术栈

- **因果推断**: PSM, IPW, Doubly Robust, Causal Forest, RDD
- **可视化**: Plotly (交互式图表)
- **机器学习**: scikit-learn (RandomForest, GradientBoosting, LogisticRegression)
- **统计**: scipy (置信区间、线性回归)
- **UI**: Gradio (Web 界面)

## 参考资源

### DoorDash
- [DoorDash ML Platform Blog](https://doordash.engineering/category/data-science-and-machine-learning/)
- 配送时间优化实践

### Netflix
- [Netflix Tech Blog - Recommendation Systems](https://netflixtechblog.com/)
- [Causal Forests - Athey & Imbens (2019)](https://arxiv.org/abs/1902.07409)

### Uber
- [Uber Surge Pricing Explained](https://www.uber.com/blog/surge-pricing/)
- [Dynamic Pricing at Uber](https://eng.uber.com/research/)
- Lee & Lemieux (2010): Regression Discontinuity Designs in Economics

## 扩展方向

1. **更多公司案例**:
   - Airbnb 定价优化
   - Amazon 推荐系统
   - LinkedIn 职位推荐

2. **更多方法**:
   - 合成控制法 (Synthetic Control)
   - 差分法 (Difference-in-Differences)
   - 工具变量 (Instrumental Variables)

3. **实时交互**:
   - 用户自定义数据生成参数
   - A/B 测试模拟器
   - 因果图编辑器

## 联系

如有问题或建议，欢迎提 Issue!
