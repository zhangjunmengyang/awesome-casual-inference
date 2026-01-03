# Challenges - 因果推断竞赛系统

Kaggle 风格的因果推断挑战，让你在实践中学习和提升技能!

## 概述

挑战系统提供三个由易到难的因果推断任务:

| 挑战 | 难度 | 任务 | 评估指标 |
|------|------|------|---------|
| **ATE Estimation** | Beginner | 从观察数据估计平均处理效应 | Relative Error |
| **CATE Prediction** | Intermediate | 预测个体条件处理效应 | PEHE |
| **Uplift Ranking** | Advanced | 按 uplift 对用户排序 | AUUC |

## 特性

- **真实场景数据**: LaLonde 职业培训、IHDP 教育干预、营销优惠券
- **基线方法**: 提供多种基线方法对比
- **Starter Code**: 每个挑战都有代码模板
- **实时排行榜**: 与其他用户竞争
- **进度追踪**: 查看个人提交历史
- **指标说明**: 详细的评估指标解释

## 快速开始

### 1. 在 Gradio 界面中使用

```bash
python app.py
```

访问 http://localhost:7860，进入 "Challenges" 标签页。

### 2. 挑战流程

1. **初始化**: 点击 "Initialize Challenge" 生成数据
2. **探索**: 查看数据预览和挑战说明
3. **基线**: 运行基线方法了解任务
4. **编码**: 在代码编辑器中实现你的方案
5. **提交**: 输入姓名并提交预测
6. **排名**: 查看排行榜上的排名

### 3. 代码示例

#### ATE Estimation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

def estimate_ate(train_df, test_df):
    # 提取变量
    X_cols = ['age', 'education', 're74', 're75', 'black', 'hispanic', 'married']
    X = train_df[X_cols].values
    T = train_df['T'].values
    Y = train_df['Y'].values

    # IPW 估计
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    # 计算 ATE
    weights_1 = T / ps
    weights_0 = (1 - T) / (1 - ps)
    ate = (Y * weights_1).sum() / weights_1.sum() - (Y * weights_0).sum() / weights_0.sum()

    return ate

# 定义预测
predictions = estimate_ate(train_data, test_data)
```

#### CATE Prediction

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_cate(train_df, test_df):
    X_cols = [f'X{i}' for i in range(1, 11)]

    X_train = train_df[X_cols].values
    T_train = train_df['T'].values
    Y_train = train_df['Y'].values
    X_test = test_df[X_cols].values

    # T-Learner
    model_0 = RandomForestRegressor(n_estimators=100)
    model_1 = RandomForestRegressor(n_estimators=100)

    model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
    model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

    cate = model_1.predict(X_test) - model_0.predict(X_test)
    return cate

# 定义预测
predictions = predict_cate(train_data, test_data)
```

#### Uplift Ranking

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def predict_uplift(train_df, test_df):
    X_cols = [f'feature_{i}' for i in range(1, 9)]

    X_train = train_df[X_cols].values
    T_train = train_df['T'].values
    Y_train = train_df['Y'].values
    X_test = test_df[X_cols].values

    # T-Learner for classification
    model_0 = RandomForestClassifier(n_estimators=100)
    model_1 = RandomForestClassifier(n_estimators=100)

    model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
    model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

    p0 = model_0.predict_proba(X_test)[:, 1]
    p1 = model_1.predict_proba(X_test)[:, 1]

    uplift = p1 - p0
    return uplift

# 定义预测
predictions = predict_uplift(train_data, test_data)
```

## 程序化使用

### 直接使用挑战类

```python
from challenges import ATEEstimationChallenge, Leaderboard
import numpy as np

# 创建挑战
challenge = ATEEstimationChallenge()

# 生成数据
train_data, test_data = challenge.generate_data(seed=42)

# 获取基线
baseline_ate = challenge.get_baseline_predictions('ipw')

# 提交你的预测
my_prediction = 1800.0  # 你的 ATE 估计
result = challenge.evaluate(my_prediction, user_name="YourName")

print(f"Score: {result.score:.2f}")
print(f"Relative Error: {result.primary_metric:.4f}")

# 添加到排行榜
leaderboard = Leaderboard("ATE Estimation")
leaderboard.add_submission(result)
```

## 评估指标详解

### ATE Estimation

**Primary Metric: Relative Error**

$$\text{Relative Error} = \frac{|\hat{\tau} - \tau|}{\tau}$$

其中:
- $\hat{\tau}$ 是估计的 ATE
- $\tau$ 是真实 ATE

**Scoring**:
- 0% 误差 → 100分
- 50% 误差 → 0分

**目标**: Relative Error < 10%

### CATE Prediction

**Primary Metric: PEHE (Precision in Estimation of Heterogeneous Effect)**

$$\text{PEHE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{\tau}(x_i) - \tau(x_i))^2}$$

其中:
- $\hat{\tau}(x_i)$ 是个体 i 的预测 CATE
- $\tau(x_i)$ 是个体 i 的真实 CATE

**Secondary Metrics**:
- **ATE Bias**: $|\text{mean}(\hat{\tau}) - \text{mean}(\tau)|$
- **Correlation**: Pearson 相关系数
- **R²**: 决定系数

**Scoring**:
- PEHE = 0 → 100分
- PEHE = 5 → 0分
- Bonus: Correlation > 0.8 → +10分

**目标**: PEHE < 2.0

### Uplift Ranking

**Primary Metric: AUUC (Area Under Uplift Curve)**

Qini 曲线下的面积，衡量排序质量:

$$\text{Qini}(k) = \sum_{i=1}^{k} Y_i \cdot T_i - \sum_{i=1}^{k} Y_i \cdot (1-T_i) \cdot \frac{n_{T,k}}{n_{C,k}}$$

**Secondary Metrics**:
- **Normalized AUUC**: $(AUUC - AUUC_{random}) / (AUUC_{perfect} - AUUC_{random})$
- **Kendall's Tau**: 排序一致性
- **Top-K Uplift**: Top K% 用户的平均 uplift
- **ROI**: 投资回报率

**Scoring**:
- 基于 Normalized AUUC (0-100)
- Bonus: ROI > 2.0 → +10分

**目标**: Normalized AUUC > 0.7

## 数据集说明

### ATE Estimation: LaLonde 职业培训

模拟真实的观察性研究数据:

- **处理**: 参加职业培训 (0/1)
- **结果**: 培训后年收入 (美元)
- **协变量**:
  - `age`: 年龄 (18-60)
  - `education`: 受教育年限 (6-18)
  - `re74`, `re75`: 1974/1975年收入
  - `black`, `hispanic`: 种族
  - `married`: 婚姻状况

**混淆**: 低收入、低教育背景的人更可能参加培训

### CATE Prediction: IHDP 教育干预

婴儿健康发展项目数据:

- **处理**: 早期教育干预 (0/1)
- **结果**: 认知测试得分
- **协变量**: X1-X10 (孩子和家庭特征)

**异质性**: 效应随特征有复杂的非线性变化

### Uplift Ranking: 营销优惠券

电商优惠券发放场景:

- **处理**: 收到优惠券 (0/1)
- **结果**: 是否购买/转化 (0/1)
- **协变量**: feature_1 - feature_8 (用户行为特征)

**四类用户**:
1. **Persuadables**: 发券会买 (正 uplift)
2. **Sure Things**: 本来就买 (零 uplift)
3. **Lost Causes**: 怎么都不买 (零 uplift)
4. **Sleeping Dogs**: 发券反而不买 (负 uplift)

## 提示和技巧

### ATE Estimation

1. **识别混淆**: 可视化处理组和控制组的协变量分布
2. **倾向得分**: 估计 $P(T=1|X)$ 并检查重叠性
3. **IPW**: 使用倾向得分加权
4. **双重稳健**: 结合倾向得分和结果模型
5. **交叉验证**: 避免过拟合

### CATE Prediction

1. **探索异质性**: 分析特征与效应的关系
2. **Meta-Learners**: 尝试 S/T/X-Learner
3. **非线性模型**: Random Forest, GBM 表现通常好
4. **特征工程**: 考虑交互项、多项式特征
5. **集成方法**: 多个模型加权组合

### Uplift Ranking

1. **排序优先**: 关注排序质量而非绝对值
2. **二分类模型**: 使用 RandomForestClassifier
3. **Class Transformation**: 4类转换方法
4. **业务视角**: 计算 ROI 找最优干预比例
5. **特征重要性**: 识别关键预测特征

## 排行榜系统

### 查看排名

```python
from challenges import Leaderboard

lb = Leaderboard("ATE Estimation")

# 获取排名
rankings = lb.get_rankings(top_n=10)
print(rankings)

# 可视化
fig = lb.plot_rankings(top_n=10)
fig.show()

# 查看用户进步
fig = lb.plot_user_progress("YourName")
fig.show()
```

### 统计信息

```python
stats = lb.get_statistics()
print(f"Total submissions: {stats['total_submissions']}")
print(f"Unique users: {stats['unique_users']}")
print(f"Best score: {stats['best_score']:.2f}")
```

### 导出数据

```python
# 导出为 CSV
lb.export_to_csv("leaderboard.csv")

# 获取 Markdown
md = lb.get_leaderboard_markdown(top_n=10)
print(md)
```

## 架构设计

### 模块结构

```
challenges/
├── __init__.py                    # 模块导出
├── challenge_base.py              # 挑战基类和数据生成器
├── challenge_1_ate_estimation.py  # ATE 估计挑战
├── challenge_2_cate_prediction.py # CATE 预测挑战
├── challenge_3_uplift_ranking.py  # Uplift 排序挑战
├── leaderboard.py                 # 排行榜系统
└── ui.py                          # Gradio 界面
```

### 核心类

**Challenge (基类)**
- `generate_data()`: 生成训练/测试数据
- `evaluate()`: 评估预测结果
- `get_baseline_predictions()`: 获取基线方法
- `get_starter_code()`: 返回代码模板
- `validate_predictions()`: 验证预测格式

**ChallengeResult (数据类)**
- 存储评估结果
- 包含得分、指标、时间戳等

**Leaderboard**
- 记录提交历史
- 排名和统计
- 可视化对比
- 数据持久化

## 扩展新挑战

创建新挑战只需继承 `Challenge` 基类:

```python
from challenges.challenge_base import Challenge, ChallengeResult

class MyChallenge(Challenge):
    def __init__(self):
        super().__init__(
            name="My Challenge",
            description="Challenge description",
            difficulty="intermediate"
        )

    def generate_data(self, seed=42):
        # 生成训练和测试数据
        train_data = ...
        test_data = ...
        self.true_targets = ...
        return train_data, test_data

    def evaluate(self, predictions, user_name="Anonymous"):
        # 计算指标
        primary_metric = ...
        score = ...

        return ChallengeResult(
            challenge_name=self.name,
            user_name=user_name,
            submission_time=...,
            primary_metric=primary_metric,
            secondary_metrics={...},
            score=score
        )

    def get_baseline_predictions(self, method='naive'):
        # 实现基线方法
        ...

    def get_starter_code(self):
        return "# Your starter code"
```

## 常见问题

### Q: 如何获得高分?

A:
1. 理解数据生成过程和混淆机制
2. 尝试多种方法并对比
3. 关注评估指标的定义
4. 使用交叉验证避免过拟合
5. 参考排行榜上的方法

### Q: 为什么我的代码报错?

A:
1. 检查预测格式 (numpy array)
2. 确保预测长度匹配测试集
3. 避免 NaN 和 Inf 值
4. 使用 try-except 捕获异常

### Q: 排行榜数据存在哪里?

A: 存储在 `./challenge_submissions/` 目录下的 JSON 文件中

### Q: 可以使用外部库吗?

A: 可以! 推荐使用:
- scikit-learn
- econml
- causalml
- xgboost
- lightgbm

## 参考资源

### 论文
- **ATE Estimation**: Rosenbaum & Rubin (1983) "The central role of the propensity score in observational studies"
- **Meta-Learners**: Künzel et al. (2019) "Metalearners for estimating heterogeneous treatment effects"
- **Uplift Modeling**: Radcliffe & Surry (2011) "Real-world uplift modelling with significance-based uplift trees"

### 代码库
- **EconML**: https://github.com/microsoft/EconML
- **CausalML**: https://github.com/uber/causalml
- **DoWhy**: https://github.com/py-why/dowhy

### 课程
- "Causal Inference: The Mixtape" by Scott Cunningham
- "Applied Causal Inference" by Brady Neal

## 贡献

欢迎贡献新的挑战、改进评估指标或优化代码!

## License

MIT License
