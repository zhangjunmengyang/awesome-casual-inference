# 挑战系统使用指南

## 快速开始

### 1. 启动应用

```bash
python app.py
```

访问 http://localhost:7860，点击 "Challenges" 标签页。

### 2. 三个挑战概览

| 挑战 | 难度 | 任务 | 评估指标 | 目标 |
|------|------|------|---------|------|
| **ATE Estimation** | Beginner | 从观察数据估计 ATE | Relative Error | < 10% |
| **CATE Prediction** | Intermediate | 预测个体处理效应 | PEHE | < 2.0 |
| **Uplift Ranking** | Advanced | 按 uplift 排序用户 | AUUC | > 0.7 |

## 挑战 1: ATE Estimation (初级)

### 任务描述

从存在混淆偏差的观察性数据中，估计职业培训对收入的平均因果效应。

### 数据集

- **场景**: LaLonde 职业培训项目
- **处理 (T)**: 是否参加培训 (0/1)
- **结果 (Y)**: 年收入 (美元)
- **协变量**: age, education, re74, re75, black, hispanic, married
- **样本**: 训练 1500, 测试 500

### 挑战点

低收入、低教育背景的人更可能参加培训 (混淆偏差)。
朴素估计 E[Y|T=1] - E[Y|T=0] 会严重有偏!

### 评估指标

**Relative Error** = |估计值 - 真实值| / |真实值|

得分计算:
- 0% 误差 → 100分
- 50% 误差 → 0分

### 代码模板

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

def estimate_ate(train_df, test_df):
    X_cols = ['age', 'education', 're74', 're75', 'black', 'hispanic', 'married']
    X = train_df[X_cols].values
    T = train_df['T'].values
    Y = train_df['Y'].values

    # TODO: 实现你的方法

    # 方法 1: 朴素估计 (会有偏!)
    # ate = Y[T==1].mean() - Y[T==0].mean()

    # 方法 2: IPW (倾向得分加权)
    # 1. 估计倾向得分 P(T=1|X)
    # 2. 使用倾向得分加权

    # 方法 3: 双重稳健 (推荐)
    # 1. 估计倾向得分和结果模型
    # 2. 结合两者

    return ate

# 定义预测
predictions = estimate_ate(train_data, test_data)
```

### 提示

1. 可视化处理组/控制组的协变量分布
2. 检查倾向得分的重叠性
3. 使用 IPW 或双重稳健估计
4. 交叉验证避免过拟合

### 基线方法

- **Naive**: 组间均值差 (有偏)
- **IPW**: 倾向得分加权
- **Matching**: 最近邻匹配

## 挑战 2: CATE Prediction (中级)

### 任务描述

预测每个个体的条件平均处理效应 (CATE)，而非仅仅估计平均效应。

### 数据集

- **场景**: IHDP 婴儿健康发展项目
- **处理 (T)**: 早期教育干预 (0/1)
- **结果 (Y)**: 认知测试得分
- **协变量**: X1-X10 (孩子和家庭特征)
- **样本**: 训练 3000, 测试 1000

### 挑战点

处理效应具有复杂的异质性!
- 线性异质性: τ(x) = 4 + 3*X1
- 二阶交互: + 2*X1*X2
- 非线性: + 1.5*sin(X3)
- 阈值效应: + 1*(X4>0)

### 评估指标

**PEHE** (Precision in Estimation of Heterogeneous Effect)
= sqrt(MSE(CATE))

次要指标:
- **Correlation**: 与真实 CATE 的相关性
- **R²**: 解释方差比例
- **ATE Bias**: 平均效应估计偏差

得分计算:
- PEHE = 0 → 100分
- PEHE = 5 → 0分
- Bonus: Correlation > 0.8 → +10分

### 代码模板

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_cate(train_df, test_df):
    X_cols = [f'X{i}' for i in range(1, 11)]

    X_train = train_df[X_cols].values
    T_train = train_df['T'].values
    Y_train = train_df['Y'].values
    X_test = test_df[X_cols].values

    # TODO: 实现你的方法

    # 方法 1: S-Learner
    # Y = f(X, T)
    # CATE = f(X, 1) - f(X, 0)

    # 方法 2: T-Learner (推荐起点)
    # mu_0 = E[Y|X, T=0]
    # mu_1 = E[Y|X, T=1]
    # CATE = mu_1 - mu_0

    # 方法 3: X-Learner (更好)
    # Stage 1: 估计 mu_0, mu_1
    # Stage 2: 计算伪效应并建模

    # 示例: T-Learner
    model_0 = RandomForestRegressor(n_estimators=100)
    model_1 = RandomForestRegressor(n_estimators=100)

    model_0.fit(X_train[T_train==0], Y_train[T_train==0])
    model_1.fit(X_train[T_train==1], Y_train[T_train==1])

    cate = model_1.predict(X_test) - model_0.predict(X_test)
    return cate

# 定义预测
predictions = predict_cate(train_data, test_data)
```

### 提示

1. 探索特征与效应的关系
2. 使用非线性模型 (RF, GBM)
3. 尝试 Meta-Learners (S/T/X-Learner)
4. 特征工程: 交互项、多项式
5. 可以用 econml 库的高级方法

### 基线方法

- **S-Learner**: 单一模型
- **T-Learner**: 两个独立模型
- **X-Learner**: 伪效应 + 加权

## 挑战 3: Uplift Ranking (高级)

### 任务描述

对用户按 uplift 排序，识别最应该接受营销干预的用户，最大化 ROI。

### 数据集

- **场景**: 电商优惠券发放
- **处理 (T)**: 收到优惠券 (0/1)
- **结果 (Y)**: 是否购买/转化 (0/1)
- **协变量**: feature_1 - feature_8 (用户特征)
- **样本**: 训练 5000, 测试 2000

### 挑战点

存在四类用户:
1. **Persuadables**: 发券会买，不发不买 (正 uplift) ✓
2. **Sure Things**: 本来就会买 (零 uplift)
3. **Lost Causes**: 怎么都不买 (零 uplift)
4. **Sleeping Dogs**: 发券反而不买 (负 uplift) ✗

需要识别 Persuadables，避免浪费和负效应!

### 评估指标

**AUUC** (Area Under Uplift Curve)
= Qini 曲线下面积

次要指标:
- **Normalized AUUC**: 归一化到 [0,1]
- **Kendall's Tau**: 排序一致性
- **Top-K Uplift**: Top K% 用户的真实 uplift
- **ROI**: 投资回报率

得分计算:
- 基于 Normalized AUUC (0-100)
- Bonus: ROI > 2.0 → +10分

### 代码模板

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def predict_uplift(train_df, test_df):
    X_cols = [f'feature_{i}' for i in range(1, 9)]

    X_train = train_df[X_cols].values
    T_train = train_df['T'].values
    Y_train = train_df['Y'].values
    X_test = test_df[X_cols].values

    # TODO: 实现你的方法

    # 注意: Y 是二分类，需要预测概率!

    # 方法 1: T-Learner (分类版本)
    # P(Y=1|T=1, X) - P(Y=1|T=0, X)

    # 方法 2: Class Transformation
    # 转换为 4 类问题

    # 方法 3: Uplift Tree/Forest
    # 直接建模 uplift

    # 示例: T-Learner
    model_0 = RandomForestClassifier(n_estimators=100)
    model_1 = RandomForestClassifier(n_estimators=100)

    model_0.fit(X_train[T_train==0], Y_train[T_train==0])
    model_1.fit(X_train[T_train==1], Y_train[T_train==1])

    p0 = model_0.predict_proba(X_test)[:, 1]
    p1 = model_1.predict_proba(X_test)[:, 1]

    uplift = p1 - p0
    return uplift

# 定义预测
predictions = predict_uplift(train_data, test_data)
```

### 提示

1. 关注排序质量而非绝对值
2. 使用分类模型预测概率
3. 可以尝试 causalml 库的 Uplift Tree
4. 计算 ROI 找最优干预比例
5. 分析特征重要性

### 基线方法

- **T-Learner**: 两个分类器
- **Class Transformation**: 4类问题

### 业务应用

**决策问题**: 应该向多少比例的用户发券?

- 成本: 1元/张券
- 收益: 10元/次转化
- ROI = (收益 - 成本) / 成本

根据 uplift 排序，对 top X% 用户发券:
- X 太小: 错失机会
- X 太大: 浪费成本
- 最优 X: 最大化 ROI

## 排行榜系统

### 查看排名

在 "Leaderboard" 标签页:
1. 选择挑战
2. 设置显示数量
3. 点击 "View Leaderboard"

### 用户进步

输入用户名查看:
- 提交历史
- 得分趋势
- 最佳成绩

### 数据持久化

提交记录保存在:
```
./challenge_submissions/
├── ATE_Estimation_Challenge.json
├── CATE_Prediction_Challenge.json
└── Uplift_Ranking_Challenge.json
```

## 高级用法

### 程序化提交

```python
from challenges import CATEPredictionChallenge, Leaderboard

# 创建挑战
challenge = CATEPredictionChallenge()
train_data, test_data = challenge.generate_data(seed=42)

# 训练你的模型
# ...

# 生成预测
predictions = your_model.predict(test_data)

# 评估
result = challenge.evaluate(predictions, user_name="YourName")

# 添加到排行榜
lb = Leaderboard("CATE Prediction")
lb.add_submission(result)

print(f"Score: {result.score:.2f}")
```

### 批量测试

```python
methods = ['s_learner', 't_learner', 'x_learner']

for method in methods:
    pred = challenge.get_baseline_predictions(method)
    result = challenge.evaluate(pred, user_name=f"Baseline-{method}")
    print(f"{method}: Score={result.score:.2f}")
```

### 导出结果

```python
from challenges import Leaderboard

lb = Leaderboard("CATE Prediction")

# 导出 CSV
lb.export_to_csv("leaderboard.csv")

# 获取 Markdown
md = lb.get_leaderboard_markdown(top_n=10)
with open("leaderboard.md", "w") as f:
    f.write(md)
```

## 常见问题

### Q: 代码执行报错怎么办?

A: 检查以下几点:
1. 预测格式是否正确 (numpy array)
2. 预测长度是否匹配测试集
3. 是否有 NaN 或 Inf 值
4. 变量名是否正确 (predictions 或 predict())

### Q: 如何提高分数?

A:
1. 理解数据生成过程
2. 尝试多种方法并对比
3. 使用交叉验证
4. 参考排行榜上的方法
5. 阅读相关论文

### Q: 可以使用哪些库?

A: 推荐使用:
- scikit-learn
- econml
- causalml
- xgboost
- lightgbm

### Q: 如何查看详细说明?

A: 在每个挑战页面:
1. 点击 "Initialize Challenge"
2. 查看右侧的 "Challenge Info"
3. 运行 Baseline 方法了解性能

## 学习路径

### 初学者

1. 从 ATE Estimation 开始
2. 理解混淆偏差和倾向得分
3. 实现 IPW 或双重稳健估计
4. 目标: 得分 > 70

### 进阶者

1. 挑战 CATE Prediction
2. 学习 Meta-Learners (S/T/X-Learner)
3. 尝试 econml 库的方法
4. 目标: 得分 > 85

### 高级玩家

1. 挑战 Uplift Ranking
2. 理解四类用户分群
3. 优化业务指标 (ROI)
4. 目标: Normalized AUUC > 0.8

## 评分标准

### 分数区间

- **90-100**: 优秀 (Excellent)
- **70-90**: 良好 (Good)
- **50-70**: 一般 (Average)
- **< 50**: 需改进 (Needs Improvement)

### 排名规则

1. 按得分降序排序
2. 同分时按提交时间
3. 每个用户可多次提交
4. 取最高分进入排行榜

## 参考资源

### 论文

- Rosenbaum & Rubin (1983) - 倾向得分
- Künzel et al. (2019) - Meta-Learners
- Radcliffe & Surry (2011) - Uplift Modeling

### 代码库

- [EconML](https://github.com/microsoft/EconML)
- [CausalML](https://github.com/uber/causalml)
- [DoWhy](https://github.com/py-why/dowhy)

### 教程

- "Causal Inference: The Mixtape" - Scott Cunningham
- "Applied Causal Inference" - Brady Neal

## 贡献

欢迎贡献:
- 新的挑战场景
- 改进的评估指标
- 更好的基线方法
- 文档优化

---

祝你在挑战中取得好成绩! Good luck!
