# 因果推断 Kaggle 比赛与数据集深度调研报告

> 本报告系统整理了因果推断、Uplift Modeling、处理效应估计相关的 Kaggle 比赛、经典数据集、工业界数据集和开源项目，按学习难度和顺序组织。

---

## 目录

1. [Kaggle 比赛](#1-kaggle-比赛)
2. [经典学术 Benchmark 数据集](#2-经典学术-benchmark-数据集)
3. [工业界公开数据集](#3-工业界公开数据集)
4. [GitHub 开源项目](#4-github-开源项目)
5. [学习路径建议](#5-学习路径建议)

---

## 1. Kaggle 比赛

### 1.1 历史相关比赛

#### ⚠️ 重要发现
目前 **没有找到** Kaggle 上专门针对因果推断或 Uplift Modeling 的大型比赛（2024-2025）。

**可能的原因：**
- 因果推断需要 RCT（随机对照试验）数据，商业敏感性高
- 真实的处理效应数据难以公开
- 多数工业应用在内部进行

#### 相关 A/B Testing 数据集（Kaggle）

虽然不是严格的因果推断比赛，但以下数据集可用于 A/B 测试和处理效应分析：

| 数据集名称 | 链接 | 规模 | 适用任务 | 难度 |
|-----------|------|------|---------|------|
| A/B Testing Dataset | [链接](https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset) | 中等 | ATE 估计 | ⭐ 入门 |
| Marketing A/B Testing | [链接](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) | 中等 | 营销效果评估 | ⭐ 入门 |
| Ad A/B Testing | [链接](https://www.kaggle.com/datasets/osuolaleemmanuel/ad-ab-testing) | 小 | 广告效果评估 | ⭐ 入门 |

**推荐用法：**
- 使用 PSM（倾向得分匹配）估计 ATE
- 使用 IPW/AIPW 进行处理效应估计
- 练习协变量平衡检查

**Baseline 方法：**
- T-test（简单对比）
- OLS 回归 + 协变量调整
- 倾向得分匹配（PSM）

---

## 2. 经典学术 Benchmark 数据集

这些是学术界广泛使用的标准 benchmark，通常是半合成数据（semi-synthetic），有真实的处理效应作为 ground truth。

### 2.1 IHDP (Infant Health and Development Program) ⭐⭐ 最推荐

**数据来源：** 1985-1988 年婴儿健康与发展项目的随机试验

**下载链接：**
- 直接下载 CSV: `https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv`
- IEEE DataPort: [Treatment Effect Estimation Benchmarks](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)
- Figshare: [Causal Machine Learning Benchmark Datasets](https://figshare.com/articles/dataset/Causal_Machine_Learning_Benchmark_Datasets/30244936)
- 原始数据: `https://github.com/vdorie/npci`

**数据特征：**
- **样本数量：** 747（139 处理组 + 608 对照组）
- **协变量：** 25 个（6 个连续 + 19 个二元）
- **结果变量：** 认知测试分数（连续）
- **处理：** 是否接受专业儿童护理

**适用任务：**
- ATE（平均处理效应）估计
- CATE（条件平均处理效应）预测
- 协变量平衡检查

**难度级别：** ⭐⭐ 中等
- 样本量适中
- 特征维度不高
- 非线性关系
- 组间不平衡（1:4）

**Baseline 方法：**
1. OLS 回归（线性基线）
2. 倾向得分匹配（PSM）
3. 双重稳健估计（AIPW）
4. S/T-Learner（Meta-Learner）
5. CEVAE（深度学习）

**SOTA 方法：**
- BCAUSS (Bayesian Causal Forest)
- Papers with Code: [IHDP Benchmark](https://paperswithcode.com/sota/causal-inference-on-ihdp)

**Python 加载示例：**
```python
import pandas as pd
# 直接加载
df = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)

# 使用 DoWhy
from dowhy import CausalModel
# 参考: https://www.pywhy.org/dowhy/v0.8/example_notebooks/dowhy_ihdp_data_example.html
```

---

### 2.2 LaLonde Jobs Dataset ⭐ 入门级

**数据来源：** 美国国家支持工作示范项目（NSW）1970s 数据

**下载链接：**
- NBER 官方数据: `https://users.nber.org/~rdehejia/data/.nswdata2.html`
- IEEE DataPort: [Treatment Effect Estimation Benchmarks](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)

**数据特征：**
- **样本数量：** 445（185 处理组 + 260 对照组）
- **协变量：** 8 个（年龄、教育、种族、婚姻状况、收入等）
- **结果变量：** 1978 年真实收入（连续）
- **处理：** 是否接受职业培训

**适用任务：**
- ATE 估计（职业培训对收入的影响）
- 匹配方法练习
- 敏感性分析

**难度级别：** ⭐ 入门
- 样本量小
- 特征维度低
- 经典问题，教学案例

**Baseline 方法：**
1. T-test 简单对比
2. OLS 回归 + 协变量
3. 倾向得分匹配（PSM）
4. 双重差分（DID，如果有面板数据版本）

**经典论文：**
- LaLonde (1986): Evaluating the Econometric Evaluations of Training Programs
- Dehejia & Wahba (1999): Causal Effects in Non-Experimental Studies

**R 加载：**
```r
library(haven)
nsw_data <- read_dta("http://www.nber.org/~rdehejia/data/nsw_dw.dta")

# 或使用 lalonde 包
library(lalonde)
data(nsw_dw)
```

---

### 2.3 Twins Dataset ⭐⭐⭐ 进阶

**数据来源：** 美国双胞胎出生数据 1989-1991

**下载链接：**
- Shalit Lab: `https://raw.githubusercontent.com/shalit-lab/Benchmarks/master/Twins/Final_data_twins.csv`
- IEEE DataPort: [Treatment Effect Estimation Benchmarks](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)
- Zenodo: `https://zenodo.org/records/14674618`

**数据特征：**
- **样本数量：** 11,984 对双胞胎（限制为同性、出生体重 < 2kg）
- **协变量：** 46 个（父母教育、种族、婚姻状况、怀孕风险因素等）
- **结果变量：** 死亡率（二元）
- **处理（半合成）：** 是否为较重的双胞胎（由 GESTAT10 模拟）

**适用任务：**
- ATE 和 CATE 估计
- 半合成数据的因果推断
- 处理高维协变量

**难度级别：** ⭐⭐⭐ 进阶
- 样本量大
- 特征维度高（46 个）
- 半合成数据（需理解数据生成过程）
- 高度混淆（GESTAT10 影响处理和结果）

**特殊性：**
- 半合成数据：真实协变量，模拟处理分配
- 可以"观察"两个潜在结果（双胞胎的对称性）
- Louizos et al. (2017) 通过隐藏一个双胞胎创建观测数据

**Baseline 方法：**
1. 倾向得分匹配（PSM）
2. 双重稳健估计（AIPW）
3. Causal Forest
4. CEVAE
5. DragonNet

---

### 2.4 News Dataset ⭐⭐

**数据来源：** 新闻文章词袋表示数据集

**下载链接：**
- IEEE DataPort: [Treatment Effect Estimation Benchmarks](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)

**数据特征：**
- 新闻文章的词袋（bag of words）表示
- 用于高维文本数据的因果推断

**适用任务：**
- 高维特征的处理效应估计
- 文本数据的因果推断

**难度级别：** ⭐⭐ 中等

---

### 2.5 ACIC 数据挑战赛 ⭐⭐⭐⭐ 竞赛级

**数据来源：** 美国因果推断会议（ACIC）年度数据挑战

**官方链接：**
- ACIC 2019: [https://sites.google.com/view/acic2019datachallenge](https://sites.google.com/view/acic2019datachallenge)
- ACIC 2023: [GitHub - zalandoresearch/ACIC23-competition](https://github.com/zalandoresearch/ACIC23-competition)

**历年挑战概览：**

| 年份 | 任务 | 数据规模 | 难度 |
|------|------|---------|------|
| ACIC 2016 | 估计 ATE | 数千个数据集 | ⭐⭐⭐ |
| ACIC 2019 | 估计 ATE | 变化的 DGP | ⭐⭐⭐ |
| ACIC 2022 | 医疗政策影响 | Medicare 数据 | ⭐⭐⭐⭐ |
| ACIC 2023 | 反事实预测 | 时间序列 + 多处理 | ⭐⭐⭐⭐⭐ |

**ACIC 2023 任务特点：**
- **目标：** 预测反事实（30 个预测/单位：5 个时间步 × 6 种处理）
- **评估：** MSE（均方误差）
- **挑战：** 传统预测方法不适用于反事实，传统因果估计方法不适用于预测

**顶级方法：**
- BART（贝叶斯可加回归树）
- Bayesian Causal Forests（BCF）
- Double Machine Learning

**R 包（ACIC 2016）：**
```r
install.packages("aciccomp2016")
library(aciccomp2016)
```

**适用人群：**
- 已掌握基础因果推断方法
- 希望在竞赛级数据上测试算法
- 研究新方法的性能边界

---

## 3. 工业界公开数据集

这些数据来自真实业务场景，具有极高的实践价值。

### 3.1 Criteo Uplift Dataset ⭐⭐⭐ 最重要的工业数据集

**数据来源：** Criteo AI Lab（广告科技公司）

**官方链接：**
- Criteo AI Lab: [https://ailab.criteo.com/criteo-uplift-prediction-dataset/](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- 直接下载: `http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz`
- Hugging Face: [criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift)

**数据特征：**
- **样本数量：** 25M（2500万）行
- **特征：** 12 个（f0-f11，连续特征）
- **处理：** exposure（是否被广告曝光，二元）
- **标签：**
  - visit（访问，二元）
  - conversion（转化，二元）
- **处理比例：** 85% treated, 15% control
- **转化率：** 0.29%
- **访问率：** 4.7%

**数据背景：**
- 来自多个增量测试（incrementality tests）的组合
- 随机试验：部分用户被阻止接受广告投放
- 真实广告投放数据

**适用任务：**
- Uplift Modeling（增益建模）
- ITE（个体处理效应）预测
- HTE（异质处理效应）估计
- 大规模数据的因果推断

**难度级别：** ⭐⭐⭐ 进阶
- **优点：** 真实工业数据，样本量巨大
- **挑战：**
  - 转化率极低（0.29%，类别不平衡）
  - 特征匿名化（无法解释）
  - 处理比例不平衡（85:15）

**Baseline 方法：**
1. S-Learner（单模型）
2. T-Learner（双模型）
3. X-Learner（交叉学习）
4. Uplift Random Forest
5. Causal Forest

**Python 加载：**
```python
# 使用 scikit-uplift
from sklift.datasets import fetch_criteo
dataset = fetch_criteo(target_col='conversion', treatment_col='exposure')
X, y, treatment = dataset.data, dataset.target, dataset.treatment

# 或直接下载
import pandas as pd
url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
df = pd.read_csv(url, compression='gzip')
```

**引用论文：**
- Diemert et al. (2018): "A Large Scale Benchmark for Uplift Modeling" (AdKDD 2018)

**适用场景：**
- 在线广告效果评估
- 用户定向投放优化
- 大规模 Uplift Modeling

---

### 3.2 Hillstrom Email Marketing Dataset ⭐⭐ 经典营销数据

**数据来源：** Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (2008)

**下载链接：**
- 直接下载: `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`

**数据特征：**
- **样本数量：** 64,000 客户
- **处理组：**
  - 1/3 接收男性商品邮件
  - 1/3 接收女性商品邮件
  - 1/3 不接收邮件（对照组）
- **观察期：** 邮件发送后两周
- **结果变量：**
  - visit（是否访问）
  - conversion（是否转化）
  - spend（消费金额）

**适用任务：**
- 多处理组的因果推断
- 邮件营销效果评估
- Uplift Modeling

**难度级别：** ⭐⭐ 中等
- 三组对比（多处理）
- 真实营销场景
- 样本量适中

**Baseline 方法：**
1. 多组 T-test
2. S/T/X-Learner（多处理版本）
3. Uplift Tree

**Python 加载：**
```python
# 使用 causeinfer
from causeinfer.data.hillstrom import load_hillstrom
df = load_hillstrom()

# 使用 pyuplift
from pyuplift.datasets import load_hillstrom_email_marketing
df = load_hillstrom_email_marketing()
```

**挑战任务：**
1. 估计男性商品 vs 对照组的 Uplift
2. 估计女性商品 vs 对照组的 Uplift
3. 找出对邮件最敏感的客户群（CATE）

---

### 3.3 X5 RetailHero Uplift Modeling ⭐⭐⭐⭐ 俄罗斯零售巨头

**数据来源：** X5 Retail Group（俄罗斯最大零售商）2019 年 Hackathon

**官方链接：**
- ODS.ai 竞赛页: [https://ods.ai/competitions/x5-retailhero-uplift-modeling](https://ods.ai/competitions/x5-retailhero-uplift-modeling)
- GitHub 解决方案: [kcostya/uplift-modeling](https://github.com/kcostya/uplift-modeling)

**数据特征：**
- **样本数量：**
  - clients.csv: 400,162 行
  - purchases.csv: 45,786,568 行（历史购买记录）
- **文件大小：** 4.17GB（未压缩）
- **特征：** 客户基本信息 + 购买历史
- **处理：** treatment_flg（是否发送 SMS）
- **结果：** target（是否购买）

**任务目标：**
- 对测试集客户预测 Uplift 值
- 排序客户（按通信效率降序）
- 选择 Top 30% 发送 SMS

**适用任务：**
- 真实零售场景的 Uplift Modeling
- 智能发券/短信优化
- 客户分层（高/低响应）

**难度级别：** ⭐⭐⭐⭐ 高难度
- 数据量大（4GB+）
- 需要特征工程（购买历史聚合）
- 真实业务约束（成本优化）
- 组间平衡（50:50）

**关键统计：**
- 处理组转化率：63.7%
- 对照组转化率：60.4%
- ATE（平均增益）：3.3%

**Baseline 方法：**
1. RFM 特征 + S-Learner
2. 购买历史聚合 + T-Learner
3. LightGBM/XGBoost
4. Uplift Random Forest

**Python 加载：**
```python
from sklift.datasets import fetch_x5
dataset = fetch_x5()
X, y, treatment = dataset.data, dataset.target, dataset.treatment
```

**评估指标：**
- Uplift@30% (Top 30% 客户的增益)
- Qini Curve
- AUUC（Area Under Uplift Curve）

---

### 3.4 Lenta Dataset ⭐⭐⭐ 俄罗斯零售 SMS 营销

**数据来源：** Lenta（俄罗斯大型零售连锁）BigTarget Hackathon 2020

**下载链接：**
```python
from sklift.datasets import fetch_lenta
dataset = fetch_lenta()
```

**数据特征：**
- **样本数量：** 687,029 行（原始）→ 176,065 行（清洗后）
- **特征：** 110 个（清洗后）
- **处理组：** 132,519（75%）
- **对照组：** 43,546（25%）
- **转化率：**
  - 处理组：11.912%
  - 对照组：10.257%
- **ATE：** 1.655%

**主要特征：**
- group（处理/对照）
- response_att（是否购买，目标变量）
- gender（性别）
- age（年龄）
- main_format（店铺类型：1=便利店, 0=大卖场）

**适用任务：**
- SMS 营销效果评估
- 小 Uplift 场景（ATE < 2%）
- 特征工程练习

**难度级别：** ⭐⭐⭐ 进阶
- 特征维度高（110+）
- 转化增益小（1.6%，需精准模型）
- 缺失值处理

**Baseline 方法：**
1. T-Learner + XGBoost
2. Causal Forest
3. S-Learner + 特征选择

---

### 3.5 Megafon Uplift Competition ⭐⭐⭐ 电信行业

**数据来源：** Megafon（俄罗斯电信公司）ODS.ai 2021 年竞赛

**下载链接：**
```python
from sklift.datasets import fetch_megafon
dataset = fetch_megafon()
```

**数据特征：**
- **样本数量：** 中等规模
- **特征：** 51 列（50 个 float 特征 + 1 个 id）
- **无缺失值**

**适用任务：**
- 电信行业 Uplift Modeling
- 匿名特征的因果推断

**难度级别：** ⭐⭐⭐ 进阶

---

## 4. GitHub 开源项目

### 4.1 核心因果推断库

#### 4.1.1 CausalML (Uber) ⭐⭐⭐⭐⭐

**GitHub:** [uber/causalml](https://github.com/uber/causalml)

**特点：**
- Uber 开源，工业级 Uplift Modeling 工具
- 统一接口估计 CATE/ITE
- 丰富的 Meta-Learner 实现

**核心功能：**
1. **Meta-Learner:**
   - S-Learner, T-Learner, X-Learner, R-Learner
2. **Uplift Tree/Forest**
3. **深度学习模型:**
   - CEVAE, DragonNet
4. **评估工具:**
   - Qini Curve, AUUC, Uplift Curve
5. **优化方法:**
   - Policy Optimization
   - Value Optimization

**示例：**
```python
from causalml.inference.meta import LRSRegressor
from causalml.metrics import plot_qini

# S-Learner
learner = LRSRegressor()
learner.fit(X, treatment, y)
cate = learner.predict(X_test)

# 评估
plot_qini(y_test, cate, treatment_test)
```

**学习资源：**
- 官方文档: [https://causalml.readthedocs.io/](https://causalml.readthedocs.io/)
- KDD 2021 Tutorial: [EconML/CausalML](https://causal-machine-learning.github.io/kdd2021-tutorial/)

**推荐指数：** ⭐⭐⭐⭐⭐
**适用人群：** 工业界从业者、Uplift Modeling 研究者

---

#### 4.1.2 EconML (Microsoft) ⭐⭐⭐⭐⭐

**GitHub:** [py-why/EconML](https://github.com/py-why/EconML)

**特点：**
- 微软 ALICE 项目
- 结合 ML + 计量经济学
- 异质处理效应（HTE）专家

**核心方法：**
1. **Orthogonal/Double Machine Learning:**
   - DML, Linear DML
2. **Meta-Learner:**
   - S/T/X-Learner
3. **Causal Forest:**
   - Generalized Random Forest
4. **Double Robustness:**
   - AIPW, Doubly Robust DML
5. **工具变量（IV）方法**

**示例：**
```python
from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor

# Double Machine Learning
dml = DML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor()
)
dml.fit(Y, T, X=X, W=W)
cate = dml.effect(X_test)
```

**与 DoWhy 集成：**
```python
from dowhy import CausalModel
import econml

# DoWhy 可调用 EconML 的估计器
```

**学习资源：**
- 官方文档: [https://econml.azurewebsites.net/](https://econml.azurewebsites.net/)
- PyWhy Discord: 社区讨论

**推荐指数：** ⭐⭐⭐⭐⭐
**适用人群：** 学术研究者、高级从业者、HTE 估计

---

#### 4.1.3 DoWhy (PyWhy) ⭐⭐⭐⭐⭐

**GitHub:** [py-why/dowhy](https://github.com/py-why/dowhy)

**特点：**
- 因果推断的"统一语言"
- 结合因果图（DAG）+ 潜在结果框架
- 显式建模和测试因果假设

**四步因果推断流程：**
1. **Model:** 构建因果图
2. **Identify:** 识别因果效应
3. **Estimate:** 估计效应大小
4. **Refute:** 敏感性分析

**示例：**
```python
from dowhy import CausalModel

# 1. 构建模型
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='outcome',
    common_causes=['confounder']
)

# 2. 识别
identified_estimand = model.identify_effect()

# 3. 估计
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

# 4. 反驳
refute = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
```

**2024 新功能：**
- DoWhy-GCM（图因果模型扩展）
- 时间序列因果推断支持
- 更稳健的分布变化方法

**学习资源：**
- 官方文档: [https://www.pywhy.org/dowhy](https://www.pywhy.org/dowhy)
- 教程: [DoWhy+EconML Tutorial](https://www.pywhy.org/dowhy/v0.8/example_notebooks/tutorial-causalinference-machinelearning-using-dowhy-econml.html)

**推荐指数：** ⭐⭐⭐⭐⭐
**适用人群：** 所有学习因果推断的人（入门到高级）

---

#### 4.1.4 scikit-uplift ⭐⭐⭐⭐

**GitHub:** [maks-sh/scikit-uplift](https://github.com/maks-sh/scikit-uplift)

**特点：**
- Sklearn 风格的 Uplift Modeling
- 丰富的评估指标
- 俄罗斯开源社区（ODS.ai）

**核心功能：**
1. **模型:**
   - Solo Model, Class Transformation
   - Two Model Approach
2. **评估指标（最全）:**
   - AUUC (Area Under Uplift Curve)
   - Qini Coefficient
   - Uplift@k
3. **可视化:**
   - plot_uplift_curve()
   - plot_qini_curve()
4. **数据集:**
   - fetch_criteo(), fetch_lenta(), fetch_x5()

**示例：**
```python
from sklift.models import SoloModel
from sklift.metrics import uplift_auc_score, plot_qini_curve
from lightgbm import LGBMClassifier

# Solo Model
model = SoloModel(LGBMClassifier())
model.fit(X_train, y_train, treat_train)
uplift = model.predict(X_test)

# 评估
score = uplift_auc_score(y_test, uplift, treat_test)
plot_qini_curve(y_test, uplift, treat_test)
```

**数据集加载：**
```python
from sklift.datasets import fetch_criteo, fetch_lenta, fetch_x5
criteo = fetch_criteo()
lenta = fetch_lenta()
x5 = fetch_x5()
```

**学习资源：**
- 官方文档: [https://www.uplift-modeling.com/](https://www.uplift-modeling.com/)
- RetailHero 教程: [GitHub Notebooks](https://github.com/maks-sh/scikit-uplift/tree/master/notebooks)

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** Uplift Modeling 实践者、营销分析师

---

#### 4.1.5 causallib (IBM BiomedSciAI) ⭐⭐⭐

**GitHub:** [BiomedSciAI/causallib](https://github.com/BiomedSciAI/causallib)

**特点：**
- IBM 开源
- Sklearn 风格 API
- 模块化设计

**核心功能：**
- IPW（逆概率加权）
- Standardization
- Matching
- 模型评估工具

**示例：**
```python
from causallib.estimation import IPW
from sklearn.linear_model import LogisticRegression

ipw = IPW(LogisticRegression())
ipw.fit(X, a, y)
effect = ipw.estimate_population_outcome(X, a, y)
```

**推荐指数：** ⭐⭐⭐
**适用人群：** 生物医学研究、学术研究

---

#### 4.1.6 CausalInference (Laurence Wong) ⭐⭐

**GitHub:** [laurencium/Causalinference](https://github.com/laurencium/Causalinference)

**特点：**
- 经典计量经济学方法
- 倾向得分专家
- 详细的理论文档

**官方网站:** [https://causalinferenceinpython.org/](https://causalinferenceinpython.org/)

**示例：**
```python
from causalinference import CausalModel
from causalinference.utils import random_data

Y, D, X = random_data()
causal = CausalModel(Y, D, X)
causal.est_via_matching()
print(causal.estimates)
```

**推荐指数：** ⭐⭐
**适用人群：** 计量经济学背景、倾向得分匹配

---

### 4.2 教程和学习项目

#### 4.2.1 Causal Inference for the Brave and True ⭐⭐⭐⭐⭐

**GitHub:** [matheusfacure/python-causality-handbook](https://github.com/matheusfacure/python-causality-handbook)

**特点：**
- 轻松幽默的写作风格
- 完整的因果推断入门书
- 多语言版本（葡萄牙语、法语、中文、西班牙语、韩语）

**内容涵盖：**
- 随机试验
- 因果图（DAG）
- 倾向得分
- 工具变量
- 双重差分（DID）
- 回归不连续（RDD）

**在线阅读:** [https://matheusfacure.github.io/python-causality-handbook/](https://matheusfacure.github.io/python-causality-handbook/)

**推荐指数：** ⭐⭐⭐⭐⭐
**适用人群：** 所有初学者（最推荐的入门资源）

---

#### 4.2.2 Deep Learning for Causal Inference ⭐⭐⭐⭐

**GitHub:** [kochbj/Deep-Learning-for-Causal-Inference](https://github.com/kochbj/Deep-Learning-for-Causal-Inference)

**特点：**
- 深度学习因果模型教程
- TensorFlow 2 + PyTorch 实现
- 从零开始构建 TARNet, DragonNet

**内容：**
1. **Representation Learning for Causal Inference**
   - TARNet 实现
   - 表示学习理论
2. **DragonNet (PyTorch)**
   - 渐近有效置信区间
   - SHAP 解释

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** 深度学习背景、想学习深度因果模型

---

#### 4.2.3 NYU Causal Inference Course (2024) ⭐⭐⭐⭐

**GitHub:** [kyunghyuncho/2024-causal-inference-machine-learning](https://github.com/kyunghyuncho/2024-causal-inference-machine-learning)

**特点：**
- 2024 年春 NYU 课程材料
- Kyunghyun Cho 教授
- 结构因果模型（SCM）+ 机器学习

**内容：**
- 结构因果模型
- 条件分布 vs 干预分布
- Google Colab 实验

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** 研究生、高级学习者

---

#### 4.2.4 CausalML/EconML KDD Tutorials ⭐⭐⭐⭐

**GitHub:** [causal-machine-learning](https://github.com/causal-machine-learning)

**特点：**
- KDD 2021, 2024, 2025 Workshop 材料
- EconML + CausalML 官方教程
- 工业界最佳实践

**官方网站:** [KDD 2021 Tutorial](https://causal-machine-learning.github.io/kdd2021-tutorial/)

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** 工业界从业者、希望了解 SOTA 方法

---

### 4.3 文本因果推断（NLP + Causality）

#### 4.3.1 CausalNLP ⭐⭐⭐⭐

**GitHub:** [amaiya/causalnlp](https://github.com/amaiya/causalnlp)

**特点：**
- 文本作为处理、结果或混淆变量
- 低代码因果推断（2 行命令）
- 内置 Autocoder（自动文本特征提取：主题、情感、情绪）

**示例：**
```python
from causalnlp import CausalInferenceModel

# 文本作为处理
model = CausalInferenceModel(df, treatment_col='text', outcome_col='clicked')
model.fit()
effect = model.estimate_ate()
```

**半合成数据集:**
- music_seed50.tsv（音乐评论 + 点击）

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** NLP 研究者、文本挖掘

---

#### 4.3.2 Corr2Cause Dataset (ICLR 2024) ⭐⭐⭐⭐

**GitHub:** [causalNLP/corr2cause](https://github.com/causalNLP/corr2cause)

**特点：**
- 测试 LLM 的因果推理能力
- 200K+ 样本
- 从相关性推断因果关系

**任务：**
- 输入：一组相关性陈述
- 输出：变量之间的因果关系

**Hugging Face:** [causalnlp/corr2cause](https://huggingface.co/datasets/causalnlp/corr2cause)

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** LLM + 因果推断交叉研究

---

### 4.4 Benchmark 和评估框架

#### 4.4.1 CausalBench (Gene Networks) ⭐⭐⭐⭐

**GitHub:** [causalbench/causalbench](https://github.com/causalbench/causalbench)

**特点：**
- 单细胞基因表达数据
- 200,000+ 训练样本（真实扰动数据）
- 基因调控网络推断

**论文：** Communications Biology (2025)

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** 生物信息学、基因网络

---

#### 4.4.2 CSuite (Microsoft) ⭐⭐⭐⭐

**GitHub:** [microsoft/csuite](https://github.com/microsoft/csuite)

**特点：**
- 因果发现 + ATE/CATE 估计 benchmark
- 已知结构方程模型（SEM）
- 提供干预数据

**推荐指数：** ⭐⭐⭐⭐
**适用人群：** 因果发现、算法评估

---

#### 4.4.3 awesome-causality-algorithms ⭐⭐⭐

**GitHub:** [rguo12/awesome-causality-algorithms](https://github.com/rguo12/awesome-causality-algorithms)

**特点：**
- 因果学习算法索引
- 论文 + 代码合集

**推荐指数：** ⭐⭐⭐
**适用人群：** 快速查找算法实现

---

## 5. 学习路径建议

### 5.1 入门路径（0-3 个月）

#### 第一步：理解基础概念
**推荐资源：**
1. **Causal Inference for the Brave and True**
   - 阅读前 5 章（随机试验、因果图、倾向得分）
   - 动手实践每章代码

**练习数据集：**
1. **LaLonde Jobs Dataset** ⭐
   - 小样本，易于理解
   - 练习 PSM、IPW

2. **Kaggle A/B Testing Datasets** ⭐
   - 简单对比分析
   - T-test + OLS 回归

#### 第二步：掌握核心方法
**推荐库：**
- DoWhy（理解因果推断流程）
- CausalInference（倾向得分匹配）

**练习数据集：**
- **IHDP Dataset** ⭐⭐
  - 标准 benchmark
  - 对比 PSM vs AIPW vs S-Learner

**目标：**
- 能独立完成 ATE 估计
- 理解协变量平衡检查
- 会画 Uplift Curve

---

### 5.2 进阶路径（3-6 个月）

#### 第一步：Uplift Modeling
**推荐库：**
- scikit-uplift
- CausalML

**练习数据集：**
1. **Hillstrom Email Marketing** ⭐⭐
   - 多处理组
   - 练习 T-Learner, X-Learner

2. **Criteo Uplift Dataset** ⭐⭐⭐
   - 真实工业数据
   - 大样本，类别不平衡

**关键技能：**
- Meta-Learner（S/T/X）
- Uplift Tree/Forest
- Qini/AUUC 评估

#### 第二步：异质效应（HTE）
**推荐库：**
- EconML

**练习数据集：**
- **Twins Dataset** ⭐⭐⭐
  - 高维协变量
  - 练习 Causal Forest

**目标：**
- 估计 CATE（条件平均处理效应）
- 识别高响应人群
- 敏感性分析

---

### 5.3 高级路径（6-12 个月）

#### 第一步：深度因果模型
**推荐资源：**
- Deep Learning for Causal Inference (GitHub)

**练习数据集：**
- **IHDP + Twins** ⭐⭐⭐
  - 实现 TARNet
  - 实现 DragonNet

**推荐库：**
- PyTorch + EconML
- CausalML (CEVAE)

#### 第二步：竞赛级挑战
**推荐数据集：**
1. **X5 RetailHero** ⭐⭐⭐⭐
   - 真实业务场景
   - 大规模数据
   - 特征工程

2. **ACIC 2022/2023** ⭐⭐⭐⭐
   - 多个 DGP
   - 时间序列 + 多处理
   - SOTA 方法对比

**目标：**
- 在 RetailHero 数据上达到 Top 10%
- 实现论文中的最新方法
- 发表研究成果

---

### 5.4 专业方向

#### 方向 1：工业界应用
**重点数据集：**
- Criteo Uplift
- X5 RetailHero
- Lenta

**核心技能：**
- 大规模 Uplift Modeling
- 成本约束优化
- A/B 测试设计

**推荐库：**
- CausalML
- scikit-uplift

---

#### 方向 2：学术研究
**重点数据集：**
- ACIC Competition
- IHDP, Twins
- CSuite (Microsoft)

**核心技能：**
- 新方法开发
- Benchmark 评估
- 理论证明

**推荐库：**
- EconML
- DoWhy

---

#### 方向 3：深度学习 + 因果推断
**重点数据集：**
- IHDP, Twins
- CausalBench (Gene Networks)

**核心技能：**
- 表示学习
- 深度因果模型
- 神经网络 + 因果理论

**推荐库：**
- PyTorch + EconML
- TensorFlow 2

---

#### 方向 4：NLP + 因果推断
**重点数据集：**
- Corr2Cause (ICLR 2024)
- CausalNLP 内置数据集

**核心技能：**
- 文本作为处理/结果/混淆
- LLM 因果推理
- 文本特征提取

**推荐库：**
- CausalNLP
- DoWhy + NLP

---

## 6. 推荐学习顺序（总结）

### 数据集难度梯度

| 难度 | 数据集 | 样本量 | 任务类型 | 推荐时间 |
|------|--------|--------|---------|---------|
| ⭐ 入门 | LaLonde Jobs | 445 | ATE | 第 1 周 |
| ⭐ 入门 | Kaggle A/B Test | 中等 | ATE | 第 2 周 |
| ⭐⭐ 中等 | IHDP | 747 | ATE/CATE | 第 3-4 周 |
| ⭐⭐ 中等 | Hillstrom | 64,000 | Uplift | 第 5-6 周 |
| ⭐⭐⭐ 进阶 | Criteo | 25M | Uplift | 第 7-10 周 |
| ⭐⭐⭐ 进阶 | Twins | 11,984 对 | CATE | 第 11-14 周 |
| ⭐⭐⭐ 进阶 | Lenta | 687,029 | Uplift | 第 15-18 周 |
| ⭐⭐⭐⭐ 高级 | X5 RetailHero | 4.17GB | Uplift | 第 19-24 周 |
| ⭐⭐⭐⭐ 高级 | ACIC 2022 | 多 DGP | ATE/CATE | 第 25-30 周 |
| ⭐⭐⭐⭐⭐ 专家 | ACIC 2023 | 时间序列 | 反事实预测 | 第 31-36 周 |

---

## 7. 数据集 Quick Access（快速访问表）

### 7.1 一键加载（Python）

```python
# ===== 经典学术数据集 =====
import pandas as pd

# IHDP
ihdp = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)

# LaLonde
lalonde = pd.read_stata("http://www.nber.org/~rdehejia/data/nsw_dw.dta")

# Twins
twins = pd.read_csv("https://raw.githubusercontent.com/shalit-lab/Benchmarks/master/Twins/Final_data_twins.csv")

# ===== 工业界数据集 =====
from sklift.datasets import fetch_criteo, fetch_lenta, fetch_x5

# Criteo (直接下载)
criteo = pd.read_csv("http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz", compression='gzip')

# 或使用 scikit-uplift
criteo_sklift = fetch_criteo()
lenta = fetch_lenta()
x5 = fetch_x5()

# Hillstrom
hillstrom = pd.read_csv("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
```

---

### 7.2 数据集元信息表

| 数据集 | 样本量 | 特征数 | 处理类型 | 结果类型 | 来源 | 下载方式 |
|--------|--------|--------|---------|---------|------|---------|
| IHDP | 747 | 25 | 二元 | 连续 | 学术 | URL/DoWhy |
| LaLonde | 445 | 8 | 二元 | 连续 | 学术 | NBER |
| Twins | 11,984 对 | 46 | 二元（半合成） | 二元 | 学术 | GitHub |
| Criteo | 25M | 12 | 二元 | 二元 | 工业 | URL/sklift |
| Hillstrom | 64,000 | 多 | 三组 | 二元+连续 | 工业 | URL |
| X5 RetailHero | 400K+ | 多 | 二元 | 二元 | 工业 | ODS.ai/sklift |
| Lenta | 687K | 110 | 二元 | 二元 | 工业 | sklift |

---

## 8. 关键论文引用

### 8.1 经典论文

1. **LaLonde (1986):** "Evaluating the Econometric Evaluations of Training Programs." *American Economic Review*
2. **Dehejia & Wahba (1999):** "Causal Effects in Non-Experimental Studies." *JASA*
3. **Louizos et al. (2017):** "Causal Effect Inference with Deep Latent-Variable Models." *NeurIPS*

### 8.2 工业应用

4. **Diemert et al. (2018):** "A Large Scale Benchmark for Uplift Modeling." *AdKDD 2018*
5. **Athey & Imbens (2016):** "Recursive Partitioning for Heterogeneous Causal Effects." *PNAS*

### 8.3 深度学习

6. **Shalit et al. (2017):** "Estimating Individual Treatment Effect: Generalization Bounds and Algorithms." *ICML*
7. **Shi et al. (2019):** "Adapting Neural Networks for the Estimation of Treatment Effects." *NeurIPS*

---

## 9. 常见问题（FAQ）

### Q1: Kaggle 为什么没有因果推断比赛？
**A:** 因果推断需要 RCT 数据（随机对照试验），这类数据商业敏感性高，且真实处理效应难以公开。大部分工业应用在内部进行。

### Q2: 半合成数据（semi-synthetic）是什么？
**A:** 使用真实协变量，但模拟处理分配和/或结果变量，确保有已知的 ground truth 处理效应。例如 IHDP、Twins。

### Q3: 应该先学 ATE 还是 Uplift Modeling？
**A:** 先学 ATE。Uplift Modeling 是 CATE 的应用，需要先掌握 ATE 估计基础（PSM, IPW, AIPW）。

### Q4: Criteo 数据集为什么这么大（25M）？
**A:** 真实广告投放数据，反映工业界规模。建议先用 10% 采样练习。

### Q5: ACIC 数据挑战适合初学者吗？
**A:** 不适合。ACIC 是竞赛级难度，建议先完成 IHDP + Twins。

---

## 10. 数据集下载地址汇总

### 一站式下载（IEEE DataPort）
[Treatment Effect Estimation Benchmarks](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)
- 包含：IHDP, Jobs, Twins, News

### Figshare
[Causal Machine Learning Benchmark Datasets](https://figshare.com/articles/dataset/Causal_Machine_Learning_Benchmark_Datasets/30244936)

### GitHub Collections
- ACIC 2023: [zalandoresearch/ACIC23-competition](https://github.com/zalandoresearch/ACIC23-competition)
- RealCause: [bradyneal/realcause](https://github.com/bradyneal/realcause)

---

## 参考资源（Sources）

### Kaggle & Competitions
- [Kaggle Uplift Modeling Discussion](https://www.kaggle.com/discussions/getting-started/132958)
- [Software Usage Promotion Campaign Dataset](https://www.kaggle.com/datasets/hwwang98/software-usage-promotion-campaign-uplift-model)

### Academic Datasets
- [Papers with Code - IHDP Benchmark](https://paperswithcode.com/sota/causal-inference-on-ihdp)
- [DoWhy IHDP Example](https://www.pywhy.org/dowhy/v0.8/example_notebooks/dowhy_ihdp_data_example.html)
- [ACIC 2019 Data Challenge](https://sites.google.com/view/acic2019datachallenge/home)
- [ACIC 2023 GitHub](https://github.com/zalandoresearch/ACIC23-competition)

### Industry Datasets
- [Criteo AI Lab - Uplift Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [Criteo on Hugging Face](https://huggingface.co/datasets/criteo/criteo-uplift)
- [scikit-uplift Criteo](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_criteo.html)
- [Hillstrom Dataset](https://causeinfer.readthedocs.io/en/latest/data/hillstrom.html)
- [X5 RetailHero Competition](https://ods.ai/competitions/x5-retailhero-uplift-modeling)
- [scikit-uplift Lenta](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html)

### GitHub Projects
- [Uber CausalML](https://github.com/uber/causalml)
- [Microsoft EconML](https://github.com/py-why/EconML)
- [PyWhy DoWhy](https://github.com/py-why/dowhy)
- [scikit-uplift](https://github.com/maks-sh/scikit-uplift)
- [Causal Inference for the Brave and True](https://github.com/matheusfacure/python-causality-handbook)
- [Deep Learning for Causal Inference](https://github.com/kochbj/Deep-Learning-for-Causal-Inference)
- [NYU 2024 Causal Inference Course](https://github.com/kyunghyuncho/2024-causal-inference-machine-learning)
- [CausalNLP](https://github.com/amaiya/causalnlp)
- [Corr2Cause (ICLR 2024)](https://github.com/causalNLP/corr2cause)
- [CausalBench](https://github.com/causalbench/causalbench)
- [Microsoft CSuite](https://github.com/microsoft/csuite)

### Papers
- [Criteo Uplift Paper (2018)](https://arxiv.org/pdf/2111.10106)
- [ACIC Competition Review](https://muse.jhu.edu/article/895650)
- [Twins Dataset (Louizos et al. 2017)](https://arxiv.org/pdf/2202.02195)

---

**最后更新：** 2026-01-04
**作者：** Claude Code
**项目：** Causal Inference Workbench

---

## 附录：推荐阅读顺序（for 你的项目）

基于你的项目（Causal Inference Workbench），建议按以下顺序集成数据集到 `challenges/` 模块：

### Phase 1: 基础挑战（已完成部分）
1. **Challenge 1: ATE 估计** → 使用 **IHDP** 或 **LaLonde**
2. **Challenge 2: CATE 预测** → 使用 **IHDP**（非线性）
3. **Challenge 3: Uplift 排序** → 使用 **Hillstrom**（小规模）

### Phase 2: 工业挑战（新增）
4. **Challenge 4: 大规模 Uplift** → 使用 **Criteo**
   - 任务：Top 30% 客户选择
   - 评估：AUUC, Qini
5. **Challenge 5: 智能发券** → 使用 **X5 RetailHero**
   - 任务：成本约束下的 ROI 最大化
   - 评估：Uplift@30%, Revenue Gain

### Phase 3: 高级挑战
6. **Challenge 6: 时间序列 + 因果** → 使用 **ACIC 2023**
   - 任务：反事实预测
   - 评估：MSE

### 数据集下载脚本（建议）
在 `datasets/utils.py` 中添加：
```python
def download_dataset(name='ihdp'):
    """一键下载标准数据集"""
    if name == 'ihdp':
        return pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)
    elif name == 'criteo':
        from sklift.datasets import fetch_criteo
        return fetch_criteo()
    # ...
```

这样可以让挑战系统更完整，覆盖入门→进阶→专家全路径。
