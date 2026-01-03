# Deep Learning for Causal Inference - 学习资料大全

> 本文档为因果推断学习提供系统化的资源整理，侧重于深度学习方法，适合准备数据科学/机器学习面试。

---

## 目录

1. [学习路线图](#学习路线图)
2. [核心论文精读](#核心论文精读)
3. [经典书籍](#经典书籍)
4. [在线课程](#在线课程)
5. [Benchmark 数据集](#benchmark-数据集)
6. [Kaggle 比赛](#kaggle-比赛)
7. [开源工具库](#开源工具库)
8. [面试准备](#面试准备)
9. [项目优化建议](#项目优化建议)

---

## 学习路线图

### Phase 1: 基础概念 (2-3周)

```
潜在结果框架 → 因果图 (DAG) → 混淆/选择偏差 → 可识别性条件
```

**关键概念**:
- Potential Outcomes Framework (Rubin)
- Directed Acyclic Graphs (Pearl)
- Confounding, Selection Bias, Collider Bias
- Backdoor Criterion, Frontdoor Criterion
- ATE, ATT, CATE, ITE

### Phase 2: 传统方法 (2-3周)

```
倾向得分匹配 → IPW/AIPW → 双重稳健估计 → 工具变量
```

**关键方法**:
- Propensity Score Matching (PSM)
- Inverse Probability Weighting (IPW)
- Doubly Robust Estimation
- Difference-in-Differences (DiD)
- Regression Discontinuity Design (RDD)
- Instrumental Variables (IV)

### Phase 3: 机器学习方法 (3-4周)

```
Meta-Learners → Causal Forest → Uplift Modeling → Double ML
```

**关键方法**:
- S-Learner, T-Learner, X-Learner
- Causal Forest (GRF)
- Uplift Trees & Random Forests
- Double Machine Learning (DML)
- Orthogonal ML

### Phase 4: 深度学习方法 (4-6周)

```
TARNet → DragonNet → CEVAE → GANITE → TransTEE
```

**关键方法**:
- TARNet / CFR (Counterfactual Regression)
- DragonNet (Targeted Regularization)
- CEVAE (Causal Effect VAE)
- GANITE (Generative Adversarial)
- VCNet (Dose-Response)
- TransTEE (Transformer-based)

### Phase 5: 高级主题 (持续学习)

```
敏感性分析 → 因果发现 → 时序因果 → 图神经网络
```

---

## 核心论文精读

### 1. 基础架构论文 (必读)

#### TARNet / CFR
- **论文**: [Estimating individual treatment effect: generalization bounds and algorithms](https://arxiv.org/abs/1606.03976)
- **作者**: Shalit, Johansson, Sontag (ICML 2017)
- **核心创新**:
  - 共享表示层 + 双头输出架构
  - IPM (Integral Probability Metric) 正则化平衡表示分布
  - 建立了表示学习与 ITE 估计误差的理论联系
- **代码**: https://github.com/clinicalml/cfrnet

#### DragonNet
- **论文**: [Adapting Neural Networks for the Estimation of Treatment Effects](https://arxiv.org/abs/1906.02120)
- **作者**: Shi, Blei, Veitch (NeurIPS 2019)
- **核心创新**:
  - 端到端学习倾向得分
  - Targeted Regularization (类似 TMLE)
  - 比 TARNet 更数据高效
- **代码**: https://github.com/claudiashi57/dragonnet

#### CEVAE
- **论文**: [Causal Effect Inference with Deep Latent-Variable Models](https://arxiv.org/abs/1705.08821)
- **作者**: Louizos et al. (NeurIPS 2017)
- **核心创新**:
  - VAE 框架处理隐变量混淆
  - 可处理 unobserved confounders
  - 生成反事实样本
- **代码**: https://github.com/AMLab-Amsterdam/CEVAE

#### GANITE
- **论文**: [GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets](https://openreview.net/forum?id=ByKWUeWA-)
- **作者**: Yoon, Jordon, van der Schaar (ICLR 2018)
- **核心创新**:
  - GAN 框架生成反事实结果
  - 两阶段: 反事实生成 + ITE 估计
  - 支持多处理场景
- **代码**: https://github.com/jsyoon0823/GANITE

### 2. 表示学习方法

#### BNN (Balancing Neural Network)
- **论文**: Learning Representations for Counterfactual Inference
- **作者**: Johansson et al. (ICML 2016)
- **要点**: 最早将表示学习用于因果推断

#### SITE
- **论文**: Representation Learning for Treatment Effect Estimation from Observational Data
- **作者**: Yao et al. (NeurIPS 2018)
- **要点**: 保持局部相似性的表示学习

#### Perfect Match
- **论文**: Perfect Match: A Simple Method for Learning Representations For Counterfactual Inference With Neural Networks
- **作者**: Schwab et al. (2018)
- **要点**: 匹配思想与神经网络结合

### 3. 连续处理/剂量响应

#### VCNet
- **论文**: VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments
- **作者**: Nie et al. (ICLR 2021)
- **核心创新**:
  - 处理连续值处理变量
  - 变分推断 + 目标正则化

#### SCIGAN
- **论文**: Estimating Counterfactual Treatment Outcomes over Time Through Adversarially Balanced Representations
- **作者**: Bica et al. (ICLR 2020)
- **要点**: 连续处理 + 时序数据

### 4. Transformer 架构

#### TransTEE
- **论文**: TransTEE: Transformer-based Treatment Effect Estimation
- **会议**: NeurIPS 2022
- **核心创新**:
  - 注意力机制学习协变量重要性
  - 更好的特征交互建模

#### Causal Transformer
- **论文**: Causal Transformer for Estimating Counterfactual Outcomes
- **会议**: ICML 2022
- **要点**: 将 Transformer 用于时序因果推断

### 5. 综述论文 (必读)

#### A Survey of Deep Causal Models and Their Industrial Applications (2024)
- **链接**: https://link.springer.com/article/10.1007/s10462-024-10886-0
- **要点**: 最新最全面的深度因果模型综述

#### Causal Inference Meets Deep Learning: A Comprehensive Survey (2024)
- **链接**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11384545/
- **要点**: 因果推断与深度学习交叉领域综述

#### Deep Causal Learning: Representation, Discovery and Inference
- **链接**: https://dl.acm.org/doi/10.1145/3762179
- **期刊**: ACM Computing Surveys
- **要点**: 系统性回顾深度因果学习

### 6. 工业界应用论文

#### Uber CausalML
- **论文**: CausalML: Python Package for Causal Machine Learning
- **会议**: KDD 2020
- **链接**: https://github.com/uber/causalml

#### Microsoft EconML
- **论文**: Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning
- **会议**: PNAS 2019
- **链接**: https://github.com/py-why/EconML

#### Netflix Causal Inference
- **博客**: https://netflixtechblog.com/
- **要点**: 推荐系统中的因果推断

---

## 经典书籍

### 入门级

#### 1. Causal Inference: What If
- **作者**: Miguel Hernán, James Robins
- **免费下载**: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
- **适合人群**: 初学者
- **特点**:
  - 免费在线版本
  - 包含 R/Stata 代码
  - 从基础概念到高级方法
  - 医学/流行病学视角

#### 2. The Book of Why
- **作者**: Judea Pearl
- **适合人群**: 所有人
- **特点**: 科普性质，通俗易懂

#### 3. Causal Inference: The Mixtape
- **作者**: Scott Cunningham
- **免费链接**: https://mixtape.scunning.com/
- **适合人群**: 经济学背景
- **特点**:
  - 完全免费在线
  - 经济学视角
  - 大量实例

### 中级

#### 4. Elements of Causal Inference
- **作者**: Peters, Janzing, Schölkopf
- **免费下载**: MIT Press 官网
- **适合人群**: ML 背景
- **特点**: 机器学习视角的因果推断

#### 5. Causal Inference for Data Science
- **作者**: Aleix Ruiz de Villa
- **出版**: Manning
- **特点**: 实践导向，Python 代码

### 高级

#### 6. Causality: Models, Reasoning, and Inference (2nd Edition)
- **作者**: Judea Pearl
- **适合人群**: 研究者
- **特点**: 因果推断圣经，理论深入

#### 7. Targeted Learning
- **作者**: van der Laan, Rose
- **适合人群**: 统计学背景
- **特点**: TMLE 理论基础

---

## 在线课程

### 免费课程

#### 1. Brady Neal's Introduction to Causal Inference (强烈推荐)
- **链接**: https://www.bradyneal.com/causal-inference-course
- **特点**:
  - 完全免费
  - 视频 + 教材 + Slack 社区
  - ML 视角
  - 循序渐进
- **配套资源**:
  - YouTube 视频
  - 免费教材 PDF
  - 读书会

#### 2. MIT 6.S091: Introduction to Causal Inference
- **平台**: MIT OpenCourseWare
- **特点**: 学术严谨

#### 3. Stanford CS 246: Mining Massive Datasets (Causal Inference部分)
- **平台**: Stanford Online
- **特点**: 大规模数据处理视角

### 付费课程

#### 4. Coursera: A Crash Course in Causality
- **提供方**: University of Pennsylvania
- **难度**: 入门
- **特点**: 系统化，有证书

#### 5. edX: Causal Diagrams
- **提供方**: Harvard University
- **特点**: DAG 专项

### 教程和文档

#### 6. Microsoft EconML Tutorials
- **链接**: https://www.pywhy.org/econml/
- **特点**: 工业级工具使用教程

#### 7. Uber CausalML Tutorials
- **链接**: https://causalml.readthedocs.io/
- **特点**: Uplift Modeling 专项

#### 8. PyWhy DoWhy Tutorials
- **链接**: https://www.pywhy.org/dowhy/
- **特点**: 因果推断完整工作流

---

## Benchmark 数据集

### 1. IHDP (Infant Health and Development Program)

```python
# 规模: ~750 samples
# 特征: 25 个协变量
# 任务: ITE/ATE 估计
# 特点: 经典小规模 benchmark，有 ground truth

# 下载方式
from datasets.ihdp import load_ihdp
X, T, Y, Y0_true, Y1_true = load_ihdp()
```

**适合练习**: 基础深度学习模型 (TARNet, DragonNet)

### 2. Jobs (LaLonde Job Training)

```python
# 规模: ~5,000 samples
# 特征: 职业培训相关
# 任务: ATT 估计
# 特点: 经济学经典数据集

# 下载方式
from causalml.dataset import make_uplift_classification
```

**适合练习**: 传统因果推断方法

### 3. Twins Dataset

```python
# 规模: ~4,000 twin pairs
# 特征: 出生体重相关
# 任务: ITE 估计
# 特点: 双胞胎自然对照

# 来源: https://github.com/AMLab-Amsterdam/CEVAE
```

**适合练习**: CEVAE 等生成模型

### 4. Criteo Uplift Dataset (工业级)

```python
# 规模: 13.9M samples
# 特征: 11 个匿名特征
# 任务: Uplift Modeling
# 特点: 最大规模公开 uplift 数据集

# 下载方式
from sklift.datasets import fetch_criteo
data = fetch_criteo()

# 或使用 TensorFlow Datasets
import tensorflow_datasets as tfds
ds = tfds.load('criteo')
```

**适合练习**: 大规模 Uplift 模型，生产级评估

### 5. Hillstrom Email Marketing

```python
# 规模: ~64,000 samples
# 特征: 客户属性
# 任务: Uplift Modeling
# 特点: 经典营销数据集

# 下载方式
from sklift.datasets import fetch_hillstrom
data = fetch_hillstrom()
```

**适合练习**: 中等规模 Uplift 模型

### 6. ACIC Competition Datasets

```python
# 规模: 变化
# 特点: 竞赛级别，设计复杂

# 来源: Atlantic Causal Inference Conference
# 链接: https://jenniferhill7.wixsite.com/acic-2016
```

**适合练习**: 竞赛准备，复杂场景

### 7. News Dataset

```python
# 规模: ~5,000 samples
# 特征: 文本相关
# 任务: ITE 估计
# 特点: 高维特征

# 来源: Perfect Match paper
```

**适合练习**: 高维数据处理

### 数据集对比

| 数据集 | 样本量 | 特征数 | 难度 | 推荐模型 |
|--------|--------|--------|------|----------|
| IHDP | 750 | 25 | 入门 | TARNet, DragonNet |
| Jobs | 5K | 8 | 入门 | PSM, IPW |
| Twins | 4K | 40 | 中级 | CEVAE, GANITE |
| Hillstrom | 64K | 8 | 中级 | Meta-Learners, Uplift Tree |
| Criteo | 13.9M | 11 | 高级 | 大规模 Uplift |
| ACIC | 变化 | 变化 | 高级 | 综合评估 |

---

## Kaggle 比赛

### 历史相关比赛

#### 1. Criteo Uplift Modeling Challenge
- **任务**: 预测广告增益效应
- **数据**: Criteo Uplift Dataset
- **关键技巧**:
  - 特征工程
  - Uplift Forest
  - 模型集成

#### 2. X5 Retail Hero: Uplift Modeling
- **任务**: 零售促销增益
- **平台**: 俄罗斯 Kaggle
- **获奖方案**: https://github.com/maks-sh/x5-uplift

#### 3. Causal Discovery Competitions
- **平台**: ChaLearn
- **任务**: 因果发现

### Kaggle 学习资源

#### Notebooks
- [Causal Inference Tutorial](https://www.kaggle.com/code)
- [Uplift Modeling with CausalML](https://www.kaggle.com/code)
- [A/B Testing Analysis](https://www.kaggle.com/code)

---

## 开源工具库

### 1. EconML (Microsoft)

```python
pip install econml

# 示例: Double ML
from econml.dml import LinearDML
model = LinearDML()
model.fit(Y, T, X=X, W=W)
ate = model.ate(X)
```

**特点**:
- Double Machine Learning
- Orthogonal Forest
- Deep IV
- 工业级稳定性

**文档**: https://www.pywhy.org/econml/

### 2. CausalML (Uber)

```python
pip install causalml

# 示例: Meta-Learners
from causalml.inference.meta import LGBMTRegressor
learner = LGBMTRegressor()
cate = learner.fit_predict(X, treatment, y)
```

**特点**:
- Uplift Modeling
- Meta-Learners (S/T/X)
- Uplift Trees
- 可视化工具

**文档**: https://causalml.readthedocs.io/

### 3. DoWhy (PyWhy)

```python
pip install dowhy

# 示例: 完整工作流
import dowhy
model = dowhy.CausalModel(data, treatment, outcome, graph)
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand)
refutation = model.refute_estimate(estimate)
```

**特点**:
- 因果图建模
- 可识别性分析
- 敏感性分析
- 与 EconML 集成

**文档**: https://www.pywhy.org/dowhy/

### 4. scikit-uplift

```python
pip install scikit-uplift

# 示例
from sklift.models import SoloModel
model = SoloModel(estimator=RandomForestClassifier())
model.fit(X_train, y_train, treat_train)
```

**特点**:
- Uplift 专用
- Qini/Uplift 曲线
- 多种模型

**文档**: https://www.uplift-modeling.com/

### 工具对比

| 工具 | 优势 | 适用场景 |
|------|------|----------|
| EconML | DML, 理论严谨 | CATE 估计, 政策评估 |
| CausalML | Uplift, 实用 | 营销优化, A/B增强 |
| DoWhy | 完整工作流, 敏感性 | 端到端因果分析 |
| scikit-uplift | 轻量, 易用 | 快速原型 |

---

## 面试准备

### 核心概念题

#### 1. 什么是因果推断？与相关性有什么区别？
**答案要点**:
- 相关性: X 和 Y 一起变化
- 因果性: X 导致 Y 变化
- 因果需要: 干预思维 (do-calculus)
- RCT 是因果推断的金标准

#### 2. 解释 Potential Outcomes Framework
**答案要点**:
- 每个个体有两个潜在结果: Y(0), Y(1)
- 因果效应: Y(1) - Y(0)
- 基本问题: 只能观测到一个结果
- SUTVA 假设

#### 3. 什么是混淆变量？如何处理？
**答案要点**:
- 同时影响 Treatment 和 Outcome
- 导致虚假因果关系
- 处理方法: 控制、匹配、加权、工具变量

#### 4. 解释倾向得分匹配 (PSM)
**答案要点**:
- 倾向得分: P(T=1|X)
- 降维: 多维协变量 → 一维得分
- 匹配: 得分相近的处理/对照配对
- 优点: 直观、减少偏差
- 缺点: 信息损失、需要重叠

#### 5. IPW 和 AIPW 的区别
**答案要点**:
- IPW: 逆倾向得分加权
- AIPW: 增强的 IPW，结合结果模型
- AIPW 是双重稳健的
- 只要倾向得分或结果模型正确即可

### 机器学习方法题

#### 6. S/T/X-Learner 的区别
**答案要点**:
- S-Learner: 单模型，T 作为特征
- T-Learner: 两个独立模型
- X-Learner: 交叉预测 + 加权
- X-Learner 在样本不平衡时更好

#### 7. 什么是 Double Machine Learning？
**答案要点**:
- 分离混淆控制和效应估计
- 两阶段: 残差化 + 估计
- 正交性保证渐近正态性
- Neyman 正交条件

#### 8. Causal Forest 如何工作？
**答案要点**:
- 随机森林 + 因果推断
- 分裂准则: 最大化 CATE 方差
- Honest 分裂: 分裂和估计使用不同数据
- 自动发现异质性

### 深度学习方法题

#### 9. TARNet 的架构和核心思想
**答案要点**:
- 共享表示层: 学习处理无关的特征
- 双头输出: 分别预测 Y(0) 和 Y(1)
- Factual Loss: 只在观测结果上训练
- CFR: 添加 IPM 正则化平衡表示

#### 10. DragonNet 相比 TARNet 的改进
**答案要点**:
- 添加倾向得分头
- Targeted Regularization (类 TMLE)
- 端到端学习
- 更数据高效

#### 11. CEVAE 如何处理隐变量混淆？
**答案要点**:
- VAE 框架建模隐变量
- 潜在空间捕获未观测混淆
- 生成模型思想
- 可以进行敏感性分析

### 实践/设计题

#### 12. 如何设计一个因果推断项目？
**答案要点**:
1. 明确因果问题和假设
2. 绘制因果图 (DAG)
3. 识别可识别性条件
4. 选择合适方法
5. 评估和敏感性分析

#### 13. A/B 测试中如何处理网络效应？
**答案要点**:
- 问题: SUTVA 违反
- 方法: 集群随机化、Switchback、Ego-network randomization

#### 14. 如何评估 CATE 估计的质量？
**答案要点**:
- 有 ground truth: PEHE, ATE Error
- 无 ground truth:
  - R-Loss (反事实预测)
  - Qini/Uplift 曲线
  - Policy value

### 行业案例题

#### 15. 描述一个因果推断在业务中的应用
**示例答案** (智能发券):
- 问题: 给谁发优惠券能最大化增量收益
- 方法: Uplift Modeling 估计 CATE
- 实施: 对高增益用户发券
- 评估: Qini 曲线, 增量 GMV

---

## 项目优化建议

基于对当前项目的分析，以下是系统性的优化建议：

### 1. 深度学习模块扩展

#### 当前状态
- TARNet ✅
- DragonNet ✅
- CEVAE ❌ (缺失)
- GANITE ❌ (缺失)

#### 建议添加

```python
# 1. CEVAE 实现
class CEVAE(nn.Module):
    """
    Causal Effect Variational Autoencoder
    - VAE 架构处理隐变量
    - 可处理 unobserved confounding
    """
    pass

# 2. GANITE 实现
class GANITE(nn.Module):
    """
    Generative Adversarial Nets for ITE
    - 生成反事实样本
    - 支持多处理
    """
    pass

# 3. TransTEE 实现
class TransTEE(nn.Module):
    """
    Transformer-based Treatment Effect Estimation
    - 注意力机制
    - 更好的特征交互
    """
    pass
```

### 2. 评估体系完善

#### 当前状态
- 基础 PEHE, ATE Error ✅
- Qini 曲线 ✅
- 置信区间 ❌
- Bootstrap 评估 ❌

#### 建议添加

```python
# 添加完整评估模块
class CATEEvaluator:
    def compute_metrics(self, y0_true, y1_true, y0_pred, y1_pred):
        return {
            'pehe': self.pehe(),
            'ate_error': self.ate_error(),
            'r_squared': self.r_squared(),
            'qini_auc': self.qini_auc(),
            'confidence_interval': self.bootstrap_ci()
        }
```

### 3. Benchmark 数据集集成

#### 建议添加

```python
# datasets/benchmarks.py
def load_benchmark(name: str):
    """统一加载 benchmark 数据集"""
    benchmarks = {
        'ihdp': load_ihdp,
        'jobs': load_jobs,
        'twins': load_twins,
        'criteo': load_criteo,
        'hillstrom': load_hillstrom,
    }
    return benchmarks[name]()
```

### 4. 测试覆盖

#### 当前状态
- 无单元测试 ❌

#### 建议添加

```python
# tests/test_deep_causal.py
class TestTARNet:
    def test_forward_pass(self):
        pass

    def test_training(self):
        pass

    def test_prediction(self):
        pass

    def test_on_ihdp(self):
        """在 IHDP 上测试，确保 PEHE < 阈值"""
        pass
```

### 5. 文档和教程

#### 建议添加

```
docs/
├── tutorials/
│   ├── 01_getting_started.ipynb
│   ├── 02_potential_outcomes.ipynb
│   ├── 03_propensity_score.ipynb
│   ├── 04_meta_learners.ipynb
│   ├── 05_deep_learning.ipynb
│   └── 06_real_world_case.ipynb
├── api/
│   └── reference.md
└── theory/
    └── mathematical_foundations.md
```

### 6. 模型对比实验

#### 建议添加

```python
# experiments/benchmark_comparison.py
def run_benchmark_comparison():
    """
    在所有 benchmark 上运行所有模型
    生成对比报告
    """
    models = [PSM, IPW, SLearner, TLearner, TARNet, DragonNet]
    datasets = [IHDP, Jobs, Twins]

    results = {}
    for model in models:
        for dataset in datasets:
            results[(model, dataset)] = evaluate(model, dataset)

    return generate_report(results)
```

### 7. 面试准备模块

#### 建议添加

```python
# interview_prep/
├── concepts/           # 概念题库
├── coding/            # 编程题
├── case_studies/      # 案例分析
└── mock_interview/    # 模拟面试
```

### 8. 实践项目

#### 建议添加

```
projects/
├── 01_ate_estimation/      # ATE 估计项目
├── 02_cate_prediction/     # CATE 预测项目
├── 03_uplift_ranking/      # Uplift 排序项目
└── 04_policy_optimization/ # 策略优化项目
```

### 优先级排序

| 优先级 | 任务 | 预计工作量 |
|--------|------|------------|
| P0 | 添加 CEVAE, GANITE | 高 |
| P0 | 完善测试覆盖 | 中 |
| P1 | Benchmark 数据集集成 | 低 |
| P1 | 评估体系完善 | 中 |
| P2 | 教程 Notebooks | 高 |
| P2 | 面试准备模块 | 中 |
| P3 | 模型对比实验 | 中 |

---

## 参考资源

### 官方文档

- [EconML](https://www.pywhy.org/econml/)
- [CausalML](https://causalml.readthedocs.io/)
- [DoWhy](https://www.pywhy.org/dowhy/)
- [scikit-uplift](https://www.uplift-modeling.com/)

### 技术博客

- [Towards Data Science - Causal Inference](https://towardsdatascience.com/tagged/causal-inference)
- [Netflix Tech Blog](https://netflixtechblog.com/)
- [Uber Engineering Blog](https://www.uber.com/blog/engineering/)

### GitHub 资源

- [Awesome Causality](https://github.com/rguo12/awesome-causality-algorithms)
- [Causal Inference Papers](https://github.com/fulifeng/Causal_Reading_Group)

---

*最后更新: 2026年1月*

*本文档持续更新中，欢迎贡献!*
