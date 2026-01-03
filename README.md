# Causal Inference Workbench

> Casual inference play ground

一站式因果推断学习与可视化平台，面向数据科学家和算法工程师的因果推断探索工具。

聚焦 **Model-Based** 方法，涵盖深度学习、机器学习因果模型，配合交互式可视化帮助理解算法原理。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Gradio 应用
python app.py
```

访问 `http://localhost:7860` 开始学习。

---

## 知识体系结构

```
因果推断学习路径

├── 1. FoundationLab - 基础概念实验室
│   ├── 潜在结果框架 (Potential Outcomes)
│   ├── 因果图与 DAG (Causal DAG)
│   ├── 混淆偏差可视化 (Confounding Bias)
│   └── 选择偏差可视化 (Selection Bias)
│
├── 2. TreatmentEffectLab - 处理效应估计
│   ├── ATE/ATT/ATC 估计器对比
│   ├── 倾向得分匹配 (PSM)
│   ├── 逆概率加权 (IPW/IPTW)
│   └── 双重稳健估计 (Doubly Robust)
│
├── 3. UpliftLab - 增益模型实验室
│   ├── Meta-Learners (S/T/X/R-Learner)
│   ├── Uplift Tree 可视化
│   ├── CATE 估计对比
│   └── Qini/Uplift 曲线评估
│
├── 4. DeepCausalLab - 深度因果模型
│   ├── CEVAE (因果变分自编码器)
│   ├── DragonNet 架构解析
│   ├── TARNet / CFR
│   ├── GANITE
│   └── Causal Transformer
│
├── 5. HeteroEffectLab - 异质性效应
│   ├── Causal Forest 可视化
│   ├── BART 贝叶斯加法树
│   ├── 条件平均处理效应 (CATE)
│   └── 敏感性分析
│
├── 6. ApplicationLab - 营销场景应用
│   ├── 智能发券策略优化
│   ├── A/B Testing 增强
│   ├── 用户分层干预
│   └── ROI 优化
│
└── 7. EvaluationLab - 评估与诊断
    ├── 平衡性诊断 (Balance Check)
    ├── 重叠假设检验 (Overlap)
    ├── 敏感性分析 (Sensitivity)
    └── 模型对比评估
```

---

## 项目结构

```
├── app.py                        # Gradio 应用入口
├── requirements.txt              # 依赖清单
│
├── foundation_lab/               # 基础概念模块
│   ├── __init__.py
│   ├── utils.py                  # 数据生成与工具函数
│   ├── potential_outcomes.py     # 潜在结果框架可视化
│   ├── causal_dag.py             # 因果图绘制与分析
│   ├── confounding_bias.py       # 混淆偏差演示
│   └── selection_bias.py         # 选择偏差演示
│
├── treatment_effect_lab/         # 处理效应估计模块
│   ├── __init__.py
│   ├── utils.py
│   ├── ate_estimators.py         # ATE 估计器对比
│   ├── propensity_score.py       # 倾向得分方法
│   ├── ipw.py                    # 逆概率加权
│   └── doubly_robust.py          # 双重稳健估计
│
├── uplift_lab/                   # 增益模型模块
│   ├── __init__.py
│   ├── utils.py
│   ├── meta_learners.py          # S/T/X/R Learner
│   ├── uplift_tree.py            # Uplift 决策树
│   ├── cate_comparison.py        # CATE 对比
│   └── evaluation.py             # Qini/Uplift 曲线
│
├── deep_causal_lab/              # 深度因果模型模块
│   ├── __init__.py
│   ├── utils.py
│   ├── tarnet.py                 # TARNet 模型
│   ├── dragonnet.py              # DragonNet 模型
│   ├── cevae.py                  # CEVAE 模型
│   └── causal_transformer.py     # Causal Transformer
│
├── hetero_effect_lab/            # 异质性效应模块
│   ├── __init__.py
│   ├── utils.py
│   ├── causal_forest.py          # 因果森林
│   ├── bart.py                   # BART 模型
│   └── sensitivity.py            # 敏感性分析
│
├── application_lab/              # 应用场景模块
│   ├── __init__.py
│   ├── utils.py
│   ├── coupon_optimization.py    # 智能发券
│   ├── ab_enhancement.py         # A/B 测试增强
│   └── user_targeting.py         # 用户定向
│
├── evaluation_lab/               # 评估诊断模块
│   ├── __init__.py
│   ├── utils.py
│   ├── balance_check.py          # 平衡性检查
│   ├── overlap_check.py          # 重叠假设
│   └── model_comparison.py       # 模型对比
│
├── exercises/                    # 练习题目录
│   ├── chapter1_foundation/
│   ├── chapter2_treatment_effect/
│   ├── chapter3_uplift/
│   ├── chapter4_deep_causal/
│   ├── chapter5_hetero_effect/
│   └── chapter6_application/
│
└── notebooks/                    # Jupyter Notebooks
    ├── 01_foundation.ipynb
    ├── 02_treatment_effect.ipynb
    ├── 03_uplift_modeling.ipynb
    ├── 04_deep_causal.ipynb
    ├── 05_hetero_effect.ipynb
    └── 06_application.ipynb
```

---

## 学习模块详解

### 1. FoundationLab - 基础概念实验室

| 模块 | 功能 | 交互元素 |
|------|------|----------|
| **潜在结果框架** | Y(0), Y(1) 可视化，理解反事实 | 滑动条调整处理效应大小 |
| **因果图 DAG** | 交互式 DAG 绘制，识别混淆变量 | 拖拽节点，添加边 |
| **混淆偏差** | 演示 omit variable bias | 控制混淆强度观察估计偏差 |
| **选择偏差** | Berkson's paradox 等经典案例 | 调整选择机制参数 |

**练习**:
- 识别给定 DAG 中的后门路径
- 模拟数据验证混淆偏差大小

---

### 2. TreatmentEffectLab - 处理效应估计

| 模块 | 功能 | 交互元素 |
|------|------|----------|
| **ATE 估计器** | 朴素估计 vs 调整估计对比 | 选择不同估计方法 |
| **倾向得分** | PS 分布可视化，匹配效果 | 调整匹配参数 |
| **IPW** | 权重分布，极端权重截断 | 权重截断阈值 |
| **双重稳健** | DR 估计量构建过程 | 切换结果模型 |

**练习**:
- 实现简单的倾向得分匹配
- 对比不同估计器的方差

---

### 3. UpliftLab - 增益模型实验室

| 模块 | 功能 | 交互元素 |
|------|------|----------|
| **Meta-Learners** | S/T/X/R Learner 对比 | 选择基学习器 |
| **Uplift Tree** | 树结构可视化，分裂准则 | 调整树深度 |
| **CATE 估计** | 异质性效应分布 | 特征选择 |
| **评估曲线** | Qini/Uplift Curve | 调整干预比例 |

**练习**:
- 实现 T-Learner
- 计算 AUUC (Area Under Uplift Curve)

---

### 4. DeepCausalLab - 深度因果模型

| 模块 | 功能 | 交互元素 |
|------|------|----------|
| **TARNet** | 双塔结构可视化 | 网络参数调整 |
| **DragonNet** | 端到端训练，倾向得分头 | Loss 权重调整 |
| **CEVAE** | VAE 潜变量因果图 | 潜空间可视化 |
| **Causal Transformer** | Attention 与因果效应 | 注意力图 |

**练习**:
- 实现简化版 TARNet
- DragonNet 训练与调参

---

### 5. HeteroEffectLab - 异质性效应

| 模块 | 功能 | 交互元素 |
|------|------|----------|
| **Causal Forest** | 树集成可视化 | 树数量调整 |
| **BART** | 后验分布采样 | MCMC 参数 |
| **敏感性分析** | Unmeasured confounding | 敏感性参数 |

**练习**:
- 使用 EconML 的 Causal Forest
- 解读 CATE 的不确定性

---

### 6. ApplicationLab - 营销场景应用

| 模块 | 功能 | 交互元素 |
|------|------|----------|
| **智能发券** | 优惠券发放策略优化 | 成本/收益调整 |
| **A/B 增强** | 因果推断增强 A/B | 协变量选择 |
| **用户定向** | 用户分层 + 个性化干预 | 分层阈值 |

**练习**:
- 设计发券策略的 Uplift 模型
- 计算 ROI 最优干预阈值

---

## 技术栈

| 分类 | 依赖 | 用途 |
|------|------|------|
| **核心** | `gradio` | Web 交互框架 |
| | `plotly` | 交互式可视化 |
| | `networkx` | 因果图绘制 |
| **因果推断** | `econml` | Microsoft 因果推断库 |
| | `causalml` | Uber 因果推断库 |
| | `dowhy` | Microsoft 因果推断框架 |
| **深度学习** | `pytorch` | 深度因果模型 |
| | `pytorch-lightning` | 训练框架 |
| **传统ML** | `scikit-learn` | 基础模型 |
| | `xgboost` / `lightgbm` | 树模型 |

---

## 设计理念

1. **体系化学习**: 从基础到高级，循序渐进
2. **交互式理解**: 通过可视化和滑动条建立直觉
3. **代码驱动**: 每章配套练习代码，学以致用
4. **实战导向**: 结合营销场景，解决实际问题

---

## License

MIT License
