# 深度学习因果推断论文清单

> 按学习路径和重要性整理的深度学习因果推断重要论文
>
> 最后更新: 2026-01-04

---

## 目录

1. [综述论文 (Survey Papers)](#1-综述论文)
2. [基础架构 (Foundational Architectures)](#2-基础架构)
3. [表示学习方法 (Representation Learning)](#3-表示学习方法)
4. [Transformer 架构 (Transformer-Based)](#4-transformer-架构)
5. [时序因果推断 (Temporal Causal Inference)](#5-时序因果推断)
6. [剂量-反应曲线 (Dose-Response)](#6-剂量-反应曲线)
7. [生成模型 (Generative Models)](#7-生成模型)
8. [图神经网络 (Graph Neural Networks)](#8-图神经网络)
9. [工具变量方法 (Instrumental Variables)](#9-工具变量方法)
10. [双重机器学习 (Double Machine Learning)](#10-双重机器学习)
11. [贝叶斯与不确定性量化 (Bayesian & Uncertainty)](#11-贝叶斯与不确定性量化)
12. [元学习与迁移学习 (Meta-Learning & Transfer)](#12-元学习与迁移学习)
13. [工业应用案例 (Industrial Applications)](#13-工业应用案例)
14. [基准数据集与工具 (Benchmarks & Tools)](#14-基准数据集与工具)

---

## 学习路线图

```
入门路径:
综述论文 → 基础架构 (TARNet/DragonNet) → 表示学习 (CFRNet) → 应用案例

进阶路径:
Transformer 方法 → 时序因果 → GNN 方法 → 元学习

深入路径:
工具变量 → 双重机器学习 → 贝叶斯方法 → 不确定性量化
```

---

## 1. 综述论文

从这里开始，建立全局视野。

### 1.1 必读综述

**📚 Causal Inference Meets Deep Learning: A Comprehensive Survey**
- **作者**: Licheng Jiao et al.
- **发表**: Research (Science Partner Journal), 2024
- **核心内容**:
  - 深度学习与因果推断的全面综述
  - 涵盖语音、文本、图、图像四大模态
  - 包含基准数据集和下载链接
- **适用场景**: 建立全局认知，了解领域全貌
- **链接**: [Paper](https://spj.science.org/doi/10.34133/research.0467) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39257419/)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**📚 Deep Causal Learning: Representation, Discovery and Inference**
- **作者**: 多作者
- **发表**: ACM Computing Surveys, 2024
- **核心内容**:
  - 因果学习的三大核心能力：表示、发现、推断
  - 神经网络在因果学习中的三大优势
  - 因果表示、因果发现、因果推断的深度学习方法
- **适用场景**: 理解深度学习如何解决因果学习难题
- **链接**: [ACM DL](https://dl.acm.org/doi/10.1145/3762179) | [arXiv](https://arxiv.org/pdf/2211.03374)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**📚 A Survey of Deep Causal Models and Their Industrial Applications**
- **作者**: 多作者
- **发表**: Artificial Intelligence Review, 2024
- **核心内容**:
  - 2016-2023 年约 50 个经典深度因果模型发展时间线
  - 模型分类与论文关系图谱
  - 工业应用案例
- **适用场景**: 了解模型演进脉络和工业落地
- **链接**: [Springer](https://link.springer.com/article/10.1007/s10462-024-10886-0)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**📚 A Primer on Deep Learning for Causal Inference**
- **作者**: Bernard J. Koch et al.
- **发表**: 2025 (Sage Journals)
- **核心内容**:
  - 深度学习因果推断入门教程
  - S-Learner 和 T-Learner 的神经网络实现
  - TARNet 和 DragonNet 的详细分析
- **适用场景**: 入门级教材，适合初学者
- **链接**: [Sage](https://journals.sagepub.com/doi/10.1177/00491241241234866) | [PDF](https://faculty.ist.psu.edu/vhonavar/Courses/causality/dl-causal2.pdf)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

### 1.2 专题综述

**📚 Causal Deep Learning (arXiv 2024)**
- **核心内容**: 因果推断与复杂建模的整合
- **链接**: [arXiv:2303.02186](https://arxiv.org/pdf/2303.02186)
- **推荐指数**: ⭐⭐⭐⭐

---

## 2. 基础架构

从这些论文开始动手实践。

### 2.1 开山之作

**🔥 Learning Representations for Counterfactual Inference**
- **作者**: Fredrik D. Johansson et al.
- **发表**: ICML 2016
- **核心创新**:
  - 首次将因果推断转化为领域适应问题
  - 提出 BNN (Balancing Neural Network) 和 BLR 方法
  - 学习平衡表示以减少选择偏差
- **适用场景**: 观察性数据的反事实推断
- **代码**: [GitHub (多个实现)](https://github.com/kochbj/Deep-Learning-for-Causal-Inference)
- **链接**: [PMLR](http://proceedings.mlr.press/v48/johansson16.pdf)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🔴 必读

---

**🔥 Estimating Individual Treatment Effect: Generalization Bounds and Algorithms**
- **作者**: Uri Shalit, Fredrik D. Johansson, David Sontag
- **发表**: ICML 2017 (PMLR)
- **核心创新**:
  - 提出 **TARNet** (Treatment-Agnostic Representation Network)
  - 双塔架构：共享表示层 + 两个输出头
  - 推导泛化界，理论保证平衡表示的有效性
  - 提出 **CFRNet** (Counterfactual Regression Network)
  - 使用 IPM (Integral Probability Metric) 如 Wasserstein 距离或 MMD 进行分布匹配
- **适用场景**: 个体处理效应估计 (ITE)
- **代码**: [TensorFlow实现](https://github.com/clinicalml/cfrnet) | [多个复现](https://github.com/kochbj/Deep-Learning-for-Causal-Inference)
- **链接**: [PMLR](https://proceedings.mlr.press/v70/shalit17a.html) | [arXiv](https://arxiv.org/abs/1606.03976)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🔴 必读

---

**🔥 Adapting Neural Networks for the Estimation of Treatment Effects**
- **作者**: Claudia Shi, David Blei, Victor Veitch
- **发表**: NeurIPS 2019
- **核心创新**:
  - 提出 **DragonNet** 架构
  - 在 TARNet 基础上增加倾向得分头 (propensity head)
  - 提出 **Targeted Regularization (t-reg)** 目标函数
  - 丢弃与混淆无关的协变量，提高数据效率
- **适用场景**: 有限数据下的处理效应估计
- **代码**: [TensorFlow 2.8实现](https://github.com/claudiashi57/dragonnet)
- **链接**: [NeurIPS](https://papers.nips.cc/paper/2019/hash/8fb5f8be2aa9d6c64a04e3ab9f63feee-Abstract.html) | [PDF](https://arxiv.org/pdf/1906.02120)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🔴 必读

---

### 2.2 扩展架构

**Perfect Match: A Simple Method for Learning Representations For Counterfactual Inference**
- **作者**: Patrick Schwab et al.
- **发表**: NeurIPS 2018 Workshop
- **核心创新**:
  - 支持多个处理 (multiple treatments)
  - 性能优于 BNN, TARNet, CFRNet, GANITE
- **适用场景**: 多处理场景的反事实推断
- **代码**: [GitHub](https://github.com/d909b/perfect_match)
- **链接**: [arXiv:1810.00656](https://arxiv.org/pdf/1810.00656)
- **推荐指数**: ⭐⭐⭐⭐

---

**Neural Networks with Causal Graph Constraints**
- **作者**: 多作者
- **发表**: 2024
- **核心创新**:
  - NN-CGC: 将因果图约束整合进神经网络
  - 减少虚假变量交互导致的误差
  - 对不完美因果图具有鲁棒性
- **适用场景**: 已知部分因果结构时的 HTE 估计
- **链接**: [arXiv:2404.12238](https://arxiv.org/abs/2404.12238)
- **推荐指数**: ⭐⭐⭐⭐

---

## 3. 表示学习方法

理解如何学习无偏的因果表示。

### 3.1 对抗平衡

**Adversarial Balancing-based Representation Learning for Causal Effect Inference**
- **作者**: 多作者
- **发表**: Data Mining and Knowledge Discovery, 2021
- **核心创新**:
  - ABCEI 框架：基于对抗网络的平衡表示学习
  - 同时处理选择偏差和反事实缺失问题
  - 使用对抗训练平衡处理组和对照组分布
- **适用场景**: 观察性数据中的 CATE 估计
- **代码**: [GitHub](https://github.com/octeufer/Adversarial-Balancing-based-representation-learning-for-Causal-Effect-Inference)
- **链接**: [Springer](https://link.springer.com/article/10.1007/s10618-021-00759-3)
- **推荐指数**: ⭐⭐⭐⭐

---

**Balancing Deep Covariate Representations (DeepMatch)**
- **作者**: Nathan Kallus
- **发表**: ICML 2020 (PMLR)
- **核心创新**:
  - 使用对抗训练平衡协变量
  - 判别性差异度量 (discriminative discrepancy metric)
  - 类似 GAN 的交替梯度方法
- **适用场景**: 高维协变量的因果推断
- **链接**: [PMLR](http://proceedings.mlr.press/v119/kallus20a.html)
- **推荐指数**: ⭐⭐⭐⭐

---

### 3.2 领域适应

**Counterfactual Domain Adversarial Training**
- **作者**: 多作者
- **发表**: IEEE Conference 2018
- **核心创新**:
  - 利用 DANN (Domain Adversarial Neural Networks) 进行因果推断
  - 使用差异距离度量进行对抗训练
- **适用场景**: 领域迁移下的因果推断
- **链接**: [IEEE Xplore](https://ieeexplore.ieee.org/document/8253217/)
- **推荐指数**: ⭐⭐⭐

---

**Estimating Conditional Average Treatment Effects via Sufficient Representation Learning**
- **作者**: 多作者
- **发表**: arXiv 2024
- **核心创新**:
  - 通过充分表示学习估计 CATE
  - 表示收敛性保证 CATE 估计一致性
- **适用场景**: CATE 估计与降维
- **链接**: [arXiv:2408.17053](https://arxiv.org/html/2408.17053)
- **推荐指数**: ⭐⭐⭐

---

### 3.3 去噪与扩散

**Denoising for Balanced Representation (DBRT)**
- **作者**: 多作者
- **发表**: Knowledge-Based Systems, 2024
- **核心创新**:
  - 从根本原因消除选择偏差
  - 扩散模型启发的去噪方法
  - 实现平衡表示以准确估计 ITE
- **适用场景**: 选择偏差严重的场景
- **链接**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705124012814)
- **推荐指数**: ⭐⭐⭐

---

## 4. Transformer 架构

最新的注意力机制在因果推断中的应用。

### 4.1 反事实预测

**🔥 Causal Transformer for Estimating Counterfactual Outcomes**
- **作者**: Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel
- **发表**: ICML 2022
- **核心创新**:
  - 专为捕捉时变混淆因子的复杂长程依赖而设计
  - 提出反事实领域混淆损失 (CDC loss)
  - 三个子网络：协变量、历史处理、历史结果
  - 子网络间交叉注意力机制
- **适用场景**: 时间序列数据的反事实预测
- **代码**: [GitHub](https://github.com/Valentyn1997/CausalTransformer)
- **链接**: [PMLR](https://proceedings.mlr.press/v162/melnychuk22a/melnychuk22a.pdf)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🟠 重要

---

**DAG-aware Transformer for Causal Effect Estimation**
- **作者**: Manqing Liu et al.
- **发表**: 2024
- **核心创新**:
  - 将因果 DAG 直接整合进注意力机制
  - 灵活估计 ATE 和 CATE
  - 准确建模潜在因果结构
- **适用场景**: 已知因果图的处理效应估计
- **链接**: [arXiv:2410.10044](https://arxiv.org/html/2410.10044v1) | [OpenReview](https://openreview.net/pdf?id=sG6tdKozS7)
- **推荐指数**: ⭐⭐⭐⭐

---

**Transformer-Variational Autoencoder for ITE (TCE-VAE)**
- **作者**: 多作者
- **发表**: Applied Intelligence, 2025
- **核心创新**:
  - 整合 Transformer 编码器-解码器与 VAE
  - 自注意力机制捕捉复杂依赖和交互
  - 直接估计因果效应
- **适用场景**: 复杂特征交互的 ITE 估计
- **链接**: [Springer](https://link.springer.com/article/10.1007/s10489-025-06738-1) | [ResearchGate](https://www.researchgate.net/publication/393538036)
- **推荐指数**: ⭐⭐⭐⭐

---

**CETransformer: Casual Effect Estimation via Transformer Based Representation Learning**
- **作者**: 多作者
- **发表**: 2021
- **核心创新**:
  - 自监督 Transformer 利用协变量间相关性
  - 自注意力机制挖掘特征关系
  - 对抗网络平衡处理组和对照组分布
- **适用场景**: 自监督学习场景的因果效应估计
- **链接**: [arXiv:2107.08714](https://arxiv.org/abs/2107.08714)
- **推荐指数**: ⭐⭐⭐

---

### 4.2 因果发现

**CausalFormer: An Interpretable Transformer for Temporal Causal Discovery**
- **作者**: 多作者
- **发表**: IEEE TKDE, 2024
- **核心创新**:
  - 因果感知 Transformer (causality-aware transformer)
  - 多核因果卷积 (multi-kernel causal convolution)
  - 基于分解的因果检测器
  - 回归相关性传播 (regression relevance propagation)
- **适用场景**: 时间序列因果发现
- **代码**: [预计有开源实现]
- **链接**: [IEEE](https://dl.acm.org/doi/10.1109/TKDE.2024.3484461) | [arXiv:2406.16708](https://arxiv.org/abs/2406.16708)
- **推荐指数**: ⭐⭐⭐⭐

---

**Teaching Transformers Causal Reasoning through Axiomatic Training**
- **作者**: 多作者
- **发表**: ICML 2025
- **核心创新**:
  - 公理化训练方案 (axiomatic training)
  - 从多个因果公理演示中学习
  - 小图上学习传递性公理，泛化到大图
- **适用场景**: 文本 AI 系统的因果推理
- **链接**: [ICML 2025](https://icml.cc/virtual/2025/poster/46158)
- **推荐指数**: ⭐⭐⭐

---

### 4.3 零样本推断

**Causal Inference with Attention (CInA)**
- **作者**: 多作者
- **发表**: ICML 2024
- **核心创新**:
  - 利用多个无标签数据集进行自监督因果学习
  - 零样本因果推断 (zero-shot causal inference)
  - 最优协变量平衡与自注意力的对偶连接
  - Transformer 架构最后一层实现零样本推断
- **适用场景**: 新任务的零样本因果推断
- **链接**: [ICML 2024](https://icml.cc/virtual/2024/session/35594)
- **推荐指数**: ⭐⭐⭐⭐

---

## 5. 时序因果推断

处理时间序列数据中的因果关系。

### 5.1 循环神经网络

**🔥 Counterfactual Recurrent Network (CRN)**
- **作者**: Ioana Bica et al.
- **发表**: NeurIPS 2020
- **核心创新**:
  - 序列到序列模型估计随时间变化的处理效应
  - RNN 跟踪上下文协变量信息
  - 处理时变混淆偏差
- **适用场景**: 时间序列观察数据的处理效应估计
- **代码**: [GitHub](https://github.com/ioanabica/Counterfactual-Recurrent-Network)
- **链接**: [NeurIPS](https://papers.nips.cc/paper/2020/hash/0d0871f0806eae32d30983b62252da50-Abstract.html)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🟠 重要

---

**Disentangled Counterfactual Recurrent Network (DCRN)**
- **作者**: 多作者
- **发表**: 待确认
- **核心创新**:
  - 将患者历史解耦为三个潜在因子
  - 处理因子、结果因子、混淆因子
  - 序列到序列架构
- **适用场景**: 医疗时序数据的处理效应估计
- **推荐指数**: ⭐⭐⭐⭐

---

### 5.2 合成对照

**SyncTwin: Transparent Treatment Effect Estimation Under Temporal Confounding**
- **作者**: 多作者
- **发表**: 会议待确认
- **核心创新**:
  - 合成孪生 (synthetic twin) 方法
  - 处理不规则采样数据
  - Seq2Seq 学习时间协变量表示
  - 优化方法构建合成孪生权重
- **适用场景**: 单次二元处理的时序数据
- **链接**: [Semantic Scholar](https://www.semanticscholar.org/paper/SYNCTWIN:-TRANSPARENT-TREATMENT-EFFECT-ESTIMATION/34c6979affab7600ab49d7009450b6bac6ae14d4)
- **推荐指数**: ⭐⭐⭐⭐

---

### 5.3 对比学习

**Causal Contrastive Learning for Counterfactual Regression Over Time**
- **作者**: 多作者
- **发表**: NeurIPS 2024
- **核心创新**:
  - 结合 RNN 与对比预测编码 (CPC)
  - 对比损失正则化，互信息指导
  - 首次将 CPC 应用于因果推断
  - 优先考虑计算效率，无需复杂 Transformer
- **适用场景**: 高效时序反事实回归
- **链接**: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/02cef2ae63853724eb99e70721d3bc65-Paper-Conference.pdf)
- **推荐指数**: ⭐⭐⭐⭐

---

## 6. 剂量-反应曲线

连续处理变量的因果效应估计。

### 6.1 变系数网络

**🔥 VCNet: Varying Coefficient Neural Network**
- **作者**: Xinkun Nie et al.
- **发表**: 2021
- **核心创新**:
  - 变系数神经网络，增强模型表达能力
  - 函数目标正则化 (functional targeted regularization)
  - 使用 B 样条建模处理水平变化
  - 倾向得分估计器强制平衡表示
  - 保持 ADRF (Average Dose-Response Function) 连续性
- **适用场景**: 连续处理变量的剂量-反应曲线估计
- **代码**: [GitHub](https://github.com/lushleaf/varying-coefficient-net-with-functional-tr)
- **链接**: 论文链接待补充
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🟠 重要

---

### 6.2 生成对抗网络

**SCIGAN: Generative Adversarial Network for Continuous Interventions**
- **作者**: Ioana Bica et al.
- **发表**: 2020
- **核心创新**:
  - 分层生成对抗网络
  - 在 GAN 框架下处理连续值干预
  - 提供因果估计的理论验证
- **缺点**: 需要数千训练样本
- **适用场景**: 连续干预的因果效应估计
- **代码**: [GitHub](https://github.com/ioanabica/SCIGAN)
- **推荐指数**: ⭐⭐⭐⭐

---

### 6.3 最新进展

**TransTEE: Transformer for Treatment Effect Estimation**
- **作者**: Zhang et al.
- **发表**: 2022
- **核心创新**:
  - 结合 SCIGAN 的分层判别器与 VCNet 的变系数结构
  - Transformer 多头注意力机制
  - 统一框架扩展到离散、连续、结构化、剂量相关处理
- **适用场景**: 多种处理类型的统一框架
- **推荐指数**: ⭐⭐⭐⭐

---

**ADMIT: Adaptive Dose-Response Modeling with IPM**
- **作者**: Wang et al.
- **发表**: 2024
- **核心创新**:
  - 基于 DRNet 和 VCNet 构建
  - 重加权方案 (re-weighting scheme)
  - IPM 估计反事实损失上界
  - 理论和实验证据支持
- **适用场景**: 剂量-反应函数估计
- **推荐指数**: ⭐⭐⭐

---

**Contrastive Balancing Representation Learning for Dose-Response**
- **作者**: 多作者
- **发表**: 2024
- **核心创新**:
  - 新型对比正则化网络
  - 同时满足平衡和预后表示条件
  - 无偏异质剂量-反应曲线估计
- **适用场景**: 异质剂量-反应曲线估计
- **链接**: [arXiv:2403.14232](https://arxiv.org/html/2403.14232)
- **推荐指数**: ⭐⭐⭐⭐

---

**Continuous Treatment Effect Estimation using Gradient Interpolation**
- **作者**: Nagalapatti et al.
- **发表**: AAAI 2024
- **核心创新**:
  - 梯度插值和核平滑
  - 处理连续处理效应估计
- **链接**: AAAI 2024 Conference
- **推荐指数**: ⭐⭐⭐

---

## 7. 生成模型

使用 VAE 和 GAN 进行因果推断。

### 7.1 变分自编码器

**🔥 CEVAE: Causal Effect Variational Autoencoder**
- **作者**: Christos Louizos et al.
- **发表**: NeurIPS 2017
- **核心创新**:
  - 使用 VAE 结构估计个体处理效应
  - 推断网络 + 模型网络同时估计潜在空间和因果效应
  - 处理隐藏混淆因子 (hidden confounders)
  - 基于 Pearl 后门准则建模噪声代理变量
- **适用场景**: 存在未观测混淆因子的场景
- **代码**: [GitHub (AMLab-Amsterdam)](https://github.com/AMLab-Amsterdam/CEVAE) | [Pyro](https://docs.pyro.ai/en/dev/contrib.cevae.html)
- **链接**: [arXiv:1705.08821](https://arxiv.org/pdf/1705.08821)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🟠 重要

---

**TEDVAE: Treatment Effect Disentangled VAE**
- **作者**: Yao et al.
- **发表**: 2018
- **核心创新**:
  - 解耦表示学习
  - 将潜在因子分为三组：预测处理、结果、或两者
  - 改进 CEVAE 的准确性
  - 支持连续处理变量
- **适用场景**: 解耦因果因子的 ITE 估计
- **推荐指数**: ⭐⭐⭐⭐

---

**UTVAE: Uniform Treatment VAE**
- **作者**: 多作者
- **发表**: 2021
- **核心创新**:
  - 使用重要性采样训练均匀处理分布
  - 缓解测试时的分布偏移
  - 优于观察性处理分布的 CEVAE
- **适用场景**: 缓解分布偏移的因果推断
- **链接**: [arXiv:2111.08656](https://arxiv.org/abs/2111.08656)
- **推荐指数**: ⭐⭐⭐

---

**CausalVAE**
- **作者**: 多作者
- **发表**: 待确认
- **核心创新**:
  - 整合线性结构因果模型 (SCM) 与 VAE
  - 利用已知因果结构生成反事实
- **适用场景**: 已知因果结构的反事实生成
- **推荐指数**: ⭐⭐⭐

---

**TECE-VAE: Task Embedding-based Causal Effect VAE**
- **作者**: 多作者
- **发表**: 待确认
- **核心创新**:
  - 通过任务嵌入扩展 CEVAE
  - 支持多处理场景
  - 编码器-解码器架构
- **适用场景**: 多处理的观察性数据 ITE 估计
- **推荐指数**: ⭐⭐⭐

---

### 7.2 生成对抗网络

**🔥 GANITE: Generative Adversarial Nets for ITE**
- **作者**: Jinsung Yoon et al.
- **发表**: ICLR 2018
- **核心创新**:
  - 基于 GAN 框架推断个体处理效应
  - 反事实生成器 G：基于 X, t, y 生成潜在结果向量
  - ITE 生成器 I：基于 X 生成潜在结果
  - 两个判别器提升训练性能
- **适用场景**: 个体处理效应推断
- **代码**: [GitHub (多个实现)](https://github.com/topics/ganite)
- **链接**: [OpenReview](https://openreview.net/forum?id=ByKWUeWA-)
- **推荐指数**: ⭐⭐⭐⭐
- **学习优先级**: 🟡 推荐

---

## 8. 图神经网络

利用图结构进行因果推断。

### 8.1 处理效应估计

**🔥 Neural Networks with Causal Graph Constraints (NN-CGC)**
- **作者**: 多作者
- **发表**: 2024
- **核心创新**:
  - 将因果信息整合进 HTE 估计
  - 归纳偏置减少虚假变量交互误差
  - 可应用于其他基于表示的模型
  - 对不完美因果图具有鲁棒性
  - 达到 SOTA 结果
- **适用场景**: 利用已知或部分因果图的处理效应估计
- **链接**: [arXiv:2404.12238](https://arxiv.org/html/2404.12238v1) | [arXiv](https://arxiv.org/abs/2404.12238)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🟠 重要

---

**Graph Neural Networks for Treatment Effect Prediction**
- **作者**: 多作者
- **发表**: 2024
- **核心创新**:
  - 将问题视为节点回归，标记实例受限
  - 双模型神经架构
  - 测试不同消息传递层进行编码
  - 结合获取函数指导训练集创建（极低实验预算）
- **适用场景**: 图数据的处理效应预测
- **链接**: [arXiv:2403.19289](https://arxiv.org/html/2403.19289v1)
- **推荐指数**: ⭐⭐⭐⭐

---

**Causal Effect Estimation on Hierarchical Spatial Graph Data**
- **作者**: 多作者
- **发表**: KDD 2023
- **核心创新**:
  - SINet: 空间干预神经网络
  - 利用空间图的分层结构
  - 学习协变量和处理的丰富表示
  - 预测时间序列结果的处理效应
- **适用场景**: 分层空间图数据的因果效应估计
- **链接**: [ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599269)
- **推荐指数**: ⭐⭐⭐⭐

---

### 8.2 因果发现

**Exploring Causal Learning Through Graph Neural Networks: An In-Depth Review**
- **作者**: Job et al.
- **发表**: WIREs Data Mining and Knowledge Discovery, 2025
- **核心内容**:
  - GNN 在因果学习中的综合综述
  - 因果发现和因果推断的 GNN 方法
  - 混淆因子的识别与处理
- **链接**: [Wiley](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70024) | [arXiv:2311.14994](https://arxiv.org/html/2311.14994)
- **推荐指数**: ⭐⭐⭐⭐

---

**Causal GNN for Mining Stable Disease Biomarkers**
- **作者**: 多作者
- **发表**: 2025
- **核心创新**:
  - Causal-GNN 方法整合因果推断与多层 GNN
  - 因果效应估计识别稳定生物标志物
  - 基于 GNN 的倾向得分机制，利用跨基因调控网络
- **适用场景**: 生物医学中的稳定标志物发现
- **链接**: [arXiv:2511.13295](https://arxiv.org/html/2511.13295v1)
- **推荐指数**: ⭐⭐⭐

---

**A Graph Neural Network Framework for Causal Inference in Brain Networks**
- **作者**: 多作者
- **发表**: Scientific Reports, 2021
- **核心创新**:
  - GNN 框架基于结构解剖布局描述功能交互
  - 处理图结构的时空信号
  - 结合 DTI 结构信息与时间神经活动
  - 数据驱动的脑区域动态交互发现
- **适用场景**: 神经科学中的因果连接强度分析
- **链接**: [Nature](https://www.nature.com/articles/s41598-021-87411-8)
- **推荐指数**: ⭐⭐⭐

---

### 8.3 资源

**Awesome Graph Causal Learning**
- **内容**: 图因果学习材料清单
- **链接**: [GitHub](https://github.com/TimeLovercc/Awesome-Graph-Causal-Learning)
- **推荐指数**: ⭐⭐⭐

---

## 9. 工具变量方法

处理未观测混淆和内生性问题。

### 9.1 深度工具变量

**🔥 Deep IV: A Flexible Approach for Counterfactual Prediction**
- **作者**: Jason Hartford, Greg Lewis, Kevin Leyton-Brown, Matt Taddy
- **发表**: ICML 2017
- **核心创新**:
  - 首次将深度学习与工具变量结合
  - 两阶段方法：第一阶段处理预测网络，第二阶段结果网络
  - 第二阶段损失函数涉及条件处理分布积分
  - 学习非线性因果关系，无需同质性和线性假设
  - 优于 2SLS (Two-Stage Least Squares)
- **适用场景**: 存在未观测混淆的观察性数据
- **代码**: [GitHub (多个实现)](https://github.com/jhartford/DeepIV)
- **链接**: [PMLR](https://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🟠 重要

---

### 9.2 扩展与改进

**DeLIVR: Deep Learning Approach to IV Regression**
- **作者**: 多作者
- **发表**: PubMed, 2023
- **核心创新**:
  - 克服 DeepIV 的缺点（慢且不稳定）
  - 估计相关但不同的目标函数
  - 包含假设检验框架
  - 支持核方法、级数方法、深度神经网络
  - 用于非线性因果效应测试
- **适用场景**: TWAS (转录组关联研究) 中的非线性因果效应
- **链接**: [PubMed](https://pubmed.ncbi.nlm.nih.gov/36610078/)
- **推荐指数**: ⭐⭐⭐⭐

---

**DeepGMM: Generalized Method of Moments**
- **作者**: 多作者
- **发表**: 待确认
- **核心创新**:
  - 基于广义矩方法 (GMM)
  - 两阶段使用神经网络学习非线性效应
  - 使用处理和 IV 的非线性函数
- **适用场景**: 非参数 IV 回归
- **推荐指数**: ⭐⭐⭐

---

### 9.3 综述

**Instrumental Variables in Causal Inference and Machine Learning: A Survey**
- **作者**: 多作者
- **发表**: ACM Computing Surveys, 2024
- **核心内容**:
  - IV 在因果推断和机器学习中的综合综述
  - 处理未观测混淆影响处理和结果变量
  - 三个关键研究领域：2SLS 回归、控制函数方法、IV 学习方法进展
  - 涵盖经典和最新机器学习研究
- **链接**: [ACM DL](https://dlnext.acm.org/doi/abs/10.1145/3735969)
- **推荐指数**: ⭐⭐⭐⭐

---

**Machine Learning Instrument Variables for Causal Inference**
- **作者**: 多作者
- **发表**: Wharton 工作论文
- **链接**: [PDF](https://marketing.wharton.upenn.edu/wp-content/uploads/2021/09/09.29.2021-Singh-Amandeep-PAPER2-mliv.pdf)
- **推荐指数**: ⭐⭐⭐

---

## 10. 双重机器学习

利用正交矩函数进行去偏估计。

### 10.1 基础理论

**🔥 Double/Debiased Machine Learning for Treatment and Structural Parameters**
- **作者**: Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins
- **发表**: The Econometrics Journal, 2018
- **核心创新**:
  - 使用正交矩函数 (Neyman-orthogonal moments) 去偏 ML 估计
  - 样本分割和交叉拟合 (cross-fitting) 缓解过拟合偏差
  - 允许使用各种现代 ML 方法（随机森林、lasso、ridge、深度神经网络、boosting 等）
  - 保证有效的 root-n 一致推断
  - 提供点估计的正态分布和置信区间
- **适用场景**: 高维设置下的因果和结构参数估计
- **代码**: [DoubleML Python包](https://github.com/DoubleML/doubleml-for-py) | [R包](https://github.com/DoubleML/doubleml-for-r)
- **链接**: [Oxford Academic](https://academic.oup.com/ectj/article/21/1/C1/5056401) | [MIT PDF](https://economics.mit.edu/sites/default/files/2022-08/2017.06%20Double%20Debiased%20Machine%20Learning%20for%20Treat.pdf)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🔴 必读

---

### 10.2 扩展应用

**Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments**
- **作者**: 多作者
- **发表**: Journal of Business & Economic Statistics, 2025
- **核心创新**:
  - 连续处理变量的双稳健推断方法
  - 无混淆假设下
  - 非参数或高维 nuisance 函数
  - 提供核方法、级数方法、深度神经网络的充分低层条件
- **适用场景**: 连续处理的因果效应推断
- **链接**: [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/07350015.2025.2505487) | [arXiv:2004.03036](https://arxiv.org/abs/2004.03036)
- **推荐指数**: ⭐⭐⭐⭐

---

**Double/Debiased Machine Learning for Logistic Partially Linear Model**
- **作者**: 多作者
- **发表**: PMC, 2024
- **核心创新**:
  - DML 用于逻辑部分线性模型
  - 扩展到分类结果变量
- **链接**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10786638/)
- **推荐指数**: ⭐⭐⭐

---

### 10.3 教程

**An Introduction to Double/Debiased Machine Learning**
- **作者**: 多作者
- **发表**: arXiv 2025
- **核心内容**:
  - DML 方法的入门介绍
  - 理论基础和实践指导
- **链接**: [arXiv:2504.08324](https://arxiv.org/abs/2504.08324)
- **推荐指数**: ⭐⭐⭐⭐

---

**DoubleML Documentation**
- **内容**: 官方文档，详细介绍 DML 基础、正交矩、交叉拟合
- **链接**: [Docs](https://docs.doubleml.org/stable/guide/basics.html)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

## 11. 贝叶斯与不确定性量化

提供置信区间和不确定性估计。

### 11.1 贝叶斯神经网络

**Bayesian Neural Controlled Differential Equations (BNCDE)**
- **作者**: 多作者
- **发表**: 2023
- **核心创新**:
  - 连续时间观察数据的处理效应估计
  - 贝叶斯不确定性量化
  - 神经控制微分方程 + 神经随机微分方程的耦合系统
  - 易处理的变分贝叶斯推断
  - 对医疗决策至关重要的不确定性估计
- **适用场景**: 连续时间医疗数据的因果推断
- **链接**: [arXiv:2310.17463](https://arxiv.org/html/2310.17463v2)
- **推荐指数**: ⭐⭐⭐⭐

---

**Foundation Models for Causal Inference via Bayesian Neural Networks**
- **作者**: 多作者
- **发表**: OpenReview
- **核心创新**:
  - 使用 BNN 提供学习算法
  - 利用 SCM 模拟干预数据进行贝叶斯推断
  - 通过上下文学习 (in-context learning) 进行推断，无需额外训练
  - 贝叶斯性质提供原则性不确定性量化
  - 检测处理重叠不佳的情况
- **适用场景**: 新数据集的零样本因果推断
- **链接**: [OpenReview](https://openreview.net/pdf?id=d2L1ndOKjq)
- **推荐指数**: ⭐⭐⭐⭐

---

### 11.2 贝叶斯 TMLE

**Bayesian Implementation of Targeted Maximum Likelihood Estimation (TMLE)**
- **作者**: 多作者
- **发表**: 2025
- **核心创新**:
  - TMLE 的贝叶斯实现
  - 基于样本的概率分布进行不确定性量化
  - 训练三个模型：结果模型、倾向模型、波动模型
  - 调整结果预测以获得无偏因果效应估计
- **适用场景**: 因果效应的不确定性量化
- **链接**: [arXiv:2507.15909](https://arxiv.org/html/2507.15909)
- **推荐指数**: ⭐⭐⭐

---

### 11.3 联邦学习

**Bayesian Federated Causal Inference**
- **作者**: 多作者
- **发表**: Journal of Intelligent Manufacturing, 2025
- **核心创新**:
  - xFBCI 框架
  - 完整后验推断和不确定性估计
  - 制造业应用
- **链接**: [Springer](https://link.springer.com/article/10.1007/s10845-025-02665-7)
- **推荐指数**: ⭐⭐⭐

---

### 11.4 综述

**A Practical Introduction to Bayesian Estimation of Causal Effects**
- **作者**: 多作者
- **发表**: PMC, 2021
- **核心内容**:
  - 贝叶斯因果效应估计的实用介绍
  - 参数和非参数方法
  - 完整后验推断
  - 先验引导正则化和稀疏性
- **链接**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8640942/)
- **推荐指数**: ⭐⭐⭐⭐

---

## 12. 元学习与迁移学习

跨任务和领域的因果推断。

### 12.1 零样本与少样本

**Zero-Shot Causal Learning (CaML)**
- **作者**: Hamed Nilforoshan, Michael Moor, Yusuf Roohani
- **发表**: NeurIPS 2023
- **核心创新**:
  - 训练单个元模型融合干预信息与个体特征
  - 预测新干预的因果效应，无需样本级训练数据
  - 例如新发现药物的效应预测
  - 将 CATE 估计表述为元学习问题
  - 每个任务对应唯一干预的 CATE 估计
- **适用场景**: 新干预的零样本因果效应预测
- **链接**: [PDF](https://cs.stanford.edu/people/jure/pubs/zero-neurips23.pdf)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🟠 重要

---

**MetaCI: Meta-Learning for Causal Inference in Heterogeneous Population**
- **作者**: Sharma, Gupta et al.
- **发表**: 待确认
- **核心创新**:
  - 采用元学习范式处理异质人群
  - 解决训练和测试阶段的分布偏移
  - 处理反事实问题，数据来自多个同质子组
- **适用场景**: 异质人群的因果推断
- **链接**: [Semantic Scholar](https://www.semanticscholar.org/paper/MetaCI:-Meta-Learning-for-Causal-Inference-in-a-Sharma-Gupta/bbcebbe3295ebd9cfada36cff91f46697dc78934)
- **推荐指数**: ⭐⭐⭐⭐

---

### 12.2 迁移学习

**Advantages and Limitations of Transfer Learning for ITE**
- **作者**: 多作者
- **发表**: arXiv 2024
- **核心创新**:
  - ITE 迁移学习的理论和实践
  - 使用 TARNet 进行迁移学习
  - 下界：性能受（不可观测的）反事实误差限制
  - 泛化界：源和目标分布差异足够小时迁移有效
  - **CITA 指标** (Causal Inference Task Affinity)：捕捉源和目标数据集相似性
  - 判断源数据集是否适合迁移到目标数据集
- **适用场景**: 跨数据集的因果推断迁移
- **链接**: [arXiv:2512.16489](https://arxiv.org/html/2512.16489)
- **推荐指数**: ⭐⭐⭐⭐

---

### 12.3 元学习器

**Meta-Learners for Estimating Heterogeneous Treatment Effects**
- **作者**: Sören R. Künzel et al.
- **发表**: PNAS, 2019
- **核心创新**:
  - 提出元学习器概念：S/T/X/R-Learner
  - 将 CATE 估计分解为多个子问题
  - 每个子问题可用任何监督学习方法解决
- **适用场景**: 灵活的 HTE 估计
- **代码**: [多个实现]
- **链接**: [arXiv:1706.03461](https://arxiv.org/pdf/1706.03461) | [PNAS](https://www.pnas.org/doi/10.1073/pnas.1804597116)
- **推荐指数**: ⭐⭐⭐⭐⭐
- **学习优先级**: 🔴 必读

---

**Meta-Learning for HTE Estimation with Closed-Form Solvers**
- **作者**: 多作者
- **发表**: Machine Learning, 2024
- **核心创新**:
  - 从少量观察数据估计 CATE 的元学习方法
  - 从多个任务中学习如何估计 CATE
  - 基于元学习器框架分解问题
  - 闭式求解器
- **适用场景**: 少样本 CATE 估计
- **链接**: [Springer](https://link.springer.com/article/10.1007/s10994-024-06546-7)
- **推荐指数**: ⭐⭐⭐⭐

---

### 12.4 教程

**21 - Meta Learners — Causal Inference for the Brave and True**
- **内容**: 元学习器的实用教程
- **链接**: [在线书籍](https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**A Tutorial Introduction to HTE Estimation with Meta-learners**
- **作者**: 多作者
- **发表**: PMC, 2024
- **链接**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11379759/)
- **推荐指数**: ⭐⭐⭐⭐

---

## 13. 工业应用案例

真实世界的因果推断应用。

### 13.1 Uber

**🏢 Uber CausalML**
- **机构**: Uber
- **项目**: CausalML 开源库
- **核心内容**:
  - Uplift 建模和因果推断的 ML 算法套件
  - 标准接口估计 CATE/ITE
  - 实验或观察数据
  - 无需对模型形式强假设
- **应用场景**: 用户干预、营销优化
- **代码**: [GitHub - uber/causalml](https://github.com/uber/causalml) ⭐ 5k+
- **链接**: [Uber Blog](https://www.uber.com/blog/causal-inference-at-uber/)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**🏢 Practical Marketplace Optimization at Uber Using Causally-Informed ML**
- **作者**: Uber 团队
- **发表**: KDD 2024 Workshop
- **核心内容**:
  - 2023 Q3 Uber 移动业务 179 亿美元总预订额
  - 使用因果知识的 ML 进行市场优化
  - 跨地区和杠杆类型分配预算
  - 优化业务目标
- **链接**: [arXiv:2407.19078](https://arxiv.org/html/2407.19078v1)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

### 13.2 Microsoft

**🏢 Microsoft EconML**
- **机构**: Microsoft Research - ALICE 项目
- **项目**: EconML 开源库
- **核心内容**:
  - 从观察数据估计异质处理效应
  - 结合 SOTA ML 技术与计量经济学
  - 自动化复杂因果推断问题
  - 支持随机森林、boosting、lasso、神经网络
  - 保持因果解释性和有效置信区间
  - 包含 **DeepIV** 估计器
- **方法**: Double ML, Causal Forests, DeepIV, Doubly Robust Learning, Dynamic DML
- **代码**: [GitHub - py-why/EconML](https://github.com/py-why/EconML) ⭐ 3.8k+
- **链接**: [Microsoft Research](https://www.microsoft.com/en-us/research/project/econml/)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**🏢 Microsoft Causica**
- **机构**: Microsoft Research
- **项目**: Causica 深度学习库
- **核心内容**:
  - 端到端因果推断深度学习库
  - 包含因果发现和推断
  - **DECI** (Deep End-to-end Causal Inference)
  - 加性噪声结构方程模型 (ANM-SEM)
  - 灵活神经网络捕捉变量间函数关系
  - Gaussian 或 spline-flow 噪声模型
- **代码**: [GitHub - microsoft/causica](https://github.com/microsoft/causica)
- **推荐指数**: ⭐⭐⭐⭐

---

### 13.3 KDD 教程与研讨会

**🎓 EconML/CausalML KDD 2021 Tutorial**
- **标题**: Causal Inference and Machine Learning in Practice
- **机构**: Microsoft, TripAdvisor, Uber
- **核心内容**:
  - 条件处理效应估计器：meta-learners、tree-based 算法
  - 模型验证和敏感性分析
  - 优化算法：policy learner、cost optimization
  - 工业用例演示
- **应用案例**:
  - 旅游网站会员计划因果效应
  - 多离散处理的联合估计
  - Doubly Robust Learner 模型
- **链接**: [KDD 2021 Tutorial](https://causal-machine-learning.github.io/kdd2021-tutorial/) | [ACM DL](https://dl.acm.org/doi/10.1145/3447548.3470792)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**🎓 KDD 2024 Workshop - Causal Inference and ML in Practice**
- **链接**: [KDD 2024 Workshop](https://causal-machine-learning.github.io/kdd2024-workshop/)
- **推荐指数**: ⭐⭐⭐⭐

---

### 13.4 Netflix

**🏢 Netflix 应用**
- **人物**: Jeong-Yoon Lee (CausalML 贡献者)
- **应用**: Netflix 推荐算法团队
- **方法**: 逆概率加权、meta-learners、switchback、工具变量
- **推荐指数**: ⭐⭐⭐⭐

---

### 13.5 综合案例

**Causal Machine Learning for Predicting Treatment Outcomes**
- **发表**: Nature Medicine, 2024
- **核心内容**:
  - 因果 ML 预测处理结果（疗效和毒性）
  - 支持药物评估和安全性
  - 个体化处理效应估计，支持个性化临床决策
- **链接**: [Nature Medicine](https://www.nature.com/articles/s41591-024-02902-1)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

## 14. 基准数据集与工具

评估和对比因果推断方法。

### 14.1 标准数据集

**📊 Treatment Effect Estimation Benchmarks**
- **包含数据集**: IHDP, Jobs, Twins, News
- **链接**: [IEEE DataPort](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**📊 IHDP (Infant Health and Development Program)**
- **来源**: 婴儿健康发展项目临床试验 (1985)
- **目标**: 预测专业儿童护理对婴儿认知测试分数的效应
- **处理**: Hill (2011) 系统性移除处理组中非白人母亲的孩子
- **用途**: 因果推断基准
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**📊 Twins**
- **来源**: 美国 1989-1991 年双胞胎出生数据
- **目标**: 预测较高体重对死亡率的效应
- **处理**: Louizos et al. (2017) 创建半合成数据集，使用孕期作为混淆因子
- **用途**: 因果推断基准
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**📊 Jobs**
- **来源**: 国家支持工作项目 + 收入动态面板研究
- **目标**: 预测职业培训对就业状态的效应
- **用途**: 因果推断基准
- **推荐指数**: ⭐⭐⭐⭐

---

**📊 ACIC Benchmark**
- **来源**: ACIC (Atlantic Causal Inference Conference)
- **用途**: 学习因果效应的标准数据集
- **推荐指数**: ⭐⭐⭐⭐

---

### 14.2 实现与工具

**🛠️ Vector Institute's Causal Inference Laboratory**
- **功能**: 使用 AutoML 估计每个 nuisance 模型的最佳模型
- **数据集**: Jobs, Twins
- **代码**: [GitHub](https://github.com/VectorInstitute/Causal_Inference_Laboratory)
- **推荐指数**: ⭐⭐⭐⭐

---

**🛠️ RealCause**
- **作者**: Brady Neal
- **功能**: 真实基准，通过拟合生成模型到假设因果结构的数据
- **支持数据集**: twins, ihdp, lbidd
- **代码**: [GitHub - bradyneal/realcause](https://github.com/bradyneal/realcause)
- **推荐指数**: ⭐⭐⭐⭐

---

**🛠️ DoWhy**
- **机构**: Microsoft
- **功能**: 端到端因果推断库
- **支持数据集**: IHDP, Twins, Lalonde Jobs
- **代码**: [GitHub - microsoft/dowhy](https://github.com/microsoft/dowhy) ⭐ 7k+
- **文档**: [DoWhy Docs](https://petergtz.github.io/dowhy/v0.5.1/index.html)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**🛠️ Awesome Causality Data**
- **内容**: 因果推断数据集策划索引
- **包含**: IHDP, Twins, ACIC 等
- **代码**: [GitHub - rguo12/awesome-causality-data](https://github.com/rguo12/awesome-causality-data)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

**🛠️ Deep Learning for Causal Inference (Koch)**
- **作者**: Bernard J. Koch
- **内容**: 使用 TensorFlow 2 和 PyTorch 构建深度学习因果推断模型的广泛教程
- **涵盖**: HTE, selection on observables
- **代码**: [GitHub - kochbj/Deep-Learning-for-Causal-Inference](https://github.com/kochbj/Deep-Learning-for-Causal-Inference)
- **推荐指数**: ⭐⭐⭐⭐⭐

---

## 附录：学习建议

### 入门路径 (1-2 个月)

1. **第 1 周**: 阅读综述论文
   - "A Primer on Deep Learning for Causal Inference"
   - "Causal Inference Meets Deep Learning: A Comprehensive Survey"

2. **第 2-3 周**: 掌握基础架构
   - 精读 TARNet/CFRNet 论文
   - 精读 DragonNet 论文
   - 动手实现简化版 TARNet

3. **第 4 周**: 理解表示学习
   - 学习 BNN 和领域适应观点
   - 阅读 CFRNet 的平衡表示理论

4. **第 5-6 周**: 实践与应用
   - 在 IHDP 数据集上复现实验
   - 使用 EconML/CausalML 库
   - 阅读 KDD 2021 Tutorial 材料

5. **第 7-8 周**: 生成模型
   - 学习 CEVAE 和 GANITE
   - 理解隐藏混淆问题

### 进阶路径 (2-3 个月)

1. **Transformer 方法**
   - Causal Transformer (ICML 2022)
   - DAG-aware Transformer
   - CausalFormer

2. **时序因果推断**
   - CRN (Counterfactual Recurrent Network)
   - SyncTwin
   - Causal Contrastive Learning (NeurIPS 2024)

3. **剂量-反应**
   - VCNet
   - SCIGAN
   - TransTEE, ADMIT

4. **GNN 方法**
   - NN-CGC (Neural Networks with Causal Graph Constraints)
   - GNN for Treatment Effect Prediction
   - Spatial Graph 应用

### 深入路径 (3+ 个月)

1. **工具变量**
   - Deep IV (ICML 2017)
   - DeLIVR
   - DeepGMM

2. **双重机器学习**
   - Chernozhukov et al. (2018) 开山之作
   - DoubleML 库实践
   - 连续处理的 DML 扩展

3. **贝叶斯与不确定性**
   - BNCDE (贝叶斯神经控制微分方程)
   - Foundation Models with BNN
   - Bayesian TMLE

4. **元学习**
   - Zero-shot Causal Learning (NeurIPS 2023)
   - MetaCI
   - Transfer Learning for ITE

### 实战建议

1. **代码复现**: 至少复现 3-5 个核心论文
2. **数据集实践**: 在 IHDP, Twins, Jobs 上测试不同方法
3. **库的使用**: 熟练掌握 EconML, CausalML, DoWhy
4. **论文笔记**: 记录核心创新、适用场景、代码链接
5. **博客写作**: 总结学习心得，加深理解

---

## 推荐阅读顺序

### 必读论文 (Top 10)

1. ⭐ Learning Representations for Counterfactual Inference (ICML 2016)
2. ⭐ Estimating ITE: TARNet/CFRNet (ICML 2017)
3. ⭐ Adapting Neural Networks: DragonNet (NeurIPS 2019)
4. ⭐ Meta-Learners for HTE (PNAS 2019)
5. ⭐ CEVAE (NeurIPS 2017)
6. ⭐ Deep IV (ICML 2017)
7. ⭐ Double/Debiased Machine Learning (Econometrics Journal 2018)
8. ⭐ Causal Transformer (ICML 2022)
9. ⭐ CRN (NeurIPS 2020)
10. ⭐ Zero-shot Causal Learning (NeurIPS 2023)

### 综述论文 (Top 5)

1. ⭐ Causal Inference Meets Deep Learning: A Comprehensive Survey (2024)
2. ⭐ Deep Causal Learning (ACM Computing Surveys 2024)
3. ⭐ A Survey of Deep Causal Models (AI Review 2024)
4. ⭐ A Primer on Deep Learning for Causal Inference (2025)
5. ⭐ Instrumental Variables in CI and ML: A Survey (ACM CS 2024)

---

## 更新日志

- **2026-01-04**: 初始版本，涵盖 2016-2024 年重要论文
- 包含 14 个主题类别，100+ 篇论文
- 添加学习路径和推荐阅读顺序

---

## 贡献

欢迎补充遗漏的重要论文或更正错误信息。

---

## Sources

本文档基于以下搜索和文献综述整理：

- [Adapting Neural Networks for the Estimation of Treatment Effects](https://arxiv.org/pdf/1906.02120)
- [A Primer on Deep Learning for Causal Inference](https://faculty.ist.psu.edu/vhonavar/Courses/causality/dl-causal2.pdf)
- [TARNet and Dragonnet: Causal Inference Between S- And T-Learners | Towards Data Science](https://towardsdatascience.com/tarnet-and-dragonnet-causal-inference-between-s-and-t-learners-0444b8cc65bd/)
- [Causal Inference with Attention (CInA) | ICML 2024](https://icml.cc/virtual/2024/session/35594)
- [Causal Contrastive Learning for Counterfactual Regression Over Time | NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/02cef2ae63853724eb99e70721d3bc65-Paper-Conference.pdf)
- [Neural Networks with Causal Graph Constraints](https://arxiv.org/html/2404.12238v1)
- [Exploring Causal Learning Through Graph Neural Networks](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70024)
- [Adversarial Balancing-based Representation Learning](https://link.springer.com/article/10.1007/s10618-021-00759-3)
- [Estimating Conditional Average Treatment Effects via Sufficient Representation Learning](https://arxiv.org/html/2408.17053)
- [Double/Debiased Machine Learning for Treatment and Structural Parameters](https://academic.oup.com/ectj/article/21/1/C1/5056401)
- [Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments](https://www.tandfonline.com/doi/full/10.1080/07350015.2025.2505487)
- [GitHub - uber/causalml](https://github.com/uber/causalml)
- [Using Causal Inference to Improve the Uber User Experience](https://www.uber.com/blog/causal-inference-at-uber/)
- [Causal Inference and Machine Learning in Practice with EconML and CausalML | KDD 2021](https://dl.acm.org/doi/10.1145/3447548.3470792)
- [Practical Marketplace Optimization at Uber](https://arxiv.org/html/2407.19078v1)
- [GitHub - py-why/EconML](https://github.com/py-why/EconML)
- [EconML - Microsoft Research](https://www.microsoft.com/en-us/research/project/econml/)
- [GitHub - microsoft/causica](https://github.com/microsoft/causica)
- [Treatment Effect Estimation Benchmarks | IEEE DataPort](https://ieee-dataport.org/documents/treatment-effect-estimation-benchmarks)
- [GitHub - VectorInstitute/Causal_Inference_Laboratory](https://github.com/VectorInstitute/Causal_Inference_Laboratory)
- [GitHub - bradyneal/realcause](https://github.com/bradyneal/realcause)
- [GitHub - rguo12/awesome-causality-data](https://github.com/rguo12/awesome-causality-data)
- [Causal Inference Meets Deep Learning: A Comprehensive Survey | PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384545/)
- [Deep Causal Learning: Representation, Discovery and Inference | ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3762179)
- [A Survey of Deep Causal Models | Artificial Intelligence Review](https://link.springer.com/article/10.1007/s10462-024-10886-0)
- [Perfect Match | arXiv:1810.00656](https://arxiv.org/pdf/1810.00656)
- [Learning Representations for Counterfactual Inference | PMLR](http://proceedings.mlr.press/v48/johansson16.pdf)
- [GitHub - AMLab-Amsterdam/CEVAE](https://github.com/AMLab-Amsterdam/CEVAE)
- [Causal Effect Variational Autoencoder with Uniform Treatment | arXiv:2111.08656](https://arxiv.org/abs/2111.08656)
- [Contrastive Balancing Representation Learning for Dose-Response | arXiv:2403.14232](https://arxiv.org/html/2403.14232)
- [CausalFormer: An Interpretable Transformer for Temporal Causal Discovery](https://arxiv.org/html/2406.16708v1)
- [Causal Transformer for Estimating Counterfactual Outcomes | PMLR](https://proceedings.mlr.press/v162/melnychuk22a/melnychuk22a.pdf)
- [DAG-aware Transformer for Causal Effect Estimation | arXiv:2410.10044](https://arxiv.org/html/2410.10044v1)
- [Transformer-Variational Autoencoder for ITE | Springer](https://link.springer.com/article/10.1007/s10489-025-06738-1)
- [CETransformer | arXiv:2107.08714](https://arxiv.org/abs/2107.08714)
- [Deep IV: A Flexible Approach for Counterfactual Prediction | PMLR](https://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf)
- [DeLIVR | PubMed](https://pubmed.ncbi.nlm.nih.gov/36610078/)
- [Instrumental Variables in Causal Inference and ML: A Survey | ACM Computing Surveys](https://dlnext.acm.org/doi/abs/10.1145/3735969)
- [Bayesian Neural Controlled Differential Equations | arXiv:2310.17463](https://arxiv.org/html/2310.17463v2)
- [Foundation Models for Causal Inference via BNN | OpenReview](https://openreview.net/pdf?id=d2L1ndOKjq)
- [Bayesian TMLE | arXiv:2507.15909](https://arxiv.org/html/2507.15909)
- [Zero-shot Causal Learning | NeurIPS 2023](https://cs.stanford.edu/people/jure/pubs/zero-neurips23.pdf)
- [MetaCI | Semantic Scholar](https://www.semanticscholar.org/paper/MetaCI:-Meta-Learning-for-Causal-Inference-in-a-Sharma-Gupta/bbcebbe3295ebd9cfada36cff91f46697dc78934)
- [Advantages and Limitations of Transfer Learning for ITE | arXiv:2512.16489](https://arxiv.org/html/2512.16489)
- [Meta-Learners for HTE | arXiv:1706.03461](https://arxiv.org/pdf/1706.03461)
- [Meta-Learning for HTE with Closed-Form Solvers | Springer](https://link.springer.com/article/10.1007/s10994-024-06546-7)
- [Causal Machine Learning for Predicting Treatment Outcomes | Nature Medicine](https://www.nature.com/articles/s41591-024-02902-1)

---

**Happy Learning!**
