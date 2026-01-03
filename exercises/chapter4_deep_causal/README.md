# Chapter 4: 深度因果推断练习

本章节包含深度学习因果推断模型的练习，循序渐进地学习从表示学习到 TARNet 和 DragonNet 的完整体系。

## 练习概览

### 练习 1: 表示学习基础 (`ex1_representation_learning.py`)

**学习目标:**
- 理解为什么需要学习表示 (Representation Learning)
- 掌握简单的神经网络特征提取
- 理解处理组和对照组表示的差异
- 为深度因果模型打下基础

**核心概念:**
- 手工特征 vs 学习表示
- 非线性特征变换
- 表示平衡性检查
- SMD 和 MMD 指标

**练习内容:**
1. 生成非线性数据
2. 对比线性模型和表示学习
3. 实现简单的表示学习网络
4. 可视化表示空间
5. 检查表示平衡性

---

### 练习 2: TARNet (`ex2_tarnet.py`)

**学习目标:**
- 理解 TARNet 的架构设计
- 实现简化版 TARNet
- 理解 Factual Loss 的含义
- 训练和评估 TARNet

**核心概念:**
- 共享表示层
- 双头输出 (Y0, Y1)
- Factual Loss
- 反事实推断

**架构:**
```
X -> [Shared Representation] -> Phi(X)
                                  |
                +----------------+----------------+
                |                                 |
            [Head 0]                         [Head 1]
                |                                 |
              Y(0)                              Y(1)
```

**练习内容:**
1. 实现 TARNet 网络架构
2. 理解和实现 Factual Loss
3. 训练 TARNet 模型
4. 评估 PEHE 和 ATE 误差

---

### 练习 3: DragonNet (`ex3_dragonnet.py`)

**学习目标:**
- 理解倾向得分头的作用
- 实现 DragonNet 架构
- 掌握 DragonNet 的复合损失函数
- 理解 Targeted Regularization

**核心概念:**
- 倾向得分头
- Targeted Regularization (TMLE 风格)
- Epsilon 层
- 端到端倾向得分估计

**架构:**
```
              +---> [Head 0] ---------> Y(0)
              |
X -> [Shared Repr] -> Phi(X) -> [Head 1] ---------> Y(1)
              |
              +---> [Propensity Head] -> e(X)
                          |
                          v
                     [Epsilon Layer] -> Targeted Reg
```

**损失函数:**
```
L = L_factual + α·L_propensity + β·L_targeted
```

**练习内容:**
1. 实现 DragonNet 三头架构
2. 实现复合损失函数
3. 理解 Targeted Regularization
4. 训练和评估 DragonNet
5. 评估倾向得分质量
6. 对比 TARNet vs DragonNet

---

## 运行练习

每个练习文件都是独立的，可以直接运行：

```bash
# 练习 1: 表示学习
python ex1_representation_learning.py

# 练习 2: TARNet
python ex2_tarnet.py

# 练习 3: DragonNet
python ex3_dragonnet.py
```

## 学习路径

建议按顺序完成练习：

1. **ex1_representation_learning.py** - 理解表示学习的基础
2. **ex2_tarnet.py** - 掌握深度因果模型的基本架构
3. **ex3_dragonnet.py** - 学习最先进的深度因果模型

## 依赖

```bash
pip install numpy pandas torch scikit-learn
```

## 学习资源

### 论文

1. **TARNet/CFR**:
   - Shalit et al., "Estimating individual treatment effect: generalization bounds and algorithms" (ICML 2017)

2. **DragonNet**:
   - Shi et al., "Adapting Neural Networks for the Estimation of Treatment Effects" (NeurIPS 2019)

### 相关代码

参考项目中的实现：
- `/deep_causal_lab/tarnet.py` - TARNet 完整实现
- `/deep_causal_lab/dragonnet.py` - DragonNet 完整实现
- `/deep_causal_lab/utils.py` - 工具函数

## 评估指标

### PEHE (Precision in Estimation of Heterogeneous Effect)
衡量个体处理效应 (ITE) 的估计精度：

```
PEHE = sqrt(E[(τ_true - τ_pred)²])
```

### ATE Error
衡量平均处理效应的估计误差：

```
ATE Error = |E[τ_true] - E[τ_pred]|
```

## 进阶挑战

完成基础练习后，可以尝试：

1. 实现 CFR (TARNet + IPM 正则化)
2. 实现 CEVAE (Causal Effect VAE)
3. 在真实数据集上测试 (IHDP, Jobs, Twins)
4. 实现更复杂的网络架构 (ResNet, Attention)
5. 添加早停和模型选择

## 思考题参考方向

练习中的思考题没有标准答案，但这里提供一些思考方向：

### 为什么需要表示学习？
- 原始特征可能无法捕获复杂的非线性关系
- 学习到的表示可以自动发现有用的特征组合
- 表示平衡有助于反事实推断

### TARNet 的关键设计
- 共享表示确保两个头使用相同的特征空间
- 双头输出允许同时学习两个潜在结果
- Factual Loss 只利用观测数据，避免反事实标签

### DragonNet 的创新
- 倾向得分头提供隐式正则化
- Targeted Regularization 类似 TMLE 的偏差校正
- 端到端训练优于两阶段方法

---

祝学习顺利！有问题可以参考源码或查阅论文。
