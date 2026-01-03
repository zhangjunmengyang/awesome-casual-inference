# Chapter 6: Application Lab 练习

本章练习涵盖因果推断在真实业务场景中的应用。

## 练习概览

### ex1_coupon_optimization.py - 智能发券优化

**业务场景**: 外卖/电商平台优惠券分配优化

**学习目标**:
- 理解营销场景中的用户分群策略 (Persuadables, Sure Things, Lost Causes, Sleeping Dogs)
- 掌握 Uplift 建模在发券场景的应用
- 学习 ROI 优化决策方法
- 对比不同发券策略的效果

**核心概念**:
- Uplift Modeling (Two-Model / T-Learner)
- 用户分群 (User Segmentation)
- ROI 优化 (Return on Investment)
- 策略对比 (Policy Comparison)

**难度**: ⭐⭐⭐

---

### ex2_ab_enhancement.py - A/B 测试增强

**业务场景**: 视频平台新功能 A/B 测试优化

**学习目标**:
- 理解传统 A/B 测试的局限性
- 掌握 CUPED 方差缩减技术
- 学习异质效应 (HTE) 分析
- 理解统计功效 (Power) 和样本量规划

**核心概念**:
- CUPED (Controlled-experiment Using Pre-Experiment Data)
- 方差缩减 (Variance Reduction)
- 异质效应分析 (Heterogeneous Treatment Effect)
- 统计功效分析 (Power Analysis)

**难度**: ⭐⭐⭐⭐

---

### ex3_user_targeting.py - 用户定向干预

**业务场景**: 网约车平台司机激励优化

**学习目标**:
- 理解 CATE (Conditional Average Treatment Effect) 估计
- 掌握 X-Learner 高级方法
- 学习最优干预策略 (Policy Learning)
- 理解成本-收益权衡

**核心概念**:
- T-Learner vs X-Learner
- CATE 估计 (Conditional Average Treatment Effect)
- 最优策略学习 (Optimal Policy Learning)
- 用户分层 (User Segmentation)
- 敏感性分析 (Sensitivity Analysis)

**难度**: ⭐⭐⭐⭐⭐

---

## 使用方法

### 1. 阅读源码

在开始练习前，先阅读 `application_lab/` 目录下的源码:
- `coupon_optimization.py` - 发券优化实现
- `ab_enhancement.py` - A/B 测试增强实现
- `user_targeting.py` - 用户定向干预实现
- `utils.py` - 数据生成和工具函数

### 2. 完成练习

每个练习文件包含:
- **TODO 注释**: 标记需要完成的部分
- **函数文档**: 详细说明函数功能和参数
- **思考题**: 加深对概念的理解
- **测试代码**: 验证实现的正确性

### 3. 运行测试

完成 TODO 后，运行脚本验证答案:

```bash
# 练习 1
python ex1_coupon_optimization.py

# 练习 2
python ex2_ab_enhancement.py

# 练习 3
python ex3_user_targeting.py
```

### 4. 对照源码

如果遇到困难，可以对照 `application_lab/` 下的源码实现。

---

## 学习路径建议

### 初学者路径
1. 先完成 **ex1_coupon_optimization.py** (相对简单)
2. 理解 Uplift 建模的基本思想
3. 再挑战 ex2 和 ex3

### 进阶路径
1. 按顺序完成三个练习
2. 认真思考每个思考题
3. 尝试优化代码实现
4. 对比不同方法的优劣

### 实践导向路径
1. 快速浏览练习代码
2. 直接运行 `application_lab/` 下的 Gradio 界面
3. 在 UI 中实验不同参数
4. 再回来完成练习加深理解

---

## 关键技术对比

| 方法 | 适用场景 | 优势 | 局限 |
|------|---------|------|------|
| **T-Learner** | 样本充足，处理组平衡 | 简单直观，易实现 | 样本小时不稳定 |
| **X-Learner** | 样本不平衡，处理比例极端 | 利用伪处理效应，更稳健 | 实现复杂 |
| **CUPED** | 有实验前数据的 A/B 测试 | 显著减少方差，提升功效 | 需要相关的实验前数据 |

---

## 实际案例参考

### 智能发券优化
- **DoorDash**: Uplift Modeling 优化促销活动，ROI 提升 30%+
- **Meituan**: 智能发券系统每年节省数亿补贴成本

### A/B 测试增强
- **Netflix**: CUPED 使实验周期缩短 30%
- **Airbnb**: HTE 分析发现不同国家的效应差异

### 用户定向干预
- **Uber**: CATE 优化司机激励，ROI 提升 40%+
- **Lyft**: 分层激励策略减少 30% 补贴浪费

---

## 扩展阅读

### 论文
- [Uber's Causal ML Platform](https://eng.uber.com/causal-inference-at-uber/)
- [Improving the Sensitivity of Online Controlled Experiments (CUPED)](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning](https://arxiv.org/abs/1706.03461)

### 博客
- [Netflix Experimentation Platform](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15)
- [DoorDash Experimentation Platform](https://doordash.engineering/2020/09/09/experimentation-analysis-platform-mvp/)

### 开源库
- [EconML](https://github.com/microsoft/EconML) - Microsoft 的因果推断库
- [CausalML](https://github.com/uber/causalml) - Uber 的 Uplift 建模库

---

## 常见问题

### Q1: 练习太难怎么办?
- 先阅读源码理解整体思路
- 逐个函数完成，不要一次性完成所有 TODO
- 利用测试代码验证每个函数

### Q2: 如何判断实现是否正确?
- 运行测试代码，检查输出是否合理
- 对照源码实现
- 检查数值范围是否符合预期

### Q3: 思考题如何回答?
- 结合代码实现和业务场景思考
- 参考扩展阅读材料
- 可以查看源码中的注释和文档

---

## 贡献

如果发现问题或有改进建议，欢迎提 Issue 或 PR!
