# Part 1: Experimentation Methods
# 实验方法论模块

## 模块概述

本模块实现了现代A/B测试和实验设计的核心方法，从基础到高级，覆盖实际业务中的各种场景。

## 模块结构

```
part1_experimentation/
├── __init__.py              # 模块导出
├── utils.py                 # 工具函数
├── ab_testing.py           # A/B测试基础
├── cuped.py                # CUPED方差缩减
├── stratified_analysis.py  # 分层分析
├── network_effects.py      # 网络效应与溢出
├── switchback.py           # Switchback实验
├── long_term_effects.py    # 长期效应估计
├── multi_armed_bandits.py  # 多臂老虎机
└── api.py                  # API适配层
```

## 核心功能

### 1. A/B Testing Basics (ab_testing.py)
**核心内容:**
- 样本量计算
- 统计检验（t检验、z检验）
- SRM检测
- 多重检验校正
- 平衡性检查

**API函数:** `analyze_ab_test()`

**典型使用:**
```python
from part1_experimentation import analyze_ab_test

result = analyze_ab_test(
    n_control=10000,
    n_treatment=10000,
    baseline_rate=0.05,
    treatment_effect=0.10
)

print(result['summary'])  # 查看分析报告
```

### 2. CUPED (cuped.py)
**核心内容:**
- 方差缩减技术
- 协变量选择
- 效果对比

**原理:**
```
Y_adj = Y - θ(X - X̄)
其中 θ = Cov(Y, X) / Var(X)
```

**API函数:** `apply_cuped()`

**优势:**
- 方差缩减 20-50%
- 更小的样本量需求
- 更短的实验周期

### 3. Stratified Analysis (stratified_analysis.py)
**核心内容:**
- 分层效应分析
- 异质性检验
- 汇总估计

**API函数:** `stratified_analysis()`

**应用场景:**
- 发现不同用户群体的差异效应
- 精准定向投放
- 个性化决策

### 4. Network Effects (network_effects.py)
**核心内容:**
- 网络效应建模
- 溢出效应估计
- SUTVA违反检测

**API函数:** `analyze_network_effects()`

**典型场景:**
- 社交网络实验
- 双边市场
- 平台效应

### 5. Switchback Experiments (switchback.py)
**核心内容:**
- 时间维度切换实验
- 固定效应分析
- 残留效应检测

**API函数:** `analyze_switchback()`

**适用场景:**
- 网约车司机端实验
- 外卖配送策略
- 供给侧实验

### 6. Long-term Effects (long_term_effects.py)
**核心内容:**
- 时变效应估计
- 短期 vs 长期对比
- Holdout分析
- 代理指标验证

**API函数:** `estimate_long_term_effects()`

**核心问题:**
短期效应 ≠ 长期效应

### 7. Multi-Armed Bandits (multi_armed_bandits.py)
**核心内容:**
- Epsilon-Greedy
- Thompson Sampling
- UCB算法
- Regret分析

**API函数:** `run_bandit_simulation()`

**vs A/B Testing:**
- A/B: 固定分流，明确因果
- Bandits: 动态优化，最大化收益

## API规范

所有API函数统一返回格式:

```python
{
    "charts": [...],      # Plotly图表JSON列表
    "tables": [...],      # 数据表格列表
    "summary": "...",     # Markdown格式的文字总结
    "metrics": {...}      # 关键指标字典
}
```

## 快速开始

### 基础A/B测试
```python
from part1_experimentation import analyze_ab_test

result = analyze_ab_test(
    n_control=10000,
    n_treatment=10000,
    baseline_rate=0.05,
    treatment_effect=0.10
)
```

### CUPED方差缩减
```python
from part1_experimentation import apply_cuped

result = apply_cuped(
    metric_col='converted',
    covariate_col='historical_conversion',
    n_samples=10000
)
```

### 分层分析
```python
from part1_experimentation import stratified_analysis

result = stratified_analysis(
    metric_col='converted',
    strata_col='user_activity',
    n_quantiles=4
)
```

### Bandit模拟
```python
from part1_experimentation import run_bandit_simulation

result = run_bandit_simulation(
    arm_means=[0.5, 0.6, 0.55, 0.7],
    algorithm='thompson',
    n_rounds=1000
)
```

## 核心概念

### 1. SUTVA (Stable Unit Treatment Value Assumption)
**定义:** 一个单元的潜在结果不受其他单元处理分配的影响

**违反场景:**
- 网络效应
- 双边市场
- 平台效应

**解决方案:**
- Cluster随机化
- Switchback实验
- 网络效应建模

### 2. 方差缩减
**目标:** 减少估计量的方差，提高统计功效

**方法:**
- CUPED (使用协变量)
- 分层 (Stratification)
- 配对 (Matching)

### 3. 异质性处理效应 (HTE)
**定义:** 不同群体的处理效应不同

**估计方法:**
- 分层分析
- 交互效应回归
- CATE估计 (见Part 4)

### 4. Explore vs Exploit
**Exploration:** 尝试新选项，获取信息
**Exploitation:** 利用已知信息，最大化收益

**权衡:**
- A/B Testing: 纯Exploration
- Greedy: 纯Exploitation
- Bandits: 平衡两者

## 面试常见问题

### A/B测试基础
Q: 如何计算样本量？
A: 根据MDE、alpha、power计算。公式见 `ab_testing.calculate_sample_size()`

Q: 什么是SRM？
A: Sample Ratio Mismatch，样本比例不匹配，表明分流有问题

Q: t检验和z检验的区别？
A: 样本量大时两者接近。小样本用t检验。

### CUPED
Q: CUPED的假设条件？
A: 协变量与处理分配独立（随机化保证）

Q: 如何选择协变量？
A: 与结果指标相关性高的实验前指标

### 网络效应
Q: 如何检测网络效应？
A: 比较有treated好友vs无treated好友的对照组

Q: Cluster随机化的代价？
A: 需要更多cluster才能达到同样功效

### Bandits
Q: Regret如何定义？
A: Σ(最优臂均值 - 实际选择臂均值)

Q: Thompson Sampling的优势？
A: 贝叶斯最优，自动平衡explore-exploit

## 最佳实践

### 1. 实验设计
- 事前功效分析
- 明确主指标
- 预注册分析计划

### 2. 数据质量
- SRM检查
- 平衡性检查
- 异常值检测

### 3. 统计分析
- 多重检验校正
- 敏感性分析
- 异质性分析

### 4. 业务决策
- 统计显著 ≠ 业务显著
- 考虑长期效应
- 权衡成本收益

## 扩展阅读

### 论文
1. Deng et al. (2013) - "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data"
2. Agarwal et al. (2017) - "The Case for Contextual Bandits in Experimentation"
3. Athey & Imbens (2017) - "The State of Applied Econometrics: Causality and Policy Evaluation"

### 工业实践
- Netflix: Long-term Holdout
- Uber: Switchback for Marketplace Experiments
- LinkedIn: Variance Reduction at Scale

## 开发者注意事项

### 添加新功能
1. 在相应模块文件中实现核心逻辑
2. 在 `api.py` 中添加API函数
3. 在 `__init__.py` 中导出
4. 编写测试用例

### 代码规范
- 函数需要完整的docstring
- 返回格式必须符合API规范
- 图表使用Plotly
- 颜色方案统一

### 测试
```bash
python -m pytest part1_experimentation/tests/
```

## 更新日志

### v1.0.0 (2026-01-04)
- ✅ 完整实现8个核心模块
- ✅ 统一API接口
- ✅ 完整文档和示例
