# Part 7: 高级主题模块

本模块实现了因果推断的高级主题，包括因果发现、连续处理效应、时变处理效应和中介分析。

## 模块结构

```
part7_advanced/
├── __init__.py              # 模块导出
├── api.py                   # API 适配层（统一返回格式）
├── utils.py                 # 数据生成和工具函数
├── causal_discovery.py      # 因果发现（PC 算法）
├── continuous_treatment.py  # 连续处理效应（GPS、DRF）
├── time_varying_treatment.py # 时变处理效应（MSM、G-computation）
├── mediation_analysis.py    # 中介分析（效应分解）
└── README.md               # 本文件
```

## API 函数

### 1. analyze_causal_discovery()
因果发现分析，使用 PC 算法从数据中学习因果结构。

**参数:**
- `n_samples`: 样本量（默认: 1000）
- `n_variables`: 变量数量（默认: 6）
- `graph_type`: 图类型（'chain', 'fork', 'collider', 'complex'）

**返回:** 
```python
{
    "charts": [...],      # Plotly 图表
    "tables": [...],      # 数据表格
    "summary": "...",     # Markdown 摘要
    "metrics": {          # 关键指标
        "precision": float,
        "recall": float,
        "f1_score": float,
        "shd": int
    }
}
```

### 2. analyze_continuous_treatment()
连续处理效应分析，估计剂量响应函数（DRF）。

**参数:**
- `n_samples`: 样本量（默认: 1000）
- `treatment_distribution`: 处理分布（'uniform', 'normal', 'gamma'）

**返回:**
```python
{
    "charts": [...],
    "tables": [...],
    "summary": "...",
    "metrics": {
        "optimal_treatment": float,  # 最优处理水平
        "max_outcome": float,        # 最大期望结果
        "mse_gps": float,            # GPS 方法 MSE
        "mse_spline": float          # 样条方法 MSE
    }
}
```

### 3. analyze_time_varying_treatment()
时变处理效应分析，使用边际结构模型（MSM）和 G-computation。

**参数:**
- `n_periods`: 时间周期数（默认: 5）
- `treatment_pattern`: 处理模式（'random', 'increasing', 'alternating'）

**返回:**
```python
{
    "charts": [...],
    "tables": [...],
    "summary": "...",
    "metrics": {
        "msm_treatment_effect": float,  # MSM 估计的处理效应
        "ate": float,                   # 平均处理效应
        "always_treat_outcome": float,
        "never_treat_outcome": float
    }
}
```

### 4. analyze_mediation()
中介分析，分解总效应为直接和间接效应。

**参数:**
- `n_samples`: 样本量（默认: 1000）
- `direct_effect`: 真实直接效应（默认: 2.0）
- `indirect_effect`: 真实间接效应（默认: 1.5）

**返回:**
```python
{
    "charts": [...],
    "tables": [...],
    "summary": "...",
    "metrics": {
        "total_effect": float,           # 总效应
        "direct_effect": float,          # 直接效应 (NDE)
        "indirect_effect": float,        # 间接效应 (NIE)
        "proportion_mediated": float,    # 中介比例
        "mediation_type": str            # 中介类型
    }
}
```

## 核心算法

### 因果发现（causal_discovery.py）
- **PC 算法**: 基于条件独立性检验的因果结构学习
- **性能评估**: Precision, Recall, F1, SHD
- **适用场景**: 探索性因果分析，变量关系发现

### 连续处理（continuous_treatment.py）
- **广义倾向得分 (GPS)**: 调整混淆偏差
- **剂量响应函数 (DRF)**: 参数和非参数估计
- **边际效应**: 计算处理剂量变化的影响
- **适用场景**: 优惠券金额优化、广告预算分配、定价策略

### 时变处理（time_varying_treatment.py）
- **边际结构模型 (MSM)**: 使用 IPW 估计时变效应
- **G-computation**: 标准化方法模拟干预
- **累积效应**: 分析长期累积影响
- **适用场景**: 重复干预、治疗方案优化、政策评估

### 中介分析（mediation_analysis.py）
- **效应分解**: 总效应 = 直接效应 + 间接效应
- **Baron-Kenny 检验**: 经典三步法
- **敏感性分析**: 评估未观测混淆的影响
- **适用场景**: 机制研究、路径分析、归因分析

## 使用示例

```python
from domains.part7_advanced import (
    analyze_causal_discovery,
    analyze_continuous_treatment,
    analyze_time_varying_treatment,
    analyze_mediation
)

# 因果发现
result1 = analyze_causal_discovery(
    n_samples=2000,
    n_variables=6,
    graph_type="complex"
)

# 连续处理
result2 = analyze_continuous_treatment(
    n_samples=1500,
    treatment_distribution="normal"
)

# 时变处理
result3 = analyze_time_varying_treatment(
    n_periods=10,
    treatment_pattern="random"
)

# 中介分析
result4 = analyze_mediation(
    n_samples=2000,
    direct_effect=2.5,
    indirect_effect=1.8
)

# 所有结果都遵循统一格式
for result in [result1, result2, result3, result4]:
    print(result['summary'])
    print(result['metrics'])
```

## 依赖项

- numpy: 数值计算
- pandas: 数据处理
- scipy: 统计检验、样条插值
- sklearn: 回归模型
- plotly: 可视化

## 理论基础

### 因果发现
- Spirtes et al. (2000): "Causation, Prediction, and Search"
- Pearl (2009): "Causality: Models, Reasoning and Inference"

### 连续处理
- Hirano & Imbens (2004): "The Propensity Score with Continuous Treatments"
- Flores et al. (2012): "Estimating the Effects of Length of Exposure"

### 时变处理
- Robins et al. (2000): "Marginal Structural Models"
- Hernán & Robins (2020): "Causal Inference: What If"

### 中介分析
- Baron & Kenny (1986): "The Moderator-Mediator Variable Distinction"
- Imai et al. (2010): "A General Approach to Causal Mediation Analysis"

## 注意事项

1. **因果发现**: 需要大样本量（建议 1000+），假设无隐变量
2. **连续处理**: GPS 方法假设条件正态分布，可能需要变换
3. **时变处理**: IPW 权重可能不稳定，建议使用稳定权重
4. **中介分析**: 假设无交互、无混淆，需要谨慎解释

## 扩展方向

- 加入 FCI 算法处理隐变量
- 实现 LiNGAM（非高斯因果发现）
- 支持多个中介变量
- 实现 Double ML 用于连续处理
- 加入 Bootstrap 置信区间
