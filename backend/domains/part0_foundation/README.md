# Part 0: 因果思维基础模块

本模块提供因果推断的核心概念和分析工具。

## 模块结构

```
part0_foundation/
├── __init__.py                    # 模块导出
├── potential_outcomes.py          # 潜在结果框架
├── causal_dag.py                  # 因果图与DAG
├── identification_strategies.py   # 识别策略框架 (核心新增)
├── bias_types.py                  # 偏差类型分析
├── utils.py                       # 工具函数
├── api.py                         # API适配层
└── README.md                      # 本文件
```

## API 接口

所有 API 函数返回统一格式:

```python
{
    "charts": [plotly_figure_dict, ...],  # Plotly 图表
    "tables": [table_data, ...],           # 表格数据
    "summary": "markdown_text",            # Markdown 格式的分析摘要
    "metrics": {...}                       # 关键指标字典
}
```

### 1. 潜在结果框架

```python
from domains.part0_foundation.api import analyze_potential_outcomes

result = analyze_potential_outcomes(
    n_samples=500,
    treatment_effect=2.0,
    noise_std=1.0,
    confounding_strength=0.0
)
```

**功能**: 
- 可视化 Y(0), Y(1), ITE 分布
- 演示反事实问题
- 计算 ATE 和估计偏差

### 2. 因果图分析

```python
from domains.part0_foundation.api import analyze_causal_dag

result = analyze_causal_dag(scenario="confounding")
# scenario: "confounding", "mediation", "collider", "complex"
```

**功能**:
- 可视化不同类型的 DAG
- 识别后门路径
- 提供识别策略建议

### 3. 混淆偏差分析

```python
from domains.part0_foundation.api import analyze_confounding_bias

result = analyze_confounding_bias(
    n_samples=1000,
    confounding_strength=1.0,
    treatment_effect=2.0
)
```

**功能**:
- 演示混淆如何产生偏差
- 对比朴素估计和调整估计
- 检查协变量平衡

### 4. 选择偏差分析

```python
from domains.part0_foundation.api import analyze_selection_bias

result = analyze_selection_bias(
    n_samples=1000,
    selection_strength=1.0,
    treatment_effect=2.0
)
```

**功能**:
- 演示样本选择如何引入偏差
- 对比完整样本和选择样本
- 量化选择偏差

### 5. 识别策略推荐 (新增核心功能)

```python
from domains.part0_foundation.api import analyze_identification_strategy

result = analyze_identification_strategy(
    data_type="observational",  # "experimental" 或 "observational"
    confounding_observed=True,
    has_instrument=False,
    has_panel=False,
    has_discontinuity=False
)
```

**功能**:
- 决策树帮助选择合适的因果推断方法
- 评估各方法的假设条件
- 提供多个推荐方法并排序
- 方法对比表格

### 6. 偏差类型对比

```python
from domains.part0_foundation.api import analyze_bias_comparison

result = analyze_bias_comparison()
```

**功能**:
- 演示 Simpson's Paradox
- 演示 Berkson's Paradox
- 分析测量偏差
- 对比各种偏差类型

## 核心概念

### 潜在结果框架 (Potential Outcomes)

- **Y(0)**: 不接受处理时的潜在结果
- **Y(1)**: 接受处理时的潜在结果
- **ITE**: Y(1) - Y(0) (个体处理效应)
- **ATE**: E[Y(1) - Y(0)] (平均处理效应)
- **基本问题**: 每个个体只能观测到一个潜在结果

### 因果图 (Causal DAG)

- **混淆变量**: 同时影响处理和结果
- **中介变量**: 传递因果效应
- **碰撞变量**: 被多个变量影响 (不要控制!)
- **后门准则**: 识别需要控制的变量

### 识别策略

根据数据特征推荐方法:

| 数据类型 | 条件 | 推荐方法 |
|---------|------|---------|
| 实验数据 | - | 简单差分 / T-test |
| 观测数据 | 观测到混淆 | PSM / IPW / DR |
| 观测数据 | 未观测混淆 + 工具变量 | IV / 2SLS |
| 面板数据 | - | DID / 固定效应 |
| 断点设计 | - | RDD |

### 偏差类型

1. **混淆偏差**: X → T, X → Y
2. **选择偏差**: 样本选择与结果相关
3. **碰撞偏差**: 控制碰撞变量引入虚假关联
4. **Simpson's Paradox**: 整体趋势与分层趋势相反
5. **测量偏差**: 变量测量误差导致衰减偏差

## 使用示例

```python
# 1. 分析潜在结果框架
from domains.part0_foundation.api import analyze_potential_outcomes

result = analyze_potential_outcomes(
    n_samples=500,
    treatment_effect=2.0,
    confounding_strength=0.5  # 添加一些混淆
)

print(result['summary'])
print("ATE估计偏差:", result['metrics']['bias'])

# 2. 获取方法推荐
from domains.part0_foundation.api import analyze_identification_strategy

result = analyze_identification_strategy(
    data_type="observational",
    confounding_observed=True
)

print("推荐方法:", result['metrics']['recommended_method'])

# 3. 对比各种偏差
from domains.part0_foundation.api import analyze_bias_comparison

result = analyze_bias_comparison()
print("Simpson's Paradox:", result['metrics']['simpsons_paradox'])
```

## 与 Notebooks 对应关系

| Notebook | 对应函数 |
|----------|---------|
| part0_1_potential_outcomes.ipynb | `analyze_potential_outcomes()` |
| part0_2_causal_dag.ipynb | `analyze_causal_dag()` |
| part0_3_identification_strategies.ipynb | `analyze_identification_strategy()` |
| part0_4_bias_types.ipynb | `analyze_confounding_bias()`, `analyze_selection_bias()`, `analyze_bias_comparison()` |

## 依赖

```python
numpy
pandas
scikit-learn
plotly
```

## 下一步

- **Part 1**: 实验设计 (A/B Testing, 样本量计算)
- **Part 2**: 观测数据方法 (PSM, IPW, DR)
- **Part 3**: 准实验设计 (DID, IV, RDD)
- **Part 4**: CATE & Uplift Modeling
