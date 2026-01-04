"""
Part 0 Foundation API 适配层

将各个模块的功能整合为统一的 API 接口,
返回格式: {"charts": [...], "tables": [], "summary": "...", "metrics": {...}}
"""

from typing import Dict, Any
import plotly.graph_objects as go

from .potential_outcomes import visualize_potential_outcomes, demonstrate_fundamental_problem
from .causal_dag import create_dag_visualization, identify_backdoor_paths, simulate_confounding_effect
from .bias_types import (
    analyze_confounding_bias as _analyze_confounding_bias,
    analyze_selection_bias as _analyze_selection_bias,
    analyze_measurement_bias as _analyze_measurement_bias,
    demonstrate_simpsons_paradox,
    demonstrate_berksons_paradox
)
from .identification_strategies import (
    create_strategy_decision_tree,
    get_identification_strategy,
    recommend_methods,
    create_method_comparison_table
)
from .utils import fig_to_dict


def analyze_potential_outcomes(
    n_samples: int = 500,
    treatment_effect: float = 2.0,
    noise_std: float = 1.0,
    confounding_strength: float = 0.0
) -> Dict[str, Any]:
    """
    潜在结果框架分析 API

    Returns:
    --------
    {
        "charts": [plotly_figure_dict],
        "tables": [],
        "summary": markdown_text,
        "metrics": {...}
    }
    """
    # 生成主图
    fig, stats = visualize_potential_outcomes(
        n_samples=n_samples,
        treatment_effect=treatment_effect,
        noise_std=noise_std,
        confounding_strength=confounding_strength
    )

    # 生成基本问题演示图
    demo_fig = demonstrate_fundamental_problem()

    # 构建摘要
    summary = f"""
## 潜在结果框架分析

### 核心概念

**潜在结果框架** (Rubin Causal Model) 是因果推断的基石:
- Y(0): 个体**不接受处理**时的潜在结果
- Y(1): 个体**接受处理**时的潜在结果
- ITE = Y(1) - Y(0): 个体处理效应
- ATE = E[Y(1) - Y(0)]: 平均处理效应

### 统计量

| 指标 | 值 | 说明 |
|------|-----|------|
| 样本量 | {stats['n_samples']} | 模拟的个体数量 |
| 真实 ATE | {stats['true_ate']:.4f} | 设定的平均处理效应 |
| 朴素估计 | {stats['naive_ate']:.4f} | E[Y|T=1] - E[Y|T=0] |
| 估计偏差 | {stats['bias']:.4f} | 朴素估计 - 真实 ATE |
| ITE 标准差 | {stats['ite_std']:.4f} | 个体间效应的异质性 |

### 关键洞察

**因果推断的基本问题**: 每个个体只能观测到一个潜在结果
- 接受处理的人: 观测到 Y(1), 无法观测 Y(0) (反事实)
- 未接受处理的人: 观测到 Y(0), 无法观测 Y(1) (反事实)

**随机实验的威力**:
- 混淆强度 = {confounding_strength:.2f}
- 当混淆强度 = 0 (完美随机化) 时, 朴素估计 ≈ 真实 ATE
- 当混淆强度 > 0 时, 会产生偏差: {stats['bias']:.4f}

**ITE 异质性**:
- 标准差 = {stats['ite_std']:.4f} 反映不同个体的处理效应差异
- 即使 ATE > 0, 某些个体的 ITE 可能 < 0
- 这就是**异质性处理效应** (CATE) 研究的动机

### 下一步

1. 如果有观测数据, 需要处理混淆偏差 → Part 2: 观测数据方法
2. 如果想估计异质性效应 → Part 4: CATE & Uplift Modeling
3. 如果想理解因果图结构 → Part 0.2: 因果图分析
    """

    return {
        "charts": [
            fig_to_dict(fig),
            fig_to_dict(demo_fig)
        ],
        "tables": [],
        "summary": summary,
        "metrics": stats
    }


def analyze_causal_dag(
    scenario: str = "confounding"
) -> Dict[str, Any]:
    """
    因果图分析 API

    Parameters:
    -----------
    scenario: 'confounding', 'mediation', 'collider', 'complex'

    Returns:
    --------
    标准 API 响应格式
    """
    # 创建 DAG 可视化
    fig, explanation = create_dag_visualization(scenario)

    # 识别后门路径
    path_analysis = identify_backdoor_paths(scenario)

    # 构建摘要
    summary = f"""
## 因果图分析: {scenario.upper()}

### DAG 结构说明

{explanation}

### 路径分析

{path_analysis}

### 为什么需要因果图?

因果图 (DAG) 提供了一个**可视化语言**来:
1. **表示因果假设**: 哪些变量影响哪些变量
2. **识别混淆**: 找到需要控制的变量
3. **避免碰撞偏差**: 识别不应该控制的变量
4. **推导识别策略**: 后门准则、前门准则等

### 关键规则

**后门准则** (Backdoor Criterion):
- 控制集 X 满足后门准则 ⟺ X 阻断所有 T 到 Y 的后门路径, 且 X 不包含 T 的后代

**碰撞变量规则**:
- 永远不要控制碰撞变量!
- 控制碰撞变量会打开虚假关联 (Berkson's Paradox)

### 实践建议

根据你的 DAG 结构:
- **混淆**: 使用倾向得分方法 (PSM, IPW, DR)
- **中介**: 区分直接效应和间接效应
- **碰撞**: 不要控制碰撞变量
- **未观测混淆**: 考虑 IV, DID, 或敏感性分析
    """

    return {
        "charts": [fig_to_dict(fig)],
        "tables": [],
        "summary": summary,
        "metrics": {"scenario": scenario}
    }


def analyze_confounding_bias(
    n_samples: int = 1000,
    confounding_strength: float = 1.0,
    treatment_effect: float = 2.0
) -> Dict[str, Any]:
    """
    混淆偏差分析 API

    Returns:
    --------
    标准 API 响应格式
    """
    fig, stats = _analyze_confounding_bias(
        n_samples=n_samples,
        confounding_strength=confounding_strength,
        treatment_effect=treatment_effect
    )

    summary = f"""
## 混淆偏差分析

### 关键指标

| 指标 | 值 | 解释 |
|------|-----|------|
| 真实 ATE | {stats['true_ate']:.4f} | 设定的真实因果效应 |
| 朴素估计 | {stats['naive_ate']:.4f} | 简单差分的估计 |
| 调整估计 | {stats['adjusted_ate']:.4f} | 控制 X 后的估计 |
| **偏差** | **{stats['bias']:.4f}** | 朴素估计的偏差程度 |
| 混淆强度 | {stats['confounding_strength']:.2f} | X 对 T 和 Y 的影响强度 |

### 偏差来源

**混淆偏差**产生的条件:
1. 混淆变量 X 影响处理分配: X → T
2. 混淆变量 X 也影响结果: X → Y
3. 结果: 处理组和控制组的 X 分布不同

在本例中:
- X 较高 → 更可能接受处理 (T=1)
- X 较高 → 结果 Y 也较高
- 导致: 处理组的均值被**人为抬高**, 产生{'+正向' if stats['bias'] > 0 else '负向'}偏差

### 解决方案

1. **随机实验 (RCT)**: 打破 X → T 的关联 (金标准)
2. **倾向得分匹配 (PSM)**: 匹配相似的 X 的个体
3. **逆概率加权 (IPW)**: 重新加权样本以平衡 X
4. **回归调整**: 控制 X 作为协变量
5. **双重稳健 (DR)**: 结合倾向得分和结果模型

### 诊断建议

观察图表:
- **左上**: 协变量分布是否平衡? 不平衡 → 混淆存在
- **右上**: 散点图中两组的 X 范围是否重叠? 重叠 → 共同支撑满足
- **左下**: 倾向得分分布是否重叠? 重叠不足 → PSM 效果受限
- **右下**: 调整估计是否接近真实值? 接近 → 方法有效

### 下一步

→ 尝试 **Part 2: 观测数据方法** 中的 PSM, IPW, DR 方法
    """

    return {
        "charts": [fig_to_dict(fig)],
        "tables": [],
        "summary": summary,
        "metrics": stats
    }


def analyze_selection_bias(
    n_samples: int = 1000,
    selection_strength: float = 1.0,
    treatment_effect: float = 2.0
) -> Dict[str, Any]:
    """
    选择偏差分析 API

    Returns:
    --------
    标准 API 响应格式
    """
    fig, stats = _analyze_selection_bias(
        n_samples=n_samples,
        selection_strength=selection_strength,
        treatment_effect=treatment_effect
    )

    summary = f"""
## 选择偏差分析

### 关键指标

| 指标 | 值 |
|------|-----|
| 真实 ATE | {stats['true_ate']:.4f} |
| 完整样本估计 | {stats['full_naive']:.4f} |
| 选择样本估计 | {stats['selected_naive'] if stats['selected_naive'] is not None else 'N/A'} |
| **选择偏差** | **{stats['selection_bias'] if stats['selection_bias'] is not None else 'N/A'}** |
| 被选中比例 | {stats['selection_rate']:.2%} |

### 选择偏差来源

**选择偏差**发生在样本选择与研究变量相关:
- 本例: 结果 Y 较高的样本更可能被观测到
- 这是**碰撞偏差** (Collider Bias) 的一种形式
- DAG: T → 选择 ← Y

### 常见场景

1. **生存偏差** (Survivorship Bias):
   - 只观测到"幸存者"
   - 例: 分析成功企业的特征 (失败的看不到)

2. **自选择偏差**:
   - 受益者更可能留在实验中
   - 例: 减肥研究中效果好的人更愿意完成研究

3. **出版偏差**:
   - 显著结果更可能被发表
   - 例: 元分析中遗漏了阴性结果

4. **Berkson's Paradox**:
   - 医院数据研究
   - 例: 住院人群中疾病间的虚假负相关

### 解决方案

1. **避免选择**: 设计研究时最小化样本流失
2. **Heckman 校正**: 建模选择机制并校正
3. **Bounds 分析**: 在最坏/最好情况下的边界
4. **敏感性分析**: 评估选择偏差的影响范围

### 图表解读

- **左图**: 完整样本 (包含所有个体)
- **右图**: 选择后样本 (只包含被观测到的个体)
- 注意: 选择后样本的分布发生了改变

### 下一步

→ 查看 **Berkson's Paradox** 演示理解碰撞偏差
    """

    return {
        "charts": [fig_to_dict(fig)],
        "tables": [],
        "summary": summary,
        "metrics": stats
    }


def analyze_identification_strategy(
    data_type: str = "observational",
    confounding_observed: bool = False,
    has_instrument: bool = False,
    has_panel: bool = False,
    has_discontinuity: bool = False
) -> Dict[str, Any]:
    """
    识别策略推荐 API

    Parameters:
    -----------
    data_type: 'experimental' 或 'observational'
    confounding_observed: 是否观测到混淆变量
    has_instrument: 是否有工具变量
    has_panel: 是否有面板数据
    has_discontinuity: 是否有断点

    Returns:
    --------
    标准 API 响应格式
    """
    # 获取推荐策略
    strategy = get_identification_strategy(
        data_type=data_type,
        confounding_observed=confounding_observed,
        has_instrument=has_instrument,
        has_panel=has_panel,
        has_discontinuity=has_discontinuity
    )

    # 创建决策树图
    decision_tree_fig = create_strategy_decision_tree()

    # 创建方法对比表
    comparison_fig = create_method_comparison_table()

    # 推荐多个方法
    data_characteristics = {
        'is_experimental': data_type == "experimental",
        'confounders_observed': confounding_observed,
        'has_instrument': has_instrument,
        'has_panel': has_panel,
        'has_discontinuity': has_discontinuity,
        'sample_size': 1000
    }
    recommendations = recommend_methods(data_characteristics)

    # 构建推荐列表
    rec_list = "\n".join([
        f"{i+1}. **{method}** (优先级: {score:.2f}) - {reason}"
        for i, (method, score, reason) in enumerate(recommendations)
    ])

    summary = f"""
## 因果推断方法识别策略

### 推荐方法: {strategy['recommended_method']}

**优先级**: {strategy['priority']}

### 识别假设

该方法需要以下假设成立:
{chr(10).join(['- ' + assumption for assumption in strategy['assumptions']])}

### 优点

{chr(10).join(['- ' + pro for pro in strategy['pros']])}

### 缺点

{chr(10).join(['- ' + con for con in strategy['cons']])}

### 代码示例

```python
{strategy['code_example']}
```

---

## 所有推荐方法 (按优先级排序)

{rec_list}

---

## 如何选择方法?

### 决策流程

1. **数据类型**:
   - 实验数据 (RCT) → 简单差分 ✓
   - 观测数据 → 继续判断

2. **混淆变量**:
   - 观测到所有混淆 → PSM / IPW / DR
   - 存在未观测混淆 → 继续判断

3. **特殊数据结构**:
   - 有工具变量 → IV / 2SLS
   - 有面板数据 → DID / FE
   - 有断点设计 → RDD
   - 都没有 → 敏感性分析

### 方法对比

参考下方的**方法对比表**了解各方法的特点。

### 实践建议

1. **不要只依赖一种方法**: 使用多种方法交叉验证
2. **检验假设**: 每种方法都有假设,尽可能检验
3. **敏感性分析**: 评估结果对假设违反的敏感性
4. **诚实汇报**: 说明假设的合理性和潜在限制

### 下一步

根据推荐的方法,探索相应的模块:
- **PSM/IPW/DR** → Part 2: 观测数据方法
- **DID** → Part 3: 准实验设计
- **CATE** → Part 4: 异质性效应分析
    """

    return {
        "charts": [
            fig_to_dict(decision_tree_fig),
            fig_to_dict(comparison_fig)
        ],
        "tables": [],
        "summary": summary,
        "metrics": {
            "recommended_method": strategy['recommended_method'],
            "priority": strategy['priority'],
            "data_type": data_type
        }
    }


def analyze_bias_comparison() -> Dict[str, Any]:
    """
    偏差类型对比分析 API

    Returns:
    --------
    标准 API 响应格式
    """
    # 生成各种偏差的演示
    simpson_fig, simpson_stats = demonstrate_simpsons_paradox()
    berkson_fig, berkson_stats = demonstrate_berksons_paradox()
    measurement_fig, measurement_stats = _analyze_measurement_bias(
        n_samples=1000,
        measurement_error_std=1.0
    )

    summary = f"""
## 偏差类型对比

因果推断中的主要偏差类型及其特征:

### 1. 混淆偏差 (Confounding Bias)

**定义**: 混淆变量同时影响处理和结果,导致虚假关联

**DAG**: T ← X → Y

**示例**: 咖啡消费与心脏病 (混淆: 吸烟)

**解决**: PSM, IPW, DR, 回归调整

---

### 2. 选择偏差 (Selection Bias)

**定义**: 样本选择与研究变量相关

**DAG**: T → 选择 ← Y (碰撞变量)

**示例**:
- 生存偏差: 只看到成功的企业
- 自选择: 效果好的人留在研究中

**解决**: Heckman 校正, Bounds 分析

---

### 3. 碰撞偏差 (Collider Bias / Berkson's Paradox)

**定义**: 控制碰撞变量打开虚假关联

**DAG**: T → C ← Y

**示例**: 医院数据中疾病的虚假负相关

**Berkson's Paradox 统计**:
- 总体相关性: {berkson_stats['full_correlation']:.3f} (独立)
- 住院人群相关性: {berkson_stats['selected_correlation']:.3f} (负相关!)

**解决**: **不要控制碰撞变量!**

---

### 4. 辛普森悖论 (Simpson's Paradox)

**定义**: 整体趋势与分层趋势相反

**本质**: 混淆偏差的一种特殊形式

**Simpson's Paradox 统计**:
- 整体效应: {simpson_stats['overall_effect']:+.2f}
- 组 A 效应: {simpson_stats['effect_group_a']:+.2f}
- 组 B 效应: {simpson_stats['effect_group_b']:+.2f}
- 真实效应: {simpson_stats['true_effect']:+.2f}

**教训**: 永远检查分层分析!

---

### 5. 测量偏差 (Measurement Error / Attenuation Bias)

**定义**: 变量测量存在误差,导致估计偏向 0

**统计**:
- 真实 ATE: {measurement_stats['true_ate']:.3f}
- 使用真实 X: {measurement_stats['ate_with_true_x']:.3f}
- 使用测量 X: {measurement_stats['ate_with_measured_x']:.3f}
- 衰减偏差: {measurement_stats['attenuation_bias']:.3f}

**解决**: 工具变量, 多重测量, 校正公式

---

## 偏差识别决策树

```
是否存在 X 同时影响 T 和 Y?
├─ 是 → 混淆偏差 → 控制 X
└─ 否 → 继续

是否存在 C 被 T 和 Y 同时影响?
├─ 是 → 碰撞变量 → 不要控制 C!
└─ 否 → 继续

样本选择是否与结果相关?
├─ 是 → 选择偏差 → Heckman 校正
└─ 否 → 继续

变量测量是否有误差?
├─ 是 → 测量偏差 → IV 或校正
└─ 否 → 无主要偏差
```

---

## 实践建议

1. **画 DAG**: 可视化你的因果假设
2. **识别偏差**: 根据 DAG 结构识别潜在偏差
3. **选择方法**: 根据偏差类型选择合适的方法
4. **检验假设**: 尽可能检验方法的假设
5. **敏感性分析**: 评估结果的稳健性

## 下一步

- 混淆偏差 → Part 2: 观测数据方法
- 选择偏差 → Heckman 模型 (Part 7: 高级主题)
- 测量误差 → 工具变量 (Part 3: 准实验)
    """

    return {
        "charts": [
            fig_to_dict(simpson_fig),
            fig_to_dict(berkson_fig),
            fig_to_dict(measurement_fig)
        ],
        "tables": [],
        "summary": summary,
        "metrics": {
            "simpsons_paradox": simpson_stats,
            "berksons_paradox": berkson_stats,
            "measurement_bias": measurement_stats
        }
    }
