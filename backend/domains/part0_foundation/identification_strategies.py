"""
识别策略框架 (Identification Strategies)

帮助用户选择合适的因果推断方法的决策系统

核心问题:
1. 你有什么数据? (实验 vs 观测)
2. 你观测到混淆变量了吗?
3. 你有工具变量吗?
4. 你有面板数据吗?
5. 你有断点/不连续性吗?

方法映射:
- 随机实验 (RCT) → 简单差分
- 观测数据 + 观测到混淆 → PSM, IPW, Doubly Robust
- 观测数据 + 未观测混淆 + 工具变量 → IV
- 面板数据 → DID, Fixed Effects
- 断点设计 → RDD
- 合成控制 → Synthetic Control
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple


def create_strategy_decision_tree() -> go.Figure:
    """创建识别策略决策树可视化"""

    # 使用 Sankey 图展示决策流程
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[
                # Level 0
                "开始",
                # Level 1
                "随机实验 (RCT)",
                "观测数据",
                # Level 2 (RCT)
                "简单差分/T-test",
                # Level 2 (观测数据)
                "观测到混淆变量",
                "未观测混淆",
                # Level 3 (观测到混淆)
                "PSM/IPW/DR",
                "回归调整",
                # Level 3 (未观测混淆)
                "有工具变量",
                "有面板数据",
                "有断点",
                "其他",
                # Level 4
                "IV/2SLS",
                "DID/FE",
                "RDD",
                "敏感性分析"
            ],
            color=[
                "#2D9CDB",  # 开始
                "#27AE60", "#EB5757",  # Level 1
                "#27AE60",  # RCT → 简单差分
                "#FFE66D", "#E74C3C",  # 观测数据分支
                "#95E1D3", "#95E1D3",  # 观测到混淆方法
                "#9B59B6", "#9B59B6", "#9B59B6", "#666",  # 未观测混淆条件
                "#9B59B6", "#9B59B6", "#9B59B6", "#666"  # 最终方法
            ]
        ),
        link=dict(
            source=[0, 0,  # 开始 → Level 1
                    1,  # RCT → 简单差分
                    2, 2,  # 观测数据 → Level 2
                    4, 4,  # 观测到混淆 → Level 3
                    5, 5, 5, 5,  # 未观测混淆 → Level 3
                    8, 9, 10, 11],  # Level 3 → Level 4
            target=[1, 2,  # Level 1
                    3,  # 简单差分
                    4, 5,  # Level 2
                    6, 7,  # PSM/回归
                    8, 9, 10, 11,  # 工具变量等
                    12, 13, 14, 15],  # 最终方法
            value=[3, 7,  # 实验vs观测分配
                   3,  # RCT权重
                   4, 3,  # 观测到vs未观测
                   2, 2,  # PSM vs 回归
                   1, 1, 0.5, 0.5,  # IV/DID/RDD/其他
                   1, 1, 0.5, 0.5],  # 最终权重
            color=[
                "#27AE60", "#EB5757",
                "#27AE60",
                "#FFE66D", "#E74C3C",
                "#95E1D3", "#95E1D3",
                "#9B59B6", "#9B59B6", "#9B59B6", "#666",
                "#9B59B6", "#9B59B6", "#9B59B6", "#666"
            ]
        )
    )])

    fig.update_layout(
        title="因果推断方法选择决策树",
        font=dict(size=12),
        height=600,
        template='plotly_white'
    )

    return fig


def get_identification_strategy(
    data_type: str,
    confounding_observed: bool = False,
    has_instrument: bool = False,
    has_panel: bool = False,
    has_discontinuity: bool = False
) -> Dict:
    """
    根据数据特征推荐识别策略

    Parameters:
    -----------
    data_type: 'experimental' (实验) 或 'observational' (观测)
    confounding_observed: 是否观测到混淆变量
    has_instrument: 是否有工具变量
    has_panel: 是否有面板数据
    has_discontinuity: 是否有断点/不连续性

    Returns:
    --------
    包含推荐方法、假设条件、优缺点的字典
    """

    if data_type == "experimental":
        return {
            "recommended_method": "简单差分 (Simple Difference) / T-test",
            "assumptions": [
                "随机分配 (Randomization)",
                "SUTVA (稳定单位处理值假设)",
                "无干扰 (No spillover)"
            ],
            "pros": [
                "无需控制混淆变量",
                "因果效应可识别",
                "方法简单直接"
            ],
            "cons": [
                "成本高",
                "可能存在外部有效性问题",
                "样本量要求"
            ],
            "code_example": "ate = Y[T==1].mean() - Y[T==0].mean()",
            "priority": "最优选择"
        }

    # 观测数据
    if confounding_observed:
        return {
            "recommended_method": "倾向得分匹配 (PSM) / 逆概率加权 (IPW) / 双重稳健 (DR)",
            "assumptions": [
                "可忽略性 (Ignorability): (Y(0), Y(1)) ⊥ T | X",
                "共同支撑 (Common Support): 0 < P(T=1|X) < 1",
                "正确的倾向得分模型"
            ],
            "pros": [
                "可以控制观测到的混淆",
                "方法成熟，软件支持好",
                "可视化检验平衡性"
            ],
            "cons": [
                "无法处理未观测混淆",
                "对模型设定敏感",
                "共同支撑假设可能违反"
            ],
            "code_example": "使用 EconML, CausalML 等库",
            "priority": "观测数据首选"
        }

    # 未观测混淆
    if has_instrument:
        return {
            "recommended_method": "工具变量 (IV) / 两阶段最小二乘 (2SLS)",
            "assumptions": [
                "相关性 (Relevance): Cov(Z, T) ≠ 0",
                "排他性 (Exclusion): Z 只通过 T 影响 Y",
                "可忽略性: (Y(0), Y(1)) ⊥ Z"
            ],
            "pros": [
                "可以处理未观测混淆",
                "识别局部平均处理效应 (LATE)",
                "广泛应用于经济学"
            ],
            "cons": [
                "需要找到有效的工具变量 (很难!)",
                "只识别 LATE,不是 ATE",
                "弱工具变量问题"
            ],
            "code_example": "使用 statsmodels.IV2SLS 或 linearmodels",
            "priority": "未观测混淆 + 有IV"
        }

    if has_panel:
        return {
            "recommended_method": "双重差分 (DID) / 固定效应 (FE)",
            "assumptions": [
                "平行趋势 (Parallel Trends)",
                "无预期效应 (No anticipation)",
                "SUTVA"
            ],
            "pros": [
                "控制时间不变的未观测混淆",
                "政策评估的黄金标准",
                "可视化检验平行趋势"
            ],
            "cons": [
                "平行趋势假设不可检验 (只能间接)",
                "需要处理前后对照",
                "可能存在动态效应"
            ],
            "code_example": "使用 linearmodels.PanelOLS 或 pyfixest",
            "priority": "面板数据首选"
        }

    if has_discontinuity:
        return {
            "recommended_method": "断点回归 (RDD)",
            "assumptions": [
                "连续性假设: E[Y(0)|X] 和 E[Y(1)|X] 在断点处连续",
                "无操纵 (No manipulation)",
                "局部随机化"
            ],
            "pros": [
                "断点处近似随机实验",
                "可信度高",
                "可视化检验假设"
            ],
            "cons": [
                "只识别断点处的局部效应",
                "需要足够的断点附近样本",
                "对带宽选择敏感"
            ],
            "code_example": "使用 rdd 包或手动实现",
            "priority": "有断点设计时的首选"
        }

    # 默认: 无法识别
    return {
        "recommended_method": "敏感性分析 / 边界分析",
        "assumptions": [
            "需要对未观测混淆的强度做假设"
        ],
        "pros": [
            "可以评估估计的稳健性",
            "诚实地承认识别问题"
        ],
        "cons": [
            "无法得到点估计",
            "需要主观假设"
        ],
        "code_example": "使用 sensemakr 或自定义",
        "priority": "最后的选择"
    }


def evaluate_identification_assumptions(
    method: str
) -> Dict:
    """
    评估特定方法的识别假设

    Returns:
    --------
    假设列表和检验方法
    """
    assumptions_tests = {
        "PSM": {
            "ignorability": {
                "assumption": "(Y(0), Y(1)) ⊥ T | X",
                "testable": False,
                "indirect_test": "检查平衡性: 匹配后协变量分布是否平衡"
            },
            "common_support": {
                "assumption": "0 < P(T=1|X) < 1",
                "testable": True,
                "test_method": "绘制倾向得分分布图，检查重叠区域"
            },
            "no_unmeasured_confounding": {
                "assumption": "没有未观测混淆变量",
                "testable": False,
                "indirect_test": "敏感性分析 (Rosenbaum bounds)"
            }
        },
        "IV": {
            "relevance": {
                "assumption": "Cov(Z, T) ≠ 0",
                "testable": True,
                "test_method": "第一阶段 F 统计量 > 10"
            },
            "exclusion": {
                "assumption": "Z 只通过 T 影响 Y",
                "testable": False,
                "indirect_test": "领域知识论证"
            },
            "independence": {
                "assumption": "(Y(0), Y(1)) ⊥ Z",
                "testable": False,
                "indirect_test": "随机分配的 IV 自动满足"
            }
        },
        "DID": {
            "parallel_trends": {
                "assumption": "无处理时,两组趋势平行",
                "testable": "部分",
                "test_method": "检查处理前趋势、事件研究图"
            },
            "no_anticipation": {
                "assumption": "处理前无预期效应",
                "testable": True,
                "test_method": "检查处理前几期的系数"
            },
            "no_composition_change": {
                "assumption": "组内构成不变",
                "testable": True,
                "test_method": "检查样本流失模式"
            }
        },
        "RDD": {
            "continuity": {
                "assumption": "Y(0), Y(1) 在断点处连续",
                "testable": "间接",
                "test_method": "检查协变量在断点处的连续性"
            },
            "no_manipulation": {
                "assumption": "个体无法精确操纵 running variable",
                "testable": True,
                "test_method": "McCrary 密度检验"
            },
            "local_randomization": {
                "assumption": "断点附近近似随机",
                "testable": True,
                "test_method": "检查协变量平衡性"
            }
        }
    }

    return assumptions_tests.get(method, {})


def recommend_methods(
    data_characteristics: Dict
) -> List[Tuple[str, float, str]]:
    """
    根据数据特征推荐多个方法,按优先级排序

    Parameters:
    -----------
    data_characteristics: 包含数据特征的字典
        {
            'is_experimental': bool,
            'confounders_observed': bool,
            'has_instrument': bool,
            'has_panel': bool,
            'has_discontinuity': bool,
            'sample_size': int
        }

    Returns:
    --------
    List of (method_name, priority_score, reason)
    """
    recommendations = []

    # 实验数据
    if data_characteristics.get('is_experimental'):
        recommendations.append((
            "随机实验分析 (Simple Difference)",
            1.0,
            "金标准: 随机分配消除混淆"
        ))
        return recommendations

    # 观测数据 + 观测到混淆
    if data_characteristics.get('confounders_observed'):
        recommendations.append((
            "倾向得分匹配 (PSM)",
            0.85,
            "可控制观测到的混淆，可视化好"
        ))
        recommendations.append((
            "逆概率加权 (IPW)",
            0.8,
            "利用所有数据，效率高"
        ))
        recommendations.append((
            "双重稳健估计 (DR)",
            0.9,
            "结合结果模型和倾向得分，更稳健"
        ))

    # 面板数据
    if data_characteristics.get('has_panel'):
        recommendations.append((
            "双重差分 (DID)",
            0.85,
            "控制时间不变的未观测混淆"
        ))
        recommendations.append((
            "固定效应模型 (FE)",
            0.8,
            "控制个体异质性"
        ))

    # 工具变量
    if data_characteristics.get('has_instrument'):
        recommendations.append((
            "工具变量 (IV/2SLS)",
            0.75,
            "可处理未观测混淆,但需要强IV"
        ))

    # 断点设计
    if data_characteristics.get('has_discontinuity'):
        recommendations.append((
            "断点回归 (RDD)",
            0.9,
            "断点处近似随机，可信度高"
        ))

    # 样本量考虑
    sample_size = data_characteristics.get('sample_size', 1000)
    if sample_size < 100:
        recommendations.append((
            "合成控制法 (Synthetic Control)",
            0.6,
            "小样本可用，但需要长时间序列"
        ))

    # 默认: 敏感性分析
    if not recommendations:
        recommendations.append((
            "敏感性分析",
            0.3,
            "无法满足其他方法的假设，评估稳健性"
        ))

    # 按优先级排序
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations


def create_method_comparison_table() -> go.Figure:
    """创建方法对比表格"""

    methods = [
        "RCT", "PSM", "IPW", "DR", "IV", "DID", "RDD"
    ]

    criteria = {
        "处理未观测混淆": ["✓", "✗", "✗", "✗", "✓", "✓(时间不变)", "✓(断点处)"],
        "数据要求": ["高", "中", "中", "中", "中", "面板", "断点"],
        "假设可检验性": ["高", "中", "中", "中", "中", "中", "高"],
        "因果可信度": ["最高", "中", "中", "中高", "高", "高", "高"],
        "适用场景": ["实验", "观测+混淆", "观测+混淆", "观测+混淆", "未观测混淆+IV", "面板数据", "断点设计"]
    }

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>方法</b>'] + methods,
            fill_color='#2D9CDB',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                ['<b>' + k + '</b>' for k in criteria.keys()],
                *[criteria[k] for k in criteria.keys()]
            ],
            fill_color=[['#F0F0F0']*len(criteria), *[['white']*len(criteria)]*(len(methods))],
            align=['left'] + ['center']*len(methods),
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title="因果推断方法对比",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig
