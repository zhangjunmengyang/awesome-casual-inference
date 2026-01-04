"""
Part 4: CATE & Uplift Modeling

异质性处理效应估计与 Uplift 建模模块

主要内容:
- Meta-Learners (S/T/X/R/DR-Learner)
- Causal Forest (因果森林)
- Uplift Tree (Uplift 决策树)
- Uplift 评估方法 (Qini, AUUC)
- CATE 可视化与子群体识别

该模块合并了原有的 uplift_lab 和 hetero_effect_lab 的功能。
"""

from .meta_learners import SLearner, TLearner, XLearner, RLearner, DRLearner
from .causal_forest import SimpleTLearner
from .uplift_tree import calculate_uplift_gain, find_best_split
from .uplift_evaluation import calculate_qini_curve, calculate_uplift_curve, calculate_auuc
from .cate_visualization import TLearnerWithCI, analyze_cate_by_features

__all__ = [
    # Meta-Learners
    'SLearner',
    'TLearner',
    'XLearner',
    'RLearner',
    'DRLearner',
    # Causal Forest
    'SimpleTLearner',
    # Uplift Tree
    'calculate_uplift_gain',
    'find_best_split',
    # Uplift Evaluation
    'calculate_qini_curve',
    'calculate_uplift_curve',
    'calculate_auuc',
    # CATE Visualization
    'TLearnerWithCI',
    'analyze_cate_by_features',
]
