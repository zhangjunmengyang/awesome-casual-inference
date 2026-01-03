"""
UpliftLab - 增益模型实验室

包含:
- meta_learners: S/T/X/R Learner 实现与对比
- uplift_tree: Uplift 决策树可视化
- cate_comparison: CATE 估计对比
- evaluation: Qini/Uplift 曲线评估
"""

from . import meta_learners
from . import uplift_tree
from . import cate_comparison
from . import evaluation
