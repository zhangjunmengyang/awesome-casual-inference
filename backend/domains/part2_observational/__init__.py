"""
Part 2: 观测数据因果推断方法

本模块实现基于观测数据的因果推断方法，主要包括:
- 倾向得分估计
- 匹配方法 (PSM, CEM, 马氏距离)
- 加权方法 (IPW, 稳定权重, 重叠权重)
- 双重稳健估计
- 敏感性分析

对应 notebooks/part2_observational
"""

from .propensity_score import (
    PropensityScoreEstimator,
    PropensityScoreMatching
)

from .matching import (
    NearestNeighborMatching,
    CovariateExactMatching,
    MahalanobisMatching,
    KernelMatching
)

from .weighting import (
    IPWEstimator,
    StabilizedIPW,
    OverlapWeighting,
    TrimmedIPW
)

from .doubly_robust import (
    DoublyRobustEstimator,
    AIPWEstimator,
    TMLEEstimator
)

from .sensitivity_analysis import (
    RosenbaumBounds,
    EValueAnalysis,
    ConfoundingBiasAnalysis
)

__all__ = [
    # Propensity Score
    'PropensityScoreEstimator',
    'PropensityScoreMatching',

    # Matching
    'NearestNeighborMatching',
    'CovariateExactMatching',
    'MahalanobisMatching',
    'KernelMatching',

    # Weighting
    'IPWEstimator',
    'StabilizedIPW',
    'OverlapWeighting',
    'TrimmedIPW',

    # Doubly Robust
    'DoublyRobustEstimator',
    'AIPWEstimator',
    'TMLEEstimator',

    # Sensitivity
    'RosenbaumBounds',
    'EValueAnalysis',
    'ConfoundingBiasAnalysis',
]
