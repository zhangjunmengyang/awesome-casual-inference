"""Part 3: 准实验方法 API 适配层

将核心分析函数适配为统一的API格式。
"""

from .did import analyze_did_basic, analyze_did_event_study, analyze_did_staggered
from .synthetic_control import analyze_synthetic_control
from .rdd import analyze_rdd_sharp, analyze_rdd_fuzzy
from .instrumental_variables import analyze_iv


# 所有API函数都已在各自模块中实现，直接导出即可
__all__ = [
    "analyze_did_basic",
    "analyze_did_event_study",
    "analyze_did_staggered",
    "analyze_synthetic_control",
    "analyze_rdd_sharp",
    "analyze_rdd_fuzzy",
    "analyze_iv",
]
