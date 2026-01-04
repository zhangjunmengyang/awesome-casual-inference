"""Part 3: 准实验方法模块

包含以下方法:
- 双重差分 (DID)
- 合成控制 (Synthetic Control)
- 断点回归 (RDD)
- 工具变量 (IV)
"""

from .did import analyze_did_basic, analyze_did_event_study, analyze_did_staggered
from .synthetic_control import analyze_synthetic_control
from .rdd import analyze_rdd_sharp, analyze_rdd_fuzzy
from .instrumental_variables import analyze_iv

__all__ = [
    "analyze_did_basic",
    "analyze_did_event_study",
    "analyze_did_staggered",
    "analyze_synthetic_control",
    "analyze_rdd_sharp",
    "analyze_rdd_fuzzy",
    "analyze_iv",
]
