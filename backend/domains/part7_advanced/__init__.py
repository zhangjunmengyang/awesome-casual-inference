"""Part 7: 高级主题模块

本模块包含因果推断的高级主题：
- 因果发现 (Causal Discovery)
- 连续处理效应 (Continuous Treatment)
- 时变处理效应 (Time-Varying Treatment)
- 中介分析 (Mediation Analysis)
"""

from .api import (
    analyze_causal_discovery,
    analyze_continuous_treatment,
    analyze_time_varying_treatment,
    analyze_mediation,
)

__all__ = [
    "analyze_causal_discovery",
    "analyze_continuous_treatment",
    "analyze_time_varying_treatment",
    "analyze_mediation",
]
