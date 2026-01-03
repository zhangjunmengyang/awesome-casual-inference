"""
SQL Analytics - 业务 SQL 分析模块

提供常见业务分析场景的 SQL 模板和最佳实践：
1. retention: 留存分析（N日留存、同期群分析）
2. funnel: 漏斗分析（转化漏斗、流失分析）
3. cohort: 同期群分析（用户生命周期、LTV）
4. attribution: 归因分析（渠道归因、转化归因）

设计理念：
---------
- 提供可复用的 SQL 模板
- 结合 Python 进行可视化
- 面试常见 SQL 题目覆盖
"""

from . import retention
from . import funnel

__all__ = [
    'retention',
    'funnel',
]
