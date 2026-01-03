"""
Metrics Framework - 指标体系设计模块

帮助设计和管理业务指标体系：
1. 指标分类：北极星指标、过程指标、护栏指标
2. 指标定义：口径、计算逻辑、数据源
3. 指标关系：指标树、因果关系
4. 监控体系：异常检测、归因分析

核心模块：
---------
- metric_design: 指标设计方法论
- metric_tree: 指标树构建
- anomaly_detection: 异常检测
- metric_attribution: 指标归因

设计理念：
---------
- 以业务目标为导向
- 指标可解释、可行动
- 避免虚荣指标
"""

from . import metric_design
from . import metric_tree

__all__ = [
    'metric_design',
    'metric_tree',
]
