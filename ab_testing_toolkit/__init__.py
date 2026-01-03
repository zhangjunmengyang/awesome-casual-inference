"""
A/B Testing Toolkit - 完整的实验分析工具箱

提供端到端的 A/B 测试支持：
1. 实验设计：样本量计算、功效分析
2. 随机化：分流策略、平衡检查
3. 结果分析：统计检验、置信区间
4. 报告生成：自动化实验报告

核心模块：
---------
- sample_size: 样本量计算器
- power_analysis: 功效分析
- experiment_analysis: 实验结果分析
- report_generator: 自动报告生成
- sequential_testing: 序贯检验

设计理念：
---------
- 覆盖完整实验生命周期
- 提供业务友好的接口
- 包含常见陷阱的警告
"""

from . import sample_size
from . import power_analysis
from . import experiment_analysis
from . import report_generator

__all__ = [
    'sample_size',
    'power_analysis',
    'experiment_analysis',
    'report_generator',
]
