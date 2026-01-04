"""
Part 1: Experimentation Methods
实验方法论模块

核心内容:
--------
1. A/B Testing Basics - A/B测试基础
2. CUPED - 方差缩减技术
3. Stratified Analysis - 分层分析
4. Network Effects - 网络效应与溢出
5. Switchback Experiments - Switchback实验
6. Long-term Effects - 长期效应估计
7. Multi-Armed Bandits - 多臂老虎机

设计理念:
--------
- 循序渐进：从基础到高级
- 理论与实践结合：公式 + 代码 + 可视化
- 业务导向：解决真实问题

API 导出:
---------
所有 API 函数统一返回格式:
{
    "charts": [...],      # Plotly图表JSON列表
    "tables": [...],      # 数据表格列表
    "summary": "...",     # 文字总结
    "metrics": {...}      # 关键指标字典
}
"""

from . import ab_testing
from . import cuped
from . import stratified_analysis
from . import network_effects
from . import switchback
from . import long_term_effects
from . import multi_armed_bandits
from . import utils
from . import api

# Export API functions
from .api import (
    analyze_ab_test,
    apply_cuped,
    stratified_analysis as run_stratified_analysis,
    analyze_network_effects,
    analyze_switchback,
    estimate_long_term_effects,
    run_bandit_simulation,
)

__all__ = [
    # Modules
    'ab_testing',
    'cuped',
    'stratified_analysis',
    'network_effects',
    'switchback',
    'long_term_effects',
    'multi_armed_bandits',
    'utils',
    'api',
    # API functions
    'analyze_ab_test',
    'apply_cuped',
    'run_stratified_analysis',
    'analyze_network_effects',
    'analyze_switchback',
    'estimate_long_term_effects',
    'run_bandit_simulation',
]
