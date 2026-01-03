"""
Business Cases - 端到端业务案例模块

从真实业务问题出发，展示完整的数据科学工作流：
问题定义 → 数据探索 → 方法选择 → 模型构建 → 业务解读 → 落地建议

核心案例：
---------
1. marketing_roi: 智能营销ROI优化 - 预算分配、人群圈选、增量归因
2. growth_attribution: 用户增长归因 - 渠道归因、LTV预测、增长实验
3. pricing_optimization: 定价策略优化 - 价格弹性、动态定价、促销效果

设计理念：
---------
- 每个案例都是完整的业务闭环，不只是算法演示
- 包含业务背景、数据探索、方法对比、结论解读
- 强调"so what" - 分析结果如何指导业务决策
"""

from . import marketing_roi
from . import growth_attribution
from . import pricing_optimization

__all__ = [
    'marketing_roi',
    'growth_attribution',
    'pricing_optimization',
]
