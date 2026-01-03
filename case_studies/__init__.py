"""
Case Studies - 精选因果推断案例

包含 4 个精选案例，展示因果推断在真实业务中的应用：
- doordash_delivery: 配送优化 (PSM/DR)
- netflix_recommendation: 推荐系统 (Causal Forest)
- growth_attribution: 渠道归因 (Shapley Value)
- coupon_optimization: 智能发券 (Uplift)
"""

from . import doordash_delivery
from . import netflix_recommendation
from . import growth_attribution
from . import coupon_optimization

__all__ = [
    'doordash_delivery',
    'netflix_recommendation',
    'growth_attribution',
    'coupon_optimization',
]
