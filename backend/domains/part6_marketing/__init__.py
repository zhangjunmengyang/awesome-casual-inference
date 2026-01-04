"""Part 6: Marketing Applications - 营销应用模块

包含营销归因、智能发券、用户定向、预算分配等营销场景的因果推断应用。
"""

from .attribution import (
    MarketingAttribution,
    RuleBasedAttribution,
    ShapleyAttribution,
    MarkovAttribution,
)
from .coupon_optimization import CouponOptimizer, UserSegmentation
from .user_targeting import TLearner, XLearner, PolicyLearner
from .budget_allocation import BudgetOptimizer, ResponseCurveModel
from .utils import (
    generate_user_journey_data,
    generate_marketing_data,
    generate_driver_data,
)

__all__ = [
    # Attribution
    "MarketingAttribution",
    "RuleBasedAttribution",
    "ShapleyAttribution",
    "MarkovAttribution",
    # Coupon Optimization
    "CouponOptimizer",
    "UserSegmentation",
    # User Targeting
    "TLearner",
    "XLearner",
    "PolicyLearner",
    # Budget Allocation
    "BudgetOptimizer",
    "ResponseCurveModel",
    # Utils
    "generate_user_journey_data",
    "generate_marketing_data",
    "generate_driver_data",
]
