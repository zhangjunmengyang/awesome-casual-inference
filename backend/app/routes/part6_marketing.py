"""Part 6: Marketing Applications API - 营销场景应用"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

from app.schemas.common import ApiResponse, AnalysisResult, convert_numpy_types

router = APIRouter()


# ==================== 请求/响应模型 ====================


class AttributionRequest(BaseModel):
    """营销归因分析请求"""

    n_users: int = Field(1000, ge=100, le=10000, description="用户数量")
    n_touchpoints: int = Field(5, ge=2, le=20, description="触点数量")
    conversion_rate: float = Field(0.05, ge=0.01, le=0.5, description="转化率")
    attribution_model: Literal[
        "last_touch",
        "first_touch",
        "linear",
        "time_decay",
        "position_based",
        "shapley",
        "markov"
    ] = Field("shapley", description="归因模型")


class CouponOptimizationRequest(BaseModel):
    """优惠券优化请求"""

    n_users: int = Field(5000, ge=1000, le=50000, description="用户数量")
    n_features: int = Field(10, ge=5, le=30, description="特征数量")
    coupon_values: List[float] = Field(
        [0, 5, 10, 20],
        description="优惠券面额列表"
    )
    budget_constraint: Optional[float] = Field(None, ge=0, description="预算约束")


class UserTargetingRequest(BaseModel):
    """用户定向请求"""

    n_users: int = Field(10000, ge=1000, le=100000, description="用户数量")
    n_features: int = Field(15, ge=5, le=50, description="特征数量")
    heterogeneity_strength: float = Field(1.0, ge=0, le=3.0, description="异质性强度")
    targeting_percentile: float = Field(0.2, ge=0.05, le=0.5, description="定向比例")


class BudgetAllocationRequest(BaseModel):
    """预算分配请求"""

    n_channels: int = Field(5, ge=2, le=20, description="渠道数量")
    total_budget: float = Field(100000, ge=10000, le=10000000, description="总预算")
    roi_variability: float = Field(0.5, ge=0.1, le=2.0, description="ROI 变异性")
    diminishing_returns: bool = Field(True, description="是否考虑边际递减")


# ==================== API 端点 ====================


@router.post("/attribution", response_model=ApiResponse[AnalysisResult])
async def analyze_attribution_api(request: AttributionRequest):
    """营销归因分析"""
    from domains.part6_marketing.api import analyze_attribution

    result = analyze_attribution(
        n_users=request.n_users,
        n_touchpoints=request.n_touchpoints,
        conversion_rate=request.conversion_rate,
        attribution_model=request.attribution_model,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/coupon-optimization", response_model=ApiResponse[AnalysisResult])
async def analyze_coupon_optimization_api(request: CouponOptimizationRequest):
    """优惠券优化分析"""
    from domains.part6_marketing.api import analyze_coupon_optimization

    result = analyze_coupon_optimization(
        n_users=request.n_users,
        n_features=request.n_features,
        coupon_values=request.coupon_values,
        budget_constraint=request.budget_constraint,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/user-targeting", response_model=ApiResponse[AnalysisResult])
async def analyze_user_targeting_api(request: UserTargetingRequest):
    """用户定向分析"""
    from domains.part6_marketing.api import analyze_user_targeting

    result = analyze_user_targeting(
        n_users=request.n_users,
        n_features=request.n_features,
        heterogeneity_strength=request.heterogeneity_strength,
        targeting_percentile=request.targeting_percentile,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/budget-allocation", response_model=ApiResponse[AnalysisResult])
async def analyze_budget_allocation_api(request: BudgetAllocationRequest):
    """预算分配分析"""
    from domains.part6_marketing.api import analyze_budget_allocation

    result = analyze_budget_allocation(
        n_channels=request.n_channels,
        total_budget=request.total_budget,
        roi_variability=request.roi_variability,
        diminishing_returns=request.diminishing_returns,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))
