"""Part 2: Observational Methods API - 观测数据因果推断方法"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, Optional

from app.schemas.common import ApiResponse, AnalysisResult, convert_numpy_types

router = APIRouter()


# ==================== 请求/响应模型 ====================


class PSMRequest(BaseModel):
    """倾向得分匹配请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    confounding_strength: float = Field(1.0, ge=0, le=3.0, description="混淆强度")
    caliper: float = Field(0.2, ge=0.01, le=1.0, description="卡尺宽度")
    n_neighbors: int = Field(1, ge=1, le=5, description="匹配邻居数")


class MatchingMethodsRequest(BaseModel):
    """匹配方法对比请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    confounding_strength: float = Field(1.0, ge=0, le=3.0, description="混淆强度")


class IPWRequest(BaseModel):
    """逆概率加权请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    confounding_strength: float = Field(1.0, ge=0, le=3.0, description="混淆强度")
    stabilized: bool = Field(True, description="是否使用稳定权重")
    trimming: float = Field(0.01, ge=0.0, le=0.2, description="倾向得分截断阈值")


class DoublyRobustRequest(BaseModel):
    """双重稳健估计请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    confounding_strength: float = Field(1.0, ge=0, le=3.0, description="混淆强度")
    method: Literal["aipw", "tmle", "dr"] = Field("aipw", description="方法选择")


class SensitivityRequest(BaseModel):
    """敏感性分析请求"""

    estimated_ate: float = Field(..., description="估计的 ATE")
    ci_lower: Optional[float] = Field(None, description="置信区间下界")
    effect_type: Literal["mean_difference", "risk_ratio", "odds_ratio"] = Field(
        "mean_difference", description="效应类型"
    )


# ==================== API 端点 ====================


@router.post("/psm", response_model=ApiResponse[AnalysisResult])
async def analyze_psm_api(request: PSMRequest):
    """倾向得分匹配分析"""
    from domains.part2_observational.api import analyze_psm

    result = analyze_psm(
        n_samples=request.n_samples,
        confounding_strength=request.confounding_strength,
        caliper=request.caliper,
        n_neighbors=request.n_neighbors,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/matching-methods", response_model=ApiResponse[AnalysisResult])
async def analyze_matching_methods_api(request: MatchingMethodsRequest):
    """匹配方法对比分析"""
    from domains.part2_observational.api import analyze_matching_methods

    result = analyze_matching_methods(
        n_samples=request.n_samples,
        confounding_strength=request.confounding_strength,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/ipw", response_model=ApiResponse[AnalysisResult])
async def analyze_ipw_api(request: IPWRequest):
    """逆概率加权分析"""
    from domains.part2_observational.api import analyze_ipw

    result = analyze_ipw(
        n_samples=request.n_samples,
        confounding_strength=request.confounding_strength,
        stabilized=request.stabilized,
        trimming=request.trimming,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/doubly-robust", response_model=ApiResponse[AnalysisResult])
async def analyze_doubly_robust_api(request: DoublyRobustRequest):
    """双重稳健估计分析"""
    from domains.part2_observational.api import analyze_doubly_robust

    result = analyze_doubly_robust(
        n_samples=request.n_samples,
        confounding_strength=request.confounding_strength,
        method=request.method,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/sensitivity", response_model=ApiResponse[AnalysisResult])
async def analyze_sensitivity_api(request: SensitivityRequest):
    """敏感性分析 (E-value)"""
    from domains.part2_observational.api import analyze_sensitivity

    result = analyze_sensitivity(
        estimated_ate=request.estimated_ate,
        ci_lower=request.ci_lower,
        effect_type=request.effect_type,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))
