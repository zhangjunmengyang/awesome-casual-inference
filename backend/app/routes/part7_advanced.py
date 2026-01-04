"""Part 7: Advanced Topics API - 高级因果推断主题"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

from app.schemas.common import ApiResponse, AnalysisResult, convert_numpy_types

router = APIRouter()


# ==================== 请求/响应模型 ====================


class CausalDiscoveryRequest(BaseModel):
    """因果发现请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_variables: int = Field(5, ge=3, le=15, description="变量数量")
    edge_density: float = Field(0.3, ge=0.1, le=0.8, description="边密度")
    algorithm: Literal["pc", "ges", "notears", "lingam"] = Field(
        "pc", description="因果发现算法"
    )


class ContinuousTreatmentRequest(BaseModel):
    """连续处理变量请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(5, ge=2, le=20, description="特征数量")
    treatment_effect_type: Literal["linear", "nonlinear", "threshold"] = Field(
        "linear", description="处理效应类型"
    )
    confounding_strength: float = Field(1.0, ge=0, le=3.0, description="混淆强度")


class TimeVaryingTreatmentRequest(BaseModel):
    """时变处理请求"""

    n_subjects: int = Field(100, ge=20, le=500, description="个体数量")
    n_timepoints: int = Field(10, ge=5, le=50, description="时间点数量")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="处理效应")
    time_varying_confounding: bool = Field(True, description="是否有时变混淆")


class MediationRequest(BaseModel):
    """中介分析请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    treatment_to_mediator: float = Field(0.5, ge=-2, le=2, description="处理到中介的效应")
    mediator_to_outcome: float = Field(0.8, ge=-2, le=2, description="中介到结果的效应")
    direct_effect: float = Field(0.3, ge=-2, le=2, description="直接效应")
    interaction: bool = Field(False, description="处理与中介是否有交互")


# ==================== API 端点 ====================


@router.post("/causal-discovery", response_model=ApiResponse[AnalysisResult])
async def analyze_causal_discovery_api(request: CausalDiscoveryRequest):
    """因果发现分析"""
    from domains.part7_advanced.api import analyze_causal_discovery

    result = analyze_causal_discovery(
        n_samples=request.n_samples,
        n_variables=request.n_variables,
        edge_density=request.edge_density,
        algorithm=request.algorithm,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/continuous-treatment", response_model=ApiResponse[AnalysisResult])
async def analyze_continuous_treatment_api(request: ContinuousTreatmentRequest):
    """连续处理变量分析"""
    from domains.part7_advanced.api import analyze_continuous_treatment

    result = analyze_continuous_treatment(
        n_samples=request.n_samples,
        n_features=request.n_features,
        treatment_effect_type=request.treatment_effect_type,
        confounding_strength=request.confounding_strength,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/time-varying-treatment", response_model=ApiResponse[AnalysisResult])
async def analyze_time_varying_treatment_api(request: TimeVaryingTreatmentRequest):
    """时变处理分析"""
    from domains.part7_advanced.api import analyze_time_varying_treatment

    result = analyze_time_varying_treatment(
        n_subjects=request.n_subjects,
        n_timepoints=request.n_timepoints,
        treatment_effect=request.treatment_effect,
        time_varying_confounding=request.time_varying_confounding,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/mediation", response_model=ApiResponse[AnalysisResult])
async def analyze_mediation_api(request: MediationRequest):
    """中介分析"""
    from domains.part7_advanced.api import analyze_mediation

    result = analyze_mediation(
        n_samples=request.n_samples,
        treatment_to_mediator=request.treatment_to_mediator,
        mediator_to_outcome=request.mediator_to_outcome,
        direct_effect=request.direct_effect,
        interaction=request.interaction,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))
