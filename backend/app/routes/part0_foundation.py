"""Part 0: Foundation Lab API - 因果推断基础"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal

from app.schemas.common import ApiResponse, AnalysisResult

router = APIRouter()


# ==================== 请求/响应模型 ====================


class PotentialOutcomesRequest(BaseModel):
    """潜在结果分析请求"""

    n_samples: int = Field(500, ge=100, le=5000, description="样本数量")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")
    noise_std: float = Field(1.0, ge=0.1, le=5.0, description="噪声标准差")
    confounding_strength: float = Field(0.0, ge=0, le=2.0, description="混淆强度")


class CausalDAGRequest(BaseModel):
    """因果图分析请求"""

    scenario: Literal["confounding", "mediation", "collider", "complex"] = Field(
        "confounding", description="场景: confounding, mediation, collider, complex"
    )


class ConfoundingBiasRequest(BaseModel):
    """混淆偏差分析请求"""

    n_samples: int = Field(1000, ge=100, le=5000)
    confounding_strength: float = Field(1.0, ge=0, le=3.0)
    treatment_effect: float = Field(2.0, ge=-5, le=5)


class SelectionBiasRequest(BaseModel):
    """选择偏差分析请求"""

    n_samples: int = Field(1000, ge=100, le=5000)
    selection_strength: float = Field(1.0, ge=0, le=3.0)
    treatment_effect: float = Field(2.0, ge=-5, le=5)


class IdentificationStrategyRequest(BaseModel):
    """识别策略推荐请求"""

    data_type: Literal["experimental", "observational"] = Field(
        "observational", description="数据类型"
    )
    confounding_observed: bool = Field(False, description="是否观测到混淆变量")
    has_instrument: bool = Field(False, description="是否有工具变量")
    has_panel: bool = Field(False, description="是否有面板数据")
    has_discontinuity: bool = Field(False, description="是否有断点")


# ==================== API 端点 ====================


@router.post("/potential-outcomes", response_model=ApiResponse[AnalysisResult])
async def analyze_potential_outcomes_api(request: PotentialOutcomesRequest):
    """潜在结果框架分析"""
    from domains.part0_foundation.api import analyze_potential_outcomes

    result = analyze_potential_outcomes(
        n_samples=request.n_samples,
        treatment_effect=request.treatment_effect,
        noise_std=request.noise_std,
        confounding_strength=request.confounding_strength,
    )
    return ApiResponse(success=True, data=result)


@router.post("/causal-dag", response_model=ApiResponse[AnalysisResult])
async def analyze_causal_dag_api(request: CausalDAGRequest):
    """因果图分析"""
    from domains.part0_foundation.api import analyze_causal_dag

    result = analyze_causal_dag(scenario=request.scenario)
    return ApiResponse(success=True, data=result)


@router.post("/confounding-bias", response_model=ApiResponse[AnalysisResult])
async def analyze_confounding_bias_api(request: ConfoundingBiasRequest):
    """混淆偏差分析"""
    from domains.part0_foundation.api import analyze_confounding_bias

    result = analyze_confounding_bias(
        n_samples=request.n_samples,
        confounding_strength=request.confounding_strength,
        treatment_effect=request.treatment_effect,
    )
    return ApiResponse(success=True, data=result)


@router.post("/selection-bias", response_model=ApiResponse[AnalysisResult])
async def analyze_selection_bias_api(request: SelectionBiasRequest):
    """选择偏差分析"""
    from domains.part0_foundation.api import analyze_selection_bias

    result = analyze_selection_bias(
        n_samples=request.n_samples,
        selection_strength=request.selection_strength,
        treatment_effect=request.treatment_effect,
    )
    return ApiResponse(success=True, data=result)


@router.post("/identification-strategy", response_model=ApiResponse[AnalysisResult])
async def analyze_identification_strategy_api(request: IdentificationStrategyRequest):
    """识别策略推荐"""
    from domains.part0_foundation.api import analyze_identification_strategy

    result = analyze_identification_strategy(
        data_type=request.data_type,
        confounding_observed=request.confounding_observed,
        has_instrument=request.has_instrument,
        has_panel=request.has_panel,
        has_discontinuity=request.has_discontinuity,
    )
    return ApiResponse(success=True, data=result)


@router.post("/bias-comparison", response_model=ApiResponse[AnalysisResult])
async def analyze_bias_comparison_api():
    """偏差类型对比分析"""
    from domains.part0_foundation.api import analyze_bias_comparison

    result = analyze_bias_comparison()
    return ApiResponse(success=True, data=result)
