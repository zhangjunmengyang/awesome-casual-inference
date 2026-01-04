"""Part 3: Quasi-Experimental Methods API - 准实验方法"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from app.schemas.common import ApiResponse, AnalysisResult, convert_numpy_types

router = APIRouter()


# ==================== 请求/响应模型 ====================


class DIDBasicRequest(BaseModel):
    """基础双重差分请求"""

    n_units: int = Field(100, ge=20, le=500, description="单元数量")
    n_periods: int = Field(10, ge=5, le=50, description="时间周期数")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")
    parallel_trends: bool = Field(True, description="是否满足平行趋势")


class DIDEventStudyRequest(BaseModel):
    """事件研究法请求"""

    n_units: int = Field(100, ge=20, le=500, description="单元数量")
    n_periods: int = Field(20, ge=10, le=100, description="时间周期数")
    treatment_period: int = Field(10, ge=5, le=50, description="处理开始时间")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")


class DIDStaggeredRequest(BaseModel):
    """交错双重差分请求"""

    n_units: int = Field(100, ge=20, le=500, description="单元数量")
    n_periods: int = Field(20, ge=10, le=100, description="时间周期数")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")


class SyntheticControlRequest(BaseModel):
    """合成控制法请求"""

    n_treated_periods: int = Field(10, ge=5, le=50, description="处理后周期数")
    n_control_units: int = Field(20, ge=5, le=100, description="控制单元数量")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")


class RDDSharpRequest(BaseModel):
    """断点回归 (Sharp RDD) 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")
    bandwidth: Optional[float] = Field(None, ge=0.01, le=1.0, description="带宽 (None 为自动选择)")


class RDDFuzzyRequest(BaseModel):
    """模糊断点回归 (Fuzzy RDD) 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")
    compliance_rate: float = Field(0.7, ge=0.1, le=1.0, description="依从率")
    bandwidth: Optional[float] = Field(None, ge=0.01, le=1.0, description="带宽")


class IVRequest(BaseModel):
    """工具变量请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    treatment_effect: float = Field(2.0, ge=-10, le=10, description="真实处理效应")
    instrument_strength: float = Field(0.5, ge=0.1, le=1.0, description="工具变量强度")
    confounding_strength: float = Field(1.0, ge=0, le=3.0, description="混淆强度")


# ==================== API 端点 ====================


@router.post("/did-basic", response_model=ApiResponse[AnalysisResult])
async def analyze_did_basic_api(request: DIDBasicRequest):
    """基础双重差分分析"""
    from domains.part3_quasi_experiments.api import analyze_did_basic

    result = analyze_did_basic(
        n_units=request.n_units,
        n_periods=request.n_periods,
        treatment_effect=request.treatment_effect,
        parallel_trends=request.parallel_trends,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/did-event-study", response_model=ApiResponse[AnalysisResult])
async def analyze_did_event_study_api(request: DIDEventStudyRequest):
    """事件研究法分析"""
    from domains.part3_quasi_experiments.api import analyze_did_event_study

    result = analyze_did_event_study(
        n_units=request.n_units,
        n_periods=request.n_periods,
        treatment_period=request.treatment_period,
        treatment_effect=request.treatment_effect,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/did-staggered", response_model=ApiResponse[AnalysisResult])
async def analyze_did_staggered_api(request: DIDStaggeredRequest):
    """交错双重差分分析"""
    from domains.part3_quasi_experiments.api import analyze_did_staggered

    result = analyze_did_staggered(
        n_units=request.n_units,
        n_periods=request.n_periods,
        treatment_effect=request.treatment_effect,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/synthetic-control", response_model=ApiResponse[AnalysisResult])
async def analyze_synthetic_control_api(request: SyntheticControlRequest):
    """合成控制法分析"""
    from domains.part3_quasi_experiments.api import analyze_synthetic_control

    result = analyze_synthetic_control(
        n_treated_periods=request.n_treated_periods,
        n_control_units=request.n_control_units,
        treatment_effect=request.treatment_effect,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/rdd-sharp", response_model=ApiResponse[AnalysisResult])
async def analyze_rdd_sharp_api(request: RDDSharpRequest):
    """断点回归 (Sharp RDD) 分析"""
    from domains.part3_quasi_experiments.api import analyze_rdd_sharp

    result = analyze_rdd_sharp(
        n_samples=request.n_samples,
        treatment_effect=request.treatment_effect,
        bandwidth=request.bandwidth,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/rdd-fuzzy", response_model=ApiResponse[AnalysisResult])
async def analyze_rdd_fuzzy_api(request: RDDFuzzyRequest):
    """模糊断点回归 (Fuzzy RDD) 分析"""
    from domains.part3_quasi_experiments.api import analyze_rdd_fuzzy

    result = analyze_rdd_fuzzy(
        n_samples=request.n_samples,
        treatment_effect=request.treatment_effect,
        compliance_rate=request.compliance_rate,
        bandwidth=request.bandwidth,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/iv", response_model=ApiResponse[AnalysisResult])
async def analyze_iv_api(request: IVRequest):
    """工具变量分析"""
    from domains.part3_quasi_experiments.api import analyze_iv

    result = analyze_iv(
        n_samples=request.n_samples,
        treatment_effect=request.treatment_effect,
        instrument_strength=request.instrument_strength,
        confounding_strength=request.confounding_strength,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))
