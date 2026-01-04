"""Part 1: Experimentation Lab API - A/B 测试与实验设计"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from app.schemas.common import ApiResponse, AnalysisResult


router = APIRouter()


# ==================== 请求模型 ====================


class ABTestRequest(BaseModel):
    """A/B 测试分析请求"""

    n_control: int = Field(10000, ge=100, description="对照组样本量")
    n_treatment: int = Field(10000, ge=100, description="实验组样本量")
    baseline_rate: float = Field(0.05, ge=0.001, le=0.999, description="基线转化率")
    treatment_effect: float = Field(0.10, ge=-1, le=10, description="处理效应")
    metrics: Optional[List[str]] = Field(None, description="要分析的指标列表")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="显著性水平")
    seed: int = Field(42, description="随机种子")


class CUPEDRequest(BaseModel):
    """CUPED 方差缩减请求"""

    metric_col: str = Field("converted", description="目标指标")
    covariate_col: str = Field("historical_conversion", description="协变量")
    n_samples: int = Field(10000, ge=100, description="样本量")
    seed: int = Field(42, description="随机种子")


class StratifiedRequest(BaseModel):
    """分层分析请求"""

    metric_col: str = Field("converted", description="目标指标")
    strata_col: str = Field("user_activity", description="分层变量")
    n_quantiles: int = Field(4, ge=2, le=10, description="分位数数量")
    n_samples: int = Field(10000, ge=100, description="样本量")
    seed: int = Field(42, description="随机种子")


class NetworkEffectsRequest(BaseModel):
    """网络效应分析请求"""

    n_users: int = Field(1000, ge=100, description="用户数量")
    avg_degree: int = Field(10, ge=1, description="平均连接数")
    direct_effect: float = Field(0.15, description="直接效应")
    spillover_effect: float = Field(0.05, description="溢出效应")
    seed: int = Field(42, description="随机种子")


class SwitchbackRequest(BaseModel):
    """Switchback 实验请求"""

    n_units: int = Field(50, ge=10, description="单元数量")
    n_periods: int = Field(100, ge=10, description="时间周期数")
    treatment_effect: float = Field(0.10, description="处理效应")
    carryover: float = Field(0.03, description="残留效应")
    seed: int = Field(42, description="随机种子")


class LongTermEffectsRequest(BaseModel):
    """长期效应估计请求"""

    n_users: int = Field(10000, ge=100, description="用户数量")
    n_days: int = Field(180, ge=30, description="天数")
    short_term_effect: float = Field(0.15, description="短期效应")
    long_term_effect: float = Field(0.05, description="长期效应")
    seed: int = Field(42, description="随机种子")


class BanditRequest(BaseModel):
    """多臂老虎机模拟请求"""

    arm_means: Optional[List[float]] = Field(None, description="各臂的真实均值")
    algorithm: str = Field("thompson", description="算法 (epsilon, thompson, ucb)")
    n_rounds: int = Field(1000, ge=100, description="模拟轮数")
    seed: int = Field(42, description="随机种子")


# ==================== API 端点 ====================


@router.get("/status", response_model=ApiResponse[dict])
async def get_status():
    """获取 Part 1 模块状态"""
    return ApiResponse(
        success=True,
        data={
            "status": "active",
            "message": "Part 1 Experimentation API is ready",
            "modules": [
                "ab_testing",
                "cuped",
                "stratified_analysis",
                "network_effects",
                "switchback",
                "long_term_effects",
                "multi_armed_bandits",
            ],
        },
    )


@router.post("/ab-test", response_model=ApiResponse[AnalysisResult])
async def analyze_ab_test_api(request: ABTestRequest):
    """
    A/B 测试完整分析

    包含:
    - 指标统计检验
    - SRM 检查
    - 平衡性检验
    - 样本量计算
    """
    from domains.part1_experimentation.api import analyze_ab_test

    result = analyze_ab_test(
        n_control=request.n_control,
        n_treatment=request.n_treatment,
        baseline_rate=request.baseline_rate,
        treatment_effect=request.treatment_effect,
        metrics=request.metrics,
        alpha=request.alpha,
        seed=request.seed,
    )
    return ApiResponse(success=True, data=AnalysisResult(**result))


@router.post("/cuped", response_model=ApiResponse[AnalysisResult])
async def apply_cuped_api(request: CUPEDRequest):
    """
    CUPED 方差缩减分析

    包含:
    - CUPED 前后对比
    - 协变量选择
    - 方差缩减估计
    """
    from domains.part1_experimentation.api import apply_cuped

    result = apply_cuped(
        metric_col=request.metric_col,
        covariate_col=request.covariate_col,
        n_samples=request.n_samples,
        seed=request.seed,
    )
    return ApiResponse(success=True, data=AnalysisResult(**result))


@router.post("/stratified", response_model=ApiResponse[AnalysisResult])
async def stratified_analysis_api(request: StratifiedRequest):
    """
    分层分析

    包含:
    - 分层效应估计
    - 异质性检验
    - 汇总估计
    """
    from domains.part1_experimentation.api import stratified_analysis

    result = stratified_analysis(
        metric_col=request.metric_col,
        strata_col=request.strata_col,
        n_quantiles=request.n_quantiles,
        n_samples=request.n_samples,
        seed=request.seed,
    )
    return ApiResponse(success=True, data=AnalysisResult(**result))


@router.post("/network-effects", response_model=ApiResponse[AnalysisResult])
async def analyze_network_effects_api(request: NetworkEffectsRequest):
    """
    网络效应分析

    包含:
    - 直接效应估计
    - 溢出效应估计
    - 效应分解
    """
    from domains.part1_experimentation.api import analyze_network_effects

    result = analyze_network_effects(
        n_users=request.n_users,
        avg_degree=request.avg_degree,
        direct_effect=request.direct_effect,
        spillover_effect=request.spillover_effect,
        seed=request.seed,
    )
    return ApiResponse(success=True, data=AnalysisResult(**result))


@router.post("/switchback", response_model=ApiResponse[AnalysisResult])
async def analyze_switchback_api(request: SwitchbackRequest):
    """
    Switchback 实验分析

    包含:
    - 朴素估计
    - 固定效应估计
    - 残留效应检测
    """
    from domains.part1_experimentation.api import analyze_switchback

    result = analyze_switchback(
        n_units=request.n_units,
        n_periods=request.n_periods,
        treatment_effect=request.treatment_effect,
        carryover=request.carryover,
        seed=request.seed,
    )
    return ApiResponse(success=True, data=AnalysisResult(**result))


@router.post("/long-term-effects", response_model=ApiResponse[AnalysisResult])
async def estimate_long_term_effects_api(request: LongTermEffectsRequest):
    """
    长期效应估计

    包含:
    - 短期效应分析
    - 长期效应分析
    - 时变效应追踪
    """
    from domains.part1_experimentation.api import estimate_long_term_effects

    result = estimate_long_term_effects(
        n_users=request.n_users,
        n_days=request.n_days,
        short_term_effect=request.short_term_effect,
        long_term_effect=request.long_term_effect,
        seed=request.seed,
    )
    return ApiResponse(success=True, data=AnalysisResult(**result))


@router.post("/bandit", response_model=ApiResponse[AnalysisResult])
async def run_bandit_simulation_api(request: BanditRequest):
    """
    多臂老虎机模拟

    包含:
    - 算法模拟 (Epsilon-Greedy, Thompson Sampling, UCB)
    - 算法对比
    - Regret 分析
    """
    from domains.part1_experimentation.api import run_bandit_simulation

    result = run_bandit_simulation(
        arm_means=request.arm_means,
        algorithm=request.algorithm,
        n_rounds=request.n_rounds,
        seed=request.seed,
    )
    return ApiResponse(success=True, data=AnalysisResult(**result))
