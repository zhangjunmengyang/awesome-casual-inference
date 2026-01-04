"""Part 4: CATE & Uplift Modeling API - 异质性处理效应与提升建模"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

from app.schemas.common import ApiResponse, AnalysisResult, convert_numpy_types

router = APIRouter()


# ==================== 请求/响应模型 ====================


class MetaLearnersRequest(BaseModel):
    """Meta-Learners 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(5, ge=2, le=20, description="特征数量")
    heterogeneity_strength: float = Field(1.0, ge=0, le=3.0, description="异质性强度")
    methods: List[Literal["s_learner", "t_learner", "x_learner", "r_learner"]] = Field(
        ["s_learner", "t_learner", "x_learner"],
        description="要比较的方法列表"
    )


class CausalForestRequest(BaseModel):
    """Causal Forest 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(5, ge=2, le=20, description="特征数量")
    heterogeneity_strength: float = Field(1.0, ge=0, le=3.0, description="异质性强度")
    n_trees: int = Field(100, ge=10, le=500, description="树的数量")
    min_samples_leaf: int = Field(5, ge=1, le=50, description="叶子节点最小样本数")


class UpliftTreeRequest(BaseModel):
    """Uplift Tree 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(5, ge=2, le=20, description="特征数量")
    heterogeneity_strength: float = Field(1.0, ge=0, le=3.0, description="异质性强度")
    max_depth: int = Field(5, ge=2, le=10, description="树的最大深度")
    min_samples_leaf: int = Field(20, ge=5, le=100, description="叶子节点最小样本数")


class UpliftEvaluationRequest(BaseModel):
    """Uplift 评估请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(5, ge=2, le=20, description="特征数量")
    heterogeneity_strength: float = Field(1.0, ge=0, le=3.0, description="异质性强度")


class CATEVisualizationRequest(BaseModel):
    """CATE 可视化请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(2, ge=2, le=10, description="特征数量")
    heterogeneity_type: Literal["linear", "nonlinear", "interaction"] = Field(
        "linear", description="异质性类型"
    )


# ==================== API 端点 ====================


@router.post("/meta-learners", response_model=ApiResponse[AnalysisResult])
async def analyze_meta_learners_api(request: MetaLearnersRequest):
    """Meta-Learners 分析"""
    from domains.part4_cate_uplift.api import analyze_meta_learners

    result = analyze_meta_learners(
        n_samples=request.n_samples,
        n_features=request.n_features,
        heterogeneity_strength=request.heterogeneity_strength,
        methods=request.methods,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/causal-forest", response_model=ApiResponse[AnalysisResult])
async def analyze_causal_forest_api(request: CausalForestRequest):
    """Causal Forest 分析"""
    from domains.part4_cate_uplift.api import analyze_causal_forest

    result = analyze_causal_forest(
        n_samples=request.n_samples,
        n_features=request.n_features,
        heterogeneity_strength=request.heterogeneity_strength,
        n_trees=request.n_trees,
        min_samples_leaf=request.min_samples_leaf,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/uplift-tree", response_model=ApiResponse[AnalysisResult])
async def analyze_uplift_tree_api(request: UpliftTreeRequest):
    """Uplift Tree 分析"""
    from domains.part4_cate_uplift.api import analyze_uplift_tree

    result = analyze_uplift_tree(
        n_samples=request.n_samples,
        n_features=request.n_features,
        heterogeneity_strength=request.heterogeneity_strength,
        max_depth=request.max_depth,
        min_samples_leaf=request.min_samples_leaf,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/uplift-evaluation", response_model=ApiResponse[AnalysisResult])
async def analyze_uplift_evaluation_api(request: UpliftEvaluationRequest):
    """Uplift 模型评估"""
    from domains.part4_cate_uplift.api import analyze_uplift_evaluation

    result = analyze_uplift_evaluation(
        n_samples=request.n_samples,
        n_features=request.n_features,
        heterogeneity_strength=request.heterogeneity_strength,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/cate-visualization", response_model=ApiResponse[AnalysisResult])
async def visualize_cate_api(request: CATEVisualizationRequest):
    """CATE 可视化"""
    from domains.part4_cate_uplift.api import visualize_cate

    result = visualize_cate(
        n_samples=request.n_samples,
        n_features=request.n_features,
        heterogeneity_type=request.heterogeneity_type,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))
