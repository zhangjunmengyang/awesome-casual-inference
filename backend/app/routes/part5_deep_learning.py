"""Part 5: Deep Learning Methods API - 深度学习因果推断方法"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from app.schemas.common import ApiResponse, AnalysisResult, convert_numpy_types

router = APIRouter()


# ==================== 请求/响应模型 ====================


class RepresentationLearningRequest(BaseModel):
    """表示学习请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(10, ge=5, le=50, description="特征数量")
    latent_dim: int = Field(5, ge=2, le=20, description="潜在表示维度")
    epochs: int = Field(50, ge=10, le=200, description="训练轮数")
    batch_size: int = Field(64, ge=16, le=256, description="批次大小")


class TARNetRequest(BaseModel):
    """TARNet 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(10, ge=5, le=50, description="特征数量")
    hidden_dim: int = Field(64, ge=16, le=256, description="隐藏层维度")
    n_shared_layers: int = Field(2, ge=1, le=5, description="共享层数量")
    epochs: int = Field(50, ge=10, le=200, description="训练轮数")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="学习率")


class DragonNetRequest(BaseModel):
    """DragonNet 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(10, ge=5, le=50, description="特征数量")
    hidden_dim: int = Field(64, ge=16, le=256, description="隐藏层维度")
    n_layers: int = Field(3, ge=1, le=5, description="网络层数")
    epochs: int = Field(50, ge=10, le=200, description="训练轮数")
    targeted_regularization: bool = Field(True, description="是否使用 Targeted Regularization")


class CEVAERequest(BaseModel):
    """CEVAE (Causal Effect VAE) 请求"""

    n_samples: int = Field(1000, ge=100, le=10000, description="样本数量")
    n_features: int = Field(10, ge=5, le=50, description="特征数量")
    latent_dim: int = Field(10, ge=5, le=30, description="潜在变量维度")
    epochs: int = Field(100, ge=20, le=300, description="训练轮数")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="学习率")


# ==================== API 端点 ====================


@router.post("/representation-learning", response_model=ApiResponse[AnalysisResult])
async def analyze_representation_learning_api(request: RepresentationLearningRequest):
    """表示学习分析"""
    from domains.part5_deep_learning.api import analyze_representation_learning

    result = analyze_representation_learning(
        n_samples=request.n_samples,
        n_features=request.n_features,
        latent_dim=request.latent_dim,
        epochs=request.epochs,
        batch_size=request.batch_size,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/tarnet", response_model=ApiResponse[AnalysisResult])
async def analyze_tarnet_api(request: TARNetRequest):
    """TARNet 分析"""
    from domains.part5_deep_learning.api import analyze_tarnet

    result = analyze_tarnet(
        n_samples=request.n_samples,
        n_features=request.n_features,
        hidden_dim=request.hidden_dim,
        n_shared_layers=request.n_shared_layers,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/dragonnet", response_model=ApiResponse[AnalysisResult])
async def analyze_dragonnet_api(request: DragonNetRequest):
    """DragonNet 分析"""
    from domains.part5_deep_learning.api import analyze_dragonnet

    result = analyze_dragonnet(
        n_samples=request.n_samples,
        n_features=request.n_features,
        hidden_dim=request.hidden_dim,
        n_layers=request.n_layers,
        epochs=request.epochs,
        targeted_regularization=request.targeted_regularization,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))


@router.post("/cevae", response_model=ApiResponse[AnalysisResult])
async def analyze_cevae_api(request: CEVAERequest):
    """CEVAE 分析"""
    from domains.part5_deep_learning.api import analyze_cevae

    result = analyze_cevae(
        n_samples=request.n_samples,
        n_features=request.n_features,
        latent_dim=request.latent_dim,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
    )
    return ApiResponse(success=True, data=convert_numpy_types(result))
