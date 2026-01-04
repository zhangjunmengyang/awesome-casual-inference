"""通用 Schema 定义"""

import json
from typing import Any, Generic, TypeVar, Optional, Union
import numpy as np
from pydantic import BaseModel, Field

T = TypeVar("T")


def convert_numpy_types(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应格式"""

    success: bool = True
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None


class Point2D(BaseModel):
    """2D 坐标点"""

    x: float
    y: float


class ChartData(BaseModel):
    """图表数据 - 支持 Plotly 原生格式"""

    model_config = {"extra": "allow"}

    # Plotly 格式的必需字段
    data: list[dict[str, Any]] = Field(default_factory=list, description="Plotly data traces")
    layout: dict[str, Any] = Field(default_factory=dict, description="Plotly layout")


class TableData(BaseModel):
    """表格数据"""

    model_config = {"extra": "allow"}

    # 支持多种表格格式
    title: Optional[str] = Field(None, description="表格标题")
    columns: Optional[list[str]] = Field(None, description="列名")
    headers: Optional[list[str]] = Field(None, description="表头 (别名)")
    rows: Optional[list[Any]] = Field(None, description="行数据")
    data: Optional[list[dict[str, Any]]] = Field(None, description="数据 (字典列表)")


class AnalysisResult(BaseModel):
    """分析结果"""

    model_config = {"extra": "allow"}

    charts: list[ChartData] = Field(default_factory=list, description="图表列表")
    tables: list[Any] = Field(default_factory=list, description="表格列表 (灵活格式)")
    summary: str = Field("", description="分析摘要 (Markdown)")
    metrics: dict[str, Any] = Field(default_factory=dict, description="关键指标 (任意类型)")
