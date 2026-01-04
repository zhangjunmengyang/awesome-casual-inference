"""API 路由聚合"""

from fastapi import APIRouter

from app.routes import (
    part0_foundation,
    part1_experimentation,
    part2_observational,
    part3_quasi_experiments,
    part4_cate_uplift,
    part5_deep_learning,
    part6_marketing,
    part7_advanced,
)

api_router = APIRouter()

# 注册新模块路由 (按知识体系重构后)
api_router.include_router(
    part0_foundation.router,
    prefix="/part0",
    tags=["Part 0: Foundation - 因果思维基础"]
)
api_router.include_router(
    part1_experimentation.router,
    prefix="/part1",
    tags=["Part 1: Experimentation - 实验方法"]
)
api_router.include_router(
    part2_observational.router,
    prefix="/part2",
    tags=["Part 2: Observational - 观测数据方法"]
)
api_router.include_router(
    part3_quasi_experiments.router,
    prefix="/part3",
    tags=["Part 3: Quasi-Experiments - 准实验方法"]
)
api_router.include_router(
    part4_cate_uplift.router,
    prefix="/part4",
    tags=["Part 4: CATE & Uplift - 异质效应与Uplift建模"]
)
api_router.include_router(
    part5_deep_learning.router,
    prefix="/part5",
    tags=["Part 5: Deep Learning - 深度学习因果推断"]
)
api_router.include_router(
    part6_marketing.router,
    prefix="/part6",
    tags=["Part 6: Marketing - 营销应用"]
)
api_router.include_router(
    part7_advanced.router,
    prefix="/part7",
    tags=["Part 7: Advanced - 高级主题"]
)
