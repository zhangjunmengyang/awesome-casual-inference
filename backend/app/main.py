"""FastAPI 主应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import api_router

app = FastAPI(
    title="Causal Inference Workbench",
    description="因果推断学习与可视化平台 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 开发服务器
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Causal Inference Workbench API",
        "docs": "/docs",
        "health": "/health",
    }
