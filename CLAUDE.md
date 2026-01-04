# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

因果推断学习与可视化平台，专注于 Model-Based 方法，涵盖深度学习和机器学习因果模型。项目目的是作为教程项目，体系化学习，在真实项目实践中学习。

## 工作模式

**核心原则: 你是调度者，不是执行者。**

- 理解用户意图，拆解任务
- 分发任务给 SubAgent
- 审核结果，统筹全局

## 常用命令

```bash
# 启动服务
make start              # Docker 模式 (推荐)
make start local        # 本地开发模式

# 停止服务
make stop               # Docker 模式
make stop local         # 本地模式

# 安装依赖
make install            # 后端 pip + 前端 pnpm

# 代码质量
make format             # 格式化 (black/isort + eslint)
make typecheck          # 类型检查 (mypy + tsc)

# Docker 相关
make logs               # 查看日志
make rebuild            # 重新构建
```

**访问地址:**
- 前端: http://localhost:5173
- 后端: http://localhost:8000
- API 文档: http://localhost:8000/docs

## 架构概览

Monorepo 全栈项目，FastAPI 后端 + React 前端。

```
backend/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── routes/              # API 路由 (part0-part7)
│   └── schemas/             # Pydantic 数据模型
└── domains/                 # 业务逻辑
    ├── part0_foundation/    # 基础概念
    ├── part1_experimentation/
    ├── part2_observational/
    ├── part3_quasi_experiments/
    ├── part4_cate_uplift/
    ├── part5_deep_learning/
    ├── part6_marketing/
    ├── part7_advanced/
    └── datasets/

frontend/
└── src/
    ├── features/            # 按模块划分的功能
    │   └── [module]/
    │       ├── types.ts     # 类型定义
    │       ├── api.ts       # API 调用
    │       ├── hooks.ts     # React Query hooks
    │       └── pages/       # 页面组件
    ├── components/          # 通用组件 (ui, layout, charts)
    └── lib/                 # API 客户端、工具函数
```

## 开发规范

### 后端 API 模式

```python
# routes/part0_foundation.py
class SomeRequest(BaseModel):
    n_samples: int = Field(500, ge=100, le=5000)

@router.post("/some-analysis", response_model=ApiResponse[AnalysisResult])
async def analyze(request: SomeRequest):
    from domains.part0_foundation.api import analyze_something
    result = analyze_something(**request.dict())
    return ApiResponse(success=True, data=result)
```

### 前端 Hook 模式

```typescript
// features/foundation/hooks.ts
export function usePotentialOutcomes() {
  return useMutation({
    mutationFn: (params) => foundationApi.analyzePotentialOutcomes(params),
  })
}
```

### 可视化规范

- Plotly 交互式可视化
- 颜色: #2D9CDB (蓝), #27AE60 (绿), #EB5757 (红)
- 模板: `template='plotly_white'`

## 技术栈

**后端:** FastAPI, Pydantic v2, Pandas, NumPy, Scikit-learn, EconML, Plotly

**前端:** React 18, TypeScript, Vite, TailwindCSS, shadcn/ui, React Query, Plotly.js

## 相关资源

- EconML: https://github.com/microsoft/EconML
- CausalML: https://github.com/uber/causalml
- DoWhy: https://github.com/py-why/dowhy
