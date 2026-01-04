# Causal Inference Workbench - 项目指南

## 项目概述

这是一个因果推断学习与可视化平台，专注于 Model-Based 方法，涵盖深度学习和机器学习因果模型。
项目除了展示核心原理，另外的目的是作为教程项目，教我相关知识，循序渐进，体系化学习，在真实项目实践中学习，避免只懂公式的纸上谈兵。

## 工作模式 (最重要)

**核心原则: 你是调度者，不是执行者。**

不要事必躬亲。你的角色是:
- 理解用户意图，拆解任务
- 分发任务给 SubAgent
- 审核结果，做出判断
- 统筹全局，确保一致性

```
用户需求 -> 拆解任务 -> 分发给 SubAgent -> 审核结果 -> 交付
```

## 快速启动

### Docker 模式 (推荐)

```bash
make start
```

### 本地开发模式

```bash
make install      # 安装依赖
make start-local  # 启动前后端
```

### 访问地址

- 前端: http://localhost:5173
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/docs

## 项目结构 (Monorepo)

```
.
├── package.json              # pnpm workspace 配置
├── pnpm-workspace.yaml       # workspace 定义
├── Makefile                  # 开发命令
│
├── backend/                  # FastAPI 后端
│   ├── app/
│   │   ├── main.py          # FastAPI 入口
│   │   ├── routes/          # API 路由
│   │   │   ├── foundation.py
│   │   │   ├── treatment.py
│   │   │   ├── uplift.py
│   │   │   ├── hetero.py
│   │   │   ├── evaluation.py
│   │   │   ├── case_studies.py
│   │   │   └── ab_testing.py
│   │   └── schemas/         # Pydantic 数据模型
│   ├── domains/             # 业务逻辑 (原 Lab 代码)
│   │   ├── foundation_lab/
│   │   ├── treatment_effect_lab/
│   │   ├── uplift_lab/
│   │   ├── hetero_effect_lab/
│   │   ├── deep_causal_lab/
│   │   ├── evaluation_lab/
│   │   ├── case_studies/
│   │   └── ab_testing_toolkit/
│   └── requirements.txt
│
├── frontend/                 # React + TypeScript 前端
│   ├── src/
│   │   ├── App.tsx
│   │   ├── routes.tsx       # 路由定义
│   │   ├── features/        # 按 Lab 划分的功能模块
│   │   │   ├── foundation/
│   │   │   │   ├── types.ts
│   │   │   │   ├── api.ts
│   │   │   │   ├── hooks.ts
│   │   │   │   └── pages/
│   │   │   ├── treatment/
│   │   │   ├── uplift/
│   │   │   ├── hetero/
│   │   │   ├── evaluation/
│   │   │   ├── case-studies/
│   │   │   └── ab-testing/
│   │   ├── components/      # 通用组件
│   │   │   ├── charts/      # 图表组件 (Plotly)
│   │   │   ├── layout/      # 布局组件
│   │   │   └── ui/          # 基础 UI (shadcn)
│   │   └── lib/
│   │       ├── api/         # HTTP 客户端
│   │       └── utils/
│   └── package.json
│
├── docker/                   # Docker 配置
│   ├── docker-compose.yml
│   ├── api/Dockerfile
│   └── frontend/Dockerfile
│
└── docs/                     # 文档
```

## 开发规范

### 后端 API 规范

每个路由模块遵循以下结构:

```python
# app/routes/foundation.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.schemas.common import ApiResponse, AnalysisResult

router = APIRouter()

class SomeRequest(BaseModel):
    n_samples: int = 500
    # ...

@router.post("/some-analysis", response_model=ApiResponse[AnalysisResult])
async def analyze_something(request: SomeRequest):
    from domains.foundation_lab.api import analyze_something
    result = analyze_something(**request.dict())
    return ApiResponse(success=True, data=result)
```

### 前端 Feature 规范

每个 feature 包含:
- `types.ts`: 类型定义
- `api.ts`: API 调用
- `hooks.ts`: React Query hooks
- `pages/`: 页面组件

```typescript
// features/foundation/hooks.ts
import { useMutation } from '@tanstack/react-query'
import { foundationApi } from './api'

export function usePotentialOutcomes() {
  return useMutation({
    mutationFn: (params) => foundationApi.analyzePotentialOutcomes(params),
  })
}
```

### 可视化规范

- 使用 Plotly 进行交互式可视化
- 颜色方案: #2D9CDB (主蓝), #27AE60 (绿), #EB5757 (红)
- 统一使用 `template='plotly_white'`

## 技术栈

### 前端
- React 18 + TypeScript
- Vite (构建工具)
- TailwindCSS + shadcn/ui
- React Query (数据获取)
- Plotly.js (图表)
- React Router v6

### 后端
- FastAPI
- Pydantic v2
- NumPy, Pandas, SciPy
- Scikit-learn, XGBoost
- EconML (因果推断)
- Plotly (图表数据生成)

## 模块状态

### 已完成
- [x] 项目骨架 (Monorepo + Docker)
- [x] Foundation Lab (潜在结果、因果图、混淆偏差、选择偏差)

### 开发中
- [ ] Treatment Effect Lab (PSM, IPW, DR)
- [ ] Uplift Lab (Meta-Learners, Uplift Tree)
- [ ] Heterogeneous Effect Lab (Causal Forest)
- [ ] Case Studies (DoorDash, Netflix)
- [ ] A/B Testing Toolkit

## 相关资源

- EconML: https://github.com/microsoft/EconML
- CausalML: https://github.com/uber/causalml
- DoWhy: https://github.com/py-why/dowhy
