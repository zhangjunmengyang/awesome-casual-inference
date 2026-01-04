.PHONY: start stop restart logs help install local format typecheck rebuild

# 检测模式: local（默认为空，即 Docker 模式）
MODE := $(word 2,$(MAKECMDGOALS))

# 默认目标
help:
	@echo "Causal Inference Workbench - 开发命令"
	@echo ""
	@echo "Docker 模式:"
	@echo "  make start       - 启动 Docker 容器"
	@echo "  make stop        - 停止 Docker 容器"
	@echo "  make logs        - 查看 Docker 日志"
	@echo "  make rebuild     - 重新构建并启动容器"
	@echo ""
	@echo "本地开发模式:"
	@echo "  make start local - 启动本地开发服务器"
	@echo "  make stop local  - 停止本地开发服务器"
	@echo "  make install     - 安装所有依赖"
	@echo ""
	@echo "访问地址:"
	@echo "  前端: http://localhost:5173"
	@echo "  后端: http://localhost:8000"
	@echo "  API文档: http://localhost:8000/docs"

# 伪目标（模式标记）
local:
	@true

# ============ 启动服务 ============

start:
ifeq ($(MODE),local)
	@$(MAKE) _start_local
else
	@$(MAKE) _start_docker
endif

_start_docker:
	docker compose -f docker/docker-compose.yml up -d

_start_local:
	@echo "启动本地开发服务器..."
	@trap 'kill 0' EXIT; \
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 & \
	cd frontend && pnpm dev & \
	wait

# ============ 停止服务 ============

stop:
ifeq ($(MODE),local)
	@$(MAKE) _stop_local
else
	@$(MAKE) _stop_docker
endif

_stop_docker:
	docker compose -f docker/docker-compose.yml down

_stop_local:
	@echo "停止本地开发服务器..."
	@pkill -f "uvicorn app.main:app" || true
	@pkill -f "vite" || true

# ============ 其他命令 ============

logs:
	docker compose -f docker/docker-compose.yml logs -f

rebuild:
	docker compose -f docker/docker-compose.yml up -d --build

install:
	@echo "安装后端依赖..."
	cd backend && pip install -r requirements.txt
	@echo "安装前端依赖..."
	pnpm install

# ============ 工具命令 ============

# 格式化代码
format:
	cd backend && black . && isort .
	cd frontend && pnpm lint --fix

# 类型检查
typecheck:
	cd backend && mypy app
	cd frontend && pnpm tsc --noEmit
