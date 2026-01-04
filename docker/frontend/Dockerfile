FROM node:20-alpine

WORKDIR /app

# 安装 pnpm
RUN corepack enable && corepack prepare pnpm@9.15.0 --activate

# 复制依赖文件
COPY frontend/package.json ./

# 安装依赖
RUN pnpm install

# 复制应用代码
COPY frontend/ .

# 暴露端口
EXPOSE 5173

# 默认启动命令
CMD ["pnpm", "dev", "--host", "0.0.0.0"]
