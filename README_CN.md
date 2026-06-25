# CF AI Proxy

一个基于 Cloudflare Worker 的 AI API 代理，使用 D1 数据库存储路由配置。它同时支持 [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 和 [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) 格式，并能为只支持一种格式的 API 进行自动格式转换。

[English](./README.md)

## 功能特性

- 🔄 **双格式支持**: 同时支持 OpenAI 和 Anthropic API 格式
- 🔀 **智能路由**: 自动检测 API 格式支持，尽可能直接代理
- 💾 **D1 数据库存储**: 路由和设置存储在 Cloudflare D1 (SQLite) 中
- 🔐 **密码保护**: 管理页面通过密码保护
- 🎛️ **Web 管理界面**: 通过网页添加、编辑和删除 API 路由
- 🔍 **API 测试**: 保存前测试 API 格式支持
- 🔑 **全局 API 令牌**: 为所有路由设置共享令牌或为每个路由配置独立令牌
- 📡 **流式支持**: 完整支持流式响应 (SSE)
- 🛠️ **工具调用**: 双向转换工具/函数调用
- 🚀 **边缘部署**: 运行在 Cloudflare 全球边缘网络上

## 架构

```
客户端 (OpenAI 或 Anthropic 格式)
    │
    ▼
┌─────────────────────────────────────┐
│      Cloudflare Worker              │
│  ┌─────────────────────────────┐   │
│  │  /api/{name}/v1/*           │   │
│  │                             │   │
│  │  如果 API 支持两种格式:      │   │
│  │    → 直接代理               │   │
│  │                             │   │
│  │  如果 API 只支持一种格式:    │   │
│  │    → 自动转换格式           │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  D1 数据库                  │   │
│  │  - 路由 (名称, URL, 密钥)   │   │
│  │  - 格式支持标志             │   │
│  │  - 设置 (全局令牌)          │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    │
    ▼
远程 AI API
(OpenAI, Anthropic, DeepSeek 等)
```

## 工作原理

1. **同时支持两种格式的 API** (如 DeepSeek): 请求直接代理，不进行转换
2. **只支持 OpenAI 格式的 API**: Anthropic 请求自动转换为 OpenAI 格式
3. **只支持 Anthropic 格式的 API**: OpenAI 请求自动转换为 Anthropic 格式

添加新路由时，可以点击"测试 API"自动检测 API 支持的格式。

## 快速开始

### 1. 克隆并安装

```bash
git clone https://github.com/zxdong262-apps/cf-ai-proxy.git
cd cf-ai-proxy
npm install
```

### 2. 创建 D1 数据库

```bash
# 创建数据库（记下输出的 ID）
wrangler d1 create ai-proxy-db

# 更新 wrangler.toml 中的数据库 ID
# 将 "placeholder-replace-with-actual-id" 替换为实际的 ID

# 初始化数据库结构
npm run db:init
```

### 3. 设置密码

编辑 `wrangler.toml` 设置你的 `MAIN_PASSWORD`：

```toml
[vars]
MAIN_PASSWORD = "your-secure-password"
```

或者在生产环境中设置为密钥：

```bash
wrangler secret put MAIN_PASSWORD
```

### 4. 本地开发

```bash
npm start
# 或
npm run dev
```

访问 `http://localhost:8787` 进入管理页面。

### 5. 部署

```bash
npm run deploy
```

## 使用方法

### 管理页面

访问你的 Worker URL (`https://your-worker.workers.dev`) 进入管理页面：

1. 使用 `MAIN_PASSWORD` 登录
2. 生成或设置全局 API 令牌
3. 添加路由并配置远程 API URL
4. 点击"测试 API"自动检测格式支持
5. 配置支持的格式 (OpenAI/Anthropic/两者)

### API 端点

#### OpenAI 兼容端点

```
POST /api/{name}/v1/chat/completions
```

#### Anthropic 兼容端点

```
POST /api/{name}/v1/messages
```

#### 模型列表

```
GET /api/{name}/v1/models
```

### 客户端配置

#### OpenAI SDK

```javascript
import OpenAI from 'openai'

const client = new OpenAI({
  apiKey: 'your-global-api-token',
  baseURL: 'https://your-worker.workers.dev/api/openai/v1'
})

const response = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }]
})
```

#### Anthropic SDK

```javascript
import Anthropic from '@anthropic-ai/sdk'

const client = new Anthropic({
  apiKey: 'your-global-api-token',
  baseURL: 'https://your-worker.workers.dev/api/deepseek'
})

const response = await client.messages.create({
  model: 'claude-3-opus-20240229',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Hello!' }]
})
```

#### Claude Code

```bash
# 对于原生支持 Anthropic 格式的 API
export ANTHROPIC_BASE_URL=https://your-worker.workers.dev/api/deepseek
export ANTHROPIC_AUTH_TOKEN=your-global-api-token

# 对于只支持 OpenAI 格式的 API
export ANTHROPIC_BASE_URL=https://your-worker.workers.dev/api/openai
export ANTHROPIC_AUTH_TOKEN=your-global-api-token

claude
```

#### cURL

```bash
# OpenAI 格式
curl https://your-worker.workers.dev/api/openai/v1/chat/completions \
  -H "Authorization: Bearer your-global-api-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Anthropic 格式
curl https://your-worker.workers.dev/api/deepseek/v1/messages \
  -H "Authorization: Bearer your-global-api-token" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 配置

### 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `MAIN_PASSWORD` | 管理页面的登录密码 | 是 |

### D1 数据库

路由存储在 D1 中，结构如下：

```sql
CREATE TABLE routes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  remote_api_url TEXT NOT NULL,
  api_key TEXT NOT NULL DEFAULT '',
  messages_endpoint TEXT NOT NULL DEFAULT '/chat/completions',
  anthropic_endpoint TEXT NOT NULL DEFAULT '/messages',
  supports_openai INTEGER NOT NULL DEFAULT 1,
  supports_anthropic INTEGER NOT NULL DEFAULT 0,
  auth_type TEXT NOT NULL DEFAULT 'bearer',
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

### 路由字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 路由名称，用作 URL 前缀（如 `openai` → `/api/openai/v1/*`） |
| `remote_api_url` | 是 | API 的基础 URL |
| `api_key` | 否 | 路由专用 API 密钥。为空时使用全局令牌 |
| `messages_endpoint` | 否 | OpenAI 聊天补全端点（默认：`/chat/completions`） |
| `anthropic_endpoint` | 否 | Anthropic 消息端点（默认：`/messages`） |
| `supports_openai` | 否 | API 是否支持 OpenAI 格式（默认：`true`） |
| `supports_anthropic` | 否 | API 是否支持 Anthropic 格式（默认：`false`） |
| `auth_type` | 否 | 认证类型：`bearer` 或 `api-key`（默认：`bearer`） |
| `enabled` | 否 | 路由是否启用（默认：`true`） |

### 支持的服务商

支持任何 AI API 服务商：

**同时支持两种格式（直接代理）：**
- **DeepSeek**: `https://api.deepseek.com/v1`
- **Anthropic**: `https://api.anthropic.com/v1`

**仅支持 OpenAI 格式（Anthropic 请求会被转换）：**
- **OpenAI**: `https://api.openai.com/v1`
- **Ollama**（本地）: `http://localhost:11434/v1`
- **vLLM**（本地）: `http://localhost:8000/v1`
- **LiteLLM**: `http://localhost:4000/v1`
- **Together AI**: `https://api.together.xyz/v1`
- **Groq**: `https://api.groq.com/openai/v1`

## API 参考

### 管理 API

所有管理端点需要通过会话 Cookie 或 `Authorization: Bearer <session-token>` 进行认证。

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/auth/login` | 使用密码登录 |
| `POST` | `/api/auth/logout` | 登出 |
| `GET` | `/api/auth/check` | 检查认证状态 |
| `GET` | `/api/routes` | 列出所有路由 |
| `POST` | `/api/routes` | 创建新路由 |
| `PUT` | `/api/routes/:id` | 更新路由 |
| `DELETE` | `/api/routes/:id` | 删除路由 |
| `POST` | `/api/routes/test` | 测试 API 格式支持 |
| `GET` | `/api/settings` | 获取设置（全局令牌） |
| `PUT` | `/api/settings` | 更新设置 |
| `POST` | `/api/settings/generate-token` | 生成新全局令牌 |

### 代理 API

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/{name}/v1/chat/completions` | OpenAI 兼容的聊天补全 |
| `POST` | `/api/{name}/v1/messages` | Anthropic 兼容的消息 |
| `GET` | `/api/{name}/v1/models` | 列出可用模型 |
| `ANY` | `/api/{name}/v1/*` | 通用代理到远程 API |

## 测试

```bash
# 运行所有测试
npm test

# 仅运行单元测试
npm run test:unit

# 仅运行集成测试
npm run test:integration

# 监视模式运行测试
npm run test:watch
```

## 部署

### GitHub Actions

项目包含一个 GitHub Actions 工作流，在推送到 `main` 分支时自动部署到 Cloudflare Workers。

需要的密钥：
- `CLOUDFLARE_API_TOKEN`: 你的 Cloudflare API 令牌
- `CLOUDFLARE_ACCOUNT_ID`: 你的 Cloudflare 账户 ID

### 手动部署

```bash
# 部署到 Cloudflare Workers
npm run deploy
```

## 开发

### 本地开发

```bash
# 启动本地开发服务器
npm start

# 初始化本地 D1 数据库
npm run db:init

# 重置本地数据库
npm run db:reset
```

### 项目结构

```
cf-ai-proxy/
├── src/
│   ├── worker.js      # 主入口和路由
│   ├── auth.js        # 认证辅助函数
│   ├── db.js          # D1 数据库操作
│   ├── proxy.js       # 代理转发逻辑
│   ├── converter.js   # Anthropic ↔ OpenAI 格式转换
│   └── html.js        # 管理页面 HTML 模板
├── tests/
│   ├── unit/          # 单元测试
│   └── integration/   # 集成测试
├── schema.sql         # D1 数据库结构
├── wrangler.toml      # Cloudflare Workers 配置
├── vitest.config.js   # 测试配置
└── package.json
```

## 许可证

MIT
