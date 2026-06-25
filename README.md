# CF AI Proxy

A Cloudflare Worker-based AI API proxy that stores route configurations in D1 database. It supports both [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) and [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) formats, with automatic format conversion for APIs that only support one format.

[中文文档](./README_CN.md)

## Features

- 🔄 **Dual Format Support**: Works with both OpenAI and Anthropic API formats
- 🔀 **Smart Routing**: Auto-detects API format support, proxies directly when possible
- 💾 **D1 Database Storage**: Routes and settings stored in Cloudflare D1 (SQLite)
- 🔐 **Password Protection**: Admin management page protected by password
- 🎛️ **Web Management UI**: Add, edit, and delete API routes via a web interface
- 🔍 **API Testing**: Test API format support before saving
- 🔑 **Global API Token**: Set a shared token for all routes or configure per-route tokens
- 📡 **Streaming Support**: Full support for streaming responses (SSE)
- 🛠️ **Tool Use**: Bidirectional conversion of tool/function calls
- 🚀 **Edge Deployment**: Runs on Cloudflare's global edge network

## Architecture

```
Client (OpenAI or Anthropic format)
    │
    ▼
┌─────────────────────────────────────┐
│      Cloudflare Worker              │
│  ┌─────────────────────────────┐   │
│  │  /api/{name}/v1/*           │   │
│  │                             │   │
│  │  If API supports both:      │   │
│  │    → Direct proxy           │   │
│  │                             │   │
│  │  If API only supports one:  │   │
│  │    → Auto-convert format    │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  D1 Database                │   │
│  │  - Routes (name, url, key)  │   │
│  │  - Format support flags     │   │
│  │  - Settings (global token)  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    │
    ▼
Remote AI API
(OpenAI, Anthropic, DeepSeek, etc.)
```

## How It Works

1. **APIs supporting both formats** (like DeepSeek): Requests are proxied directly without conversion
2. **APIs supporting only OpenAI**: Anthropic requests are automatically converted to OpenAI format
3. **APIs supporting only Anthropic**: OpenAI requests are automatically converted to Anthropic format

When adding a new route, you can click "Test API" to auto-detect which formats the API supports.

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/zxdong262-apps/cf-ai-proxy.git
cd cf-ai-proxy
npm install
```

### 2. Create D1 Database

```bash
# Create the database (note the ID from output)
wrangler d1 create ai-proxy-db

# Update wrangler.toml with the database ID
# Replace "placeholder-replace-with-actual-id" with the actual ID

# Initialize the schema
npm run db:init
```

### 3. Set Password

Edit `wrangler.toml` and set your `MAIN_PASSWORD`:

```toml
[vars]
MAIN_PASSWORD = "your-secure-password"
```

Or set it as a secret for production:

```bash
wrangler secret put MAIN_PASSWORD
```

### 4. Local Development

```bash
npm start
# or
npm run dev
```

Visit `http://localhost:8787` to access the management page.

### 5. Deploy

```bash
npm run deploy
```

## Usage

### Management Page

Visit your worker URL (`https://your-worker.workers.dev`) to access the management page:

1. Login with your `MAIN_PASSWORD`
2. Generate or set a global API token
3. Add routes with remote API URLs
4. Click "Test API" to auto-detect format support
5. Configure supported formats (OpenAI/Anthropic/Both)

### API Endpoints

#### OpenAI-Compatible Endpoint

```
POST /api/{name}/v1/chat/completions
```

#### Anthropic-Compatible Endpoint

```
POST /api/{name}/v1/messages
```

#### Models List

```
GET /api/{name}/v1/models
```

### Client Configuration

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
# For APIs that support Anthropic format natively
export ANTHROPIC_BASE_URL=https://your-worker.workers.dev/api/deepseek
export ANTHROPIC_AUTH_TOKEN=your-global-api-token

# For APIs that only support OpenAI format
export ANTHROPIC_BASE_URL=https://your-worker.workers.dev/api/openai
export ANTHROPIC_AUTH_TOKEN=your-global-api-token

claude
```

#### cURL

```bash
# OpenAI format
curl https://your-worker.workers.dev/api/openai/v1/chat/completions \
  -H "Authorization: Bearer your-global-api-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Anthropic format
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

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MAIN_PASSWORD` | Password for the admin management page | Yes |

### D1 Database

Routes are stored in D1 with the following schema:

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

### Route Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Route name, used as URL prefix (e.g., `openai` → `/api/openai/v1/*`) |
| `remote_api_url` | Yes | Base URL of the API |
| `api_key` | No | Route-specific API key. Falls back to global token if empty |
| `messages_endpoint` | No | OpenAI chat completions endpoint (default: `/chat/completions`) |
| `anthropic_endpoint` | No | Anthropic messages endpoint (default: `/messages`) |
| `supports_openai` | No | Whether API supports OpenAI format (default: `true`) |
| `supports_anthropic` | No | Whether API supports Anthropic format (default: `false`) |
| `auth_type` | No | Authentication type: `bearer` or `api-key` (default: `bearer`) |
| `enabled` | No | Whether the route is active (default: `true`) |

### Supported Providers

Works with any AI API provider:

**Supports both formats (direct proxy):**
- **DeepSeek**: `https://api.deepseek.com/v1`
- **Anthropic**: `https://api.anthropic.com/v1`

**Supports OpenAI format only (Anthropic requests are converted):**
- **OpenAI**: `https://api.openai.com/v1`
- **Ollama** (local): `http://localhost:11434/v1`
- **vLLM** (local): `http://localhost:8000/v1`
- **LiteLLM**: `http://localhost:4000/v1`
- **Together AI**: `https://api.together.xyz/v1`
- **Groq**: `https://api.groq.com/openai/v1`

## API Reference

### Admin API

All admin endpoints require authentication via session cookie or `Authorization: Bearer <session-token>`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | Login with password |
| `POST` | `/api/auth/logout` | Logout |
| `GET` | `/api/auth/check` | Check authentication status |
| `GET` | `/api/routes` | List all routes |
| `POST` | `/api/routes` | Create a new route |
| `PUT` | `/api/routes/:id` | Update a route |
| `DELETE` | `/api/routes/:id` | Delete a route |
| `POST` | `/api/routes/test` | Test API format support |
| `GET` | `/api/settings` | Get settings (global token) |
| `PUT` | `/api/settings` | Update settings |
| `POST` | `/api/settings/generate-token` | Generate new global token |

### Proxy API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/{name}/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/api/{name}/v1/messages` | Anthropic-compatible messages |
| `GET` | `/api/{name}/v1/models` | List available models |
| `ANY` | `/api/{name}/v1/*` | Catch-all proxy to remote API |

## Testing

```bash
# Run all tests
npm test

# Run unit tests only
npm run test:unit

# Run integration tests only
npm run test:integration

# Run tests in watch mode
npm run test:watch
```

## Deployment

### GitHub Actions

The project includes a GitHub Actions workflow that automatically deploys to Cloudflare Workers on push to `main`.

Required secrets:
- `CLOUDFLARE_API_TOKEN`: Your Cloudflare API token
- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID

### Manual Deployment

```bash
# Deploy to Cloudflare Workers
npm run deploy
```

## Development

### Local Development

```bash
# Start local development server
npm start

# Initialize local D1 database
npm run db:init

# Reset local database
npm run db:reset
```

### Project Structure

```
cf-ai-proxy/
├── src/
│   ├── worker.js      # Main entry point and routing
│   ├── auth.js        # Authentication helpers
│   ├── db.js          # D1 database operations
│   ├── proxy.js       # Proxy forwarding logic
│   ├── converter.js   # Anthropic ↔ OpenAI format conversion
│   └── html.js        # Admin page HTML template
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── schema.sql         # D1 database schema
├── wrangler.toml      # Cloudflare Workers configuration
├── vitest.config.js   # Test configuration
└── package.json
```

## License

MIT
