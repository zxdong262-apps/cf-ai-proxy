/**
 * Cloudflare AI API Proxy Worker
 *
 * Routes:
 *   GET  /                          → Admin management page (password protected)
 *   POST /api/auth/login            → Login endpoint
 *   POST /api/auth/logout           → Logout endpoint
 *   GET  /api/auth/check            → Check auth status
 *   GET  /api/routes                → List all routes (auth required)
 *   POST /api/routes                → Create route (auth required)
 *   PUT  /api/routes/:id            → Update route (auth required)
 *   DELETE /api/routes/:id          → Delete route (auth required)
 *   POST /api/routes/test           → Test API format support (auth required)
 *   GET  /api/settings              → Get settings (auth required)
 *   PUT  /api/settings              → Update settings (auth required)
 *   POST /api/settings/generate-token → Generate new global token (auth required)
 *   POST /api/{name}/v1/chat/completions → OpenAI-compatible proxy
 *   POST /api/{name}/v1/messages    → Anthropic-compatible proxy
 *   GET  /api/{name}/v1/models      → Models list proxy
 *   ANY  /api/{name}/v1/*           → Catch-all proxy
 */

import { createSession, validateSession } from './auth.js'
import { getRoutes, createRoute, updateRoute, deleteRoute, getSetting, updateSetting } from './db.js'
import { handleProxy, testApiSupport } from './proxy.js'
import { getAdminPage } from './html.js'

export default {
  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') {
      return cors(new Response(null, { status: 204 }))
    }

    const url = new URL(request.url)
    const path = url.pathname

    try {
      // Admin page
      if (path === '/' || path === '/index.html') {
        return new Response(getAdminPage(), {
          headers: { 'Content-Type': 'text/html; charset=utf-8' }
        })
      }

      // Auth endpoints
      if (path === '/api/auth/login' && request.method === 'POST') {
        return await handleLogin(request, env)
      }
      if (path === '/api/auth/logout' && request.method === 'POST') {
        return handleLogout(env)
      }
      if (path === '/api/auth/check' && request.method === 'GET') {
        return handleAuthCheck(request, env)
      }

      // Settings endpoints (auth required)
      if (path === '/api/settings' && request.method === 'GET') {
        return await handleGetSettings(request, env)
      }
      if (path === '/api/settings' && request.method === 'PUT') {
        return await handleUpdateSettings(request, env)
      }
      if (path === '/api/settings/generate-token' && request.method === 'POST') {
        return await handleGenerateToken(request, env)
      }

      // Test API support endpoint (must be before routes CRUD)
      if (path === '/api/routes/test' && request.method === 'POST') {
        return await handleTestApiSupport(request, env)
      }

      // Routes CRUD (auth required)
      if (path === '/api/routes' && request.method === 'GET') {
        return await handleListRoutes(request, env)
      }
      if (path === '/api/routes' && request.method === 'POST') {
        return await handleCreateRoute(request, env)
      }

      const routeMatch = path.match(/^\/api\/routes\/(\d+)$/)
      if (routeMatch) {
        const id = parseInt(routeMatch[1])
        if (request.method === 'PUT') return await handleUpdateRoute(request, env, id)
        if (request.method === 'DELETE') return await handleDeleteRoute(request, env, id)
      }

      // API proxy: /api/{name}/v1/*
      const proxyMatch = path.match(/^\/api\/([^/]+)\/v1\/(.*)$/)
      if (proxyMatch) {
        const name = proxyMatch[1]
        const subPath = proxyMatch[2]
        return await handleProxy(request, env, name, subPath)
      }

      // Also handle /api/{name}/v1 (no trailing path)
      const proxyMatch2 = path.match(/^\/api\/([^/]+)\/v1$/)
      if (proxyMatch2) {
        const name = proxyMatch2[1]
        return await handleProxy(request, env, name, '')
      }

      return json({ error: { type: 'not_found', message: `Unknown path: ${path}` } }, 404)
    } catch (e) {
      console.error('Worker error:', e)
      return json({ error: { type: 'server_error', message: e.message } }, 500)
    }
  }
}

// ---------------------------------------------------------------------------
// Auth handlers
// ---------------------------------------------------------------------------

async function handleLogin(request, env) {
  const { password } = await request.json()
  const mainPassword = env.MAIN_PASSWORD

  if (!mainPassword) {
    return json({ error: { type: 'config_error', message: 'MAIN_PASSWORD not set' } }, 500)
  }

  if (!password || password !== mainPassword) {
    return json({ error: { type: 'auth_error', message: 'Invalid password' } }, 401)
  }

  const token = await createSession(env)
  const response = json({ success: true, token })
  response.headers.set('Set-Cookie', `session=${token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=86400`)
  return response
}

function handleLogout(env) {
  const response = json({ success: true })
  response.headers.set('Set-Cookie', 'session=; Path=/; HttpOnly; Max-Age=0')
  return response
}

function handleAuthCheck(request, env) {
  const isAuth = validateSession(request, env)
  return json({ authenticated: isAuth })
}

// ---------------------------------------------------------------------------
// Settings handlers
// ---------------------------------------------------------------------------

async function handleGetSettings(request, env) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  const token = await getSetting(env.DB, 'global_api_token')
  return json({ global_api_token: token || '' })
}

async function handleUpdateSettings(request, env) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  const { global_api_token } = await request.json()
  await updateSetting(env.DB, 'global_api_token', global_api_token || '')
  return json({ success: true })
}

async function handleGenerateToken(request, env) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  const token = generateToken()
  await updateSetting(env.DB, 'global_api_token', token)
  return json({ token })
}

// ---------------------------------------------------------------------------
// Routes CRUD handlers
// ---------------------------------------------------------------------------

async function handleListRoutes(request, env) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  const routes = await getRoutes(env.DB)
  return json(routes)
}

async function handleCreateRoute(request, env) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  const body = await request.json()
  const {
    name, remote_api_url, api_key, messages_endpoint, anthropic_endpoint,
    supports_openai, supports_anthropic, auth_type, enabled
  } = body

  if (!name || !remote_api_url) {
    return json({ error: { type: 'validation_error', message: 'name and remote_api_url are required' } }, 400)
  }

  // Validate name format (alphanumeric, hyphens, underscores)
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    return json({ error: { type: 'validation_error', message: 'name can only contain letters, numbers, hyphens, and underscores' } }, 400)
  }

  try {
    const id = await createRoute(env.DB, {
      name,
      remote_api_url,
      api_key: api_key || '',
      messages_endpoint: messages_endpoint || '/chat/completions',
      anthropic_endpoint: anthropic_endpoint || '/messages',
      supports_openai: supports_openai !== undefined ? supports_openai : true,
      supports_anthropic: supports_anthropic !== undefined ? supports_anthropic : false,
      auth_type: auth_type || 'bearer',
      enabled: enabled !== undefined ? (enabled ? 1 : 0) : 1
    })
    return json({ id, success: true }, 201)
  } catch (e) {
    if (e.message.includes('UNIQUE')) {
      return json({ error: { type: 'validation_error', message: 'Route name already exists' } }, 409)
    }
    throw e
  }
}

async function handleUpdateRoute(request, env, id) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  const body = await request.json()
  const {
    name, remote_api_url, api_key, messages_endpoint, anthropic_endpoint,
    supports_openai, supports_anthropic, auth_type, enabled
  } = body

  if (name && !/^[a-zA-Z0-9_-]+$/.test(name)) {
    return json({ error: { type: 'validation_error', message: 'name can only contain letters, numbers, hyphens, and underscores' } }, 400)
  }

  try {
    await updateRoute(env.DB, id, {
      name,
      remote_api_url,
      api_key,
      messages_endpoint,
      anthropic_endpoint,
      supports_openai,
      supports_anthropic,
      auth_type,
      enabled: enabled !== undefined ? (enabled ? 1 : 0) : undefined
    })
    return json({ success: true })
  } catch (e) {
    if (e.message.includes('UNIQUE')) {
      return json({ error: { type: 'validation_error', message: 'Route name already exists' } }, 409)
    }
    throw e
  }
}

async function handleDeleteRoute(request, env, id) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  await deleteRoute(env.DB, id)
  return json({ success: true })
}

// ---------------------------------------------------------------------------
// Test API support handler
// ---------------------------------------------------------------------------

async function handleTestApiSupport(request, env) {
  if (!validateSession(request, env)) {
    return json({ error: { type: 'auth_error', message: 'Authentication required' } }, 401)
  }

  const { url, api_key } = await request.json()

  if (!url) {
    return json({ error: { type: 'validation_error', message: 'url is required' } }, 400)
  }

  const results = await testApiSupport(env, url, api_key || '')
  return json(results)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function generateToken() {
  const bytes = new Uint8Array(32)
  crypto.getRandomValues(bytes)
  return 'sk-' + Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' }
  })
}

function cors(response) {
  const r = new Response(response.body, response)
  r.headers.set('Access-Control-Allow-Origin', '*')
  r.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
  r.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, Cookie')
  r.headers.set('Access-Control-Allow-Credentials', 'true')
  return r
}
