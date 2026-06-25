import { describe, test, expect, beforeAll, beforeEach } from 'vitest'

/**
 * Integration tests for the Cloudflare Worker.
 *
 * These tests use miniflare's built-in worker environment.
 * They test the actual HTTP request/response flow.
 */

// Helper to create a mock D1 database
function createMockDB() {
  const routes = []
  const settings = new Map()
  settings.set('global_api_token', '')

  return {
    prepare(sql) {
      const self = {
        _sql: sql,
        _params: [],
        bind(...params) {
          self._params = params
          return self
        },
        async first() {
          if (self._sql.includes('FROM routes') && self._sql.includes('WHERE name')) {
            const name = self._params[0]
            return routes.find(r => r.name === name && r.enabled) || null
          }
          if (self._sql.includes('FROM settings')) {
            const key = self._params[0]
            const value = settings.get(key)
            return value !== undefined ? { value } : null
          }
          return null
        },
        async all() {
          if (self._sql.includes('FROM routes')) {
            return { results: [...routes] }
          }
          return { results: [] }
        },
        async run() {
          if (self._sql.includes('INSERT INTO routes')) {
            const id = routes.length + 1
            routes.push({
              id,
              name: self._params[0],
              remote_api_url: self._params[1],
              api_key: self._params[2],
              messages_endpoint: self._params[3],
              auth_type: self._params[4],
              enabled: self._params[5],
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            })
            return { meta: { last_row_id: id } }
          }
          if (self._sql.includes('INSERT INTO settings') || self._sql.includes('UPDATE settings')) {
            settings.set(self._params[0], self._params[1])
          }
          if (self._sql.includes('DELETE FROM routes')) {
            const id = self._params[0]
            const idx = routes.findIndex(r => r.id === id)
            if (idx >= 0) routes.splice(idx, 1)
          }
          return { meta: {} }
        }
      }
      return self
    },
    _routes: routes,
    _settings: settings
  }
}

describe('Worker Integration', () => {
  let worker
  let env
  let mockDB

  beforeAll(async () => {
    // Import the worker module
    const mod = await import('../../src/worker.js')
    worker = mod.default
  })

  beforeEach(() => {
    mockDB = createMockDB()
    env = {
      DB: mockDB,
      MAIN_PASSWORD: 'test-password'
    }
  })

  describe('GET /', () => {
    test('returns HTML page', async () => {
      const req = new Request('http://localhost/')
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(200)
      const html = await res.text()
      expect(html).toContain('AI Proxy Manager')
      expect(html).toContain('loginForm')
    })
  })

  describe('POST /api/auth/login', () => {
    test('returns 401 for wrong password', async () => {
      const req = new Request('http://localhost/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: 'wrong' })
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(401)
      const data = await res.json()
      expect(data.error.message).toBe('Invalid password')
    })

    test('returns success for correct password', async () => {
      const req = new Request('http://localhost/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: 'test-password' })
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(200)
      const data = await res.json()
      expect(data.success).toBe(true)
      expect(data.token).toBeDefined()
      // Check Set-Cookie header
      const cookie = res.headers.get('Set-Cookie')
      expect(cookie).toContain('session=')
    })

    test('returns 500 if MAIN_PASSWORD not set', async () => {
      env.MAIN_PASSWORD = ''
      const req = new Request('http://localhost/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: 'test' })
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(500)
    })
  })

  describe('Auth check', () => {
    test('returns unauthenticated without session', async () => {
      const req = new Request('http://localhost/api/auth/check')
      const res = await worker.fetch(req, env)
      const data = await res.json()
      expect(data.authenticated).toBe(false)
    })
  })

  describe('Protected endpoints', () => {
    test('GET /api/routes returns 401 without auth', async () => {
      const req = new Request('http://localhost/api/routes')
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(401)
    })

    test('GET /api/settings returns 401 without auth', async () => {
      const req = new Request('http://localhost/api/settings')
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(401)
    })
  })

  describe('Routes CRUD with auth', () => {
    let sessionToken

    beforeEach(async () => {
      // Login to get session token
      const loginReq = new Request('http://localhost/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: 'test-password' })
      })
      const loginRes = await worker.fetch(loginReq, env)
      const loginData = await loginRes.json()
      sessionToken = loginData.token
    })

    test('POST /api/routes creates a route', async () => {
      const req = new Request('http://localhost/api/routes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        },
        body: JSON.stringify({
          name: 'openai',
          remote_api_url: 'https://api.openai.com/v1'
        })
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(201)
      const data = await res.json()
      expect(data.success).toBe(true)
      expect(data.id).toBeDefined()
    })

    test('POST /api/routes rejects duplicate name', async () => {
      // Create first route
      const req1 = new Request('http://localhost/api/routes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        },
        body: JSON.stringify({
          name: 'openai',
          remote_api_url: 'https://api.openai.com/v1'
        })
      })
      await worker.fetch(req1, env)

      // Try duplicate
      const req2 = new Request('http://localhost/api/routes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        },
        body: JSON.stringify({
          name: 'openai',
          remote_api_url: 'https://api.openai.com/v1'
        })
      })
      const res = await worker.fetch(req2, env)
      expect(res.status).toBe(409)
    })

    test('POST /api/routes rejects invalid name', async () => {
      const req = new Request('http://localhost/api/routes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        },
        body: JSON.stringify({
          name: 'invalid name!',
          remote_api_url: 'https://api.openai.com/v1'
        })
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(400)
    })

    test('GET /api/routes returns routes list', async () => {
      // Create a route first
      const createReq = new Request('http://localhost/api/routes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        },
        body: JSON.stringify({
          name: 'test-route',
          remote_api_url: 'https://api.example.com/v1'
        })
      })
      await worker.fetch(createReq, env)

      // List routes
      const req = new Request('http://localhost/api/routes', {
        headers: { 'Authorization': `Bearer ${sessionToken}` }
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(200)
      const data = await res.json()
      expect(Array.isArray(data)).toBe(true)
      expect(data.length).toBeGreaterThan(0)
    })
  })

  describe('Proxy endpoints', () => {
    test('returns 404 for non-existent route', async () => {
      const req = new Request('http://localhost/api/nonexistent/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'test', messages: [] })
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(404)
    })
  })

  describe('Unknown routes', () => {
    test('returns 404 for unknown paths', async () => {
      const req = new Request('http://localhost/unknown')
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(404)
    })
  })

  describe('CORS', () => {
    test('handles OPTIONS request', async () => {
      const req = new Request('http://localhost/api/routes', {
        method: 'OPTIONS'
      })
      const res = await worker.fetch(req, env)
      expect(res.status).toBe(204)
      expect(res.headers.get('Access-Control-Allow-Origin')).toBe('*')
    })
  })
})
