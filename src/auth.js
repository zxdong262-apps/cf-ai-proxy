/**
 * Authentication helpers for session management.
 *
 * Sessions are stored in-memory (Map) with a TTL.
 * For a single-worker deployment this is sufficient.
 */

// In-memory session store: token → expiry timestamp
const sessions = new Map()

const SESSION_TTL = 24 * 60 * 60 * 1000 // 24 hours

/**
 * Create a new session and return the token.
 */
export async function createSession(env) {
  const token = crypto.randomUUID()
  sessions.set(token, Date.now() + SESSION_TTL)
  return token
}

/**
 * Validate a session from the request (Cookie or Authorization header).
 */
export function validateSession(request, env) {
  const token = extractToken(request)
  if (!token) return false

  const expiry = sessions.get(token)
  if (!expiry) return false

  if (Date.now() > expiry) {
    sessions.delete(token)
    return false
  }

  return true
}

/**
 * Clear a session (logout).
 */
export function clearSession(request) {
  const token = extractToken(request)
  if (token) sessions.delete(token)
}

/**
 * Extract session token from Cookie or Authorization header.
 */
function extractToken(request) {
  // Try Cookie first
  const cookie = request.headers.get('Cookie') || ''
  const sessionMatch = cookie.match(/session=([^;]+)/)
  if (sessionMatch) return sessionMatch[1]

  // Try Authorization header
  const auth = request.headers.get('Authorization') || ''
  if (auth.startsWith('Bearer ')) return auth.slice(7)

  return null
}

/**
 * Hash password using Web Crypto API (SHA-256).
 * Note: For MAIN_PASSWORD comparison we do plain text since it's env-configured.
 */
export async function hashPassword(password) {
  const encoder = new TextEncoder()
  const data = encoder.encode(password)
  const hash = await crypto.subtle.digest('SHA-256', data)
  return Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * Constant-time string comparison to prevent timing attacks.
 */
export function safeCompare(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string') return false
  if (a.length !== b.length) return false

  let result = 0
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i)
  }
  return result === 0
}
