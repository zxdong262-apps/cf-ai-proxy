/**
 * D1 database operations for routes and settings.
 */

/**
 * Get all routes.
 */
export async function getRoutes(db) {
  const { results } = await db.prepare(
    'SELECT * FROM routes ORDER BY created_at DESC'
  ).all()
  return results
}

/**
 * Get a route by name.
 */
export async function getRouteByName(db, name) {
  return await db.prepare(
    'SELECT * FROM routes WHERE name = ? AND enabled = 1'
  ).bind(name).first()
}

/**
 * Get a route by ID.
 */
export async function getRouteById(db, id) {
  return await db.prepare(
    'SELECT * FROM routes WHERE id = ?'
  ).bind(id).first()
}

/**
 * Create a new route.
 */
export async function createRoute(db, route) {
  const result = await db.prepare(
    `INSERT INTO routes (name, remote_api_url, api_key, messages_endpoint, anthropic_endpoint,
     supports_openai, supports_anthropic, auth_type, enabled)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).bind(
    route.name,
    route.remote_api_url,
    route.api_key || '',
    route.messages_endpoint || '/chat/completions',
    route.anthropic_endpoint || '/messages',
    route.supports_openai !== undefined ? (route.supports_openai ? 1 : 0) : 1,
    route.supports_anthropic !== undefined ? (route.supports_anthropic ? 1 : 0) : 0,
    route.auth_type || 'bearer',
    route.enabled !== undefined ? route.enabled : 1
  ).run()
  return result.meta.last_row_id
}

/**
 * Update a route.
 */
export async function updateRoute(db, id, updates) {
  const fields = []
  const values = []

  if (updates.name !== undefined) { fields.push('name = ?'); values.push(updates.name) }
  if (updates.remote_api_url !== undefined) { fields.push('remote_api_url = ?'); values.push(updates.remote_api_url) }
  if (updates.api_key !== undefined) { fields.push('api_key = ?'); values.push(updates.api_key) }
  if (updates.messages_endpoint !== undefined) { fields.push('messages_endpoint = ?'); values.push(updates.messages_endpoint) }
  if (updates.anthropic_endpoint !== undefined) { fields.push('anthropic_endpoint = ?'); values.push(updates.anthropic_endpoint) }
  if (updates.supports_openai !== undefined) { fields.push('supports_openai = ?'); values.push(updates.supports_openai ? 1 : 0) }
  if (updates.supports_anthropic !== undefined) { fields.push('supports_anthropic = ?'); values.push(updates.supports_anthropic ? 1 : 0) }
  if (updates.auth_type !== undefined) { fields.push('auth_type = ?'); values.push(updates.auth_type) }
  if (updates.enabled !== undefined) { fields.push('enabled = ?'); values.push(updates.enabled) }

  if (fields.length === 0) return

  fields.push("updated_at = datetime('now')")
  values.push(id)

  await db.prepare(
    `UPDATE routes SET ${fields.join(', ')} WHERE id = ?`
  ).bind(...values).run()
}

/**
 * Delete a route.
 */
export async function deleteRoute(db, id) {
  await db.prepare('DELETE FROM routes WHERE id = ?').bind(id).run()
}

/**
 * Get a setting value by key.
 */
export async function getSetting(db, key) {
  const row = await db.prepare(
    'SELECT value FROM settings WHERE key = ?'
  ).bind(key).first()
  return row ? row.value : null
}

/**
 * Update a setting value.
 */
export async function updateSetting(db, key, value) {
  await db.prepare(
    `INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))
     ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`
  ).bind(key, value).run()
}
