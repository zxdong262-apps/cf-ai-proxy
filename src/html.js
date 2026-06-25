/**
 * Admin HTML page template.
 */

export function getAdminPage() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Proxy Manager</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f5f5f5;
      color: #333;
      line-height: 1.6;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    header {
      background: #fff;
      border-bottom: 1px solid #e0e0e0;
      padding: 16px 0;
      margin-bottom: 24px;
    }
    header .container {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    h1 { font-size: 24px; color: #1a1a1a; }
    h2 { font-size: 20px; margin-bottom: 16px; color: #1a1a1a; }
    h3 { font-size: 16px; margin-bottom: 12px; color: #444; }

    /* Login */
    .login-container {
      max-width: 400px;
      margin: 100px auto;
      background: #fff;
      padding: 32px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .login-container h2 { text-align: center; }

    /* Forms */
    .form-group {
      margin-bottom: 16px;
    }
    .form-group label {
      display: block;
      font-weight: 500;
      margin-bottom: 4px;
      font-size: 14px;
    }
    .form-group input, .form-group select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid #d0d0d0;
      border-radius: 6px;
      font-size: 14px;
    }
    .form-group input:focus, .form-group select:focus {
      outline: none;
      border-color: #4a90d9;
      box-shadow: 0 0 0 2px rgba(74,144,217,0.2);
    }
    .form-group .hint {
      font-size: 12px;
      color: #666;
      margin-top: 4px;
    }

    /* Checkbox group */
    .checkbox-group {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
    }
    .checkbox-group label {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: normal;
      cursor: pointer;
    }
    .checkbox-group input[type="checkbox"] {
      width: auto;
    }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 10px 16px;
      border: none;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn:hover { opacity: 0.9; }
    .btn-primary { background: #4a90d9; color: #fff; }
    .btn-success { background: #28a745; color: #fff; }
    .btn-danger { background: #dc3545; color: #fff; }
    .btn-secondary { background: #6c757d; color: #fff; }
    .btn-warning { background: #ffc107; color: #333; }
    .btn-sm { padding: 6px 12px; font-size: 13px; }
    .btn-block { width: 100%; justify-content: center; }
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    /* Cards */
    .card {
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      padding: 24px;
      margin-bottom: 24px;
    }

    /* Table */
    .table-container {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #e0e0e0;
      font-size: 14px;
    }
    th {
      background: #f8f9fa;
      font-weight: 600;
    }
    tr:hover { background: #f8f9fa; }

    /* Status badges */
    .badge {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 500;
    }
    .badge-success { background: #d4edda; color: #155724; }
    .badge-danger { background: #f8d7da; color: #721c24; }
    .badge-info { background: #d1ecf1; color: #0c5460; }
    .badge-warning { background: #fff3cd; color: #856404; }

    /* Token display */
    .token-display {
      display: flex;
      align-items: center;
      gap: 8px;
      background: #f8f9fa;
      padding: 12px;
      border-radius: 6px;
      margin-bottom: 16px;
    }
    .token-value {
      font-family: monospace;
      font-size: 14px;
      word-break: break-all;
      flex: 1;
    }
    .copy-btn {
      padding: 6px 12px;
      background: #4a90d9;
      color: #fff;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 13px;
      white-space: nowrap;
    }
    .copy-btn:hover { background: #357abd; }

    /* Modal */
    .modal-overlay {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.5);
      z-index: 1000;
      justify-content: center;
      align-items: center;
    }
    .modal-overlay.active { display: flex; }
    .modal {
      background: #fff;
      border-radius: 8px;
      padding: 24px;
      width: 90%;
      max-width: 600px;
      max-height: 90vh;
      overflow-y: auto;
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
    }
    .modal-header h3 { margin: 0; }
    .modal-close {
      background: none;
      border: none;
      font-size: 24px;
      cursor: pointer;
      color: #666;
    }

    /* Actions */
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    /* Toast */
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      padding: 12px 20px;
      background: #28a745;
      color: #fff;
      border-radius: 6px;
      font-size: 14px;
      z-index: 2000;
      opacity: 0;
      transition: opacity 0.3s;
    }
    .toast.show { opacity: 1; }
    .toast.error { background: #dc3545; }

    /* URL format helper */
    .url-format {
      font-size: 12px;
      color: #666;
      margin-top: 4px;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .container { padding: 12px; }
      .card { padding: 16px; }
      th, td { padding: 8px; font-size: 13px; }
      .btn { padding: 8px 12px; }
    }

    /* Logout button */
    .logout-btn {
      background: #6c757d;
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
    }
    .logout-btn:hover { background: #5a6268; }

    /* Usage section */
    .usage-section {
      background: #f8f9fa;
      padding: 16px;
      border-radius: 6px;
      margin-top: 16px;
    }
    .usage-section h3 {
      margin-bottom: 12px;
    }
    .usage-item {
      background: #fff;
      padding: 12px;
      border-radius: 4px;
      margin-bottom: 8px;
      font-family: monospace;
      font-size: 13px;
    }
    .usage-label {
      font-weight: 500;
      color: #666;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin-bottom: 4px;
    }

    /* Test result */
    .test-result {
      margin-top: 12px;
      padding: 12px;
      border-radius: 6px;
      font-size: 14px;
    }
    .test-result.success { background: #d4edda; color: #155724; }
    .test-result.error { background: #f8d7da; color: #721c24; }
    .test-result.info { background: #d1ecf1; color: #0c5460; }
  </style>
</head>
<body>
  <!-- Login Page -->
  <div id="loginPage" class="login-container">
    <h2>🔐 AI Proxy Manager</h2>
    <form id="loginForm">
      <div class="form-group">
        <label for="password">Password</label>
        <input type="password" id="password" placeholder="Enter password" required>
      </div>
      <button type="submit" class="btn btn-primary btn-block">Login</button>
    </form>
  </div>

  <!-- Main App -->
  <div id="app" style="display: none;">
    <header>
      <div class="container">
        <h1>🤖 AI Proxy Manager</h1>
        <button class="logout-btn" onclick="logout()">Logout</button>
      </div>
    </header>

    <div class="container">
      <!-- Global Token Section -->
      <div class="card">
        <h2>🔑 Global API Token</h2>
        <p style="margin-bottom: 16px; color: #666; font-size: 14px;">
          This token is used for all routes that don't have their own API key configured.
          Clients can pass this token in the <code>Authorization: Bearer &lt;token&gt;</code> header.
        </p>
        <div class="token-display">
          <span id="globalTokenValue" class="token-value">Not set</span>
          <button class="copy-btn" onclick="copyToken()">Copy</button>
        </div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-success" onclick="generateToken()">Generate New Token</button>
          <button class="btn btn-secondary" onclick="saveToken()">Save Token</button>
        </div>
        <div class="form-group" style="margin-top: 12px;">
          <label for="customToken">Or set custom token:</label>
          <input type="text" id="customToken" placeholder="sk-...">
        </div>
      </div>

      <!-- Routes Section -->
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h2 style="margin: 0;">📡 API Routes</h2>
          <button class="btn btn-primary" onclick="openAddModal()">+ Add Route</button>
        </div>

        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Remote URL</th>
                <th>Formats</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="routesTable">
            </tbody>
          </table>
        </div>

        <div id="noRoutes" style="display: none; text-align: center; padding: 32px; color: #666;">
          No routes configured. Click "Add Route" to get started.
        </div>
      </div>

      <!-- Usage Examples -->
      <div class="card">
        <h2>📖 Usage Examples</h2>
        <p style="margin-bottom: 16px; color: #666;">
          Use these endpoints in your applications. Replace <code>{name}</code> with your route name.
        </p>

        <div class="usage-section">
          <h3>OpenAI-Compatible Endpoint</h3>
          <div class="usage-item">
            <div class="usage-label">Chat Completions:</div>
            POST <span id="usageOpenAI">/api/{name}/v1/chat/completions</span>
          </div>
          <div class="usage-item">
            <div class="usage-label">Models:</div>
            GET <span id="usageModels">/api/{name}/v1/models</span>
          </div>
        </div>

        <div class="usage-section">
          <h3>Anthropic-Compatible Endpoint</h3>
          <div class="usage-item">
            <div class="usage-label">Messages:</div>
            POST <span id="usageAnthropic">/api/{name}/v1/messages</span>
          </div>
          <p style="margin-top: 8px; font-size: 13px; color: #666;">
            If the API natively supports Anthropic format, requests are proxied directly.
            Otherwise, they are automatically converted to/from OpenAI format.
          </p>
        </div>

        <div class="usage-section">
          <h3>Client Configuration Example</h3>
          <div class="usage-item">
            <div class="usage-label">OpenAI SDK:</div>
            <pre style="margin: 0; white-space: pre-wrap;">import OpenAI from 'openai'

const client = new OpenAI({
  apiKey: 'your-api-token',
  baseURL: '${typeof location !== 'undefined' ? location.origin : 'https://your-worker.workers.dev'}/api/openai/v1'
})

const response = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }]
})</pre>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Add/Edit Route Modal -->
  <div id="routeModal" class="modal-overlay">
    <div class="modal">
      <div class="modal-header">
        <h3 id="modalTitle">Add Route</h3>
        <button class="modal-close" onclick="closeModal()">&times;</button>
      </div>
      <form id="routeForm">
        <input type="hidden" id="routeId">
        <div class="form-group">
          <label for="routeName">Name *</label>
          <input type="text" id="routeName" placeholder="e.g., openai, deepseek" required>
          <div class="hint">Letters, numbers, hyphens, underscores only</div>
        </div>
        <div class="form-group">
          <label for="routeUrl">Remote API URL *</label>
          <input type="url" id="routeUrl" placeholder="https://api.openai.com/v1" required>
          <div class="hint">Base URL without trailing path (e.g., https://api.openai.com/v1)</div>
        </div>
        <div class="form-group">
          <label for="routeKey">API Key (optional)</label>
          <input type="password" id="routeKey" placeholder="Leave empty to use global token">
        </div>

        <div class="form-group">
          <label>Supported Formats</label>
          <div class="checkbox-group">
            <label>
              <input type="checkbox" id="supportsOpenai" checked>
              OpenAI (chat/completions)
            </label>
            <label>
              <input type="checkbox" id="supportsAnthropic">
              Anthropic (messages)
            </label>
          </div>
          <div class="hint">Click "Test API" to auto-detect supported formats</div>
        </div>

        <div class="form-group">
          <label for="routeEndpoint">OpenAI Endpoint</label>
          <input type="text" id="routeEndpoint" placeholder="/chat/completions" value="/chat/completions">
          <div class="hint">Path for OpenAI chat completions (default: /chat/completions)</div>
        </div>
        <div class="form-group">
          <label for="routeAnthropicEndpoint">Anthropic Endpoint</label>
          <input type="text" id="routeAnthropicEndpoint" placeholder="/messages" value="/messages">
          <div class="hint">Path for Anthropic messages (default: /messages)</div>
        </div>
        <div class="form-group">
          <label for="routeAuth">Auth Type</label>
          <select id="routeAuth">
            <option value="bearer">Bearer</option>
            <option value="api-key">API Key Header</option>
          </select>
        </div>
        <div class="form-group">
          <label>
            <input type="checkbox" id="routeEnabled" checked> Enabled
          </label>
        </div>

        <div id="testResult" style="display: none;"></div>

        <div style="display: flex; gap: 8px; justify-content: space-between; margin-top: 16px;">
          <button type="button" class="btn btn-warning" onclick="testApi()" id="testBtn">
            🔍 Test API
          </button>
          <div style="display: flex; gap: 8px;">
            <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
            <button type="submit" class="btn btn-primary">Save</button>
          </div>
        </div>
      </form>
    </div>
  </div>

  <!-- Toast -->
  <div id="toast" class="toast"></div>

  <script>
    const API_BASE = ''
    let currentToken = ''

    // Check auth on load
    async function checkAuth() {
      try {
        const res = await fetch(API_BASE + '/api/auth/check', {
          credentials: 'include'
        })
        const data = await res.json()
        if (data.authenticated) {
          showApp()
        } else {
          showLogin()
        }
      } catch {
        showLogin()
      }
    }

    function showLogin() {
      document.getElementById('loginPage').style.display = 'block'
      document.getElementById('app').style.display = 'none'
    }

    function showApp() {
      document.getElementById('loginPage').style.display = 'none'
      document.getElementById('app').style.display = 'block'
      loadSettings()
      loadRoutes()
    }

    // Login
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
      e.preventDefault()
      const password = document.getElementById('password').value
      try {
        const res = await fetch(API_BASE + '/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ password })
        })
        const data = await res.json()
        if (data.success) {
          showApp()
        } else {
          showToast(data.error?.message || 'Login failed', true)
        }
      } catch (e) {
        showToast('Login failed', true)
      }
    })

    // Logout
    async function logout() {
      await fetch(API_BASE + '/api/auth/logout', {
        method: 'POST',
        credentials: 'include'
      })
      showLogin()
    }

    // Load settings
    async function loadSettings() {
      try {
        const res = await fetch(API_BASE + '/api/settings', {
          credentials: 'include'
        })
        const data = await res.json()
        currentToken = data.global_api_token || ''
        document.getElementById('globalTokenValue').textContent = currentToken || 'Not set'
        document.getElementById('customToken').value = currentToken
      } catch (e) {
        console.error('Failed to load settings:', e)
      }
    }

    // Generate token
    async function generateToken() {
      try {
        const res = await fetch(API_BASE + '/api/settings/generate-token', {
          method: 'POST',
          credentials: 'include'
        })
        const data = await res.json()
        if (data.token) {
          currentToken = data.token
          document.getElementById('globalTokenValue').textContent = data.token
          document.getElementById('customToken').value = data.token
          showToast('Token generated successfully')
        }
      } catch (e) {
        showToast('Failed to generate token', true)
      }
    }

    // Save custom token
    async function saveToken() {
      const token = document.getElementById('customToken').value
      try {
        const res = await fetch(API_BASE + '/api/settings', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ global_api_token: token })
        })
        const data = await res.json()
        if (data.success) {
          currentToken = token
          document.getElementById('globalTokenValue').textContent = token || 'Not set'
          showToast('Token saved successfully')
        }
      } catch (e) {
        showToast('Failed to save token', true)
      }
    }

    // Copy token
    function copyToken() {
      if (!currentToken) {
        showToast('No token to copy', true)
        return
      }
      navigator.clipboard.writeText(currentToken).then(() => {
        showToast('Token copied to clipboard')
      }).catch(() => {
        // Fallback
        const textarea = document.createElement('textarea')
        textarea.value = currentToken
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand('copy')
        document.body.removeChild(textarea)
        showToast('Token copied to clipboard')
      })
    }

    // Load routes
    async function loadRoutes() {
      try {
        const res = await fetch(API_BASE + '/api/routes', {
          credentials: 'include'
        })
        const routes = await res.json()
        renderRoutes(routes)
      } catch (e) {
        console.error('Failed to load routes:', e)
      }
    }

    // Format badges
    function formatBadges(route) {
      const badges = []
      if (route.supports_openai) {
        badges.push('<span class="badge badge-info">OpenAI</span>')
      }
      if (route.supports_anthropic) {
        badges.push('<span class="badge badge-warning">Anthropic</span>')
      }
      if (!route.supports_openai && !route.supports_anthropic) {
        badges.push('<span class="badge badge-danger">None</span>')
      }
      return badges.join(' ')
    }

    // Render routes table
    function renderRoutes(routes) {
      const tbody = document.getElementById('routesTable')
      const noRoutes = document.getElementById('noRoutes')

      if (routes.length === 0) {
        tbody.innerHTML = ''
        noRoutes.style.display = 'block'
        return
      }

      noRoutes.style.display = 'none'
      tbody.innerHTML = routes.map(route => \`
        <tr>
          <td><strong>\${escapeHtml(route.name)}</strong></td>
          <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis;">\${escapeHtml(route.remote_api_url)}</td>
          <td>\${formatBadges(route)}</td>
          <td>
            <span class="badge \${route.enabled ? 'badge-success' : 'badge-danger'}">
              \${route.enabled ? 'Active' : 'Disabled'}
            </span>
          </td>
          <td class="actions">
            <button class="btn btn-sm btn-secondary" onclick="editRoute(\${route.id})">Edit</button>
            <button class="btn btn-sm btn-danger" onclick="deleteRoute(\${route.id}, '\${escapeHtml(route.name)}')">Delete</button>
          </td>
        </tr>
      \`).join('')
    }

    // Open add modal
    function openAddModal() {
      document.getElementById('modalTitle').textContent = 'Add Route'
      document.getElementById('routeForm').reset()
      document.getElementById('routeId').value = ''
      document.getElementById('routeEnabled').checked = true
      document.getElementById('supportsOpenai').checked = true
      document.getElementById('supportsAnthropic').checked = false
      document.getElementById('routeEndpoint').value = '/chat/completions'
      document.getElementById('routeAnthropicEndpoint').value = '/messages'
      document.getElementById('testResult').style.display = 'none'
      document.getElementById('routeModal').classList.add('active')
    }

    // Edit route
    async function editRoute(id) {
      try {
        const res = await fetch(API_BASE + '/api/routes', {
          credentials: 'include'
        })
        const routes = await res.json()
        const route = routes.find(r => r.id === id)
        if (!route) return

        document.getElementById('modalTitle').textContent = 'Edit Route'
        document.getElementById('routeId').value = route.id
        document.getElementById('routeName').value = route.name
        document.getElementById('routeUrl').value = route.remote_api_url
        document.getElementById('routeKey').value = route.api_key
        document.getElementById('routeEndpoint').value = route.messages_endpoint
        document.getElementById('routeAnthropicEndpoint').value = route.anthropic_endpoint || '/messages'
        document.getElementById('supportsOpenai').checked = !!route.supports_openai
        document.getElementById('supportsAnthropic').checked = !!route.supports_anthropic
        document.getElementById('routeAuth').value = route.auth_type
        document.getElementById('routeEnabled').checked = !!route.enabled
        document.getElementById('testResult').style.display = 'none'
        document.getElementById('routeModal').classList.add('active')
      } catch (e) {
        showToast('Failed to load route', true)
      }
    }

    // Close modal
    function closeModal() {
      document.getElementById('routeModal').classList.remove('active')
    }

    // Test API support
    async function testApi() {
      const url = document.getElementById('routeUrl').value
      const apiKey = document.getElementById('routeKey').value

      if (!url) {
        showToast('Please enter a URL first', true)
        return
      }

      const testBtn = document.getElementById('testBtn')
      const testResult = document.getElementById('testResult')
      testBtn.disabled = true
      testBtn.textContent = '⏳ Testing...'
      testResult.style.display = 'none'

      try {
        const res = await fetch(API_BASE + '/api/routes/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ url, api_key: apiKey })
        })
        const data = await res.json()

        // Update checkboxes based on results
        document.getElementById('supportsOpenai').checked = data.openai
        document.getElementById('supportsAnthropic').checked = data.anthropic

        // Show result
        let resultHtml = '<div class="test-result '
        if (data.openai || data.anthropic) {
          resultHtml += 'success'
        } else {
          resultHtml += 'error'
        }
        resultHtml += '">'

        if (data.openai) {
          resultHtml += '✅ <strong>OpenAI format:</strong> Supported<br>'
        } else {
          resultHtml += '❌ <strong>OpenAI format:</strong> Not supported'
          if (data.openai_error) resultHtml += ' (' + escapeHtml(data.openai_error) + ')'
          resultHtml += '<br>'
        }

        if (data.anthropic) {
          resultHtml += '✅ <strong>Anthropic format:</strong> Supported<br>'
        } else {
          resultHtml += '❌ <strong>Anthropic format:</strong> Not supported'
          if (data.anthropic_error) resultHtml += ' (' + escapeHtml(data.anthropic_error) + ')'
          resultHtml += '<br>'
        }

        resultHtml += '</div>'
        testResult.innerHTML = resultHtml
        testResult.style.display = 'block'

        if (data.openai || data.anthropic) {
          showToast('API test completed')
        } else {
          showToast('API test failed - no supported formats found', true)
        }
      } catch (e) {
        showToast('Failed to test API: ' + e.message, true)
      } finally {
        testBtn.disabled = false
        testBtn.textContent = '🔍 Test API'
      }
    }

    // Save route
    document.getElementById('routeForm').addEventListener('submit', async (e) => {
      e.preventDefault()
      const id = document.getElementById('routeId').value
      const data = {
        name: document.getElementById('routeName').value,
        remote_api_url: document.getElementById('routeUrl').value,
        api_key: document.getElementById('routeKey').value,
        messages_endpoint: document.getElementById('routeEndpoint').value,
        anthropic_endpoint: document.getElementById('routeAnthropicEndpoint').value,
        supports_openai: document.getElementById('supportsOpenai').checked,
        supports_anthropic: document.getElementById('supportsAnthropic').checked,
        auth_type: document.getElementById('routeAuth').value,
        enabled: document.getElementById('routeEnabled').checked
      }

      try {
        const url = id ? \`\${API_BASE}/api/routes/\${id}\` : \`\${API_BASE}/api/routes\`
        const method = id ? 'PUT' : 'POST'
        const res = await fetch(url, {
          method,
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(data)
        })
        const result = await res.json()
        if (result.success || result.id) {
          closeModal()
          loadRoutes()
          showToast(id ? 'Route updated' : 'Route created')
        } else {
          showToast(result.error?.message || 'Failed to save route', true)
        }
      } catch (e) {
        showToast('Failed to save route', true)
      }
    })

    // Delete route
    async function deleteRoute(id, name) {
      if (!confirm(\`Are you sure you want to delete route "\${name}"?\`)) return
      try {
        const res = await fetch(\`\${API_BASE}/api/routes/\${id}\`, {
          method: 'DELETE',
          credentials: 'include'
        })
        const data = await res.json()
        if (data.success) {
          loadRoutes()
          showToast('Route deleted')
        }
      } catch (e) {
        showToast('Failed to delete route', true)
      }
    }

    // Toast notification
    function showToast(message, isError = false) {
      const toast = document.getElementById('toast')
      toast.textContent = message
      toast.className = 'toast' + (isError ? ' error' : '') + ' show'
      setTimeout(() => {
        toast.className = 'toast'
      }, 3000)
    }

    // Escape HTML
    function escapeHtml(str) {
      if (!str) return ''
      const div = document.createElement('div')
      div.textContent = str
      return div.innerHTML
    }

    // Init
    checkAuth()
  </script>
</body>
</html>`
}
