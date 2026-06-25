/**
 * Proxy handler for forwarding requests to remote AI APIs.
 *
 * Supports:
 * - OpenAI-compatible passthrough (POST /api/{name}/v1/chat/completions)
 * - Anthropic passthrough for APIs that support it natively (POST /api/{name}/v1/messages)
 * - Anthropic → OpenAI conversion for APIs that only support OpenAI format
 * - Generic proxy for other /api/{name}/v1/* endpoints
 */

import { getRouteByName, getSetting } from './db.js'
import { anthropicToOpenAI, openAIToAnthropic, openAIChunkToAnthropicEvents } from './converter.js'

/**
 * Main proxy handler.
 */
export async function handleProxy(request, env, name, subPath) {
  // Look up route in D1
  const route = await getRouteByName(env.DB, name)
  if (!route) {
    return json({ error: { type: 'not_found', message: `Route '${name}' not found or disabled` } }, 404)
  }

  // Determine API key: route-specific or global
  let apiKey = route.api_key
  if (!apiKey) {
    apiKey = await getSetting(env.DB, 'global_api_token') || ''
  }

  // Check if client provided a token (passthrough mode)
  const clientAuth = request.headers.get('Authorization') || ''
  const clientToken = clientAuth.replace(/^Bearer\s+/i, '')
  if (!apiKey && clientToken) {
    apiKey = clientToken
  }

  if (!apiKey) {
    return json({ error: { type: 'auth_error', message: 'No API key configured and no client token provided' } }, 401)
  }

  // Handle Anthropic messages endpoint
  if (subPath === 'messages' && request.method === 'POST') {
    // If API natively supports Anthropic, proxy directly
    if (route.supports_anthropic) {
      return await handleAnthropicPassthrough(request, route, apiKey)
    }
    // Otherwise convert Anthropic → OpenAI
    return await handleAnthropicToOpenAI(request, route, apiKey)
  }

  // Handle OpenAI chat completions endpoint
  if (subPath === 'chat/completions' && request.method === 'POST') {
    return await handleDirectProxy(request, route, apiKey, subPath)
  }

  // Handle all other endpoints - direct proxy
  return await handleDirectProxy(request, route, apiKey, subPath)
}

/**
 * Direct passthrough for APIs that natively support Anthropic format.
 */
async function handleAnthropicPassthrough(request, route, apiKey) {
  const anthropicEndpoint = route.anthropic_endpoint || '/messages'
  const url = `${route.remote_api_url}${anthropicEndpoint}`

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`,
    'anthropic-version': request.headers.get('anthropic-version') || '2023-06-01'
  }

  // Copy x-api-key header if present (some Anthropic APIs use this)
  const xApiKey = request.headers.get('x-api-key')
  if (xApiKey) {
    headers['x-api-key'] = xApiKey
  }

  const body = await request.text()

  try {
    const upstreamRes = await fetch(url, {
      method: 'POST',
      headers,
      body
    })

    // Pass through the response directly (streaming or non-streaming)
    const responseHeaders = new Headers()
    const contentType = upstreamRes.headers.get('Content-Type')
    if (contentType) responseHeaders.set('Content-Type', contentType)

    return new Response(upstreamRes.body, {
      status: upstreamRes.status,
      headers: responseHeaders
    })
  } catch (e) {
    return json({
      type: 'error',
      error: { type: 'proxy_error', message: e.message }
    }, 502)
  }
}

/**
 * Handle Anthropic Messages API → OpenAI Chat Completions conversion.
 */
async function handleAnthropicToOpenAI(request, route, apiKey) {
  const body = await request.json()
  const isStream = body.stream === true

  // Convert Anthropic request to OpenAI format
  const openaiBody = anthropicToOpenAI(body)
  openaiBody.stream = isStream

  const endpoint = route.messages_endpoint || '/chat/completions'
  const url = `${route.remote_api_url}${endpoint}`

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  }

  try {
    const upstreamRes = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(openaiBody)
    })

    if (!upstreamRes.ok) {
      const errorText = await upstreamRes.text()
      return json({
        type: 'error',
        error: {
          type: 'api_error',
          message: `Upstream returned ${upstreamRes.status}: ${errorText}`
        }
      }, upstreamRes.status)
    }

    if (isStream) {
      return handleAnthropicStreamingResponse(upstreamRes, body.model)
    }

    // Non-streaming: convert OpenAI response back to Anthropic format
    const openaiRes = await upstreamRes.json()
    const anthropicRes = openAIToAnthropic(openaiRes, body.model)
    return json(anthropicRes)
  } catch (e) {
    return json({
      type: 'error',
      error: { type: 'proxy_error', message: e.message }
    }, 502)
  }
}

/**
 * Convert OpenAI SSE stream to Anthropic SSE stream.
 */
function handleAnthropicStreamingResponse(upstreamRes, model) {
  const encoder = new TextEncoder()
  const decoder = new TextDecoder()

  const stream = new ReadableStream({
    async start(controller) {
      const reader = upstreamRes.body.getReader()
      const state = { started: false, outputTokens: 0, model }
      let buffer = ''

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed || !trimmed.startsWith('data: ')) continue

            const payload = trimmed.slice(6)
            if (payload === '[DONE]') continue

            try {
              const openaiChunk = JSON.parse(payload)
              const events = openAIChunkToAnthropicEvents(openaiChunk, state)
              for (const evt of events) {
                controller.enqueue(encoder.encode(`event: ${evt.event}\ndata: ${evt.data}\n\n`))
              }
            } catch {
              // skip unparseable chunks
            }
          }
        }
      } catch (e) {
        console.error('Stream error:', e)
      } finally {
        reader.releaseLock()
        controller.close()
      }
    }
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    }
  })
}

/**
 * Direct proxy for OpenAI-compatible endpoints.
 */
async function handleDirectProxy(request, route, apiKey, subPath) {
  const url = `${route.remote_api_url}/${subPath}`

  const headers = new Headers()
  headers.set('Authorization', `Bearer ${apiKey}`)
  headers.set('Accept', 'application/json')

  // Copy relevant headers from the original request
  const contentType = request.headers.get('Content-Type')
  if (contentType) headers.set('Content-Type', contentType)

  const fetchOptions = {
    method: request.method,
    headers
  }

  if (request.method !== 'GET' && request.method !== 'HEAD') {
    fetchOptions.body = await request.text()
  }

  try {
    const upstreamRes = await fetch(url, fetchOptions)

    // If it's a streaming response, pass through directly
    if (upstreamRes.headers.get('content-type')?.includes('text/event-stream')) {
      return new Response(upstreamRes.body, {
        status: upstreamRes.status,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive'
        }
      })
    }

    // For non-streaming, proxy the response
    const responseHeaders = new Headers()
    responseHeaders.set('Content-Type', upstreamRes.headers.get('Content-Type') || 'application/json')

    return new Response(upstreamRes.body, {
      status: upstreamRes.status,
      headers: responseHeaders
    })
  } catch (e) {
    return json({
      error: { type: 'proxy_error', message: e.message }
    }, 502)
  }
}

/**
 * Test if an API supports OpenAI and/or Anthropic formats.
 */
export async function testApiSupport(env, url, apiKey) {
  const results = {
    openai: false,
    anthropic: false,
    openai_error: null,
    anthropic_error: null
  }

  // Test OpenAI format
  try {
    const openaiRes = await fetch(`${url}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: 'test',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1
      })
    })

    // If we get a response (even an error), the endpoint exists
    if (openaiRes.status !== 404 && openaiRes.status !== 405) {
      results.openai = true
    } else {
      results.openai_error = `Status ${openaiRes.status}`
    }
  } catch (e) {
    results.openai_error = e.message
  }

  // Test Anthropic format
  try {
    const anthropicRes = await fetch(`${url}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'test',
        max_tokens: 1,
        messages: [{ role: 'user', content: 'hi' }]
      })
    })

    // If we get a response (even an error), the endpoint exists
    if (anthropicRes.status !== 404 && anthropicRes.status !== 405) {
      results.anthropic = true
    } else {
      results.anthropic_error = `Status ${anthropicRes.status}`
    }
  } catch (e) {
    results.anthropic_error = e.message
  }

  return results
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' }
  })
}
