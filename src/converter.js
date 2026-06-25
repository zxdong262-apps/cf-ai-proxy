/**
 * Converts between Anthropic Messages API and OpenAI Chat Completions API formats.
 * Reused from temp/ai-proxy/src/converter.js
 */

/**
 * Extract text from an Anthropic content block (string or array).
 */
export function extractText(content) {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join('')
  }
  return ''
}

/**
 * Convert Anthropic message content to OpenAI content format.
 */
export function convertContent(content) {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content.map((block) => {
      if (block.type === 'text') {
        return { type: 'text', text: block.text }
      }
      if (block.type === 'image') {
        return {
          type: 'image_url',
          image_url: {
            url: `data:${block.source.media_type};base64,${block.source.data}`
          }
        }
      }
      return block
    })
  }
  return String(content)
}

/**
 * Convert Anthropic tools to OpenAI tools format.
 */
function convertTools(tools) {
  if (!tools || !tools.length) return undefined
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.input_schema || { type: 'object', properties: {} }
    }
  }))
}

/**
 * Convert Anthropic tool_choice to OpenAI tool_choice.
 */
function convertToolChoice(toolChoice) {
  if (!toolChoice) return undefined
  if (toolChoice.type === 'auto') return 'auto'
  if (toolChoice.type === 'any') return 'required'
  if (toolChoice.type === 'tool' && toolChoice.name) {
    return { type: 'function', function: { name: toolChoice.name } }
  }
  return 'auto'
}

/**
 * Convert a single Anthropic message to one or more OpenAI messages.
 */
function convertMessage(msg) {
  if (typeof msg.content === 'string') {
    return [{ role: msg.role, content: msg.content }]
  }

  if (!Array.isArray(msg.content)) {
    return [{ role: msg.role, content: String(msg.content) }]
  }

  if (msg.role === 'assistant') {
    const textParts = msg.content.filter((b) => b.type === 'text')
    const toolUseBlocks = msg.content.filter((b) => b.type === 'tool_use')
    const text = textParts.map((b) => b.text).join('')

    const openaiMsg = { role: 'assistant', content: text || null }

    if (toolUseBlocks.length > 0) {
      openaiMsg.tool_calls = toolUseBlocks.map((block) => ({
        id: block.id,
        type: 'function',
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input || {})
        }
      }))
    }

    return [openaiMsg]
  }

  if (msg.role === 'user') {
    const toolResults = msg.content.filter((b) => b.type === 'tool_result')
    const nonTool = msg.content.filter((b) => b.type !== 'tool_result')

    const messages = []

    if (nonTool.length > 0) {
      const converted = convertContent(nonTool)
      if (typeof converted === 'string' ? converted : converted.length > 0) {
        messages.push({ role: 'user', content: converted })
      }
    }

    for (const result of toolResults) {
      let toolContent
      if (result.is_error) {
        toolContent = typeof result.content === 'string'
          ? result.content
          : extractText(result.content)
      } else if (typeof result.content === 'string') {
        toolContent = result.content
      } else if (Array.isArray(result.content)) {
        toolContent = extractText(result.content)
      } else {
        toolContent = String(result.content || '')
      }

      messages.push({
        role: 'tool',
        tool_call_id: result.tool_use_id,
        content: toolContent
      })
    }

    return messages
  }

  return [{ role: msg.role, content: convertContent(msg.content) }]
}

/**
 * Convert an Anthropic messages request body to OpenAI chat completions format.
 */
export function anthropicToOpenAI(body) {
  const messages = []

  if (body.system) {
    const systemText =
      typeof body.system === 'string'
        ? body.system
        : extractText(body.system)
    messages.push({ role: 'system', content: systemText })
  }

  for (const msg of body.messages || []) {
    messages.push(...convertMessage(msg))
  }

  const openai = {
    model: body.model,
    max_tokens: body.max_tokens,
    messages
  }

  if (body.stream) openai.stream = true
  if (body.temperature !== undefined) openai.temperature = body.temperature
  if (body.top_p !== undefined) openai.top_p = body.top_p
  if (body.stop_sequences) openai.stop = body.stop_sequences

  const tools = convertTools(body.tools)
  if (tools) {
    openai.tools = tools
    const toolChoice = convertToolChoice(body.tool_choice)
    if (toolChoice) openai.tool_choice = toolChoice
  }

  return openai
}

/**
 * Map OpenAI finish_reason to Anthropic stop_reason.
 */
export function mapStopReason(finishReason) {
  const map = {
    stop: 'end_turn',
    length: 'max_tokens',
    tool_calls: 'tool_use',
    content_filter: 'end_turn'
  }
  return map[finishReason] || 'end_turn'
}

/**
 * Generate a simple message ID.
 */
export function generateId() {
  return 'msg_' + Math.random().toString(36).slice(2, 14)
}

/**
 * Convert an OpenAI chat completions response to Anthropic messages format.
 */
export function openAIToAnthropic(openaiRes, requestModel) {
  const choice = openaiRes.choices?.[0]
  const message = choice?.message
  const text = message?.content || ''

  const content = []

  if (text) {
    content.push({ type: 'text', text })
  }

  if (message?.tool_calls) {
    for (const tc of message.tool_calls) {
      let input = {}
      try {
        input = JSON.parse(tc.function.arguments)
      } catch {
        input = { raw: tc.function.arguments }
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input
      })
    }
  }

  if (content.length === 0) {
    content.push({ type: 'text', text: '' })
  }

  return {
    id: openaiRes.id || generateId(),
    type: 'message',
    role: 'assistant',
    content,
    model: openaiRes.model || requestModel,
    stop_reason: mapStopReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: openaiRes.usage?.prompt_tokens || 0,
      output_tokens: openaiRes.usage?.completion_tokens || 0
    }
  }
}

/**
 * Convert a single OpenAI streaming chunk to zero or more Anthropic SSE events.
 */
export function openAIChunkToAnthropicEvents(chunk, state) {
  const events = []
  const delta = chunk.choices?.[0]?.delta
  const finishReason = chunk.choices?.[0]?.finish_reason

  if (!state.started) {
    state.started = true
    state.contentIndex = 0
    state.toolCallBlocks = {}

    events.push({
      event: 'message_start',
      data: JSON.stringify({
        type: 'message_start',
        message: {
          id: chunk.id || generateId(),
          type: 'message',
          role: 'assistant',
          content: [],
          model: chunk.model || state.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 }
        }
      })
    })

    events.push({
      event: 'content_block_start',
      data: JSON.stringify({
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' }
      })
    })
  }

  if (delta?.content) {
    state.outputTokens++
    events.push({
      event: 'content_block_delta',
      data: JSON.stringify({
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: delta.content }
      })
    })
  }

  if (delta?.tool_calls) {
    for (const tc of delta.tool_calls) {
      const tcIndex = tc.index

      if (!state.toolCallBlocks[tcIndex]) {
        if (!state.textBlockClosed && state.started) {
          state.textBlockClosed = true
          events.push({
            event: 'content_block_stop',
            data: JSON.stringify({ type: 'content_block_stop', index: 0 })
          })
        }

        state.contentIndex++
        state.toolCallBlocks[tcIndex] = {
          contentIndex: state.contentIndex,
          name: tc.function?.name || '',
          arguments: ''
        }

        events.push({
          event: 'content_block_start',
          data: JSON.stringify({
            type: 'content_block_start',
            index: state.contentIndex,
            content_block: {
              type: 'tool_use',
              id: tc.id || `toolu_${generateId()}`,
              name: tc.function?.name || '',
              input: {}
            }
          })
        })
      }

      if (tc.function?.arguments) {
        state.toolCallBlocks[tcIndex].arguments += tc.function.arguments
        events.push({
          event: 'content_block_delta',
          data: JSON.stringify({
            type: 'content_block_delta',
            index: state.toolCallBlocks[tcIndex].contentIndex,
            delta: {
              type: 'input_json_delta',
              partial_json: tc.function.arguments
            }
          })
        })
      }
    }
  }

  if (finishReason) {
    if (!state.textBlockClosed) {
      state.textBlockClosed = true
      events.push({
        event: 'content_block_stop',
        data: JSON.stringify({ type: 'content_block_stop', index: 0 })
      })
    }

    for (const tcIndex of Object.keys(state.toolCallBlocks || {})) {
      const block = state.toolCallBlocks[tcIndex]
      events.push({
        event: 'content_block_stop',
        data: JSON.stringify({ type: 'content_block_stop', index: block.contentIndex })
      })
    }

    events.push({
      event: 'message_delta',
      data: JSON.stringify({
        type: 'message_delta',
        delta: {
          stop_reason: mapStopReason(finishReason),
          stop_sequence: null
        },
        usage: { output_tokens: state.outputTokens }
      })
    })

    events.push({
      event: 'message_stop',
      data: JSON.stringify({ type: 'message_stop' })
    })
  }

  return events
}
