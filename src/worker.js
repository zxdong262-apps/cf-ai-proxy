/**
 * Cloudflare AI API Proxy Worker
 *
 * Translates OpenAI-compatible and Anthropic Claude-compatible API requests
 * to Cloudflare AI REST API. The Bearer token from the incoming request is
 * forwarded as-is to Cloudflare — no token management needed in this worker.
 *
 * Supported endpoints:
 *   POST /v1/chat/completions      (OpenAI format)
 *   POST /v1/messages              (Anthropic/Claude format)
 *   GET  /v1/models                (live list from Cloudflare AI)
 *
 * Required env var (wrangler.toml [vars] or secret):
 *   CF_ACCOUNT_ID  – your Cloudflare account ID
 *
 * Client usage: pass your Cloudflare API token as the Bearer token.
 */

const DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const CF_AI_BASE    = (accountId) =>
  `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai`;

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return cors(new Response(null, { status: 204 }));
    }

    // Extract the client's bearer token — forwarded to Cloudflare as-is
    const cfToken = (request.headers.get("Authorization") ?? "").replace(/^Bearer\s+/i, "");
    if (!cfToken) {
      return cors(err(401, "missing_token", "Provide your Cloudflare API token as a Bearer token."));
    }

    if (!env.CF_ACCOUNT_ID) {
      return cors(err(500, "config_error", "CF_ACCOUNT_ID is not set on this worker."));
    }

    const { pathname } = new URL(request.url);

    try {
      if (request.method === "GET"  && pathname === "/v1/models")           return cors(await handleModels(cfToken, env));
      if (request.method === "POST" && pathname === "/v1/chat/completions") return cors(await handleOpenAI(request, cfToken, env));
      if (request.method === "POST" && pathname === "/v1/messages")         return cors(await handleAnthropic(request, cfToken, env));

      return cors(err(404, "not_found", `Unknown path: ${pathname}`));
    } catch (e) {
      console.error(e);
      return cors(err(500, "server_error", e.message));
    }
  },
};

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------
async function handleModels(cfToken, env) {
  const res = await cfFetch(`${CF_AI_BASE(env.CF_ACCOUNT_ID)}/models/search`, cfToken);
  if (!res.ok) return proxyError(res);

  const { result = [] } = await res.json();
  const now = Math.floor(Date.now() / 1000);
  return json({
    object: "list",
    data: result.map((m) => ({
      id: m.name,
      object: "model",
      created: now,
      owned_by: "cloudflare",
      description: m.description ?? "",
      task: m.task?.name ?? "",
    })),
  });
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions  (OpenAI format)
// ---------------------------------------------------------------------------
async function handleOpenAI(request, cfToken, env) {
  const { model, messages, stream = false, max_tokens, temperature, top_p, stop, system } =
    await request.json();

  const cfModel    = model || DEFAULT_MODEL;
  const cfMessages = normalizeMessages(messages, system);
  const cfBody     = {
    messages: cfMessages,
    stream,
    ...(max_tokens  != null && { max_tokens }),
    ...(temperature != null && { temperature }),
    ...(top_p       != null && { top_p }),
    ...(stop        != null && { stop }),
  };

  const res = await cfFetch(`${CF_AI_BASE(env.CF_ACCOUNT_ID)}/run/${cfModel}`, cfToken, cfBody);
  if (!res.ok) return proxyError(res);

  if (stream) {
    return new Response(openaiSSEStream(res.body, cfModel), { headers: sseHeaders() });
  }

  const { result } = await res.json();
  const text = result?.response ?? "";
  const now  = Math.floor(Date.now() / 1000);

  return json({
    id: `chatcmpl-${uid()}`,
    object: "chat.completion",
    created: now,
    model: cfModel,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
    usage: usageStats(cfMessages, text),
  });
}

// ---------------------------------------------------------------------------
// POST /v1/messages  (Anthropic / Claude format)
// ---------------------------------------------------------------------------
async function handleAnthropic(request, cfToken, env) {
  const { model, messages, stream = false, max_tokens = 1024, temperature, top_p, top_k, system, stop_sequences } =
    await request.json();

  const cfModel    = model || DEFAULT_MODEL;
  const cfMessages = normalizeMessages(messages, system);
  const cfBody     = {
    messages: cfMessages,
    max_tokens,
    stream,
    ...(temperature    != null && { temperature }),
    ...(top_p          != null && { top_p }),
    ...(top_k          != null && { top_k }),
    ...(stop_sequences != null && { stop: stop_sequences }),
  };

  const res = await cfFetch(`${CF_AI_BASE(env.CF_ACCOUNT_ID)}/run/${cfModel}`, cfToken, cfBody);
  if (!res.ok) return proxyError(res);

  if (stream) {
    return new Response(anthropicSSEStream(res.body, cfModel), { headers: sseHeaders() });
  }

  const { result } = await res.json();
  const text = result?.response ?? "";

  return json({
    id: `msg_${uid()}`,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: cfModel,
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: countTokens(cfMessages), output_tokens: countTokens(text) },
  });
}

// ---------------------------------------------------------------------------
// Streaming: CF AI SSE stream → OpenAI SSE chunks
// ---------------------------------------------------------------------------
function openaiSSEStream(body, model) {
  const id      = `chatcmpl-${uid()}`;
  const now     = Math.floor(Date.now() / 1000);
  const encoder = new TextEncoder();

  return new ReadableStream({
    async start(controller) {
      const reader = body.getReader();
      const send   = (chunk) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));

      try {
        for await (const token of readTokens(reader)) {
          send({ id, object: "chat.completion.chunk", created: now, model,
            choices: [{ index: 0, delta: { role: "assistant", content: token }, finish_reason: null }] });
        }
        send({ id, object: "chat.completion.chunk", created: now, model,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      } finally {
        reader.releaseLock();
        controller.close();
      }
    },
  });
}

// ---------------------------------------------------------------------------
// Streaming: CF AI SSE stream → Anthropic SSE events
// ---------------------------------------------------------------------------
function anthropicSSEStream(body, model) {
  const id      = `msg_${uid()}`;
  const encoder = new TextEncoder();

  return new ReadableStream({
    async start(controller) {
      const reader = body.getReader();
      const send   = (event, data) =>
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));

      send("message_start", { type: "message_start",
        message: { id, type: "message", role: "assistant", content: [], model,
          stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } });
      send("content_block_start", { type: "content_block_start", index: 0,
        content_block: { type: "text", text: "" } });
      send("ping", { type: "ping" });

      let outTokens = 0;
      try {
        for await (const token of readTokens(reader)) {
          outTokens += countTokens(token);
          send("content_block_delta", { type: "content_block_delta", index: 0,
            delta: { type: "text_delta", text: token } });
        }
      } finally {
        reader.releaseLock();
      }

      send("content_block_stop",  { type: "content_block_stop", index: 0 });
      send("message_delta", { type: "message_delta",
        delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: outTokens } });
      send("message_stop", { type: "message_stop" });
      controller.close();
    },
  });
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** POST to Cloudflare AI REST API with the client's token */
function cfFetch(url, cfToken, body) {
  return fetch(url, {
    method: body ? "POST" : "GET",
    headers: {
      "Authorization": `Bearer ${cfToken}`,
      "Content-Type": "application/json",
    },
    ...(body && { body: JSON.stringify(body) }),
  });
}

/** Proxy a non-OK Cloudflare response back to the client */
async function proxyError(res) {
  const text = await res.text();
  return new Response(text, { status: res.status, headers: { "Content-Type": "application/json" } });
}

/** Async generator: yields decoded token strings from a CF AI SSE stream reader */
async function* readTokens(reader) {
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const raw = typeof value === "string" ? value : new TextDecoder().decode(value);
    for (const line of raw.split("\n").filter((l) => l.trim())) {
      const dataStr = line.startsWith("data: ") ? line.slice(6) : line;
      if (dataStr === "[DONE]") continue;
      try {
        const parsed = JSON.parse(dataStr);
        const token  = parsed?.response ?? parsed?.token ?? "";
        if (token) yield token;
      } catch {
        if (dataStr) yield dataStr;
      }
    }
  }
}

/**
 * Normalise messages for Cloudflare AI:
 * - Prepend top-level `system` string if provided
 * - Flatten Anthropic multi-part content blocks to plain strings
 * - Merge consecutive system messages
 */
function normalizeMessages(messages = [], systemPrompt) {
  const out = [];

  if (systemPrompt) out.push({ role: "system", content: systemPrompt });

  for (const msg of messages) {
    const role    = msg.role === "assistant" ? "assistant"
                  : msg.role === "system"    ? "system"
                  :                            "user";

    const content = typeof msg.content === "string"
      ? msg.content
      : Array.isArray(msg.content)
        ? msg.content.map((b) =>
            typeof b === "string"    ? b
          : b.type === "text"        ? (b.text ?? "")
          : b.type === "tool_result" ? JSON.stringify(b.content ?? "")
          :                            ""
          ).join("\n").trim()
        : "";

    const existingSystem = out.find((m) => m.role === "system");
    if (role === "system" && existingSystem) {
      existingSystem.content += "\n" + content;
    } else {
      out.push({ role, content });
    }
  }

  return out;
}

function countTokens(input) {
  if (!input) return 0;
  return Math.ceil((typeof input === "string" ? input : JSON.stringify(input)).length / 4);
}

function usageStats(msgs, text) {
  const p = countTokens(msgs), c = countTokens(text);
  return { prompt_tokens: p, completion_tokens: c, total_tokens: p + c };
}

function uid() { return Math.random().toString(36).slice(2, 11); }

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
}

function err(status, type, message) { return json({ error: { type, message } }, status); }

function sseHeaders() {
  return { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" };
}

function cors(response) {
  const r = new Response(response.body, response);
  r.headers.set("Access-Control-Allow-Origin", "*");
  r.headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  r.headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");
  return r;
}