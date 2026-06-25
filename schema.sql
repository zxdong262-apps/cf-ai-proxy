-- AI Proxy D1 Database Schema

CREATE TABLE IF NOT EXISTS routes (
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
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Insert default global API token (will be generated on first login)
INSERT OR IGNORE INTO settings (key, value) VALUES ('global_api_token', '');
