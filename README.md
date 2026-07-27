# llm-fallback

An OpenAI-compatible HTTP proxy that routes each model request to the first
**reachable** backend provider in a configured fallback list.

## How it works

- Exposes a transparent OpenAI-compatible API. Request/response bodies, headers,
  status codes and streaming are passed through unchanged — only the **model
  name** and **auth token** are rewritten for the chosen backend.
- Providers are defined once at the top level and referenced by models. For a
  request to model `m`, the providers listed for `m` are checked **in order**
  with a plain **TCP connectivity probe** (no LLM call is made to test
  availability). The request goes to the first provider that is online.
- **Fallback also triggers on errors:** if a chosen provider answers with an
  HTTP server error (5xx, e.g. `502 Bad Gateway`) or `429`, the router moves on
  to the next provider. Other 4xx responses (e.g. `400`, `401`) are passed back
  to the client unchanged, since they would fail identically everywhere.
- **`retry_after` (optional, per provider, seconds, default `0`):** after a
  provider fails, it is skipped outright — no availability check — for this many
  seconds before being tried again. `0` means retry on every request.
- **`wake` (optional, per provider):** if the provider is unreachable, send a
  **Wake-on-LAN** magic packet to wake the machine, then retry the request.
  Fields:
  - ``mac_address`` — MAC address of the provider's machine (required)
  - ``max_retries`` — number of WoL + retry attempts (default `1`, minimum `1`)
  - ``retry_delay`` — seconds to wait after each WoL packet before retrying
    (default `1.0`)
  If all retries fail, the normal fallback chain continues to the next provider.
- Clients authenticate with an **access token**; each token is scoped to a set
  of model names it is allowed to use.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python llm-fallback.py --config config.yaml [--host 0.0.0.0] [--port 8000]
```

## Configuration

```yaml
providers:
  - id: openai                          # unique identifier
    url: https://api.openai.com/v1       # base url of an OpenAI-compatible API
    token: sk-...                        # auth token sent to this backend
    retry_after: 30                      # optional; skip for 30s after a failure

  - id: remote-ollama
    url: http://192.168.1.50:11434/v1
    token: ollama
    wake:                                # optional; wake the machine on failure
      mac_address: "aa:bb:cc:dd:ee:ff"
      max_retries: 3                     # default 1
      retry_delay: 5                     # default 1.0 (seconds)

models:
  - name: gpt-4o                        # name clients request
    providers:                           # tried in this order
      - id: openai
        model_name: gpt-4o               # optional; model name sent to backend
                                          # if omitted, defaults to model's "name"
      - id: remote-ollama
        model_name: llama3.1:70b

tokens:
  - token: my-secret-access-token       # token clients send to THIS server
    models: [gpt-4o]                    # models this token may use
```

See [config.example.yaml](config.example.yaml).

### Provider `url`

The base URL of the backend's OpenAI-compatible API. The client's request path
is appended to it; a shared prefix is not duplicated, so both
`https://host/v1` and `https://host` work for a client calling
`/v1/chat/completions`.

### Model `providers` entries

Each entry references a top-level provider by `id` and optionally overrides the
`model_name` sent to that backend. When `model_name` is omitted, the model's
own `name` is used, which is convenient when the same model name exists on
multiple backends.

## Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-access-token" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
```

`GET /v1/models` returns the models the presented token may access.

## Tests

126 tests covering config validation, provider routing, authentication,
fallback logic, cooldown behaviour, Wake-on-LAN, and header forwarding.

```bash
pip install pytest
python -m pytest tests/ -v
```

Works on both Linux and Windows. No external services required — all backend
interactions are mocked.
