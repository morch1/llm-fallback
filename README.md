# llm-fallback

An OpenAI-compatible HTTP proxy that routes each model request to the first
**reachable** backend provider in a configured fallback list.

## How it works

- Exposes a transparent OpenAI-compatible API. Request/response bodies, headers,
  status codes and streaming are passed through unchanged — only the **model
  name** and **auth token** are rewritten for the chosen backend.
- For a request to model `m`, the providers listed for `m` are checked **in
  order** with a plain **TCP connectivity probe** (no LLM call is made to test
  availability). The request goes to the first provider that is online.
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
models:
  - name: gpt-4o                      # name clients request
    providers:                        # tried in this order
      - url: https://api.openai.com/v1  # base url of an OpenAI-compatible API
        model_name: gpt-4o              # model name sent to this backend
        token: sk-...                   # auth token sent to this backend
      - url: http://localhost:11434/v1
        model_name: llama3.1:70b
        token: ollama

tokens:
  - token: my-secret-access-token     # token clients send to THIS server
    models: [gpt-4o]                  # models this token may use
```

See [config.example.yaml](config.example.yaml).

### Provider `url`

The base URL of the backend's OpenAI-compatible API. The client's request path
is appended to it; a shared prefix is not duplicated, so both
`https://host/v1` and `https://host` work for a client calling
`/v1/chat/completions`.

## Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-secret-access-token" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
```

`GET /v1/models` returns the models the presented token may access.
