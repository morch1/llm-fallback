#!/usr/bin/env python3
"""llm-fallback.py — OpenAI-compatible LLM router with provider fallback.

Starts an HTTP server that exposes an OpenAI-compatible API. Each configured
model maps to an ordered list of backend providers. An incoming request for a
model is routed to the first usable provider, where a provider is tried when:

  * it is not in a post-failure cooldown (see ``retry_after`` below), and
  * its endpoint is reachable (a plain TCP connectivity check — no LLM request
    is made to probe availability).

Fallback to the next provider is triggered both when a provider is unreachable
and when it responds with a server error (HTTP 5xx, e.g. 502 Bad Gateway) or
429. A failed provider is put into cooldown for its ``retry_after`` seconds, and
is skipped outright (no availability check) for that period. ``retry_after`` is
optional and defaults to 0 (the provider is retried on every request).

The proxy is intentionally transparent: request/response bodies, headers,
status codes and streaming behaviour of the chosen provider are passed through
unchanged. The only things rewritten are the model name (client name -> the
provider's ``model_name``) and the auth token (client access token -> the
provider's ``token``).

Usage:
    python llm-fallback.py --config config.yaml [--host 0.0.0.0] [--port 8000]

Config (YAML):
    models:
      - name: gpt-4o
        providers:
          - url: https://api.openai.com/v1
            model_name: gpt-4o
            token: sk-...
            retry_after: 30        # optional, seconds; default 0
          - url: http://localhost:11434/v1
            model_name: llama3.1:70b
            token: ollama
    tokens:
      - token: my-secret-access-token
        models: [gpt-4o]
"""

import argparse
import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set
from urllib.parse import urlsplit

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

# Hop-by-hop headers must not be forwarded by a proxy (RFC 7230 6.1).
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class Provider:
    """A single backend OpenAI-compatible endpoint for a model."""

    def __init__(
        self, url: str, model_name: str, token: str, retry_after: float = 0.0
    ):
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.token = token

        try:
            self.retry_after = float(retry_after)
        except (TypeError, ValueError):
            raise ValueError(f"retry_after must be a number, got {retry_after!r}")
        if self.retry_after < 0:
            raise ValueError("retry_after must be >= 0")

        # Monotonic timestamp before which this provider is skipped outright
        # (no availability check, no request). 0 means "eligible now".
        self.cooldown_until: float = 0.0

        parts = urlsplit(url)
        if not parts.hostname:
            raise ValueError(f"provider url has no host: {url!r}")
        self.host: str = parts.hostname
        self.port: int = parts.port or (443 if parts.scheme == "https" else 80)
        # Path component of the base url, e.g. "/v1" for https://host/v1
        self.base_path: str = parts.path.rstrip("/")

    def in_cooldown(self) -> bool:
        """True if this provider failed recently and is still within retry_after."""
        return time.monotonic() < self.cooldown_until

    def mark_failed(self) -> None:
        """Record a failure; the provider is skipped for retry_after seconds.

        With the default retry_after of 0 this is a no-op in practice — the
        provider becomes eligible again on the very next request.
        """
        self.cooldown_until = time.monotonic() + self.retry_after

    def target_url(self, request_path: str) -> str:
        """Combine this provider's base url with the incoming request path.

        Avoids duplicating a shared prefix: if the base url already ends with
        the path prefix the client sent (e.g. base ".../v1" and request
        "/v1/chat/completions"), the prefix is not repeated.
        """
        path = request_path
        if self.base_path and path.startswith(self.base_path):
            path = path[len(self.base_path):]
        return self.url + path


class Model:
    """A logical model name backed by an ordered list of providers."""

    def __init__(self, name: str, providers: List[Provider]):
        self.name = name
        self.providers = providers


# Populated by load_config() before the server starts.
MODELS: Dict[str, Model] = {}
TOKENS: Dict[str, Set[str]] = {}

client: httpx.AsyncClient  # created on startup


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global client
    # No read timeout: LLM responses (especially streamed) can take a while.
    # A connect timeout still guards against dead sockets.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=None, write=None, pool=None),
        follow_redirects=True,
    )
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="llm-fallback", docs_url=None, redoc_url=None, lifespan=lifespan)


def load_config(path: str) -> None:
    """Load and validate the YAML config into the MODELS/TOKENS globals."""
    try:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        sys.exit(f"config file not found: {path}")
    except yaml.YAMLError as exc:
        sys.exit(f"failed to parse config {path}: {exc}")

    if not isinstance(raw, dict):
        sys.exit("config root must be a mapping with 'models' and 'tokens'")

    models_cfg = raw.get("models")
    if not isinstance(models_cfg, list) or not models_cfg:
        sys.exit("config must define a non-empty 'models' list")

    for entry in models_cfg:
        if not isinstance(entry, dict):
            sys.exit("each model must be a mapping")
        name = entry.get("name")
        if not name:
            sys.exit("each model needs a 'name'")
        providers_cfg = entry.get("providers")
        if not isinstance(providers_cfg, list) or not providers_cfg:
            sys.exit(f"model {name!r} must define a non-empty 'providers' list")

        providers: List[Provider] = []
        for p in providers_cfg:
            if not isinstance(p, dict):
                sys.exit(f"each provider of model {name!r} must be a mapping")
            missing = [k for k in ("url", "model_name", "token") if not p.get(k)]
            if missing:
                sys.exit(
                    f"provider of model {name!r} missing fields: {', '.join(missing)}"
                )
            try:
                providers.append(
                    Provider(
                        p["url"],
                        p["model_name"],
                        p["token"],
                        p.get("retry_after", 0),
                    )
                )
            except ValueError as exc:
                sys.exit(f"model {name!r}: {exc}")

        if name in MODELS:
            sys.exit(f"duplicate model name: {name!r}")
        MODELS[name] = Model(name, providers)

    tokens_cfg = raw.get("tokens")
    if not isinstance(tokens_cfg, list) or not tokens_cfg:
        sys.exit("config must define a non-empty 'tokens' list")

    for entry in tokens_cfg:
        if not isinstance(entry, dict):
            sys.exit("each token must be a mapping")
        token = entry.get("token")
        if not token:
            sys.exit("each token entry needs a 'token'")
        allowed = entry.get("models")
        if not isinstance(allowed, list) or not allowed:
            sys.exit(f"token {token!r} must define a non-empty 'models' list")
        for m in allowed:
            if m not in MODELS:
                sys.exit(
                    f"token {token!r} references unknown model {m!r} "
                    f"(not defined in 'models')"
                )
        TOKENS.setdefault(token, set()).update(allowed)


def error_response(status: int, message: str, err_type: str, code: str) -> JSONResponse:
    """Return an OpenAI-style error body."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code,
            }
        },
    )


def access_token_from(request: Request) -> Optional[str]:
    """Extract the bearer/access token from the request headers."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    if auth.strip():
        return auth.strip()
    # Azure-style header, just in case a client uses it.
    api_key = request.headers.get("api-key", "").strip()
    return api_key or None


async def is_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Plain TCP connectivity check — no protocol traffic, no LLM request."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


def is_failure_status(status: int) -> bool:
    """Statuses that should trigger fallback to the next provider.

    Server errors (5xx, e.g. 502 Bad Gateway / 503 / 504) and rate limiting
    (429) are treated as the provider being unable to serve the request, so we
    fall through to the next one. Other 4xx responses (400, 401, 404, ...) are
    request-level errors that would fail identically on every provider, so they
    are passed back to the client transparently instead.
    """
    return status >= 500 or status == 429


@app.api_route(
    "/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy(full_path: str, request: Request):
    # --- authentication ---------------------------------------------------
    token = access_token_from(request)
    if token is None or token not in TOKENS:
        return error_response(
            401, "Invalid authentication token.", "invalid_request_error", "invalid_api_key"
        )
    allowed = TOKENS[token]

    path = request.url.path

    # --- /v1/models discovery endpoint -----------------------------------
    if request.method == "GET" and path.rstrip("/").endswith("/models"):
        data = [
            {"id": name, "object": "model", "owned_by": "llm-fallback"}
            for name in MODELS
            if name in allowed
        ]
        return JSONResponse({"object": "list", "data": data})

    # --- determine requested model from the JSON body --------------------
    body = await request.body()
    payload = None
    if body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                payload = parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None

    requested_model = payload.get("model") if payload else None
    if not requested_model:
        return error_response(
            400,
            "Could not determine the target model: request body must be JSON "
            "containing a 'model' field.",
            "invalid_request_error",
            "model_not_specified",
        )

    if requested_model not in allowed:
        return error_response(
            403,
            f"This token is not authorized to access model '{requested_model}'.",
            "invalid_request_error",
            "model_not_authorized",
        )

    model = MODELS.get(requested_model)
    if model is None:  # token validation guarantees this, but be safe
        return error_response(
            404,
            f"Model '{requested_model}' does not exist.",
            "invalid_request_error",
            "model_not_found",
        )

    assert payload is not None  # requested_model was read from payload

    # Headers shared across providers; the authorization header is set per
    # provider below since each backend has its own token.
    base_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
        and k.lower() not in ("host", "content-length", "authorization", "api-key")
    }
    query = request.url.query

    # --- try providers in order, falling back on failure -----------------
    last_failure: Optional[str] = None
    for provider in model.providers:
        # Skip providers still in their post-failure cooldown window.
        if provider.in_cooldown():
            last_failure = f"{provider.url} in cooldown"
            continue

        # Plain TCP availability check (no LLM request).
        if not await is_reachable(provider.host, provider.port):
            provider.mark_failed()
            last_failure = f"{provider.url} unreachable"
            continue

        # Rewrite body (model name) and headers (auth token) per provider.
        payload["model"] = provider.model_name
        new_body = json.dumps(payload).encode("utf-8")
        fwd_headers = dict(base_headers)
        fwd_headers["authorization"] = f"Bearer {provider.token}"

        target = provider.target_url(path)
        if query:
            target = f"{target}?{query}"

        upstream_req = client.build_request(
            request.method, target, headers=fwd_headers, content=new_body
        )
        try:
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            provider.mark_failed()
            last_failure = f"{provider.url} request error: {exc}"
            continue

        # An HTTP error status (5xx/429) means this provider could not serve
        # the request — close it and fall through to the next provider.
        if is_failure_status(upstream_resp.status_code):
            status = upstream_resp.status_code
            await upstream_resp.aclose()
            provider.mark_failed()
            last_failure = f"{provider.url} returned HTTP {status}"
            continue

        # Success (or a non-retryable client error) — stream it back as-is.
        resp_headers = {
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in HOP_BY_HOP and k.lower() != "content-length"
        }
        media_type = resp_headers.pop("content-type", None)
        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=media_type,
            background=BackgroundTask(upstream_resp.aclose),
        )

    # No provider could serve the request.
    return error_response(
        503,
        f"No available provider for model '{requested_model}'"
        + (f" (last: {last_failure})." if last_failure else "."),
        "service_unavailable",
        "no_provider_available",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible LLM router with provider fallback."
    )
    parser.add_argument("--config", required=True, help="path to YAML config file")
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=8000, help="bind port (default 8000)"
    )
    args = parser.parse_args()

    load_config(args.config)
    print(
        f"llm-fallback: loaded {len(MODELS)} model(s), {len(TOKENS)} token(s); "
        f"listening on {args.host}:{args.port}"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
