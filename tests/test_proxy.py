import json
import asyncio
import pytest

import httpx
from starlette.testclient import TestClient
from unittest.mock import patch

import llm_fallback as app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BytesStream(httpx.AsyncByteStream):
    """Simple async byte stream that yields a single chunk."""

    def __init__(self, data: bytes):
        self._data = data
        self._consumed = False

    async def __aiter__(self):
        if self._consumed:
            raise httpx.StreamConsumed()
        self._consumed = True
        yield self._data

    async def aclose(self):
        pass


class StreamingMockTransport(httpx.AsyncBaseTransport):
    """Async transport that returns proper httpx.Response with streaming body."""

    def __init__(self, handler):
        self._handler = handler

    async def handle_async_request(self, request):
        return self._handler(request)


def make_response(body, status=200, headers=None):
    """Create an httpx.Response with a streaming body."""
    if isinstance(body, dict):
        raw = json.dumps(body).encode()
    elif isinstance(body, str):
        raw = body.encode()
    else:
        raw = body
    resp_headers = dict(headers or {})
    resp_headers["content-type"] = "application/json"
    resp_headers["content-length"] = str(len(raw))
    stream = BytesStream(raw)
    return httpx.Response(
        status_code=status,
        headers=httpx.Headers(resp_headers),
        stream=stream,
    )


class MockHandler:
    """Callable that returns httpx.Responses based on a list of (body, status) tuples."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.call_count = 0
        self.calls = []

    def __call__(self, request):
        self.call_count += 1
        self.calls.append(request)
        idx = (self.call_count - 1) % len(self.responses)
        resp = self.responses[idx]
        if isinstance(resp, tuple):
            body, status = resp
            return make_response(body, status)
        return make_response(resp)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_globals():
    app.PROVIDERS.clear()
    app.MODELS.clear()
    app.TOKENS.clear()
    yield
    app.PROVIDERS.clear()
    app.MODELS.clear()
    app.TOKENS.clear()


@pytest.fixture(autouse=True)
def patch_is_reachable():
    with patch("llm_fallback.is_reachable", return_value=True):
        yield


def make_client(handler):
    """Create an httpx.AsyncClient with our streaming mock transport."""
    transport = StreamingMockTransport(handler)
    return httpx.AsyncClient(transport=transport, follow_redirects=True)


# ---------------------------------------------------------------------------
# /v1/models endpoint
# ---------------------------------------------------------------------------

class TestModelsEndpoint:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        app.load_config(tmp_config(FULL_CONFIG))
        app.client = make_client(MockHandler({"default": "response"}))
        self.client = TestClient(app.app)

    def test_list_models_full_access(self):
        resp = self.client.get(
            "/v1/models",
            headers={"Authorization": "Bearer full-access"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = {m["id"] for m in data["data"]}
        assert model_ids == {"gpt-4o", "local-llm", "fast"}

    def test_list_models_limited_access(self):
        resp = self.client.get(
            "/v1/models",
            headers={"Authorization": "Bearer limited"},
        )
        assert resp.status_code == 200
        model_ids = {m["id"] for m in resp.json()["data"]}
        assert model_ids == {"fast"}

    def test_list_models_no_auth(self):
        resp = self.client.get("/v1/models")
        assert resp.status_code == 401

    def test_list_models_invalid_token(self):
        resp = self.client.get(
            "/v1/models",
            headers={"Authorization": "Bearer bad-token"},
        )
        assert resp.status_code == 401

    def test_model_object_format(self):
        resp = self.client.get(
            "/v1/models",
            headers={"Authorization": "Bearer full-access"},
        )
        for m in resp.json()["data"]:
            assert m["object"] == "model"
            assert m["owned_by"] == "llm-fallback"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        app.load_config(tmp_config(MINIMAL_CONFIG))
        app.client = make_client(MockHandler({"ok": True}))
        self.client = TestClient(app.app)

    def _post(self, headers=None):
        return self.client.post(
            "/v1/chat/completions",
            headers=headers or {},
            json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
        )

    def test_no_auth_returns_401(self):
        resp = self._post()
        assert resp.status_code == 401
        assert "invalid_api_key" in resp.json()["error"]["code"]

    def test_bearer_token_accepted(self):
        resp = self._post(headers={"Authorization": "Bearer mytoken"})
        assert resp.status_code == 200

    def test_raw_token_accepted(self):
        resp = self._post(headers={"Authorization": "mytoken"})
        assert resp.status_code == 200

    def test_api_key_header_accepted(self):
        resp = self._post(headers={"api-key": "mytoken"})
        assert resp.status_code == 200

    def test_invalid_token_returns_401(self):
        resp = self._post(headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_token_scope_restriction(self, tmp_config, patch_is_reachable):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        app.load_config(tmp_config(FULL_CONFIG))
        app.client = make_client(MockHandler({"ok": True}))
        client = TestClient(app.app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer limited"},
            json={"model": "gpt-4o", "messages": []},
        )
        assert resp.status_code == 403
        assert "model_not_authorized" in resp.json()["error"]["code"]


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

class TestRequestValidation:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        app.load_config(tmp_config(MINIMAL_CONFIG))
        app.client = make_client(MockHandler({"ok": True}))
        self.client = TestClient(app.app)

    def test_missing_model_field(self):
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400
        assert "model_not_specified" in resp.json()["error"]["code"]

    def test_non_json_body(self):
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken", "Content-Type": "text/plain"},
            content=b"not json",
        )
        assert resp.status_code == 400

    def test_model_not_authorized(self, tmp_config, patch_is_reachable):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        app.load_config(tmp_config(FULL_CONFIG))
        app.client = make_client(MockHandler({"ok": True}))
        client = TestClient(app.app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer limited"},
            json={"model": "gpt-4o", "messages": []},
        )
        assert resp.status_code == 403
        assert "model_not_authorized" in resp.json()["error"]["code"]


# ---------------------------------------------------------------------------
# Proxy success
# ---------------------------------------------------------------------------

class TestProxySuccess:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        app.load_config(tmp_config(MINIMAL_CONFIG))
        app.client = make_client(MockHandler({"default": "response"}))
        self.client = TestClient(app.app)

    def test_proxy_passes_response(self):
        handler = MockHandler({"choices": [{"message": {"content": "hello"}}]})
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"},
            json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello"

    def test_model_name_rewritten(self):
        received = {}
        def capture(request):
            body = json.loads(request.content)
            received["model"] = body.get("model")
            received["auth"] = request.headers.get("authorization", "")
            return make_response({"ok": True})
        app.client = make_client(capture)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200
        assert received["model"] == "m1backend"
        assert received["auth"] == "Bearer tok1"

    def test_passthrough_4xx(self):
        handler = MockHandler(({"error": "bad"}, 400))
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class TestFallback:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        config = """\
providers:
  - id: p1
    url: http://127.0.0.1:19998/v1
    token: tok1
    retry_after: 0
  - id: p2
    url: http://127.0.0.1:19999/v1
    token: tok2
    retry_after: 0
models:
  - name: m1
    providers:
      - id: p1
        model_name: m1-on-p1
      - id: p2
        model_name: m1-on-p2
tokens:
  - token: tok
    models: [m1]
"""
        app.load_config(tmp_config(config))
        self.client = TestClient(app.app)

    def test_fallback_on_500(self):
        handler = MockHandler(
            ({"error": "err"}, 500),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "p2"

    def test_fallback_on_502(self):
        handler = MockHandler(
            ({"error": "bg"}, 502),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200

    def test_fallback_on_503(self):
        handler = MockHandler(
            ({"error": "unavail"}, 503),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200

    def test_fallback_on_504(self):
        handler = MockHandler(
            ({"error": "timeout"}, 504),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200

    def test_fallback_on_429(self):
        handler = MockHandler(
            ({"error": "rate"}, 429),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "p2"

    def test_no_fallback_on_400(self):
        handler = MockHandler(
            ({"error": "bad"}, 400),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 400
        assert handler.call_count == 1

    def test_no_fallback_on_401(self):
        handler = MockHandler(
            ({"error": "unauth"}, 401),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 401
        assert handler.call_count == 1

    def test_no_fallback_on_404(self):
        handler = MockHandler(
            ({"error": "not found"}, 404),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 404
        assert handler.call_count == 1

    def test_all_unreachable_returns_503(self):
        with patch("llm_fallback.is_reachable", return_value=False):
            handler = MockHandler(({"error": "x"}, 500))
            app.client = make_client(handler)
            resp = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer tok"},
                json={"model": "m1", "messages": []},
            )
            assert resp.status_code == 503
            assert "no_provider_available" in resp.json()["error"]["code"]

    def test_all_return_500_returns_503(self):
        handler = MockHandler(
            ({"error": "err"}, 500),
            ({"error": "err"}, 500),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        config = """\
providers:
  - id: p1
    url: http://127.0.0.1:19998/v1
    token: tok1
    retry_after: 2
  - id: p2
    url: http://127.0.0.1:19999/v1
    token: tok2
    retry_after: 0
models:
  - name: m1
    providers:
      - id: p1
        model_name: on-p1
      - id: p2
        model_name: on-p2
tokens:
  - token: tok
    models: [m1]
"""
        app.load_config(tmp_config(config))
        self.client = TestClient(app.app)

    def test_provider_in_cooldown_after_failure(self):
        handler = MockHandler(
            ({"error": "err"}, 500),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "p2"
        p1 = app.PROVIDERS["p1"]
        assert p1.in_cooldown()

    def test_cooldown_skips_provider(self):
        # First call: p1 fails (500), falls back to p2
        handler1 = MockHandler(
            ({"error": "err"}, 500),
            ({"source": "p2"}, 200),
        )
        app.client = make_client(handler1)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200
        assert app.PROVIDERS["p1"].in_cooldown()

        # Second call: p1 in cooldown, goes straight to p2
        handler2 = MockHandler(({"source": "p2-again"}, 200))
        app.client = make_client(handler2)
        resp2 = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp2.status_code == 200
        assert handler2.call_count == 1  # Only p2 was called


# ---------------------------------------------------------------------------
# Config strings
# ---------------------------------------------------------------------------

FULL_CONFIG = """\
providers:
  - id: openai
    url: https://api.openai.com/v1
    token: sk-openai-token
    retry_after: 30
  - id: ollama
    url: http://localhost:11434/v1
    token: ollama
    retry_after: 0
  - id: remote
    url: http://192.168.1.50:11434/v1
    token: rem-token
    retry_after: 60
    wake:
      mac_address: "aa:bb:cc:dd:ee:ff"
      max_retries: 3
      retry_delay: 5
models:
  - name: gpt-4o
    providers:
      - id: openai
        model_name: gpt-4o
      - id: ollama
        model_name: llama3.1:70b
  - name: local-llm
    providers:
      - id: remote
        model_name: llama3.1:70b
  - name: fast
    providers:
      - id: ollama
        model_name: llama3.1:8b
tokens:
  - token: full-access
    models: [gpt-4o, local-llm, fast]
  - token: limited
    models: [fast]
"""

MINIMAL_CONFIG = """\
providers:
  - id: p1
    url: http://127.0.0.1:19997/v1
    token: tok1
models:
  - name: m1
    providers:
      - id: p1
        model_name: m1backend
tokens:
  - token: mytoken
    models: [m1]
"""
