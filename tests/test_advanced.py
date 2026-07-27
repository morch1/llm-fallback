import json
import pytest

import httpx
from starlette.testclient import TestClient
from unittest.mock import patch

import llm_fallback as app


# Reuse helpers from test_proxy
from tests.test_proxy import (
    BytesStream, StreamingMockTransport, MockHandler, make_response, make_client,
)


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


# ---------------------------------------------------------------------------
# Wake-on-LAN
# ---------------------------------------------------------------------------

class TestWakeOnLan:
    """Test Wake-on-LAN retry behavior."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        config = """\
providers:
  - id: wol-p1
    url: http://10.0.0.1:11434/v1
    token: tok1
    retry_after: 0
    wake:
      mac_address: "aa:bb:cc:dd:ee:ff"
      max_retries: 3
      retry_delay: 0.01
  - id: p2
    url: http://127.0.0.1:19999/v1
    token: tok2
    retry_after: 0
models:
  - name: m1
    providers:
      - id: wol-p1
        model_name: m1-on-p1
      - id: p2
        model_name: m1-on-p2
tokens:
  - token: tok
    models: [m1]
"""
        app.load_config(tmp_config(config))
        self.client = TestClient(app.app)

    def test_wol_retries_on_unreachable(self):
        """When provider is unreachable, WoL is sent and retry is attempted."""
        wol_calls = []

        def mock_wol(mac, host):
            wol_calls.append((mac, host))

        reachable_count = [0]
        def mock_reachable(host, port, timeout=3.0):
            reachable_count[0] += 1
            return reachable_count[0] > 3  # Fail first 3 times, succeed on 4th

        handler = MockHandler({"source": "wol-p1"})
        app.client = make_client(handler)

        with patch("llm_fallback.send_wol", side_effect=mock_wol):
            with patch("llm_fallback.is_reachable", side_effect=mock_reachable):
                resp = self.client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer tok"},
                    json={"model": "m1", "messages": []},
                )

        # WoL should have been sent (up to max_retries times)
        assert len(wol_calls) > 0
        assert wol_calls[0][0] == "aa:bb:cc:dd:ee:ff"

    def test_wol_falls_back_after_exhausted_retries(self):
        """After WoL retries are exhausted, falls back to next provider."""
        wol_calls = []

        def mock_wol(mac, host):
            wol_calls.append((mac, host))

        # p1 unreachable, p2 reachable and succeeds
        call_count = [0]
        def handler_func(request):
            call_count[0] += 1
            return make_response({"source": "p2"})

        app.client = make_client(handler_func)

        reachable_count = [0]
        def mock_reachable(host, port, timeout=3.0):
            reachable_count[0] += 1
            # p1 (10.0.0.1) always unreachable, p2 (127.0.0.1) reachable
            return host == "127.0.0.1"

        with patch("llm_fallback.send_wol", side_effect=mock_wol):
            with patch("llm_fallback.is_reachable", side_effect=mock_reachable):
                resp = self.client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer tok"},
                    json={"model": "m1", "messages": []},
                )

        assert resp.status_code == 200
        assert resp.json()["source"] == "p2"
        assert len(wol_calls) > 0


# ---------------------------------------------------------------------------
# Header forwarding
# ---------------------------------------------------------------------------

class TestHeaderForwarding:
    """Test that headers are forwarded correctly (minus hop-by-hop)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        app.load_config(tmp_config(MINIMAL_CONFIG))
        self.client = TestClient(app.app)

    def test_custom_headers_forwarded(self):
        """Custom request headers are forwarded to the backend."""
        received_headers = {}

        def capture(request):
            for k, v in request.headers.items():
                received_headers[k.lower()] = v
            return make_response({"ok": True})

        app.client = make_client(capture)
        self.client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer mytoken",
                "X-Custom-Header": "custom-value",
                "X-Request-Id": "abc-123",
            },
            json={"model": "m1", "messages": []},
        )
        assert received_headers.get("x-custom-header") == "custom-value"
        assert received_headers.get("x-request-id") == "abc-123"

    def test_proxy_auth_stripped(self):
        """Proxy-Authorization header is not forwarded."""
        received_headers = {}

        def capture(request):
            for k, v in request.headers.items():
                received_headers[k.lower()] = v
            return make_response({"ok": True})

        app.client = make_client(capture)
        self.client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer mytoken",
                "Proxy-Authorization": "Basic secret",
            },
            json={"model": "m1", "messages": []},
        )
        assert "proxy-authorization" not in received_headers

    def test_api_key_header_stripped(self):
        """api-key header is stripped (not forwarded to backend)."""
        received_headers = {}

        def capture(request):
            for k, v in request.headers.items():
                received_headers[k.lower()] = v
            return make_response({"ok": True})

        app.client = make_client(capture)
        self.client.post(
            "/v1/chat/completions",
            headers={
                "api-key": "mytoken",
                "X-Custom": "value",
            },
            json={"model": "m1", "messages": []},
        )
        assert "api-key" not in received_headers


# ---------------------------------------------------------------------------
# Multi-provider chain
# ---------------------------------------------------------------------------

class TestMultiProviderChain:
    """Test fallback with 3+ providers."""

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
  - id: p3
    url: http://127.0.0.1:19997/v1
    token: tok3
    retry_after: 0
models:
  - name: m1
    providers:
      - id: p1
        model_name: on-p1
      - id: p2
        model_name: on-p2
      - id: p3
        model_name: on-p3
tokens:
  - token: tok
    models: [m1]
"""
        app.load_config(tmp_config(config))
        self.client = TestClient(app.app)

    def test_fallback_to_third_provider(self):
        """When p1 and p2 fail, falls back to p3."""
        handler = MockHandler(
            ({"error": "e1"}, 500),
            ({"error": "e2"}, 500),
            ({"source": "p3"}, 200),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "p3"

    def test_all_three_fail(self):
        """When all three providers fail, returns 503."""
        handler = MockHandler(
            ({"error": "e1"}, 500),
            ({"error": "e2"}, 500),
            ({"error": "e3"}, 500),
        )
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 503

    def test_first_provider_succeeds(self):
        """When first provider succeeds, no fallback."""
        handler = MockHandler({"source": "p1"})
        app.client = make_client(handler)
        resp = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "p1"
        assert handler.call_count == 1


# ---------------------------------------------------------------------------
# URL path handling
# ---------------------------------------------------------------------------

class TestUrlPathHandling:
    """Test URL path construction for different base URLs."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_config, patch_is_reachable):
        config = """\
providers:
  - id: with_path
    url: http://localhost:8080/v1
    token: tok1
    retry_after: 0
  - id: no_path
    url: http://localhost:8081
    token: tok2
    retry_after: 0
models:
  - name: m1
    providers:
      - id: with_path
        model_name: on-with-path
      - id: no_path
        model_name: on-no-path
  - name: m2
    providers:
      - id: no_path
        model_name: m2-on-no-path
tokens:
  - token: tok
    models: [m1, m2]
"""
        app.load_config(tmp_config(config))
        self.client = TestClient(app.app)

    def test_url_with_base_path(self):
        """Provider URL with /v1 path doesn't duplicate the prefix."""
        received_urls = []

        def capture(request):
            received_urls.append(str(request.url))
            return make_response({"ok": True})

        app.client = make_client(capture)
        self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m1", "messages": []},
        )
        # Should be http://localhost:8080/v1/chat/completions (not /v1/v1/...)
        assert received_urls[0] == "http://localhost:8080/v1/chat/completions"

    def test_url_without_base_path(self):
        """Provider URL without path appends the full path."""
        received_urls = []

        def capture(request):
            received_urls.append(str(request.url))
            return make_response({"ok": True})

        app.client = make_client(capture)
        self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok"},
            json={"model": "m2", "messages": []},
        )
        assert received_urls[0] == "http://localhost:8081/v1/chat/completions"


# Config strings
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
models:
  - name: gpt-4o
    providers:
      - id: openai
        model_name: gpt-4o
      - id: ollama
        model_name: llama3.1:70b
tokens:
  - token: full-access
    models: [gpt-4o]
  - token: limited
    models: []
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
