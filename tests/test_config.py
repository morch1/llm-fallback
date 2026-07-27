import sys
import pytest

import llm_fallback as app


class TestConfigMissingFields:
    """Config validation: missing or malformed required fields."""

    def test_missing_config_file(self, tmp_path):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="config file not found"):
            app.load_config(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_yaml(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        path = tmp_config("providers: [\n  - id: p1\n    url: [invalid")
        with pytest.raises(SystemExit, match="failed to parse config"):
            app.load_config(path)

    def test_root_not_mapping(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="config root must be a mapping"):
            app.load_config(tmp_config("[1, 2, 3]"))

    def test_empty_providers(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="non-empty 'providers' list"):
            app.load_config(tmp_config("providers: []\nmodels: []\ntokens: []"))

    def test_missing_provider_id(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="each provider needs an 'id'"):
            app.load_config(tmp_config(
                "providers:\n  - url: http://x.com\n    token: t\n"
                "models: []\ntokens: []"
            ))

    def test_duplicate_provider_id(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="duplicate provider id"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://a.com\n    token: t\n"
                "  - id: p1\n    url: http://b.com\n    token: t\n"
                "models: []\ntokens: []"
            ))

    def test_missing_provider_url(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="missing fields.*url.*"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    token: t\n"
                "models: []\ntokens: []"
            ))

    def test_missing_provider_token(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="missing fields.*token.*"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n"
                "models: []\ntokens: []"
            ))

    def test_empty_models(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="non-empty 'models' list"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models: []\ntokens: []"
            ))

    def test_missing_model_name(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="each model needs a 'name'"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - providers: []\ntokens: []"
            ))

    def test_empty_model_providers(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="non-empty 'providers' list"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers: []\ntokens: []"
            ))

    def test_unknown_provider_ref(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="references unknown provider"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - id: unknown\n"
                "tokens: []"
            ))

    def test_missing_provider_ref_id(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="missing 'id'"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - model_name: x\n"
                "tokens: []"
            ))

    def test_duplicate_model_name(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="duplicate model name"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "  - name: m1\n    providers:\n      - id: p1\n"
                "tokens: []"
            ))

    def test_empty_tokens(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="non-empty 'tokens' list"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens: []"
            ))

    def test_missing_token_value(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="each token entry needs a 'token'"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - models: [m1]"
            ))

    def test_empty_token_models(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="non-empty 'models' list"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: t1\n    models: []"
            ))

    def test_unknown_model_in_token(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="references unknown model"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: t1\n    models: [unknown]"
            ))


class TestConfigValid:
    """Config loads successfully with valid YAML."""

    def test_minimal_config(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        app.load_config(tmp_config(MINIMAL_CONFIG))
        assert "p1" in app.PROVIDERS
        assert "m1" in app.MODELS
        assert "mytoken" in app.TOKENS

    def test_full_config(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        app.load_config(tmp_config(FULL_CONFIG))
        assert len(app.PROVIDERS) == 3
        assert len(app.MODELS) == 3
        assert len(app.TOKENS) == 2
        assert app.TOKENS["full-access"] == {"gpt-4o", "local-llm", "fast"}
        assert app.TOKENS["limited"] == {"fast"}

    def test_model_name_default(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        app.load_config(tmp_config(
            "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
            "models:\n  - name: m1\n    providers:\n      - id: p1\n"
            "tokens:\n  - token: tok\n    models: [m1]"
        ))
        ref = app.MODELS["m1"].providers[0]
        assert ref.model_name == "m1"

    def test_model_name_override(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        app.load_config(tmp_config(
            "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
            "models:\n  - name: m1\n    providers:\n      - id: p1\n        model_name: backend-name\n"
            "tokens:\n  - token: tok\n    models: [m1]"
        ))
        ref = app.MODELS["m1"].providers[0]
        assert ref.model_name == "backend-name"

    def test_token_merges_models(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        app.load_config(tmp_config(
            "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n"
            "models:\n  - name: m1\n    providers:\n      - id: p1\n"
            "  - name: m2\n    providers:\n      - id: p1\n"
            "tokens:\n  - token: tok\n    models: [m1]\n"
            "  - token: tok\n    models: [m2]"
        ))
        assert app.TOKENS["tok"] == {"m1", "m2"}


class TestProviderValidation:
    """Provider-specific field validation."""

    def test_invalid_retry_after(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="retry_after must be a number"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    retry_after: abc\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_negative_retry_after(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="retry_after must be >= 0"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    retry_after: -1\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_wake_not_mapping(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="wake must be a mapping"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    wake: true\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_wake_with_mac_missing_raises(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="wake requires 'mac_address'"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    wake:\n      other_field: val\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_wake_invalid_mac(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="invalid mac_address"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    wake:\n      mac_address: 'not-a-mac'\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_wake_invalid_max_retries(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="wake.max_retries must be >= 1"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    wake:\n      mac_address: 'aa:bb:cc:dd:ee:ff'\n      max_retries: 0\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_wake_invalid_retry_delay(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="wake.retry_delay must be a number"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    wake:\n      mac_address: 'aa:bb:cc:dd:ee:ff'\n      retry_delay: abc\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_wake_negative_retry_delay(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="wake.retry_delay must be >= 0"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: http://x.com\n    token: t\n    wake:\n      mac_address: 'aa:bb:cc:dd:ee:ff'\n      retry_delay: -1\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))

    def test_provider_url_no_host(self, tmp_config):
        app.PROVIDERS.clear()
        app.MODELS.clear()
        app.TOKENS.clear()
        with pytest.raises(SystemExit, match="provider url has no host"):
            app.load_config(tmp_config(
                "providers:\n  - id: p1\n    url: 'not-a-url'\n    token: t\n"
                "models:\n  - name: m1\n    providers:\n      - id: p1\n"
                "tokens:\n  - token: tok\n    models: [m1]"
            ))


# ---------------------------------------------------------------------------
# Config strings
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = """\
providers:
  - id: p1
    url: http://localhost:9999/v1
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
