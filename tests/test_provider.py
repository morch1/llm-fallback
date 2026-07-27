import pytest
import time

from llm_fallback import Provider


class TestProviderInit:
    """Test Provider.__init__ parsing and defaults."""

    def test_basic(self):
        p = Provider("p1", "http://localhost:8080/v1", "tok")
        assert p.id == "p1"
        assert p.url == "http://localhost:8080/v1"
        assert p.token == "tok"
        assert p.host == "localhost"
        assert p.port == 8080
        assert p.base_path == "/v1"
        assert p.retry_after == 0.0
        assert p.wol_mac is None

    def test_trailing_slash_removed(self):
        p = Provider("p1", "http://localhost:8080/v1/", "tok")
        assert p.url == "http://localhost:8080/v1"

    def test_https_default_port(self):
        p = Provider("p1", "https://api.example.com/v1", "tok")
        assert p.port == 443

    def test_http_default_port(self):
        p = Provider("p1", "http://api.example.com", "tok")
        assert p.port == 80

    def test_no_path(self):
        p = Provider("p1", "http://localhost:9999", "tok")
        assert p.base_path == ""

    def test_retry_after_default(self):
        p = Provider("p1", "http://localhost:8080", "tok")
        assert p.retry_after == 0.0

    def test_retry_after_set(self):
        p = Provider("p1", "http://localhost:8080", "tok", retry_after=45)
        assert p.retry_after == 45.0

    def test_retry_after_float(self):
        p = Provider("p1", "http://localhost:8080", "tok", retry_after=1.5)
        assert p.retry_after == 1.5

    def test_invalid_retry_after_type(self):
        with pytest.raises(ValueError, match="retry_after must be a number"):
            Provider("p1", "http://localhost:8080", "tok", retry_after="abc")

    def test_negative_retry_after(self):
        with pytest.raises(ValueError, match="retry_after must be >= 0"):
            Provider("p1", "http://localhost:8080", "tok", retry_after=-5)

    def test_url_no_host(self):
        with pytest.raises(ValueError, match="provider url has no host"):
            Provider("p1", "not-a-url", "tok")


class TestProviderWake:
    """Test Wake-on-LAN configuration in Provider."""

    def test_wake_defaults(self):
        p = Provider(
            "p1", "http://10.0.0.1:8080", "tok",
            wake={"mac_address": "aa:bb:cc:dd:ee:ff"},
        )
        assert p.wol_mac == "aa:bb:cc:dd:ee:ff"
        assert p.wol_max_retries == 1
        assert p.wol_retry_delay == 1.0

    def test_wake_hyphen_mac_normalized(self):
        p = Provider(
            "p1", "http://10.0.0.1:8080", "tok",
            wake={"mac_address": "AA-BB-CC-DD-EE-FF"},
        )
        assert p.wol_mac == "aa:bb:cc:dd:ee:ff"

    def test_wake_custom_retries(self):
        p = Provider(
            "p1", "http://10.0.0.1:8080", "tok",
            wake={"mac_address": "aa:bb:cc:dd:ee:ff", "max_retries": 5},
        )
        assert p.wol_max_retries == 5

    def test_wake_custom_delay(self):
        p = Provider(
            "p1", "http://10.0.0.1:8080", "tok",
            wake={"mac_address": "aa:bb:cc:dd:ee:ff", "retry_delay": 3.5},
        )
        assert p.wol_retry_delay == 3.5

    def test_wake_not_dict(self):
        with pytest.raises(ValueError, match="wake must be a mapping"):
            Provider("p1", "http://localhost", "tok", wake="invalid")

    def test_wake_missing_mac(self):
        with pytest.raises(ValueError, match="wake requires 'mac_address'"):
            Provider("p1", "http://localhost", "tok", wake={"other": "val"})

    def test_wake_invalid_mac(self):
        with pytest.raises(ValueError, match="invalid mac_address"):
            Provider("p1", "http://localhost", "tok", wake={"mac_address": "zz"})

    def test_wake_zero_retries(self):
        with pytest.raises(ValueError, match="wake.max_retries must be >= 1"):
            Provider("p1", "http://localhost", "tok",
                     wake={"mac_address": "aa:bb:cc:dd:ee:ff", "max_retries": 0})

    def test_wake_invalid_delay(self):
        with pytest.raises(ValueError, match="wake.retry_delay must be a number"):
            Provider("p1", "http://localhost", "tok",
                     wake={"mac_address": "aa:bb:cc:dd:ee:ff", "retry_delay": "x"})

    def test_wake_negative_delay(self):
        with pytest.raises(ValueError, match="wake.retry_delay must be >= 0"):
            Provider("p1", "http://localhost", "tok",
                     wake={"mac_address": "aa:bb:cc:dd:ee:ff", "retry_delay": -1})


class TestProviderCooldown:
    """Test cooldown / mark_failed behavior."""

    def test_not_in_cooldown_initially(self):
        p = Provider("p1", "http://localhost", "tok")
        assert not p.in_cooldown()

    def test_in_cooldown_after_mark_failed(self):
        p = Provider("p1", "http://localhost", "tok", retry_after=10)
        p.mark_failed()
        assert p.in_cooldown()

    def test_cooldown_expires(self):
        p = Provider("p1", "http://localhost", "tok", retry_after=0.05)
        p.mark_failed()
        assert p.in_cooldown()
        time.sleep(0.06)
        assert not p.in_cooldown()

    def test_zero_retry_not_in_cooldown(self):
        p = Provider("p1", "http://localhost", "tok", retry_after=0)
        p.mark_failed()
        assert not p.in_cooldown()


class TestProviderTargetUrl:
    """Test URL path construction."""

    def test_no_base_path(self):
        p = Provider("p1", "http://localhost:8080", "tok")
        assert p.target_url("/v1/chat/completions") == "http://localhost:8080/v1/chat/completions"

    def test_base_path_prefix_stripped(self):
        p = Provider("p1", "http://localhost:8080/v1", "tok")
        assert p.target_url("/v1/chat/completions") == "http://localhost:8080/v1/chat/completions"

    def test_base_path_no_overlap(self):
        p = Provider("p1", "http://localhost:8080/v1", "tok")
        assert p.target_url("/other/path") == "http://localhost:8080/v1/other/path"

    def test_empty_base_path(self):
        p = Provider("p1", "http://localhost:8080", "tok")
        assert p.target_url("/chat") == "http://localhost:8080/chat"
