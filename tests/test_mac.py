import pytest

from llm_fallback import validate_mac_address


class TestMacValidation:
    """Test MAC address validation and normalization."""

    @pytest.mark.parametrize(
        "mac, expected",
        [
            ("aa:bb:cc:dd:ee:ff", "aa:bb:cc:dd:ee:ff"),
            ("AA:BB:CC:DD:EE:FF", "aa:bb:cc:dd:ee:ff"),
            ("aa-bb-cc-dd-ee-ff", "aa:bb:cc:dd:ee:ff"),
            ("AA-BB-CC-DD-EE-FF", "aa:bb:cc:dd:ee:ff"),
            ("00:00:00:00:00:00", "00:00:00:00:00:00"),
            ("ff:ff:ff:ff:ff:ff", "ff:ff:ff:ff:ff:ff"),
            ("1a:2b:3c:4d:5e:6f", "1a:2b:3c:4d:5e:6f"),
            ("1a-2b-3c-4d-5e-6f", "1a:2b:3c:4d:5e:6f"),
            ("Ab:Cd:Ef:01:23:45", "ab:cd:ef:01:23:45"),
        ],
    )
    def test_valid_mac(self, mac, expected):
        assert validate_mac_address(mac) == expected

    @pytest.mark.parametrize(
        "mac",
        [
            "",
            "aa:bb:cc:dd:ee",
            "aa:bb:cc:dd:ee:ff:gg",
            "aa:bb:cc:dd:ee:f",
            "aa:bb:cc:dd:ee:gg",
            "aa bb cc dd ee ff",
            "aa.bb.cc.dd.ee.ff",
            "a:b:c:d:e:f",
            "aa:bb:cc:dd:ee:ff:00",
            "not-a-mac",
            "123456",
            "12-34-56-78-9A",
            "12:34:56:78:9A",
            "123456789012",
            ":aa:bb:cc:dd:ee:ff",
            "aa:bb:cc:dd:ee:ff:",
            "aa::bb:cc:dd:ee:ff",
        ],
    )
    def test_invalid_mac(self, mac):
        with pytest.raises(ValueError, match="invalid mac_address"):
            validate_mac_address(mac)
