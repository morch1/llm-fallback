import pytest


@pytest.fixture
def tmp_config(tmp_path):
    """Write a YAML string to a temp file and return its path."""
    def _write(yaml_content: str) -> str:
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content)
        return str(p)
    return _write
