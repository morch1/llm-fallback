import importlib.util
import sys
from pathlib import Path

import pytest

# Import llm-fallback.py (hyphenated) as 'llm_fallback'
_spec = importlib.util.spec_from_file_location(
    "llm_fallback",
    str(Path(__file__).resolve().parent.parent / "llm-fallback.py"),
)
_llm_fallback = importlib.util.module_from_spec(_spec)
sys.modules["llm_fallback"] = _llm_fallback
_spec.loader.exec_module(_llm_fallback)


@pytest.fixture
def tmp_config(tmp_path):
    """Write a YAML string to a temp file and return its path."""
    def _write(yaml_content: str) -> str:
        p = tmp_path / "config.yaml"
        p.write_text(yaml_content)
        return str(p)
    return _write
