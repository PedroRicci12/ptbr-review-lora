import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_hf", ROOT / "scripts" / "publish_hf.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


def test_read_token_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nHF_TOKEN='hf_test_value'\n", encoding="utf-8")
    assert publish.read_token(env) == "hf_test_value"


def test_read_token_prefers_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert publish.read_token(tmp_path / "missing.env") == "from_env"


def test_read_token_fails_when_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        publish.read_token(tmp_path / ".env")


def _adapter(tmp_path, **config):
    base = {"base_model_name_or_path": "HuggingFaceTB/SmolLM3-3B", "r": 16, "lora_alpha": 32}
    (tmp_path / "adapter_config.json").write_text(json.dumps({**base, **config}), encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    return tmp_path


def test_check_adapter_accepts_the_expected_config(tmp_path):
    assert len(publish.check_adapter(_adapter(tmp_path))) == 64  # sha256 hex


def test_check_adapter_rejects_another_config(tmp_path):
    with pytest.raises(SystemExit, match="lora_alpha"):
        publish.check_adapter(_adapter(tmp_path, lora_alpha=8))


def test_check_adapter_rejects_missing_weights(tmp_path):
    _adapter(tmp_path)
    (tmp_path / "adapter_model.safetensors").unlink()
    with pytest.raises(SystemExit, match="safetensors"):
        publish.check_adapter(tmp_path)
