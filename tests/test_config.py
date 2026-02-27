"""Tests for vps_fastsearch.config — no ML models required."""

from vps_fastsearch.config import FastSearchConfig, load_config


def test_default_has_expected_models():
    """Default config should include embedder and reranker model slots."""
    config = FastSearchConfig.default()
    assert "embedder" in config.models
    assert "reranker" in config.models
    assert "bge-base-en" in config.models["embedder"].name
    assert "ms-marco" in config.models["reranker"].name


def test_from_dict_roundtrip():
    """to_dict -> from_dict should preserve config values."""
    original = FastSearchConfig.default()
    data = original.to_dict()
    restored = FastSearchConfig.from_dict(data)

    assert restored.to_dict() == data


def test_load_config_nonexistent_path():
    """Loading from a nonexistent path should return defaults."""
    config = load_config("/tmp/nonexistent_fastsearch_config_xyz_12345.yaml")
    default = FastSearchConfig.default()
    assert config.daemon.socket_path == default.daemon.socket_path
    assert "embedder" in config.models


def test_to_yaml_produces_valid_output():
    """to_yaml should produce a non-empty string with expected keys."""
    config = FastSearchConfig.default()
    yaml_str = config.to_yaml()
    assert isinstance(yaml_str, str)
    assert len(yaml_str) > 0
    assert "daemon" in yaml_str
    assert "models" in yaml_str
    assert "memory" in yaml_str
