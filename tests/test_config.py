"""Tests for vps_fastsearch.config — no ML models required."""

from pathlib import Path

import pytest

from vps_fastsearch.config import (
    DaemonConfig,
    FastSearchConfig,
    MemoryConfig,
    ModelConfig,
    create_default_config,
    load_config,
)


def test_default_has_expected_models() -> None:
    """Default config should include embedder and reranker model slots."""
    config = FastSearchConfig.default()
    assert "embedder" in config.models
    assert "reranker" in config.models
    assert "bge-base-en" in config.models["embedder"].name
    assert "ms-marco" in config.models["reranker"].name


def test_from_dict_roundtrip() -> None:
    """to_dict -> from_dict should preserve config values."""
    original = FastSearchConfig.default()
    data = original.to_dict()
    restored = FastSearchConfig.from_dict(data)

    assert restored.to_dict() == data


def test_load_config_nonexistent_path() -> None:
    """Loading from an explicit nonexistent path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/tmp/nonexistent_fastsearch_config_xyz_12345.yaml")


def test_to_yaml_produces_valid_output() -> None:
    """to_yaml should produce a non-empty string with expected keys."""
    config = FastSearchConfig.default()
    yaml_str = config.to_yaml()
    assert isinstance(yaml_str, str)
    assert len(yaml_str) > 0
    assert "daemon" in yaml_str
    assert "models" in yaml_str
    assert "memory" in yaml_str


# ---------------------------------------------------------------------------
# Edge case tests for FastSearchConfig.from_yaml / load_config
# ---------------------------------------------------------------------------


def test_from_yaml_missing_file_returns_defaults() -> None:
    """from_yaml with a nonexistent path should return default config."""
    config = FastSearchConfig.from_yaml("/tmp/definitely_does_not_exist_xyz_99999.yaml")
    default = FastSearchConfig.default()
    assert config.to_dict() == default.to_dict()


def test_from_yaml_partial_config(tmp_path) -> None:
    """A YAML file with only some fields should fill the rest with defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("daemon:\n  log_level: DEBUG\n")

    config = FastSearchConfig.from_yaml(config_file)
    assert config.daemon.log_level == "DEBUG"
    # Other daemon fields should be defaults
    assert config.daemon.pid_path is not None
    assert config.daemon.socket_path is not None
    # Models should still have defaults
    assert "embedder" in config.models
    assert "reranker" in config.models


def test_from_yaml_invalid_yaml_uses_defaults(tmp_path) -> None:
    """Invalid YAML content should fall back to defaults, not crash."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(":::invalid yaml content{{{")

    config = FastSearchConfig.from_yaml(config_file)
    default = FastSearchConfig.default()
    # Should get default config when YAML parsing fails
    assert config.to_dict() == default.to_dict()


def test_from_yaml_empty_file(tmp_path) -> None:
    """An empty YAML file should return defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")

    config = FastSearchConfig.from_yaml(config_file)
    default = FastSearchConfig.default()
    assert config.to_dict() == default.to_dict()


def test_from_yaml_unknown_keys(tmp_path) -> None:
    """Unknown top-level keys should not crash config loading."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "daemon:\n  log_level: INFO\nunknown_section:\n  foo: bar\nanother_unknown: 42\n"
    )
    config = FastSearchConfig.from_yaml(config_file)
    assert config.daemon.log_level == "INFO"


# ---------------------------------------------------------------------------
# Edge case tests for from_dict validation
# ---------------------------------------------------------------------------


def test_from_dict_empty() -> None:
    """An empty dict should produce a valid config with defaults."""
    config = FastSearchConfig.from_dict({})
    assert config.daemon.log_level == "INFO"
    assert config.memory.max_ram_mb == 4000
    assert config.memory.eviction_policy == "lru"


def test_from_dict_invalid_log_level() -> None:
    """Invalid log level should fall back to INFO."""
    data = {"daemon": {"log_level": "BOGUS"}}
    config = FastSearchConfig.from_dict(data)
    assert config.daemon.log_level == "INFO"


def test_from_dict_invalid_keep_loaded() -> None:
    """Invalid keep_loaded value should fall back to on_demand."""
    data = {
        "models": {
            "test_model": {
                "name": "test/model",
                "keep_loaded": "sometimes",
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].keep_loaded == "on_demand"


def test_from_dict_negative_idle_timeout() -> None:
    """Negative idle_timeout_seconds should fall back to 300."""
    data = {
        "models": {
            "test_model": {
                "name": "test/model",
                "idle_timeout_seconds": -10,
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].idle_timeout_seconds == 300


def test_from_dict_non_numeric_idle_timeout() -> None:
    """Non-numeric idle_timeout_seconds should fall back to 300."""
    data = {
        "models": {
            "test_model": {
                "name": "test/model",
                "idle_timeout_seconds": "not_a_number",
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].idle_timeout_seconds == 300


def test_from_dict_invalid_threads() -> None:
    """Invalid threads value (0 or negative) should fall back to 2."""
    data = {
        "models": {
            "test_model": {
                "name": "test/model",
                "threads": 0,
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].threads == 2

    data["models"]["test_model"]["threads"] = -1
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].threads == 2


def test_from_dict_non_int_threads() -> None:
    """Non-integer threads value should fall back to 2."""
    data = {
        "models": {
            "test_model": {
                "name": "test/model",
                "threads": "four",
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].threads == 2


def test_from_dict_model_not_a_dict() -> None:
    """Non-dict model entry should be skipped (not crash)."""
    data = {
        "models": {
            "bad_model": "just a string",
            "good_model": {"name": "test/model"},
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert "bad_model" not in config.models
    assert "good_model" in config.models


def test_from_dict_model_missing_name() -> None:
    """Model entry with no name field should get empty string name."""
    data = {
        "models": {
            "test_model": {
                "keep_loaded": "always",
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].name == ""


# ---------------------------------------------------------------------------
# Edge case tests for MemoryConfig validation
# ---------------------------------------------------------------------------


def test_from_dict_zero_max_ram() -> None:
    """Zero max_ram_mb should fall back to default (4000)."""
    data = {"memory": {"max_ram_mb": 0}}
    config = FastSearchConfig.from_dict(data)
    assert config.memory.max_ram_mb == 4000


def test_from_dict_negative_max_ram() -> None:
    """Negative max_ram_mb should fall back to default (4000)."""
    data = {"memory": {"max_ram_mb": -512}}
    config = FastSearchConfig.from_dict(data)
    assert config.memory.max_ram_mb == 4000


def test_from_dict_non_numeric_max_ram() -> None:
    """Non-numeric max_ram_mb should fall back to default (4000)."""
    data = {"memory": {"max_ram_mb": "lots"}}
    config = FastSearchConfig.from_dict(data)
    assert config.memory.max_ram_mb == 4000


def test_from_dict_invalid_eviction_policy() -> None:
    """Invalid eviction policy should fall back to lru."""
    data = {"memory": {"eviction_policy": "random"}}
    config = FastSearchConfig.from_dict(data)
    assert config.memory.eviction_policy == "lru"


def test_from_dict_float_max_ram() -> None:
    """Float max_ram_mb should be accepted and truncated to int."""
    data = {"memory": {"max_ram_mb": 2048.7}}
    config = FastSearchConfig.from_dict(data)
    assert config.memory.max_ram_mb == 2048


# ---------------------------------------------------------------------------
# Edge case tests for DaemonConfig
# ---------------------------------------------------------------------------


def test_from_dict_null_socket_path() -> None:
    """None/null socket_path should fall back to default."""
    data = {"daemon": {"socket_path": None}}
    config = FastSearchConfig.from_dict(data)
    assert config.daemon.socket_path is not None
    assert len(config.daemon.socket_path) > 0


def test_from_dict_null_pid_path() -> None:
    """None/null pid_path should fall back to default."""
    data = {"daemon": {"pid_path": None}}
    config = FastSearchConfig.from_dict(data)
    assert config.daemon.pid_path is not None
    assert len(config.daemon.pid_path) > 0


def test_daemon_config_defaults() -> None:
    """DaemonConfig direct instantiation should have expected defaults."""
    dc = DaemonConfig()
    assert dc.log_level == "INFO"
    assert dc.socket_path is not None
    assert dc.pid_path is not None


# ---------------------------------------------------------------------------
# Edge case tests for ModelConfig and MemoryConfig dataclasses
# ---------------------------------------------------------------------------


def test_model_config_defaults() -> None:
    """ModelConfig should have expected default values."""
    mc = ModelConfig(name="test/model")
    assert mc.keep_loaded == "on_demand"
    assert mc.idle_timeout_seconds == 300
    assert mc.threads == 2


def test_memory_config_defaults() -> None:
    """MemoryConfig should have expected default values."""
    mem = MemoryConfig()
    assert mem.max_ram_mb == 4000
    assert mem.eviction_policy == "lru"


# ---------------------------------------------------------------------------
# Edge case tests for load_config
# ---------------------------------------------------------------------------


def test_load_config_from_explicit_path(tmp_path) -> None:
    """load_config with an explicit valid path should load that file."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("daemon:\n  log_level: WARNING\n")

    config = load_config(config_file)
    assert config.daemon.log_level == "WARNING"


def test_load_config_env_var_override(tmp_path, monkeypatch) -> None:
    """FASTSEARCH_CONFIG env var should be used when no explicit path is given."""
    config_file = tmp_path / "env_config.yaml"
    config_file.write_text("daemon:\n  log_level: ERROR\n")

    monkeypatch.setenv("FASTSEARCH_CONFIG", str(config_file))
    # Also ensure the default config path does not exist
    monkeypatch.setattr(
        "vps_fastsearch.config.DEFAULT_CONFIG_PATH",
        Path("/tmp/definitely_nonexistent_default_config_xyz.yaml"),
    )

    config = load_config()
    assert config.daemon.log_level == "ERROR"


def test_load_config_env_var_nonexistent(monkeypatch) -> None:
    """FASTSEARCH_CONFIG pointing to nonexistent file should return defaults (from_yaml behavior)."""
    monkeypatch.setenv("FASTSEARCH_CONFIG", "/tmp/does_not_exist_env_config.yaml")
    monkeypatch.setattr(
        "vps_fastsearch.config.DEFAULT_CONFIG_PATH",
        Path("/tmp/definitely_nonexistent_default_config_xyz.yaml"),
    )
    # from_yaml returns defaults for nonexistent paths (no explicit path = no FileNotFoundError)
    config = load_config()
    default = FastSearchConfig.default()
    assert config.to_dict() == default.to_dict()


def test_load_config_no_path_no_env_no_default(monkeypatch) -> None:
    """With no path, no env var, and no default file, should return defaults."""
    monkeypatch.delenv("FASTSEARCH_CONFIG", raising=False)
    monkeypatch.setattr(
        "vps_fastsearch.config.DEFAULT_CONFIG_PATH",
        Path("/tmp/definitely_nonexistent_default_config_xyz.yaml"),
    )
    config = load_config()
    default = FastSearchConfig.default()
    assert config.to_dict() == default.to_dict()


# ---------------------------------------------------------------------------
# Edge case tests for create_default_config
# ---------------------------------------------------------------------------


def test_create_default_config(tmp_path) -> None:
    """create_default_config should write a valid YAML file."""
    config_path = tmp_path / "subdir" / "config.yaml"
    result = create_default_config(config_path)
    assert result == config_path
    assert config_path.exists()
    # Verify the written file can be loaded back
    config = FastSearchConfig.from_yaml(config_path)
    assert "embedder" in config.models


# ---------------------------------------------------------------------------
# Edge case tests for to_dict / to_yaml roundtrip
# ---------------------------------------------------------------------------


def test_to_dict_to_yaml_roundtrip(tmp_path) -> None:
    """Config written to YAML and read back should match original."""
    original = FastSearchConfig.default()
    yaml_str = original.to_yaml()

    config_file = tmp_path / "roundtrip.yaml"
    config_file.write_text(yaml_str)

    restored = FastSearchConfig.from_yaml(config_file)
    assert restored.to_dict() == original.to_dict()


def test_from_dict_all_valid_log_levels() -> None:
    """All valid log levels should be accepted."""
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        data = {"daemon": {"log_level": level}}
        config = FastSearchConfig.from_dict(data)
        assert config.daemon.log_level == level


def test_from_dict_all_valid_keep_loaded() -> None:
    """All valid keep_loaded values should be accepted."""
    for mode in ("always", "on_demand", "never"):
        data = {"models": {"m": {"name": "test", "keep_loaded": mode}}}
        config = FastSearchConfig.from_dict(data)
        assert config.models["m"].keep_loaded == mode


def test_from_dict_all_valid_eviction_policies() -> None:
    """All valid eviction policies should be accepted."""
    for policy in ("lru", "fifo"):
        data = {"memory": {"eviction_policy": policy}}
        config = FastSearchConfig.from_dict(data)
        assert config.memory.eviction_policy == policy


def test_from_dict_float_idle_timeout() -> None:
    """Float idle_timeout_seconds should be accepted and truncated to int."""
    data = {
        "models": {
            "test_model": {
                "name": "test/model",
                "idle_timeout_seconds": 120.9,
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["test_model"].idle_timeout_seconds == 120


# ---------------------------------------------------------------------------
# Instruction prefix tests (#15)
# ---------------------------------------------------------------------------


def test_model_config_prefix_defaults() -> None:
    """ModelConfig should default to empty prefix strings."""
    mc = ModelConfig(name="test/model")
    assert mc.document_prefix == ""
    assert mc.query_prefix == ""


def test_from_dict_flat_prefix_keys() -> None:
    """Flat document_prefix and query_prefix keys should be parsed."""
    data = {
        "models": {
            "embedder": {
                "name": "test/model",
                "document_prefix": "Represent this document: ",
                "query_prefix": "Represent this query: ",
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["embedder"].document_prefix == "Represent this document: "
    assert config.models["embedder"].query_prefix == "Represent this query: "


def test_from_dict_nested_instruction_prefix() -> None:
    """Nested instruction_prefix dict should be parsed."""
    data = {
        "models": {
            "embedder": {
                "name": "test/model",
                "instruction_prefix": {
                    "document": "Doc prefix: ",
                    "query": "Query prefix: ",
                },
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["embedder"].document_prefix == "Doc prefix: "
    assert config.models["embedder"].query_prefix == "Query prefix: "


def test_from_dict_flat_prefix_overrides_nested() -> None:
    """Flat prefix keys should take priority over nested instruction_prefix."""
    data = {
        "models": {
            "embedder": {
                "name": "test/model",
                "document_prefix": "Flat doc: ",
                "instruction_prefix": {
                    "document": "Nested doc: ",
                },
            }
        }
    }
    config = FastSearchConfig.from_dict(data)
    assert config.models["embedder"].document_prefix == "Flat doc: "


def test_to_dict_omits_empty_prefixes() -> None:
    """to_dict should not include prefix keys when they are empty."""
    config = FastSearchConfig.default()
    data = config.to_dict()
    embedder_data = data["models"]["embedder"]
    assert "document_prefix" not in embedder_data
    assert "query_prefix" not in embedder_data


def test_to_dict_includes_non_empty_prefixes() -> None:
    """to_dict should include prefix keys when they are set."""
    config = FastSearchConfig.default()
    config.models["embedder"].document_prefix = "Doc: "
    config.models["embedder"].query_prefix = "Query: "
    data = config.to_dict()
    embedder_data = data["models"]["embedder"]
    assert embedder_data["document_prefix"] == "Doc: "
    assert embedder_data["query_prefix"] == "Query: "


def test_prefix_roundtrip_via_dict() -> None:
    """Prefixes should survive to_dict -> from_dict roundtrip."""
    config = FastSearchConfig.default()
    config.models["embedder"].document_prefix = "Doc: "
    config.models["embedder"].query_prefix = "Query: "
    data = config.to_dict()
    restored = FastSearchConfig.from_dict(data)
    assert restored.models["embedder"].document_prefix == "Doc: "
    assert restored.models["embedder"].query_prefix == "Query: "


def test_from_yaml_yaml_list_instead_of_dict(tmp_path) -> None:
    """YAML content that parses to a list (not dict) should fall back to defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- item1\n- item2\n")

    config = FastSearchConfig.from_yaml(config_file)
    default = FastSearchConfig.default()
    assert config.to_dict() == default.to_dict()
