"""VPS-FastSearch configuration system with YAML support."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Try to import yaml, fall back to simple parsing
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Default paths — respect XDG Base Directory specification
_xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.join(Path.home(), ".config"))
DEFAULT_CONFIG_PATH = Path(_xdg_config) / "fastsearch" / "config.yaml"

_xdg_data = os.environ.get("XDG_DATA_HOME", os.path.join(Path.home(), ".local", "share"))
DEFAULT_DB_PATH = os.path.join(_xdg_data, "fastsearch", "fastsearch.db")

_xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
if _xdg_runtime:
    DEFAULT_SOCKET_PATH = os.path.join(_xdg_runtime, "fastsearch.sock")
    DEFAULT_PID_PATH = os.path.join(_xdg_runtime, "fastsearch.pid")
else:
    DEFAULT_SOCKET_PATH = "/tmp/fastsearch.sock"
    DEFAULT_PID_PATH = "/tmp/fastsearch.pid"


@dataclass
class ModelConfig:
    """Configuration for a single model slot."""

    name: str
    keep_loaded: Literal["always", "on_demand", "never"] = "on_demand"
    idle_timeout_seconds: int = 300
    threads: int = 2
    document_prefix: str = ""
    query_prefix: str = ""


@dataclass
class MemoryConfig:
    """Memory management configuration."""

    max_ram_mb: int = 4000
    eviction_policy: Literal["lru", "fifo"] = "lru"


@dataclass
class DaemonConfig:
    """Daemon server configuration."""

    socket_path: str = DEFAULT_SOCKET_PATH
    pid_path: str = DEFAULT_PID_PATH
    log_level: str = "INFO"


@dataclass
class FastSearchConfig:
    """Complete FastSearch configuration."""

    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def default(cls) -> "FastSearchConfig":
        """Create configuration with sensible defaults."""
        return cls(
            daemon=DaemonConfig(),
            models={
                "embedder": ModelConfig(
                    name="BAAI/bge-base-en-v1.5",
                    keep_loaded="always",
                    idle_timeout_seconds=0,
                ),
                "reranker": ModelConfig(
                    name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    keep_loaded="on_demand",
                    idle_timeout_seconds=300,
                ),
            },
            memory=MemoryConfig(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FastSearchConfig":
        """Create configuration from dictionary."""
        daemon_data = data.get("daemon", {})
        socket_path = daemon_data.get("socket_path", DEFAULT_SOCKET_PATH) or DEFAULT_SOCKET_PATH
        pid_path = daemon_data.get("pid_path", DEFAULT_PID_PATH) or DEFAULT_PID_PATH
        log_level = daemon_data.get("log_level", "INFO")
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            logger.warning(f"Invalid log_level: {log_level!r}, using default 'INFO'")
            log_level = "INFO"
        daemon = DaemonConfig(
            socket_path=socket_path,
            pid_path=pid_path,
            log_level=log_level,
        )

        models = {}
        for name, model_data in data.get("models", {}).items():
            if isinstance(model_data, dict):
                keep_loaded = model_data.get("keep_loaded", "on_demand")
                if keep_loaded not in ("always", "on_demand", "never"):
                    logger.warning(
                        f"Invalid keep_loaded: {keep_loaded!r}, using default 'on_demand'"
                    )
                    keep_loaded = "on_demand"
                idle_timeout = model_data.get("idle_timeout_seconds", 300)
                if not isinstance(idle_timeout, (int, float)) or idle_timeout < 0:
                    logger.warning(
                        f"Invalid idle_timeout_seconds: {idle_timeout!r}, using default 300"
                    )
                    idle_timeout = 300
                idle_timeout = int(idle_timeout)
                threads = model_data.get("threads", 2)
                if not isinstance(threads, int) or threads < 1:
                    logger.warning(f"Invalid threads: {threads!r}, using default 2")
                    threads = 2
                # Instruction prefixes for embedding models
                doc_prefix = model_data.get("document_prefix", "")
                query_prefix = model_data.get("query_prefix", "")
                # Also accept nested instruction_prefix dict
                prefix_data = model_data.get("instruction_prefix", {})
                if isinstance(prefix_data, dict):
                    doc_prefix = doc_prefix or prefix_data.get("document", "")
                    query_prefix = query_prefix or prefix_data.get("query", "")

                models[name] = ModelConfig(
                    name=model_data.get("name", ""),
                    keep_loaded=keep_loaded,
                    idle_timeout_seconds=idle_timeout,
                    threads=threads,
                    document_prefix=str(doc_prefix),
                    query_prefix=str(query_prefix),
                )

        memory_data = data.get("memory", {})
        max_ram = memory_data.get("max_ram_mb", 4000)
        if not isinstance(max_ram, (int, float)) or max_ram <= 0:
            logger.warning(f"Invalid max_ram_mb: {max_ram!r}, using default 4000")
            max_ram = 4000
        eviction_policy = memory_data.get("eviction_policy", "lru")
        if eviction_policy not in ("lru", "fifo"):
            logger.warning(f"Invalid eviction_policy: {eviction_policy!r}, using default 'lru'")
            eviction_policy = "lru"
        memory = MemoryConfig(
            max_ram_mb=int(max_ram),
            eviction_policy=eviction_policy,
        )

        return cls(daemon=daemon, models=models, memory=memory)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "FastSearchConfig":
        """Load configuration from YAML file."""
        path = Path(path)

        if not path.exists():
            return cls.default()

        content = path.read_text()

        if HAS_YAML:
            try:
                data = yaml.safe_load(content)
            except yaml.YAMLError as e:
                logger.warning(f"Config file has syntax errors, using defaults: {e}")
                data = {}
            if not isinstance(data, dict):
                data = {}
        else:
            # Simple fallback parser for basic YAML
            data = _simple_yaml_parse(content)

        # Merge with defaults
        config = cls.default()
        loaded = cls.from_dict(data)

        # Override defaults with loaded values
        config.daemon = loaded.daemon
        if loaded.models:
            config.models.update(loaded.models)
        config.memory = loaded.memory

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "daemon": {
                "socket_path": self.daemon.socket_path,
                "pid_path": self.daemon.pid_path,
                "log_level": self.daemon.log_level,
            },
            "models": {
                name: {
                    "name": model.name,
                    "keep_loaded": model.keep_loaded,
                    "idle_timeout_seconds": model.idle_timeout_seconds,
                    "threads": model.threads,
                    **({"document_prefix": model.document_prefix} if model.document_prefix else {}),
                    **({"query_prefix": model.query_prefix} if model.query_prefix else {}),
                }
                for name, model in self.models.items()
            },
            "memory": {
                "max_ram_mb": self.memory.max_ram_mb,
                "eviction_policy": self.memory.eviction_policy,
            },
        }

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        if HAS_YAML:
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        else:
            # Simple YAML serialization
            return _simple_yaml_dump(self.to_dict())


def _simple_yaml_parse(content: str) -> dict[str, Any]:
    """Simple YAML parser for basic nested structures."""
    result: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_subsection: dict[str, Any] | None = None

    for line in content.split("\n"):
        stripped = line.rstrip()
        if not stripped or stripped.startswith("#"):
            continue

        # Count indentation
        indent = len(line) - len(line.lstrip())

        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip("\"'")

            if indent == 0:
                # Top-level key
                if value:
                    result[key] = _parse_value(value)
                else:
                    result[key] = {}
                    current_section = result[key]
                    current_subsection = None
            elif indent == 2 and current_section is not None:
                # Second level
                if value:
                    current_section[key] = _parse_value(value)
                else:
                    current_section[key] = {}
                    current_subsection = current_section[key]
            elif indent == 4 and current_subsection is not None:
                # Third level
                current_subsection[key] = _parse_value(value)

    return result


def _parse_value(value: str) -> Any:
    """Parse a simple YAML value."""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_dump(data: dict[str, Any], indent: int = 0) -> str:
    """Simple YAML dumper."""
    lines = []
    prefix = "  " * indent

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_simple_yaml_dump(value, indent + 1))
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


def load_config(path: Path | str | None = None) -> FastSearchConfig:
    """
    Load configuration from file or environment.

    Priority:
    1. Explicit path argument
    2. FASTSEARCH_CONFIG environment variable
    3. ~/.config/fastsearch/config.yaml
    4. Default configuration
    """
    if path is not None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return FastSearchConfig.from_yaml(path)

    env_path = os.environ.get("FASTSEARCH_CONFIG")
    if env_path:
        return FastSearchConfig.from_yaml(env_path)

    if DEFAULT_CONFIG_PATH.exists():
        return FastSearchConfig.from_yaml(DEFAULT_CONFIG_PATH)

    return FastSearchConfig.default()


def create_default_config(path: Path | str | None = None) -> Path:
    """Create default configuration file."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    config = FastSearchConfig.default()
    path.write_text(config.to_yaml())

    return path
