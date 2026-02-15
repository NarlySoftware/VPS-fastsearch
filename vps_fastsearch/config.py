"""FastSearch configuration system with YAML support."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Try to import yaml, fall back to simple parsing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Default paths
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "fastsearch" / "config.yaml"
DEFAULT_SOCKET_PATH = "/tmp/fastsearch.sock"
DEFAULT_PID_PATH = "/tmp/fastsearch.pid"


@dataclass
class ModelConfig:
    """Configuration for a single model slot."""
    name: str
    keep_loaded: Literal["always", "on_demand", "never"] = "on_demand"
    idle_timeout_seconds: int = 300


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
        daemon = DaemonConfig(
            socket_path=daemon_data.get("socket_path", DEFAULT_SOCKET_PATH),
            pid_path=daemon_data.get("pid_path", DEFAULT_PID_PATH),
            log_level=daemon_data.get("log_level", "INFO"),
        )
        
        models = {}
        for name, model_data in data.get("models", {}).items():
            if isinstance(model_data, dict):
                models[name] = ModelConfig(
                    name=model_data.get("name", ""),
                    keep_loaded=model_data.get("keep_loaded", "on_demand"),
                    idle_timeout_seconds=model_data.get("idle_timeout_seconds", 300),
                )
        
        memory_data = data.get("memory", {})
        memory = MemoryConfig(
            max_ram_mb=memory_data.get("max_ram_mb", 4000),
            eviction_policy=memory_data.get("eviction_policy", "lru"),
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
            data = yaml.safe_load(content) or {}
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
    section_name = ""
    subsection_name = ""
    
    for line in content.split("\n"):
        stripped = line.rstrip()
        if not stripped or stripped.startswith("#"):
            continue
        
        # Count indentation
        indent = len(line) - len(line.lstrip())
        
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"\'')
            
            if indent == 0:
                # Top-level key
                if value:
                    result[key] = _parse_value(value)
                else:
                    result[key] = {}
                    current_section = result[key]
                    section_name = key
                    current_subsection = None
            elif indent == 2 and current_section is not None:
                # Second level
                if value:
                    current_section[key] = _parse_value(value)
                else:
                    current_section[key] = {}
                    current_subsection = current_section[key]
                    subsection_name = key
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
    elif value.isdigit():
        return int(value)
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
