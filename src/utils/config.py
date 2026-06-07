"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_section(section: str, config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Get a specific section from configuration.
    
    Args:
        section: Section name to retrieve
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the requested section
    """
    config = load_config(config_path)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config")
    
    return config[section]


class Config:
    """Configuration class for easy access to config values."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
    
    def __getattr__(self, name: str) -> Any:
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default fallback."""
        return self.config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary."""
        return self.config.copy()
