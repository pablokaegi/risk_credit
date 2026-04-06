"""Utilities for configuration management and logging."""
import logging
from pathlib import Path
from typing import Dict, Any
import yaml


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class Config:
    """Configuration management."""
    
    def __init__(self, config_file: str = 'config.yaml'):
        """Load configuration from YAML file."""
        self.config_file = Path(config_file)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._config.get(key, default)