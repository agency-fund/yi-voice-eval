"""Configuration management for voice evaluation pipeline."""

import yaml
from pathlib import Path
from typing import Any, Optional


_config_cache: Optional[dict] = None


def load_config(*keys: str, config_path: str = "config/manifest_kannada_stt_haveri.yaml") -> Any:
    """
    Load configuration value from YAML file.

    Args:
        *keys: Nested keys to access (e.g., 'input', 'audio_dir')
        config_path: Path to YAML config file (relative to project root)

    Returns:
        Configuration value at the specified path

    Examples:
        >>> load_config('input', 'audio_dir')
        'files/Haveri_Audios'

        >>> load_config('whisper', 'model')
        'large-v3'

        >>> load_config('dataset')
        {'name': '...', 'description': '...'}
    """
    global _config_cache

    if _config_cache is None:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        full_path = project_root / config_path

        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")

        with open(full_path) as f:
            _config_cache = yaml.safe_load(f)

    result = _config_cache
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
            if result is None:
                raise KeyError(f"Config key not found: {' -> '.join(keys)}")
        else:
            raise TypeError(f"Cannot access key '{key}' in non-dict value")

    return result


def reload_config():
    """Force reload of config file (useful for testing)."""
    global _config_cache
    _config_cache = None
