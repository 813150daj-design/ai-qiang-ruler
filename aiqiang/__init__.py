"""AI腔尺子：只量文风的离线工具。"""

__version__ = "0.1.0"

from .analyzer import DEFAULT_MODEL_PATH, analyze, load_model

__all__ = ["__version__", "analyze", "load_model", "DEFAULT_MODEL_PATH"]
