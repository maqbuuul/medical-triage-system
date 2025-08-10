"""Model training and tuning utilities."""

import importlib
from src.config import CLASSIFIERS, RND
from src.logging_config import get_logger

logger = get_logger("model_utils")


def get_model_instance(module_path, params):
    """Instantiate model from module path."""
    try:
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**params) if params else cls()
    except Exception as e:
        logger.error(f"Model instantiation failed: {e}")
        raise
